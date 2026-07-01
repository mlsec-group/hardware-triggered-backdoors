from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch

from strategies.chimera_config import ChimeraSearchConfig


@dataclass(frozen=True)
class MarginState:
    abs_margin: float
    margin: float
    competitor: int


@dataclass(frozen=True)
class CandidateProposal:
    tensor: torch.Tensor
    source: str


def quantize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return torch.floor(tensor.clamp(0, 1) * 255.0 + 0.5) / 255.0


def margin_stats(logits: torch.Tensor, cls: int):
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask[:, cls] = True
    competitor_logits = logits.masked_fill(mask, -1e9)
    competitor_values, competitor_idx = competitor_logits.max(dim=1)
    margin = logits[:, cls] - competitor_values
    return margin, competitor_idx


class ChimeraSearch:
    """Resumable chimera search.

    The generator client owns this object. It performs all differentiable search
    work locally, but external BLIS/OpenBLAS oracle results are fed back through
    :meth:`update_with_probe_results`.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        image: torch.Tensor,
        label: int,
        *,
        device: torch.device | str,
        seed: int,
        config: ChimeraSearchConfig,
    ):
        self.model = model
        self.device = torch.device(device)
        self.label = int(label)
        self.config = config
        seed = int(seed) & 0xFFFFFFFF
        self.rng = np.random.default_rng(seed)
        try:
            self.torch_rng = torch.Generator(device=self.device).manual_seed(seed)
        except RuntimeError:
            self.torch_rng = torch.Generator().manual_seed(seed)
        self.trace_events = []

        self.base_tensor = self.prepare_tensor(image)
        with torch.inference_mode():
            logits = self.run_model(self.base_tensor)
            self.original_cls = int(torch.argmax(logits, dim=1).item())
            margin, competitor = margin_stats(logits, self.original_cls)
            self.initial_state = MarginState(
                abs_margin=abs(float(margin.item())),
                margin=float(margin.item()),
                competitor=int(competitor.item()),
            )
        self.trace(
            "initial_prediction",
            label=self.label,
            original_class=self.original_cls,
            initial_abs_margin=self.initial_state.abs_margin,
            initial_margin=self.initial_state.margin,
            initial_competitor=self.initial_state.competitor,
            device=str(self.device),
        )

        optimized, _ = self.gradient_descent_boundary(
            self.base_tensor, self.original_cls
        )
        quantized, abs_margin, margin, competitor = self.best_quantized_dithered_start(
            optimized, self.original_cls
        )
        self.best_tensor = quantize_tensor(quantized)
        self.best_state = MarginState(abs_margin, margin, competitor)
        self.current_tensor = self.best_tensor

        self.round_idx = 0
        self.done = False
        self.success = False
        self.result_tensor: Optional[torch.Tensor] = None
        self.result_index: Optional[int] = None
        self.last_candidates: Optional[torch.Tensor] = None
        self.last_states: list[MarginState] = []
        self.oracle_bridge_queue: list[CandidateProposal] = []

    def jsonable(self, value):
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.detach().cpu().item()
            return value.detach().cpu().tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, MarginState):
            return {
                "abs_margin": value.abs_margin,
                "margin": value.margin,
                "competitor": value.competitor,
            }
        return value

    def trace(self, event: str, **payload) -> None:
        self.trace_events.append(
            {
                "event": event,
                **{key: self.jsonable(value) for key, value in payload.items()},
            }
        )

    def pop_trace_events(self):
        events = self.trace_events
        self.trace_events = []
        return events

    def with_trace(self, payload: Dict[str, object]) -> Dict[str, object]:
        events = self.pop_trace_events()
        if events:
            payload["trace_events"] = events
        return payload

    def prepare_tensor(self, image: torch.Tensor) -> torch.Tensor:
        tensor = image.detach().to(self.device, dtype=torch.float32)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        assert tensor.shape[1:] == (3, 32, 32), f"Expected CIFAR image, got {tensor.shape}"
        return tensor.clamp(0.0, 1.0).contiguous()

    def run_model(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.model(tensor.to(self.device))

    def margin_value(self, tensor: torch.Tensor, cls: int) -> float:
        with torch.inference_mode():
            logits = self.run_model(tensor)
            margin, _ = margin_stats(logits, cls)
            return float(margin.item())

    def binary_search_boundary(self, base, crossing, cls, steps=24):
        margin_base = self.margin_value(base, cls)
        margin_cross = self.margin_value(crossing, cls)
        if not (margin_base > 0 and margin_cross <= 0):
            self.trace(
                "binary_search_invalid_bracket",
                margin_base=margin_base,
                margin_cross=margin_cross,
            )
            return base if margin_base <= margin_cross else crossing

        low = base.clone()
        high = crossing.clone()
        final_margin = margin_cross
        for _ in range(steps):
            mid = ((low + high) * 0.5).detach()
            margin = self.margin_value(mid, cls)
            final_margin = margin
            if margin > 0:
                low = mid
            else:
                high = mid
        self.trace("binary_search_complete", steps=steps, margin=final_margin)
        return high

    def project_linf(self, tensor, center, epsilon):
        if epsilon <= 0:
            return tensor
        return torch.max(torch.min(tensor, center + epsilon), center - epsilon)

    def gradient_descent_boundary(self, original, cls):
        cfg = self.config
        current = original.clone()
        velocity = torch.zeros_like(current)
        best = current.clone()
        best_margin = float("inf")
        crossing = None
        lr = float(cfg.gd_step_size)

        self.trace(
            "gradient_descent_start",
            steps=int(cfg.gd_steps),
            step_size=float(cfg.gd_step_size),
            momentum=float(cfg.gd_momentum),
            epsilon=float(cfg.gd_epsilon),
        )

        stop_reason = "max_steps"
        step = -1
        for step in range(int(cfg.gd_steps)):
            working = current.detach().requires_grad_(True)
            logits = self.run_model(working)
            margin_tensor, _ = margin_stats(logits, cls)
            loss = margin_tensor.mean()
            margin_value = float(margin_tensor.item())

            if margin_value < best_margin:
                best_margin = margin_value
                best = working.detach()
                self.trace(
                    "gradient_descent_new_best",
                    step=step,
                    margin=best_margin,
                    lr=lr,
                )
            if margin_value <= 0.0:
                crossing = working.detach()
                stop_reason = "crossing"
                self.trace("gradient_descent_crossing", step=step, margin=margin_value)
                break

            loss.backward()
            grad = working.grad
            if grad is None:
                stop_reason = "missing_gradient"
                break
            flat_grad = grad.view(grad.size(0), -1)
            grad_norm = flat_grad.norm(p=2, dim=1, keepdim=True)
            if torch.any(~torch.isfinite(grad_norm)) or torch.min(grad_norm).item() < 1e-6:
                stop_reason = "bad_gradient_norm"
                self.trace(
                    "gradient_descent_stop",
                    step=step,
                    reason=stop_reason,
                    grad_norm=float(torch.min(grad_norm).item()),
                )
                break
            grad_norm = torch.clamp(grad_norm, min=1e-8)
            normalized_grad = grad / grad_norm.view(-1, 1, 1, 1)
            velocity = float(cfg.gd_momentum) * velocity - lr * normalized_grad
            current = (working + velocity).detach()
            current = self.project_linf(current, original, float(cfg.gd_epsilon))
            current = current.clamp(0.0, 1.0)
            lr *= 0.98

        self.trace(
            "gradient_descent_finish",
            step=step,
            reason=stop_reason,
            best_margin=best_margin,
            crossed=crossing is not None,
        )
        if crossing is None:
            return best, False
        return self.binary_search_boundary(original, crossing, cls), True

    def best_quantized_dithered_start(self, tensor, cls):
        cfg = self.config
        center = tensor.detach().clamp(0.0, 1.0)
        best_tensor = None
        best_abs_margin = float("inf")
        best_margin = 0.0
        best_competitor = -1
        current_radius = float(cfg.quant_dither_radius)

        self.trace(
            "quantized_dither_start",
            samples=int(cfg.quant_dither_samples),
            rounds=int(cfg.quant_dither_rounds),
            radius=float(cfg.quant_dither_radius),
        )

        for round_idx in range(max(1, int(cfg.quant_dither_rounds))):
            batch_size = max(1, int(cfg.quant_dither_samples)) + 1
            float_batch = center.repeat(batch_size, 1, 1, 1)
            if batch_size > 1 and current_radius > 0:
                noise = (
                    torch.rand(
                        batch_size - 1,
                        *center.shape[1:],
                        device=self.device,
                        dtype=torch.float32,
                        generator=self.torch_rng,
                    )
                    - 0.5
                ) * (2.0 * current_radius)
                float_batch[1:] = (float_batch[1:] + noise).clamp(0.0, 1.0)
            quant_batch = quantize_tensor(float_batch)

            with torch.inference_mode():
                logits = self.run_model(quant_batch)
                margin, competitor_idx = margin_stats(logits, cls)
                abs_margin = margin.abs()
                idx_best = int(torch.argmin(abs_margin).item())

            if float(abs_margin[idx_best].item()) < best_abs_margin:
                best_abs_margin = float(abs_margin[idx_best].item())
                best_margin = float(margin[idx_best].item())
                best_competitor = int(competitor_idx[idx_best].item())
                best_tensor = quant_batch[idx_best : idx_best + 1].detach()

            self.trace(
                "quantized_dither_round",
                round=round_idx,
                radius=current_radius,
                round_best_abs_margin=float(abs_margin[idx_best].item()),
                round_best_margin=float(margin[idx_best].item()),
                round_best_competitor=int(competitor_idx[idx_best].item()),
                best_abs_margin=best_abs_margin,
            )
            center = float_batch[idx_best : idx_best + 1].detach()
            current_radius *= float(cfg.quant_dither_radius_decay)

        assert best_tensor is not None
        self.trace(
            "quantized_dither_finish",
            best_abs_margin=best_abs_margin,
            best_margin=best_margin,
            best_competitor=best_competitor,
        )
        return best_tensor, best_abs_margin, best_margin, best_competitor

    def tangent_kick_quantized(self, tensor, cls, *, round_idx=None, candidate_idx=None):
        cfg = self.config
        base = quantize_tensor(tensor).detach()
        _, channels, height, width = base.shape

        work = base.clone().detach().requires_grad_(True)
        logits = self.run_model(work)
        _, competitor_idx = margin_stats(logits, cls)
        competitor = int(competitor_idx.item())
        delta = logits[:, cls] - logits[:, competitor]
        loss = 0.5 * delta.pow(2).mean()
        loss.backward()
        grad = work.grad
        if grad is None:
            self.trace(
                "tangent_kick_skipped",
                round=round_idx,
                candidate=candidate_idx,
                reason="missing_gradient",
            )
            return base

        flat = grad.view(-1).abs()
        pool_size = int(
            max(1, min(flat.numel(), round(float(cfg.tangent_pool_frac) * flat.numel())))
        )
        _, pool_indices = torch.topk(flat, pool_size, largest=False)
        pool_np = pool_indices.detach().cpu().numpy()

        u8 = torch.round(base.clamp(0.0, 1.0) * 255.0).to(torch.int16)
        per_img = channels * height * width
        moved = 0
        attempts = 0
        target = max(0, int(cfg.tangent_pixels))
        while moved < target and attempts < max(1, target * 20):
            attempts += 1
            idx = int(self.rng.choice(pool_np)) % per_img
            channel = idx // (height * width)
            rem = idx - channel * height * width
            row = rem // width
            col = rem - row * width
            current = int(u8[0, channel, row, col].item())
            if current == 0:
                direction = 1
            elif current == 255:
                direction = -1
            else:
                direction = -1 if self.rng.random() < 0.5 else 1
            u8[0, channel, row, col] = current + direction
            moved += 1
        kicked = u8.to(torch.float32).div(255.0).detach()
        before_u8 = torch.round(base.clamp(0.0, 1.0) * 255.0).to(torch.int16)
        changed = int((before_u8 != u8).sum().item())
        self.trace(
            "tangent_kick",
            round=round_idx,
            candidate=candidate_idx,
            moved=moved,
            changed=changed,
            attempts=attempts,
            pool_size=pool_size,
            target=target,
        )
        return kicked

    def pixel_sweep(self, tensor, cls, best_abs_margin):
        cfg = self.config
        base = tensor.detach().clamp(0.0, 1.0)
        base_u8 = torch.round(base * 255.0).to(torch.int16)
        base_quant = base_u8.to(torch.float32).div(255.0)
        _, channels, height, width = base_u8.shape
        total_coords = int(channels * height * width)

        if cfg.sweep_coords_per_round is None:
            flat_idx = np.arange(total_coords, dtype=np.int64)
        else:
            k = int(max(1, min(total_coords, int(cfg.sweep_coords_per_round))))
            flat_idx = self.rng.choice(total_coords, size=k, replace=False).astype(np.int64)

        per_ch = int(height * width)
        ch_idx = torch.from_numpy(flat_idx // per_ch).to(self.device, dtype=torch.long)
        rem = flat_idx - (flat_idx // per_ch) * per_ch
        r_idx = torch.from_numpy(rem // int(width)).to(self.device, dtype=torch.long)
        c_idx = torch.from_numpy(rem - (rem // int(width)) * int(width)).to(
            self.device, dtype=torch.long
        )

        cur = base_u8[0, ch_idx, r_idx, c_idx]
        coord_count = int(ch_idx.numel())
        dirs = torch.tensor([-1, 1], device=self.device, dtype=torch.int16).repeat_interleave(
            coord_count
        )
        cur_rep = cur.repeat(2)
        ch_rep = ch_idx.repeat(2)
        r_rep = r_idx.repeat(2)
        c_rep = c_idx.repeat(2)
        new_u8 = torch.clamp(cur_rep + dirs, 0, 255)
        valid = new_u8 != cur_rep
        if not torch.any(valid):
            return base, best_abs_margin, 0.0, -1, False

        ch_rep = ch_rep[valid]
        r_rep = r_rep[valid]
        c_rep = c_rep[valid]
        new_u8 = new_u8[valid]
        move_count = int(new_u8.numel())

        best_move_idx = -1
        best_move_abs_margin = float("inf")
        best_move_margin = 0.0
        best_move_competitor = -1
        max_batch = max(1, int(cfg.sweep_max_batch))

        with torch.inference_mode():
            for start in range(0, move_count, max_batch):
                end = min(move_count, start + max_batch)
                chunk = end - start
                batch = base_quant.expand(chunk, -1, -1, -1).clone()
                batch_indices = torch.arange(chunk, device=self.device, dtype=torch.long)
                batch[
                    batch_indices,
                    ch_rep[start:end],
                    r_rep[start:end],
                    c_rep[start:end],
                ] = new_u8[start:end].to(torch.float32).div(255.0)

                logits_batch = self.run_model(batch)
                margin_batch, competitor_idx_batch = margin_stats(logits_batch, cls)
                abs_margin_batch = margin_batch.abs()
                chunk_best = int(torch.argmin(abs_margin_batch).item())
                chunk_best_abs_margin = float(abs_margin_batch[chunk_best].item())
                if chunk_best_abs_margin < best_move_abs_margin:
                    best_move_abs_margin = chunk_best_abs_margin
                    best_move_margin = float(margin_batch[chunk_best].item())
                    best_move_competitor = int(competitor_idx_batch[chunk_best].item())
                    best_move_idx = start + chunk_best

        if best_move_idx >= 0 and best_move_abs_margin + 1e-12 < best_abs_margin:
            chosen = base_quant.clone()
            chosen[
                0,
                int(ch_rep[best_move_idx].item()),
                int(r_rep[best_move_idx].item()),
                int(c_rep[best_move_idx].item()),
            ] = float(new_u8[best_move_idx].item()) / 255.0
            return chosen.detach(), best_move_abs_margin, best_move_margin, best_move_competitor, True

        return base, best_abs_margin, 0.0, -1, False

    def margin_state_quantized(self, quant, cls):
        with torch.inference_mode():
            logits = self.run_model(quant)
            margin_tensor, competitor_idx = margin_stats(logits, cls)
        margin = float(margin_tensor.item())
        return MarginState(
            abs_margin=abs(margin),
            margin=margin,
            competitor=int(competitor_idx.item()),
        )

    def _prediction_gaps(
        self, logits: torch.Tensor, predictions: torch.Tensor
    ) -> torch.Tensor:
        logits = logits.detach().cpu().to(torch.float32)
        predictions = predictions.detach().cpu().to(torch.long)
        rows = torch.arange(logits.size(0), dtype=torch.long)
        pred_values = logits[rows, predictions]
        masked = logits.clone()
        masked[rows, predictions] = -float("inf")
        runner_up, _ = masked.max(dim=1)
        return (pred_values - runner_up).abs()

    def _runner_up_edges(
        self,
        logits: torch.Tensor,
        predictions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = logits.detach().cpu().to(torch.float32)
        predictions = predictions.detach().cpu().to(torch.long)
        rows = torch.arange(logits.size(0), dtype=torch.long)
        pred_values = logits[rows, predictions]
        masked = logits.clone()
        masked[rows, predictions] = -float("inf")
        runner_values, runner_idx = masked.max(dim=1)
        return runner_idx.to(torch.long), (pred_values - runner_values).abs()

    def _update_from_oracle_logits(
        self,
        *,
        blis_predictions: torch.Tensor,
        openblas_predictions: torch.Tensor,
        blis_logits: torch.Tensor,
        openblas_logits: torch.Tensor,
    ) -> None:
        if self.last_candidates is None:
            return

        if self.config.oracle_guided_seed:
            blis_gap = self._prediction_gaps(blis_logits, blis_predictions)
            openblas_gap = self._prediction_gaps(openblas_logits, openblas_predictions)
            oracle_gap = torch.minimum(blis_gap, openblas_gap)
            idx = int(torch.argmin(oracle_gap).item())
            if idx < int(self.last_candidates.size(0)):
                self.current_tensor = self.last_candidates[idx : idx + 1].to(self.device)
                self.trace(
                    "oracle_guided_seed",
                    round=self.round_idx,
                    candidate=idx,
                    oracle_gap=float(oracle_gap[idx].item()),
                    blis_gap=float(blis_gap[idx].item()),
                    openblas_gap=float(openblas_gap[idx].item()),
                    prediction=int(blis_predictions[idx].item()),
                )

        self._build_oracle_bridge_queue(
            blis_predictions=blis_predictions,
            openblas_predictions=openblas_predictions,
            blis_logits=blis_logits,
            openblas_logits=openblas_logits,
        )

    def _build_oracle_bridge_queue(
        self,
        *,
        blis_predictions: torch.Tensor,
        openblas_predictions: torch.Tensor,
        blis_logits: torch.Tensor,
        openblas_logits: torch.Tensor,
    ) -> None:
        self.oracle_bridge_queue = []
        if self.last_candidates is None:
            return

        budget = max(0, int(self.config.oracle_bridge_candidates))
        if budget == 0:
            return

        blis_predictions = blis_predictions.detach().cpu().to(torch.long)
        openblas_predictions = openblas_predictions.detach().cpu().to(torch.long)
        agreed = blis_predictions == openblas_predictions
        if not bool(torch.any(agreed).item()):
            return

        blis_runner, blis_gap = self._runner_up_edges(blis_logits, blis_predictions)
        openblas_runner, openblas_gap = self._runner_up_edges(
            openblas_logits, openblas_predictions
        )
        use_blis = blis_gap <= openblas_gap

        directed = {}
        for idx in torch.nonzero(agreed, as_tuple=False).flatten().tolist():
            idx = int(idx)
            top = int(blis_predictions[idx].item())
            blis_top_runner = int(blis_runner[idx].item())
            openblas_top_runner = int(openblas_runner[idx].item())
            if bool(use_blis[idx].item()):
                runner = blis_top_runner
                gap = float(blis_gap[idx].item())
                source = "blis"
            else:
                runner = openblas_top_runner
                gap = float(openblas_gap[idx].item())
                source = "openblas"
            if runner == top:
                continue
            directed.setdefault((top, runner), []).append(
                {"idx": idx, "gap": gap, "source": source}
            )

        for candidates in directed.values():
            candidates.sort(key=lambda item: (item["gap"], item["idx"]))

        bridge_pairs = []
        seen_keys = set()
        for (left_cls, right_cls), left_candidates in directed.items():
            if (right_cls, left_cls) in seen_keys:
                continue
            right_candidates = directed.get((right_cls, left_cls))
            if not right_candidates:
                continue
            seen_keys.add((left_cls, right_cls))
            max_rank = min(3, len(left_candidates), len(right_candidates))
            for rank in range(max_rank):
                left_item = left_candidates[rank]
                right_item = right_candidates[rank]
                bridge_pairs.append(
                    (
                        left_item["gap"] + right_item["gap"],
                        rank,
                        left_cls,
                        right_cls,
                        left_item,
                        right_item,
                    )
                )

        bridge_pairs.sort()
        if not bridge_pairs:
            return

        alphas = np.linspace(0.0, 1.0, budget + 2, dtype=np.float32)[1:-1]
        alphas = sorted(
            (float(alpha) for alpha in alphas), key=lambda alpha: abs(alpha - 0.5)
        )
        seen = set()
        alpha_pos = {pair_idx: 0 for pair_idx in range(len(bridge_pairs))}
        pair_pos = 0
        attempts = 0
        while len(self.oracle_bridge_queue) < budget and attempts < budget * 4:
            attempts += 1
            pair_idx = pair_pos % len(bridge_pairs)
            pair_pos += 1
            alpha_idx = alpha_pos[pair_idx]
            if alpha_idx >= len(alphas):
                continue
            alpha_pos[pair_idx] = alpha_idx + 1
            alpha = alphas[alpha_idx]
            _, rank, left_cls, right_cls, left_item, right_item = bridge_pairs[pair_idx]
            left_idx = int(left_item["idx"])
            right_idx = int(right_item["idx"])
            left = self.last_candidates[left_idx : left_idx + 1]
            right = self.last_candidates[right_idx : right_idx + 1]
            candidate = quantize_tensor((1.0 - alpha) * left + alpha * right)
            key = candidate.mul(255.0).round().to(torch.uint8).numpy().tobytes()
            if key in seen:
                continue
            seen.add(key)
            self.oracle_bridge_queue.append(
                CandidateProposal(candidate.detach().cpu(), "bridge")
            )
            self.trace(
                "oracle_bridge_candidate_queued",
                round=self.round_idx,
                left_candidate=left_idx,
                right_candidate=right_idx,
                left_prediction=left_cls,
                right_prediction=right_cls,
                left_runner_up=right_cls,
                right_runner_up=left_cls,
                left_oracle_gap=left_item["gap"],
                right_oracle_gap=right_item["gap"],
                left_oracle_source=left_item["source"],
                right_oracle_source=right_item["source"],
                pair_rank=rank,
                alpha=alpha,
            )

    def _candidate_distance_to_original(self, idx: int):
        original = quantize_tensor(self.base_tensor).detach().cpu()
        candidate = self.last_candidates[idx : idx + 1].detach().cpu()
        diff = (candidate - original).abs()
        return {
            "l0": int((diff > 0).sum().item()),
            "l1": float(diff.sum().item()),
            "linf": float(diff.max().item()) if diff.numel() else 0.0,
        }

    def _select_closest_disagreement(self, disagreements: torch.Tensor):
        disagreement_indices = [
            int(idx.item()) for idx in torch.nonzero(disagreements, as_tuple=False)
        ]
        ranked = []
        for idx in disagreement_indices:
            distance = self._candidate_distance_to_original(idx)
            ranked.append((distance["l1"], distance["l0"], distance["linf"], idx, distance))
        ranked.sort()
        return ranked[0][3], [item[4] | {"candidate": item[3]} for item in ranked]

    def sweep_rounds(self, tensor, cls, state, rounds, *, round_idx=None, candidate_idx=None):
        candidate = quantize_tensor(tensor.detach())
        current_state = state
        for sweep_idx in range(max(0, int(rounds))):
            swept_tensor, swept_abs_margin, swept_margin, swept_competitor, improved = (
                self.pixel_sweep(candidate, cls, current_state.abs_margin)
            )
            if not improved:
                self.trace(
                    "sweep_no_improvement",
                    round=round_idx,
                    candidate=candidate_idx,
                    sweep=sweep_idx,
                    abs_margin=current_state.abs_margin,
                    competitor=current_state.competitor,
                )
                break
            candidate = swept_tensor
            current_state = MarginState(
                swept_abs_margin, swept_margin, swept_competitor
            )
            self.trace(
                "sweep_round",
                round=round_idx,
                candidate=candidate_idx,
                sweep=sweep_idx,
                abs_margin=current_state.abs_margin,
                competitor=current_state.competitor,
            )
            if current_state.abs_margin <= self.config.target_abs_margin:
                self.trace(
                    "sweep_target_reached",
                    round=round_idx,
                    candidate=candidate_idx,
                    sweep=sweep_idx,
                    abs_margin=current_state.abs_margin,
                )
                break
        return candidate, current_state

    def _candidate_for_probe_slot(self, cand_idx: int) -> CandidateProposal:
        if self.last_candidates is None and cand_idx == 0:
            return CandidateProposal(self.current_tensor, "seed")

        if self.oracle_bridge_queue:
            proposal = self.oracle_bridge_queue.pop(0)
            self.trace(
                "oracle_bridge_candidate_used",
                round=self.round_idx,
                candidate=cand_idx,
                remaining=len(self.oracle_bridge_queue),
            )
            return CandidateProposal(proposal.tensor.to(self.device), proposal.source)

        return CandidateProposal(
            self.tangent_kick_quantized(
                self.current_tensor,
                self.original_cls,
                round_idx=self.round_idx,
                candidate_idx=cand_idx,
            ),
            "tangent",
        )

    def next_probe_batch(self) -> Dict[str, object]:
        if self.done:
            result = self.result_tensor if self.result_tensor is not None else self.best_tensor
            return self.with_trace({
                "done": True,
                "success": self.success,
                "candidate": result.detach().cpu(),
                "best_abs_margin": self.best_state.abs_margin,
            })

        if self.round_idx >= int(self.config.walk_rounds):
            self.done = True
            self.result_tensor = self.best_tensor
            self.trace(
                "search_done",
                reason="max_rounds",
                round=self.round_idx,
                success=self.success,
                best_abs_margin=self.best_state.abs_margin,
            )
            return self.next_probe_batch()

        self.trace(
            "probe_batch_start",
            round=self.round_idx,
            probe_batch_size=int(self.config.probe_batch_size),
            best_abs_margin=self.best_state.abs_margin,
        )
        candidates = []
        states = []
        for cand_idx in range(int(self.config.probe_batch_size)):
            proposal = self._candidate_for_probe_slot(cand_idx)
            candidate = proposal.tensor
            candidate = quantize_tensor(candidate)
            state = self.margin_state_quantized(candidate, self.original_cls)
            self.trace(
                "candidate_pre_sweep",
                round=self.round_idx,
                candidate=cand_idx,
                source=proposal.source,
                abs_margin=state.abs_margin,
                competitor=state.competitor,
            )
            if state.abs_margin > self.config.target_abs_margin:
                rounds = (
                    self.config.sweep_rounds_first
                    if self.round_idx == 0 and cand_idx == 0
                    else self.config.sweep_rounds_per_step
                )
                candidate, state = self.sweep_rounds(
                    candidate,
                    self.original_cls,
                    state,
                    rounds,
                    round_idx=self.round_idx,
                    candidate_idx=cand_idx,
                )
            else:
                self.trace(
                    "candidate_skip_sweep",
                    round=self.round_idx,
                    candidate=cand_idx,
                    source=proposal.source,
                    reason="target_reached",
                    abs_margin=state.abs_margin,
                    target_abs_margin=self.config.target_abs_margin,
                )

            if state.abs_margin + 1e-12 < self.best_state.abs_margin:
                self.best_tensor = candidate
                self.best_state = state
                self.trace(
                    "new_best_candidate",
                    round=self.round_idx,
                    candidate=cand_idx,
                    abs_margin=state.abs_margin,
                    competitor=state.competitor,
                )

            candidates.append(candidate.squeeze(0).detach().cpu())
            states.append(state)

        self.last_candidates = torch.stack(candidates, dim=0)
        self.last_states = states
        self.current_tensor = self.best_tensor
        self.trace(
            "probe_batch_ready",
            round=self.round_idx,
            best_abs_margin=self.best_state.abs_margin,
        )
        return self.with_trace({
            "done": False,
            "round": self.round_idx,
            "original_class": self.original_cls,
            "candidates": self.last_candidates,
            "candidate_abs_margins": torch.tensor(
                [s.abs_margin for s in states], dtype=torch.float32
            ),
            "best_abs_margin": self.best_state.abs_margin,
        })

    def update_with_probe_results(
        self,
        *,
        blis_predictions: torch.Tensor,
        openblas_predictions: torch.Tensor,
        blis_logits: torch.Tensor,
        openblas_logits: torch.Tensor,
    ) -> Dict[str, object]:
        if self.last_candidates is None:
            raise RuntimeError("No pending candidates; call next_probe_batch first")

        blis_predictions = blis_predictions.detach().cpu().to(torch.long)
        openblas_predictions = openblas_predictions.detach().cpu().to(torch.long)
        disagreements = blis_predictions != openblas_predictions
        self.trace(
            "probe_result",
            round=self.round_idx,
            blis_predictions=blis_predictions.tolist(),
            openblas_predictions=openblas_predictions.tolist(),
            disagreement=disagreements.tolist(),
        )
        if bool(torch.any(disagreements).item()):
            idx, disagreement_distances = self._select_closest_disagreement(
                disagreements
            )
            chosen_distance = self._candidate_distance_to_original(idx)
            self.success = True
            self.done = True
            self.result_index = idx
            self.result_tensor = self.last_candidates[idx : idx + 1].to(self.device)
            if idx < len(self.last_states):
                self.best_state = self.last_states[idx]
            self.trace(
                "search_done",
                reason="chimera_found",
                round=self.round_idx,
                candidate=idx,
                disagreement_candidates=[
                    int(item.item())
                    for item in torch.nonzero(disagreements, as_tuple=False)
                ],
                disagreement_distances=disagreement_distances,
                blis_prediction=int(blis_predictions[idx].item()),
                openblas_prediction=int(openblas_predictions[idx].item()),
                best_abs_margin=self.best_state.abs_margin,
            )
            return self.status(
                extra={
                    "chimera_index": idx,
                    "chimera_abs_margin": self.best_state.abs_margin,
                    "chimera_margin": self.best_state.margin,
                    "chimera_competitor": self.best_state.competitor,
                    "chimera_l0_distance": chosen_distance["l0"],
                    "chimera_l1_distance": chosen_distance["l1"],
                    "chimera_linf_distance": chosen_distance["linf"],
                    "blis_prediction": int(blis_predictions[idx].item()),
                    "openblas_prediction": int(openblas_predictions[idx].item()),
                }
            )

        self._update_from_oracle_logits(
            blis_predictions=blis_predictions,
            openblas_predictions=openblas_predictions,
            blis_logits=blis_logits,
            openblas_logits=openblas_logits,
        )
        self.round_idx += 1
        if self.round_idx >= int(self.config.walk_rounds):
            self.done = True
            self.result_tensor = self.best_tensor
            self.trace(
                "search_done",
                reason="max_rounds",
                round=self.round_idx,
                success=self.success,
                best_abs_margin=self.best_state.abs_margin,
            )
        return self.status()

    def status(self, *, extra: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        result = {
            "done": self.done,
            "success": self.success,
            "round": self.round_idx,
            "original_class": self.original_cls,
            "initial_abs_margin": self.initial_state.abs_margin,
            "best_abs_margin": self.best_state.abs_margin,
            "best_margin": self.best_state.margin,
            "best_competitor": self.best_state.competitor,
        }
        if self.result_tensor is not None:
            result["candidate"] = self.result_tensor.detach().cpu()
        if extra:
            result.update(extra)
        return self.with_trace(result)
