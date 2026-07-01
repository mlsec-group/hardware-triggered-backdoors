from dataclasses import dataclass
from typing import Optional


@dataclass
class ChimeraSearchConfig:
    target_abs_margin: float = 4e-7
    probe_abs_margin: float = 2e-6
    sweep_max_batch: int = 256
    sweep_rounds_first: int = 5
    sweep_rounds_per_step: int = 4
    sweep_coords_per_round: Optional[int] = 128
    walk_rounds: int = 20
    probe_batch_size: int = 10
    tangent_pixels: int = 128
    tangent_pool_frac: float = 0.2
    quant_dither_samples: int = 64
    quant_dither_radius: float = 2.0 / 255.0
    quant_dither_rounds: int = 15
    quant_dither_radius_decay: float = 0.5
    gd_steps: int = 200
    gd_step_size: float = 0.02
    gd_momentum: float = 0.9
    gd_epsilon: float = 16.0 / 255.0
    oracle_guided_seed: bool = True
    save_preview_images: bool = True
