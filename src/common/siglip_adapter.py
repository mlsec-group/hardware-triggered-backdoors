import torch
import torch.nn as nn
from torch import inference_mode
from transformers import AutoModel, AutoProcessor
from transformers.models.siglip.modeling_siglip import SiglipModel, SiglipOutput

from common.imagenet_labels import IMAGENET_LABELS


class SigLIPAdapter(nn.Module):
    def __init__(
        self,
        model_name="google/siglip-so400m-patch14-384",
        device="cpu",
        cache_dir=".cache",
    ):
        super().__init__()

        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir

        self.siglip_model: SiglipModel = AutoModel.from_pretrained(
            model_name, cache_dir=cache_dir, local_files_only=True
        )
        self.siglip_model = self.siglip_model.to(device)

        self.processor = AutoProcessor.from_pretrained(
            self.model_name, cache_dir=self.cache_dir, local_files_only=True
        )

        self.label_outputs = None
        self.label_embeds = None

    @inference_mode(False)
    @torch.no_grad()
    def get_label_embed(self):
        # Precompute text prompts from ImageNet labels
        labels = [f"a photo of a {label}" for label in IMAGENET_LABELS]

        label_inputs = self.processor(
            text=labels, padding="max_length", return_tensors="pt"
        )
        input_ids = {k: v.to(self.device) for k, v in label_inputs.items()}["input_ids"]

        label_outputs = self.siglip_model.text_model(
            input_ids=input_ids,
            attention_mask=None,
            position_ids=None,
        )
        label_embeds = label_outputs.pooler_output
        label_embeds = label_embeds / label_embeds.norm(p=2, dim=-1, keepdim=True)

        self.label_outputs = label_outputs
        self.label_embeds = label_embeds

        return label_outputs, label_embeds

    def forward(self, x):
        """
        x: torch.Tensor of shape (n_batch, 3, 224, 224)
        returns: logits of shape (n_batch, num_classes)
        """

        model = self.siglip_model

        with torch.no_grad():
            if self.label_outputs is None or self.label_embeds is None:
                label_outputs, label_embeds = self.get_label_embed()
            else:
                label_outputs, label_embeds = (self.label_outputs, self.label_embeds)

        # Convert tensor -> list of PIL images (processor expects this)
        # Normalize to [0, 255] if needed
        images = []
        for img in x:
            img = img.detach().cpu()
            img = (img - img.min()) / (img.max() - img.min() + 1e-6)
            img = (img * 255).clamp(0, 255).byte()
            img = img.permute(1, 2, 0).numpy()
            images.append(img)

        image_inputs = self.processor(images=images, return_tensors="pt")

        # Move everything to device
        pixel_values = {k: v.to(self.device) for k, v in image_inputs.items()}[
            "pixel_values"
        ]

        # SiglipModel::forward
        # def forward(
        #     self,
        #     input_ids: torch.LongTensor | None = None,
        #     pixel_values: torch.FloatTensor | None = None,
        #     attention_mask: torch.Tensor | None = None,
        #     position_ids: torch.LongTensor | None = None,
        #     return_loss: bool | None = None,
        #     interpolate_pos_encoding: bool = False,
        #     **kwargs: Unpack[TransformersKwargs],
        # ) -> SiglipOutput:
        vision_outputs = model.vision_model(
            pixel_values=pixel_values,
            interpolate_pos_encoding=False,
        )

        image_embeds = vision_outputs.pooler_output

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logits_per_text = torch.matmul(
            label_embeds, image_embeds.t().to(label_embeds.device)
        )

        logit_scale, logit_bias = model.logit_scale.to(
            label_embeds.device
        ), model.logit_bias.to(label_embeds.device)
        logits_per_text = logits_per_text * logit_scale.exp() + logit_bias

        logits_per_image = logits_per_text.t()

        return SiglipOutput(
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=label_embeds,
            image_embeds=image_embeds,
            text_model_output=label_outputs,
            vision_model_output=vision_outputs,
        ).logits_per_image
