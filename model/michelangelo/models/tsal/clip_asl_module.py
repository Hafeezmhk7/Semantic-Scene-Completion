# -*- coding: utf-8 -*-

import torch
from torch import nn
from einops import rearrange
from transformers import CLIPModel

from model.michelangelo.models.tsal.tsal_base import AlignedShapeAsLatentModule


class CLIPAlignedShapeAsLatentModule(AlignedShapeAsLatentModule):

    def __init__(self, *,
                 shape_model,
                 clip_model_version: str = "openai/clip-vit-large-patch14"):

        super().__init__()

        self.clip_model: CLIPModel = CLIPModel.from_pretrained(clip_model_version)
        for params in self.clip_model.parameters():
            params.requires_grad = False

        self.shape_model = shape_model
        self.shape_projection = nn.Parameter(torch.empty(self.shape_model.width, self.clip_model.projection_dim))
        nn.init.normal_(self.shape_projection, std=self.clip_model.projection_dim ** -0.5)

    def set_shape_model_only(self):
        self.clip_model = None

    def encode_shape_embed(self, surface, return_latents: bool = False):
        """

        Args:
            surface (torch.FloatTensor): [bs, n, 3 + c]
            return_latents (bool):

        Returns:
            x (torch.FloatTensor): [bs, projection_dim]
            shape_latents (torch.FloatTensor): [bs, m, d]
        """

        pc = surface[..., 0:3]
        feats = surface[..., 3:]

        shape_embed, shape_latents = self.shape_model.encode_latents(pc, feats)
        x = shape_embed @ self.shape_projection

        if return_latents:
            return x, shape_latents
        else:
            return x

    def encode_image_embed(self, image):
        """

        Args:
            image (torch.FloatTensor): [bs, 3, h, w]

        Returns:
            x (torch.FloatTensor): [bs, projection_dim]
        """

        x = self.clip_model.get_image_features(image)

        return x

    def encode_text_embed(self, text):
        x = self.clip_model.get_text_features(text)
        return x

    def forward(self,
            surface: torch.FloatTensor,
            image: torch.FloatTensor,
            text: torch.FloatTensor,
            volume_queries: torch.FloatTensor):
    
    pc = surface[..., :4]
    feats = surface[..., 4:]
    
    # Encode
    shape_embed, mu, log_var, shape_zq, posterior = self.shape_model.encode(
        pc=surface, feats=surface, sample_posterior=True
    )
    
    # Decode with semantic features if training
    if self.training:
        UV_gs_recover, per_gaussian_features = self.shape_model.decode(
            shape_zq.reshape([shape_zq.shape[0], 512, 32]), 
            volume_queries,
            return_semantic_features=True
        )
    else:
        UV_gs_recover = self.shape_model.decode(
            shape_zq.reshape([shape_zq.shape[0], 512, 32]), 
            volume_queries,
            return_semantic_features=False
        )
        per_gaussian_features = None
    
    return shape_embed, mu, log_var, shape_zq, UV_gs_recover, per_gaussian_features


    def encode(self, surface: torch.FloatTensor, sample_posterior=True):
        pc = surface[..., 0:3]
        feats = surface[..., 3:]

        shape_embed, mu, log_var, shape_zq, posterior = self.shape_model.encode(
            pc=pc, feats=feats, sample_posterior=sample_posterior
        )

        # No per_gaussian_features during encode anymore
        return shape_embed, mu, log_var, shape_zq, posterior, None


    def decode(self,
            z_q,
            bounds: Union[Tuple[float], List[float], float] = 1.1,
            octree_depth: int = 7,
            num_chunks: int = 10000) -> List[Latent2MeshOutput]:

        latents = self.shape_model.decode(z_q, return_semantic_features=False)
        return latents
