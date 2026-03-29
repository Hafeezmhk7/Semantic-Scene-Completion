# -*- coding: utf-8 -*-
"""
asl_pl_module.py  — AlignedShapeAsLatentPLModule
=================================================
CHANGES vs original:
  forward() now accepts scaffold_anchors and scaffold_token_ids (both optional).
  These are passed through to self.shape_model for:
    - Idea 2A (token_cond approach A): scaffold_anchors [B,512,3]
    - Idea 2B (token_cond approach B): uses pred_centroids internally (no extra arg)
    - Idea 3  (query_decoder):         scaffold_anchors + scaffold_token_ids [B,40000]

  If the model flags are inactive the args are simply ignored.
  Fully backward-compatible: old code calling forward(surface,image,text,volume_queries)
  still works because both new args default to None.
"""

from typing import List, Tuple, Dict, Optional
from omegaconf import DictConfig

import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
import pytorch_lightning as pl
from typing import Union
from functools import partial

from model.michelangelo.utils import instantiate_from_config

from .inference_utils import extract_geometry
from .tsal_base import (
    AlignedShapeAsLatentModule,
    ShapeAsLatentModule,
    Latent2MeshOutput,
    AlignedMeshOutput
)


class AlignedShapeAsLatentPLModule(pl.LightningModule):

    def __init__(self, *,
                 shape_module_cfg,
                 aligned_module_cfg,
                 loss_cfg,
                 optimizer_cfg: Optional[DictConfig] = None,
                 ckpt_path: Optional[str] = None,
                 ignore_keys: Union[Tuple[str], List[str]] = ()):

        super().__init__()

        self.shape_model: ShapeAsLatentModule = instantiate_from_config(
            shape_module_cfg, device=None, dtype=None
        )

        self.loss = instantiate_from_config(loss_cfg)
        self.optimizer_cfg = optimizer_cfg

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        self.save_hyperparameters()

    def set_shape_model_only(self):
        self.model.set_shape_model_only()

    @property
    def latent_shape(self):
        return self.model.shape_model.latent_shape

    @property
    def zero_rank(self):
        if self._trainer:
            zero_rank = self.trainer.local_rank == 0
        else:
            zero_rank = True
        return zero_rank

    def init_from_ckpt(self, path, ignore_keys=()):
        state_dict = torch.load(path, map_location="cpu")["state_dict"]

        keys = list(state_dict.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del state_dict[k]

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def configure_optimizers(self) -> Tuple[List, List]:
        lr = self.learning_rate

        trainable_parameters = list(self.model.parameters())

        if self.optimizer_cfg is None:
            optimizers = [torch.optim.AdamW(trainable_parameters, lr=lr, betas=(0.9, 0.99), weight_decay=1e-3)]
            schedulers = []
        else:
            optimizer = instantiate_from_config(self.optimizer_cfg.optimizer, params=trainable_parameters)
            scheduler_func = instantiate_from_config(
                self.optimizer_cfg.scheduler,
                max_decay_steps=self.trainer.max_steps,
                lr_max=lr
            )
            scheduler = {
                "scheduler": lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler_func.schedule),
                "interval": "step",
                "frequency": 1
            }
            optimizers = [optimizer]
            schedulers = [scheduler]

        return optimizers, schedulers

    def forward(self,
                surface: torch.FloatTensor,
                image: torch.FloatTensor,
                text: torch.FloatTensor,
                volume_queries: torch.FloatTensor,
                scaffold_anchors=None,
                scaffold_token_ids=None):
        """
        Forward pass through the model.

        Args:
            surface:            [B, N, feats]   — Gaussian features (pc + params)
            image:              [B, N, feats]   — same as surface (not used separately)
            text:               [B, N, feats]   — same as surface (not used separately)
            volume_queries:     [B, N, 3]       — query positions for geo decoder
            scaffold_anchors:   [B, 512, 3]     — voxel centroid positions (optional)
                                                   Needed for:
                                                   - Idea 2A (token_cond approach A)
                                                   - Idea 3  (query_decoder)
            scaffold_token_ids: [B, 40000]      — per-Gaussian voxel assignment (optional)
                                                   Needed for:
                                                   - Idea 3 (query_decoder)

        Returns:
            shape_embed:            [B, width]
            mu:                     [B, 16384]
            log_var:                [B, 16384]
            z:                      [B, 16384]
            UV_gs_recover:          [B, 560000]
            per_gaussian_features:  [B, 40000, 32] or None
        """
        shape_embed, mu, log_var, z, UV_gs_recover, per_gaussian_features = self.shape_model(
            pc=surface,
            feats=surface,
            volume_queries=volume_queries,
            sample_posterior=True,
            scaffold_anchors=scaffold_anchors,
            scaffold_token_ids=scaffold_token_ids)

        return shape_embed, mu, log_var, z, UV_gs_recover, per_gaussian_features

    def encode(self, surface: torch.FloatTensor, sample_posterior=True):
        pc = surface[..., 0:3]
        feats = surface[..., 3:]

        shape_embed, mu, log_var, shape_zq, posterior = self.shape_model.encode(
            pc=pc, feats=feats, sample_posterior=sample_posterior
        )

        return shape_embed, mu, log_var, shape_zq, posterior

    def decode(self,
               z_q,
               bounds: Union[Tuple[float], List[float], float] = 1.1,
               octree_depth: int = 7,
               num_chunks: int = 10000) -> List[Latent2MeshOutput]:

        latents = self.shape_model.decode(z_q, return_semantic_features=False)
        return latents

    def training_step(self, batch: Dict[str, torch.FloatTensor],
                      batch_idx: int, optimizer_idx: int = 0) -> torch.FloatTensor:

        surface = batch["surface"]
        image = batch["image"]
        text = batch["text"]

        volume_queries = batch["geo_points"][..., 0:3]
        shape_labels = batch["geo_points"][..., -1]

        embed_outputs, shape_logits, posteriors = self(surface, image, text, volume_queries)

        aeloss, log_dict_ae = self.loss(
            **embed_outputs,
            posteriors=posteriors,
            shape_logits=shape_logits,
            shape_labels=shape_labels,
            split="train"
        )

        self.log_dict(log_dict_ae, prog_bar=True, logger=True, batch_size=shape_logits.shape[0],
                      sync_dist=False, rank_zero_only=True)

        return aeloss

    def validation_step(self, batch: Dict[str, torch.FloatTensor], batch_idx: int) -> torch.FloatTensor:

        surface = batch["surface"]
        image = batch["image"]
        text = batch["text"]

        volume_queries = batch["geo_points"][..., 0:3]
        shape_labels = batch["geo_points"][..., -1]

        embed_outputs, shape_logits, posteriors = self(surface, image, text, volume_queries)

        aeloss, log_dict_ae = self.loss(
            **embed_outputs,
            posteriors=posteriors,
            shape_logits=shape_logits,
            shape_labels=shape_labels,
            split="val"
        )
        self.log_dict(log_dict_ae, prog_bar=True, logger=True, batch_size=shape_logits.shape[0],
                      sync_dist=False, rank_zero_only=True)

        return aeloss

    def visual_alignment(self,
                         surface: torch.FloatTensor,
                         image: torch.FloatTensor,
                         text: torch.FloatTensor,
                         description: Optional[List[str]] = None,
                         bounds: Union[Tuple[float], List[float]] = (-1.25, -1.25, -1.25, 1.25, 1.25, 1.25),
                         octree_depth: int = 7,
                         num_chunks: int = 10000) -> List[AlignedMeshOutput]:

        outputs = []

        device = surface.device
        bs = surface.shape[0]

        embed_outputs, shape_z = self.model(surface, image, text)

        image_embed = embed_outputs["image_embed"]
        text_embed = embed_outputs["text_embed"]
        shape_embed = embed_outputs["shape_embed"]

        shape_embed = F.normalize(shape_embed, dim=-1, p=2)
        text_embed = F.normalize(text_embed, dim=-1, p=2)
        image_embed = F.normalize(image_embed, dim=-1, p=2)

        shape_text_similarity = (100.0 * shape_embed @ text_embed.T).softmax(dim=-1)
        shape_image_similarity = (100.0 * shape_embed @ image_embed.T).softmax(dim=-1)

        shape_zq, posterior = self.model.shape_model.encode_kl_embed(shape_z)
        latents = self.model.shape_model.decode(shape_zq)
        geometric_func = partial(self.model.shape_model.query_geometry, latents=latents)

        mesh_v_f, has_surface = extract_geometry(
            geometric_func=geometric_func,
            device=device,
            batch_size=bs,
            bounds=bounds,
            octree_depth=octree_depth,
            num_chunks=num_chunks,
            disable=not self.zero_rank
        )

        for i, ((mesh_v, mesh_f), is_surface) in enumerate(zip(mesh_v_f, has_surface)):
            if not is_surface:
                outputs.append(None)
                continue

            out = AlignedMeshOutput()
            out.mesh_v = mesh_v
            out.mesh_f = mesh_f
            out.surface = surface[i].cpu().numpy()
            out.image = image[i].cpu().numpy()
            if description is not None:
                out.text = description[i]
            out.shape_text_similarity = shape_text_similarity[i, i]
            out.shape_image_similarity = shape_image_similarity[i, i]

            outputs.append(out)

        return outputs

    def latent2mesh(self,
                    latents: torch.FloatTensor,
                    bounds: Union[Tuple[float], List[float], float] = 1.1,
                    octree_depth: int = 7,
                    num_chunks: int = 10000) -> List[Latent2MeshOutput]:

        outputs = []

        geometric_func = partial(self.shape_model.query_geometry, latents=latents)

        device = latents.device
        mesh_v_f, has_surface = extract_geometry(
            geometric_func=geometric_func,
            device=device,
            batch_size=len(latents),
            bounds=bounds,
            octree_depth=octree_depth,
            num_chunks=num_chunks,
            disable=not self.zero_rank
        )

        for i, ((mesh_v, mesh_f), is_surface) in enumerate(zip(mesh_v_f, has_surface)):
            if not is_surface:
                outputs.append(None)
                continue

            out = Latent2MeshOutput()
            out.mesh_v = mesh_v
            out.mesh_f = mesh_f

            outputs.append(out)

        return outputs