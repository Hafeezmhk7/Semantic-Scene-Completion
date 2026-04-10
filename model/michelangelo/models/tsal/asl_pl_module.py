# -*- coding: utf-8 -*-
"""asl_pl_module.py — AlignedShapeAsLatentPLModule"""

from typing import List, Tuple, Dict, Optional, Union
from omegaconf import DictConfig
import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
import pytorch_lightning as pl
from functools import partial
from model.michelangelo.utils import instantiate_from_config
from .inference_utils import extract_geometry
from .tsal_base import AlignedShapeAsLatentModule, ShapeAsLatentModule, Latent2MeshOutput, AlignedMeshOutput


class AlignedShapeAsLatentPLModule(pl.LightningModule):

    def __init__(self, *, shape_module_cfg, aligned_module_cfg, loss_cfg,
                 optimizer_cfg: Optional[DictConfig] = None,
                 ckpt_path: Optional[str] = None,
                 ignore_keys: Union[Tuple[str], List[str]] = ()):
        super().__init__()
        self.shape_model: ShapeAsLatentModule = instantiate_from_config(
            shape_module_cfg, device=None, dtype=None)
        self.loss        = instantiate_from_config(loss_cfg)
        self.optimizer_cfg = optimizer_cfg
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.save_hyperparameters()

    @property
    def zero_rank(self):
        return (self.trainer.local_rank == 0) if self._trainer else True

    def init_from_ckpt(self, path, ignore_keys=()):
        state_dict = torch.load(path, map_location="cpu")["state_dict"]
        for k in list(state_dict.keys()):
            for ik in ignore_keys:
                if k.startswith(ik): del state_dict[k]
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        print(f"Restored from {path}: {len(missing)} missing, {len(unexpected)} unexpected")

    def configure_optimizers(self):
        lr     = self.learning_rate
        params = list(self.model.parameters())
        if self.optimizer_cfg is None:
            return [torch.optim.AdamW(params, lr=lr, betas=(0.9,0.99), weight_decay=1e-3)], []
        optimizer = instantiate_from_config(self.optimizer_cfg.optimizer, params=params)
        scheduler_func = instantiate_from_config(
            self.optimizer_cfg.scheduler, max_decay_steps=self.trainer.max_steps, lr_max=lr)
        scheduler = {"scheduler": lr_scheduler.LambdaLR(
                        optimizer, lr_lambda=scheduler_func.schedule),
                     "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def forward(self, surface, image, text, volume_queries,
                scaffold_anchors=None, scaffold_token_ids=None,
                return_semantic_features=None):
        return self.shape_model(
            pc=surface, feats=surface, volume_queries=volume_queries,
            sample_posterior=True,
            scaffold_anchors=scaffold_anchors,
            scaffold_token_ids=scaffold_token_ids,
            return_semantic_features=return_semantic_features)

    def encode(self, surface, sample_posterior=True):
        pc    = surface[..., 0:3]
        feats = surface[..., 3:]
        return self.shape_model.encode(pc=pc, feats=feats, sample_posterior=sample_posterior)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        surface        = batch["surface"]
        volume_queries = batch["geo_points"][..., 0:3]
        shape_labels   = batch["geo_points"][..., -1]
        embed_outputs, shape_logits, posteriors = self(surface, surface, surface, volume_queries)
        aeloss, log_dict = self.loss(
            **embed_outputs, posteriors=posteriors,
            shape_logits=shape_logits, shape_labels=shape_labels, split="train")
        self.log_dict(log_dict, prog_bar=True, logger=True,
                      batch_size=shape_logits.shape[0], sync_dist=False, rank_zero_only=True)
        return aeloss

    def validation_step(self, batch, batch_idx):
        surface        = batch["surface"]
        volume_queries = batch["geo_points"][..., 0:3]
        shape_labels   = batch["geo_points"][..., -1]
        embed_outputs, shape_logits, posteriors = self(surface, surface, surface, volume_queries)
        aeloss, log_dict = self.loss(
            **embed_outputs, posteriors=posteriors,
            shape_logits=shape_logits, shape_labels=shape_labels, split="val")
        self.log_dict(log_dict, prog_bar=True, logger=True,
                      batch_size=shape_logits.shape[0], sync_dist=False, rank_zero_only=True)
        return aeloss

    def latent2mesh(self, latents, bounds=1.1, octree_depth=7, num_chunks=10000):
        outputs = []
        geometric_func = partial(self.shape_model.query_geometry, latents=latents)
        mesh_v_f, has_surface = extract_geometry(
            geometric_func=geometric_func, device=latents.device,
            batch_size=len(latents), bounds=bounds, octree_depth=octree_depth,
            num_chunks=num_chunks, disable=not self.zero_rank)
        for (mesh_v, mesh_f), is_surface in zip(mesh_v_f, has_surface):
            if not is_surface: outputs.append(None); continue
            out = Latent2MeshOutput(); out.mesh_v = mesh_v; out.mesh_f = mesh_f
            outputs.append(out)
        return outputs