"""
stage2/stage1_bridge.py
=======================
Schema-aware bridge between the SEVEN Stage 1 experiments and Stage 2.

The Stage 2 latent has TWO possible layouts depending on the Stage 1 flags. The
old code assumed a third (z_s as a 16-token prefix inside the 512), which NONE of
the seven experiments use, which is why the old loader/encoder break on them.

  FLAT   (local_disentangle=False): the encoder produces ONE latent of 512 tokens
         (mu has 512*embed_dim numbers). There is no separate z_s. Experiments 1-5.
         Within FLAT, 'structured' (local_encoder or structured_latent True, exp 5)
         means token k is a spatial Hilbert block; otherwise (1-4) token k is a
         global mixture.

  SPLIT  (local_disentangle=True): mu = cat([mu_s (semantic_dims), mu_g (512*ed)]).
         z_s = mu_s reshaped to [16,32] is a SEPARATE stochastic layout code;
         z_g = mu_g reshaped to [512,32] is the geometry the decoder reconstructs,
         with z_s injected via cross-attention. Experiments 6-7.

This module centralises:
  read_stage1_flags / detect_schema / is_structured
  load_stage1            -- reads ALL relevant flags (incl. local_encoder,
                            structured_latent, local_disentangle, token_local_decoder,
                            position_scaffold, num_gaussians, embed_dim), calls
                            set_num_gaussians, and rebuilds the exact architecture.
  encode_clean           -- frozen-encoder clean targets (mode, not sampled).
  decode_latent          -- schema-correct decode (z_layout for SPLIT, scaffold
                            token ids for position_scaffold).
  build_scaffold_token_ids
  build_stage2_model     -- factory over {scene, layout, geometry, completion}.
  validate_stage_for_schema
  stage1_data_kwargs     -- dataset kwargs mirroring the Stage 1 input distribution.
"""

import math
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.michelangelo.utils import instantiate_from_config
from model.michelangelo.utils.misc import get_config_from_file
from model.michelangelo.models.tsal.sal_perceiver_dist_changes import set_num_gaussians

from stage2.models.layout_dit      import LayoutDiT_models
from stage2.models.geometry_dit512 import GeometryDiT512_models
from stage2.models.flat_dit        import SceneDiT_models, CompletionDiTUncond_models, DCHead
from stage2.models.completion_dit  import CompletionDiT_models


# Defaults match the Stage 1 argparse defaults, so a key missing from an older
# checkpoint resolves to the same value Stage 1 would have used.
_FLAG_DEFAULTS = {
    # architecture
    'color_residual': False, 'latent_disentangle': False, 'semantic_dims': 512,
    'decoder_fourier_pe': False, 'decoder_layout_cross_attn': False,
    'decoder_layout_additive': False, 'decoder_zs_cross_attn': False,
    'structured_layout_tokens': False, 'scene_layout_head': False,
    'scene_semantic_head': False, 'semantic_token_heads': False,
    'position_layout_residual': False,
    'token_cond': False, 'token_cond_approach': 'B', 'token_cond_adaln': False,
    # the NEW experiment flags the old loader ignored
    'local_encoder': False, 'local_window': 1, 'structured_latent': False,
    'local_disentangle': False, 'token_local_decoder': False,
    'position_scaffold': False, 'scaffold_mode': 'voxel',
    'anchor_relative_decode': False, 'anchor_teacher_force': False,
    'offset_scale_init': 2.0, 'num_gaussians': 10000, 'embed_dim': 32,
    # data config needed to reproduce the encoder's input distribution
    'morton_order': False, 'order_curve': 'hilbert', 'order_frame_radius': 10.0,
    'crop_percentile': 100.0, 'scale_norm_mode': 'linear',
    'use_canonical_norm': True, 'chunk_norm_factor': True, 'sampling_method': 'opacity',
}


def read_stage1_flags(ckpt: dict) -> dict:
    return {k: ckpt.get(k, dv) for k, dv in _FLAG_DEFAULTS.items()}


def detect_schema(flags: dict) -> str:
    """'split' when there is a separate stochastic z_s (local_disentangle),
    otherwise 'flat'. The old latent_disentangle / decoder_*_cross_attn schemas are
    not produced by the seven experiments and are treated as flat here."""
    return 'split' if flags['local_disentangle'] else 'flat'


def is_structured(flags: dict) -> bool:
    """True when token k is a spatial Hilbert block (local encoder / structured
    latent). Determines whether completion is meaningful in the FLAT case."""
    return bool(flags['local_encoder'] or flags['structured_latent'])


# ============================================================================
# Stage 1 loading
# ============================================================================

def load_stage1(checkpoint_path: str, config_path: str, device):
    """
    Rebuild and load the frozen Stage 1 model for ANY of the seven experiments.
    Returns (shape_model, flags, schema).
    """
    ckpt  = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    flags = read_stage1_flags(ckpt)

    # CRITICAL: the Gaussian count sizes the fixed token-id buffers AND the decoder
    # output. Must be set BEFORE the model is instantiated.
    set_num_gaussians(int(flags['num_gaussians']))

    model_config = get_config_from_file(config_path).model
    p = model_config.params.shape_module_cfg.params

    # embed_dim is a YAML param (capacity knob); set it explicitly.
    p.embed_dim = int(flags['embed_dim'])

    # Every flag that changes the architecture, read straight from the checkpoint.
    arch = dict(
        color_residual            = flags['color_residual'],
        scene_semantic_head       = flags['scene_semantic_head'],
        position_scaffold         = flags['position_scaffold'],
        latent_disentangle        = flags['latent_disentangle'],
        semantic_dims             = int(flags['semantic_dims']),
        scene_layout_head         = flags['scene_layout_head'],
        decoder_fourier_pe        = flags['decoder_fourier_pe'],
        semantic_token_heads      = flags['semantic_token_heads'],
        token_local_decoder       = flags['token_local_decoder'],
        anchor_relative_decode    = flags['anchor_relative_decode'],
        anchor_teacher_force      = False,    # FORCE off at Stage 2 (no GT anchors)
        offset_scale_init         = float(flags['offset_scale_init']),
        structured_latent         = flags['structured_latent'],
        local_encoder             = flags['local_encoder'],
        local_window              = int(flags['local_window']),
        local_disentangle         = flags['local_disentangle'],
        decoder_zs_cross_attn     = flags['decoder_zs_cross_attn'],
        decoder_layout_cross_attn = flags['decoder_layout_cross_attn'],
        decoder_layout_additive   = flags['decoder_layout_additive'],
        structured_layout_tokens  = flags['structured_layout_tokens'],
        position_layout_residual  = flags['position_layout_residual'],
        token_cond                = flags['token_cond'],
        token_cond_approach       = flags['token_cond_approach'],
        token_cond_adaln          = flags['token_cond_adaln'],
    )
    for k, v in arch.items():
        setattr(p, k, v)

    # Inference-only defaults: the per-Gaussian InfoNCE / seg heads are never called
    # at Stage 2 (we pass return_semantic_features=False), so don't build them.
    p.semantic_mode      = 'none'
    p.predict_seg_labels = False
    p.jepa_idea1         = False
    p.query_decoder      = False
    p.decoder_pos_enc    = False

    stage1 = instantiate_from_config(model_config)
    missing, unexpected = stage1.load_state_dict(ckpt['model_state_dict'], strict=False)
    if missing:
        print(f"  [Stage 1] {len(missing)} missing keys (expected: unused heads)")
    if unexpected:
        print(f"  [Stage 1] {len(unexpected)} unexpected keys (expected: dropped InfoNCE heads)")

    shape_model = stage1.shape_model
    shape_model.to(device).eval()
    for prm in shape_model.parameters():
        prm.requires_grad_(False)

    schema = detect_schema(flags)
    print(f"  Stage 1 loaded: {checkpoint_path}")
    print(f"  schema={schema}  structured={is_structured(flags)}  "
          f"local_disentangle={flags['local_disentangle']}  "
          f"local_encoder={flags['local_encoder']}  "
          f"token_local_decoder={flags['token_local_decoder']}  "
          f"position_scaffold={flags['position_scaffold']}  "
          f"num_gaussians={flags['num_gaussians']}  embed_dim={flags['embed_dim']}")
    return shape_model, flags, schema


# ============================================================================
# Clean-target encoding (frozen encoder, mode not sampled)
# ============================================================================

@torch.no_grad()
def encode_clean(shape_model, features: torch.Tensor, flags: dict, schema: str):
    """
    Returns (z_s_clean, z_g_clean):
      SPLIT -> z_s_clean [B,16,32] (mode of mu_s), z_g_clean [B,512,32] (mode of mu_g)
      FLAT  -> z_s_clean None,      z_g_clean [B,512,32] (the single latent)

    Uses sample_posterior=False so the target is the deterministic mode. We slice
    mu (NOT the model's _z_s_latent, which is always sampled and carries noise).
    """
    B  = features.shape[0]
    ed = int(flags['embed_dim'])
    shape_embed, mu, log_var, z, _ = shape_model.encode(
        pc=features, feats=features, sample_posterior=False)

    if schema == 'split':
        sd   = int(flags['semantic_dims'])
        n_zs = sd // ed
        z_s_clean = mu[:, :sd].reshape(B, n_zs, ed)        # [B, 16, 32]
        z_g_clean = mu[:, sd:].reshape(B, 512, ed)         # [B, 512, 32]
        return z_s_clean, z_g_clean

    z_full = mu.reshape(B, 512, ed)                        # [B, 512, 32]
    return None, z_full


# ============================================================================
# Decoding (schema-correct: z_layout for SPLIT, scaffold ids for position_scaffold)
# ============================================================================

def build_scaffold_token_ids(num_gaussians: int, num_tokens: int = 512, device=None):
    """
    Index-based hilbert_block assignment used at Stage 2: Gaussian i -> block i // g
    with g = ceil(num_gaussians / num_tokens). This depends ONLY on the (Hilbert-
    ordered) Gaussian index, so it is reproducible without GT positions, which is
    exactly why hilbert_block is Stage-2 feasible (voxel mode is not).

    NOTE: this must match gs_dataset_scenesplat's compute_hilbert_block_scaffold.
    It is the natural definition of a contiguous block, and differs from the model's
    fallback FIXED_TOKEN_IDS_512 = arange(N)*512//N. If generated positions look
    striped/scrambled for the scaffold experiments (3, 6), this formula is the first
    thing to check against the dataset.
    """
    g   = math.ceil(num_gaussians / num_tokens)
    ids = (torch.arange(num_gaussians, dtype=torch.long) // g).clamp(max=num_tokens - 1)
    if device is not None:
        ids = ids.to(device)
    return ids


@torch.no_grad()
def decode_latent(shape_model, flags: dict, z_g: torch.Tensor,
                  z_s: torch.Tensor = None, mean_color: torch.Tensor = None):
    """
    Decode a latent into Gaussians [B, num_gaussians, 14] (numpy).
      z_g : [B, 512, 32] geometry / full latent (decoder sequence)
      z_s : [B, 16, 32]  layout latent for SPLIT checkpoints; None for FLAT
    Passes z_layout=z_s (the local_disentangle decoder needs it), and synthesises
    scaffold_token_ids when the checkpoint used position_scaffold.
    """
    import numpy as np
    B   = z_g.shape[0]
    ed  = int(flags['embed_dim'])
    N   = int(flags['num_gaussians'])
    latents = z_g.reshape(B, 512, ed)

    sti = None
    if flags['position_scaffold']:
        sti = build_scaffold_token_ids(N, 512, z_g.device).unsqueeze(0).expand(B, -1)

    recon, _ = shape_model.decode(
        latents, volume_queries=None, return_semantic_features=False,
        shape_embed=None, scaffold_anchors=None, scaffold_token_ids=sti,
        z_layout=z_s)
    preds = recon.reshape(B, N, 14).float().cpu().numpy()
    if flags['color_residual'] and mean_color is not None:
        mc = mean_color.cpu().numpy() if isinstance(mean_color, torch.Tensor) else mean_color
        for i in range(B):
            preds[i, :, 3:6] = np.clip(preds[i, :, 3:6] + mc[i], 0.0, 1.0)
    return preds


# ============================================================================
# Stage 2 model factory + validation
# ============================================================================

def validate_stage_for_schema(stage: str, schema: str, structured: bool):
    if stage == 'scene':
        if schema != 'flat':
            raise ValueError(
                "--stage scene is for FLAT checkpoints (local_disentangle=False). "
                "This checkpoint is SPLIT: use --stage layout then --stage geometry.")
    elif stage == 'dc':
        if schema != 'flat':
            raise ValueError(
                "--stage dc is the DC-colour model for FLAT generation. SPLIT "
                "checkpoints recover DC from z_s via lay_color_head, so no dc stage "
                "is needed there.")
    elif stage in ('layout', 'geometry'):
        if schema != 'split':
            raise ValueError(
                f"--stage {stage} is for SPLIT checkpoints (local_disentangle=True). "
                "This checkpoint is FLAT: use --stage scene.")
    elif stage == 'completion':
        if schema == 'flat' and not structured:
            raise ValueError(
                "Completion needs spatial tokens. This is a global-flat checkpoint "
                "(experiments 1-4): masking a token masks an arbitrary latent "
                "dimension, not a region. Completion is only meaningful for the "
                "structured (experiment 5) and local-disentangle (6, 7) checkpoints.")
    else:
        raise ValueError(f"unknown stage '{stage}'")


def build_stage2_model(schema: str, structured: bool, stage: str, size: str,
                       rope_type: str = 'learned_ape', embed_dim: int = 32):
    """
    stage in {scene, dc, layout, geometry, completion}
      FLAT  generation : scene      -> SceneDiT (512 tokens)
                         dc         -> DCHead   (recovers the DC mean colour)
      SPLIT generation : layout     -> LayoutDiT (16 tokens)
                         geometry   -> GeometryDiT512 (z_g 512 cond z_s 16)
      completion       : SPLIT      -> CompletionDiT (z_s cross-attn)
                         FLAT struct -> CompletionDiTUncond (no z_s)
    """
    if stage == 'scene':
        return SceneDiT_models[f"SceneDiT-{size}"](rope_type=rope_type)
    if stage == 'dc':
        return DCHead(embed_dim=embed_dim)
    if stage == 'layout':
        return LayoutDiT_models[f"LayoutDiT-{size}"]()
    if stage == 'geometry':
        return GeometryDiT512_models[f"GeometryDiT512-{size}"](rope_type=rope_type)
    if stage == 'completion':
        if schema == 'split':
            return CompletionDiT_models[f"CompletionDiT-{size}"]()
        return CompletionDiTUncond_models[f"CompletionDiTUncond-{size}"](rope_type=rope_type)
    raise ValueError(f"unknown stage '{stage}'")


# ============================================================================
# Dataset kwargs mirroring the Stage 1 input distribution
# ============================================================================

def stage1_data_kwargs(flags: dict) -> dict:
    """
    Dataset kwargs (WITHOUT root) so the frozen encoder sees the SAME input
    distribution it was trained on. The order/sampling/frame/Gaussian-count fields
    determine which Gaussians and in what order reach the encoder, so they must
    match Stage 1. The caller sets root per gs_dataset call (the training set is a
    ConcatDataset over the chunk root plus the extra full-scene roots, exactly as in
    Stage 1). Also set gs_dataset.TARGET_POINTS = flags['num_gaussians'] before
    constructing. position_scaffold / scene_layout_head are False because Stage 2
    only needs 'features' and 'mean_color' from the batch.
    """
    return dict(
        resol=100,                                   # matches Stage 1 _ds_kwargs
        sampling_method=flags['sampling_method'],
        normalize=flags['use_canonical_norm'],
        normalize_colors=True,
        use_chunk_norm_factor=flags['chunk_norm_factor'],
        target_radius=10.0,
        scale_norm_mode=flags['scale_norm_mode'],
        label_input=False,
        color_residual=flags['color_residual'],
        position_scaffold=False,
        scaffold_mode=flags['scaffold_mode'],
        scene_layout_head=False,
        crop_percentile=flags['crop_percentile'],
        morton_order=flags['morton_order'],
        order_curve=flags['order_curve'],
        order_frame_radius=flags['order_frame_radius'],
        sample_voxel_res=96,
    )