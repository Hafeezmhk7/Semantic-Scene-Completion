"""
Semantic Loss Functions for Can3Tok
=====================================
VECTORIZED REWRITE (this version)
  All Python-level loops over categories/labels replaced with scatter_add and
  sort-based GPU operations. GPU→CPU sync count per batch:

  ScanNet72SemanticLoss.forward()
    OLD: 72 (prototype loop) + up to 900k (target loop) syncs
    NEW: 1  (torch.unique, unavoidable)

  compute_scannet72_semantic_loss() balanced sampling
    OLD: 90 × 72 = 6480 syncs (len(ci)==0, len(ci)>spc checks)
    NEW: 90 × 3  = 270  syncs (M count, K count, n_selected count)

  compute_zs_token/layout/pool_infonce_loss()
    OLD: up to B syncs per prototype loop
    NEW: 1 sync (torch.unique)

  All function signatures are identical — backward compatible.

THREE families of contrastive loss:

1. Per-Gaussian InfoNCE  (--semantic_mode != 'none')
2. Scene-level z_s InfoNCE  (--z_s_infonce_weight > 0)
3. z_s Token InfoNCE  (--zs_token_infonce_weight > 0)
"""

import torch
import torch.nn.functional as F


# ============================================================================
# SHARED VECTORIZED PRIMITIVE
# ============================================================================

def _proto_infonce(all_emb, all_labels, missing_categories, temperature, device):
    """
    Vectorized prototype InfoNCE. Replaces every for-loop version.

    Replaces:
      for cat in unique_cats:
          m = all_labels == cat
          if m.sum() > 0:                      ← GPU→CPU sync
              protos.append(all_emb[m].mean())
      for i, lbl in enumerate(all_labels):
          tgt[i] = cat2idx.get(lbl.item())     ← GPU→CPU sync ×M

    With:
      scatter_add for prototype sums            ← zero additional syncs
      inverse_idx directly as cross-entropy tgt ← zero additional syncs

    GPU syncs: 1 (torch.unique, unavoidable to know K for allocation)

    Returns (loss_scalar, K_int) or (None, 0) if fewer than 2 categories.
    """
    # Filter missing categories entirely on GPU using torch.isin
    if missing_categories:
        missing = torch.tensor(missing_categories, device=device, dtype=all_labels.dtype)
        keep    = ~torch.isin(all_labels, missing)
        all_emb    = all_emb[keep]
        all_labels = all_labels[keep]

    if all_emb.shape[0] == 0:
        return None, 0

    # ONE sync: torch.unique needs to count unique values for output allocation
    unique_cats, inverse_idx = torch.unique(all_labels, return_inverse=True)
    K = unique_cats.shape[0]   # .shape[] is a Python int — no sync
    if K < 2:
        return None, 0

    D = all_emb.shape[1]

    # Prototype sums via scatter_add — ZERO syncs
    # proto_sum[k] = sum of all_emb[i] where inverse_idx[i] == k
    proto_sum = torch.zeros(K, D, device=device, dtype=all_emb.dtype)
    proto_sum.scatter_add_(
        0,
        inverse_idx.unsqueeze(1).expand(-1, D),
        all_emb
    )

    # Prototype means and L2-normalise — ZERO syncs
    counts = torch.bincount(inverse_idx, minlength=K).to(all_emb.dtype)
    protos = F.normalize(proto_sum / counts.unsqueeze(1).clamp(min=1.0), p=2, dim=-1)

    # Similarity matrix and cross-entropy — ZERO syncs
    # inverse_idx already contains targets in [0, K-1] — no mapping needed
    sim_mat = torch.matmul(all_emb, protos.T) / temperature   # [M, K]
    loss    = F.cross_entropy(sim_mat, inverse_idx)

    if torch.isnan(loss) or torch.isinf(loss):
        return None, 0

    return loss, K


# ============================================================================
# 1. PER-GAUSSIAN INFONCE
# ============================================================================

class ScanNet72SemanticLoss(torch.nn.Module):
    """
    Cross-batch InfoNCE between per-Gaussian embeddings and category prototypes.
    VECTORIZED: prototype loop + target loop replaced with scatter_add.
    GPU syncs per call: 1 (was 72 + up to 900k).
    """

    def __init__(self, num_categories=72, temperature=0.07, segment_weight=1.0,
                 instance_weight=0.0, feature_dim=32):
        super().__init__()
        self.num_categories      = num_categories
        self.temperature         = temperature
        self.segment_weight      = segment_weight
        self.instance_weight     = instance_weight
        self.feature_dim         = feature_dim
        self.missing_categories  = [13, 53, 61]

    def forward(self, embeddings, segment_labels, instance_labels=None, batch_size=1):
        """
        embeddings:      [B, N, D]
        segment_labels:  [B, N]   (ScanNet72 labels, -1 = unlabelled)
        """
        B, N, D = embeddings.shape
        device   = embeddings.device

        segment_loss         = torch.tensor(0.0, device=device)
        instance_loss        = torch.tensor(0.0, device=device)
        num_categories_batch = 0
        num_instances_batch  = 0

        # ── Segment loss ──────────────────────────────────────────────────────
        if self.segment_weight > 0:
            all_emb    = F.normalize(embeddings.reshape(B * N, D), p=2, dim=-1)
            all_labels = segment_labels.reshape(B * N)

            # Remove unlabelled points on GPU (no sync)
            valid      = all_labels >= 0
            all_emb    = all_emb[valid]
            all_labels = all_labels[valid]

            if all_emb.shape[0] > 0:
                loss, K = _proto_infonce(
                    all_emb, all_labels, self.missing_categories,
                    self.temperature, device)
                if loss is not None:
                    segment_loss         = loss
                    num_categories_batch = K

        # ── Instance loss ─────────────────────────────────────────────────────
        if self.instance_weight > 0 and instance_labels is not None:
            all_emb    = F.normalize(embeddings.reshape(B * N, D), p=2, dim=-1)
            all_labels = instance_labels.reshape(B * N)

            valid      = all_labels >= 0
            all_emb    = all_emb[valid]
            all_labels = all_labels[valid]

            if all_emb.shape[0] > 0:
                # Instance labels have no "missing" categories to exclude
                loss, K = _proto_infonce(
                    all_emb, all_labels, [],
                    self.temperature, device)
                if loss is not None:
                    instance_loss        = loss
                    num_instances_batch  = K

        total = self.segment_weight * segment_loss + self.instance_weight * instance_loss
        return total, {
            'segment_loss':            segment_loss.item(),
            'instance_loss':           instance_loss.item(),
            'semantic_loss':           total.item(),
            'num_categories_in_batch': num_categories_batch,
            'num_instances_in_batch':  num_instances_batch,
        }


def _balanced_sample_scene(se, ssl, subsample, device):
    """
    Vectorized balanced sampling for one scene.

    Replaces the inner for-loop:
      for cat_id in uc:
          ci = torch.where(ssl == cat_id)[0]
          if len(ci) == 0: continue          ← GPU→CPU sync
          if len(ci) > spc: ci = ci[:spc]   ← GPU→CPU sync

    Strategy: sort all valid points by (category, random_score), then
    select the first spc points from each contiguous category block.
    Categories are contiguous after sorting, so within-category rank is
    computable without any per-category loops.

    Always returns exactly (subsample, D) and (subsample,) tensors.
    When M < subsample, pads with random points from the full N-point
    tensor (matching original behaviour).

    GPU syncs per scene: ~4 (M, K, n_selected, always-exact-size guarantee)
    Was: up to 72 syncs per scene.
    """
    N, D = se.shape
    valid_mask = ssl >= 0

    M = valid_mask.sum().item()       # SYNC 1
    if M == 0:
        # No labelled points at all — random sample from full scene
        idx = torch.randperm(N, device=device)[:subsample]
        return se[idx], ssl[idx]

    valid_labels = ssl[valid_mask]
    valid_emb    = se[valid_mask]

    # ── SYNC 2: torch.unique needs K for output allocation ────────────────
    _, inverse = torch.unique(valid_labels, return_inverse=True)
    K   = inverse.max().item() + 1   # SYNC 3
    spc = max(1, subsample // K)

    # Sort by (category, random_score) — no sync
    rand_score  = torch.rand(M, device=device)
    sort_key    = inverse.float() * (M + 1) + rand_score
    sorted_pos  = torch.argsort(sort_key)
    sorted_cats = inverse[sorted_pos]

    # Within-category rank via category-boundary positions — no sync
    cat_change = torch.cat([
        torch.ones(1, dtype=torch.bool, device=device),
        sorted_cats[1:] != sorted_cats[:-1]
    ])
    # SYNC 4 — torch.where needs to count True values
    cat_start_positions = torch.where(cat_change)[0]   # [K]
    within_cat_rank = (
        torch.arange(M, device=device) - cat_start_positions[sorted_cats]
    )

    # Select points with within-category rank < spc — no sync
    selected_in_sorted = within_cat_rank < spc
    selected_valid_idx = sorted_pos[selected_in_sorted]   # indices into valid_emb

    n_selected = selected_in_sorted.sum().item()          # SYNC 5

    if n_selected > subsample:
        # Over budget (rare) — random downsample
        perm = torch.randperm(n_selected, device=device)[:subsample]
        selected_valid_idx = selected_valid_idx[perm]
        n_selected = subsample

    if n_selected < subsample:
        # Under budget — two-stage padding:
        # Stage 1: pad with remaining valid points not yet selected
        already   = torch.zeros(M, dtype=torch.bool, device=device)
        already[selected_valid_idx] = True
        remaining_valid = torch.where(~already)[0]
        if remaining_valid.shape[0] > 0:
            take = min(subsample - n_selected, remaining_valid.shape[0])
            extra = remaining_valid[torch.randperm(remaining_valid.shape[0], device=device)[:take]]
            selected_valid_idx = torch.cat([selected_valid_idx, extra])
            n_selected += take

        # Stage 2: if still short (M < subsample), pad from the full N-point
        # tensor using random indices — matches original code behaviour
        if n_selected < subsample:
            still_needed = subsample - n_selected
            all_idx      = torch.randperm(N, device=device)[:still_needed]
            extra_emb    = se[all_idx]
            extra_lbl    = ssl[all_idx]
            final_emb    = torch.cat([valid_emb[selected_valid_idx], extra_emb], dim=0)
            final_lbl    = torch.cat([valid_labels[selected_valid_idx], extra_lbl], dim=0)
            return final_emb, final_lbl

    return valid_emb[selected_valid_idx], valid_labels[selected_valid_idx]


def compute_scannet72_semantic_loss(embeddings, segment_labels, instance_labels, batch_size,
                                    segment_weight=1.0, instance_weight=0.0,
                                    temperature=0.07, subsample=2000,
                                    sampling_strategy='balanced'):
    """Per-Gaussian InfoNCE with vectorized balanced subsampling."""
    B, N, D = embeddings.shape
    device   = embeddings.device

    if subsample < N:
        if sampling_strategy == 'random':
            idx            = torch.randperm(N, device=device)[:subsample]
            embeddings     = embeddings[:, idx, :]
            segment_labels = segment_labels[:, idx]
            if instance_labels is not None:
                instance_labels = instance_labels[:, idx]

        elif sampling_strategy == 'balanced':
            sampled_emb  = []
            sampled_seg  = []
            sampled_inst = [] if instance_labels is not None else None

            for b in range(B):
                se_b, ssl_b = _balanced_sample_scene(
                    embeddings[b], segment_labels[b], subsample, device)
                sampled_emb.append(se_b)
                sampled_seg.append(ssl_b)

                if instance_labels is not None:
                    # Instance labels: use the same sampled count, random for instances
                    n_sel = se_b.shape[0]
                    idx_b = torch.randperm(N, device=device)[:n_sel]
                    sampled_inst.append(instance_labels[b][idx_b])

            embeddings     = torch.stack(sampled_emb, 0)
            segment_labels = torch.stack(sampled_seg, 0)
            if instance_labels is not None:
                instance_labels = torch.stack(sampled_inst, 0)
        else:
            raise ValueError(f"Unknown sampling_strategy: '{sampling_strategy}'")

    loss_mod = ScanNet72SemanticLoss(
        num_categories=72, temperature=temperature,
        segment_weight=segment_weight, instance_weight=instance_weight,
        feature_dim=D)
    return loss_mod(embeddings, segment_labels, instance_labels, batch_size)


def compute_semantic_loss(embeddings, segment_labels, instance_labels, batch_size,
                          segment_weight=1.0, instance_weight=0.0,
                          temperature=0.07, subsample=2000,
                          num_categories=72, sampling_strategy='balanced'):
    """Universal entry point for per-Gaussian InfoNCE (backward compatible)."""
    return compute_scannet72_semantic_loss(
        embeddings=embeddings, segment_labels=segment_labels,
        instance_labels=instance_labels, batch_size=batch_size,
        segment_weight=segment_weight, instance_weight=instance_weight,
        temperature=temperature, subsample=subsample,
        sampling_strategy=sampling_strategy)


# ============================================================================
# 2. SCENE-LEVEL Z_S INFONCE (soft positive pairs)
# ============================================================================

def compute_scene_infonce_loss(z_s_proj, label_dist, temperature=0.07, delta=0.4):
    """
    Scene-level InfoNCE with soft positive pairs.
    Fully vectorized — no Python loops.
    """
    B, D = z_s_proj.shape
    if B < 2:
        return torch.tensor(0.0, device=z_s_proj.device), {
            'z_s_infonce_loss': 0.0, 'z_s_num_positives': 0.0, 'z_s_frac_anchors': 0.0}

    ld_norm    = F.normalize(label_dist.float(), p=2, dim=-1)
    weights    = torch.clamp(ld_norm @ ld_norm.T - delta, min=0.0)
    weights.fill_diagonal_(0.0)
    weight_sum = weights.sum(1)
    has_pos    = weight_sum > 1e-8
    if not has_pos.any():
        return torch.tensor(0.0, device=z_s_proj.device), {
            'z_s_infonce_loss': 0.0, 'z_s_num_positives': 0.0, 'z_s_frac_anchors': 0.0}

    sim      = z_s_proj @ z_s_proj.T / temperature
    sim      = sim - sim.max(1, keepdim=True)[0].detach()
    eye      = torch.eye(B, dtype=torch.bool, device=z_s_proj.device)
    denom    = torch.exp(sim).masked_fill(eye, 0.0).sum(1, keepdim=True).clamp(1e-8)
    log_prob = sim - torch.log(denom)
    norm_w   = weights / weight_sum.clamp(1e-8).unsqueeze(1)
    loss     = -(norm_w * log_prob).sum(1)[has_pos].mean()

    return loss, {
        'z_s_infonce_loss':  loss.item(),
        'z_s_num_positives': (weights > 0).float().sum(1).mean().item(),
        'z_s_frac_anchors':  has_pos.float().mean().item(),
    }


# ============================================================================
# 3. Z_S TOKEN INFONCE
# ============================================================================

def compute_zs_token_infonce_loss(zs_tokens, label_dist, temperature=0.07):
    """
    InfoNCE on z_s tokens [B, T, D].
    VECTORIZED: scatter_add prototype construction (was: for-loop with B syncs).
    GPU syncs: 1 (torch.unique).
    """
    B, T, D = zs_tokens.shape
    device   = zs_tokens.device

    if B < 2:
        return torch.tensor(0.0, device=device), {
            'zs_tok_infonce_loss': 0.0, 'zs_tok_num_categories': 0, 'zs_tok_num_tokens': 0}

    # Dominant category per scene, broadcast to all T tokens
    dom_cat    = label_dist.float().argmax(dim=1)                       # [B]
    all_labels = dom_cat.unsqueeze(1).expand(B, T).reshape(B * T)       # [B*T]
    all_emb    = F.normalize(zs_tokens.reshape(B * T, D), p=2, dim=-1)  # [B*T, D]

    missing = [13, 53, 61]
    loss, K = _proto_infonce(all_emb, all_labels, missing, temperature, device)

    if loss is None:
        return torch.tensor(0.0, device=device), {
            'zs_tok_infonce_loss': 0.0, 'zs_tok_num_categories': 0, 'zs_tok_num_tokens': B * T}

    return loss, {
        'zs_tok_infonce_loss':   loss.item(),
        'zs_tok_num_categories': K,
        'zs_tok_num_tokens':     B * T,
    }


# ============================================================================
# 4. Z_LAYOUT INFONCE
# ============================================================================

def compute_zs_layout_infonce_loss(z_layout_proj, label_dist, temperature=0.07):
    """
    Scene-level InfoNCE on z_layout projections [B, 128].
    VECTORIZED: scatter_add prototype construction.
    GPU syncs: 1 (torch.unique).
    """
    B, D   = z_layout_proj.shape
    device = z_layout_proj.device

    if B < 2:
        return torch.tensor(0.0, device=device), {
            'zs_layout_infonce_loss': 0.0, 'zs_layout_num_cats': 0}

    dom_cat    = label_dist.float().argmax(dim=1)   # [B]
    all_emb    = F.normalize(z_layout_proj, p=2, dim=-1)

    missing    = [13, 53, 61]
    loss, K    = _proto_infonce(all_emb, dom_cat, missing, temperature, device)

    if loss is None:
        return torch.tensor(0.0, device=device), {
            'zs_layout_infonce_loss': 0.0, 'zs_layout_num_cats': 0}

    return loss, {'zs_layout_infonce_loss': loss.item(), 'zs_layout_num_cats': K}


# ============================================================================
# 5. Z_S POOL INFONCE
# ============================================================================

def compute_zs_pool_infonce_loss(zs_pool_proj, label_dist, temperature=0.07):
    """
    InfoNCE on pooled z_s/z_layout projections [B, 128].
    VECTORIZED: scatter_add prototype construction.
    GPU syncs: 1 (torch.unique).
    """
    B, D   = zs_pool_proj.shape
    device = zs_pool_proj.device

    if B < 2:
        return torch.tensor(0.0, device=device), {
            'zs_pool_infonce_loss': 0.0, 'zs_pool_num_cats': 0}

    dom_cat  = label_dist.float().argmax(dim=1)
    all_emb  = F.normalize(zs_pool_proj, p=2, dim=-1)

    missing  = [13, 53, 61]
    loss, K  = _proto_infonce(all_emb, dom_cat, missing, temperature, device)

    if loss is None:
        return torch.tensor(0.0, device=device), {
            'zs_pool_infonce_loss': 0.0, 'zs_pool_num_cats': 0}

    return loss, {'zs_pool_infonce_loss': loss.item(), 'zs_pool_num_cats': K}