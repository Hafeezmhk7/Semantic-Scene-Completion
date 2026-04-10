"""
Semantic Loss Functions for Can3Tok
=====================================

Two families of contrastive loss:

1. Per-Gaussian InfoNCE (original, semantic_mode != 'none')
   ─ operates on decoder per-Gaussian features [B, 40000, 32]
   ─ groups Gaussians by ScanNet72 category across the batch
   ─ flag: --semantic_mode  --segment_loss_weight
   ─ enabled via enable_semantic in training loop
   ─ PCA visualisation: scene{i}_semantic_infonce.ply

2. Scene-level z_s InfoNCE (NEW, z_s_infonce_weight > 0)
   ─ operates on z_s projection [B, 128] from SemanticTokenInfoNCEHead
   ─ soft positive pairs defined by cosine similarity of label_dist vectors
   ─ flag: --z_s_infonce_weight  --z_s_infonce_temperature  --z_s_infonce_delta
   ─ PCA visualisation: z_s_space_epoch_NNN.ply (one point per eval scene)

References:
  SimCLR (Chen et al., ICML 2020): cross-batch contrastive
  SupCon (Khosla et al., NeurIPS 2020): supervised contrastive, multiple positives
  PixelCLR (Xie et al., 2021): cross-image dense contrastive
"""

import torch
import torch.nn.functional as F
import numpy as np


# ============================================================================
# 1. PER-GAUSSIAN INFONCE (original, kept for ablation)
# ============================================================================

class ScanNet72SemanticLoss(torch.nn.Module):
    """
    CROSS-BATCH contrastive loss between Gaussian features and category prototypes.

    Operates on per-Gaussian embeddings [B, N, D] produced by SemanticProjectionHead
    (which runs on the decoder's internal hidden state).  Groups Gaussians by
    ScanNet72 category, builds one mean prototype per category across the WHOLE
    batch, then applies InfoNCE pushing each Gaussian toward its prototype.

    HISTORY:
      Original (broken): per-scene loss  → plateau at log(10) = 2.3
      Fixed (this):      cross-batch      → converges to ~0.8
    """

    def __init__(self, num_categories=72, temperature=0.07, segment_weight=1.0,
                 instance_weight=0.0, feature_dim=32):
        super().__init__()
        self.num_categories   = num_categories
        self.temperature      = temperature
        self.segment_weight   = segment_weight
        self.instance_weight  = instance_weight
        self.feature_dim      = feature_dim
        # Categories absent from SceneSplat dataset (from 1000-scene analysis)
        self.missing_categories = [13, 53, 61]

    def forward(self, embeddings, segment_labels, instance_labels=None, batch_size=1):
        """
        Args:
            embeddings:      [B, N, D]  per-Gaussian L2-normalized features
            segment_labels:  [B, N]     ScanNet72 labels (0-71, -1=unlabelled)
            instance_labels: [B, N]     instance IDs (optional)
            batch_size:      B

        Returns:
            total_loss: scalar tensor
            metrics:    dict
        """
        B, N, D = embeddings.shape

        segment_loss          = torch.tensor(0.0, device=embeddings.device)
        instance_loss         = torch.tensor(0.0, device=embeddings.device)
        num_categories_batch  = 0
        num_instances_batch   = 0

        # ── Segment-level cross-batch InfoNCE ─────────────────────────────────
        if self.segment_weight > 0:
            all_emb    = embeddings.reshape(B * N, D)
            all_labels = segment_labels.reshape(B * N)
            valid_mask = all_labels >= 0
            all_emb    = all_emb[valid_mask]
            all_labels = all_labels[valid_mask]

            if len(all_emb) > 0:
                all_emb = F.normalize(all_emb, p=2, dim=-1)
                unique_cats = torch.unique(all_labels).cpu().numpy()
                unique_cats = [c for c in unique_cats
                               if c not in self.missing_categories]

                if len(unique_cats) >= 2:
                    prototypes, proto_ids = [], []
                    for cat_id in unique_cats:
                        mask = all_labels == cat_id
                        if mask.sum() > 0:
                            feat = F.normalize(all_emb[mask].mean(dim=0, keepdim=True), p=2, dim=-1)
                            prototypes.append(feat)
                            proto_ids.append(cat_id)

                    if len(prototypes) >= 2:
                        prototypes = F.normalize(torch.cat(prototypes, dim=0), p=2, dim=-1)
                        sim_mat    = torch.matmul(all_emb, prototypes.T) / self.temperature

                        cat2idx   = {c: i for i, c in enumerate(proto_ids)}
                        tgt       = torch.zeros_like(all_labels, dtype=torch.long)
                        for i, lbl in enumerate(all_labels):
                            tgt[i] = cat2idx.get(lbl.item(), -100)

                        seg_loss  = F.cross_entropy(sim_mat, tgt, ignore_index=-100)
                        if not (torch.isnan(seg_loss) or torch.isinf(seg_loss)):
                            segment_loss = seg_loss
                        num_categories_batch = len(prototypes)

        # ── Instance-level cross-batch InfoNCE ────────────────────────────────
        if self.instance_weight > 0 and instance_labels is not None:
            all_emb    = embeddings.reshape(B * N, D)
            all_labels = instance_labels.reshape(B * N)
            valid_mask = all_labels >= 0
            all_emb    = all_emb[valid_mask]
            all_labels = all_labels[valid_mask]

            if len(all_emb) > 0:
                all_emb     = F.normalize(all_emb, p=2, dim=-1)
                unique_insts = torch.unique(all_labels)

                if len(unique_insts) >= 2:
                    inst_feats, inst_ids = [], []
                    for iid in unique_insts:
                        mask = all_labels == iid
                        if mask.sum() > 0:
                            feat = all_emb[mask].mean(dim=0, keepdim=True)
                            inst_feats.append(feat)
                            inst_ids.append(iid.item())

                    if len(inst_feats) >= 2:
                        prototypes = F.normalize(torch.cat(inst_feats, dim=0), p=2, dim=-1)
                        sim_mat    = torch.matmul(all_emb, prototypes.T) / self.temperature
                        i2idx      = {iid: idx for idx, iid in enumerate(inst_ids)}
                        tgt        = torch.zeros_like(all_labels, dtype=torch.long)
                        for i, lbl in enumerate(all_labels):
                            tgt[i] = i2idx.get(lbl.item(), -100)

                        inst_l = F.cross_entropy(sim_mat, tgt, ignore_index=-100)
                        if not (torch.isnan(inst_l) or torch.isinf(inst_l)):
                            instance_loss = inst_l
                        num_instances_batch = len(prototypes)

        total_loss = self.segment_weight * segment_loss + self.instance_weight * instance_loss
        metrics = {
            'segment_loss':            segment_loss.item() if segment_loss > 0 else 0.0,
            'instance_loss':           instance_loss.item() if instance_loss > 0 else 0.0,
            'semantic_loss':           total_loss.item(),
            'num_categories_in_batch': num_categories_batch,
            'num_instances_in_batch':  num_instances_batch,
        }
        return total_loss, metrics


def compute_scannet72_semantic_loss(embeddings, segment_labels, instance_labels, batch_size,
                                    segment_weight=1.0, instance_weight=0.0,
                                    temperature=0.07, subsample=2000,
                                    sampling_strategy='balanced'):
    """
    Per-Gaussian InfoNCE with optional balanced subsampling.

    SUBSAMPLING STRATEGIES
      'random'   — uniform random, biased toward common categories
      'balanced' — equal samples per category within each scene (recommended)

    RECOMMENDED FOR SceneSplat
      subsample=8000-10000, sampling_strategy='balanced', segment_weight=1.0
    """
    B, N, D = embeddings.shape

    if subsample < N:
        if sampling_strategy == 'random':
            idx = torch.randperm(N, device=embeddings.device)[:subsample]
            embeddings     = embeddings[:, idx, :]
            segment_labels = segment_labels[:, idx]
            if instance_labels is not None:
                instance_labels = instance_labels[:, idx]

        elif sampling_strategy == 'balanced':
            sampled_emb  = []
            sampled_seg  = []
            sampled_inst = [] if instance_labels is not None else None

            for b in range(B):
                valid_mask = segment_labels[b] >= 0
                if valid_mask.sum() == 0:
                    idx = torch.randperm(N, device=embeddings.device)[:subsample]
                    sampled_emb.append(embeddings[b][idx])
                    sampled_seg.append(segment_labels[b][idx])
                    if instance_labels is not None:
                        sampled_inst.append(instance_labels[b][idx])
                    continue

                se       = embeddings[b]
                ssl      = segment_labels[b]
                sil      = instance_labels[b] if instance_labels is not None else None
                unique_c = torch.unique(ssl[valid_mask])
                n_cats   = len(unique_c)
                spc      = max(1, subsample // n_cats)
                cat_idx  = []

                for cat_id in unique_c:
                    ci = torch.where(ssl == cat_id)[0]
                    if len(ci) == 0:
                        continue
                    if len(ci) > spc:
                        perm = torch.randperm(len(ci), device=embeddings.device)[:spc]
                        ci   = ci[perm]
                    cat_idx.append(ci)

                if cat_idx:
                    combined = torch.cat(cat_idx)
                    if len(combined) < subsample:
                        all_idx  = torch.arange(N, device=embeddings.device)
                        used     = torch.zeros(N, dtype=torch.bool, device=embeddings.device)
                        used[combined] = True
                        remaining = all_idx[~used]
                        if len(remaining) > 0:
                            extra = min(subsample - len(combined), len(remaining))
                            perm  = torch.randperm(len(remaining), device=embeddings.device)[:extra]
                            combined = torch.cat([combined, remaining[perm]])
                    if len(combined) > subsample:
                        perm     = torch.randperm(len(combined), device=embeddings.device)[:subsample]
                        combined = combined[perm]
                    sampled_emb.append(se[combined])
                    sampled_seg.append(ssl[combined])
                    if sil is not None:
                        sampled_inst.append(sil[combined])
                else:
                    idx = torch.randperm(N, device=embeddings.device)[:subsample]
                    sampled_emb.append(se[idx])
                    sampled_seg.append(ssl[idx])
                    if sil is not None:
                        sampled_inst.append(sil[idx])

            embeddings     = torch.stack(sampled_emb,  dim=0)
            segment_labels = torch.stack(sampled_seg,  dim=0)
            if instance_labels is not None:
                instance_labels = torch.stack(sampled_inst, dim=0)
        else:
            raise ValueError(f"Unknown sampling_strategy: '{sampling_strategy}'")

    loss_module = ScanNet72SemanticLoss(
        num_categories=72, temperature=temperature,
        segment_weight=segment_weight, instance_weight=instance_weight, feature_dim=D)
    return loss_module(embeddings, segment_labels, instance_labels, batch_size)


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
# 2. SCENE-LEVEL Z_S INFONCE (NEW — soft positive pairs from label_dist)
# ============================================================================

def compute_scene_infonce_loss(z_s_proj, label_dist, temperature=0.07, delta=0.4):
    """
    Scene-level InfoNCE with soft positive pairs defined by label_dist similarity.

    MOTIVATION
      The per-Gaussian InfoNCE operates on decoder hidden features and does not
      directly supervise z_s.  This loss operates on z_s projections [B, 128]
      produced by SemanticTokenInfoNCEHead, providing a direct gradient path to
      the semantic subspace of the latent.

    POSITIVE PAIRS
      Instead of hard scene-category labels (which SceneSplat does not expose),
      we use the cosine similarity between label_dist vectors as a continuous
      soft-positive weight.  Two scenes with similar category distributions are
      treated as positives with weight proportional to their overlap above delta.

      w_ij = max(0, cos_sim(label_dist_i, label_dist_j) - delta)

      Setting delta=0.4 removes pairs that share only background categories
      (wall, floor) which are near-universal in indoor scenes and would otherwise
      inflate the number of false positives.

    FORMULATION (SupCon generalisation, Khosla et al. 2020)
      For each anchor i:
        norm_w_ij  = w_ij / sum_j(w_ij)           (per-anchor normalised weights)
        L_i        = -sum_j norm_w_ij * log P(j|i)
        P(j|i)     = exp(z_i·z_j / tau) / sum_{k!=i} exp(z_i·z_k / tau)

      Anchors with no positive above threshold are excluded from the mean.

    Args:
        z_s_proj:    [B, D_proj]  L2-normalized scene embeddings (from head)
        label_dist:  [B, 72]      empirical category distributions per scene
        temperature: float        InfoNCE temperature (default 0.07)
        delta:       float        min cosine similarity threshold (default 0.4)

    Returns:
        loss:    scalar tensor
        metrics: dict with keys:
                   z_s_infonce_loss      — loss value
                   z_s_num_positives     — avg # positives per anchor
                   z_s_frac_anchors      — fraction of anchors with ≥1 positive
    """
    B, D = z_s_proj.shape

    if B < 2:
        return torch.tensor(0.0, device=z_s_proj.device), {
            'z_s_infonce_loss':  0.0,
            'z_s_num_positives': 0.0,
            'z_s_frac_anchors':  0.0,
        }

    # ── Soft positive weights ─────────────────────────────────────────────────
    ld      = label_dist.float()
    ld_norm = ld / (ld.norm(p=2, dim=-1, keepdim=True) + 1e-8)   # [B, 72]
    sim_ld  = ld_norm @ ld_norm.T                                  # [B, B]

    weights = torch.clamp(sim_ld - delta, min=0.0)                # [B, B]
    weights.fill_diagonal_(0.0)

    weight_sum  = weights.sum(dim=1)                               # [B]
    has_pos     = weight_sum > 1e-8

    if not has_pos.any():
        return torch.tensor(0.0, device=z_s_proj.device), {
            'z_s_infonce_loss':  0.0,
            'z_s_num_positives': 0.0,
            'z_s_frac_anchors':  0.0,
        }

    # ── InfoNCE log-probabilities ─────────────────────────────────────────────
    sim     = z_s_proj @ z_s_proj.T / temperature                 # [B, B]
    sim     = sim - sim.max(dim=1, keepdim=True)[0].detach()       # stability

    eye     = torch.eye(B, dtype=torch.bool, device=z_s_proj.device)
    exp_sim = torch.exp(sim)
    denom   = exp_sim.masked_fill(eye, 0.0).sum(dim=1, keepdim=True).clamp(min=1e-8)

    log_prob = sim - torch.log(denom)                              # [B, B]

    # ── Weighted per-anchor loss ──────────────────────────────────────────────
    norm_w   = weights / weight_sum.clamp(min=1e-8).unsqueeze(1)  # [B, B]
    per_anch = -(norm_w * log_prob).sum(dim=1)                    # [B]
    loss     = per_anch[has_pos].mean()

    # ── Metrics ───────────────────────────────────────────────────────────────
    avg_pos       = (weights > 0).float().sum(dim=1).mean().item()
    frac_anchors  = has_pos.float().mean().item()

    return loss, {
        'z_s_infonce_loss':  loss.item(),
        'z_s_num_positives': avg_pos,
        'z_s_frac_anchors':  frac_anchors,
    }