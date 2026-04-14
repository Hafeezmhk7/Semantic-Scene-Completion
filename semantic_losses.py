"""
Semantic Loss Functions for Can3Tok
=====================================

THREE families of contrastive loss:

1. Per-Gaussian InfoNCE  (original, --semantic_mode != 'none')
   ─ operates on decoder hidden features → per-Gaussian projections [B, 40000, 32]
   ─ groups by ScanNet72 category ACROSS the batch → cross-batch prototypes → InfoNCE
   ─ flag: --semantic_mode  --segment_loss_weight
   ─ PCA vis: scene{i}_semantic_infonce.ply  (40k points per scene)

2. Scene-level z_s InfoNCE  (--z_s_infonce_weight > 0)
   ─ operates on z_s projection head output [B, 128] (SemanticTokenInfoNCEHead)
   ─ soft positive pairs via cosine similarity of label_dist vectors
   ─ flag: --z_s_infonce_weight  --z_s_infonce_temperature  --z_s_infonce_delta
   ─ PCA vis: z_s_space_epoch_NNN.ply  (one point per eval scene)

3. z_s Token InfoNCE  (NEW, --zs_token_infonce_weight > 0)
   ─ SAME mechanism as per-Gaussian InfoNCE (#1) but applied to the 16 z_s tokens
   ─ each of the 16 tokens of scene i gets labelled with scene i's dominant category
   ─ cross-batch: B×16 tokens grouped by dominant category → prototypes → InfoNCE
   ─ visualisable with PCA exactly like per-Gaussian (B×16 points per eval batch)
   ─ flag: --zs_token_infonce_weight  --zs_token_infonce_temperature
   ─ PCA vis: zs_tokens_epoch_NNN.ply  (16 points per scene, colored by dominant cat)

   WHY THIS IS THE RIGHT APPROACH:
     Per-Gaussian InfoNCE operates on decoder outputs — the gradient reaches z_s
     only indirectly through the reconstruction path. z_s token InfoNCE applies the
     same prototype mechanism DIRECTLY on z_s, giving z_s an unambiguous gradient
     signal: tokens from same-category scenes should cluster.
     The PCA visualisation directly shows whether this clustering is happening,
     exactly as the per-Gaussian PCA shows Gaussian clustering.

References:
  SimCLR (Chen et al., ICML 2020)
  SupCon (Khosla et al., NeurIPS 2020)
  MoCo (He et al., CVPR 2020)
"""

import torch
import torch.nn.functional as F
import numpy as np


# ============================================================================
# 1. PER-GAUSSIAN INFONCE (original, decoder hidden features)
# ============================================================================

class ScanNet72SemanticLoss(torch.nn.Module):
    """
    Cross-batch InfoNCE between per-Gaussian embeddings and category prototypes.

    Groups Gaussians by ScanNet72 category across the whole batch, builds one
    mean prototype per category, then applies InfoNCE pushing each Gaussian
    toward its prototype.

    History:
      per-scene version  → plateau at log(10) = 2.3
      cross-batch (this) → converges to ~0.8
    """

    def __init__(self, num_categories=72, temperature=0.07, segment_weight=1.0,
                 instance_weight=0.0, feature_dim=32):
        super().__init__()
        self.num_categories   = num_categories
        self.temperature      = temperature
        self.segment_weight   = segment_weight
        self.instance_weight  = instance_weight
        self.feature_dim      = feature_dim
        self.missing_categories = [13, 53, 61]   # absent from SceneSplat

    def forward(self, embeddings, segment_labels, instance_labels=None, batch_size=1):
        """
        embeddings:      [B, N, D]  per-Gaussian features (L2-normalised inside)
        segment_labels:  [B, N]     ScanNet72 labels (0-71, -1=unlabelled)
        """
        B, N, D = embeddings.shape

        segment_loss         = torch.tensor(0.0, device=embeddings.device)
        instance_loss        = torch.tensor(0.0, device=embeddings.device)
        num_categories_batch = 0
        num_instances_batch  = 0

        if self.segment_weight > 0:
            all_emb    = embeddings.reshape(B * N, D)
            all_labels = segment_labels.reshape(B * N)
            valid      = all_labels >= 0
            all_emb    = all_emb[valid]
            all_labels = all_labels[valid]

            if len(all_emb) > 0:
                all_emb     = F.normalize(all_emb, p=2, dim=-1)
                unique_cats = [c for c in torch.unique(all_labels).cpu().tolist()
                               if c not in self.missing_categories]

                if len(unique_cats) >= 2:
                    protos, proto_ids = [], []
                    for cat in unique_cats:
                        m = all_labels == cat
                        if m.sum() > 0:
                            protos.append(F.normalize(all_emb[m].mean(0, keepdim=True), p=2, dim=-1))
                            proto_ids.append(cat)

                    if len(protos) >= 2:
                        protos   = F.normalize(torch.cat(protos, 0), p=2, dim=-1)
                        sim_mat  = torch.matmul(all_emb, protos.T) / self.temperature
                        cat2idx  = {c: i for i, c in enumerate(proto_ids)}
                        tgt      = torch.zeros_like(all_labels, dtype=torch.long)
                        for i, lbl in enumerate(all_labels):
                            tgt[i] = cat2idx.get(lbl.item(), -100)
                        seg_l = F.cross_entropy(sim_mat, tgt, ignore_index=-100)
                        if not (torch.isnan(seg_l) or torch.isinf(seg_l)):
                            segment_loss = seg_l
                        num_categories_batch = len(protos)

        if self.instance_weight > 0 and instance_labels is not None:
            all_emb    = embeddings.reshape(B * N, D)
            all_labels = instance_labels.reshape(B * N)
            valid      = all_labels >= 0
            all_emb    = all_emb[valid]
            all_labels = all_labels[valid]

            if len(all_emb) > 0:
                all_emb      = F.normalize(all_emb, p=2, dim=-1)
                unique_insts = torch.unique(all_labels)
                if len(unique_insts) >= 2:
                    inst_feats, inst_ids = [], []
                    for iid in unique_insts:
                        m = all_labels == iid
                        if m.sum() > 0:
                            inst_feats.append(all_emb[m].mean(0, keepdim=True))
                            inst_ids.append(iid.item())
                    if len(inst_feats) >= 2:
                        protos  = F.normalize(torch.cat(inst_feats, 0), p=2, dim=-1)
                        sim_mat = torch.matmul(all_emb, protos.T) / self.temperature
                        i2idx   = {iid: idx for idx, iid in enumerate(inst_ids)}
                        tgt     = torch.zeros_like(all_labels, dtype=torch.long)
                        for i, lbl in enumerate(all_labels):
                            tgt[i] = i2idx.get(lbl.item(), -100)
                        inst_l = F.cross_entropy(sim_mat, tgt, ignore_index=-100)
                        if not (torch.isnan(inst_l) or torch.isinf(inst_l)):
                            instance_loss = inst_l
                        num_instances_batch = len(inst_feats)

        total = self.segment_weight * segment_loss + self.instance_weight * instance_loss
        return total, {
            'segment_loss':            segment_loss.item() if segment_loss > 0 else 0.0,
            'instance_loss':           instance_loss.item() if instance_loss > 0 else 0.0,
            'semantic_loss':           total.item(),
            'num_categories_in_batch': num_categories_batch,
            'num_instances_in_batch':  num_instances_batch,
        }


def compute_scannet72_semantic_loss(embeddings, segment_labels, instance_labels, batch_size,
                                    segment_weight=1.0, instance_weight=0.0,
                                    temperature=0.07, subsample=2000,
                                    sampling_strategy='balanced'):
    """Per-Gaussian InfoNCE with optional balanced subsampling."""
    B, N, D = embeddings.shape

    if subsample < N:
        if sampling_strategy == 'random':
            idx            = torch.randperm(N, device=embeddings.device)[:subsample]
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

                se   = embeddings[b]
                ssl  = segment_labels[b]
                sil  = instance_labels[b] if instance_labels is not None else None
                uc   = torch.unique(ssl[valid_mask])
                spc  = max(1, subsample // len(uc))
                cat_idx = []
                for cat_id in uc:
                    ci = torch.where(ssl == cat_id)[0]
                    if len(ci) == 0: continue
                    if len(ci) > spc:
                        ci = ci[torch.randperm(len(ci), device=embeddings.device)[:spc]]
                    cat_idx.append(ci)

                if cat_idx:
                    combined = torch.cat(cat_idx)
                    if len(combined) < subsample:
                        all_idx   = torch.arange(N, device=embeddings.device)
                        used      = torch.zeros(N, dtype=torch.bool, device=embeddings.device)
                        used[combined] = True
                        remaining = all_idx[~used]
                        if len(remaining) > 0:
                            extra    = min(subsample - len(combined), len(remaining))
                            combined = torch.cat([combined,
                                remaining[torch.randperm(len(remaining),
                                          device=embeddings.device)[:extra]]])
                    if len(combined) > subsample:
                        combined = combined[torch.randperm(len(combined),
                                             device=embeddings.device)[:subsample]]
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
# 2. SCENE-LEVEL Z_S INFONCE (soft positive pairs from label_dist similarity)
# ============================================================================

def compute_scene_infonce_loss(z_s_proj, label_dist, temperature=0.07, delta=0.4):
    """
    Scene-level InfoNCE with soft positive pairs defined by label_dist similarity.

    Operates on z_s projections [B, 128] from SemanticTokenInfoNCEHead.
    Positive pairs: scenes with cos_sim(label_dist_i, label_dist_j) > delta.

    Args:
        z_s_proj:   [B, D_proj]  L2-normalised scene embeddings (from projection head)
        label_dist: [B, 72]      per-scene category distributions
        temperature: float
        delta:       float       min cos_sim threshold for positives (default 0.4)
    """
    B, D = z_s_proj.shape
    if B < 2:
        return torch.tensor(0.0, device=z_s_proj.device), {
            'z_s_infonce_loss': 0.0, 'z_s_num_positives': 0.0, 'z_s_frac_anchors': 0.0}

    ld_norm = F.normalize(label_dist.float(), p=2, dim=-1)
    weights = torch.clamp(ld_norm @ ld_norm.T - delta, min=0.0)
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
# 3. Z_S TOKEN INFONCE  (NEW — same mechanism as per-Gaussian, on z_s tokens)
# ============================================================================

def compute_zs_token_infonce_loss(zs_tokens, label_dist, temperature=0.07):
    """
    InfoNCE on z_s tokens [B, T, D] using SAME cross-batch prototype mechanism
    as per-Gaussian InfoNCE (#1 above).

    MECHANISM (identical to ScanNet72SemanticLoss):
      1. Assign each token a label = dominant ScanNet72 category of its scene
             dom_cat[b] = argmax(label_dist[b])   scalar per scene
             label[b, t] = dom_cat[b]             same label for all 16 tokens
      2. Pool all B×T tokens from across the batch: [B*T, D]
      3. Build one mean prototype per unique dominant category
      4. Apply cross-entropy InfoNCE: each token pushed toward its category prototype

    WHY DOMINANT CATEGORY:
      label_dist gives a full distribution but InfoNCE needs discrete assignments
      for prototype construction. argmax gives the strongest, most discriminative
      scene-level signal. Scenes with identical dominant categories form tight
      positive clusters.

    DIFFERENCE FROM PER-GAUSSIAN InfoNCE:
      Per-Gaussian: B × 40000 points, per-Gaussian ScanNet72 label (fine-grained)
      z_s token:    B × 16 points,    dominant-category scene label (coarse)
      Both use identical cross-batch prototype mechanism → directly comparable PCA.

    VISUALISATION (see pca_feature_visualization.visualize_zs_tokens):
      B×16 points in 3D PCA space, colored by dominant category.
      If InfoNCE working: points from same-category scenes cluster spatially.
      Directly analogous to per-Gaussian PCA PLY (scene{i}_semantic_infonce.ply).

    Args:
        zs_tokens:   [B, T, D]  raw z_s tokens from latent Z[:, :T, :]
                                 (L2-normalised internally — no projection head needed)
        label_dist:  [B, 72]    per-scene category distributions
        temperature: float

    Returns:
        loss:    scalar tensor (0.0 if fewer than 2 dominant categories in batch)
        metrics: dict with keys:
                   zs_tok_infonce_loss      — loss value
                   zs_tok_num_categories    — number of distinct dominant categories
                   zs_tok_num_tokens        — number of tokens used (B*T)
    """
    B, T, D = zs_tokens.shape
    missing_categories = [13, 53, 61]   # absent from SceneSplat (1000-scene analysis)

    if B < 2:
        return torch.tensor(0.0, device=zs_tokens.device), {
            'zs_tok_infonce_loss':   0.0,
            'zs_tok_num_categories': 0,
            'zs_tok_num_tokens':     0,
        }

    # ── Step 1: assign dominant-category label to every token ─────────────────
    # dom_cat[b] = argmax(label_dist[b])  →  all T tokens of scene b get label dom_cat[b]
    dom_cat   = label_dist.float().argmax(dim=1)                # [B]
    all_labels = dom_cat.unsqueeze(1).expand(B, T).reshape(B * T)  # [B*T]

    # ── Step 2: L2-normalise and flatten ──────────────────────────────────────
    all_emb = F.normalize(zs_tokens.reshape(B * T, D), p=2, dim=-1)   # [B*T, D]

    # ── Step 3: cross-batch prototypes (identical to ScanNet72SemanticLoss) ───
    unique_cats = [c for c in torch.unique(all_labels).cpu().tolist()
                   if c not in missing_categories]

    if len(unique_cats) < 2:
        return torch.tensor(0.0, device=zs_tokens.device), {
            'zs_tok_infonce_loss':   0.0,
            'zs_tok_num_categories': len(unique_cats),
            'zs_tok_num_tokens':     B * T,
        }

    protos, proto_ids = [], []
    for cat in unique_cats:
        m = all_labels == cat
        if m.sum() > 0:
            protos.append(F.normalize(all_emb[m].mean(0, keepdim=True), p=2, dim=-1))
            proto_ids.append(cat)

    if len(protos) < 2:
        return torch.tensor(0.0, device=zs_tokens.device), {
            'zs_tok_infonce_loss':   0.0,
            'zs_tok_num_categories': len(protos),
            'zs_tok_num_tokens':     B * T,
        }

    # ── Step 4: InfoNCE cross-entropy ─────────────────────────────────────────
    protos  = F.normalize(torch.cat(protos, 0), p=2, dim=-1)      # [K, D]
    sim_mat = torch.matmul(all_emb, protos.T) / temperature        # [B*T, K]

    cat2idx = {c: i for i, c in enumerate(proto_ids)}
    tgt     = torch.tensor([cat2idx.get(l.item(), -100) for l in all_labels],
                            dtype=torch.long, device=zs_tokens.device)

    loss = F.cross_entropy(sim_mat, tgt, ignore_index=-100)

    if torch.isnan(loss) or torch.isinf(loss):
        loss = torch.tensor(0.0, device=zs_tokens.device)

    return loss, {
        'zs_tok_infonce_loss':   loss.item(),
        'zs_tok_num_categories': len(protos),
        'zs_tok_num_tokens':     B * T,
    }


# ============================================================================
# 4. Z_LAYOUT INFONCE — scene-level pooled prototype InfoNCE (Strategy B)
# ============================================================================

def compute_zs_layout_infonce_loss(z_layout_proj, label_dist, temperature=0.07):
    """
    Scene-level InfoNCE on z_layout projections [B, 128].

    SAME MECHANISM as per-Gaussian InfoNCE but at the scene level:
      Each scene = one point. Label = dominant ScanNet72 category (argmax).
      Cross-batch prototypes per category → InfoNCE cross-entropy.

    INPUT:
      z_layout_proj: [B, 128]  L2-normalised  (from z_layout_infonce_head)
                     Computed as: flatten(z_layout [B,16,32]) → [B,512] → MLP → [B,128]
      label_dist:    [B, 72]   per-scene category distributions
    """
    B, D = z_layout_proj.shape
    missing_categories = [13, 53, 61]

    if B < 2:
        return torch.tensor(0.0, device=z_layout_proj.device), {
            'zs_layout_infonce_loss': 0.0, 'zs_layout_num_cats': 0}

    dom_cat     = label_dist.float().argmax(dim=1)
    unique_cats = [c for c in torch.unique(dom_cat).cpu().tolist()
                   if c not in missing_categories]

    if len(unique_cats) < 2:
        return torch.tensor(0.0, device=z_layout_proj.device), {
            'zs_layout_infonce_loss': 0.0, 'zs_layout_num_cats': len(unique_cats)}

    protos, proto_ids = [], []
    for cat in unique_cats:
        m = dom_cat == cat
        if m.sum() > 0:
            protos.append(F.normalize(z_layout_proj[m].mean(0, keepdim=True), p=2, dim=-1))
            proto_ids.append(cat)

    if len(protos) < 2:
        return torch.tensor(0.0, device=z_layout_proj.device), {
            'zs_layout_infonce_loss': 0.0, 'zs_layout_num_cats': len(protos)}

    protos  = F.normalize(torch.cat(protos, 0), p=2, dim=-1)
    sim_mat = z_layout_proj @ protos.T / temperature

    cat2idx = {c: i for i, c in enumerate(proto_ids)}
    tgt     = torch.tensor([cat2idx.get(l.item(), -100) for l in dom_cat],
                            dtype=torch.long, device=z_layout_proj.device)

    loss = F.cross_entropy(sim_mat, tgt, ignore_index=-100)
    if torch.isnan(loss) or torch.isinf(loss):
        loss = torch.tensor(0.0, device=z_layout_proj.device)

    return loss, {
        'zs_layout_infonce_loss': loss.item(),
        'zs_layout_num_cats':     len(protos),
    }