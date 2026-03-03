"""
distribution_loss.py — Label Distribution Learning loss for Can3Tok
====================================================================
Add this function to semantic_losses.py, or import it from here.

USAGE in gs_can3tok_2.py training loop:
    from distribution_loss import compute_distribution_loss
    
    if args.semantic_mode == 'dist' and per_gaussian_features is not None:
        dist_loss, dist_metrics = compute_distribution_loss(
            dist_logits=per_gaussian_features,    # [B, 72]  — head output
            segment_labels=segment_labels,         # [B, N]   — from dataset
            weight=args.segment_loss_weight,       # γ hyperparameter
        )
"""

import torch
import torch.nn.functional as F


def compute_distribution_loss(
    dist_logits: torch.Tensor,
    segment_labels: torch.Tensor,
    weight: float = 1.0,
) -> tuple:
    """
    Forward KL divergence between predicted and true label distribution.

    For each scene s in the batch:
        p_s(k)  = fraction of Gaussians with ScanNet72 label k  (from segment.npy)
        p̂_s(k) = softmax(dist_logits[s])[k]                    (model prediction)
        loss    = D_KL(p_s ∥ p̂_s) = Σ_k p_s(k) · log(p_s(k) / p̂_s(k))

    Implemented as:
        F.kl_div(log_softmax(dist_logits), p_s, reduction='batchmean')
    which is numerically equivalent and more stable than computing KL directly.

    WHY FORWARD KL (not reverse):
        Forward KL is mode-covering — if p_s(k) > 0 and p̂_s(k) → 0,
        KL → ∞. This severely penalises the model for missing a label that
        is genuinely present in the scene. Correct behaviour: if label_19
        accounts for 78% of a scene's Gaussians, the model must not predict
        near-zero for label_19.

    Args:
        dist_logits:    [B, 72]  raw logits from SemanticDistributionHead
                        (before softmax — log_softmax applied internally)
        segment_labels: [B, N]   integer ScanNet72 labels per Gaussian (0-71)
                        -1 or other negatives treated as unlabelled and excluded
        weight:         γ scalar multiplier (args.segment_loss_weight)

    Returns:
        loss:    scalar tensor  (γ · mean KL over batch)
        metrics: dict with diagnostic values for logging
    """
    B = dist_logits.shape[0]
    device = dist_logits.device

    # ── 1. Build ground-truth label distributions p_s ────────────────────────
    # p_s[b, k] = count(label==k in scene b) / count(label>=0 in scene b)
    p_s = torch.zeros(B, 72, dtype=torch.float32, device=device)

    for b in range(B):
        labels_b = segment_labels[b]           # [N]
        valid    = labels_b >= 0               # exclude unlabelled Gaussians
        n_valid  = valid.sum().item()

        if n_valid == 0:
            # Scene has no valid labels — uniform distribution as fallback
            p_s[b] = 1.0 / 72
            continue

        valid_labels = labels_b[valid].long()
        # bincount is the vectorised way to build the histogram
        counts = torch.bincount(valid_labels, minlength=72).float()
        p_s[b] = counts / n_valid             # normalise → probability simplex

    # ── 2. Compute KL(p_s ∥ p̂_s) ────────────────────────────────────────────
    # F.kl_div expects log-probabilities for the first argument.
    # reduction='batchmean' → sums over labels, averages over batch.
    log_p_hat = F.log_softmax(dist_logits, dim=-1)   # [B, 72]

    kl_loss = F.kl_div(
        input=log_p_hat,      # log p̂_s
        target=p_s,           # p_s  (probabilities, not log)
        reduction='batchmean',
    )

    # ── 3. Diagnostics ───────────────────────────────────────────────────────
    with torch.no_grad():
        p_hat = log_p_hat.exp()                        # [B, 72]

        # Cosine similarity between predicted and true distributions
        cos_sim = F.cosine_similarity(p_hat, p_s, dim=-1).mean().item()

        # Dominant label accuracy: does argmax(p̂) == argmax(p)?
        dominant_acc = (p_hat.argmax(dim=-1) == p_s.argmax(dim=-1)).float().mean().item()

        # Jensen-Shannon divergence (symmetric, bounded in [0, log2])
        m = 0.5 * (p_s + p_hat)
        log_m = m.clamp(min=1e-10).log()
        js_div = 0.5 * (
            F.kl_div(log_m, p_s, reduction='batchmean') +
            F.kl_div(log_m, p_hat, reduction='batchmean')
        ).item()

        # Mean proportion of active labels (labels with p_s > 0)
        active_labels = (p_s > 0).float().sum(dim=-1).mean().item()

    metrics = {
        'dist_kl':         kl_loss.item(),
        'dist_cos_sim':    cos_sim,
        'dist_dom_acc':    dominant_acc,
        'dist_js':         js_div,
        'dist_active_labels': active_labels,
        'semantic_loss':   (weight * kl_loss).item(),
    }

    return weight * kl_loss, metrics