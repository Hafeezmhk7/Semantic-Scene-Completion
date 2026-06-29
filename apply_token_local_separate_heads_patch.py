#!/usr/bin/env python3
"""
apply_token_local_separate_heads_patch.py
==========================================
Plumb --token_local_separate_heads and --token_local_head_cross_stitch
through gs_can3tok_2.py.

The model class (AlignedShapeLatentPerceiver) already ACCEPTS these as
constructor parameters and the TokenLocalDecoder already uses them, but
the training script never surfaced them as CLI flags. So a SLURM job
that passes --token_local_separate_heads crashes with:
    error: unrecognized arguments: --token_local_separate_heads

This patch:
  1. Adds the two argparse arguments
  2. Sets p.token_local_separate_heads / p.token_local_head_cross_stitch
     on the model config (so the model is built with them)
  3. Adds them to _ckpt_meta (so checkpoints record the setting)

Idempotent: re-running detects the sentinel and exits without changes.
"""
import sys
from pathlib import Path

SENTINEL = "# === TOKEN_LOCAL_SEPARATE_HEADS_PATCH_APPLIED ==="

DEFAULT_TARGET = "/home/yli11/scratch-project/Hafeez_thesis/Semantic-Scene-Completion/gs_can3tok_2.py"


def patch(target_path: Path):
    src = target_path.read_text()
    if SENTINEL in src:
        print(f"[skip] sentinel present: {target_path} already patched.")
        return

    # ── 1. ADD ARGPARSE FLAGS ────────────────────────────────────────────────
    # Insert immediately after the existing --token_local_decoder argument.
    arg_anchor = (
        "parser.add_argument('--token_local_decoder', action='store_true', default=False,\n"
        "    help='Replace flat GS_decoder with shared per-token MLP. See token_local_decoder.py.')"
    )
    arg_insert = (
        "\nparser.add_argument('--token_local_separate_heads', action='store_true', default=False,\n"
        "    help='In the TokenLocalDecoder, use 5 independent per-attribute heads (pos / col / opa / scl / rot)\n"
        "         off a shared trunk h2 instead of one shared out_linear. Fixes the gradient-starvation\n"
        "         failure where position\\'s gradient drowns out color and rotation in the shared head\n"
        "         (Col / Rot stuck near random init while Pos and Scl converge). Requires --token_local_decoder.')\n"
        "parser.add_argument('--token_local_head_cross_stitch', action='store_true', default=False,\n"
        "    help='Cross-stitch the 5 per-attribute heads (only meaningful with --token_local_separate_heads).\n"
        "         Lets each attribute head see a small learned combination of the others\\' trunks. Defer to\n"
        "         an ablation once --token_local_separate_heads is validated alone.')"
    )
    if arg_anchor not in src:
        print(f"[error] expected argparse anchor not found in {target_path}; aborting.")
        sys.exit(1)
    src = src.replace(arg_anchor, arg_anchor + arg_insert, 1)
    print("  [ok] added --token_local_separate_heads, --token_local_head_cross_stitch to argparse")

    # ── 2. SET ON MODEL CONFIG p ────────────────────────────────────────────
    # Insert after the existing p.token_local_decoder assignment.
    cfg_anchor = "p.token_local_decoder     = args.token_local_decoder"
    cfg_insert = (
        "\np.token_local_separate_heads    = args.token_local_separate_heads"
        "\np.token_local_head_cross_stitch = args.token_local_head_cross_stitch"
    )
    if cfg_anchor not in src:
        print(f"[error] expected model-config anchor not found in {target_path}; aborting.")
        sys.exit(1)
    src = src.replace(cfg_anchor, cfg_anchor + cfg_insert, 1)
    print("  [ok] set p.token_local_separate_heads, p.token_local_head_cross_stitch on model config")

    # ── 3. ADD TO CHECKPOINT METADATA ───────────────────────────────────────
    # Insert after the existing 'token_local_decoder' entry in _ckpt_meta.
    meta_anchor = "    'token_local_decoder':        args.token_local_decoder,"
    meta_insert = (
        "\n    'token_local_separate_heads':    args.token_local_separate_heads,"
        "\n    'token_local_head_cross_stitch': args.token_local_head_cross_stitch,"
    )
    if meta_anchor not in src:
        print(f"[error] expected checkpoint-meta anchor not found in {target_path}; aborting.")
        sys.exit(1)
    src = src.replace(meta_anchor, meta_anchor + meta_insert, 1)
    print("  [ok] added entries to _ckpt_meta")

    # ── 4. SENTINEL ─────────────────────────────────────────────────────────
    if not src.rstrip().endswith(SENTINEL):
        src = src.rstrip() + "\n\n" + SENTINEL + "\n"

    target_path.write_text(src)
    print(f"[done] patched {target_path}")


def main():
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_TARGET)
    if not target.exists():
        print(f"[error] target not found: {target}")
        sys.exit(1)
    patch(target)


if __name__ == "__main__":
    main()