"""
upload_logs_to_wandb.py — Upload Can3Tok training logs to W&B
==============================================================

Handles BOTH old and new gs_can3tok_2.py log formats:

  OLD format (job 20204707 style):
    Epoch 0 | Loss=126.3663 | Recon=126.2792 | KL=8710.7749 | Sem=0.0000 | ColorPred=0.000000
      Pos=6479.273 | Col=1343.513 | Opa=757.367 | Scl=2761.432 | Rot=2067.130
      mu range: [-4.530, 4.437]

  NEW format (Move 1 style):
    Epoch 0 | Loss=3.2341 | Recon=3.1200 | KL=0.0012 | InfoNCE=0.0000 | ColorPred=0.007823 | SceneSem=0.3421
      Pos=0.312 | Col=2.814 | Opa=0.041 | Scl=0.023 | Rot=0.019
      mu range: [-0.123, 0.456]

  Validation (same across both):
    --- Validation (epoch 20) ---
      L2:           117.6326
      Position:     113.2698
      Color:        16.9321  (absolute)
      Opacity:      2.2746
      Scale:        6.5796
      Rotation:     25.9269
      ColorPredMSE:      0.007823
      SceneSemanticKL:   0.3421
      [NEW BEST] L2=117.6326 saved

USAGE
-----
# Single file:
python upload_logs_to_wandb.py \\
    --log /home/yli11/scratch/Hafeez_thesis/Can3Tok/logs/train_colorresid_baseline_20204707.out

# Multiple files:
python upload_logs_to_wandb.py \\
    --log logs/run_A.out logs/run_B.out logs/run_C.out

# Custom run name:
python upload_logs_to_wandb.py \\
    --log logs/run_A.out --run_name "Run_A_colorresidual_only"

# Dry run — parse only, no upload:
python upload_logs_to_wandb.py --log logs/run_A.out --dry_run

# Debug — print parsed rows to verify parsing before upload:
python upload_logs_to_wandb.py --log logs/run_A.out --debug --dry_run
"""

import re
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional


# ============================================================================
# TEXT CLEANING
# ============================================================================

def clean_text(raw: str) -> str:
    """
    Normalise SLURM .out file before parsing.
    Removes ANSI codes (tqdm colours) and converts all carriage returns
    so every log line is separated by a single newline.
    """
    # Strip ANSI escape codes from tqdm
    text = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', raw)
    # \r\n → \n, remaining \r → \n
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # Collapse 3+ blank lines into 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text


# ============================================================================
# CONFIG PARSER
# ============================================================================

def parse_config(text: str, filename: str) -> Dict:
    """Extract run config from the startup summary block."""
    cfg = {}

    patterns = {
        'semantic_mode':         r'Semantic mode:\s+(\S+)',
        'color_residual':        r'Color residual:\s+(True|False)',
        'scene_semantic_head':   r'Scene semantic head:\s+(True|False)',
        'mean_color_weight':     r'Mean color weight:\s+([\d.]+)',
        'scene_semantic_weight': r'Scene semantic weight:\s+([\d.]+)',
        'label_input':           r'Label input:\s+(True|False)',
        'scale_norm_mode':       r'Scale norm mode:\s+(\S+)',
        'kl_weight':             r'--kl_weight\s+([\d.e-]+)',
        'lr':                    r'--lr\s+([\d.e-]+)',
        'batch_size':            r'--batch_size\s+(\d+)',
        'train_scenes':          r'--train_scenes\s+(\d+)',
        'val_scenes':            r'--val_scenes\s+(\d+)',
    }

    for key, pattern in patterns.items():
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            val = m.group(1)
            if val in ('True', 'False'):
                cfg[key] = (val == 'True')
            else:
                try:
                    cfg[key] = float(val) if ('.' in val or 'e' in val.lower()) else int(val)
                except ValueError:
                    cfg[key] = val

    # Job ID
    m = re.search(r'Job ID:\s*(\d+)', text)
    if m:
        cfg['job_id'] = m.group(1)

    # ── Filename fallbacks ────────────────────────────────────────────────────
    stem = Path(filename).stem.lower()

    if 'job_id' not in cfg:
        m = re.search(r'(\d{7,})', stem)
        if m:
            cfg['job_id'] = m.group(1)

    if 'color_residual' not in cfg:
        cfg['color_residual'] = ('colorresidual' in stem or 'colorresid' in stem)

    if 'scene_semantic_head' not in cfg:
        cfg['scene_semantic_head'] = ('scenesemantic' in stem or 'scenesem' in stem)

    if 'semantic_mode' not in cfg:
        if 'hidden' in stem:
            cfg['semantic_mode'] = 'hidden'
        elif 'geometric' in stem:
            cfg['semantic_mode'] = 'geometric'
        elif 'baseline' in stem:
            cfg['semantic_mode'] = 'none'
        else:
            cfg['semantic_mode'] = 'none'

    m = re.search(r'beta([\d.]+)', stem)
    if m and 'segment_weight' not in cfg:
        cfg['segment_weight'] = float(m.group(1))

    return cfg


# ============================================================================
# TRAINING EPOCH PARSER
# ============================================================================

# Mapping from raw key names in the log to clean W&B metric names.
# Covers both old format (Sem=) and new format (InfoNCE=, SceneSem=).
EPOCH_LINE_KEY_MAP = {
    'Loss':      'train/loss',
    'Recon':     'train/recon',
    'KL':        'train/kl',
    'Sem':       'train/infonce',       # old format
    'InfoNCE':   'train/infonce',       # new format
    'ColorPred': 'train/color_pred',
    'SceneSem':  'train/scene_sem',     # new format only
}

PARAM_LINE_KEY_MAP = {
    'Pos': 'train/position',
    'Col': 'train/color',
    'Opa': 'train/opacity',
    'Scl': 'train/scale',
    'Rot': 'train/rotation',
}

# Matches any "KEY=VALUE" pair on a line
KV_RE = re.compile(r'(\w+)\s*=\s*([-\d.]+(?:e[-+]?\d+)?)', re.IGNORECASE)

# Detects the start of an epoch line
EPOCH_START_RE = re.compile(r'^Epoch\s+(\d+)\s*\|', re.IGNORECASE)

# Detects the mu range line
MU_RE = re.compile(r'mu range:\s*\[([-\d.]+),\s*([-\d.]+)\]')


def parse_epoch_lines(text: str, debug: bool = False) -> List[Dict]:
    """
    Parse every training epoch summary using key=value extraction.
    Works for both old (Sem=) and new (InfoNCE=, SceneSem=) formats.
    """
    records = []
    lines = text.splitlines()

    if debug:
        hits = [l for l in lines if EPOCH_START_RE.match(l.strip())]
        print(f"\n  [DEBUG] Lines starting with 'Epoch <N> |': {len(hits)}")
        for l in hits[:5]:
            print(f"    {repr(l[:120])}")

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m = EPOCH_START_RE.match(line)
        if m:
            epoch = int(m.group(1))
            rec = {'epoch': epoch}

            # Extract key=value pairs from this line
            for kv in KV_RE.finditer(line):
                key, val = kv.group(1), float(kv.group(2))
                if key in EPOCH_LINE_KEY_MAP:
                    rec[EPOCH_LINE_KEY_MAP[key]] = val

            # Look ahead up to 4 lines for Pos/Col/... and mu range
            for j in range(i + 1, min(i + 5, len(lines))):
                next_line = lines[j].strip()

                # Stop if we hit next epoch or validation block
                if EPOCH_START_RE.match(next_line) or '--- Validation' in next_line:
                    break

                # Param line: Pos=... Col=... Opa=... Scl=... Rot=...
                for kv in KV_RE.finditer(next_line):
                    key, val = kv.group(1), float(kv.group(2))
                    if key in PARAM_LINE_KEY_MAP:
                        rec[PARAM_LINE_KEY_MAP[key]] = val

                # mu range line
                mm = MU_RE.search(next_line)
                if mm:
                    rec['train/mu_min'] = float(mm.group(1))
                    rec['train/mu_max'] = float(mm.group(2))

            records.append(rec)

            if debug and len(records) <= 3:
                print(f"\n  [DEBUG] Epoch {epoch} parsed:")
                for k, v in rec.items():
                    print(f"    {k}: {v}")

        i += 1

    return records


# ============================================================================
# VALIDATION BLOCK PARSER
# ============================================================================

VAL_FIELD_MAP = {
    'L2':              'val/l2',
    'Position':        'val/position',
    'Color':           'val/color',
    'Opacity':         'val/opacity',
    'Scale':           'val/scale',
    'Rotation':        'val/rotation',
    'ColorPredMSE':    'val/color_pred_mse',
    'SceneSemanticKL': 'val/scene_semantic_kl',
}

VAL_HEADER_RE  = re.compile(r'---\s*Validation\s*\(epoch\s+(\d+)\)\s*---', re.IGNORECASE)
VAL_FIELD_RE   = re.compile(r'^\s*([\w]+):\s+([\d.]+)')
BEST_RE        = re.compile(r'\[NEW BEST\]\s*L2\s*=\s*([\d.]+)')


def parse_validation_blocks(text: str, debug: bool = False) -> List[Dict]:
    """
    Parse every validation block. Works for both old and new formats —
    field names are identical, only whitespace differs.
    """
    records = []
    lines = text.splitlines()

    if debug:
        hits = [l for l in lines if VAL_HEADER_RE.search(l)]
        print(f"\n  [DEBUG] Validation headers found: {len(hits)}")
        for l in hits[:3]:
            print(f"    {repr(l[:120])}")

    i = 0
    while i < len(lines):
        m = VAL_HEADER_RE.search(lines[i])
        if m:
            rec = {'epoch': int(m.group(1))}

            # Scan ahead up to 20 lines for all val fields
            for j in range(i + 1, min(i + 21, len(lines))):
                line = lines[j]

                # Stop at next section
                if EPOCH_START_RE.match(line.strip()) or VAL_HEADER_RE.search(line):
                    break

                # Field line: "  L2:    3.1200" or "  L2:           129.2973"
                fm = VAL_FIELD_RE.match(line)
                if fm:
                    label = fm.group(1)
                    value = float(fm.group(2))
                    if label in VAL_FIELD_MAP and VAL_FIELD_MAP[label] not in rec:
                        rec[VAL_FIELD_MAP[label]] = value

                bm = BEST_RE.search(line)
                if bm:
                    rec['val/best_l2_so_far'] = float(bm.group(1))
                    rec['val/new_best']        = 1.0

            records.append(rec)

            if debug and len(records) <= 3:
                print(f"\n  [DEBUG] Val epoch {rec['epoch']}:")
                for k, v in rec.items():
                    print(f"    {k}: {v}")

        i += 1

    return records


# ============================================================================
# FINAL SUMMARY PARSER
# ============================================================================

def parse_final_summary(text: str) -> Dict:
    summary = {}

    m = re.search(r'Final L2:\s+([\d.]+)', text)
    if m:
        summary['summary/final_l2'] = float(m.group(1))

    m = re.search(r'Best L2:\s+([\d.]+)\s+\(epoch\s+(\d+)\)', text)
    if m:
        summary['summary/best_l2']    = float(m.group(1))
        summary['summary/best_epoch'] = int(m.group(2))

    for label, key in [
        ('ColorPredMSE',    'summary/final_color_pred_mse'),
        ('SceneSemanticKL', 'summary/final_scene_semantic_kl'),
    ]:
        m = re.search(rf'{label}[:\s]+([\d.]+)', text)
        if m:
            summary[key] = float(m.group(1))

    return summary


# ============================================================================
# UPLOAD
# ============================================================================

def build_run_name(cfg: Dict, log_path: Path, override: Optional[str]) -> str:
    if override:
        return override
    parts = []
    if 'job_id' in cfg:
        parts.append(f"job_{cfg['job_id']}")
    mode = cfg.get('semantic_mode', 'none')
    parts.append(mode if (mode and mode != 'none') else 'baseline')
    if cfg.get('color_residual'):
        parts.append('colorresid')
    if cfg.get('scene_semantic_head'):
        parts.append('scenesem')
    sw = cfg.get('segment_weight', 0)
    if sw and float(sw) > 0:
        parts.append(f"beta{sw}")
    return '_'.join(parts) if parts else log_path.stem


def build_tags(cfg: Dict) -> List[str]:
    tags = []
    mode = cfg.get('semantic_mode', 'none')
    tags.append(f"semantic_{mode}")
    if cfg.get('color_residual'):
        tags.append('color_residual')
    if cfg.get('scene_semantic_head'):
        tags.append('scene_semantic_head')
    return tags


def upload_log(
    log_path:  Path,
    project:   str,
    entity:    Optional[str],
    run_name:  Optional[str],
    dry_run:   bool,
    debug:     bool,
    wandb_key: Optional[str],
) -> bool:

    print(f"\n{'='*70}")
    print(f"  Log: {log_path}")
    print(f"{'='*70}")

    if not log_path.exists():
        print(f"  ERROR: file not found")
        return False

    raw  = log_path.read_text(encoding='utf-8', errors='ignore')
    text = clean_text(raw)
    print(f"  File: {len(raw):,} chars  →  {len(text):,} chars after cleaning")

    # ── Parse all sections ────────────────────────────────────────────────────
    cfg        = parse_config(text, log_path.name)
    epoch_rows = parse_epoch_lines(text, debug=debug)
    val_rows   = parse_validation_blocks(text, debug=debug)
    final      = parse_final_summary(text)

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n  Config:")
    for k, v in cfg.items():
        print(f"    {k}: {v}")

    print(f"\n  Parsed metrics:")
    if epoch_rows:
        sample = epoch_rows[0]
        print(f"    Training epochs:    {len(epoch_rows)}"
              f"  (epoch {epoch_rows[0]['epoch']} → {epoch_rows[-1]['epoch']})")
        print(f"    Training keys:      {[k for k in sample if k != 'epoch']}")
    else:
        print(f"    Training epochs:    0  ← WARNING: none found, run with --debug")

    if val_rows:
        sample = val_rows[0]
        print(f"    Validation blocks:  {len(val_rows)}"
              f"  (epochs: {[r['epoch'] for r in val_rows[:5]]}{'...' if len(val_rows)>5 else ''})")
        print(f"    Validation keys:    {[k for k in sample if k != 'epoch']}")
    else:
        print(f"    Validation blocks:  0  ← WARNING: none found, run with --debug")

    print(f"    Final summary:      {list(final.keys())}")

    if not epoch_rows and not val_rows:
        print(f"\n  WARNING: no metrics found. Run with --debug to diagnose.")
        return False

    name = build_run_name(cfg, log_path, run_name)
    tags = build_tags(cfg)
    print(f"\n  W&B run name: {name}")
    print(f"  W&B project:  {project}  |  entity: {entity}")
    print(f"  Tags:         {tags}")

    if dry_run:
        print(f"\n  DRY RUN — no upload.")
        return True

    # ── Upload ───────────────────────────────────────────────────────────────
    try:
        import wandb
    except ImportError:
        print("  ERROR: wandb not installed. Run: pip install wandb")
        return False

    if wandb_key:
        import os
        os.environ['WANDB_API_KEY'] = wandb_key

    run = wandb.init(
        project=project,
        entity=entity,
        name=name,
        config=cfg,
        tags=tags,
        reinit=True,
    )

    train_by_epoch = {r['epoch']: r for r in epoch_rows}
    val_by_epoch   = {r['epoch']: r for r in val_rows}
    all_epochs     = sorted(set(train_by_epoch) | set(val_by_epoch))

    for ep in all_epochs:
        payload = {}

        if ep in train_by_epoch:
            row = train_by_epoch[ep].copy()
            row.pop('epoch', None)
            payload.update(row)

        if ep in val_by_epoch:
            row = val_by_epoch[ep].copy()
            row.pop('epoch', None)
            payload.update(row)

        if payload:
            wandb.log(payload, step=ep)

    if final:
        wandb.run.summary.update(final)

    print(f"\n  Logged {len(all_epochs)} epochs to W&B")
    print(f"    Training:   {len(train_by_epoch)} epochs")
    print(f"    Validation: {len(val_by_epoch)} epochs")
    print(f"  Run URL: {run.url}")
    wandb.finish()
    return True


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Upload Can3Tok training logs to W&B',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file
  python upload_logs_to_wandb.py \\
      --log /home/yli11/scratch/Hafeez_thesis/Can3Tok/logs/train_colorresid_baseline_20204707.out

  # Multiple files
  python upload_logs_to_wandb.py \\
      --log logs/run_A.out logs/run_B.out logs/run_C.out

  # Custom run name
  python upload_logs_to_wandb.py \\
      --log logs/run_A.out --run_name "Run_A_colorresidual_only"

  # Dry run — parse and print, no upload
  python upload_logs_to_wandb.py --log logs/run_A.out --dry_run

  # Debug — verify parsing before uploading (use this if metrics are missing)
  python upload_logs_to_wandb.py --log logs/run_A.out --debug --dry_run
        """
    )

    parser.add_argument('--log', nargs='+', required=True, metavar='PATH',
        help='Path(s) to .out log file(s).')
    parser.add_argument('--project', type=str, default='Can3Tok-SceenSplat-7K',
        help='W&B project name (default: Can3Tok-SceenSplat-7K)')
    parser.add_argument('--entity', type=str, default='3D-SSC',
        help='W&B entity / team name (default: 3D-SSC)')
    parser.add_argument('--run_name', type=str, default=None,
        help='Override the W&B run name. Only applies when uploading a single file.')
    parser.add_argument('--wandb_key', type=str, default=None,
        help='W&B API key. Falls back to WANDB_API_KEY env var or existing login.')
    parser.add_argument('--dry_run', action='store_true',
        help='Parse and print metrics but do not upload to W&B.')
    parser.add_argument('--debug', action='store_true',
        help='Print detailed line-by-line parsing info. Use with --dry_run to '
             'diagnose missing metrics without uploading.')

    args = parser.parse_args()

    log_paths = [Path(p) for p in args.log]
    missing   = [p for p in log_paths if not p.exists()]
    if missing:
        print("ERROR: files not found:")
        for p in missing:
            print(f"  {p}")
        sys.exit(1)

    print(f"\nCan3Tok W&B Log Uploader")
    print(f"  Files:   {len(log_paths)}")
    print(f"  Project: {args.project}  |  Entity: {args.entity}")
    if args.dry_run:
        print(f"  Mode:    DRY RUN")
    if args.debug:
        print(f"  Debug:   ON")

    results = {}
    for path in log_paths:
        ok = upload_log(
            log_path  = path,
            project   = args.project,
            entity    = args.entity,
            run_name  = args.run_name if len(log_paths) == 1 else None,
            dry_run   = args.dry_run,
            debug     = args.debug,
            wandb_key = args.wandb_key,
        )
        results[path.name] = ok

    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    for name, ok in results.items():
        print(f"  {'✓' if ok else '✗'}  {name}")

    n_ok = sum(results.values())
    print(f"\n  {n_ok}/{len(results)} succeeded")

    if not args.dry_run and n_ok > 0:
        print(f"\n  View: https://wandb.ai/{args.entity}/{args.project}")

    sys.exit(0 if (len(results) - n_ok) == 0 else 1)


if __name__ == '__main__':
    main()