"""
scene_index_to_config.py
=========================
Convert the annotated scene_index.csv → scene_config.json
for latent_tsne_analysis.py.

Handles messy free-text labels from Excel:
  "appartment" → apartment
  "coffe shop" → coffee_shop
  "goKart"     → go_kart
  "Convinience store" → convenience_store
  etc.

USAGE:
    python scene_index_to_config.py \\
        --index ply_for_labeling/scene_index.csv \\
        --output scene_config.json

Then update CHECKPOINT_BASELINE and CHECKPOINT_SEMANTIC in
run_tsne_analysis.sh and run:
    sbatch run_tsne_analysis.sh
"""

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


# ── Canonical label map ────────────────────────────────────────────────────────
# Maps any reasonable free-text variant → clean snake_case label.
# Add new mappings here if your CSV uses other spellings.

LABEL_MAP = {
    # apartment
    'apartment':           'apartment',
    'appartment':          'apartment',
    'appartements':        'apartment',
    'flat':                'apartment',
    'residential':         'apartment',

    # coffee shop
    'coffee shop':         'coffee_shop',
    'coffeeshop':          'coffee_shop',
    'coffe shop':          'coffee_shop',
    'cafe':                'coffee_shop',
    'café':                'coffee_shop',
    'coffee':              'coffee_shop',

    # convenience store
    'convenience store':   'convenience_store',
    'convinience store':   'convenience_store',
    'convenience_store':   'convenience_store',
    'shop':                'convenience_store',
    'minimart':            'convenience_store',
    'grocery':             'convenience_store',

    # concert hall
    'concert hall':        'concert_hall',
    'concert_hall':        'concert_hall',
    'concerthall':         'concert_hall',
    'auditorium':          'concert_hall',
    'performance hall':    'concert_hall',

    # cinema
    'cinema':              'cinema',
    'movie theater':       'cinema',
    'theater':             'cinema',
    'theatre':             'cinema',

    # go-kart
    'go kart':             'go_kart',
    'go-kart':             'go_kart',
    'gokart':              'go_kart',
    'go_kart':             'go_kart',
    'kart':                'go_kart',
    'karting':             'go_kart',

    # gym
    'gym':                 'gym',
    'fitness':             'gym',
    'fitness center':      'gym',
    'workout':             'gym',

    # library
    'library':             'library',
    'bookstore':           'library',

    # office
    'office':              'office',
    'workspace':           'office',
    'coworking':           'office',

    # hotel / lobby
    'hotel':               'hotel',
    'lobby':               'lobby',
    'reception':           'lobby',

    # salon / spa
    'salon':               'spa_pool',
    'hair salon':          'spa_pool',
    'barbershop':          'spa_pool',
    'spa':                 'spa_pool',
    'spa & pool':          'spa_pool',
    'spa pool':            'spa_pool',
    'spa & pool':          'spa_pool',
    'pool':                'spa_pool',
    'indoor pool':         'spa_pool',
    'swimming pool':       'spa_pool',
    'poll relaxing area':  'spa_pool',
    'indoor pool (poll relaxing area)': 'spa_pool',

    # club
    'club':                'club',
    'nightclub':           'club',
    'night club':          'club',
    'bar':                 'club',
    'lounge':              'club',

    # washroom / bathroom
    'washroom':            'washroom',
    'bathroom':            'washroom',
    'restroom':            'washroom',
    'toilet':              'washroom',

    # museum / exhibition
    'museum':              'museum',
    'gallery':             'museum',
    'exhibition':          'museum',
    'painting exhibition': 'museum',
    'art gallery':         'museum',

    # school / classroom
    'school':              'school',
    'classroom':           'school',
    'lecture hall':        'school',

    # hospital / medical
    'hospital':            'hospital',
    'clinic':              'hospital',
    'medical':             'hospital',

    # restaurant
    'restaurant':          'restaurant',
    'dining':              'restaurant',
    'food court':          'restaurant',

    # wedding / banquet
    'wedding hall':        'wedding_hall',
    'wedding':             'wedding_hall',
    'banquet':             'wedding_hall',
    'banquet hall':        'wedding_hall',

    # futuristic / unusual — keep as-is
    'futuristic pod':      'futuristic_pod',
    'pod':                 'futuristic_pod',

    # unknown / skip
    'unknown':             'unknown',
    '':                    'unknown',
}


def normalize_label(raw: str) -> str:
    """
    Normalise a free-text category label to a clean snake_case string.

    Steps:
      1. Strip, lowercase, collapse whitespace
      2. Exact match in LABEL_MAP
      3. Partial / substring match in LABEL_MAP
      4. Fall back to snake_case of cleaned string
    """
    cleaned = re.sub(r'\s+', ' ', raw.strip().lower())

    # Exact match
    if cleaned in LABEL_MAP:
        return LABEL_MAP[cleaned]

    # Partial match — find any LABEL_MAP key that is a substring of cleaned
    for key, val in LABEL_MAP.items():
        if key and key in cleaned:
            return val

    # Fallback: snake_case the raw string
    snake = re.sub(r'[^a-z0-9]+', '_', cleaned).strip('_')
    return snake if snake else 'unknown'


def parse_args():
    ap = argparse.ArgumentParser(
        description='Convert annotated scene_index.csv → scene_config.json'
    )
    ap.add_argument('--index',  type=Path, required=True,
                    help='Annotated scene_index.csv')
    ap.add_argument('--output', type=Path, default=Path('scene_config.json'),
                    help='Output path for scene_config.json (default: scene_config.json)')
    ap.add_argument('--min-per-category', type=int, default=2,
                    help='Warn if category has fewer scenes than this (default: 2)')
    return ap.parse_args()


def main():
    args = parse_args()

    if not args.index.exists():
        print(f"[ERROR] Not found: {args.index}")
        return

    entries   = []
    skipped   = []
    cat_count = Counter()

    with open(args.index, newline='', encoding='utf-8-sig') as f:
        # Auto-detect tab vs comma (Excel TSV vs CSV)
        first_line = f.readline()
        f.seek(0)
        delimiter = '\t' if '\t' in first_line else ','
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            # Skip comment / header rows
            sn = row.get('scene_number', '').strip()
            if not sn or sn.startswith('#'):
                continue

            scene_id = row.get('scene_id', '').strip()
            raw_cat  = row.get('category', '').strip()
            notes    = row.get('notes',    '').strip()

            if not scene_id or 'REPLACE' in scene_id:
                continue

            if not raw_cat:
                skipped.append((scene_id, 'no category filled in'))
                continue

            category = normalize_label(raw_cat)

            entries.append({
                'scene_id': scene_id,
                'category': category,
                'raw_label': raw_cat,           # keep original for debugging
                'split':    'train',
                'note':     notes,
            })
            cat_count[category] += 1

    with open(args.output, 'w') as f:
        json.dump(entries, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"scene_config.json  —  {len(entries)} scenes")
    print(f"{'='*60}")
    if skipped:
        print(f"  Skipped (no label): {len(skipped)}")
        for sid, reason in skipped[:5]:
            print(f"    {sid}: {reason}")
        print()

    print(f"  {'Category':<25} {'N':>4}  {'t-SNE?'}")
    print(f"  {'─'*25}  {'─'*4}  {'─'*20}")

    for cat, n in sorted(cat_count.items(), key=lambda x: -x[1]):
        if cat == 'unknown':
            continue
        if n >= args.min_per_category:
            status = f"✓  ({n} scenes)"
        else:
            status = f"—  only {n} scene — will still appear as singleton"
        print(f"  {cat:<25} {n:>4}  {status}")

    if 'unknown' in cat_count:
        print(f"\n  unknown (excluded from t-SNE): {cat_count['unknown']}")

    viable = [c for c, n in cat_count.items()
              if c != 'unknown' and n >= args.min_per_category]
    singletons = [c for c, n in cat_count.items()
                  if c != 'unknown' and n < args.min_per_category]

    print(f"\n  Categories with ≥{args.min_per_category} scenes: {len(viable)}  "
          f"→  {', '.join(viable)}")
    if singletons:
        print(f"  Singletons (shown but no cluster): {len(singletons)}")

    print(f"\n  Saved: {args.output}")
    print(f"\nNext step:")
    print(f"  sbatch run_tsne_analysis.sh")
    print(f"  (make sure CHECKPOINT_BASELINE and CHECKPOINT_SEMANTIC are")
    print(f"   set correctly in run_tsne_analysis.sh first)")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()