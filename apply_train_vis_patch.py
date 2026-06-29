"""
apply_train_vis_patch.py
========================
Apply the TRAIN-VIS patch to gs_can3tok_2.py in place.

WHAT THIS PATCH DOES (and why):
  The current evaluate-time visualization saves PLYs ONLY for held-out scenes:
    reconstructed_gaussians/         -- val full scenes (different distribution)
    reconstructed_gaussians_chunk/   -- held-out chunks (in-distribution)
  This means there is no rendered way to see how well the model fits the SCENES IT
  TRAINED ON, which is exactly the diagnostic needed when val_L2 plateaus and you
  want to ask "is the model failing to fit, or failing to generalize?".

  This patch adds a SEPARATE, deterministic, non-augmented, non-shuffled loader
  built from the FIRST --train_vis_scenes SORTED training scenes -- the exact same
  scenes the model is being trained on. At every eval epoch the model is run on
  this loader with vis_tag="train", producing scene-for-scene comparable PLYs at:
    reconstructed_gaussians_train/       -- model output on training scenes
    ground_truth_gaussians_train/        -- GT for the SAME scenes (saved once @ epoch 0)
    pca_visualisations_train/            -- PCA of latent / per-Gaussian features
  alongside the existing held-out folders.

DETERMINISTIC SELECTION (the "overfit experiment" guarantee):
  - When --random_subset_seed is UNSET (the default in your job script), training
    already takes the FIRST --train_scenes SORTED scenes from the root, fully
    deterministic and reproducible. The chunk-val split is also deterministic by
    construction (skip the first --train_scenes sorted, take the next).
  - The new train-vis loader takes the FIRST --train_vis_scenes SORTED scenes,
    so it is guaranteed to be a prefix-subset of the actual training scenes.
  - The patch ADDS a runtime check: --train_vis_scenes > 0 with --random_subset_seed
    SET is rejected with a clear error, because in that combination "first N sorted"
    no longer matches the actual random training subset.
  - A subset-verification print runs at startup confirming all train-vis scenes
    are inside the training set, and lists the first/last 3 by name.

HOW THIS RUNS:
  python3 apply_train_vis_patch.py path/to/gs_can3tok_2.py
  -- patches the file IN PLACE. A backup at gs_can3tok_2.py.bak is written first.
  Idempotent: running twice is a no-op (the second run detects the patch is
  already applied and exits without changes).

REQUIRED JOB-SCRIPT ADDITIONS (mirrored in can3tok_overfit.job below):
  TRAIN_VIS_SCENES=3       # how many training scenes to reconstruct
  # ... and at the bottom of the TRAIN_CMD build:
  TRAIN_CMD="$TRAIN_CMD --train_vis_scenes $TRAIN_VIS_SCENES"
"""

import sys
import os
import shutil
from pathlib import Path


PATCH_MARKER = "# === TRAIN_VIS_PATCH_APPLIED ==="


# =============================================================================
# PATCH BLOCKS
# =============================================================================
# Each entry is (description, find_text, replacement_text). find_text must occur
# EXACTLY ONCE in the file. replacement_text is what it gets replaced with. The
# script verifies each match before applying any edit so a failed match aborts
# without partial-patching the file.

PATCHES = []


# -- Patch 1: insert --train_vis_scenes argparse arg after --chunk_val_scenes ---
PATCHES.append((
    "1/12: Add --train_vis_scenes argparse argument",
    """parser.add_argument('--chunk_val_scenes',     type=int,   default=None,
    help='Held-out chunk count for the chunk-val split (chunks/combined only). These are '
         'the chunks sorted AFTER the first --train_scenes, so they are DISJOINT from '
         'training by construction. None = use all remaining chunks (e.g. 3888 total - '
         '3800 train = 88). For a clean split do NOT set --random_subset_seed: a random '
         'training subset overlaps the skipped val chunks, which the disjointness check '
         'will reject.')""",
    """parser.add_argument('--chunk_val_scenes',     type=int,   default=None,
    help='Held-out chunk count for the chunk-val split (chunks/combined only). These are '
         'the chunks sorted AFTER the first --train_scenes, so they are DISJOINT from '
         'training by construction. None = use all remaining chunks (e.g. 3888 total - '
         '3800 train = 88). For a clean split do NOT set --random_subset_seed: a random '
         'training subset overlaps the skipped val chunks, which the disjointness check '
         'will reject.')
parser.add_argument('--train_vis_scenes',     type=int,   default=0,
    help='Number of TRAINING scenes to ALSO reconstruct during evaluation '
         '(visualization-only -- never used for the training loss/gradient updates). '
         'Picks the FIRST N sorted training scenes deterministically and freezes them '
         'at construction (no augmentation, no shuffle, preload=True), so the PLYs/PCAs '
         'saved at reconstructed_gaussians_train/ , ground_truth_gaussians_train/ and '
         'pca_visualisations_train/ are the SAME scenes every epoch and across reruns. '
         'The diagnostic question this loader answers is "can the model fit what it '
         'actually sees during training?", directly comparable scene-for-scene to the '
         'held-out PLYs at reconstructed_gaussians/ (full val) and '
         'reconstructed_gaussians_chunk/ (held-out chunks). REQUIRES --random_subset_seed '
         'UNSET (otherwise the first N sorted scenes do NOT match the actual random '
         'training subset, and a runtime check rejects the configuration). 0 = disabled.')""",
))


# -- Patch 2: hard-fail if train-vis is on while random_subset_seed is set ------
PATCHES.append((
    "2/12: Add hard-fail check that train-vis requires deterministic training",
    """if args.anchor_relative_decode and not args.position_scaffold:
    print("[INFO] --anchor_relative_decode requires --position_scaffold. Enabling.")
    args.position_scaffold = True

need_scaffold_data = args.position_scaffold""",
    """if args.anchor_relative_decode and not args.position_scaffold:
    print("[INFO] --anchor_relative_decode requires --position_scaffold. Enabling.")
    args.position_scaffold = True

# DETERMINISTIC TRAIN-VIS REQUIRES DETERMINISTIC TRAIN SET. The train-vis loader
# takes the first N SORTED scenes; this only equals the actual training set when
# training is ALSO sorted-first-N, i.e. when --random_subset_seed is NOT set.
# Reject the inconsistent combination loudly so the saved "train" reconstructions
# are GUARANTEED to be a subset of what the model actually trained on.
if args.train_vis_scenes > 0 and args.random_subset_seed is not None:
    raise ValueError(
        f"--train_vis_scenes={args.train_vis_scenes} requires --random_subset_seed "
        f"UNSET (got {args.random_subset_seed}). The train-vis loader takes the "
        f"first N SORTED training scenes, which only equals the actual training "
        f"subset when training is also sorted-first-N. Unset --random_subset_seed "
        f"(RANDOM_SUBSET_SEED=\\"\\" in the job).")

need_scaffold_data = args.position_scaffold""",
))


# -- Patch 3: build the train-vis dataset after the chunk-val block ------------
PATCHES.append((
    "3/12: Insert train-vis dataset + scene-subset verification",
    """    except _ChunkSplitError:
        raise  # contamination is fatal -- never disable-and-continue on a dirty split
    except Exception as e:
        if accelerator.is_main_process:
            print(f"  [WARNING] Could not create held-out chunk val dataset: {e}")
        gs_dataset_val_chunk = None
        _has_chunk_val = False

# Extra training datasets (multi-path support)""",
    """    except _ChunkSplitError:
        raise  # contamination is fatal -- never disable-and-continue on a dirty split
    except Exception as e:
        if accelerator.is_main_process:
            print(f"  [WARNING] Could not create held-out chunk val dataset: {e}")
        gs_dataset_val_chunk = None
        _has_chunk_val = False

# ── TRAIN-VIS DATASET (NEW: visualize the model's fit on its OWN training data) ──
# A SEPARATE, non-augmented, non-shuffled, preloaded dataset of the FIRST
# --train_vis_scenes sorted scenes from the training root. The same scenes the
# model is being trained on, but graded and rendered without augmentation -- so
# the saved PLYs are directly scene-for-scene comparable to the GT PLYs in
# ground_truth_gaussians_train/ and to the held-out PLYs in
# reconstructed_gaussians[_chunk]/.
#
# Why a SEPARATE loader (instead of just re-using gs_dataset_train):
#   - gs_dataset_train shuffles each epoch (random_permute=True), so picking the
#     "first N" of the train dataloader gives DIFFERENT scenes every epoch.
#   - gs_dataset_train has yaw augmentation (when --aug_yaw is on), so the same
#     scene reconstructs against a DIFFERENT rotated GT each epoch.
#   - The train-vis loader exists to give a STABLE, scene-locked picture of
#     training fit over time; only a frozen-sample dataset (preload=True, no aug,
#     no shuffle) can provide that.
#
# Determinism: random_permute=False, train=False, default aug_yaw=False (from
# _ds_kwargs), default preload=True (sampled ONCE at construction, frozen for the
# whole run). The same scenes load every epoch, every rerun.
gs_dataset_train_vis = None
trainVisDataLoader   = None
_has_train_vis       = False

if args.train_vis_scenes > 0:
    # Pick the source root that matches the training data mode:
    #   chunks   -> chunk root (same chunks the model trains on)
    #   full     -> full root  (same full scenes the model trains on)
    #   combined -> full root  (training's full-scene portion; cleanest deterministic
    #                          source. The chunk portion is already covered by
    #                          reconstructed_gaussians_chunk/ via the held-out chunk
    #                          val.)
    if args.train_data == 'chunks':
        _train_vis_root = _chunk_root
    elif args.train_data == 'full':
        _train_vis_root = _full_root
    else:  # combined
        _train_vis_root = _full_root

    if accelerator.is_main_process:
        print(f"\\n--- Train-vis Dataset: first {args.train_vis_scenes} sorted scenes "
              f"from {os.path.basename(_train_vis_root)} ---")
        print(f"    purpose : visualize the model's fit on its OWN training data, "
              f"directly scene-for-scene comparable to the held-out reconstructions")
        print(f"    mode    : no augmentation, no shuffle, preload=True (frozen sample)")

    try:
        gs_dataset_train_vis = gs_dataset(
            root=_train_vis_root,
            random_permute=False, train=False,
            max_scenes=args.train_vis_scenes, skip_scenes=None,
            **_ds_kwargs)   # uses _ds_kwargs defaults: aug_yaw=False, preload=True

        if len(gs_dataset_train_vis) > 0:
            _has_train_vis = True

            # DETERMINISTIC SUBSET CHECK. The train-vis loader takes the first N
            # SORTED scenes; the actual training set also takes the first
            # --train_scenes SORTED scenes (since --random_subset_seed is unset --
            # enforced earlier). Therefore the train-vis scenes MUST be a
            # prefix-subset of the training scenes. Verify and report it.
            if args.train_data == 'chunks':
                _train_dirs_for_check = set(gs_dataset_train.scene_dirs)
            elif args.train_data == 'full':
                _train_dirs_for_check = set(gs_dataset_train.scene_dirs)
            else:  # combined
                _train_dirs_for_check = set(_ds_full.scene_dirs)

            _tv_dirs   = set(gs_dataset_train_vis.scene_dirs)
            _missing   = _tv_dirs - _train_dirs_for_check
            if _missing:
                if accelerator.is_main_process:
                    print(f"  [WARNING] {len(_missing)} train-vis scene(s) are NOT in "
                          f"the training set. This should not happen with "
                          f"--random_subset_seed unset; investigate.")
            elif accelerator.is_main_process:
                _first3 = [os.path.basename(d) for d in gs_dataset_train_vis.scene_dirs[:3]]
                _last3  = [os.path.basename(d) for d in gs_dataset_train_vis.scene_dirs[-3:]]
                print(f"  Train-vis subset verified: {len(_tv_dirs)} scenes, all in the "
                      f"training set [OK]")
                print(f"    First 3 vis scenes : {_first3}")
                if len(_tv_dirs) > 3:
                    print(f"    Last 3 vis scenes  : {_last3}")
        else:
            if accelerator.is_main_process:
                print(f"  [INFO] No train-vis scenes available. Train-vis disabled.")
            gs_dataset_train_vis = None
    except Exception as e:
        if accelerator.is_main_process:
            print(f"  [WARNING] Could not create train-vis dataset: {e}")
        gs_dataset_train_vis = None
        _has_train_vis = False

# Extra training datasets (multi-path support)""",
))


# -- Patch 4: add the train-vis DataLoader after the chunk-val DataLoader -------
PATCHES.append((
    "4/12: Add the train-vis DataLoader",
    """if _has_chunk_val:
    valChunkDataLoader = Data.DataLoader(
        dataset=gs_dataset_val_chunk, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

# ============================================================================
# NORMALIZATION VERIFICATION
# ============================================================================""",
    """if _has_chunk_val:
    valChunkDataLoader = Data.DataLoader(
        dataset=gs_dataset_val_chunk, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

if _has_train_vis:
    # Train-vis is small and frozen (preload=True), so 2 workers are enough.
    trainVisDataLoader = Data.DataLoader(
        dataset=gs_dataset_train_vis, batch_size=args.batch_size,
        shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)

# ============================================================================
# NORMALIZATION VERIFICATION
# ============================================================================""",
))


# -- Patch 5: add train-vis check to the normalization-verification block ------
PATCHES.append((
    "5/12: Add train-vis to normalization verification",
    """    if _has_chunk_val:
        _ok_chunk_val = _check_nf("Val held-out chunks", gs_dataset_val_chunk.scene_dirs, True)

    print(f"{'='*70}\\n")""",
    """    if _has_chunk_val:
        _ok_chunk_val = _check_nf("Val held-out chunks", gs_dataset_val_chunk.scene_dirs, True)

    if _has_train_vis:
        # Same root as training, so expected_present matches the training source.
        _train_vis_expected = (args.train_data == 'chunks')
        _check_nf("Train-vis scenes", gs_dataset_train_vis.scene_dirs, _train_vis_expected)

    print(f"{'='*70}\\n")""",
))


# -- Patch 6: add train-vis to the dataset summary printout -------------------
PATCHES.append((
    "6/12: Add train-vis to dataset summary",
    """    if _has_chunk_val:
        print(f"  Val held-out chunks: {len(gs_dataset_val_chunk)}  "
              f"(CLEAN split: first {_n_train_chunks} sorted = train, "
              f"remaining = val, disjoint)")
    else:
        print(f"  Val held-out chunks: N/A")
    print(f"  Gaussian order     :""",
    """    if _has_chunk_val:
        print(f"  Val held-out chunks: {len(gs_dataset_val_chunk)}  "
              f"(CLEAN split: first {_n_train_chunks} sorted = train, "
              f"remaining = val, disjoint)")
    else:
        print(f"  Val held-out chunks: N/A")
    if _has_train_vis:
        print(f"  Train-vis scenes   : {len(gs_dataset_train_vis)}  "
              f"(DETERMINISTIC subset of training -- same scenes every epoch; "
              f"PLYs at reconstructed_gaussians_train/)")
    else:
        print(f"  Train-vis scenes   : N/A")
    print(f"  Gaussian order     :""",
))


# -- Patch 7: replace accelerator.prepare with generic 4-combo handling ---------
PATCHES.append((
    "7/12: Generalize accelerator.prepare to handle train-vis loader",
    """if _has_chunk_val:
    (gs_autoencoder, optimizer, trainDataLoader,
     valDataLoader, valChunkDataLoader, scheduler) = accelerator.prepare(
        gs_autoencoder, optimizer, trainDataLoader,
        valDataLoader, valChunkDataLoader, scheduler)
else:
    (gs_autoencoder, optimizer, trainDataLoader,
     valDataLoader, scheduler) = accelerator.prepare(
        gs_autoencoder, optimizer, trainDataLoader,
        valDataLoader, scheduler)""",
    """# Build a flat list of things to prepare so we can handle the 4 combinations of
# (chunk_val present?, train_vis present?) without duplicating the call.
_prep_objs = [gs_autoencoder, optimizer, trainDataLoader, valDataLoader]
if _has_chunk_val:  _prep_objs.append(valChunkDataLoader)
if _has_train_vis:  _prep_objs.append(trainVisDataLoader)
_prep_objs.append(scheduler)

_prepared = accelerator.prepare(*_prep_objs)
_idx = 0
gs_autoencoder  = _prepared[_idx]; _idx += 1
optimizer       = _prepared[_idx]; _idx += 1
trainDataLoader = _prepared[_idx]; _idx += 1
valDataLoader   = _prepared[_idx]; _idx += 1
if _has_chunk_val:
    valChunkDataLoader = _prepared[_idx]; _idx += 1
if _has_train_vis:
    trainVisDataLoader = _prepared[_idx]; _idx += 1
scheduler       = _prepared[_idx]; _idx += 1""",
))


# -- Patch 8: add train_vis_scenes to checkpoint metadata ----------------------
PATCHES.append((
    "8/12: Add train_vis_scenes to checkpoint metadata",
    """    'n_train_chunks':             _n_train_chunks,
    'chunk_val_scenes':           args.chunk_val_scenes,
    'kl_anneal_steps':            args.kl_anneal_steps,""",
    """    'n_train_chunks':             _n_train_chunks,
    'chunk_val_scenes':           args.chunk_val_scenes,
    'train_vis_scenes':           args.train_vis_scenes,
    'kl_anneal_steps':            args.kl_anneal_steps,""",
))


# -- Patch 9: add train-vis evaluation block in the training loop --------------
PATCHES.append((
    "9/12: Add train-vis evaluation block inside the training loop",
    """        chunk_metrics = None
        if _has_chunk_val:
            # do_vis=True + vis_tag="chunk": save held-out-chunk reconstructions to
            # reconstructed_gaussians_chunk/ (separate from the full-scene PLYs).
            chunk_metrics = evaluate_model(gs_autoencoder, raw_model, valChunkDataLoader,
                                           device, accelerator, epoch=epoch,
                                           do_vis=True, vis_tag="chunk")
            if accelerator.is_main_process:
                print(f"\\n--- Val HELD-OUT CHUNKS epoch {epoch} "
                      f"(skip={_n_train_chunks}, n={len(gs_dataset_val_chunk)}) ---")
                print(f"  L2={chunk_metrics['avg_l2_error']:.4f}  "
                      f"Pos={chunk_metrics['position_loss']:.4f}  "
                      f"Col={chunk_metrics['color_loss']:.4f}  "
                      f"Opa={chunk_metrics['opacity_loss']:.4f}  "
                      f"Scl={chunk_metrics['scale_loss']:.4f}  "
                      f"Rot={chunk_metrics['rotation_loss']:.4f}")
                if chunk_metrics['avg_l2_error'] > 1e-6:
                    _gap = val_metrics['avg_l2_error'] / chunk_metrics['avg_l2_error']
                    print(f"  DISTRIBUTION GAP  full_L2 / chunk_L2 = {_gap:.2f}x  "
                          f"({'negligible' if _gap < 1.3 else 'moderate' if _gap < 2.0 else 'large -- chunks much easier'})")""",
    """        chunk_metrics = None
        if _has_chunk_val:
            # do_vis=True + vis_tag="chunk": save held-out-chunk reconstructions to
            # reconstructed_gaussians_chunk/ (separate from the full-scene PLYs).
            chunk_metrics = evaluate_model(gs_autoencoder, raw_model, valChunkDataLoader,
                                           device, accelerator, epoch=epoch,
                                           do_vis=True, vis_tag="chunk")
            if accelerator.is_main_process:
                print(f"\\n--- Val HELD-OUT CHUNKS epoch {epoch} "
                      f"(skip={_n_train_chunks}, n={len(gs_dataset_val_chunk)}) ---")
                print(f"  L2={chunk_metrics['avg_l2_error']:.4f}  "
                      f"Pos={chunk_metrics['position_loss']:.4f}  "
                      f"Col={chunk_metrics['color_loss']:.4f}  "
                      f"Opa={chunk_metrics['opacity_loss']:.4f}  "
                      f"Scl={chunk_metrics['scale_loss']:.4f}  "
                      f"Rot={chunk_metrics['rotation_loss']:.4f}")
                if chunk_metrics['avg_l2_error'] > 1e-6:
                    _gap = val_metrics['avg_l2_error'] / chunk_metrics['avg_l2_error']
                    print(f"  DISTRIBUTION GAP  full_L2 / chunk_L2 = {_gap:.2f}x  "
                          f"({'negligible' if _gap < 1.3 else 'moderate' if _gap < 2.0 else 'large -- chunks much easier'})")

        train_vis_metrics = None
        if _has_train_vis:
            # do_vis=True + vis_tag="train": save TRAINING-set reconstructions to
            # reconstructed_gaussians_train/ -- the diagnostic for "is the model
            # fitting what it actually sees?" Same scenes every epoch (no aug, no
            # shuffle), so the per-epoch PLYs are directly comparable over time.
            train_vis_metrics = evaluate_model(gs_autoencoder, raw_model, trainVisDataLoader,
                                               device, accelerator, epoch=epoch,
                                               do_vis=True, vis_tag="train")
            if accelerator.is_main_process:
                print(f"\\n--- TRAIN-SET VISUALIZATION epoch {epoch} "
                      f"(n={len(gs_dataset_train_vis)}, deterministic subset of training) ---")
                print(f"  L2={train_vis_metrics['avg_l2_error']:.4f}  "
                      f"Pos={train_vis_metrics['position_loss']:.4f}  "
                      f"Col={train_vis_metrics['color_loss']:.4f}  "
                      f"Opa={train_vis_metrics['opacity_loss']:.4f}  "
                      f"Scl={train_vis_metrics['scale_loss']:.4f}  "
                      f"Rot={train_vis_metrics['rotation_loss']:.4f}")
                # Generalization gap = val_L2 / train_vis_L2. > 1 = train fits better
                # than held-out (overfitting). ~ 1 = model generalizes; high values
                # = strong overfitting; very high in the early epochs is expected.
                if train_vis_metrics['avg_l2_error'] > 1e-6:
                    _gen_gap = val_metrics['avg_l2_error'] / train_vis_metrics['avg_l2_error']
                    print(f"  GENERALIZATION GAP val_L2 / train_vis_L2 = {_gen_gap:.2f}x  "
                          f"({'no overfit' if _gen_gap < 1.1 else 'mild overfit' if _gen_gap < 1.5 else 'strong overfit -- val much worse than train'})")
                if chunk_metrics is not None and chunk_metrics['avg_l2_error'] > 1e-6:
                    _gen_gap_c = chunk_metrics['avg_l2_error'] / train_vis_metrics['avg_l2_error']
                    print(f"  GENERALIZATION GAP chunk_L2 / train_vis_L2 = {_gen_gap_c:.2f}x  "
                          f"(in-distribution generalization)")""",
))


# -- Patch 10: update W&B logging to include train-vis metrics -----------------
PATCHES.append((
    "10/12: Add train-vis metrics to W&B logging",
    """        if accelerator.is_main_process and wandb_enabled:
            log_dict = {
                'epoch': epoch,
                'val_full_l2': val_metrics['avg_l2_error'],
                'val_full_pos': val_metrics['position_loss'],
                'val_full_col': val_metrics['color_loss'],
            }
            if chunk_metrics is not None:
                log_dict['val_chunk_l2']  = chunk_metrics['avg_l2_error']
                log_dict['val_chunk_pos'] = chunk_metrics['position_loss']
                if chunk_metrics['avg_l2_error'] > 1e-6:
                    log_dict['val_dist_gap'] = (val_metrics['avg_l2_error']
                                                / chunk_metrics['avg_l2_error'])
            wandb_run.log(log_dict)""",
    """        if accelerator.is_main_process and wandb_enabled:
            log_dict = {
                'epoch': epoch,
                'val_full_l2': val_metrics['avg_l2_error'],
                'val_full_pos': val_metrics['position_loss'],
                'val_full_col': val_metrics['color_loss'],
            }
            if chunk_metrics is not None:
                log_dict['val_chunk_l2']  = chunk_metrics['avg_l2_error']
                log_dict['val_chunk_pos'] = chunk_metrics['position_loss']
                if chunk_metrics['avg_l2_error'] > 1e-6:
                    log_dict['val_dist_gap'] = (val_metrics['avg_l2_error']
                                                / chunk_metrics['avg_l2_error'])
            if train_vis_metrics is not None:
                log_dict['train_vis_l2']  = train_vis_metrics['avg_l2_error']
                log_dict['train_vis_pos'] = train_vis_metrics['position_loss']
                log_dict['train_vis_col'] = train_vis_metrics['color_loss']
                if train_vis_metrics['avg_l2_error'] > 1e-6:
                    log_dict['val_gen_gap'] = (val_metrics['avg_l2_error']
                                               / train_vis_metrics['avg_l2_error'])
            wandb_run.log(log_dict)""",
))


# -- Patch 11: add final train-vis eval + saved-metric ------------------------
PATCHES.append((
    "11/12: Add final train-vis eval + final saved metric",
    """accelerator.wait_for_everyone()
final_metrics = evaluate_model(gs_autoencoder, raw_model, valDataLoader, device,
                               accelerator, epoch=args.num_epochs-1, do_vis=True)
final_chunk_metrics = None
if _has_chunk_val:
    final_chunk_metrics = evaluate_model(gs_autoencoder, raw_model, valChunkDataLoader,
                                         device, accelerator, epoch=args.num_epochs-1,
                                         do_vis=True, vis_tag="chunk")

if accelerator.is_main_process:
    print(f"\\nFinal full_L2 : {final_metrics['avg_l2_error']:.4f}")
    if final_chunk_metrics is not None:
        print(f"Final chunk_L2: {final_chunk_metrics['avg_l2_error']:.4f}")
        if final_chunk_metrics['avg_l2_error'] > 1e-6:
            print(f"Final gap     : {final_metrics['avg_l2_error']/final_chunk_metrics['avg_l2_error']:.2f}x")
    print(f"Best full_L2  : {best_val_loss:.4f}  (epoch {best_epoch})")

    final_dict = {
        'epoch':            args.num_epochs - 1,
        'model_state_dict': raw_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_val_l2':     final_metrics['avg_l2_error'],
        'best_val_l2':      best_val_loss,
        'best_epoch':       best_epoch,
        **_ckpt_meta,
        'individual_losses': {k: final_metrics[f'{k}_loss'] for k in PARAM_SLICES},
    }
    if final_chunk_metrics is not None:
        final_dict['final_chunk_val_l2'] = final_chunk_metrics['avg_l2_error']
    torch.save(final_dict, os.path.join(save_path, "final.pth"))
    print(f"Saved: {save_path}final.pth")""",
    """accelerator.wait_for_everyone()
final_metrics = evaluate_model(gs_autoencoder, raw_model, valDataLoader, device,
                               accelerator, epoch=args.num_epochs-1, do_vis=True)
final_chunk_metrics = None
if _has_chunk_val:
    final_chunk_metrics = evaluate_model(gs_autoencoder, raw_model, valChunkDataLoader,
                                         device, accelerator, epoch=args.num_epochs-1,
                                         do_vis=True, vis_tag="chunk")
final_train_vis_metrics = None
if _has_train_vis:
    final_train_vis_metrics = evaluate_model(gs_autoencoder, raw_model, trainVisDataLoader,
                                             device, accelerator, epoch=args.num_epochs-1,
                                             do_vis=True, vis_tag="train")

if accelerator.is_main_process:
    print(f"\\nFinal full_L2 : {final_metrics['avg_l2_error']:.4f}")
    if final_chunk_metrics is not None:
        print(f"Final chunk_L2: {final_chunk_metrics['avg_l2_error']:.4f}")
        if final_chunk_metrics['avg_l2_error'] > 1e-6:
            print(f"Final dist gap : {final_metrics['avg_l2_error']/final_chunk_metrics['avg_l2_error']:.2f}x")
    if final_train_vis_metrics is not None:
        print(f"Final train_L2 : {final_train_vis_metrics['avg_l2_error']:.4f}  "
              f"(model fit on its own training data)")
        if final_train_vis_metrics['avg_l2_error'] > 1e-6:
            print(f"Final gen gap  : {final_metrics['avg_l2_error']/final_train_vis_metrics['avg_l2_error']:.2f}x  "
                  f"(val_L2 / train_vis_L2)")
    print(f"Best full_L2  : {best_val_loss:.4f}  (epoch {best_epoch})")

    final_dict = {
        'epoch':            args.num_epochs - 1,
        'model_state_dict': raw_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_val_l2':     final_metrics['avg_l2_error'],
        'best_val_l2':      best_val_loss,
        'best_epoch':       best_epoch,
        **_ckpt_meta,
        'individual_losses': {k: final_metrics[f'{k}_loss'] for k in PARAM_SLICES},
    }
    if final_chunk_metrics is not None:
        final_dict['final_chunk_val_l2'] = final_chunk_metrics['avg_l2_error']
    if final_train_vis_metrics is not None:
        final_dict['final_train_vis_l2'] = final_train_vis_metrics['avg_l2_error']
    torch.save(final_dict, os.path.join(save_path, "final.pth"))
    print(f"Saved: {save_path}final.pth")""",
))


# -- Patch 12: add train-vis to W&B final summary -----------------------------
PATCHES.append((
    "12/12: Add train-vis to W&B final summary",
    """if wandb_enabled and accelerator.is_main_process:
    summary = {"final_val_l2": final_metrics['avg_l2_error'],
               "best_val_l2": best_val_loss, "best_epoch": best_epoch}
    if final_chunk_metrics is not None:
        summary["final_chunk_val_l2"] = final_chunk_metrics['avg_l2_error']
    wandb_run.summary.update(summary)
    wandb_run.finish()""",
    """if wandb_enabled and accelerator.is_main_process:
    summary = {"final_val_l2": final_metrics['avg_l2_error'],
               "best_val_l2": best_val_loss, "best_epoch": best_epoch}
    if final_chunk_metrics is not None:
        summary["final_chunk_val_l2"] = final_chunk_metrics['avg_l2_error']
    if final_train_vis_metrics is not None:
        summary["final_train_vis_l2"] = final_train_vis_metrics['avg_l2_error']
    wandb_run.summary.update(summary)
    wandb_run.finish()""",
))


# =============================================================================
# APPLICATION DRIVER
# =============================================================================

def apply_patch(target_path: Path) -> int:
    if not target_path.exists():
        print(f"ERROR: {target_path} does not exist", file=sys.stderr)
        return 1

    source = target_path.read_text(encoding='utf-8')

    # Idempotency: detect that the patch was already applied (any one of the
    # 12 inserted strings is sufficient -- pick a distinctive one).
    if "--train_vis_scenes" in source and "TRAIN-VIS DATASET" in source:
        print(f"[OK] Patch already applied to {target_path}. No changes made.")
        print(f"     (detected '--train_vis_scenes' and 'TRAIN-VIS DATASET' in the file)")
        return 0
    if PATCH_MARKER in source:
        print(f"[OK] Patch marker {PATCH_MARKER} found. No changes made.")
        return 0

    # Pre-flight: verify every find_text matches EXACTLY ONCE before any write.
    # This guards against partial patching if a later block doesn't match: we
    # detect the issue up front and abort cleanly.
    print(f"Pre-flight: verifying all {len(PATCHES)} patch blocks match in {target_path}")
    print(f"           (each find_text must occur exactly once)")
    print()

    failures = []
    for i, (desc, find, _repl) in enumerate(PATCHES, start=1):
        count = source.count(find)
        status = "OK"   if count == 1 else f"FAIL ({count} matches)"
        print(f"  [{status:>13s}] {desc}")
        if count != 1:
            failures.append((i, desc, count))

    if failures:
        print()
        print("PRE-FLIGHT FAILED: one or more patch blocks did not match exactly once.")
        print("This usually means your gs_can3tok_2.py is a different version than the")
        print("one this patch was written for. Check the failing blocks above and either")
        print("update the patch to match your file, or apply the changes manually.")
        for i, desc, count in failures:
            print(f"  - Block {i}: {desc} ({count} matches; expected 1)")
        return 2

    # Backup before writing.
    backup_path = target_path.with_suffix(target_path.suffix + ".bak")
    shutil.copy2(target_path, backup_path)
    print()
    print(f"Backup saved to {backup_path}")

    # Apply patches sequentially. Each operates on the running buffer.
    print()
    print(f"Applying {len(PATCHES)} patches...")
    patched = source
    for i, (desc, find, repl) in enumerate(PATCHES, start=1):
        patched = patched.replace(find, repl, 1)
        print(f"  [APPLIED] {desc}")

    # Add a footer marker so re-runs short-circuit even if individual block
    # detection were to fail in the future.
    patched += f"\n\n{PATCH_MARKER}\n"

    # Write atomically (write to temp file, then rename) to avoid leaving a
    # truncated file if the process is interrupted mid-write.
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    tmp_path.write_text(patched, encoding='utf-8')
    os.replace(tmp_path, target_path)

    print()
    print(f"[DONE] Patched {target_path} successfully.")
    print(f"       Original at  {backup_path}")
    print(f"       Lines added: {patched.count(chr(10)) - source.count(chr(10))}")
    print()
    print("Next steps:")
    print("  1) Sanity-check with:  python3 -c \"import ast; ast.parse(open('{0}').read())\"".format(target_path))
    print("  2) In your .job file, add:  TRAIN_VIS_SCENES=3   (or however many scenes)")
    print("                              and:  TRAIN_CMD=\"$TRAIN_CMD --train_vis_scenes $TRAIN_VIS_SCENES\"")
    print("  3) Make sure RANDOM_SUBSET_SEED=\"\" (the train-vis check rejects a set seed)")
    print("  4) Submit the job; reconstructed_gaussians_train/ will appear at every")
    print("     RECON_PLY_FREQ epoch alongside reconstructed_gaussians/ and")
    print("     reconstructed_gaussians_chunk/ , with ground_truth_gaussians_train/")
    print("     written once at epoch 0.")
    return 0


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 apply_train_vis_patch.py <path-to-gs_can3tok_2.py>", file=sys.stderr)
        print()
        print("Example:")
        print("  python3 apply_train_vis_patch.py /home/yli11/scratch-project/Hafeez_thesis/Semantic-Scene-Completion/gs_can3tok_2.py")
        return 1
    return apply_patch(Path(sys.argv[1]))


if __name__ == "__main__":
    sys.exit(main())