"""
run_journal.py - Journal Extension Experiment Controller
=========================================================
Runs two categories of new experiments on top of the completed
original experiment (stored in outputs4/):

  1. LoadSim Component Ablation  (7 configs x 4 folds)
  2. Multi-backbone Comparison   (PointNeXt / KPConv / PTv3 / RandLA-Net
                                   x all 7 data branches x 4 folds)

Skip logic (resume support):
  - Each sub-experiment checks for test_metrics.json in its output directory.
  - If the file already exists the step is skipped; no work is repeated.
  - Partial runs (e.g. S1 done, S2 not yet) are detected step-by-step.
  - PointNet++ results come from outputs4/ (original experiment).

Usage:
    python run_journal.py --data-dir ./data                  # all
    python run_journal.py --data-dir ./data --experiment ablation
    python run_journal.py --data-dir ./data --experiment backbone
    python run_journal.py --data-dir ./data --debug          # 1-fold, 3 epochs
"""

import argparse, subprocess, sys, time, json, io, shutil
from pathlib import Path

# ── Windows UTF-8 safe fix ────────────────────────────────────────────────
if sys.platform == 'win32':
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        elif (hasattr(sys.stdout, 'buffer') and
              not isinstance(sys.stdout, io.TextIOWrapper)):
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer, encoding='utf-8', errors='replace')
            sys.stderr = io.TextIOWrapper(
                sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))
from config import (
    DATASET, CV_SPLITS, N_CV_FOLDS, AUGMENTATION, PSEUDO_LABEL,
    MODEL, TRAIN, TRAIN_BRANCHES,
    ABLATION_CONFIGS, ABLATION_TRAIN,
    BACKBONE_COMPARISON, BACKBONE_DATA_BRANCHES,
    BACKBONE_TRAIN, BACKBONE_N_FOLDS,
    DEBUG,
)

OUTROOT = BASE / 'outputs4'
JOURNAL = BASE / 'outputs_journal'


# ── Utility helpers ────────────────────────────────────────────────────────

def ply_count(d: Path) -> int:
    """Number of PLY files in directory d."""
    return len(list(d.glob('*.ply'))) if d.exists() else 0


def is_valid_json(path) -> bool:
    """Return True iff *path* exists and contains non-empty JSON."""
    path = Path(path)
    if not path.exists():
        return False
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        return bool(data)
    except Exception:
        return False


def run_step(label: str, script: str, args_list: list,
             optional: bool = False):
    """
    Launch *script* as a subprocess with *args_list*.

    - Uses list-based subprocess.run (shell=False) to avoid Windows
      shell encoding issues with CJK characters in paths.
    - stdout/stderr are NOT captured; child output goes directly to
      the terminal for real-time monitoring.
    - Returns (elapsed_seconds, success_bool).
    """
    print(f'\n{"=" * 70}', flush=True)
    print(f'  {label}', flush=True)
    print(f'{"=" * 70}', flush=True)
    sys.stdout.flush()

    cmd = [sys.executable, str(BASE / script)] + [str(a) for a in args_list]
    t0  = time.time()
    try:
        ret = subprocess.run(cmd, shell=False)
        rc  = ret.returncode
    except KeyboardInterrupt:
        print(f'\n  ⚠️  {label} interrupted by user', flush=True)
        return time.time() - t0, False
    except Exception as exc:
        print(f'\n  ⚠️  launch failed: {exc}', flush=True)
        if optional:
            return time.time() - t0, False
        sys.exit(1)

    el = time.time() - t0
    if rc != 0:
        if optional:
            print(f'  ⚠️  {label} failed [{el:.1f}s]', flush=True)
            return el, False
        print(f'  ❌  {label} failed', flush=True)
        sys.exit(1)
    print(f'  ✅  {label} done [{el:.1f}s]', flush=True)
    return el, True


def check_outputs4_complete():
    """Print completion status of the original experiment."""
    done = sum(
        1 for fi in range(N_CV_FOLDS) for br in TRAIN_BRANCHES
        if (OUTROOT / f'fold_{fi:02d}' / br / 'model' / 'best_model.pth').exists()
    )
    total = N_CV_FOLDS * len(TRAIN_BRANCHES)
    print(f'\n  {"─" * 52}', flush=True)
    print(f'  Original experiment: {done}/{total}'
          f' ({100 * done // max(total, 1)}%)', flush=True)
    for fi in range(N_CV_FOLDS):
        done_br = [
            br for br in TRAIN_BRANCHES
            if (OUTROOT / f'fold_{fi:02d}' / br / 'model' /
                'best_model.pth').exists()]
        print(f'    fold_{fi:02d}: {len(done_br)}/{len(TRAIN_BRANCHES)}'
              ' branches', flush=True)
    print(f'  {"─" * 52}', flush=True)
    return done, total


def get_branch_data(fold_idx: int, branch: str):
    """Return (aug_hq_dir, orig_dir, label_file) for an original branch."""
    fd  = OUTROOT / f'fold_{fold_idx:02d}'
    src = TRAIN_BRANCHES[branch][0]
    hq_map = {
        'lsda':    fd / 'aug_lsda'    / 'high_quality',
        'trad':    fd / 'aug_trad'    / 'high_quality',
        'loadsim': fd / 'aug_loadsim' / 'high_quality',
    }
    aug_hq  = hq_map.get(src) if src != 'none' else None
    orig    = fd / 'data_pool' if TRAIN_BRANCHES[branch][2] else None
    lbl     = fd / branch / 'pseudo_labels.json'
    return aug_hq, orig, lbl


# ── Ablation experiments ───────────────────────────────────────────────────

def run_ablation_experiments(P_TR, P_MDL, P_AUG, n_folds: int) -> float:
    print(f"\n{'#' * 70}", flush=True)
    print('  [ABLATION] LoadSim component contribution analysis', flush=True)
    print(f'  Configs: {len(ABLATION_CONFIGS)}  Folds: {n_folds}'
          f'  Max samples: {P_AUG["n_sample_150"]}'
          f'  Epochs: {ABLATION_TRAIN["epochs"]}', flush=True)
    print(f"{'#' * 70}", flush=True)

    total_time = 0.0
    n_done, n_skip = 0, 0

    for fold_idx in range(min(n_folds, N_CV_FOLDS)):
        fold_dir      = OUTROOT / f'fold_{fold_idx:02d}'
        data_pool_dir = fold_dir / 'data_pool'
        abl_fold_dir  = JOURNAL / 'ablation' / f'fold_{fold_idx:02d}'
        test_lbl      = OUTROOT / 'labels_test' / 'pseudo_labels.json'
        test_dir_path = OUTROOT / 'test_data'

        print(f'\n  ── Fold {fold_idx + 1}/{n_folds}'
              f' ─────────────────────────────', flush=True)
        if not data_pool_dir.exists():
            print(f'  [SKIP] data_pool not found: {data_pool_dir}',
                  flush=True)
            continue

        abl_fold_dir.mkdir(parents=True, exist_ok=True)

        for ai, (abl_name,
                 (en_r, en_s, en_c, label_cn, label_en)) in \
                enumerate(ABLATION_CONFIGS.items()):

            abl_dir      = abl_fold_dir / abl_name
            aug_dir      = abl_dir / 'augmented'
            hq_dir       = abl_dir / 'high_quality'
            model_dir    = abl_dir / 'model'
            lbl_file     = abl_dir / 'pseudo_labels.json'
            metrics_file = abl_dir / 'test_metrics.json'

            # ── skip if already completed ────────────────────────────────
            if metrics_file.exists():
                print(f'  ✓ [{ai + 1}/{len(ABLATION_CONFIGS)}]'
                      f' {abl_name} fold{fold_idx} — done, skipping',
                      flush=True)
                n_skip += 1
                continue

            print(f'\n  [{ai + 1}/{len(ABLATION_CONFIGS)}] {abl_name}:'
                  f' {label_en} | Fold {fold_idx + 1}', flush=True)
            print(f'  removal={en_r} slope={en_s} collapse={en_c}',
                  flush=True)
            step_ok = True

            # S1: augmentation
            aug_ready = aug_dir.exists() and ply_count(aug_dir) > 0
            if step_ok and not aug_ready:
                aug_dir.mkdir(parents=True, exist_ok=True)
                t, ok = run_step(
                    f'S1-aug {abl_name} fold{fold_idx}',
                    'step1_ablation_aug.py',
                    ['--data-dir',        str(data_pool_dir),
                     '--output-dir',      str(aug_dir),
                     '--ablation-config', abl_name,
                     '--variants',        str(P_AUG['n_variants'])],
                    optional=True)
                total_time += t
                if not ok:
                    # remove empty dir so next run re-detects it as missing
                    if aug_dir.exists() and ply_count(aug_dir) == 0:
                        shutil.rmtree(aug_dir, ignore_errors=True)
                    print(f'  [SKIP] S1 failed, skipping {abl_name}',
                          flush=True)
                    step_ok = False
            elif aug_ready:
                print(f'  ✓ S1 done ({ply_count(aug_dir)} PLY)', flush=True)

            # S2: quality filter
            hq_ready = hq_dir.exists() and ply_count(hq_dir) > 0
            if step_ok and not hq_ready:
                t, ok = run_step(
                    f'S2-quality {abl_name} fold{fold_idx}',
                    'step2_quality.py',
                    ['--original-dir', str(data_pool_dir),
                     '--aug-dir',      str(aug_dir),
                     '--threshold',    str(P_AUG['quality_threshold'])],
                    optional=True)
                total_time += t
                if not ok or ply_count(hq_dir) == 0:
                    print('  [SKIP] S2 no output', flush=True)
                    step_ok = False
            elif step_ok and hq_ready:
                print(f'  ✓ S2 done ({ply_count(hq_dir)} PLY)', flush=True)

            # S3: pseudo-labels
            if step_ok and not is_valid_json(lbl_file):
                t, ok = run_step(
                    f'S3-labels {abl_name} fold{fold_idx}',
                    'step3_labels.py',
                    ['--input-dirs',       str(hq_dir),
                     '--output',           str(lbl_file),
                     '--front-percentile', str(PSEUDO_LABEL['front_percentile']),
                     '--slope-min',        str(PSEUDO_LABEL['slope_min']),
                     '--slope-max',        str(PSEUDO_LABEL['slope_max']),
                     '--convex-radius',    str(PSEUDO_LABEL['convex_radius_cm']),
                     '--k-normal',         str(PSEUDO_LABEL['k_normal']),
                     '--tag',             f'{abl_name}_fold{fold_idx}'],
                    optional=True)
                total_time += t
                if not ok:
                    step_ok = False
            elif step_ok and is_valid_json(lbl_file):
                print('  ✓ S3 labels done', flush=True)

            # S4: training
            best_pth = model_dir / 'best_model.pth'
            if step_ok and not best_pth.exists():
                model_dir.mkdir(parents=True, exist_ok=True)
                sn = min(ABLATION_TRAIN['n_sample'],
                         P_AUG['n_sample_150'])
                t, ok = run_step(
                    f'S4-train {abl_name} fold{fold_idx}',
                    'step4_train.py',
                    ['--train-dirs',   str(hq_dir),
                     '--label-file',   str(lbl_file),
                     '--model-dir',    str(model_dir),
                     '--sample-n',     str(sn),
                     '--n-pts',        str(P_MDL['n_points']),
                     '--epochs',       str(ABLATION_TRAIN['epochs']),
                     '--batch-size',   str(ABLATION_TRAIN['batch_size']),
                     '--lr',           str(P_TR['lr']),
                     '--early-stop',   str(P_TR['early_stop']),
                     '--class-weight',
                         str(P_TR['class_weight'][0]),
                         str(P_TR['class_weight'][1]),
                     '--device',       P_TR['device'],
                     '--fold-tag',    f'{abl_name}_fold{fold_idx}'],
                    optional=True)
                total_time += t
                if not ok:
                    step_ok = False
            elif step_ok and best_pth.exists():
                print('  ✓ S4 model done', flush=True)

            # S5: evaluation
            if (step_ok and best_pth.exists()
                    and is_valid_json(test_lbl)):
                cfg_tmp = abl_fold_dir / f'_eval_{abl_name}.json'
                with open(cfg_tmp, 'w', encoding='utf-8') as _f:
                    json.dump(
                        {'model_path':  str(best_pth),
                         'label_file':  str(test_lbl),
                         'test_dir':    str(test_dir_path),
                         'n_pts':       P_MDL['n_points'],
                         'out_metrics': str(metrics_file),
                         'out_val':     str(abl_dir / 'best_val_metrics.json'),
                         'branch':      abl_name,
                         'hist_path':   str(model_dir / 'training_history.json')},
                        _f, indent=2)
                run_step(
                    f'S5-eval {abl_name} fold{fold_idx}',
                    'step5_evaluate_fixed.py',
                    ['--mode', 'test-single', '--cfg', str(cfg_tmp)],
                    optional=True)
                n_done += 1

    run_step('Ablation summary charts', 'step6_ablation_analysis.py',
             ['--ablation-dir', str(JOURNAL / 'ablation'),
              '--output-dir',   str(JOURNAL / 'figures'),
              '--mode',         'ablation'],
             optional=True)

    print(f'\n  Ablation: done={n_done} skipped={n_skip}', flush=True)
    return total_time


# ── Backbone comparison experiments ───────────────────────────────────────

def run_backbone_experiments(P_TR, P_MDL, P_AUG, n_folds: int) -> float:
    """
    Train PointNeXt / KPConv / PTv3 / RandLA-Net on every data branch.
    PointNet++ results are reused from the original experiment (outputs4/).
    Already-finished runs (test_metrics.json present) are automatically
    skipped, enabling safe resume after interruption.
    """
    backbones_to_run = [b for b in BACKBONE_COMPARISON if b != 'pointnet2']

    print(f"\n{'#' * 70}", flush=True)
    print('  [BACKBONE] Multi-architecture model-agnostic validation',
          flush=True)
    print(f'  Backbones to train: {backbones_to_run}', flush=True)
    print(f'  Data branches: {len(BACKBONE_DATA_BRANCHES)} (all original)',
          flush=True)
    print(f'  Folds: {n_folds}', flush=True)
    print(f"{'#' * 70}", flush=True)

    total_time = 0.0
    test_lbl   = OUTROOT / 'labels_test' / 'pseudo_labels.json'
    test_dir_p = OUTROOT / 'test_data'
    n_done, n_skip = 0, 0

    for fold_idx in range(min(n_folds, N_CV_FOLDS)):
        print(f'\n  ── Fold {fold_idx + 1}/{n_folds}'
              f' ─────────────────────────────', flush=True)

        for branch in BACKBONE_DATA_BRANCHES:
            aug_hq, orig_dir, lbl_file = get_branch_data(fold_idx, branch)
            if not is_valid_json(lbl_file):
                print(f'  [SKIP] {branch} fold{fold_idx}: label missing',
                      flush=True)
                continue

            n_sample = TRAIN_BRANCHES[branch][1]
            use_orig = TRAIN_BRANCHES[branch][2]
            actual_n = (min(n_sample, P_AUG['n_sample_150'])
                        if n_sample > 11
                        else min(n_sample, P_AUG['n_sample_11']))
            # small-data branches use smaller batch
            bs = (min(P_TR['batch_size'], 8)
                  if n_sample <= 11 or branch == 'baseline_cv'
                  else P_TR['batch_size'])

            for backbone in backbones_to_run:
                bk_dir  = (JOURNAL / 'backbones'
                           / f'fold_{fold_idx:02d}'
                           / backbone / branch)
                metrics = bk_dir / 'test_metrics.json'
                best    = bk_dir / 'best_model.pth'
                bk_cfg  = BACKBONE_TRAIN[backbone]

                # ── skip if already done ─────────────────────────────────
                if metrics.exists():
                    print(f'  ✓ {backbone}/{branch} fold{fold_idx}'
                          ' — done, skipping', flush=True)
                    n_skip += 1
                    continue

                print(f'\n  {backbone.upper()} | {branch}'
                      f' | Fold {fold_idx + 1}', flush=True)
                print(f'  lr={bk_cfg["lr"]}  ep={bk_cfg["epochs"]}'
                      f'  es={bk_cfg["early_stop"]}'
                      f'  n={actual_n}  bs={bs}', flush=True)
                bk_dir.mkdir(parents=True, exist_ok=True)

                train_args = [
                    '--backbone',    backbone,
                    '--label-file',  str(lbl_file),
                    '--model-dir',   str(bk_dir),
                    '--sample-n',    str(actual_n),
                    '--n-pts',       str(P_MDL['n_points']),
                    '--epochs',      str(bk_cfg['epochs']),
                    '--batch-size',  str(bs),
                    '--lr',          str(bk_cfg['lr']),
                    '--early-stop',  str(bk_cfg['early_stop']),
                    '--class-weight',
                        str(P_TR['class_weight'][0]),
                        str(P_TR['class_weight'][1]),
                    '--device',      P_TR['device'],
                    '--fold-tag',   f'{backbone}_{branch}_fold{fold_idx}',
                ]
                if (aug_hq and aug_hq.exists()
                        and ply_count(aug_hq) > 0):
                    train_args += ['--train-dirs', str(aug_hq)]
                if use_orig and orig_dir and orig_dir.exists():
                    train_args += ['--orig-train-dirs', str(orig_dir)]

                t, ok = run_step(
                    f'train {backbone}|{branch}|fold{fold_idx}',
                    'step4_train_backbones.py',
                    train_args, optional=True)
                total_time += t

                if ok and best.exists() and is_valid_json(test_lbl):
                    cfg_tmp = bk_dir / '_eval_cfg.json'
                    with open(cfg_tmp, 'w', encoding='utf-8') as _f:
                        json.dump(
                            {'model_path':  str(best),
                             'label_file':  str(test_lbl),
                             'test_dir':    str(test_dir_p),
                             'n_pts':       P_MDL['n_points'],
                             'out_metrics': str(metrics),
                             'out_val':     str(bk_dir / 'best_val_metrics.json'),
                             'branch':     f'{backbone}_{branch}',
                             'hist_path':   str(bk_dir / 'training_history.json')},
                            _f, indent=2)
                    run_step(
                        f'eval {backbone}|{branch}|fold{fold_idx}',
                        'step5_evaluate_fixed.py',
                        ['--mode', 'test-single', '--cfg', str(cfg_tmp)],
                        optional=True)
                    n_done += 1

    run_step('Backbone comparison summary', 'step6_ablation_analysis.py',
             ['--backbone-dir', str(JOURNAL / 'backbones'),
              '--orig-dir',     str(OUTROOT),
              '--output-dir',   str(JOURNAL / 'figures'),
              '--mode',         'backbone'],
             optional=True)

    print(f'\n  Backbone: done={n_done} skipped={n_skip}', flush=True)
    return total_time


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='LSDA Journal Extension Experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument('--data-dir',   default='./data',
                        help='Directory containing raw PLY files')
    parser.add_argument('--debug',      action='store_true',
                        help='Quick sanity-check mode: 1 fold, 3 epochs')
    parser.add_argument('--experiment', default='all',
                        choices=['all', 'ablation', 'backbone', 'analysis'],
                        help='Which experiment category to run')
    args = parser.parse_args()

    P_AUG = {**AUGMENTATION}
    P_TR  = {**TRAIN}
    P_MDL = {**MODEL}
    abl_folds = ABLATION_TRAIN['n_folds']
    bk_folds  = BACKBONE_N_FOLDS

    if args.debug:
        P_AUG.update({'n_variants':   DEBUG['n_variants'],
                      'n_sample_150': DEBUG['n_sample_150'],
                      'n_sample_11':  DEBUG['n_sample_11']})
        P_MDL.update({'n_points':     DEBUG['n_points']})
        P_TR.update({'epochs':        DEBUG['epochs'],
                     'batch_size':    DEBUG['batch_size'],
                     'early_stop':    DEBUG['early_stop']})
        abl_folds = DEBUG['ablation_folds']
        bk_folds  = DEBUG['backbone_folds']
        print('\n[DEBUG MODE] activated', flush=True)
        print(f'  abl_folds={abl_folds} bk_folds={bk_folds}'
              f' n_variants={P_AUG["n_variants"]}'
              f' epochs={P_TR["epochs"]}', flush=True)

    JOURNAL.mkdir(parents=True, exist_ok=True)
    (JOURNAL / 'figures').mkdir(exist_ok=True)
    (JOURNAL / 'reports').mkdir(exist_ok=True)

    print(f'\n{"=" * 70}', flush=True)
    print(f'  LSDA Journal Extension  |  experiment={args.experiment}',
          flush=True)
    print(f'{"=" * 70}', flush=True)

    if OUTROOT.exists():
        check_outputs4_complete()
    else:
        print(f'  [WARN] outputs4/ not found — skipping completeness check',
              flush=True)
    print(f'  Output dir: {JOURNAL}', flush=True)

    total_time = 0.0

    if args.experiment in ('all', 'ablation'):
        total_time += run_ablation_experiments(P_TR, P_MDL, P_AUG, abl_folds)

    if args.experiment in ('all', 'backbone'):
        total_time += run_backbone_experiments(P_TR, P_MDL, P_AUG, bk_folds)

    if args.experiment in ('all', 'analysis'):
        run_step('Full analysis report', 'step6_ablation_analysis.py',
                 ['--ablation-dir', str(JOURNAL / 'ablation'),
                  '--backbone-dir', str(JOURNAL / 'backbones'),
                  '--orig-dir',     str(OUTROOT),
                  '--output-dir',   str(JOURNAL / 'figures'),
                  '--report-dir',   str(JOURNAL / 'reports'),
                  '--mode',         'all'],
                 optional=True)

    print(f'\n{"=" * 70}', flush=True)
    print(f'  Journal extension complete! Elapsed: {total_time / 60:.1f} min',
          flush=True)
    print(f'  Results: {JOURNAL}', flush=True)
    print(f'{"=" * 70}\n', flush=True)


if __name__ == '__main__':
    main()
