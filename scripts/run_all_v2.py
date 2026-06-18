"""
run_all_v2.py — LSDA 期刊版完整主控（智能跳过 + 详细输出）
"""
import argparse, subprocess, sys, time, json, io, shutil
from pathlib import Path

if sys.platform == 'win32':
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        elif hasattr(sys.stdout, 'buffer') and not isinstance(sys.stdout, io.TextIOWrapper):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))
from config import (DATASET, CV_SPLITS, N_CV_FOLDS, AUGMENTATION, PSEUDO_LABEL,
                    MODEL, TRAIN, TRAIN_BRANCHES, DEBUG)
OUTROOT = BASE / 'outputs4'


def ply_count(d):
    return len(list(Path(d).glob('*.ply'))) if Path(d).exists() else 0

def is_valid_json(path):
    if not Path(path).exists(): return False
    try:
        with open(path, encoding='utf-8') as f: data = json.load(f)
        return bool(data)
    except Exception: return False


def run_step(label, script, args_list, optional=False):
    print(f'\n{"="*70}', flush=True)
    print(f'  {label}', flush=True)
    print(f'{"="*70}', flush=True)
    cmd = [sys.executable, str(BASE / script)] + [str(a) for a in args_list]
    t0  = time.time()
    try:
        ret = subprocess.run(cmd, shell=False)
        rc  = ret.returncode
    except KeyboardInterrupt:
        print(f'\n  ⚠️  {label} 被中断', flush=True)
        return time.time() - t0, False
    except Exception as e:
        print(f'\n  ⚠️  启动失败: {e}', flush=True)
        if optional: return time.time() - t0, False
        sys.exit(1)
    el = time.time() - t0
    if rc != 0:
        if optional:
            print(f'  ⚠️  {label} 失败 [{el:.1f}s]', flush=True)
            return el, False
        print(f'  ❌ {label} 失败', flush=True); sys.exit(1)
    print(f'  ✅ {label} 完成 [{el:.1f}s]', flush=True)
    return el, True


# 完成度检查
def check_s1(fd, k): d=fd/k/'augmented'; return d.exists() and ply_count(d)>0
def check_s2(fd, k): d=fd/k/'high_quality'; return d.exists() and ply_count(d)>0
def check_s3(fd, br): return is_valid_json(fd/br/'pseudo_labels.json')
def check_s4(fd, br): return ((fd/br/'model'/'best_model.pth').exists() and
                               (fd/br/'model'/'training_history.json').exists())
def check_s5(fd, br): return (fd/br/'test_metrics.json').exists()


def setup_fold_data(data_dir, train_stems, val_stems, fold_dir):
    data_dir = Path(data_dir)
    for split, stems in [('data_train', train_stems), ('data_val', val_stems)]:
        sd = fold_dir / split; sd.mkdir(parents=True, exist_ok=True)
        for stem in stems:
            for ext in ('ply','PLY'):
                ml = list(data_dir.glob(f'{stem}.{ext}'))
                if ml and not (sd/ml[0].name).exists():
                    shutil.copy2(ml[0], sd/ml[0].name); break
    pool = fold_dir/'data_pool'; pool.mkdir(parents=True, exist_ok=True)
    for stem in train_stems+val_stems:
        for ext in ('ply','PLY'):
            ml = list(data_dir.glob(f'{stem}.{ext}'))
            if ml and not (pool/ml[0].name).exists():
                shutil.copy2(ml[0], pool/ml[0].name); break
    print(f"  数据: train={len(train_stems)} val={len(val_stems)} pool={len(train_stems)+len(val_stems)}", flush=True)
    return pool


def setup_test_dir(data_dir, stems, test_dir):
    Path(test_dir).mkdir(parents=True, exist_ok=True)
    for stem in stems:
        for ext in ('ply','PLY'):
            ml = list(Path(data_dir).glob(f'{stem}.{ext}'))
            if ml and not (Path(test_dir)/ml[0].name).exists():
                shutil.copy2(ml[0], Path(test_dir)/ml[0].name); break


def main():
    parser = argparse.ArgumentParser(description='LSDA 期刊版完整主控')
    parser.add_argument('--data-dir',  default='./data')
    parser.add_argument('--debug',     action='store_true')
    parser.add_argument('--from-fold', type=int, default=0)
    parser.add_argument('--n-folds',   type=int, default=None)
    parser.add_argument('--step',      type=int, default=0)
    parser.add_argument('--skip-orig', action='store_true')
    parser.add_argument('--only-new',  action='store_true')
    args = parser.parse_args()

    if args.only_new:
        print("\n[--only-new] 直接运行期刊扩展实验", flush=True)
        run_step('期刊扩展实验', 'run_journal.py', ['--data-dir', args.data_dir])
        return

    P_AUG = {**AUGMENTATION}; P_PL = {**PSEUDO_LABEL}
    P_TR  = {**TRAIN};        P_MDL = {**MODEL}

    if args.debug:
        P_AUG.update({'n_variants': DEBUG['n_variants'],
                      'n_sample_150': DEBUG['n_sample_150'],
                      'n_sample_11': DEBUG['n_sample_11']})
        P_MDL.update({'n_points': DEBUG['n_points']})
        P_TR.update({'epochs': DEBUG['epochs'], 'batch_size': DEBUG['batch_size'],
                     'early_stop': DEBUG['early_stop']})
        n_folds = DEBUG['n_cv_folds']
        print('\n[调试模式]\n', flush=True)
    else:
        n_folds = args.n_folds if args.n_folds else N_CV_FOLDS

    OUTROOT.mkdir(parents=True, exist_ok=True)
    (OUTROOT/'results').mkdir(exist_ok=True)
    (OUTROOT/'figures').mkdir(exist_ok=True)

    test_dir = OUTROOT/'test_data'
    test_lbl = OUTROOT/'labels_test'/'pseudo_labels.json'
    test_lbl.parent.mkdir(parents=True, exist_ok=True)
    setup_test_dir(args.data_dir, DATASET['test_set'], test_dir)

    total_time = 0.0

    if not is_valid_json(test_lbl):
        run_step('测试集伪标签', 'step3_labels.py',
                 ['--input-dirs', str(test_dir), '--output', str(test_lbl),
                  '--front-percentile', str(P_PL['front_percentile']),
                  '--slope-min', str(P_PL['slope_min']),
                  '--slope-max', str(P_PL['slope_max']),
                  '--convex-radius', str(P_PL['convex_radius_cm']),
                  '--k-normal', str(P_PL['k_normal']), '--tag', 'test_set'])
    else:
        print(f"\n  ✓ 测试集伪标签已存在", flush=True)

    for fold_idx in range(args.from_fold, n_folds):
        val_stems, train_stems = CV_SPLITS[fold_idx]
        fold_dir = OUTROOT / f'fold_{fold_idx:02d}'
        fold_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'#'*70}", flush=True)
        print(f"  FOLD {fold_idx+1}/{n_folds} | train={train_stems}", flush=True)
        print(f"{'#'*70}", flush=True)
        data_pool = setup_fold_data(args.data_dir, train_stems, val_stems, fold_dir)
        sd = lambda s: args.step == 0 or args.step == s

        # Step 1
        if sd(1) and not args.skip_orig:
            for mode, key in [('lsda','aug_lsda'),('traditional_only','aug_trad'),
                               ('loading_only','aug_loadsim')]:
                if check_s1(fold_dir, key):
                    print(f"\n  ✓ S1/{key} 已完成 ({ply_count(fold_dir/key/'augmented')}变体)", flush=True); continue
                aug_dir = fold_dir/key/'augmented'; aug_dir.mkdir(parents=True, exist_ok=True)
                t, _ = run_step(f'S1增强 {key} fold{fold_idx}', 'step1_augmentation.py',
                                ['--data-dir', str(data_pool), '--output-dir', str(aug_dir),
                                 '--variants', str(P_AUG['n_variants']),
                                 '--workers', str(P_AUG['n_workers']), '--mode', mode])
                total_time += t

        # Step 2
        if sd(2) and not args.skip_orig:
            for key in ['aug_lsda','aug_trad','aug_loadsim']:
                if check_s2(fold_dir, key):
                    print(f"\n  ✓ S2/{key} 已完成 ({ply_count(fold_dir/key/'high_quality')}高质量)", flush=True); continue
                t, _ = run_step(f'S2筛选 {key} fold{fold_idx}', 'step2_quality.py',
                                ['--original-dir', str(data_pool),
                                 '--aug-dir', str(fold_dir/key/'augmented'),
                                 '--threshold', str(P_AUG['quality_threshold'])])
                total_time += t

        # Step 3
        if sd(3) and not args.skip_orig:
            hq = {'lsda': fold_dir/'aug_lsda'/'high_quality',
                  'trad': fold_dir/'aug_trad'/'high_quality',
                  'loadsim': fold_dir/'aug_loadsim'/'high_quality'}
            for br, (src, ns, uo, lcn, grp) in TRAIN_BRANCHES.items():
                if check_s3(fold_dir, br):
                    print(f"\n  ✓ S3/{br} 已完成", flush=True); continue
                (fold_dir/br).mkdir(parents=True, exist_ok=True)
                lf = fold_dir/br/'pseudo_labels.json'
                idirs = ([str(data_pool)] if br=='baseline_cv'
                         else ([str(hq[src]), str(data_pool)] if uo else [str(hq[src])]))
                t, _ = run_step(f'S3伪标签 {br} fold{fold_idx}', 'step3_labels.py',
                                ['--input-dirs']+idirs+
                                ['--output', str(lf),
                                 '--front-percentile', str(P_PL['front_percentile']),
                                 '--slope-min', str(P_PL['slope_min']),
                                 '--slope-max', str(P_PL['slope_max']),
                                 '--convex-radius', str(P_PL['convex_radius_cm']),
                                 '--k-normal', str(P_PL['k_normal']),
                                 '--tag', f'{br}_fold{fold_idx}'])
                total_time += t

        # Step 4
        if sd(4) and not args.skip_orig:
            hq = {'lsda': fold_dir/'aug_lsda'/'high_quality',
                  'trad': fold_dir/'aug_trad'/'high_quality',
                  'loadsim': fold_dir/'aug_loadsim'/'high_quality'}
            for br, (src, ns, uo, lcn, grp) in TRAIN_BRANCHES.items():
                if check_s4(fold_dir, br):
                    print(f"\n  ✓ S4/{br} 已完成", flush=True); continue
                lf = fold_dir/br/'pseudo_labels.json'; md = fold_dir/br/'model'
                md.mkdir(parents=True, exist_ok=True)
                aug_hq = None if src=='none' else hq.get(src)
                orig   = data_pool if uo else None
                bs = min(P_TR['batch_size'], 8) if (br=='baseline_cv' or ns<=11) else P_TR['batch_size']
                actual = min(ns, P_AUG['n_sample_11']) if ns<=11 else min(ns, P_AUG['n_sample_150'])
                ta = []
                if aug_hq: ta += ['--train-dirs', str(aug_hq)]
                if orig:   ta += ['--orig-train-dirs', str(orig)]
                ta += ['--label-file', str(lf), '--model-dir', str(md),
                       '--sample-n', str(actual), '--n-pts', str(P_MDL['n_points']),
                       '--epochs', str(P_TR['epochs']), '--batch-size', str(bs),
                       '--lr', str(P_TR['lr']), '--early-stop', str(P_TR['early_stop']),
                       '--class-weight', str(P_TR['class_weight'][0]), str(P_TR['class_weight'][1]),
                       '--device', P_TR['device'], '--fold-tag', f'{br}_fold{fold_idx}']
                t, _ = run_step(f'S4训练 {br} [{lcn}] fold{fold_idx}', 'step4_train.py', ta,
                                optional=(ns<=11 or br=='baseline_cv'))
                total_time += t

        # Step 5
        if sd(5):
            for br in TRAIN_BRANCHES:
                if check_s5(fold_dir, br):
                    print(f"\n  ✓ S5/{br} 已完成", flush=True); continue
                mp = fold_dir/br/'model'/'best_model.pth'
                cfg = fold_dir/f'_eval_{br}.json'
                with open(cfg,'w',encoding='utf-8') as _f:
                    json.dump({'model_path': str(mp), 'label_file': str(test_lbl),
                               'test_dir': str(test_dir), 'n_pts': P_MDL['n_points'],
                               'out_metrics': str(fold_dir/br/'test_metrics.json'),
                               'out_val': str(fold_dir/br/'best_val_metrics.json'),
                               'branch': br,
                               'hist_path': str(fold_dir/br/'model'/'training_history.json')},
                              _f, indent=2)
                run_step(f'S5评估 {br} fold{fold_idx}', 'step5_evaluate_fixed.py',
                         ['--mode','test-single','--cfg', str(cfg)], optional=True)

        print(f"\n  ✅ Fold {fold_idx+1} 完成", flush=True)

    if args.step == 0 and not args.skip_orig:
        t, _ = run_step('原始实验汇总', 'step5_evaluate_fixed.py',
                        ['--outputs-root', str(OUTROOT),
                         '--results-dir', str(OUTROOT/'results'),
                         '--figures-dir', str(OUTROOT/'figures'), '--mode', 'summary'])
        total_time += t

    if args.step == 0:
        run_step('期刊扩展实验', 'run_journal.py',
                 ['--data-dir', args.data_dir], optional=True)

    print(f'\n{"="*70}', flush=True)
    print(f'  完成！耗时: {total_time/60:.1f} 分钟', flush=True)
    print(f'{"="*70}\n', flush=True)


if __name__ == '__main__':
    main()
