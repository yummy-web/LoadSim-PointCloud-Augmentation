"""
run_all.py — LSDA 7分支双组对照实验主控脚本 (v4)
=================================================

【实验目的】
  研究在散料堆铲料区域识别任务中，装载模拟形变增强相比传统几何增强
  是否能提升分割性能，并通过等量对照排除样本量影响。

【7个训练分支】
  第一组 A (N=150, 纯变体):
    baseline_cv  — 仅11个原始文件 (交叉验证基线)
    lsda_150     — 150个LSDA变体 (装载模拟+传统)
    trad_150     — 150个传统变体 (旋转/缩放/噪声/RBF)
    loadsim_150  — 150个纯装载模拟变体

  第二组 B (N=11, 等量交叉验证):
    lsda_11      — 11个LSDA变体
    trad_11      — 11个传统变体
    loadsim_11   — 11个纯装载模拟变体

【流水线 (每折)】
  Step 1: 3模式增强 → lsda/trad/loadsim (各40变体/文件)
  Step 2: 3组质量筛选 → high_quality/
  Step 3: 7分支伪标签生成 (算法完全相同，保证公平性)
  Step 4: 7分支模型训练
  Step 5: 测试集评估 + 本折图表
  综合:   两组汇总图表与统计分析

用法:
    python run_all.py --data-dir "./data"
    python run_all.py --data-dir "./data" --debug
    python run_all.py --data-dir "./data" --from-fold 1
    python run_all.py --data-dir "./data" --step 4
"""
import argparse, subprocess, sys, time, json, io, shutil
from pathlib import Path

if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except Exception: pass

BASE    = Path(__file__).parent
sys.path.insert(0, str(BASE))
from config import (DATASET, CV_SPLITS, N_CV_FOLDS, AUGMENTATION, PSEUDO_LABEL,
                     MODEL, TRAIN, TRAIN_BRANCHES, DEBUG)

OUTROOT = BASE / 'outputs4'


# ── 辅助函数 ──────────────────────────────────────────────────────────────

def run_step(label, script, args_str, optional=False):
    print(f'\n{"="*70}\n  {label}\n{"="*70}')
    t0  = time.time()
    cmd = f'{sys.executable} "{BASE / script}" {args_str}'
    ret = subprocess.run(cmd, shell=True)
    el  = time.time() - t0
    if ret.returncode != 0:
        if optional:
            print(f'  ⚠️  {label} 失败，继续...')
            return el, False
        print(f'  ❌ {label} 失败'); sys.exit(1)
    print(f'  ✅ {label} 完成 ({el:.1f}s)')
    return el, True


def p(path_obj):
    """Path → 正斜杠字符串，避免Windows路径转义"""
    return Path(path_obj).as_posix()


def setup_fold_data(data_dir, train_stems, val_stems, fold_dir):
    """准备折数据: data_train/ data_val/ data_pool/"""
    data_dir = Path(data_dir)
    for split, stems in [('data_train', train_stems), ('data_val', val_stems)]:
        split_dir = fold_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for stem in stems:
            matches = (list(data_dir.glob(f'{stem}.ply')) +
                       list(data_dir.glob(f'{stem}.PLY')))
            if matches and not (split_dir / matches[0].name).exists():
                shutil.copy2(matches[0], split_dir / matches[0].name)

    pool_dir = fold_dir / 'data_pool'
    pool_dir.mkdir(parents=True, exist_ok=True)
    for stem in train_stems + val_stems:
        matches = (list(data_dir.glob(f'{stem}.ply')) +
                   list(data_dir.glob(f'{stem}.PLY')))
        if matches and not (pool_dir / matches[0].name).exists():
            shutil.copy2(matches[0], pool_dir / matches[0].name)

    print(f"  数据准备: train={len(train_stems)}, val={len(val_stems)}, pool={len(train_stems)+len(val_stems)}")
    return pool_dir


def setup_test_dir(data_dir, test_stems, test_dir):
    test_dir = Path(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)
    for stem in test_stems:
        matches = (list(Path(data_dir).glob(f'{stem}.ply')) +
                   list(Path(data_dir).glob(f'{stem}.PLY')))
        if matches and not (test_dir / matches[0].name).exists():
            shutil.copy2(matches[0], test_dir / matches[0].name)


def mk_train_args(aug_hq_dir, orig_dir, lbl_file, model_dir,
                  sample_n, fold_tag, bs, P_TR, P_MDL):
    """构建 step4_train.py 命令行参数 (正斜杠避免Windows转义)"""
    aug_part  = f'--train-dirs "{p(aug_hq_dir)}" ' if aug_hq_dir else ''
    orig_part = f'--orig-train-dirs "{p(orig_dir)}" ' if orig_dir else ''
    return (
        f'{aug_part}'
        f'{orig_part}'
        f'--label-file "{p(lbl_file)}" '
        f'--model-dir "{p(model_dir)}" '
        f'--sample-n {sample_n} '
        f'--n-pts {P_MDL["n_points"]} '
        f'--epochs {P_TR["epochs"]} '
        f'--batch-size {bs} '
        f'--lr {P_TR["lr"]} '
        f'--early-stop {P_TR["early_stop"]} '
        f'--class-weight {P_TR["class_weight"][0]} {P_TR["class_weight"][1]} '
        f'--device {P_TR["device"]} '
        f'--fold-tag {fold_tag}'
    )


# ── 主程序 ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='LSDA 7分支双组对照实验 v4',
                                     epilog=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir',  default='./data')
    parser.add_argument('--debug',     action='store_true')
    parser.add_argument('--from-fold', type=int, default=0)
    parser.add_argument('--n-folds',   type=int, default=None)
    parser.add_argument('--step',      type=int, default=0)
    args = parser.parse_args()

    P_AUG = {**AUGMENTATION}
    P_PL  = {**PSEUDO_LABEL}
    P_TR  = {**TRAIN}
    P_MDL = {**MODEL}

    if args.debug:
        P_AUG.update({'n_variants':   DEBUG['n_variants'],
                      'n_sample_150': DEBUG['n_sample_150'],
                      'n_sample_11':  DEBUG['n_sample_11']})
        P_MDL.update({'n_points':     DEBUG['n_points']})
        P_TR.update({'epochs':        DEBUG['epochs'],
                     'batch_size':    DEBUG['batch_size'],
                     'early_stop':    DEBUG['early_stop']})
        n_folds = DEBUG['n_cv_folds']
        print('\n[调试模式] 已激活\n')
    else:
        n_folds = args.n_folds if args.n_folds else N_CV_FOLDS

    OUTROOT.mkdir(parents=True, exist_ok=True)
    (OUTROOT / 'results').mkdir(exist_ok=True)
    (OUTROOT / 'figures').mkdir(exist_ok=True)

    # 固定测试集 & 测试集伪标签 (仅生成一次)
    test_dir        = OUTROOT / 'test_data'
    test_label_file = OUTROOT / 'labels_test' / 'pseudo_labels.json'
    test_label_file.parent.mkdir(parents=True, exist_ok=True)
    setup_test_dir(args.data_dir, DATASET['test_set'], test_dir)

    if not test_label_file.exists():
        run_step(
            '测试集伪标签 (一次性)',
            'step3_labels.py',
            f'--input-dirs "{p(test_dir)}" '
            f'--output "{p(test_label_file)}" '
            f'--front-percentile {P_PL["front_percentile"]} '
            f'--slope-min {P_PL["slope_min"]} --slope-max {P_PL["slope_max"]} '
            f'--convex-radius {P_PL["convex_radius_cm"]} '
            f'--k-normal {P_PL["k_normal"]} --tag test_set')

    total_time = 0.0

    for fold_idx in range(args.from_fold, n_folds):
        val_stems, train_stems = CV_SPLITS[fold_idx]
        fold_dir = OUTROOT / f'fold_{fold_idx:02d}'
        fold_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'#'*70}")
        print(f"  FOLD {fold_idx+1}/{n_folds} — 7分支双组对照实验")
        print(f"  训练(含验证): {train_stems}")
        print(f"  宏观折验证:   {val_stems}")
        print(f"{'#'*70}")

        data_pool_dir = setup_fold_data(args.data_dir, train_stems, val_stems, fold_dir)

        def sd(step): return args.step == 0 or args.step == step

        # ── Step 1: 3种模式数据增强 ──────────────────────────────────────
        if sd(1):
            for mode, aug_key in [
                ('lsda',             'aug_lsda'),
                ('traditional_only', 'aug_trad'),
                ('loading_only',     'aug_loadsim'),
            ]:
                aug_dir = fold_dir / aug_key / 'augmented'
                aug_dir.mkdir(parents=True, exist_ok=True)
                t, _ = run_step(
                    f'Fold{fold_idx+1} Step1: {aug_key} 增强 (mode={mode})',
                    'step1_augmentation.py',
                    f'--data-dir "{p(data_pool_dir)}" '
                    f'--output-dir "{p(aug_dir)}" '
                    f'--variants {P_AUG["n_variants"]} '
                    f'--workers {P_AUG["n_workers"]} '
                    f'--mode {mode}')
                total_time += t

        # ── Step 2: 3组质量筛选 ──────────────────────────────────────────
        if sd(2):
            for aug_key in ['aug_lsda', 'aug_trad', 'aug_loadsim']:
                aug_dir = fold_dir / aug_key / 'augmented'
                t, _ = run_step(
                    f'Fold{fold_idx+1} Step2: {aug_key} 质量筛选',
                    'step2_quality.py',
                    f'--original-dir "{p(data_pool_dir)}" '
                    f'--aug-dir "{p(aug_dir)}" '
                    f'--threshold {P_AUG["quality_threshold"]}')
                total_time += t

        # ── Step 3: 7分支伪标签生成 (算法完全相同) ───────────────────────
        if sd(3):
            # aug_src → high_quality 目录映射
            hq_dirs = {
                'lsda':    fold_dir / 'aug_lsda'    / 'high_quality',
                'trad':    fold_dir / 'aug_trad'    / 'high_quality',
                'loadsim': fold_dir / 'aug_loadsim' / 'high_quality',
            }

            for branch, (aug_src, n_sample, use_orig, label_cn, group) in TRAIN_BRANCHES.items():
                branch_dir = fold_dir / branch
                branch_dir.mkdir(parents=True, exist_ok=True)
                lbl_file = branch_dir / 'pseudo_labels.json'

                # 决定输入目录
                if branch == 'baseline_cv':
                    # 仅原始文件
                    input_dirs = f'"{p(data_pool_dir)}"'
                elif use_orig:
                    # 变体 + 原始
                    input_dirs = f'"{p(hq_dirs[aug_src])}" "{p(data_pool_dir)}"'
                else:
                    # 仅变体
                    input_dirs = f'"{p(hq_dirs[aug_src])}"'

                t, _ = run_step(
                    f'Fold{fold_idx+1} Step3: {branch} 伪标签 [{label_cn}]',
                    'step3_labels.py',
                    f'--input-dirs {input_dirs} '
                    f'--output "{p(lbl_file)}" '
                    f'--front-percentile {P_PL["front_percentile"]} '
                    f'--slope-min {P_PL["slope_min"]} '
                    f'--slope-max {P_PL["slope_max"]} '
                    f'--convex-radius {P_PL["convex_radius_cm"]} '
                    f'--k-normal {P_PL["k_normal"]} '
                    f'--tag {branch}_fold{fold_idx}')
                total_time += t

        # ── Step 4: 7分支模型训练 ─────────────────────────────────────────
        if sd(4):
            hq_dirs = {
                'lsda':    fold_dir / 'aug_lsda'    / 'high_quality',
                'trad':    fold_dir / 'aug_trad'    / 'high_quality',
                'loadsim': fold_dir / 'aug_loadsim' / 'high_quality',
            }

            for branch, (aug_src, n_sample, use_orig, label_cn, group) in TRAIN_BRANCHES.items():
                lbl_file  = fold_dir / branch / 'pseudo_labels.json'
                model_dir = fold_dir / branch / 'model'
                model_dir.mkdir(parents=True, exist_ok=True)

                # 确定训练数据目录
                aug_hq = None if (aug_src == 'none') else hq_dirs.get(aug_src)
                orig   = data_pool_dir if use_orig else None

                # baseline和N=11分支用小batch
                bs = min(P_TR['batch_size'], 8) if (branch == 'baseline_cv' or n_sample <= 11) else P_TR['batch_size']

                # N=11等量分支: 全部作为训练，不做split (数据太少split无意义)
                # 实际上ShovelSegDataset的split='all'会在80/20中产生8+3，
                # 对于N=11这本身就很少，让其自然分割
                actual_sample = min(n_sample, P_AUG.get('n_sample_11', n_sample)) if n_sample <= 11 else min(n_sample, P_AUG.get('n_sample_150', n_sample))

                t, ok = run_step(
                    f'Fold{fold_idx+1} Step4: {branch} 训练 — [{label_cn}] (Group {group})',
                    'step4_train.py',
                    mk_train_args(
                        aug_hq_dir = aug_hq,
                        orig_dir   = orig,
                        lbl_file   = lbl_file,
                        model_dir  = model_dir,
                        sample_n   = actual_sample,
                        fold_tag   = f'{branch}_fold{fold_idx}',
                        bs         = bs,
                        P_TR       = P_TR,
                        P_MDL      = P_MDL,
                    ),
                    optional=(n_sample <= 11 or branch == 'baseline_cv'))
                total_time += t

        # ── Step 5: 测试集评估 + 本折图表 ────────────────────────────────
        if sd(5):
            for branch in TRAIN_BRANCHES:
                model_path  = fold_dir / branch / 'model' / 'best_model.pth'
                cfg_tmp     = fold_dir / f'_eval_{branch}.json'
                with open(cfg_tmp, 'w', encoding='utf-8') as _f:
                    json.dump({
                        'model_path':  str(model_path),
                        'label_file':  str(test_label_file),
                        'test_dir':    str(test_dir),
                        'n_pts':       P_MDL['n_points'],
                        'out_metrics': str(fold_dir / branch / 'test_metrics.json'),
                        'out_val':     str(fold_dir / branch / 'best_val_metrics.json'),
                        'branch':      branch,
                        'hist_path':   str(fold_dir / branch / 'model' / 'training_history.json'),
                    }, _f, indent=2)
                run_step(
                    f'Fold{fold_idx+1} Step5: {branch} 测试集评估',
                    'step5_evaluate.py',
                    f'--mode test-single --cfg "{p(cfg_tmp)}"',
                    optional=True)

            # 本折图表
            t, _ = run_step(
                f'Fold{fold_idx+1} Step5: 本折图表',
                'step5_evaluate.py',
                f'--outputs-root "{p(OUTROOT)}" '
                f'--results-dir "{p(OUTROOT/"results")}" '
                f'--figures-dir "{p(OUTROOT/"figures")}" '
                f'--mode fold --fold-idx {fold_idx}')
            total_time += t

        print(f"\n  ✅ Fold {fold_idx+1} 完成")

    # ── 综合汇总 (两组) ──────────────────────────────────────────────────
    if args.step == 0:
        t, _ = run_step(
            '最终汇总: 双组综合分析',
            'step5_evaluate.py',
            f'--outputs-root "{p(OUTROOT)}" '
            f'--results-dir "{p(OUTROOT/"results")}" '
            f'--figures-dir "{p(OUTROOT/"figures")}" '
            f'--mode summary')
        total_time += t

    print(f'\n{"="*70}')
    print(f'🎉 7分支双组实验完成！总耗时: {total_time/60:.1f} 分钟')
    print(f'   结果目录: {OUTROOT}')
    print(f'{"="*70}\n')


if __name__ == '__main__':
    main()
