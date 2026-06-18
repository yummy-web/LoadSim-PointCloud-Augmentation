import sys, json, warnings, os
import numpy as np
from pathlib import Path
warnings.filterwarnings('ignore')
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

try:
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    from mpl_toolkits.mplot3d import Axes3D
    HAS_PLT = True
except ImportError:
    HAS_PLT = False; print("[WARN] matplotlib missing")

try:
    import torch; HAS_TORCH = True
except ImportError:
    HAS_TORCH = False; print("[WARN] torch missing")

try:
    import open3d as o3d; HAS_O3D = True
except ImportError:
    HAS_O3D = False

BASE_DIR    = Path(__file__).parent
OUTPUTS_DIR = BASE_DIR / 'outputs4'
TEST_DIR    = OUTPUTS_DIR / 'test_data'
LABELS_TEST = OUTPUTS_DIR / 'labels_test' / 'pseudo_labels.json'
VIZ_DIR     = OUTPUTS_DIR / 'visualization'
VIZ_DIR.mkdir(parents=True, exist_ok=True)

TEST_STEMS       = ['002', 'DJI_3', 'DJI_7']
GROUP_A_BRANCHES = ['baseline_cv', 'trad_150', 'lsda_150', 'loadsim_150']

# 3-colour scheme: gray=bg, red=GT, blue=pred (overlap shows blue over red)
OV_GT_ONLY   = np.array([220,  50,  47], dtype=np.uint8)   # red
OV_PRED_ONLY = np.array([ 31, 119, 180], dtype=np.uint8)   # blue
OV_OVERLAP   = np.array([148, 103, 189], dtype=np.uint8)   # purple (kept for reference)
OV_BG        = np.array([210, 210, 210], dtype=np.uint8)   # gray

BRANCH_TITLE = {
    'baseline_cv': 'Baseline (11 orig.)',
    'trad_150':    'Traditional (N=150)',
    'lsda_150':    'LSDA (N=150)',
    'loadsim_150': 'LoadSim (N=150)',
}
LAYOUT = [['baseline_cv', 'trad_150'], ['lsda_150', 'loadsim_150']]

VC = {
    'original': {'rgb': (220,  50,  47), 'label': 'Original'},
    'trad':     {'rgb': (255, 127,  14), 'label': 'Traditional Variant'},
    'loadsim':  {'rgb': (148, 103, 189), 'label': 'LoadSim Variant'},
    'lsda':     {'rgb': ( 44, 160,  44), 'label': 'LSDA Variant'},
    'bg':       {'rgb': (210, 210, 210), 'label': 'Background'},
}


def load_ply_xyzn(path):
    path = Path(path)
    if HAS_O3D:
        try:
            pcd = o3d.io.read_point_cloud(str(path))
            pts = np.asarray(pcd.points, dtype=np.float32)
            if len(pts) == 0: return None, None
            nrm = np.asarray(pcd.normals, dtype=np.float32)
            return pts, (nrm if len(nrm)==len(pts) else None)
        except Exception: pass
    pts_l, nrms_l, has_nx, n_vert, is_bin = [], [], False, 0, False
    try:
        with open(path, 'rb') as f:
            while True:
                ln = f.readline().decode('ascii', errors='ignore').strip()
                if ln.startswith('element vertex'): n_vert = int(ln.split()[-1])
                elif 'property' in ln.lower() and ' nx' in ln.lower(): has_nx = True
                elif 'binary_little_endian' in ln: is_bin = True
                elif ln == 'end_header': break
            if is_bin:
                df = [('x','f4'),('y','f4'),('z','f4')]
                if has_nx: df += [('nx','f4'),('ny','f4'),('nz','f4')]
                dt = np.dtype(df); d = np.frombuffer(f.read(dt.itemsize*n_vert), dtype=dt)
                p = np.column_stack([d['x'],d['y'],d['z']]).astype(np.float32)
                n = np.column_stack([d['nx'],d['ny'],d['nz']]).astype(np.float32) if has_nx else None
                return p, n
            else:
                for _ in range(n_vert):
                    vs = f.readline().decode('ascii', errors='ignore').split()
                    if len(vs) >= 3:
                        try:
                            pts_l.append([float(v) for v in vs[:3]])
                            if has_nx and len(vs) >= 6:
                                nrms_l.append([float(v) for v in vs[3:6]])
                        except ValueError: pass
    except Exception as e:
        print(f"  [WARN] {path.name}: {e}"); return None, None
    p = np.array(pts_l, dtype=np.float32) if pts_l else None
    n = np.array(nrms_l, dtype=np.float32) if (has_nx and nrms_l and len(nrms_l)==len(pts_l)) else None
    return p, n


def save_ply_colored(path, pts, rgb):
    path = Path(path)
    if HAS_O3D:
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64)/255.0)
            o3d.io.write_point_cloud(str(path), pcd, write_ascii=False); return
        except Exception: pass
    with open(path, 'w', encoding='ascii') as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for i in range(len(pts)):
            f.write(f"{pts[i,0]:.6f} {pts[i,1]:.6f} {pts[i,2]:.6f} "
                    f"{rgb[i,0]} {rgb[i,1]} {rgb[i,2]}\n")


def align(arr, n):
    if len(arr) == n: return arr
    if len(arr) > n: return arr[:n]
    return np.concatenate([arr, np.zeros(n-len(arr), dtype=np.int32)])


def overlay_colors(gt, pred):
    # gray=bg, red=GT, blue=pred, overlap=blue over red
    c = np.tile(OV_BG, (len(gt), 1))
    c[gt.astype(bool)]   = OV_GT_ONLY
    c[pred.astype(bool)] = OV_PRED_ONLY
    return c


def single_colors(labels, rgb_tuple):
    c = np.tile(OV_BG, (len(labels), 1))
    c[labels.astype(bool)] = np.array(rgb_tuple, dtype=np.uint8)
    return c


def best_view(pts, gt):
    sp = pts[gt.astype(bool)] if gt.sum() > 10 else pts
    c  = sp[:, :2].mean(0)
    cv = (sp[:, :2]-c).T @ (sp[:, :2]-c) / max(len(sp), 1)
    try:
        _, ev = np.linalg.eigh(cv)
        azim = float(np.degrees(np.arctan2(ev[-1, 1], ev[-1, 0]))) + 90.0
    except Exception:
        azim = 135.0
    zr = float(pts[:, 2].max()-pts[:, 2].min()) + 1e-8
    xr = float(max(pts[:, 0].max()-pts[:, 0].min(),
                   pts[:, 1].max()-pts[:, 1].min())) + 1e-8
    elev = float(np.clip(np.degrees(np.arctan2(zr, xr*0.6)), 25, 50))
    return elev, azim


def draw_pc(ax, pts, colors_rgb, max_pts=120000, s=3.0):
    N = len(pts)
    if N > max_pts:
        idx = np.random.default_rng(0).choice(N, max_pts, replace=False)
        pts, colors_rgb = pts[idx], colors_rgb[idx]
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
               c=colors_rgb.astype(np.float32)/255.0,
               s=s, linewidths=0, alpha=1.0, rasterized=True)
    ax.set_title('')
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    ax.set_xlabel(''); ax.set_ylabel(''); ax.set_zlabel('')
    ax.grid(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False; pane.set_edgecolor('none')


def find_all_fold_models(branch):
    models = []
    for fi in range(4):
        mp = OUTPUTS_DIR / f'fold_{fi:02d}' / branch / 'model' / 'best_model.pth'
        tm = OUTPUTS_DIR / f'fold_{fi:02d}' / branch / 'test_metrics.json'
        if mp.exists() and tm.exists():
            try:
                miou = float(json.loads(tm.read_text(encoding='utf-8')).get('mIoU', -1))
                models.append((fi, mp, miou))
            except Exception:
                pass
    if models:
        avg = sum(m[2] for m in models) / len(models)
        print(f"  [{branch}] {len(models)} folds  mIoU=[{', '.join(f'{m[2]:.4f}' for m in models)}]  avg={avg:.4f}")
    return models


def predict_ensemble(branch_models, pts, normals, n_pts=4096, n_passes=20):
    if not HAS_TORCH:
        return np.zeros(len(pts), dtype=np.int32)

    sys.path.insert(0, str(BASE_DIR))
    from step4_train import PointNetPPSeg

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    N = len(pts)
    if normals is None or len(normals) != N:
        normals = np.zeros_like(pts)

    all_pred  = np.zeros(N, dtype=np.int32)
    all_count = np.zeros(N, dtype=np.int32)

    for fold_idx, model_path, _ in branch_models:
        model = PointNetPPSeg(in_ch=6, n_cls=2).to(device)
        try:
            ck = torch.load(model_path, map_location=device, weights_only=False)
        except TypeError:
            ck = torch.load(model_path, map_location=device)
        # Use strict=False to allow loading with possible key mismatches
        model.load_state_dict(ck['model_state'], strict=False)
        model.eval()

        with torch.no_grad():
            for pass_i in range(n_passes):
                rng = np.random.default_rng(fold_idx * 1000 + pass_i)
                if N >= n_pts:
                    idx = rng.choice(N, n_pts, replace=False)
                else:
                    idx = np.concatenate([np.arange(N), rng.integers(0, N, n_pts - N)])

                p_sel = pts[idx].astype(np.float32)
                n_sel = normals[idx].astype(np.float32)

                ctr   = p_sel.mean(0)
                scale = np.abs(p_sel - ctr).max() + 1e-8
                p_norm = (p_sel - ctr) / scale
                n_norm = n_sel / (np.linalg.norm(n_sel, axis=1, keepdims=True) + 1e-8)

                xyzn = np.concatenate([p_norm, n_norm], axis=1)
                t    = torch.from_numpy(xyzn.T[None]).to(device)

                try:
                    from torch.cuda.amp import autocast
                    with autocast():
                        logits = model(t)
                except Exception:
                    logits = model(t)

                pred = logits.argmax(1).cpu().numpy().flatten()
                unique_idx = idx[:min(N, n_pts)]
                for i, oi in enumerate(unique_idx):
                    all_pred[oi]  += int(pred[i])
                    all_count[oi] += 1

    sampled = all_count > 0
    labels  = np.zeros(N, dtype=np.int32)
    labels[sampled] = (all_pred[sampled] > all_count[sampled] / 2).astype(np.int32)

    if (~sampled).any():
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(pts[sampled])
            _, nn = tree.query(pts[~sampled], k=1)
            labels[~sampled] = labels[np.where(sampled)[0][nn]]
        except ImportError:
            pass

    return labels


def viz_test_file(stem, pts, all_labels, elev, azim, out_dir):
    """Each branch gets its own full-page figure."""
    if not HAS_PLT: return
    gt = all_labels['gt']
    for branch in GROUP_A_BRANCHES:
        pred = all_labels[branch]
        colors = overlay_colors(gt, pred)
        n_gt   = int(gt.sum())
        n_pred = int(pred.sum())
        n_ovlp = int((gt.astype(bool) & pred.astype(bool)).sum())
        iou    = n_ovlp / max(n_gt + n_pred - n_ovlp, 1)
        fig = plt.figure(figsize=(12, 10))
        ax  = fig.add_subplot(111, projection='3d')
        draw_pc(ax, pts, colors, max_pts=120000, s=3.0)
        ax.view_init(elev=elev, azim=azim)
        fig.subplots_adjust(left=-0.05, right=1.05, top=1.05, bottom=-0.05)
        for fmt in ('svg', 'pdf'):
            fig.savefig(out_dir / f'{stem}_{branch}.{fmt}',
                        format=fmt, bbox_inches='tight', dpi=150)
        fig.savefig(out_dir / f'{stem}_{branch}.png',
                    format='png', bbox_inches='tight', dpi=200)
        plt.close(fig)
        print(f"  {stem}_{branch}.svg/.pdf/.png saved")


def viz_combined(all_stems_data, out_dir):
    if not HAS_PLT: return
    valid = [s for s in TEST_STEMS if s in all_stems_data]
    fig = plt.figure(figsize=(24, 22*len(valid)))
    outer = GridSpec(len(valid), 1, figure=fig, hspace=0.06,
                     top=0.999, bottom=0.001, left=0.01, right=0.99)
    for si, stem in enumerate(valid):
        data = all_stems_data[stem]; pts = data['pts']
        elev, azim = data['elev'], data['azim']; gt = data['labels']['gt']
        inner = GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[si],
                                        hspace=0.10, wspace=0.02)
        for ri, row in enumerate(LAYOUT):
            for ci, branch in enumerate(row):
                ax = fig.add_subplot(inner[ri, ci], projection='3d')
                pred = data['labels'][branch]
                colors = overlay_colors(gt, pred)
                n_gt = int(gt.sum()); n_pred = int(pred.sum())
                n_ovlp = int((gt.astype(bool) & pred.astype(bool)).sum())
                iou = n_ovlp / max(n_gt+n_pred-n_ovlp, 1)
                draw_pc(ax, pts, colors)
                ax.view_init(elev=elev, azim=azim)
    for fmt in ('svg', 'pdf'):
        fig.savefig(out_dir/f'overview_all_tests.{fmt}', format=fmt, bbox_inches='tight')
    plt.close(fig); print("  overview_all_tests.svg/.pdf saved")


def load_json_labels(label_json, stem):
    p = Path(label_json)
    if not p.exists(): return None
    v = json.loads(p.read_text(encoding='utf-8')).get('labels', {}).get(stem)
    if v is None: return None
    arr = v.get('labels', v) if isinstance(v, dict) else v
    return np.array(arr, dtype=np.int32)


def load_gt_labels():
    if not LABELS_TEST.exists(): print(f"[ERROR] {LABELS_TEST}"); return {}
    data = json.loads(LABELS_TEST.read_text(encoding='utf-8'))
    result = {}
    for stem, v in data.get('labels', {}).items():
        arr = v.get('labels', v) if isinstance(v, dict) else v
        result[stem] = np.array(arr, dtype=np.int32)
    print(f"  GT labels: {list(result.keys())}"); return result


def run():
    print("="*65+"\n  Step6: Scooping Region Label Visualization\n"+"="*65)
    print("\n[1/4] Loading GT labels..."); gt_labels = load_gt_labels()
    print("\n[2/4] Finding fold models (all folds, ensemble)...")
    branch_fold_models = {}
    for br in GROUP_A_BRANCHES:
        models = find_all_fold_models(br)
        if models:
            branch_fold_models[br] = models
        else:
            print(f"  [WARN] no models found for {br}")
    print("\n[3/4] Per-file inference (ensemble)..."); all_stems_data = {}
    for stem in TEST_STEMS:
        ply_path = TEST_DIR/f'{stem}.ply'
        if not ply_path.exists(): print(f"  [SKIP] {ply_path}"); continue
        print(f"\n  -- {stem} --")
        pts, normals = load_ply_xyzn(ply_path)
        if pts is None: print("  [ERROR] load failed"); continue
        print(f"  points: {len(pts):,}")
        all_labels = {}
        gt = align(gt_labels.get(stem, np.zeros(len(pts), dtype=np.int32)), len(pts))
        all_labels['gt'] = gt
        print(f"  GT shovel: {gt.sum():,} ({gt.mean()*100:.1f}%)")
        for br in GROUP_A_BRANCHES:
            if br not in branch_fold_models:
                all_labels[br] = np.zeros(len(pts), dtype=np.int32); continue
            print(f"  [{br}] inferring (ensemble {len(branch_fold_models[br])} folds)...", flush=True)
            pred = align(predict_ensemble(branch_fold_models[br], pts, normals), len(pts))
            all_labels[br] = pred
            print(f"  [{br}] shovel={pred.sum():,} ({pred.mean()*100:.1f}%)")
        stem_dir = VIZ_DIR/stem; stem_dir.mkdir(parents=True, exist_ok=True)
        save_ply_colored(stem_dir/f'{stem}_gt.ply', pts,
                         single_colors(gt, VC['original']['rgb']))
        for br in GROUP_A_BRANCHES:
            save_ply_colored(stem_dir/f'{stem}_{br}_overlay.ply',
                             pts, overlay_colors(gt, all_labels[br]))
        print(f"  PLY saved -> {stem_dir.relative_to(BASE_DIR)}/")
        elev, azim = best_view(pts, gt)
        print(f"  view elev={elev:.1f} azim={azim:.1f}")
        viz_test_file(stem, pts, all_labels, elev, azim, stem_dir)
        all_stems_data[stem] = {'pts': pts, 'labels': all_labels,
                                 'elev': elev, 'azim': azim}
    print("\n[4/4] Combined overview...")
    if all_stems_data: viz_combined(all_stems_data, VIZ_DIR)
    print("\n"+"="*65+f"\n  Done -> {VIZ_DIR}\n"+"="*65)


if __name__ == '__main__':
    run()
