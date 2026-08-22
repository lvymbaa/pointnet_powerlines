"""
api.py — FastAPI бэкенд для классификации LAS-файлов нейросетью PointNet.

Установка:
    pip install fastapi uvicorn python-multipart laspy numpy torch

Запуск:
    python api.py

API доступен на http://localhost:8000
"""

import io, os, time, logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import laspy

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────
# КОНФИГ — поменяйте путь к вашей модели
# ─────────────────────────────────────────────────────────
CHECKPOINT   = "checkpoints_fast/best_powerlines.pt"
MAX_SEND_PTS = 250_000   # сколько точек максимум отдаём браузеру
PATCH_SIZE   = 2048
BATCH_SIZE   = 16
NUM_PASSES   = 3

CLASS_MAP_INV = {0: 2, 1: 4, 2: 14}
# Цвета RGB [0..1] для каждого класса
CLASS_COLORS = {
    2:  [0.55, 0.45, 0.33],   # коричневый — ground
    4:  [0.18, 0.55, 0.18],   # зелёный    — vegetation
    14: [1.0,  0.20, 0.0 ],   # красный    — powerlines
    1:  [0.50, 0.50, 0.50],   # серый      — unclassified
}


# ─────────────────────────────────────────────────────────
# АРХИТЕКТУРА (должна совпадать с fast_train.py)
# ─────────────────────────────────────────────────────────
class TNet(nn.Module):
    def __init__(self, k=64):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k,   64,   1)
        self.conv2 = nn.Conv1d(64,  128,  1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1   = nn.Linear(1024, 512)
        self.fc2   = nn.Linear(512,  256)
        self.fc3   = nn.Linear(256,  k * k)
        self.bn1   = nn.BatchNorm1d(64)
        self.bn2   = nn.BatchNorm1d(128)
        self.bn3   = nn.BatchNorm1d(1024)
        self.bn4   = nn.BatchNorm1d(512)
        self.bn5   = nn.BatchNorm1d(256)

    def forward(self, x):
        B = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.max(dim=2).values
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        eye = torch.eye(self.k, device=x.device).flatten().unsqueeze(0).expand(B, -1)
        return (x + eye).view(B, self.k, self.k)


class PointNetSeg(nn.Module):
    def __init__(self, num_classes=3, in_features=6, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, 64,   1)
        self.conv2 = nn.Conv1d(64,          64,   1)
        self.tnet  = TNet(k=64)
        self.conv3 = nn.Conv1d(64,  64,   1)
        self.conv4 = nn.Conv1d(64,  128,  1)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.bn1   = nn.BatchNorm1d(64)
        self.bn2   = nn.BatchNorm1d(64)
        self.bn3   = nn.BatchNorm1d(64)
        self.bn4   = nn.BatchNorm1d(128)
        self.bn5   = nn.BatchNorm1d(1024)
        self.seg1  = nn.Conv1d(1088, 512, 1)
        self.seg2  = nn.Conv1d(512,  256, 1)
        self.seg3  = nn.Conv1d(256,  128, 1)
        self.seg4  = nn.Conv1d(128,  num_classes, 1)
        self.sbn1  = nn.BatchNorm1d(512)
        self.sbn2  = nn.BatchNorm1d(256)
        self.sbn3  = nn.BatchNorm1d(128)
        self.drop  = nn.Dropout(p=dropout)

    def forward(self, x):
        B, N, _ = x.shape
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        local_feat = x
        x = torch.bmm(self.tnet(x), x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        gf = x.max(dim=2).values.unsqueeze(2).expand(-1, -1, N)
        x  = torch.cat([local_feat, gf], dim=1)
        x  = F.relu(self.sbn1(self.seg1(x))); x = self.drop(x)
        x  = F.relu(self.sbn2(self.seg2(x))); x = self.drop(x)
        x  = F.relu(self.sbn3(self.seg3(x))); x = self.seg4(x)
        return x.transpose(2, 1)


# ─────────────────────────────────────────────────────────
# ЗАГРУЗКА МОДЕЛИ
# ─────────────────────────────────────────────────────────
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = None
model_cfg = None


def try_load_model():
    global model, model_cfg
    if not os.path.exists(CHECKPOINT):
        log.warning(f"Чекпоинт не найден: {CHECKPOINT}")
        return
    ck        = torch.load(CHECKPOINT, map_location=device)
    model_cfg = ck["config"]
    model     = PointNetSeg(
        num_classes=model_cfg["num_classes"],
        in_features=model_cfg["in_features"],
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    log.info(f"Модель загружена | эпоха {ck['epoch']} "
             f"| mIoU={ck.get('val_miou',0)*100:.1f}% "
             f"| pw IoU={ck.get('pw_iou',0)*100:.1f}%")
    log.info(f"Устройство: {device}")


# ─────────────────────────────────────────────────────────
# КЛАССИФИКАЦИЯ
# ─────────────────────────────────────────────────────────
def run_classification(x_r, y_r, z_r, inten, rn, rt):
    N = len(x_r)

    def n01(v):
        lo, hi = v.min(), v.max()
        return (v - lo) / (hi - lo + 1e-8)

    feats = np.stack([
        n01(x_r), n01(y_r), n01(z_r), n01(z_r),
        inten / (inten.max() + 1e-8),
        rn / (rt + 1e-8),
    ], axis=1).astype(np.float32)              # (N, 6)

    scores  = np.zeros((N, model_cfg["num_classes"]), dtype=np.float32)
    xy_f    = feats[:, :2]
    rng     = np.random.default_rng(42)
    buf_pts, buf_idx = [], []

    def flush():
        if not buf_pts:
            return
        t = torch.tensor(np.stack(buf_pts), dtype=torch.float32).to(device)
        with torch.no_grad():
            lg = model(t).cpu().numpy()
        for i, idx in enumerate(buf_idx):
            np.add.at(scores, idx, lg[i])
        buf_pts.clear(); buf_idx.clear()

    for _ in range(NUM_PASSES):
        for ci in rng.permutation(N)[::PATCH_SIZE]:
            cx, cy  = xy_f[ci]
            cand    = rng.choice(N, min(N, PATCH_SIZE * 8), replace=False)
            dx = xy_f[cand, 0] - cx
            dy = xy_f[cand, 1] - cy
            pidx = cand[np.argsort(dx*dx + dy*dy)[:PATCH_SIZE]]
            if len(pidx) < PATCH_SIZE:
                pidx = np.concatenate([
                    pidx,
                    rng.choice(pidx, PATCH_SIZE - len(pidx), replace=True)
                ])
            pts = feats[pidx].copy()
            pts[:, 0] -= pts[:, 0].mean()
            pts[:, 1] -= pts[:, 1].mean()
            buf_pts.append(pts); buf_idx.append(pidx)
            if len(buf_pts) >= BATCH_SIZE:
                flush()
    flush()

    # Целевой дополнительный проход по точкам, не получившим ни одного голоса.
    # Это случается с разреженными проводами — их точки редко попадают в ядро
    # случайного патча. Берём их как центры патчей принудительно.
    no_vote_idx = np.where((scores == 0).all(axis=1))[0]
    if len(no_vote_idx) > 0:
        log.info(f"[infer] точек без голосов: {len(no_vote_idx)} — делаем целевой проход")
        for ci in no_vote_idx[::max(1, PATCH_SIZE // 4)]:
            cx, cy  = xy_f[ci]
            dx = xy_f[:, 0] - cx
            dy = xy_f[:, 1] - cy
            near_all = np.argsort(dx*dx + dy*dy)[:PATCH_SIZE]
            pidx = near_all if len(near_all) >= PATCH_SIZE else np.concatenate([
                near_all, rng.choice(near_all, PATCH_SIZE - len(near_all), replace=True)
            ])
            pts = feats[pidx].copy()
            pts[:, 0] -= pts[:, 0].mean()
            pts[:, 1] -= pts[:, 1].mean()
            buf_pts.append(pts); buf_idx.append(pidx)
            if len(buf_pts) >= BATCH_SIZE:
                flush()
        flush()

    # Точки, которые всё равно без голосов — по умолчанию земля (класс 0)
    no_vote = (scores == 0).all(axis=1)
    scores[no_vote, 0] = 1.0
    return scores.argmax(axis=1)   # индексы 0/1/2


# ─────────────────────────────────────────────────────────
# ПРИЛОЖЕНИЕ
# ─────────────────────────────────────────────────────────
app = FastAPI(title="LAS Classifier")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    try_load_model()


@app.get("/health")
def health():
    return {
        "model": "ok" if model else "not loaded",
        "checkpoint": CHECKPOINT,
        "device": str(device),
    }


@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(503, f"Модель не загружена. Путь: {CHECKPOINT}")
    if not file.filename.lower().endswith(".las"):
        raise HTTPException(400, "Нужен .las файл")

    raw = await file.read()
    log.info(f"Файл: {file.filename}  {len(raw)/1e6:.1f} MB")

    try:
        las = laspy.read(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(400, f"Ошибка чтения LAS: {e}")

    N     = las.header.point_count
    x_r   = np.array(las.x, dtype=np.float32)
    y_r   = np.array(las.y, dtype=np.float32)
    z_r   = np.array(las.z, dtype=np.float32)
    dims  = las.point_format.dimension_names
    inten = np.array(las.intensity,         dtype=np.float32) if "intensity"         in dims else np.zeros(N,  np.float32)
    rn    = np.array(las.return_number,     dtype=np.float32) if "return_number"     in dims else np.ones(N,   np.float32)
    rt    = np.array(las.number_of_returns, dtype=np.float32) if "number_of_returns" in dims else np.ones(N,   np.float32)

    t0 = time.time()
    log.info(f"Классификация {N:,} точек...")
    pred_idx = run_classification(x_r, y_r, z_r, inten, rn, rt)
    pred_cls = np.vectorize(CLASS_MAP_INV.get)(pred_idx)   # реальные классы LAS
    elapsed  = round(time.time() - t0, 2)
    log.info(f"Готово за {elapsed}с")

    # Прореживание: провода берём все, остальное — случайно до MAX_SEND_PTS
    pw_idx    = np.where(pred_cls == 14)[0]
    other_idx = np.where(pred_cls != 14)[0]
    n_other   = max(0, MAX_SEND_PTS - len(pw_idx))
    if len(other_idx) > n_other:
        other_idx = np.random.default_rng(0).choice(other_idx, n_other, replace=False)
    keep = np.concatenate([pw_idx, other_idx])

    x_s   = x_r[keep].tolist()
    y_s   = y_r[keep].tolist()
    z_s   = z_r[keep].tolist()
    cls_s = pred_cls[keep].tolist()
    col_s = [CLASS_COLORS.get(int(c), [0.5, 0.5, 0.5]) for c in cls_s]

    stats = {
        "total":        int(N),
        "displayed":    int(len(keep)),
        "ground":       int((pred_cls == 2).sum()),
        "vegetation":   int((pred_cls == 4).sum()),
        "powerlines":   int((pred_cls == 14).sum()),
        "unclassified": int((pred_cls == 1).sum()),
    }
    log.info(f"Статистика: {stats}")

    return JSONResponse({
        "x":      x_s,
        "y":      y_s,
        "z":      z_s,
        "colors": col_s,
        "labels": cls_s,
        "stats":  stats,
        "time_s": elapsed,
    })


@app.post("/classify_filtered")
async def classify_filtered(
    file:        UploadFile = File(...),
    linearity:   float = 0.80,
    elongation:  float = 3.0,
    min_points:  int   = 10,
    eps:         float = 0.8,
    min_samples: int   = 3,
    fallback_cls: int  = 1,
):
    """
    Классификация нейросетью + геометрическая фильтрация ложных ЛЭП за один запрос.
    Параметры фильтрации передаются как query-параметры.
    """
    try:
        from sklearn.cluster import DBSCAN
    except ImportError:
        raise HTTPException(500, "Установите scikit-learn: pip install scikit-learn")

    if model is None:
        raise HTTPException(503, f"Модель не загружена. Путь: {CHECKPOINT}")
    if not file.filename.lower().endswith(".las"):
        raise HTTPException(400, "Нужен .las файл")

    raw = await file.read()
    log.info(f"[classify_filtered] Файл: {file.filename}  {len(raw)/1e6:.1f} MB")

    try:
        las = laspy.read(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(400, f"Ошибка чтения LAS: {e}")

    N     = las.header.point_count
    x_r   = np.array(las.x, dtype=np.float32)
    y_r   = np.array(las.y, dtype=np.float32)
    z_r   = np.array(las.z, dtype=np.float32)
    dims  = las.point_format.dimension_names
    inten = np.array(las.intensity,         dtype=np.float32) if "intensity"         in dims else np.zeros(N, np.float32)
    rn    = np.array(las.return_number,     dtype=np.float32) if "return_number"     in dims else np.ones(N,  np.float32)
    rt    = np.array(las.number_of_returns, dtype=np.float32) if "number_of_returns" in dims else np.ones(N,  np.float32)

    t0 = time.time()
    log.info(f"[classify_filtered] Классификация {N:,} точек...")
    pred_idx = run_classification(x_r, y_r, z_r, inten, rn, rt)
    pred_cls = np.vectorize(CLASS_MAP_INV.get)(pred_idx)
    t_cls = round(time.time() - t0, 2)
    log.info(f"[classify_filtered] Классификация за {t_cls}с")

    # ── Геометрическая фильтрация ──────────────────────────
    pw_mask = pred_cls == 14
    n_pw    = int(pw_mask.sum())
    log.info(f"[classify_filtered] Точек класса 14 до фильтрации: {n_pw:,}")

    cluster_report = []
    removed = 0

    if n_pw > 0:
        pw_xyz = np.stack([x_r[pw_mask], y_r[pw_mask], z_r[pw_mask]], axis=1).astype(np.float64)
        t1 = time.time()
        db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = db.fit_predict(pw_xyz)
        log.info(f"[classify_filtered] DBSCAN за {time.time()-t1:.1f}с")

        pw_indices = np.where(pw_mask)[0]
        result_cls = pred_cls.copy()

        # Шумовые точки — ложные
        noise_global = pw_indices[labels == -1]
        result_cls[noise_global] = fallback_cls
        removed += len(noise_global)

        n_clusters = int(labels.max()) + 1
        for cid in range(n_clusters):
            cmask    = labels == cid
            n_pts    = int(cmask.sum())
            c_pts    = pw_xyz[cmask]
            c_global = pw_indices[cmask]
            lin, elong = pca_linearity(c_pts)
            is_false = n_pts < min_points or lin < linearity or elong < elongation
            reason = []
            if n_pts < min_points: reason.append(f"мало точек ({n_pts}<{min_points})")
            if lin   < linearity:  reason.append(f"не линейный ({lin:.2f}<{linearity})")
            if elong < elongation: reason.append(f"не вытянутый ({elong:.1f}<{elongation})")
            cluster_report.append({
                "id": cid, "n_points": n_pts,
                "linearity": round(lin, 3), "elongation": round(elong, 2),
                "is_false": bool(is_false),
                "reason": ", ".join(reason) if is_false else "провод",
            })
            if is_false:
                result_cls[c_global] = fallback_cls
                removed += n_pts

        cluster_report.sort(key=lambda x: -x["n_points"])
        pred_cls = result_cls

    n_pw_after = int((pred_cls == 14).sum())
    log.info(f"[classify_filtered] Проводов после фильтрации: {n_pw_after:,} (убрано {removed:,})")
    elapsed = round(time.time() - t0, 2)

    # ── Прореживание ───────────────────────────────────────
    pw_idx    = np.where(pred_cls == 14)[0]
    other_idx = np.where(pred_cls != 14)[0]
    n_other   = max(0, MAX_SEND_PTS - len(pw_idx))
    if len(other_idx) > n_other:
        other_idx = np.random.default_rng(0).choice(other_idx, n_other, replace=False)
    keep = np.concatenate([pw_idx, other_idx])

    col_s = [CLASS_COLORS.get(int(c), [0.5, 0.5, 0.5]) for c in pred_cls[keep]]

    stats = {
        "total":        int(N),
        "displayed":    int(len(keep)),
        "ground":       int((pred_cls == 2).sum()),
        "vegetation":   int((pred_cls == 4).sum()),
        "powerlines":   int(n_pw_after),
        "unclassified": int((pred_cls == 1).sum()),
    }
    filter_stats = {
        "powerlines_before": n_pw,
        "powerlines_after":  n_pw_after,
        "removed":           removed,
        "clusters_total":    len(cluster_report),
        "clusters_real":     sum(1 for c in cluster_report if not c["is_false"]),
        "clusters_false":    sum(1 for c in cluster_report if c["is_false"]),
        "clusters":          cluster_report[:30],
    }

    return JSONResponse({
        "x":            x_r[keep].tolist(),
        "y":            y_r[keep].tolist(),
        "z":            z_r[keep].tolist(),
        "colors":       col_s,
        "labels":       pred_cls[keep].tolist(),
        "stats":        stats,
        "filter_stats": filter_stats,
        "time_s":       elapsed,
        "mode":         "classify_filtered",
    })


def prepare_response(x_r, y_r, z_r, pred_cls, N, elapsed=0.0):
    """Общая логика прореживания и формирования ответа."""
    # Провода всегда целиком, остальное — случайно до MAX_SEND_PTS
    pw_idx    = np.where(pred_cls == 14)[0]
    other_idx = np.where(pred_cls != 14)[0]
    n_other   = max(0, MAX_SEND_PTS - len(pw_idx))
    if len(other_idx) > n_other:
        other_idx = np.random.default_rng(0).choice(other_idx, n_other, replace=False)
    keep = np.concatenate([pw_idx, other_idx])

    cls_s = pred_cls[keep].tolist()
    col_s = [CLASS_COLORS.get(int(c), [0.5, 0.5, 0.5]) for c in cls_s]

    # Подсчёт уникальных классов
    unique_cls = np.unique(pred_cls)
    cls_counts = {}
    for c in unique_cls:
        cls_counts[int(c)] = int((pred_cls == c).sum())

    stats = {
        "total":        int(N),
        "displayed":    int(len(keep)),
        "ground":       int((pred_cls == 2).sum()),
        "vegetation":   int((pred_cls == 4).sum()),
        "powerlines":   int((pred_cls == 14).sum()),
        "unclassified": int((pred_cls == 1).sum()),
        "all_classes":  cls_counts,
    }
    return JSONResponse({
        "x":      x_r[keep].tolist(),
        "y":      y_r[keep].tolist(),
        "z":      z_r[keep].tolist(),
        "colors": col_s,
        "labels": cls_s,
        "stats":  stats,
        "time_s": elapsed,
    })


@app.post("/view")
async def view_classified(file: UploadFile = File(...)):
    """
    Просмотр уже классифицированного LAS-файла.
    Читает атрибут classification напрямую — нейросеть не нужна.
    Поддерживает любые классы: 0,1,2,4,6,14 и т.д.
    """
    if not file.filename.lower().endswith(".las"):
        raise HTTPException(400, "Нужен .las файл")

    raw = await file.read()
    log.info(f"[view] Файл: {file.filename}  {len(raw)/1e6:.1f} MB")

    try:
        las = laspy.read(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(400, f"Ошибка чтения LAS: {e}")

    N       = las.header.point_count
    x_r     = np.array(las.x,              dtype=np.float32)
    y_r     = np.array(las.y,              dtype=np.float32)
    z_r     = np.array(las.z,              dtype=np.float32)
    cls_raw = np.array(las.classification, dtype=np.int32)

    unique, counts = np.unique(cls_raw, return_counts=True)
    log.info(f"[view] {N:,} точек | классы: { {int(c): int(n) for c,n in zip(unique, counts)} }")

    # Строим цвета для любых классов, не только известных нейросети
    # Расширенная палитра LAS-классов
    EXTENDED_COLORS = {
        **CLASS_COLORS,
        0:  [0.40, 0.40, 0.40],   # never classified
        3:  [0.13, 0.70, 0.13],   # low vegetation
        5:  [0.07, 0.45, 0.07],   # high vegetation
        6:  [0.80, 0.60, 0.20],   # building
        7:  [0.90, 0.10, 0.10],   # noise
        9:  [0.20, 0.50, 0.90],   # water
        10: [0.60, 0.30, 0.10],   # rail
        11: [0.70, 0.70, 0.70],   # road surface
        15: [0.95, 0.80, 0.10],   # transmission tower
        16: [0.80, 0.20, 0.80],   # overhead structure
    }

    col_s = [EXTENDED_COLORS.get(int(c), [0.55, 0.55, 0.55]) for c in cls_raw]
    col_arr = np.array(col_s, dtype=np.float32)

    # Прореживание — для просмотра провода тоже приоритетны
    pw_mask   = cls_raw == 14
    pw_idx    = np.where(pw_mask)[0]
    other_idx = np.where(~pw_mask)[0]
    n_other   = max(0, MAX_SEND_PTS - len(pw_idx))
    if len(other_idx) > n_other:
        other_idx = np.random.default_rng(0).choice(other_idx, n_other, replace=False)
    keep = np.concatenate([pw_idx, other_idx])

    cls_counts = {int(c): int(n) for c, n in zip(unique, counts)}

    stats = {
        "total":        int(N),
        "displayed":    int(len(keep)),
        "ground":       int((cls_raw == 2).sum()),
        "vegetation":   int((cls_raw == 4).sum()),
        "powerlines":   int((cls_raw == 14).sum()),
        "unclassified": int(np.isin(cls_raw, [0, 1]).sum()),
        "all_classes":  cls_counts,
    }
    log.info(f"[view] Отправляем {len(keep):,} точек")

    return JSONResponse({
        "x":      x_r[keep].tolist(),
        "y":      y_r[keep].tolist(),
        "z":      z_r[keep].tolist(),
        "colors": col_arr[keep].tolist(),
        "labels": cls_raw[keep].tolist(),
        "stats":  stats,
        "time_s": 0.0,
        "mode":   "view",
    })



# ─────────────────────────────────────────────────────────
# ГЕОМЕТРИЧЕСКАЯ ФИЛЬТРАЦИЯ ПРОВОДОВ
# ─────────────────────────────────────────────────────────
def pca_linearity(pts: np.ndarray):
    """
    Линейность кластера через PCA: (λ1-λ2)/сумма → 0..1.
    1.0 = идеальная прямая, 0.0 = шар.
    """
    if len(pts) < 3:
        return 0.0, 1.0
    cov    = np.cov(pts.T)
    if cov.ndim < 2:
        return 0.0, 1.0
    eigvals = np.linalg.eigvalsh(cov)          # возрастающий порядок
    eigvals = np.sort(eigvals)[::-1]            # убывающий
    total   = eigvals.sum()
    if total < 1e-10:
        return 0.0, 1.0
    norm = eigvals / total
    linearity = float(norm[0] - norm[1])

    # Вытянутость: размах по главной оси / размах по второй
    from numpy.linalg import eigh
    _, vecs = eigh(cov)
    vecs = vecs[:, ::-1]                        # убывающий порядок
    proj = pts @ vecs
    ranges = proj.max(axis=0) - proj.min(axis=0)
    elongation = float(ranges[0] / (ranges[1] + 1e-6))

    return linearity, elongation


@app.post("/filter")
async def filter_powerlines(
    file:        UploadFile = File(...),
    linearity:   float = 0.80,
    elongation:  float = 3.0,
    min_points:  int   = 10,
    eps:         float = 0.8,
    min_samples: int   = 3,
    fallback_cls: int  = 1,
):
    """
    Геометрическая фильтрация ложных срабатываний класса 14 (провода).

    Алгоритм:
      1. Читаем LAS, берём точки с cls == 14
      2. DBSCAN кластеризация
      3. Каждый кластер проверяем на линейность (PCA) и вытянутость
      4. Нелинейные кластеры → fallback_cls (по умолчанию 1 = unclassified)
      5. Возвращаем весь файл с исправленной классификацией для отображения

    Параметры:
      linearity   — порог линейности 0..1 (выше = строже, рекомендуется 0.75-0.90)
      elongation  — порог вытянутости (провод обычно >5, рекомендуется 3-6)
      min_points  — кластеры меньше этого → ложные
      eps         — радиус соседства DBSCAN в единицах координат файла (метры)
      min_samples — мин. точек для ядра DBSCAN
      fallback_cls — класс для ложных точек (1=unclassified, 2=ground)
    """
    try:
        from sklearn.cluster import DBSCAN
    except ImportError:
        raise HTTPException(500, "Установите scikit-learn: pip install scikit-learn")

    if not file.filename.lower().endswith(".las"):
        raise HTTPException(400, "Нужен .las файл")

    raw = await file.read()
    log.info(f"[filter] Файл: {file.filename}  {len(raw)/1e6:.1f} MB")

    try:
        las = laspy.read(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(400, f"Ошибка чтения LAS: {e}")

    N       = las.header.point_count
    x_r     = np.array(las.x,              dtype=np.float64)
    y_r     = np.array(las.y,              dtype=np.float64)
    z_r     = np.array(las.z,              dtype=np.float64)
    cls_raw = np.array(las.classification, dtype=np.int32)

    pw_mask = cls_raw == 14
    n_pw    = int(pw_mask.sum())
    log.info(f"[filter] Точек класса 14: {n_pw:,} из {N:,}")

    if n_pw == 0:
        log.info("[filter] Проводов нет — фильтрация не нужна")
        # Всё равно возвращаем файл для просмотра
        result_cls = cls_raw.copy()
        removed = 0
        cluster_report = []
    else:
        pw_xyz = np.stack([x_r[pw_mask], y_r[pw_mask], z_r[pw_mask]], axis=1)

        log.info(f"[filter] DBSCAN (eps={eps}, min_samples={min_samples})...")
        t0 = time.time()
        db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = db.fit_predict(pw_xyz)
        log.info(f"[filter] DBSCAN за {time.time()-t0:.1f}с")

        pw_indices   = np.where(pw_mask)[0]   # глобальные индексы
        result_cls   = cls_raw.copy()
        cluster_report = []
        removed      = 0

        # Шумовые точки DBSCAN (-1) → сразу ложные
        noise_global = pw_indices[labels == -1]
        result_cls[noise_global] = fallback_cls
        removed += len(noise_global)

        n_clusters = int(labels.max()) + 1
        log.info(f"[filter] Кластеров: {n_clusters}, шум: {len(noise_global):,}")

        for cid in range(n_clusters):
            cmask    = labels == cid
            n_pts    = int(cmask.sum())
            c_pts    = pw_xyz[cmask]
            c_global = pw_indices[cmask]

            lin, elong = pca_linearity(c_pts)

            is_false = (
                n_pts    < min_points   or
                lin      < linearity    or
                elong    < elongation
            )

            reason = []
            if n_pts < min_points:  reason.append(f"мало точек ({n_pts}<{min_points})")
            if lin   < linearity:   reason.append(f"не линейный ({lin:.2f}<{linearity})")
            if elong < elongation:  reason.append(f"не вытянутый ({elong:.1f}<{elongation})")

            cluster_report.append({
                "id":          cid,
                "n_points":    n_pts,
                "linearity":   round(lin, 3),
                "elongation":  round(elong, 2),
                "is_false":    bool(is_false),
                "reason":      ", ".join(reason) if is_false else "провод",
            })

            if is_false:
                result_cls[c_global] = fallback_cls
                removed += n_pts

        cluster_report.sort(key=lambda x: -x["n_points"])

    # ── Статистика до/после ───────────────────────────────
    stats_before = {
        int(c): int((cls_raw    == c).sum())
        for c in np.unique(cls_raw)
    }
    stats_after = {
        int(c): int((result_cls == c).sum())
        for c in np.unique(result_cls)
    }

    log.info(f"[filter] Убрано ложных проводов: {removed:,} | осталось проводов: {int((result_cls==14).sum()):,}")

    # ── Прореживание для отображения ─────────────────────
    EXTENDED_COLORS = {
        0:'#404040', 1:'#888888', 2:'#8B7355', 3:'#3cb371',
        4:'#2E8B57', 5:'#1a6b3a', 6:'#c8a050', 7:'#e74c3c',
        9:'#3498db', 10:'#777777', 11:'#aaaaaa',
        14:'#FF3300', 15:'#e8c030', 16:'#cc55cc',
    }
    def hex_to_rgb(h):
        h = h.lstrip('#')
        return [int(h[i:i+2], 16)/255 for i in (0, 2, 4)]

    pw_after  = np.where(result_cls == 14)[0]
    oth_after = np.where(result_cls != 14)[0]
    n_other   = max(0, MAX_SEND_PTS - len(pw_after))
    if len(oth_after) > n_other:
        oth_after = np.random.default_rng(0).choice(oth_after, n_other, replace=False)
    keep = np.concatenate([pw_after, oth_after])

    col_s = [hex_to_rgb(EXTENDED_COLORS.get(int(result_cls[i]), '#888888')) for i in keep]

    return JSONResponse({
        "x":      x_r[keep].tolist(),
        "y":      y_r[keep].tolist(),
        "z":      z_r[keep].tolist(),
        "colors": col_s,
        "labels": result_cls[keep].tolist(),
        "stats": {
            "total":           int(N),
            "displayed":       int(len(keep)),
            "powerlines_before": int(pw_mask.sum()),
            "powerlines_after":  int((result_cls == 14).sum()),
            "removed":           removed,
            "clusters_total":    len(cluster_report),
            "clusters_real":     sum(1 for c in cluster_report if not c["is_false"]),
            "clusters_false":    sum(1 for c in cluster_report if c["is_false"]),
            "before":   stats_before,
            "after":    stats_after,
        },
        "clusters": cluster_report[:50],   # топ-50 кластеров по размеру
        "time_s":  0.0,
        "mode":    "filter",
    })


# ─────────────────────────────────────────────────────────
# ОХРАННЫЕ ЗОНЫ ЛЭП
# ─────────────────────────────────────────────────────────
@app.post("/safety_zones")
async def safety_zones(
    file:     UploadFile = File(...),
    distance: float = 5.0,   # расстояние от провода до плоскости (м)
    min_pts:  int   = 5,     # мин. точек для формирования сегмента
    eps:      float = 1.5,   # DBSCAN eps для кластеризации проводов
):
    """
    Вычисляет ограничительные плоскости вдоль ЛЭП.

    Алгоритм:
      1. Читаем точки класса 14 (провода)
      2. DBSCAN — разбиваем на отдельные пролёты
      3. Для каждого пролёта: PCA → главная ось (направление провода)
      4. Две плоскости по обе стороны на расстоянии distance
         Плоскость перпендикулярна земле и параллельна проводу
      5. Земля = медиана Z точек класса 2

    Возвращает список плоскостей:
      { center:[x,y,z], normal:[nx,ny,nz], up:[ux,uy,uz],
        half_length, half_height, distance }
    """
    try:
        from sklearn.cluster import DBSCAN
    except ImportError:
        raise HTTPException(500, "pip install scikit-learn")

    if not file.filename.lower().endswith(".las"):
        raise HTTPException(400, "Нужен .las файл")

    raw = await file.read()
    try:
        las = laspy.read(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(400, f"Ошибка чтения LAS: {e}")

    x_r = np.array(las.x, dtype=np.float64)
    y_r = np.array(las.y, dtype=np.float64)
    z_r = np.array(las.z, dtype=np.float64)
    cls = np.array(las.classification, dtype=np.int32)

    # Земля — для определения высоты сцены
    ground_mask = cls == 2
    ground_z    = float(np.median(z_r[ground_mask])) if ground_mask.sum() > 0 else float(z_r.min())

    pw_mask = cls == 14
    if pw_mask.sum() < min_pts:
        return JSONResponse({"planes": [], "ground_z": ground_z,
                             "msg": "Нет точек класса 14 (провода)"})

    pw_xyz = np.stack([x_r[pw_mask], y_r[pw_mask], z_r[pw_mask]], axis=1)

    # Кластеризация DBSCAN — каждый кластер = один пролёт провода
    db     = DBSCAN(eps=eps, min_samples=min_pts, n_jobs=-1)
    labels = db.fit_predict(pw_xyz)

    # Центроид всего облака (для нормализации, как в showCloud фронтенда)
    cx = float(x_r.mean()); cy = float(y_r.mean()); cz = float(z_r.mean())

    planes = []
    n_clusters = int(labels.max()) + 1

    for cid in range(n_clusters):
        pts = pw_xyz[labels == cid]
        if len(pts) < min_pts:
            continue

        # PCA — главная ось = направление провода
        centroid = pts.mean(axis=0)
        cov = np.cov((pts - centroid).T)
        if cov.ndim < 2:
            continue
        _, vecs = np.linalg.eigh(cov)
        axis = vecs[:, -1]   # главная ось (вдоль провода)

        # Делаем ось горизонтальной — проецируем на XY
        axis_h = np.array([axis[0], axis[1], 0.0])
        norm_h = np.linalg.norm(axis_h)
        if norm_h < 1e-6:
            continue
        axis_h /= norm_h   # единичный горизонтальный вектор вдоль провода

        # Нормаль к плоскости = перпендикуляр в горизонтальной плоскости
        normal = np.array([-axis_h[1], axis_h[0], 0.0])   # 90° поворот в XY

        # Вертикальный вектор (вверх)
        up = np.array([0.0, 0.0, 1.0])

        # Размеры плоскости
        # half_length = полдлины провода + запас 20%
        proj      = (pts - centroid) @ axis_h[:, np.newaxis] if axis_h.ndim == 1 else (pts - centroid) @ axis_h
        proj      = (pts - centroid) @ axis_h
        half_len  = float(np.abs(proj).max()) * 1.2

        z_min   = float(pts[:, 2].min())
        z_max   = float(pts[:, 2].max())
        z_span  = z_max - z_min
        # Плоскость от земли до верхушки провода + 2 м
        plane_z_bot = ground_z
        plane_z_top = z_max + 2.0
        half_h  = (plane_z_top - plane_z_bot) / 2.0
        plane_z_center = (plane_z_bot + plane_z_top) / 2.0

        # Центр плоскости — над центроидом провода, на нужной высоте
        center = np.array([centroid[0], centroid[1], plane_z_center])

        # Нормализуем координаты как на фронтенде: вычитаем центроид облака
        # Фронтенд делает: px=x-cx, py=z-cz (высота), pz=y-cy
        def to_scene(pt3):
            """Конвертирует абсолютные LAS-координаты в координаты сцены Three.js"""
            return [
                float(pt3[0] - cx),    # X → X
                float(pt3[2] - cz),    # Z → Y (высота)
                float(pt3[1] - cy),    # Y → Z
            ]

        def rot_normal(n3):
            """Поворачивает нормаль под ту же перестановку осей"""
            return [float(n3[0]), float(n3[2]), float(n3[1])]

        planes.append({
            "id":          cid,
            "n_points":    int(len(pts)),
            "center":      to_scene(center),
            "normal":      rot_normal(normal),       # вектор нормали
            "axis":        rot_normal(axis_h),       # вдоль провода (для width)
            "up":          [0.0, 1.0, 0.0],          # вверх в Three.js = Y
            "half_length": float(half_len),
            "half_height": float(half_h),
            "distance":    float(distance),
            # Две плоскости — смещённые на ±distance по нормали
            "center_pos":  to_scene(center + normal * distance),
            "center_neg":  to_scene(center - normal * distance),
            # Для отладки
            "wire_z_min": float(z_min - cz),
            "wire_z_max": float(z_max - cz),
        })

    log.info(f"[safety_zones] пролётов: {n_clusters}, плоскостей: {len(planes)*2}")

    return JSONResponse({
        "planes":   planes,
        "ground_z": float(ground_z - cz),
        "n_wires":  int(pw_mask.sum()),
        "distance": float(distance),
    })


from pydantic import BaseModel
from typing import List

class PointCloudData(BaseModel):
    """Облако точек с метками — передаётся с фронтенда после фильтрации."""
    x:      List[float]
    y:      List[float]
    z:      List[float]
    labels: List[int]


@app.post("/safety_zones_from_labels")
async def safety_zones_from_labels(
    body:     PointCloudData,
    distance: float = 5.0,
    min_pts:  int   = 5,
    eps:      float = 1.5,
):
    """Охранные зоны по уже отфильтрованному облаку (без чтения файла)."""
    try:
        from sklearn.cluster import DBSCAN
    except ImportError:
        raise HTTPException(500, "pip install scikit-learn")

    x_r = np.array(body.x,      dtype=np.float64)
    y_r = np.array(body.y,      dtype=np.float64)
    z_r = np.array(body.z,      dtype=np.float64)
    cls = np.array(body.labels, dtype=np.int32)

    N = len(x_r)
    if N == 0:
        raise HTTPException(400, "Пустое облако точек")

    cx = float(x_r.mean()); cy = float(y_r.mean()); cz = float(z_r.mean())
    ground_mask = cls == 2
    ground_z    = float(np.median(z_r[ground_mask])) if ground_mask.sum() > 0 else float(z_r.min())
    pw_mask = cls == 14
    n_pw = int(pw_mask.sum())
    log.info(f"[zones_from_labels] N={N}, pw={n_pw}")

    if n_pw < min_pts:
        return JSONResponse({"planes": [], "ground_z": float(ground_z - cz),
                             "n_wires": n_pw, "distance": distance,
                             "msg": "Мало точек класса 14"})

    pw_xyz = np.stack([x_r[pw_mask], y_r[pw_mask], z_r[pw_mask]], axis=1)
    db     = DBSCAN(eps=eps, min_samples=min_pts, n_jobs=-1)
    labels = db.fit_predict(pw_xyz)

    def to_scene(pt3):
        return [float(pt3[0]-cx), float(pt3[2]-cz), float(pt3[1]-cy)]
    def rot_normal(n3):
        return [float(n3[0]), float(n3[2]), float(n3[1])]

    planes = []
    n_clusters = int(labels.max()) + 1
    for cid in range(n_clusters):
        pts = pw_xyz[labels == cid]
        if len(pts) < min_pts: continue
        centroid = pts.mean(axis=0)
        cov = np.cov((pts - centroid).T)
        if cov.ndim < 2: continue
        _, vecs = np.linalg.eigh(cov)
        axis = vecs[:, -1]
        axis_h = np.array([axis[0], axis[1], 0.0])
        norm_h = np.linalg.norm(axis_h)
        if norm_h < 1e-6: continue
        axis_h /= norm_h
        normal   = np.array([-axis_h[1], axis_h[0], 0.0])
        proj     = (pts - centroid) @ axis_h
        half_len = float(np.abs(proj).max()) * 1.2
        z_min = float(pts[:,2].min()); z_max = float(pts[:,2].max())
        plane_z_bot = ground_z; plane_z_top = z_max + 2.0
        half_h = (plane_z_top - plane_z_bot) / 2.0
        plane_z_center = (plane_z_bot + plane_z_top) / 2.0
        center = np.array([centroid[0], centroid[1], plane_z_center])
        planes.append({
            "id": cid, "n_points": int(len(pts)),
            "center":      to_scene(center),
            "normal":      rot_normal(normal),
            "axis":        rot_normal(axis_h),
            "up":          [0.0, 1.0, 0.0],
            "half_length": float(half_len),
            "half_height": float(half_h),
            "distance":    float(distance),
            "center_pos":  to_scene(center + normal * distance),
            "center_neg":  to_scene(center - normal * distance),
            "wire_z_min":  float(z_min - cz),
            "wire_z_max":  float(z_max - cz),
        })

    log.info(f"[zones_from_labels] пролётов: {n_clusters}, плоскостей: {len(planes)*2}")
    return JSONResponse({
        "planes": planes, "ground_z": float(ground_z-cz),
        "n_wires": n_pw, "distance": distance,
    })


@app.post("/vegetation_projection")
async def vegetation_projection(
    body:        PointCloudData,
    wire_height: float = None,
    distance:    float = 5.0,
):
    """
    Горизонтальная проекция растительности внутри коридора охранной зоны.

    Плоскость строится ОТДЕЛЬНО для каждого пролёта ЛЭП (кластера по оси провода).
    Высота каждого сегмента = минимум Z точек провода в данном пролёте (точка провиса).
    Слайдер на клиенте задаёт смещение вниз от точки провиса: 0 = уровень провода.
    """
    x_r = np.array(body.x,      dtype=np.float64)
    y_r = np.array(body.y,      dtype=np.float64)
    z_r = np.array(body.z,      dtype=np.float64)
    cls = np.array(body.labels, dtype=np.int32)

    N = len(x_r)
    if N == 0:
        raise HTTPException(400, "Пустое облако точек")

    try:
        from sklearn.cluster import DBSCAN as _DBSCAN
    except ImportError:
        raise HTTPException(500, "pip install scikit-learn")

    cx = float(x_r.mean()); cy = float(y_r.mean()); cz = float(z_r.mean())
    pw_mask  = cls == 14
    veg_mask = cls == 4

    ground_mask = cls == 2
    ground_z    = float(np.percentile(z_r[ground_mask], 5)) if ground_mask.sum() > 0 else float(z_r.min())

    if pw_mask.sum() == 0:
        raise HTTPException(400, "Нет точек класса 14 (провода ЛЭП)")

    log.info(f"[veg_proj] veg={veg_mask.sum()}, pw={pw_mask.sum()}")

    # ── Глобальная ось провода (PCA по всем точкам класса 14) ────────────────
    pw_xy   = np.stack([x_r[pw_mask], y_r[pw_mask]], axis=1)
    pw_mean = pw_xy.mean(axis=0)
    pw_xy_c = pw_xy - pw_mean
    cov2    = np.cov(pw_xy_c.T)
    if cov2.ndim == 2:
        _, vecs2 = np.linalg.eigh(cov2)
        wire_axis_xy = vecs2[:, -1]
    else:
        wire_axis_xy = np.array([1.0, 0.0])
    perp_xy = np.array([-wire_axis_xy[1], wire_axis_xy[0]])

    # ── Разбиваем точки проводов на пролёты (DBSCAN вдоль главной оси) ───────
    # Проецируем на ось провода и кластеризуем по этой проекции + Z
    pw_proj_along = pw_xy_c @ wire_axis_xy   # расстояние вдоль оси
    pw_xyz_all    = np.stack([x_r[pw_mask], y_r[pw_mask], z_r[pw_mask]], axis=1)

    # eps выбирается адаптивно: ~10% от длины трассы, но не меньше 5м и не больше 100м
    span_along = float(pw_proj_along.max() - pw_proj_along.min()) if len(pw_proj_along) > 1 else 50.0
    eps_span   = float(np.clip(span_along * 0.10, 5.0, 100.0))

    db = _DBSCAN(eps=eps_span, min_samples=5, n_jobs=-1)
    # Кластеризуем только вдоль оси провода (1D) — это разбивает на пролёты
    span_col   = pw_proj_along.reshape(-1, 1)
    span_labels = db.fit_predict(span_col)

    n_spans = int(span_labels.max()) + 1 if span_labels.max() >= 0 else 0
    log.info(f"[veg_proj] пролётов: {n_spans}, eps_span={eps_span:.1f}м")

    # Центр всех проводов для определения коридора
    wire_cx = float(x_r[pw_mask].mean())
    wire_cy = float(y_r[pw_mask].mean())

    # ── Маска коридора по поперечному расстоянию ─────────────────────────────
    if veg_mask.sum() > 0:
        veg_indices = np.where(veg_mask)[0]
        vax = x_r[veg_mask] - wire_cx
        vay = y_r[veg_mask] - wire_cy
        dist_perp_veg = np.abs(vax * perp_xy[0] + vay * perp_xy[1])
        in_corr      = dist_perp_veg <= distance
        corr_global  = veg_indices[in_corr]
    else:
        corr_global = np.array([], dtype=np.int64)

    # ── Строим сегменты плоскости для каждого пролёта ────────────────────────
    spans = []
    if n_spans == 0:
        # Все провода — один пролёт
        span_ids = [np.arange(pw_mask.sum())]
    else:
        span_ids = [np.where(span_labels == sid)[0] for sid in range(n_spans)]

    for sid, s_idx in enumerate(span_ids):
        if len(s_idx) < 3:
            continue
        s_z      = z_r[pw_mask][s_idx]
        s_xy     = pw_xy[s_idx]
        s_cx     = float(s_xy[:, 0].mean())
        s_cy     = float(s_xy[:, 1].mean())

        # Точка провиса = минимум Z в пролёте (5-й перцентиль для устойчивости)
        sag_z    = float(np.percentile(s_z, 5))

        # Длина пролёта вдоль оси
        s_proj   = (s_xy - pw_mean) @ wire_axis_xy
        s_half_l = float((s_proj.max() - s_proj.min()) / 2.0) * 1.05

        # Высота над землёй у точки провиса
        sag_above_ground = float(sag_z - ground_z)

        spans.append({
            "span_id":           sid,
            "center_scene":      [float(s_cx - cx), float(sag_z - cz), float(s_cy - cy)],
            "half_len":          s_half_l,
            "half_width":        distance,
            "sag_z_scene":       float(sag_z - cz),     # Y плоскости в scene
            "sag_above_ground":  round(sag_above_ground, 2),  # высота над землёй в метрах LAS
            "ground_z_scene":    float(ground_z - cz),
        })

    # ── Точки вегетации в коридоре (все, слайдер фильтрует на клиенте) ───────
    MAX_PROJ = 80000
    proj_points = []
    idx_sample = corr_global
    if len(idx_sample) > MAX_PROJ:
        idx_sample = np.random.default_rng(0).choice(idx_sample, MAX_PROJ, replace=False)
    for i in idx_sample:
        proj_points.append([
            float(x_r[i] - cx),
            float(z_r[i] - cz),
            float(y_r[i] - cy),
        ])

    # Зарастание: точки в коридоре выше минимальной точки провиса (наихудший пролёт)
    min_sag_z = min((s["sag_z_scene"] for s in spans), default=(float(z_r[pw_mask].min() - cz) if pw_mask.sum() > 0 else 0))
    above_wire  = [p for p in proj_points if p[1] >= min_sag_z]
    overgrowth_pct = round(len(above_wire) / len(proj_points) * 100, 1) if proj_points else 0.0

    log.info(f"[veg_proj] spans={len(spans)}, corridor_pts={len(proj_points)}, overgrowth={overgrowth_pct}%")

    return JSONResponse({
        "proj_points":      proj_points,
        "spans":            spans,               # список пролётов с их геометрией
        "wire_axis_scene":  [float(wire_axis_xy[0]), 0.0, float(wire_axis_xy[1])],
        "plane_half_width": distance,
        "n_veg_total":      int(veg_mask.sum()),
        "overgrowth_pct":   overgrowth_pct,
        "ground_z_scene":   float(ground_z - cz),
        # Для обратной совместимости слайдера — самая низкая точка провиса
        "wire_height_scene": min_sag_z,
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
