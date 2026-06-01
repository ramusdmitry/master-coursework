"""
sub_polars.py — субминутный замер через polars (read+resample) + БЕЗ MAX_TRAIN.

polars scan_csv + group_by_dynamic (streaming) делает тяжёлое чтение/ресемпл сырого
1s дёшево (~3 ГБ вместо ~9 у pandas), затем pandas-фичи на уже маленьком фрейме и
обучение LSTM на ПОЛНЫХ 70% (без обрезки MAX_TRAIN, которая давала 1s всего ~18 дней).
Один актив за процесс (memory-safe). Метод иначе тот же, что full_spectrum.run_one.

    python sub_polars.py --asset BTC --tf 1s --raw-dir /abs/data/raw --out sub_polars_results.csv
    python sub_polars.py --asset BTC                # все 4 субминутных tf
"""
from __future__ import annotations

import argparse
import gc
import os
import resource
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import features_polars as fp
from full_spectrum import SUB
from src.common import SimpleLSTM, apply_min_holding, sharpe_ratio, set_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW = 60
# Прорежение обучающих окон по tf (соседние мелкие окна почти идентичны).
# НЕ кап: полный период виден, режется только избыточность перекрытия.
STRIDE = {"1s": 10, "5s": 3, "15s": 1, "30s": 1}


def ram_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def _windows_gpu(Xg, idx, window):
    """Нарезка окон прямо на GPU: (B,) стартов -> (B, window, F). Без CPU/DataLoader."""
    offs = torch.arange(window, device=Xg.device)
    return Xg[idx[:, None] + offs[None, :]]


def gpu_train(Xtr, ytr, Xva, yva_lab, window, nfeat, wt, epochs=12, patience=4, bs=2048,
              stride=1, amp=True):
    """GPU-резидентное обучение: весь X на GPU, окна на GPU, CPU вне цикла -> ~90% GPU.
    stride: обучаемся на каждом stride-м стартовом окне (соседние 1s-окна почти
    идентичны -> stride=N режет компьют в N раз без потери информации; НЕ кап —
    полный период виден). amp: mixed-precision (fp16 matmul) для скорости."""
    torch.manual_seed(42)
    m = SimpleLSTM(nfeat, 64, 1, 0.2).to(DEVICE)
    crit = nn.CrossEntropyLoss(weight=wt.to(DEVICE)); opt = optim.Adam(m.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler(enabled=amp and DEVICE.type == "cuda")
    Xtr_g = torch.from_numpy(np.ascontiguousarray(Xtr, np.float32)).to(DEVICE)
    ytr_g = torch.from_numpy(ytr.astype(np.int64)).to(DEVICE)
    Xva_g = torch.from_numpy(np.ascontiguousarray(Xva, np.float32)).to(DEVICE)
    ntr, nva = len(Xtr) - window, len(Xva) - window
    base = torch.arange(0, ntr, stride, device=DEVICE)   # прорежённые старты окон
    best, st, stale = -1, None, 0
    for _ in range(epochs):
        m.train()
        perm = base[torch.randperm(len(base), device=DEVICE)]
        for s in range(0, len(perm), bs):
            idx = perm[s:s + bs]
            opt.zero_grad()
            with torch.autocast(device_type=DEVICE.type, enabled=scaler.is_enabled()):
                loss = crit(m(_windows_gpu(Xtr_g, idx, window)), ytr_g[idx + window])
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        m.eval(); ps = []
        with torch.no_grad():
            for s in range(0, nva, bs):
                idx = torch.arange(s, min(s + bs, nva), device=DEVICE)
                ps.append(torch.softmax(m(_windows_gpu(Xva_g, idx, window)), 1)[:, 1])
        p1 = torch.cat(ps).cpu().numpy()
        a = roc_auc_score(yva_lab[:len(p1)], p1) if len(np.unique(yva_lab[:len(p1)])) > 1 else 0.5
        if a > best: best, st, stale = a, {k: v.cpu().clone() for k, v in m.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= patience: break
    if st: m.load_state_dict(st)
    del Xtr_g, ytr_g, Xva_g; torch.cuda.empty_cache()
    return m


def gpu_predict(m, Xte, window, bs=2048):
    Xte_g = torch.from_numpy(np.ascontiguousarray(Xte, np.float32)).to(DEVICE)
    nte = len(Xte) - window
    m.eval(); ps = []
    with torch.no_grad():
        for s in range(0, nte, bs):
            idx = torch.arange(s, min(s + bs, nte), device=DEVICE)
            ps.append(torch.softmax(m(_windows_gpu(Xte_g, idx, window)), 1)[:, 1])
    p1 = torch.cat(ps).cpu().numpy()
    del Xte_g; torch.cuda.empty_cache()
    return p1




def run(asset, tf, path):
    freq, bd = SUB[tf]
    per_year = bd * 365
    t0 = time.time()
    # polars: read+resample+features+y_reg одной ленивой цепочкой (без pandas-OOM)
    pdf = fp.compute(path, freq, bd)
    t_rs = time.time() - t0
    idx = pd.DatetimeIndex(pd.to_datetime(pdf["timestamp"].to_numpy(), utc=True))
    X = pdf.select(fp.FEATURE_COLS).to_numpy().astype(np.float32)
    r1 = pdf["y_reg"].to_numpy().astype(float)
    yb = (r1 > 0).astype(np.int64)
    del pdf; gc.collect()
    n = len(idx); a, b = int(n * .70), int(n * .85)
    tr_i, va_i, te_i = np.arange(a), np.arange(a, b), np.arange(b, n)   # БЕЗ MAX_TRAIN
    if len(tr_i) < WINDOW + 200 or len(te_i) < WINDOW + 10:
        return None
    sc = StandardScaler().fit(X[tr_i])
    Xtr, Xva, Xte = sc.transform(X[tr_i]), sc.transform(X[va_i]), sc.transform(X[te_i])
    tr_lab = yb[tr_i][WINDOW:WINDOW + (len(tr_i) - WINDOW)]
    va_lab = yb[va_i][WINDOW:WINDOW + (len(va_i) - WINDOW)]
    te_lab = yb[te_i][WINDOW:WINDOW + (len(te_i) - WINDOW)]
    ct = np.bincount(tr_lab, minlength=2)
    wt = torch.tensor(ct.sum() / (2.0 * np.maximum(ct, 1)), dtype=torch.float32)
    t1 = time.time()
    m = gpu_train(Xtr, yb[tr_i], Xva, va_lab, WINDOW, X.shape[1], wt,
                  stride=STRIDE.get(tf, 1))
    t_tr = time.time() - t1
    p1 = gpu_predict(m, Xte, WINDOW)
    yte_w = te_lab
    k = len(p1)
    te_idx = idx[te_i][WINDOW:WINDOW + k]
    r1_te = pd.Series(r1[te_i][WINDOW:WINDOW + k], index=te_idx)
    roc = roc_auc_score(yte_w[:k], p1[:k]) if len(np.unique(yte_w[:k])) > 1 else float("nan")
    pos = apply_min_holding(pd.Series((p1[:k] >= 0.5).astype(float), index=te_idx), 1)
    turn = pos.diff().abs().fillna(pos.abs())
    sh = sharpe_ratio(pos * r1_te - turn * 1e-4, per_year)
    bh = sharpe_ratio(pd.Series(1.0, index=te_idx) * r1_te, per_year)
    return {"Актив": asset, "TF": tf, "ROC-AUC": round(roc, 4), "Sh maker": round(sh, 2),
            "BH": round(bh, 2), "train_bars": len(tr_i), "test": k,
            "t_resample": round(t_rs, 1), "t_train": round(t_tr, 1), "RAM": round(ram_gb(), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True)
    ap.add_argument("--tf", default="all")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out", default="sub_polars_results.csv")
    args = ap.parse_args()
    set_seed(42)
    path = str(Path(args.raw_dir) / f"{args.asset}_1s.csv")
    if not os.path.exists(path):
        print(f"[skip] нет {path}"); return
    tfs = list(SUB.keys()) if args.tf == "all" else [args.tf]
    rows = []
    for tf in tfs:
        r = run(args.asset, tf, path)
        if r:
            rows.append(r)
            print(f"{args.asset} {tf}: ROC={r['ROC-AUC']} Sh={r['Sh maker']} BH={r['BH']} "
                  f"| train={r['train_bars']:,} test={r['test']:,} "
                  f"| ресемпл={r['t_resample']}с обуч={r['t_train']}с RAM={r['RAM']}ГБ", flush=True)
        gc.collect()
    if rows:
        outp = Path(args.out)
        pd.DataFrame(rows).to_csv(outp, mode="a", header=not outp.exists(), index=False, encoding="utf-8")
        print(f"[OK] {args.asset}: дописано {len(rows)} -> {outp}", flush=True)


if __name__ == "__main__":
    main()
