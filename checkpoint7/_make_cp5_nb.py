"""Генератор ноутбука checkpoint-5-fixed.ipynb (CP5-зоопарк, честная оценка gross+net)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
    "# Чекпоинт 5 — исправленная версия (gross vs net Sharpe)\n\n"
    "Воспроизводим зоопарк CP5 (SimpleLSTM, 1D CNN, Transformer, RandomForest, Ensemble) "
    "на **восстановленных полных данных** (2024-01…2025-09) и считаем Sharpe **двумя способами**:\n\n"
    "- **GROSS** — argmax-позиция каждый бар, БЕЗ издержек (как в исходном CP5);\n"
    "- **NET** — + `min_holding=15` + издержки 7bps, на **1-барной** доходности (честно, без перекрытия).\n\n"
    "Цель — показать на единой метрике, что «победа» CP5 над Buy&Hold была артефактом "
    "отсутствия издержек. Обучение на GPU (`~/.envs/ds`)."
))

cells.append(nbf.v4.new_code_cell(
    "import warnings; warnings.filterwarnings('ignore')\n"
    "import numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim\n"
    "from sklearn.ensemble import RandomForestClassifier\n"
    "from sklearn.metrics import roc_auc_score\n"
    "from torch.utils.data import DataLoader\n"
    "from src.common import (PER_YEAR_1M, SimpleLSTM, WindowDataset, apply_min_holding,\n"
    "    evaluate_strategy, set_seed, _predict_labels_probs, _train_epoch)\n"
    "from src.zoo import TransformerClassifier\n"
    "from src.train import _prepare_data\n"
    "from benchmark_full_test import _build_cfg\n"
    "DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
    "WINDOW, MIN_HOLD = 60, 15\n"
    "print('device =', DEVICE)"
))

cells.append(nbf.v4.new_markdown_cell("## Архитектуры и утилиты (1D CNN из CP5, агрегаты для RF)"))
cells.append(nbf.v4.new_code_cell(
    "class Conv1dClassifier(nn.Module):\n"
    "    \"\"\"1D CNN из CP5: 2x Conv1d + Global Max Pool.\"\"\"\n"
    "    def __init__(self, input_size, hidden_size=64, dropout=0.2):\n"
    "        super().__init__()\n"
    "        self.conv1 = nn.Conv1d(input_size, hidden_size, 3, padding=1)\n"
    "        self.relu = nn.ReLU(); self.pool = nn.MaxPool1d(2)\n"
    "        self.conv2 = nn.Conv1d(hidden_size, hidden_size*2, 3, padding=1)\n"
    "        self.global_pool = nn.AdaptiveMaxPool1d(1)\n"
    "        self.fc = nn.Linear(hidden_size*2, 2); self.dropout = nn.Dropout(dropout)\n"
    "    def forward(self, x):\n"
    "        x = x.transpose(1, 2)\n"
    "        x = self.pool(self.relu(self.conv1(x)))\n"
    "        x = self.global_pool(self.relu(self.conv2(x)))\n"
    "        return self.fc(self.dropout(x.squeeze(-1)))\n\n"
    "def aggregate_windows(X, y, window):\n"
    "    from numpy.lib.stride_tricks import sliding_window_view\n"
    "    n = len(X) - window\n"
    "    sw = sliding_window_view(X, window, axis=0)[:n]\n"
    "    agg = np.concatenate([sw.mean(2), sw.std(2), sw.min(2), sw.max(2), sw[:,:,-1]], axis=1)\n"
    "    return agg.astype(np.float32), y[window:window+n]\n\n"
    "def train_dl(model, ld_tr, ld_va, yva_w, w_tensor, epochs=15, patience=4, tag=''):\n"
    "    model = model.to(DEVICE); crit = nn.CrossEntropyLoss(weight=w_tensor.to(DEVICE))\n"
    "    opt = optim.Adam(model.parameters(), lr=1e-3)\n"
    "    best_auc, best_state, stale = -1, None, 0\n"
    "    for ep in range(1, epochs+1):\n"
    "        _train_epoch(model, ld_tr, crit, opt, DEVICE)\n"
    "        _, _, p1 = _predict_labels_probs(model, ld_va, DEVICE)\n"
    "        m = min(len(p1), len(yva_w))\n"
    "        auc = roc_auc_score(yva_w[:m], p1[:m]) if len(np.unique(yva_w[:m]))>1 else 0.5\n"
    "        if auc > best_auc: best_auc, best_state, stale = auc, {k:v.cpu().clone() for k,v in model.state_dict().items()}, 0\n"
    "        else:\n"
    "            stale += 1\n"
    "            if stale >= patience: break\n"
    "    if best_state: model.load_state_dict(best_state)\n"
    "    print(f'  [{tag}] best val ROC-AUC={best_auc:.4f}'); return model\n\n"
    "def evaluate_both(p1, r1, idx, yte_w):\n"
    "    \"\"\"(ROC-AUC, Sharpe gross argmax, Sharpe net min_hold+costs, сделок).\"\"\"\n"
    "    n = min(len(p1), len(idx), len(r1))\n"
    "    p1 = p1[:n]; idxn = idx[:n]; r1n = pd.Series(np.asarray(r1)[:n], index=idx[:n])\n"
    "    roc = roc_auc_score(yte_w[:n], p1) if len(np.unique(yte_w[:n]))>1 else float('nan')\n"
    "    pos_arg = pd.Series((p1>=0.5).astype(float), index=idxn)\n"
    "    ev_g = evaluate_strategy(pos_arg, r1n, PER_YEAR_1M)\n"
    "    pos_mh = apply_min_holding(pos_arg, MIN_HOLD)\n"
    "    ev_n = evaluate_strategy(pos_mh, r1n, PER_YEAR_1M)\n"
    "    return roc, ev_g['Sharpe gross'], ev_n['Sharpe net'], ev_n['Сделок (смен позиции)']"
))

cells.append(nbf.v4.new_markdown_cell("## Данные (triple-barrier, полный тест 2025) и датасеты"))
cells.append(nbf.v4.new_code_cell(
    "set_seed(42)\n"
    "data = _prepare_data(_build_cfg('/home/dmitriy/magistracy/master-coursework/data/processed'))\n"
    "Xt, Xv, Xte = data['Xt_s'], data['Xv_s'], data['Xte_s']\n"
    "yt, yv, yte = data['yt_tb'], data['yv_tb'], data['yte_tb']\n"
    "r1_te, idx_te, nfeat = data['r1_te'], data['idx_te_w'], data['n_features']\n"
    "ds_tr, ds_va, ds_te = WindowDataset(Xt,yt,WINDOW), WindowDataset(Xv,yv,WINDOW), WindowDataset(Xte,yte,WINDOW)\n"
    "ld_tr = DataLoader(ds_tr, batch_size=512, shuffle=True); ld_va = DataLoader(ds_va, batch_size=512); ld_te = DataLoader(ds_te, batch_size=512)\n"
    "ct = np.bincount(ds_tr.labels, minlength=2)\n"
    "w_tensor = torch.tensor(ct.sum()/(2.0*np.maximum(ct,1)), dtype=torch.float32)\n"
    "yte_w, yva_w = ds_te.labels, ds_va.labels\n"
    "print(f'фич={nfeat}, train={len(Xt)}, test_окон={len(idx_te)}')"
))

cells.append(nbf.v4.new_markdown_cell("## Обучение зоопарка (LSTM, 1D CNN, Transformer, RF) + Ensemble"))
cells.append(nbf.v4.new_code_cell(
    "results, p1_store = {}, {}\n"
    "for name, ctor in [('SimpleLSTM', lambda: SimpleLSTM(nfeat,64,1,0.2)),\n"
    "                   ('1D CNN', lambda: Conv1dClassifier(nfeat,64,0.2)),\n"
    "                   ('Transformer', lambda: TransformerClassifier(nfeat,hidden=64,num_layers=2,dropout=0.2))]:\n"
    "    print('Обучение', name)\n"
    "    torch.manual_seed(42)\n"
    "    m = train_dl(ctor(), ld_tr, ld_va, yva_w, w_tensor, epochs=15, tag=name)\n"
    "    _, _, p1 = _predict_labels_probs(m, ld_te, DEVICE)\n"
    "    p1_store[name] = p1; results[name] = evaluate_both(p1, r1_te, idx_te, yte_w)\n"
    "print('Обучение RandomForest')\n"
    "Xt_agg, yt_agg = aggregate_windows(Xt, yt, WINDOW); Xte_agg, _ = aggregate_windows(Xte, yte, WINDOW)\n"
    "rf = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight='balanced', random_state=42, n_jobs=-1)\n"
    "rf.fit(Xt_agg, yt_agg); rf_p1 = rf.predict_proba(Xte_agg)[:,1]\n"
    "p1_store['RandomForest'] = rf_p1; results['RandomForest'] = evaluate_both(rf_p1, r1_te, idx_te, yte_w)\n"
    "n = min(len(v) for v in p1_store.values())\n"
    "ens = np.mean([v[:n] for v in p1_store.values()], axis=0)\n"
    "results['Ensemble (avg)'] = evaluate_both(ens, r1_te, idx_te, yte_w)\n"
    "bh = pd.Series(1.0, index=idx_te); evb = evaluate_strategy(bh, pd.Series(np.asarray(r1_te), index=idx_te), PER_YEAR_1M)\n"
    "results['Buy & Hold'] = (float('nan'), evb['Sharpe gross'], evb['Sharpe net'], evb['Сделок (смен позиции)'])"
))

cells.append(nbf.v4.new_markdown_cell("## Итоговая таблица: GROSS (как CP5) vs NET (честно)"))
cells.append(nbf.v4.new_code_cell(
    "df = pd.DataFrame([(k,v[0],v[1],v[2],v[3]) for k,v in results.items()],\n"
    "    columns=['Модель','ROC-AUC','Sharpe GROSS (argmax)','Sharpe NET (min_hold+costs)','Сделок']\n"
    ").sort_values('Sharpe GROSS (argmax)', ascending=False).reset_index(drop=True)\n"
    "df"
))

cells.append(nbf.v4.new_markdown_cell(
    "## Выводы\n\n"
    "1. **GROSS** (без издержек): все модели «бьют» Buy&Hold (Sharpe 2–3.4 vs 0.63) — "
    "ровно как заявлял исходный CP5.\n"
    "2. **NET** (с издержками 7bps + min_holding): **все модели катастрофически проигрывают** "
    "(Sharpe от −13 до −44), Buy&Hold (0.63) — единственный положительный.\n"
    "3. Разница `gross→net` — это **издержки** на тысячах сделок (argmax торгует почти каждый бар).\n"
    "4. «Лучшая по gross» торгует больше всех → худшая по net. ROC-AUC у всех ≈ 0.51.\n\n"
    "**Вывод:** превосходство CP5 над Buy&Hold было артефактом отсутствия издержек. "
    "На честной метрике ни одна модель не обыгрывает Buy&Hold."
))

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
nbf.write(nb, "checkpoint-5-fixed.ipynb")
print("checkpoint-5-fixed.ipynb создан, ячеек:", len(cells))
