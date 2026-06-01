"""
zoo.py — зоопарк DL-архитектур из CP6 (для загрузки весов коллеги и переоценки).

Классы перенесены из checkpoint-6-dl-models.ipynb без изменения логики:
- GRUClassifier, BiLSTMAttnClassifier, TCNClassifier, TransformerClassifier,
  CNNLSTMClassifier — зоопарк классификаторов направления (вход (B,T,F) -> логиты (B,2));
- LSTMEncoder, EncoderClassifier — общий энкодер + голова (SSL masked / contrastive);
- MaskedReconstructionModel, ContrastiveModel — SSL-предобучение;
- PolicyNet — differentiable-Sharpe политика (выход pos in [0,1]);
- QNetwork — DQN-агент (выход Q-значений для действий {flat, long}).

reconstruct_model() собирает нужную архитектуру по class_name из метаданных .pt,
чтобы загрузить state_dict обученной модели.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common import SimpleLSTM

HIDDEN_SIZE = 64
DROPOUT = 0.2


# =============================================================================
# Зоопарк классификаторов (вход (B,T,F) -> логиты (B,2))
# =============================================================================


class GRUClassifier(nn.Module):
    """GRU-классификатор: (B,T,F) -> логиты (B,2)."""

    def __init__(self, input_size, hidden=64, num_layers=1, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden, num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 2)

    def forward(self, x):
        out, _ = self.gru(x)
        h = out[:, -1, :]
        return self.fc(self.drop(h))


class BiLSTMAttnClassifier(nn.Module):
    """Двунаправленный LSTM + attention-pooling по времени -> логиты (B,2)."""

    def __init__(self, input_size, hidden=64, num_layers=1, dropout=0.2):
        super().__init__()
        self.bilstm = nn.LSTM(input_size, hidden, num_layers, batch_first=True,
                              bidirectional=True,
                              dropout=dropout if num_layers > 1 else 0.0)
        self.attn_w = nn.Linear(hidden * 2, 1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, 2)

    def forward(self, x):
        out, _ = self.bilstm(x)
        scores = self.attn_w(out).squeeze(-1)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        pooled = (out * weights).sum(dim=1)
        return self.fc(self.drop(pooled))


class _TCNBlock(nn.Module):
    """Резидуальный блок TCN: dilated causal Conv1d + residual."""

    def __init__(self, n_channels, kernel_size, dilation, dropout):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(n_channels, n_channels, kernel_size, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(n_channels, n_channels, kernel_size, padding=pad, dilation=dilation)
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self._pad = pad

    def _causal_crop(self, x):
        if self._pad > 0:
            x = x[:, :, :-self._pad]
        return x

    def forward(self, x):
        residual = x
        out = self.relu(self._causal_crop(self.conv1(x)))
        out = self.drop(out)
        out = self.relu(self._causal_crop(self.conv2(out)))
        out = self.drop(out)
        return self.relu(out + residual)


class TCNClassifier(nn.Module):
    """Temporal Convolutional Network -> логиты (B,2)."""

    def __init__(self, input_size, hidden=64, num_layers=2, dropout=0.2, kernel_size=3):
        super().__init__()
        self.proj = nn.Conv1d(input_size, hidden, kernel_size=1)
        blocks = [_TCNBlock(hidden, kernel_size, 2 ** i, dropout) for i in range(num_layers)]
        self.tcn = nn.Sequential(*blocks)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 2)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.proj(x)
        x = self.tcn(x)
        x = x.mean(dim=2)
        return self.fc(self.drop(x))


class TransformerClassifier(nn.Module):
    """Transformer: projection + pos-encoding + TransformerEncoder + mean-pool -> (B,2)."""

    def __init__(self, input_size, hidden=64, num_layers=2, dropout=0.2, nhead=4):
        super().__init__()
        while hidden % nhead != 0 and nhead > 1:
            nhead -= 1
        self.proj = nn.Linear(input_size, hidden)
        self.dropout_emb = nn.Dropout(dropout)
        enc_layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=nhead,
                                               dim_feedforward=hidden * 4,
                                               dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 2)
        self._hidden = hidden

    def _pos_encoding(self, T, device):
        pe = torch.zeros(T, self._hidden, device=device)
        pos = torch.arange(0, T, device=device).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, self._hidden, 2, device=device).float()
                        * (-math.log(10000.0) / self._hidden))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:self._hidden // 2])
        return pe.unsqueeze(0)

    def forward(self, x):
        B, T, _ = x.shape
        h = self.proj(x)
        h = self.dropout_emb(h + self._pos_encoding(T, x.device))
        h = self.transformer(h)
        h = h.mean(dim=1)
        return self.fc(self.drop(h))


class CNNLSTMClassifier(nn.Module):
    """CNN feature extractor -> LSTM -> голова. (B,T,F) -> (B,2)."""

    def __init__(self, input_size, hidden=64, num_layers=1, dropout=0.2,
                 cnn_channels=32, kernel_size=3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, cnn_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(cnn_channels, hidden, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 2)

    def forward(self, x):
        c = self.cnn(x.transpose(1, 2))
        c = c.transpose(1, 2)
        out, _ = self.lstm(c)
        h = out[:, -1, :]
        return self.fc(self.drop(h))


# =============================================================================
# Энкодер + голова (SSL masked / contrastive) и SSL-модели
# =============================================================================


class LSTMEncoder(nn.Module):
    """Общий энкодер: LSTM -> представления по всем шагам (B,T,H)."""

    def __init__(self, input_size, hidden_size=HIDDEN_SIZE, num_layers=1, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.hidden_size = hidden_size

    def forward(self, x):
        out, _ = self.lstm(x)
        return out


class MaskedReconstructionModel(nn.Module):
    """SSL masked reconstruction: encoder + линейный декодер."""

    def __init__(self, input_size, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.encoder = LSTMEncoder(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class EncoderClassifier(nn.Module):
    """Голова поверх энкодера (представление последнего шага) -> логиты (B,2)."""

    def __init__(self, encoder, hidden_size=HIDDEN_SIZE, dropout=DROPOUT, freeze_encoder=False):
        super().__init__()
        self.encoder = encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        h = self.encoder(x)[:, -1, :]
        return self.fc(self.dropout(h))


class ContrastiveModel(nn.Module):
    """Contrastive (NT-Xent): encoder + projector -> нормализованный эмбеддинг."""

    def __init__(self, input_size, hidden_size=HIDDEN_SIZE, proj_dim=64):
        super().__init__()
        self.encoder = LSTMEncoder(input_size, hidden_size)
        self.projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, proj_dim)
        )

    def forward(self, x):
        h = self.encoder(x)[:, -1, :]
        z = self.projector(h)
        return F.normalize(z, dim=1)


# =============================================================================
# Политики (выход — не логиты классов)
# =============================================================================


class PolicyNet(nn.Module):
    """LSTM -> сила позиции pos in [0,1] (differentiable Sharpe)."""

    def __init__(self, input_size, hidden_size=HIDDEN_SIZE, dropout=DROPOUT):
        super().__init__()
        self.encoder = LSTMEncoder(input_size, hidden_size)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 1))

    def forward(self, x):
        h = self.encoder(x)[:, -1, :]
        return torch.sigmoid(self.head(h)).squeeze(-1)


class QNetwork(nn.Module):
    """Q(s,a): окно -> Q-значения для действий {flat, long}."""

    def __init__(self, input_size, hidden_size=HIDDEN_SIZE, n_actions=2, dropout=DROPOUT):
        super().__init__()
        self.encoder = LSTMEncoder(input_size, hidden_size)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, n_actions))

    def forward(self, x):
        return self.head(self.encoder(x)[:, -1, :])


# =============================================================================
# Фабрики
# =============================================================================


def build_model(name, n_features_in, **hp):
    """Фабрика по короткому имени (для обучения новых моделей)."""
    hidden = int(hp.get("hidden", 64))
    num_layers = int(hp.get("num_layers", 1))
    dropout = float(hp.get("dropout", 0.2))
    table = {
        "gru": lambda: GRUClassifier(n_features_in, hidden=hidden, num_layers=num_layers, dropout=dropout),
        "bilstm_attn": lambda: BiLSTMAttnClassifier(n_features_in, hidden=hidden, num_layers=num_layers, dropout=dropout),
        "tcn": lambda: TCNClassifier(n_features_in, hidden=hidden, num_layers=num_layers, dropout=dropout),
        "transformer": lambda: TransformerClassifier(n_features_in, hidden=hidden, num_layers=num_layers, dropout=dropout),
        "cnn_lstm": lambda: CNNLSTMClassifier(n_features_in, hidden=hidden, num_layers=num_layers, dropout=dropout),
        "simple_lstm": lambda: SimpleLSTM(input_size=n_features_in, hidden_size=hidden, num_layers=num_layers, dropout=dropout),
    }
    if name not in table:
        raise ValueError(f"Неизвестная архитектура: {name!r}. Доступны: {list(table)}")
    return table[name]()


def reconstruct_model(class_name, input_size, hp=None):
    """Собирает модель по class_name из метаданных .pt (для загрузки state_dict).

    Возвращает (model, kind), где kind:
        'logits'  — классификатор, выход (B,2) логиты (softmax -> p1);
        'policy'  — PolicyNet, выход pos in [0,1];
        'qvalues' — QNetwork, выход (B,2) Q, argmax -> действие.
    """
    hp = hp or {}
    hidden = int(hp.get("hidden", hp.get("hidden_size", 64)))
    num_layers = int(hp.get("num_layers", 1))
    dropout = float(hp.get("dropout", 0.2))

    if class_name == "SimpleLSTM":
        return SimpleLSTM(input_size, hidden_size=hidden, num_layers=num_layers, dropout=dropout), "logits"
    if class_name == "GRUClassifier":
        return GRUClassifier(input_size, hidden=hidden, num_layers=num_layers, dropout=dropout), "logits"
    if class_name == "BiLSTMAttnClassifier":
        return BiLSTMAttnClassifier(input_size, hidden=hidden, num_layers=num_layers, dropout=dropout), "logits"
    if class_name == "TCNClassifier":
        return TCNClassifier(input_size, hidden=hidden, num_layers=num_layers, dropout=dropout), "logits"
    if class_name == "TransformerClassifier":
        return TransformerClassifier(input_size, hidden=hidden, num_layers=num_layers, dropout=dropout), "logits"
    if class_name == "CNNLSTMClassifier":
        return CNNLSTMClassifier(input_size, hidden=hidden, num_layers=num_layers, dropout=dropout), "logits"
    if class_name == "EncoderClassifier":
        enc = LSTMEncoder(input_size, hidden_size=hidden, num_layers=num_layers, dropout=dropout)
        return EncoderClassifier(enc, hidden_size=hidden, dropout=dropout), "logits"
    if class_name == "PolicyNet":
        return PolicyNet(input_size, hidden_size=hidden, dropout=dropout), "policy"
    if class_name == "QNetwork":
        return QNetwork(input_size, hidden_size=hidden, dropout=dropout), "qvalues"
    raise ValueError(f"Неизвестный class_name для реконструкции: {class_name!r}")
