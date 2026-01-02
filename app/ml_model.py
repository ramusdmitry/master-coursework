import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class SimpleLSTM(nn.Module):
    """Простая LSTM модель для бинарной классификации временных рядов."""
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.2):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 2)  # бинарная классификация
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        dropped = self.dropout(last_output)
        output = self.fc(dropped)
        return output


class MLModelService:
    """Сервис для работы с ML моделью"""
    
    def __init__(self, model_path: str = None, scaler_path: str = None):
        self.model = None
        self.scaler = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = 15  # количество фичей для BTC
        self.window_size = 60  # размер временного окна
        
        if model_path:
            self.load_model(model_path)
        else:
            # Создаем модель по умолчанию (для демонстрации)
            self.model = SimpleLSTM(
                input_size=self.input_size,
                hidden_size=64,
                num_layers=1,
                dropout=0.2
            ).to(self.device)
            logger.warning("Используется модель по умолчанию без весов. Для продакшена загрузите обученную модель.")
        
        if scaler_path:
            self.load_scaler(scaler_path)
        else:
            logger.warning("Scaler не загружен. Используется стандартизация по умолчанию.")
    
    def load_model(self, model_path: str):
        """Загрузка обученной модели"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Используем параметры из checkpoint, если они есть
                input_size = checkpoint.get('input_size', self.input_size)
                hidden_size = checkpoint.get('hidden_size', 64)
                num_layers = checkpoint.get('num_layers', 1)
                dropout = checkpoint.get('dropout', 0.2)
                
                # Обновляем параметры экземпляра
                self.input_size = input_size
                self.model = SimpleLSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout
                ).to(self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # Предполагаем, что это state_dict напрямую
                self.model = SimpleLSTM(
                    input_size=self.input_size,
                    hidden_size=64,
                    num_layers=1,
                    dropout=0.2
                ).to(self.device)
                self.model.load_state_dict(checkpoint)
            
            self.model.eval()
            logger.info(f"Модель загружена из {model_path}")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            raise
    
    def load_scaler(self, scaler_path: str):
        """Загрузка scaler для нормализации данных"""
        try:
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            logger.info(f"Scaler загружен из {scaler_path}")
        except Exception as e:
            logger.error(f"Ошибка загрузки scaler: {e}")
            raise
    
    def predict(self, data: List[List[float]]) -> Tuple[int, Dict[str, float]]:
        """
        Предсказание на основе входных данных
        
        Args:
            data: Список временных окон, каждое окно - список фичей
            
        Returns:
            prediction: Предсказанный класс (0 или 1)
            probabilities: Словарь с вероятностями классов
        """
        try:
            # Проверка размера данных
            if len(data) < self.window_size:
                raise ValueError(f"Недостаточно данных. Требуется минимум {self.window_size} временных точек, получено {len(data)}")
            
            # Берем последние window_size точек
            window_data = data[-self.window_size:]
            
            # Проверка количества фичей
            if len(window_data[0]) != self.input_size:
                raise ValueError(f"Неверное количество фичей. Ожидается {self.input_size}, получено {len(window_data[0])}")
            
            # Преобразование в numpy array
            X = np.array(window_data, dtype=np.float32)
            
            # Нормализация
            if self.scaler is not None:
                X = self.scaler.transform(X)
            else:
                # Простая стандартизация по умолчанию
                X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
            
            # Преобразование в тензор и добавление batch dimension
            X_tensor = torch.FloatTensor(X).unsqueeze(0).to(self.device)  # (1, window_size, features)
            
            # Предсказание
            with torch.no_grad():
                outputs = self.model(X_tensor)
                probs = torch.softmax(outputs, dim=1)
                prediction = torch.argmax(probs, dim=1).item()
                
                probabilities = {
                    "class_0": probs[0][0].item(),
                    "class_1": probs[0][1].item()
                }
            
            return prediction, probabilities
            
        except Exception as e:
            logger.error(f"Ошибка при предсказании: {e}")
            raise
    
    def predict_image(self, image_data: bytes) -> Tuple[str, Dict[str, any]]:
        """
        Обработка изображения (для демонстрации - возвращает base64)
        В реальном проекте здесь была бы обработка изображения через модель
        
        Args:
            image_data: Байты изображения
            
        Returns:
            image_base64: Изображение в формате base64
            metadata: Метаданные обработки
        """
        import base64
        from PIL import Image
        import io
        
        try:
            # Загружаем изображение
            image = Image.open(io.BytesIO(image_data))
            
            # Здесь должна быть обработка через модель для изображений
            # Для демонстрации просто возвращаем изображение обратно в base64
            
            # Конвертируем обратно в base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            metadata = {
                "width": image.width,
                "height": image.height,
                "format": image.format,
                "size_bytes": len(image_data)
            }
            
            return img_base64, metadata
            
        except Exception as e:
            logger.error(f"Ошибка при обработке изображения: {e}")
            raise


# Глобальный экземпляр сервиса
ml_service = None


def get_ml_service() -> MLModelService:
    """Получить экземпляр ML сервиса (singleton)"""
    global ml_service
    if ml_service is None:
        # Попытка загрузить модель из checkpoint-3
        model_path = Path("models/model.pth")
        scaler_path = Path("models/scaler.pkl")
        
        if not model_path.exists():
            logger.warning("Модель не найдена. Используется модель по умолчанию.")
            model_path = None
        
        if not scaler_path.exists():
            logger.warning("Scaler не найден. Используется стандартизация по умолчанию.")
            scaler_path = None
        
        ml_service = MLModelService(
            model_path=str(model_path) if model_path and model_path.exists() else None,
            scaler_path=str(scaler_path) if scaler_path and scaler_path.exists() else None
        )
    
    return ml_service

