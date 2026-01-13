import json
import numpy as np

# Генерируем 60 временных точек с 15 фичами каждая
np.random.seed(42)
data = []

for i in range(60):
    # Генерируем реалистичные данные для криптовалюты
    point = [
        np.random.uniform(-0.01, 0.01),      # ret_1
        abs(np.random.uniform(0, 0.01)),      # ret_abs
        np.random.uniform(0, 0.02),           # hl_spread
        np.random.uniform(-0.01, 0.01),       # oc_spread
        np.random.uniform(0, 0.02),           # vol_roll
        np.random.uniform(30, 70),             # rsi_14
        np.random.uniform(50000, 60000),       # ema_12
        np.random.uniform(50000, 60000),       # ema_26
        np.random.uniform(100, 500),           # atr_14
        np.random.uniform(1000000, 2000000),  # vol_ma_24
        np.random.uniform(0, 0.01),           # дополнительные фичи
        np.random.uniform(0, 0.01),
        np.random.uniform(0, 0.01),
        np.random.uniform(0, 0.01),
        np.random.uniform(0, 0.01)
    ]
    data.append([float(x) for x in point])

demo_request = {
    "data": data,
    "asset": "BTC"
}

with open("demo_data.json", "w") as f:
    json.dump(demo_request, f, indent=2)

print("✅ Файл demo_data.json создан!")
print(f"   Количество точек: {len(data)}")
print(f"   Фичей на точку: {len(data[0])}")