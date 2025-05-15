# Глава 58: FlashAttention для Алгоритмической Торговли

В этой главе рассматривается **FlashAttention** — IO-оптимизированный алгоритм точного внимания, который обеспечивает более быстрое и эффективное по памяти обучение и инференс Трансформеров. Мы применяем FlashAttention к прогнозированию финансовых временных рядов, демонстрируя, как его преимущества в эффективности позволяют использовать более длинные контекстные окна для захвата рыночных паттернов.

<p align="center">
<img src="https://i.imgur.com/9K8xYQf.png" width="70%">
</p>

## Содержание

1. [Введение в FlashAttention](#введение-в-flashattention)
    * [Проблема узкого места памяти](#проблема-узкого-места-памяти)
    * [Ключевые инновации](#ключевые-инновации)
    * [Преимущества для торговых моделей](#преимущества-для-торговых-моделей)
2. [Алгоритм FlashAttention](#алгоритм-flashattention)
    * [Обзор стандартного внимания](#обзор-стандартного-внимания)
    * [IO-осведомленные вычисления](#io-осведомленные-вычисления)
    * [Тайлинг и перевычисление](#тайлинг-и-перевычисление)
    * [Улучшения FlashAttention-2](#улучшения-flashattention-2)
3. [Применение в трейдинге](#применение-в-трейдинге)
    * [Прогнозирование цен с длинным контекстом](#прогнозирование-цен-с-длинным-контекстом)
    * [Высокочастотный анализ книги заявок](#высокочастотный-анализ-книги-заявок)
    * [Мультиактивное портфельное моделирование](#мультиактивное-портфельное-моделирование)
4. [Практические примеры](#практические-примеры)
    * [01: Подготовка данных](#01-подготовка-данных)
    * [02: FlashAttention Трансформер](#02-flashattention-трансформер)
    * [03: Обучение модели](#03-обучение-модели)
    * [04: Прогнозирование цен](#04-прогнозирование-цен)
    * [05: Бэктестинг торговой стратегии](#05-бэктестинг-торговой-стратегии)
5. [Реализация на Python](#реализация-на-python)
6. [Реализация на Rust](#реализация-на-rust)
7. [Бенчмарки производительности](#бенчмарки-производительности)
8. [Лучшие практики](#лучшие-практики)
9. [Ресурсы](#ресурсы)

## Введение в FlashAttention

FlashAttention — это прорывной алгоритм, разработанный Tri Dao и др. (2022), который делает вычисление внимания в Трансформерах значительно быстрее и эффективнее по памяти без потери точности. В отличие от приближенных методов внимания, которые жертвуют качеством ради скорости, FlashAttention вычисляет **точное внимание**, достигая при этом ускорения в 2-4 раза.

### Проблема узкого места памяти

Стандартное внимание Трансформера имеет сложность O(N²) по времени и памяти, где N — длина последовательности. Для торговых приложений это создает существенные ограничения:

```
Использование памяти традиционным вниманием:
┌────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  Длина последовательности (N)    Память для матрицы внимания    Практика  │
│  ─────────────────────────────────────────────────────────────────────────  │
│       512                           ~1 МБ                     ✓ Легко      │
│      2,048                         ~16 МБ                     ✓ Стандарт   │
│      8,192                        ~256 МБ                     ⚠ Сложно    │
│     32,768                          ~4 ГБ                     ✗ Часто      │
│                                                                 невозможно │
│    131,072                         ~64 ГБ                     ✗ Требует    │
│                                                                 спец. железа│
└────────────────────────────────────────────────────────────────────────────┘
```

Для трейдинга длинные последовательности критически важны:
- **1 год дневных данных**: ~252 временных шага (управляемо)
- **1 месяц часовых данных**: ~720 временных шагов (управляемо)
- **1 неделя минутных данных**: ~10,080 временных шагов (проблематично)
- **1 день тиковых данных**: ~100,000+ временных шагов (очень проблематично)

### Ключевые инновации

FlashAttention вводит две основные техники:

1. **Тайлинг (Tiling)**: Разбивает вычисление внимания на меньшие блоки, помещающиеся в GPU SRAM
2. **Перевычисление (Recomputation)**: Перевычисляет внимание в обратном проходе вместо хранения больших промежуточных матриц

```
Стандартный поток внимания (Интенсивный по памяти):
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│    Q, K, V                                                                   │
│       │                                                                      │
│       ▼                                                                      │
│   ┌───────────────┐                                                          │
│   │ Вычислить S=QK^T │  ← Хранить всю матрицу N×N в HBM (дорого!)           │
│   └───────┬───────┘                                                          │
│           │                                                                  │
│           ▼                                                                  │
│   ┌───────────────┐                                                          │
│   │ Вычислить P=softmax(S) │  ← Хранить еще одну матрицу N×N               │
│   └───────┬───────┘                                                          │
│           │                                                                  │
│           ▼                                                                  │
│   ┌───────────────┐                                                          │
│   │ Вычислить O=PV │  ← Наконец вычислить выход                             │
│   └───────────────┘                                                          │
│                                                                              │
│   Всего обращений к HBM: O(N² + N²) = O(N²)                                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

Поток FlashAttention (IO-эффективный):
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│    Q, K, V (в HBM)                                                           │
│       │                                                                      │
│       │  Загрузить блоки Q, K, V в SRAM                                     │
│       ▼                                                                      │
│   ┌───────────────────────────────────────────────────────────────────────┐ │
│   │                        ДЛЯ каждого блока:                               │ │
│   │   ┌─────────────────────────────────────────────────────────────────┐ │ │
│   │   │ 1. Загрузить Q_block, K_block, V_block из HBM в SRAM            │ │ │
│   │   │ 2. Вычислить S_block = Q_block × K_block^T  (в SRAM)            │ │ │
│   │   │ 3. Вычислить P_block = softmax(S_block)      (в SRAM)           │ │ │
│   │   │ 4. Вычислить O_block = P_block × V_block     (в SRAM)           │ │ │
│   │   │ 5. Обновить накопленный выход и статистику                      │ │ │
│   │   │ 6. Записать только финальный выход в HBM                        │ │ │
│   │   └─────────────────────────────────────────────────────────────────┘ │ │
│   └───────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│   Всего обращений к HBM: O(N² / M), где M = размер SRAM                     │
│   Обычно в 10-20 раз меньше обращений к памяти!                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Преимущества для торговых моделей

| Преимущество | Стандартное внимание | FlashAttention | Влияние на трейдинг |
|--------------|---------------------|----------------|---------------------|
| Память | O(N²) | O(N) | Обработка в 10 раз более длинных последовательностей |
| Скорость | Базовая | В 2-4 раза быстрее | Быстрее бэктесты, инференс в реальном времени |
| Точность | Точное | Точное | Без компромисса в качестве |
| Контекст | ~2K токенов типично | ~16K+ токенов | Захват более длинных рыночных паттернов |

## Алгоритм FlashAttention

### Обзор стандартного внимания

Стандартный механизм внимания вычисляет:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

Где:
- Q (Query, Запрос): Что мы ищем
- K (Key, Ключ): Какая информация у нас есть
- V (Value, Значение): Фактическое информационное содержание
- d_k: Размерность ключей (для масштабирования)

Для финансовых временных рядов:
- Q может представлять "текущее состояние рынка"
- K может представлять "исторические паттерны"
- V содержит фактическую информацию о ценах/объемах

### IO-осведомленные вычисления

Ключевое понимание FlashAttention в том, что память GPU имеет иерархию:

```
Иерархия памяти GPU:
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                         SRAM (На кристалле)                          │   │
│   │   • Размер: ~20 МБ (A100)                                            │   │
│   │   • Скорость: ~19 ТБ/с                                               │   │
│   │   • Задержка: ~1 цикл                                                │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                   │                                          │
│                                   │ ← Узкое место!                          │
│                                   ▼                                          │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                         HBM (Внешняя память)                         │   │
│   │   • Размер: 40-80 ГБ (A100)                                          │   │
│   │   • Скорость: ~2 ТБ/с                                                │   │
│   │   • Задержка: ~100 циклов                                            │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   SRAM примерно в 10 раз быстрее HBM, но в ~1000 раз меньше                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Стандартное внимание** многократно записывает промежуточные матрицы N×N в HBM.
**FlashAttention** хранит всё в SRAM используя тайлинг.

### Тайлинг и перевычисление

FlashAttention обрабатывает внимание блоками:

```python
# Псевдокод прямого прохода FlashAttention
def flash_attention_forward(Q, K, V, block_size=256):
    """
    IO-осведомленное вычисление внимания.

    Ключевые идеи:
    1. Обрабатывать Q, K, V блоками, помещающимися в SRAM
    2. Поддерживать накопленную статистику для нормализации softmax
    3. Никогда не материализовать полную матрицу внимания N×N
    """
    N, d = Q.shape
    O = zeros_like(Q)  # Выход
    l = zeros(N)       # Накопленная сумма для знаменателя softmax
    m = full(N, -inf)  # Накопленный максимум для численной стабильности

    # Обработка K, V блоками
    for j in range(0, N, block_size):
        Kj = K[j:j+block_size]
        Vj = V[j:j+block_size]

        # Обработка Q блоками
        for i in range(0, N, block_size):
            Qi = Q[i:i+block_size]

            # Вычислить блок оценок внимания (в SRAM)
            Sij = Qi @ Kj.T / sqrt(d)

            # Обновить накопленный максимум
            m_new = maximum(m[i:i+block_size], Sij.max(axis=-1))

            # Вычислить локальный softmax с коррекцией
            P_ij = exp(Sij - m_new[:, None])

            # Обновить накопленную сумму
            l_new = exp(m[i:i+block_size] - m_new) * l[i:i+block_size] + P_ij.sum(axis=-1)

            # Обновить выход с коррекционным множителем
            O[i:i+block_size] = (
                exp(m[i:i+block_size] - m_new)[:, None] * O[i:i+block_size] +
                P_ij @ Vj
            )

            # Сохранить новую статистику
            m[i:i+block_size] = m_new
            l[i:i+block_size] = l_new

    # Финальная нормализация
    O = O / l[:, None]
    return O
```

Критическое понимание — трюк с **онлайн softmax**: мы можем вычислять softmax инкрементально, отслеживая накопленный максимум и сумму, затем применяя корректирующие множители.

### Улучшения FlashAttention-2

FlashAttention-2 (Dao, 2023) улучшает оригинал с помощью:

1. **Сокращение не-матричных FLOP**: Современные GPU имеют специализированные Tensor Cores, которые делают матричное умножение в 16 раз быстрее других операций. FlashAttention-2 минимизирует не-матричные операции.

2. **Лучший параллелизм**: Параллелизует по размерности длины последовательности в дополнение к batch и heads, обеспечивая лучшую утилизацию GPU для длинных последовательностей.

3. **Улучшенное распределение работы**: Лучшее распределение работы между warp'ами внутри каждого блока потоков.

```
Производительность FlashAttention vs FlashAttention-2:
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│   Метрика                FlashAttention    FlashAttention-2    Улучшение   │
│   ─────────────────────────────────────────────────────────────────────────  │
│   Утилизация GPU         25-40%            50-73%              ~2x          │
│   Скорость обучения      Быстро            Очень быстро        ~2x          │
│   Длина последовательности  До 16K         До 64K+             4x+          │
│   Эффективность памяти   Линейная          Линейная            Так же       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Применение в трейдинге

### Прогнозирование цен с длинным контекстом

FlashAttention позволяет моделям учитывать гораздо более длинный исторический контекст:

```python
# Традиционный подход: Ограниченный контекст
lookback_traditional = 512  # ~1 месяц часовых данных

# С FlashAttention: Расширенный контекст
lookback_flash = 4096  # ~6 месяцев часовых данных
# или
lookback_flash = 16384  # ~2 года часовых данных

# Это важно потому что:
# - Сезонные паттерны могут охватывать месяцы
# - Крупные рыночные события имеют долгосрочные последствия
# - Корреляции между активами эволюционируют со временем
```

**Пример: Прогнозирование крипторынка**

```python
import torch
from flash_attention_trading import FlashAttentionTrader

# Конфигурация для криптотрейдинга
config = {
    'context_length': 8192,    # 2+ недели часовых данных
    'd_model': 256,
    'n_heads': 8,
    'n_layers': 6,
    'symbols': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT'],
    'data_source': 'bybit',
    'use_flash_attention': True  # Включить FlashAttention
}

model = FlashAttentionTrader(**config)

# Стандартное внимание потребовало бы 8192² × 4 байта = 256МБ на слой
# FlashAttention сокращает это до O(8192) = ~32КБ на слой
```

### Высокочастотный анализ книги заявок

Для данных книги заявок нам часто нужно обрабатывать много уровней и быстрые обновления:

```python
# Анализ книги заявок с FlashAttention
class OrderBookFlashAttention:
    def __init__(self, n_levels=50, history_length=1000):
        """
        Анализ лимитной книги заявок с вниманием.

        n_levels: Количество уровней bid/ask для рассмотрения
        history_length: Количество исторических снимков
        """
        self.sequence_length = n_levels * 2 * history_length
        # Традиционно: 100,000² внимание = 40 ГБ
        # FlashAttention: Легко справляется с ~100 МБ

    def predict_mid_price_movement(self, order_book_history):
        """
        Использовать внимание для поиска паттернов в динамике книги заявок.

        Внимание может обнаружить:
        - Какие ценовые уровни наиболее предсказательны
        - Как дисбалансы на разных уровнях взаимодействуют
        - Временные паттерны в потоке заявок
        """
        pass
```

### Мультиактивное портфельное моделирование

FlashAttention позволяет моделировать отношения между многими активами:

```python
# Мультиактивный портфель с кросс-активным вниманием
class FlashPortfolioModel:
    def __init__(self, n_assets=100, lookback=2048):
        """
        Модель с кросс-активным вниманием.

        При n_assets=100 и lookback=2048:
        - Длина последовательности = 100 × 2048 = 204,800
        - Традиционное внимание: 204,800² = 158 ГБ (невозможно!)
        - FlashAttention: Справляется с ~1 ГБ
        """
        self.model = TransformerWithFlashAttention(
            seq_len=n_assets * lookback,
            d_model=128,
            n_heads=8,
            n_layers=4,
            use_flash=True
        )
```

## Практические примеры

### 01: Подготовка данных

```python
# python/data_loader.py

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import requests
from datetime import datetime, timedelta

def fetch_bybit_klines(
    symbol: str,
    interval: str = '60',  # 1 час
    limit: int = 1000
) -> pd.DataFrame:
    """
    Получить OHLCV данные с Bybit.

    Args:
        symbol: Торговая пара (например, 'BTCUSDT')
        interval: Интервал свечи в минутах
        limit: Количество свечей для получения

    Returns:
        DataFrame с OHLCV данными
    """
    url = 'https://api.bybit.com/v5/market/kline'

    params = {
        'category': 'linear',
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }

    response = requests.get(url, params=params)
    data = response.json()

    if data['retCode'] != 0:
        raise ValueError(f"Ошибка API: {data['retMsg']}")

    df = pd.DataFrame(data['result']['list'], columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
    ])

    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
        df[col] = df[col].astype(float)

    return df.sort_values('timestamp').reset_index(drop=True)


def prepare_flash_attention_data(
    symbols: List[str],
    lookback: int = 2048,
    horizon: int = 24
) -> Dict[str, np.ndarray]:
    """
    Подготовить данные для торговой модели с FlashAttention.

    Длинное контекстное окно (2048) практично только с FlashAttention.
    Стандартное внимание потребовало бы 2048² × n_symbols = запретительно много памяти.

    Args:
        symbols: Список торговых пар
        lookback: Длина исторического контекста
        horizon: Горизонт прогноза

    Returns:
        Словарь с X (признаки) и y (цели)
    """
    all_data = []

    for symbol in symbols:
        df = fetch_bybit_klines(symbol, limit=lookback + horizon + 100)

        # Вычислить признаки
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['log_return'].rolling(24).std()
        df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(24).mean()
        df['price_ma_ratio'] = df['close'] / df['close'].rolling(24).mean()

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / loss))

        df = df.dropna()
        all_data.append(df)

    # Выровнять все датафреймы
    min_len = min(len(df) for df in all_data)
    aligned = [df.iloc[-min_len:].reset_index(drop=True) for df in all_data]

    # Создать последовательности
    features = ['log_return', 'volatility', 'volume_ma_ratio', 'price_ma_ratio', 'rsi']
    n_features = len(features) * len(symbols)

    X, y = [], []

    for i in range(lookback, min_len - horizon):
        # Объединить признаки всех символов
        x_sample = np.zeros((lookback, n_features))
        for j, df in enumerate(aligned):
            for k, feat in enumerate(features):
                x_sample[:, j * len(features) + k] = df[feat].iloc[i-lookback:i].values

        # Цель: будущие доходности для всех символов
        y_sample = np.array([
            df['log_return'].iloc[i:i+horizon].sum()
            for df in aligned
        ])

        X.append(x_sample)
        y.append(y_sample)

    return {
        'X': np.array(X),
        'y': np.array(y),
        'symbols': symbols,
        'feature_names': [f"{s}_{f}" for s in symbols for f in features]
    }
```

### 02: FlashAttention Трансформер

```python
# python/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

# Попытка импорта FlashAttention
try:
    from flash_attn import flash_attn_func
    FLASH_AVAILABLE = True
except ImportError:
    FLASH_AVAILABLE = False
    print("FlashAttention не установлен. Используется стандартное внимание.")


class FlashMultiHeadAttention(nn.Module):
    """
    Multi-head внимание с поддержкой FlashAttention.
    Откатывается на стандартное внимание если FlashAttention недоступен.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        use_flash: bool = True
    ):
        super().__init__()

        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.use_flash = use_flash and FLASH_AVAILABLE

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_k)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Прямой проход с FlashAttention или стандартным вниманием.

        Args:
            x: Входной тензор [batch, seq_len, d_model]
            mask: Опциональная маска внимания
            return_attention: Возвращать ли веса внимания

        Returns:
            Выходной тензор и опционально веса внимания
        """
        batch_size, seq_len, _ = x.shape

        # Проецировать в Q, K, V
        Q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        K = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        V = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k)

        if self.use_flash and not return_attention:
            # Использовать FlashAttention (не поддерживает возврат весов внимания)
            # FlashAttention ожидает [batch, seq, n_heads, d_k]
            output = flash_attn_func(Q, K, V, dropout_p=self.dropout.p if self.training else 0.0)
            output = output.view(batch_size, seq_len, self.d_model)
            attn_weights = None
        else:
            # Стандартное внимание (откат или когда нужны веса внимания)
            Q = Q.transpose(1, 2)  # [batch, n_heads, seq_len, d_k]
            K = K.transpose(1, 2)
            V = V.transpose(1, 2)

            # Вычислить оценки внимания
            scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))

            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)

            output = torch.matmul(attn_weights, V)
            output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        output = self.out_proj(output)

        return output, attn_weights
```

### 03: Обучение модели

```python
# python/train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from typing import Dict, Optional
import logging

from model import FlashAttentionTrader
from data_loader import prepare_flash_attention_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_model(
    model: FlashAttentionTrader,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    lr: float = 1e-4,
    device: str = 'cuda'
) -> Dict[str, list]:
    """
    Обучить торговую модель с FlashAttention.

    Args:
        model: Модель FlashAttentionTrader
        train_loader: Загрузчик обучающих данных
        val_loader: Загрузчик валидационных данных
        epochs: Количество эпох обучения
        lr: Скорость обучения
        device: Устройство для обучения

    Returns:
        Словарь с историей обучения
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Обучение
        model.train()
        train_losses = []

        for batch_x, batch_y in tqdm(train_loader, desc=f'Эпоха {epoch+1}/{epochs}'):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            predictions, _ = model(batch_x)
            loss = model.compute_loss(predictions, batch_y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())

        # Валидация
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                predictions, _ = model(batch_x)
                loss = model.compute_loss(predictions, batch_y)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        logger.info(f'Эпоха {epoch+1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}')

        # Сохранить лучшую модель
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')
            logger.info(f'Сохранена лучшая модель с val_loss = {val_loss:.6f}')

        scheduler.step()

    return history
```

### 04: Прогнозирование цен

```python
# python/predict.py

import torch
import numpy as np
import pandas as pd
from typing import List, Dict
import matplotlib.pyplot as plt

from model import FlashAttentionTrader
from data_loader import prepare_flash_attention_data, fetch_bybit_klines


def predict_returns(
    model: FlashAttentionTrader,
    X: np.ndarray,
    device: str = 'cuda'
) -> np.ndarray:
    """
    Сгенерировать прогнозы доходностей.

    Args:
        model: Обученный FlashAttentionTrader
        X: Входные признаки [n_samples, seq_len, n_features]
        device: Устройство для инференса

    Returns:
        Прогнозируемые доходности [n_samples, n_assets]
    """
    model = model.to(device)
    model.eval()

    X_tensor = torch.FloatTensor(X).to(device)

    with torch.no_grad():
        predictions, _ = model(X_tensor)

    return predictions.cpu().numpy()


def predict_with_attention_analysis(
    model: FlashAttentionTrader,
    X: np.ndarray,
    symbols: List[str],
    device: str = 'cuda'
) -> Dict:
    """
    Сделать прогнозы и проанализировать паттерны внимания.

    Примечание: Анализ внимания требует стандартного внимания (FlashAttention
    не возвращает веса внимания). Это полезно для интерпретируемости.
    """
    model = model.to(device)
    model.eval()

    # Временно отключить FlashAttention для получения весов внимания
    original_use_flash = model.use_flash
    model.use_flash = False
    for layer in model.layers:
        layer.attention.use_flash = False

    X_tensor = torch.FloatTensor(X).to(device)

    with torch.no_grad():
        predictions, attention_weights = model(X_tensor, return_attention=True)

    # Восстановить настройку FlashAttention
    model.use_flash = original_use_flash
    for layer in model.layers:
        layer.attention.use_flash = original_use_flash

    return {
        'predictions': predictions.cpu().numpy(),
        'attention_weights': attention_weights
    }
```

### 05: Бэктестинг торговой стратегии

```python
# python/strategy.py

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

@dataclass
class BacktestResult:
    """Контейнер для результатов бэктеста."""
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    portfolio_values: np.ndarray
    trades: List[Dict]


def calculate_metrics(returns: np.ndarray, risk_free_rate: float = 0.0) -> Dict[str, float]:
    """Вычислить метрики торговой эффективности."""

    excess_returns = returns - risk_free_rate / 252  # Дневная безрисковая ставка

    # Коэффициент Шарпа (годовой)
    sharpe = np.sqrt(252) * excess_returns.mean() / (returns.std() + 1e-8)

    # Коэффициент Сортино (штраф только за нисходящую волатильность)
    downside_returns = returns[returns < 0]
    downside_std = np.sqrt(np.mean(downside_returns**2)) if len(downside_returns) > 0 else 1e-8
    sortino = np.sqrt(252) * excess_returns.mean() / (downside_std + 1e-8)

    # Максимальная просадка
    cumulative = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    max_drawdown = drawdowns.min()

    # Процент выигрышных сделок
    win_rate = (returns > 0).mean()

    return {
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'total_return': cumulative[-1] - 1
    }


def backtest_flash_attention_strategy(
    model,
    test_data: Dict,
    symbols: List[str],
    initial_capital: float = 100000,
    transaction_cost: float = 0.001,
    position_size: float = 0.1,
    device: str = 'cuda'
) -> BacktestResult:
    """
    Бэктест торговой стратегии с использованием прогнозов модели FlashAttention.

    Стратегия:
    - Long когда прогнозируемая доходность > порога
    - Short когда прогнозируемая доходность < -порога
    - Размер позиции пропорционален уверенности прогноза

    Args:
        model: Обученный FlashAttentionTrader
        test_data: Тестовый датасет с X и y
        symbols: Список торговых символов
        initial_capital: Начальный капитал
        transaction_cost: Стоимость сделки (как доля)
        position_size: Максимальный размер позиции как доля капитала
        device: Устройство для инференса

    Returns:
        BacktestResult с метриками эффективности
    """
    import torch

    model = model.to(device)
    model.eval()

    X = test_data['X']
    y = test_data['y']  # Фактические доходности

    n_samples = len(X)
    n_assets = len(symbols)

    # Отслеживание портфеля
    capital = initial_capital
    portfolio_values = [capital]
    positions = np.zeros(n_assets)
    trades = []

    # Сгенерировать все прогнозы
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        predictions, _ = model(X_tensor)
        predictions = predictions.cpu().numpy()

    # Запустить бэктест
    for i in range(n_samples):
        pred = predictions[i]
        actual_returns = y[i]

        # Сгенерировать сигналы (нормализованные по величине прогноза)
        signals = np.tanh(pred * 10)  # Масштабировать и ограничить [-1, 1]

        # Вычислить целевые позиции
        target_positions = signals * position_size

        # Вычислить изменения позиций и издержки
        position_changes = target_positions - positions
        trade_cost = np.abs(position_changes).sum() * transaction_cost * capital

        # Записать сделки
        for j, symbol in enumerate(symbols):
            if abs(position_changes[j]) > 0.001:
                trades.append({
                    'step': i,
                    'symbol': symbol,
                    'action': 'buy' if position_changes[j] > 0 else 'sell',
                    'size': abs(position_changes[j]),
                    'predicted_return': pred[j],
                    'actual_return': actual_returns[j]
                })

        # Обновить позиции
        positions = target_positions

        # Вычислить доходности
        portfolio_return = np.sum(positions * actual_returns)
        capital = capital * (1 + portfolio_return) - trade_cost
        portfolio_values.append(capital)

    portfolio_values = np.array(portfolio_values)
    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]

    # Вычислить метрики
    metrics = calculate_metrics(daily_returns)

    return BacktestResult(
        total_return=metrics['total_return'],
        sharpe_ratio=metrics['sharpe_ratio'],
        sortino_ratio=metrics['sortino_ratio'],
        max_drawdown=metrics['max_drawdown'],
        win_rate=metrics['win_rate'],
        portfolio_values=portfolio_values,
        trades=trades
    )
```

## Реализация на Python

```
python/
├── __init__.py
├── model.py                # FlashAttention Трансформер
├── data_loader.py          # Загрузка данных Bybit и инженерия признаков
├── train.py                # Скрипт обучения
├── predict.py              # Утилиты прогнозирования
├── strategy.py             # Торговая стратегия и бэктестинг
├── requirements.txt        # Python зависимости
└── examples/
    ├── 01_data_preparation.ipynb
    ├── 02_model_architecture.ipynb
    ├── 03_training.ipynb
    ├── 04_prediction.ipynb
    └── 05_backtesting.ipynb
```

### Быстрый старт (Python)

```bash
# Установить зависимости
cd python
pip install -r requirements.txt

# Опционально: Установить FlashAttention (требует CUDA)
pip install flash-attn --no-build-isolation

# Получить данные и обучить
python data_loader.py --symbols BTCUSDT,ETHUSDT,SOLUSDT
python train.py --epochs 50 --batch-size 16

# Запустить бэктест
python strategy.py --model best_model.pt
```

## Реализация на Rust

Смотрите [rust/](rust/) для production-ready реализации на Rust.

```
rust/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs                 # Экспорты библиотеки
│   ├── attention/
│   │   ├── mod.rs
│   │   ├── standard.rs        # Стандартное внимание (откат)
│   │   └── flash.rs           # Flash-стиль реализация внимания
│   ├── model/
│   │   ├── mod.rs
│   │   ├── transformer.rs     # Архитектура трансформера
│   │   └── trading.rs         # Торговая модель
│   ├── data/
│   │   ├── mod.rs
│   │   ├── bybit.rs           # Клиент Bybit API
│   │   ├── yahoo.rs           # Интеграция Yahoo Finance
│   │   └── features.rs        # Инженерия признаков
│   └── strategy/
│       ├── mod.rs
│       ├── signals.rs         # Генерация сигналов
│       └── backtest.rs        # Движок бэктестинга
└── examples/
    ├── fetch_data.rs
    ├── train.rs
    └── backtest.rs
```

### Быстрый старт (Rust)

```bash
cd rust

# Собрать проект
cargo build --release

# Получить данные
cargo run --example fetch_data -- --symbols BTCUSDT,ETHUSDT

# Обучить модель
cargo run --example train -- --epochs 50

# Запустить бэктест
cargo run --example backtest
```

## Бенчмарки производительности

### Сравнение использования памяти

| Длина последовательности | Стандартное внимание | FlashAttention | Сокращение |
|--------------------------|---------------------|----------------|------------|
| 512 | 1.0 МБ | 0.1 МБ | 10x |
| 2,048 | 16 МБ | 0.4 МБ | 40x |
| 8,192 | 256 МБ | 1.6 МБ | 160x |
| 32,768 | 4 ГБ | 6.4 МБ | 640x |

### Сравнение скорости (A100 GPU)

| Операция | Стандартное внимание | FlashAttention | FlashAttention-2 |
|----------|---------------------|----------------|------------------|
| Forward (seq=2K) | 100 мс | 45 мс | 25 мс |
| Forward (seq=8K) | 1600 мс | 180 мс | 95 мс |
| Backward (seq=2K) | 300 мс | 135 мс | 70 мс |
| Backward (seq=8K) | 4800 мс | 540 мс | 280 мс |

## Лучшие практики

### Когда использовать FlashAttention

**Рекомендуемые сценарии:**
- Длинные временные ряды (>1000 временных шагов)
- Множество активов с кросс-вниманием
- Инференс в реальном времени где важна скорость
- Обучение на GPU с ограниченной памятью

**Может не понадобиться:**
- Короткие последовательности (<512)
- Простые модели без внимания
- Развертывание только на CPU

### Типичные ошибки

1. **Не использование смешанной точности**: FlashAttention лучше работает с FP16/BF16
   ```python
   # Использовать автоматическую смешанную точность
   with torch.autocast(device_type='cuda', dtype=torch.float16):
       output = model(x)
   ```

2. **Ожидание весов внимания**: FlashAttention не сохраняет матрицу внимания
   ```python
   # Для интерпретируемости временно отключить FlashAttention
   model.use_flash = False
   output, attention = model(x, return_attention=True)
   ```

## Ресурсы

### Научные статьи

- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135) — Оригинальная статья (2022)
- [FlashAttention-2: Faster Attention with Better Parallelism](https://arxiv.org/abs/2307.08691) — Улучшенная версия (2023)
- [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608) — Последняя итерация (2024)

### Реализации

- [FlashAttention GitHub](https://github.com/Dao-AILab/flash-attention) — Официальная реализация
- [PyTorch SDPA](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) — Встроенный flash attention PyTorch
- [xFormers](https://github.com/facebookresearch/xformers) — Эффективное по памяти внимание от Facebook

### Связанные главы

- [Глава 52: Performer Efficient Attention](../52_performer_efficient_attention) — Приближенное линейное внимание
- [Глава 54: Reformer LSH Attention](../54_reformer_lsh_attention) — Внимание с локально-чувствительным хэшированием
- [Глава 57: Longformer Financial](../57_longformer_financial) — Внимание со скользящим окном

---

## Уровень сложности

**Продвинутый**

Пререквизиты:
- Архитектура трансформера и механизм self-attention
- Иерархия памяти GPU и оптимизация
- PyTorch или аналогичный фреймворк глубокого обучения
- Базовые знания торговых стратегий
