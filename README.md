# AQI Forecasting System
### CNN-LSTM-GRU + Bahdanau Attention | 26 Indian Cities | 5-Day Forecast

## Key Results
| Metric | Value |
|--------|-------|
| Overall Bucket Accuracy | 86.65% |
| Best city MAE (Bengaluru) | 7.71 AQI |
| Worst city MAE (Patna) | 31.02 AQI |
| Average MAE (all cities) | ~12.8 AQI |
| R² (t+1) | 0.527 |
| Model parameters | 190,309 |
| Best val_loss | 0.010878 (Huber) |

## Architecture
```
Input (30, 34)
   ↓
Conv1D(64, k=3, causal) × 2   ← local pollutant patterns
   ↓
LSTM(128, return_sequences=True)  ← long-range dependencies
   ↓
GRU(64, return_sequences=True)    ← gated refinement
   ↓
BahdanauAttention(32)             ← day-weighting
   ↓
Dense(64) → Dense(32) → Dense(5)  ← 5-day AQI forecast
```

## Dataset
- Source: CPCB city_day.csv via Kaggle (rohanrao/air-quality-data-in-india)
- 29,531 rows | 26 cities | 2015-2020

## How to Run Inference
```python
from Phase6_Inference import forecast_city
result = forecast_city(city='Delhi', anchor_date='2019-10-15')
print(result)
```