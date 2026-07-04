import os, json, warnings, pickle
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import timedelta
import tensorflow as tf
from tensorflow import keras

BASE       = Path(__file__).resolve().parent.parent   # app/ -> project root
MODEL_DIR  = BASE / "models"
SCALER_DIR = BASE / "scalers"
DATA_FILE  = BASE / "data" / "city_day_processed.csv"
LOOKBACK   = 30
HORIZON    = 5

BUCKETS = [
    (0,   50,  "Good",         "#00b050"),
    (51,  100, "Satisfactory", "#92d050"),
    (101, 200, "Moderate",     "#ffff00"),
    (201, 300, "Poor",         "#ff9900"),
    (301, 400, "Very Poor",    "#ff0000"),
    (401, 9999,"Severe",       "#c00000"),
]

def aqi_bucket(v):
    for lo, hi, label, color in BUCKETS:
        if lo <= v <= hi:
            return label, color
    return "Severe", "#c00000"

class BahdanauAttention(keras.layers.Layer):
    def __init__(self, units=32, **kw):
        super().__init__(**kw)
        self.W1    = keras.layers.Dense(units, use_bias=False)
        self.W2    = keras.layers.Dense(units, use_bias=False)
        self.V     = keras.layers.Dense(1,     use_bias=False)
        self.units = units

    def call(self, encoder_outputs, query):
        query_exp = tf.expand_dims(query, 1)
        score     = self.V(tf.nn.tanh(self.W1(encoder_outputs) + self.W2(query_exp)))
        weights   = tf.nn.softmax(score, axis=1)
        context   = tf.reduce_sum(weights * encoder_outputs, axis=1)
        return context, tf.squeeze(weights, -1)

    def get_config(self):
        c = super().get_config()
        c.update({"units": self.units})
        return c

@st.cache_resource(show_spinner="Loading model...")
def load_model():
    custom = {"BahdanauAttention": BahdanauAttention}
    for fname in ["cnn_lstm_gru_aqi.keras", "cnn_lstm_gru_aqi.h5"]:
        path = MODEL_DIR / fname
        if path.exists():
            try:
                m = keras.models.load_model(path, custom_objects=custom, compile=False)
                m.compile(optimizer="adam", loss="huber")
                st.success(f"Model loaded from {fname}")
                return m
            except Exception as e:
                st.warning(f"Could not load {fname}: {e}")
    st.error("No model file found in models/ folder.")
    st.stop()

@st.cache_resource(show_spinner="Loading scalers...")
def load_scalers():
    city_path = SCALER_DIR / "city_scalers.pkl"
    if not city_path.exists():
        st.error("city_scalers.pkl not found. Run Phase 2 first.")
        st.stop()
    with open(city_path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "scalers" in obj:
        city_scalers = obj["scalers"]
        scale_cols   = obj.get("scale_cols", [])
    else:
        city_scalers = obj
        scale_cols   = []
    aqi_idx = scale_cols.index("AQI") if "AQI" in scale_cols else -1
    return city_scalers, scale_cols, aqi_idx

@st.cache_data(show_spinner="Loading data...")
def load_data():
    df = pd.read_csv(DATA_FILE, parse_dates=["Date"])
    return df.sort_values(["City", "Date"]).reset_index(drop=True)

FEATURE_COLS = [
    "AQI","PM2.5","PM10","NO","NO2","NOx","NH3","CO","SO2","O3","Benzene","Toluene","Xylene",
    "AQI_lag1","AQI_lag3","AQI_lag7","AQI_lag14","AQI_lag30",
    "AQI_roll_mean_7","AQI_roll_mean_14","AQI_roll_mean_30",
    "AQI_roll_std_7","AQI_roll_std_14","AQI_roll_std_30",
    "day_sin","day_cos","month_sin","month_cos","dow_sin","dow_cos",
    "season_winter","season_summer","season_monsoon","season_post",
]

def add_features(cdf):
    cdf = cdf.copy().reset_index(drop=True)
    for lag in [1, 3, 7, 14, 30]:
        cdf[f"AQI_lag{lag}"] = cdf["AQI"].shift(lag)
    for w in [7, 14, 30]:
        cdf[f"AQI_roll_mean_{w}"] = cdf["AQI"].shift(1).rolling(w).mean()
        cdf[f"AQI_roll_std_{w}"]  = cdf["AQI"].shift(1).rolling(w).std()
    cdf["day_sin"]     = np.sin(2*np.pi*cdf["Date"].dt.dayofyear/365)
    cdf["day_cos"]     = np.cos(2*np.pi*cdf["Date"].dt.dayofyear/365)
    cdf["month_sin"]   = np.sin(2*np.pi*cdf["Date"].dt.month/12)
    cdf["month_cos"]   = np.cos(2*np.pi*cdf["Date"].dt.month/12)
    cdf["dow_sin"]     = np.sin(2*np.pi*cdf["Date"].dt.dayofweek/7)
    cdf["dow_cos"]     = np.cos(2*np.pi*cdf["Date"].dt.dayofweek/7)
    m = cdf["Date"].dt.month
    cdf["season_winter"]  = ((m>=12)|(m<=2)).astype(int)
    cdf["season_summer"]  = ((m>=3) &(m<=5)).astype(int)
    cdf["season_monsoon"] = ((m>=6) &(m<=9)).astype(int)
    cdf["season_post"]    = ((m>=10)&(m<=11)).astype(int)
    return cdf

def predict_aqi(city, anchor_date, model, city_scalers, scale_cols, aqi_idx, df):
    if city not in city_scalers:
        return None, f"City '{city}' not found in scaler dict"
    sc = city_scalers[city]

    cdf = df[df["City"] == city].reset_index(drop=True)

    if scale_cols and len(scale_cols) == sc.n_features_in_:
        avail_scale = [c for c in scale_cols if c in cdf.columns]
        if len(avail_scale) == sc.n_features_in_:
            cdf[scale_cols] = sc.transform(cdf[scale_cols])

    cdf = add_features(cdf)
    available = [c for c in FEATURE_COLS if c in cdf.columns]
    cdf = cdf.dropna(subset=available).reset_index(drop=True)
    window = cdf[cdf["Date"] <= pd.Timestamp(anchor_date)].tail(LOOKBACK)
    if len(window) < LOOKBACK:
        return None, f"Need {LOOKBACK} days before anchor (only {len(window)} available)"

    X_raw = window[available].values.astype("float32")

    if scale_cols and len(scale_cols) == sc.n_features_in_:
        X_sc = X_raw
    else:
        if X_raw.shape[1] == sc.n_features_in_:
            X_sc = sc.transform(X_raw)
        else:
            X_sc = X_raw.copy()
            X_sc[:, :sc.n_features_in_] = sc.transform(X_raw[:, :sc.n_features_in_])

    # --- SHAPE ALIGNMENT FIX ---
    # The model was trained on 33 features (due to a dropped dummy column like
    # 'season_post' or 'is_weekend'), but we generated 34 manually above.
    # We dynamically crop to exactly what the model expects to prevent crashes.
    expected_feats = model.input_shape[-1]
    if X_sc.shape[1] > expected_feats:
        X_sc = X_sc[:, :expected_feats]
    elif X_sc.shape[1] < expected_feats:
        pad = np.zeros((X_sc.shape[0], expected_feats - X_sc.shape[1]), dtype="float32")
        X_sc = np.concatenate([X_sc, pad], axis=1)

    preds = model.predict(X_sc[np.newaxis], verbose=0)[0]  # (HORIZON,) scaled

    reals = []
    for scaled_val in preds:
        dummy = np.zeros((1, sc.n_features_in_), dtype="float32")
        dummy[0, aqi_idx] = scaled_val
        real_aqi = float(sc.inverse_transform(dummy)[0, aqi_idx])
        real_aqi = max(0.0, min(600.0, real_aqi))
        reals.append(real_aqi)

    fdates = [pd.Timestamp(anchor_date)+timedelta(days=i+1) for i in range(HORIZON)]
    rows = []
    for i, (d, v) in enumerate(zip(fdates, reals)):
        bk, col = aqi_bucket(v)
        rows.append({"Day": f"t+{i+1}", "Date": d.strftime("%Y-%m-%d"),
                     "AQI": round(v, 1), "Bucket": bk, "Color": col})
    return pd.DataFrame(rows), None

st.set_page_config(page_title="AQI Forecast", page_icon="🌬️", layout="wide")
model = load_model()
city_scalers, scale_cols, aqi_idx = load_scalers()
df = load_data()
CITIES = sorted(df["City"].unique().tolist())

with st.sidebar:
    st.title("🌬️ AQI Forecast")
    st.markdown("---")
    city = st.selectbox("City", CITIES, index=CITIES.index("Delhi") if "Delhi" in CITIES else 0)
    anchor = st.date_input("Anchor Date", value=pd.Timestamp("2019-10-15").date(),
                           min_value=df["Date"].min().date(), max_value=df["Date"].max().date())
    run_batch = st.checkbox("Forecast all cities (batch)")
    st.markdown("---")
    st.caption("CNN-LSTM-GRU + Bahdanau Attention | 190K params | 86.65% bucket acc")

st.title(f"🌬️ AQI 5-Day Forecast — {city}")
st.caption(f"Anchor: {anchor}  |  t+1 through t+5")
result, err = predict_aqi(city, anchor, model, city_scalers, scale_cols, aqi_idx, df)

if err:
    st.error(err)
else:
    cols = st.columns(5)
    for i, row in result.iterrows():
        with cols[i]:
            st.metric(f"{row['Day']} · {row['Date']}", f"{row['AQI']} AQI", row["Bucket"])
    fig = go.Figure(go.Bar(x=result["Day"], y=result["AQI"], marker_color=result["Color"],
                           text=result["AQI"], textposition="outside"))
    for lo, hi, label, color in BUCKETS:
        if hi < 9999:
            fig.add_hline(y=hi, line_dash="dot", line_color=color, opacity=0.4,
                          annotation_text=label, annotation_position="right")
    fig.update_layout(title=f"5-Day AQI Forecast — {city}", yaxis_title="AQI", plot_bgcolor="white", height=380)
    st.plotly_chart(fig, use_container_width=True)
    c1, c2 = st.columns([2, 1])
    with c1:
        hist = df[(df["City"]==city)&(df["Date"]<=pd.Timestamp(anchor))].tail(90)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=hist["Date"], y=hist["AQI"], mode="lines",
                                  name="Historical", line=dict(color="#4a90d9", width=2)))
        fd = [pd.Timestamp(anchor)+timedelta(days=i+1) for i in range(HORIZON)]
        fig2.add_trace(go.Scatter(x=fd, y=result["AQI"].tolist(), mode="markers+lines", name="Forecast",
                                  marker=dict(size=10, color=result["Color"].tolist()),
                                  line=dict(dash="dash", color="#ff7043")))

        # FIX: Ensure anchor is converted to a Timestamp or datetime object Plotly can recognize properly
        # sometimes string dates containing only dates crash with add_vline
        anchor_ts = pd.Timestamp(anchor)
        fig2.add_vline(x=anchor_ts.timestamp() * 1000 if pd.api.types.is_datetime64_any_dtype(hist["Date"]) else anchor,
                       line_dash="dash", line_color="gray", annotation_text="Anchor")

        fig2.update_layout(title="Historical + Forecast", height=300, plot_bgcolor="white", yaxis_title="AQI")
        st.plotly_chart(fig2, use_container_width=True)
    with c2:
        st.dataframe(result[["Day","Date","AQI","Bucket"]],
                     hide_index=True, use_container_width=True)
        st.download_button("Download CSV", result.to_csv(index=False), f"forecast_{city}_{anchor}.csv", "text/csv")

if run_batch:
    st.markdown("---")
    st.subheader(f"Batch Forecast — All Cities on {anchor}")
    prog = st.progress(0)
    rows = []
    for i, c in enumerate(CITIES):
        res, e = predict_aqi(c, anchor, model, city_scalers, scale_cols, aqi_idx, df)
        if res is not None:
            rows.append({"City": c, "t+1": res.loc[0,"AQI"], "t+2": res.loc[1,"AQI"],
                         "t+3": res.loc[2,"AQI"], "t+4": res.loc[3,"AQI"], "t+5": res.loc[4,"AQI"],
                         "Avg": round(res["AQI"].mean(),1)})
        prog.progress((i+1)/len(CITIES))
    bdf = pd.DataFrame(rows).sort_values("Avg", ascending=False)
    hm  = bdf.set_index("City")[["t+1","t+2","t+3","t+4","t+5"]]
    fig3 = px.imshow(hm, color_continuous_scale="RdYlGn_r", title="All-City Forecast Heatmap", aspect="auto", height=600)
    st.plotly_chart(fig3, use_container_width=True)
    st.dataframe(bdf, hide_index=True, use_container_width=True)
    st.download_button("Download Batch CSV", bdf.to_csv(index=False), f"batch_{anchor}.csv", "text/csv")

st.markdown("---")
st.caption("AQI Forecasting | CNN-LSTM-GRU + Bahdanau Attention | Phase 7 Dashboard")
