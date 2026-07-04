import os, json, warnings, pickle, time
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import requests
from pathlib import Path
from datetime import timedelta, datetime, date
import tensorflow as tf
from tensorflow import keras

BASE       = Path(__file__).resolve().parent.parent
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

# --- City coordinates for OpenWeatherMap API ---
CITY_COORDS = {
    "Ahmedabad": (23.0225, 72.5714), "Aizawl": (23.7271, 92.7176),
    "Amaravati": (16.5062, 80.6480), "Amritsar": (31.6340, 74.8723),
    "Bengaluru": (12.9716, 77.5946), "Bhopal": (23.2599, 77.4126),
    "Brajrajnagar": (21.8167, 83.9167), "Chandigarh": (30.7333, 76.7794),
    "Chennai": (13.0827, 80.2707), "Coimbatore": (11.0168, 76.9558),
    "Delhi": (28.6139, 77.2090), "Ernakulam": (9.9816, 76.2999),
    "Gurugram": (28.4595, 77.0266), "Guwahati": (26.1445, 91.7362),
    "Hyderabad": (17.3850, 78.4867), "Jaipur": (26.9124, 75.7873),
    "Jorapokhar": (23.7100, 86.4200), "Kochi": (9.9312, 76.2673),
    "Kolkata": (22.5726, 88.3639), "Lucknow": (26.8467, 80.9462),
    "Mumbai": (19.0760, 72.8777), "Patna": (25.6093, 85.1376),
    "Shillong": (25.5788, 91.8933), "Talcher": (20.9500, 85.2167),
    "Thiruvananthapuram": (8.5241, 76.9366), "Visakhapatnam": (17.6868, 83.2185),
}

# --- Indian AQI sub-index breakpoints (CPCB standard) ---
AQI_BREAKPOINTS = {
    "PM2.5": [(0,30,0,50),(31,60,51,100),(61,90,101,200),(91,120,201,300),(121,250,301,400),(250,500,401,500)],
    "PM10":  [(0,50,0,50),(51,100,51,100),(101,250,101,200),(251,350,201,300),(351,430,301,400),(430,600,401,500)],
    "NO2":   [(0,40,0,50),(41,80,51,100),(81,180,101,200),(181,280,201,300),(281,400,301,400),(400,800,401,500)],
    "SO2":   [(0,40,0,50),(41,80,51,100),(81,380,101,200),(381,800,201,300),(801,1600,301,400),(1600,3200,401,500)],
    "CO":    [(0,1.0,0,50),(1.1,2.0,51,100),(2.1,10,101,200),(10.1,17,201,300),(17.1,34,301,400),(34,60,401,500)],
    "O3":    [(0,50,0,50),(51,100,51,100),(101,168,101,200),(169,208,201,300),(209,748,301,400),(748,1500,401,500)],
}

def calc_sub_index(val, breakpoints):
    for c_lo, c_hi, i_lo, i_hi in breakpoints:
        if c_lo <= val <= c_hi:
            return ((i_hi - i_lo) / (c_hi - c_lo)) * (val - c_lo) + i_lo
    return breakpoints[-1][3]

def calc_indian_aqi(row):
    subs = []
    mapping = {"PM2.5": "PM2.5", "PM10": "PM10", "NO2": "NO2", "SO2": "SO2", "CO": "CO", "O3": "O3"}
    for poll, col in mapping.items():
        if col in row and pd.notna(row[col]) and row[col] > 0:
            subs.append(calc_sub_index(row[col], AQI_BREAKPOINTS[poll]))
    return max(subs) if subs else 0

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

# ── Live data fetching from OpenWeatherMap ──────────────────────────

@st.cache_data(ttl=3600, show_spinner="Fetching live air quality data...")
def fetch_live_data(city, api_key):
    """Fetch last 35 days of hourly pollution data and aggregate to daily."""
    api_key = api_key.strip() if api_key else ""
    if city not in CITY_COORDS:
        return None, f"City '{city}' coordinates not available"
    lat, lon = CITY_COORDS[city]
    now = int(time.time())
    start = now - 70 * 86400  # 70 days back for lag-30 + lookback-30 window

    url = (f"http://api.openweathermap.org/data/2.5/air_pollution/history"
           f"?lat={lat}&lon={lon}&start={start}&end={now}&appid={api_key}")
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 401:
            return None, "Invalid API key. Check your OpenWeatherMap API key."
        if resp.status_code != 200:
            return None, f"API error {resp.status_code}: {resp.text[:200]}"
        data = resp.json()
    except requests.exceptions.RequestException as e:
        return None, f"Network error: {e}"

    if "list" not in data or len(data["list"]) == 0:
        return None, "No data returned from API"

    records = []
    for entry in data["list"]:
        dt = datetime.utcfromtimestamp(entry["dt"])
        comp = entry.get("components", {})
        records.append({
            "datetime": dt,
            "Date": dt.date(),
            "PM2.5": comp.get("pm2_5", 0),
            "PM10": comp.get("pm10", 0),
            "NO": comp.get("no", 0),
            "NO2": comp.get("no2", 0),
            "NH3": comp.get("nh3", 0),
            "CO": comp.get("co", 0) / 1000.0,  # API gives µg/m³, model expects mg/m³
            "SO2": comp.get("so2", 0),
            "O3": comp.get("o3", 0),
        })

    hourly = pd.DataFrame(records)
    daily = hourly.groupby("Date").agg({
        "PM2.5": "mean", "PM10": "mean", "NO": "mean", "NO2": "mean",
        "NH3": "mean", "CO": "mean", "SO2": "mean", "O3": "mean",
    }).reset_index()

    # Derived columns
    daily["NOx"] = daily["NO"] + daily["NO2"]
    daily["Benzene"] = 0.0
    daily["Toluene"] = 0.0
    daily["Xylene"] = 0.0
    daily["City"] = city
    daily["Date"] = pd.to_datetime(daily["Date"])
    daily["AQI"] = daily.apply(calc_indian_aqi, axis=1)
    daily = daily.sort_values("Date").reset_index(drop=True)
    return daily, None

def predict_aqi_from_df(cdf, anchor_date, model, city_scalers, city, scale_cols, aqi_idx):
    """Run prediction from a prepared city DataFrame (works for both live and historical)."""
    if city not in city_scalers:
        return None, f"City '{city}' not found in scaler dict"
    sc = city_scalers[city]

    cdf = cdf.copy()
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

    expected_feats = model.input_shape[-1]
    if X_sc.shape[1] > expected_feats:
        X_sc = X_sc[:, :expected_feats]
    elif X_sc.shape[1] < expected_feats:
        pad = np.zeros((X_sc.shape[0], expected_feats - X_sc.shape[1]), dtype="float32")
        X_sc = np.concatenate([X_sc, pad], axis=1)

    preds = model.predict(X_sc[np.newaxis], verbose=0)[0]

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

def predict_aqi(city, anchor_date, model, city_scalers, scale_cols, aqi_idx, df):
    """Historical prediction wrapper."""
    cdf = df[df["City"] == city].reset_index(drop=True)
    return predict_aqi_from_df(cdf, anchor_date, model, city_scalers, city, scale_cols, aqi_idx)

# ── Render forecast results (shared by both tabs) ──────────────────

def render_forecast(result, city, anchor, hist_df=None):
    """Display forecast cards, bar chart, historical line, and table."""
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
    fig.update_layout(title=f"5-Day AQI Forecast — {city}", yaxis_title="AQI",
                      plot_bgcolor="white", height=380)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns([2, 1])
    with c1:
        if hist_df is not None and len(hist_df) > 0:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=hist_df["Date"], y=hist_df["AQI"], mode="lines",
                                      name="Recent Data", line=dict(color="#4a90d9", width=2)))
            fd = [pd.Timestamp(anchor)+timedelta(days=i+1) for i in range(HORIZON)]
            fig2.add_trace(go.Scatter(x=fd, y=result["AQI"].tolist(), mode="markers+lines",
                                      name="Forecast",
                                      marker=dict(size=10, color=result["Color"].tolist()),
                                      line=dict(dash="dash", color="#ff7043")))
            anchor_ts = pd.Timestamp(anchor)
            fig2.add_vline(x=anchor_ts.timestamp()*1000, line_dash="dash",
                           line_color="gray", annotation_text="Today")
            fig2.update_layout(title="Recent + Forecast", height=300,
                               plot_bgcolor="white", yaxis_title="AQI")
            st.plotly_chart(fig2, use_container_width=True)
    with c2:
        st.dataframe(result[["Day","Date","AQI","Bucket"]], hide_index=True, use_container_width=True)
        st.download_button("Download CSV", result.to_csv(index=False),
                           f"forecast_{city}_{anchor}.csv", "text/csv")

# ══════════════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="AQI Forecast — Live & Historical", page_icon="🌬️", layout="wide")
model = load_model()
city_scalers, scale_cols, aqi_idx = load_scalers()
df = load_data()
CITIES = sorted(df["City"].unique().tolist())
LIVE_CITIES = sorted([c for c in CITIES if c in CITY_COORDS])

# ── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🌬️ AQI Forecast")
    st.markdown("---")
    mode = st.radio("Mode", ["🔴 Live Forecast", "📊 Historical Forecast"], index=0)
    st.markdown("---")

    if mode == "🔴 Live Forecast":
        city = st.selectbox("City", LIVE_CITIES,
                            index=LIVE_CITIES.index("Delhi") if "Delhi" in LIVE_CITIES else 0)
        api_key = st.text_input("OpenWeatherMap API Key", type="password",
                                help="Get free key at openweathermap.org")
    else:
        city = st.selectbox("City", CITIES,
                            index=CITIES.index("Delhi") if "Delhi" in CITIES else 0)
        anchor = st.date_input("Anchor Date", value=pd.Timestamp("2019-10-15").date(),
                               min_value=df["Date"].min().date(), max_value=df["Date"].max().date())
        run_batch = st.checkbox("Forecast all cities (batch)")

    st.markdown("---")
    st.caption("CNN-LSTM-GRU + Bahdanau Attention | 190K params | 86.65% bucket acc")

# ── Live Forecast Tab ──────────────────────────────────────────────
if mode == "🔴 Live Forecast":
    st.title(f"🔴 Live AQI 5-Day Forecast — {city}")
    today = date.today()
    st.caption(f"Today: {today}  |  Forecast: {today + timedelta(1)} → {today + timedelta(5)}")

    if not api_key:
        st.info("👈 Enter your **OpenWeatherMap API key** in the sidebar to enable live forecasting. "
                "Get a free key at [openweathermap.org](https://home.openweathermap.org/users/sign_up)")
        st.stop()

    live_df, err = fetch_live_data(city, api_key)
    if err:
        st.error(f"❌ {err}")
        st.stop()

    # Show current pollutant levels
    latest = live_df.iloc[-1]
    current_aqi = latest["AQI"]
    bk, col = aqi_bucket(current_aqi)
    st.markdown(f"### Current AQI: **{current_aqi:.0f}** — <span style='color: {col}; font-weight: bold;'>{bk}</span>", unsafe_allow_html=True)

    poll_cols = st.columns(4)
    for i, (poll, val) in enumerate([("PM2.5", latest["PM2.5"]), ("PM10", latest["PM10"]),
                                      ("NO2", latest["NO2"]), ("O3", latest["O3"])]):
        with poll_cols[i]:
            st.metric(poll, f"{val:.1f} µg/m³")

    st.markdown("---")

    # Run prediction
    result, pred_err = predict_aqi_from_df(live_df, today, model, city_scalers, city, scale_cols, aqi_idx)
    if pred_err:
        st.error(pred_err)
    else:
        render_forecast(result, city, today, hist_df=live_df)

# ── Historical Forecast Tab ────────────────────────────────────────
else:
    st.title(f"📊 AQI 5-Day Forecast — {city}")
    st.caption(f"Anchor: {anchor}  |  t+1 through t+5")
    result, err = predict_aqi(city, anchor, model, city_scalers, scale_cols, aqi_idx, df)

    if err:
        st.error(err)
    else:
        hist = df[(df["City"]==city)&(df["Date"]<=pd.Timestamp(anchor))].tail(90)
        render_forecast(result, city, anchor, hist_df=hist)

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
        fig3 = px.imshow(hm, color_continuous_scale="RdYlGn_r",
                         title="All-City Forecast Heatmap", aspect="auto", height=600)
        st.plotly_chart(fig3, use_container_width=True)
        st.dataframe(bdf, hide_index=True, use_container_width=True)
        st.download_button("Download Batch CSV", bdf.to_csv(index=False),
                           f"batch_{anchor}.csv", "text/csv")

st.markdown("---")
st.caption("AQI Forecasting | CNN-LSTM-GRU + Bahdanau Attention | Live + Historical Dashboard")
