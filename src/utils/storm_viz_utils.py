import os
import numpy as np
import joblib
import pandas as pd
from datetime import timedelta

LOG_COLS = {2: "area_km2", 4: "speed_kmh", 5: "growth"}

PATHS = {
    "future"  : "data/prediction/predictions.npy",
    "seq"     : "data/lstm_dataset/X_sequences.npy",
    "scaler"  : "data/lstm_dataset/scaler.save",
    "metrics" : "data/metrics/evaluation_metrics.npy",
    "tracks"  : "data/storm_tracks/storm_tracks.csv",
    "bob_idx" : "data/lstm_dataset/bob_sequence_idx.npy",
    "output"  : "output/gru_predict_img",
}


def load_all():
    preds  = np.load(PATHS["future"])
    X      = np.load(PATHS["seq"])
    scaler = joblib.load(PATHS["scaler"])
    return preds, X, scaler


def inverse_transform(scaler, arr):
    real = scaler.inverse_transform(arr)
    for col in LOG_COLS:
        real[:, col] = np.expm1(real[:, col])
    return real


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def calculate_direction(lat1, lon1, lat2, lon2):
    v = "N" if lat2 - lat1 > 0 else "S"
    h = "E" if lon2 - lon1 > 0 else "W"
    return f"{v}{h}"


def safe_format(value):
    return f"{value:.2f}" if value is not None else "N/A"


def load_metrics():
    try:
        m = np.load(PATHS["metrics"], allow_pickle=True).item()
        return m.get("MAE"), m.get("RMSE"), m.get("track_error_mean")
    except Exception:
        return None, None, None


def find_bob_sequence(X, scaler):
    if os.path.exists(PATHS["bob_idx"]):
        idx = int(np.load(PATHS["bob_idx"])[0])
        seq = scaler.inverse_transform(X[idx])
        print(f"  Using saved BoB sequence: {idx} "
              f"({seq[-1,0]:.2f}N, {seq[-1,1]:.2f}E)")
        return idx

    for i in range(len(X) - 1, -1, -1):
        seq = scaler.inverse_transform(X[i])
        lat, lon = seq[-1, 0], seq[-1, 1]
        if 5 <= lat <= 25 and 65 <= lon <= 100:
            print(f"  Found BoB sequence: {i} ({lat:.2f}N, {lon:.2f}E)")
            np.save(PATHS["bob_idx"], np.array([i]))
            return i

    return len(X) - 1


def get_history_and_future(X, preds, scaler, idx):
    history = inverse_transform(scaler, X[idx])
    if preds.shape[0] > 100:
        future = inverse_transform(scaler, [preds[idx]])
    else:
        future = inverse_transform(scaler, preds)
    return history, future


def get_timestamps(seq_idx, scaler, X):
    try:
        df         = pd.read_csv(PATHS["tracks"])
        seq_real   = scaler.inverse_transform(X[seq_idx])
        target_lat = round(float(seq_real[-1, 0]), 2)
        target_lon = round(float(seq_real[-1, 1]), 2)

        match = df[
            (df["centroid_lat"].round(2) == target_lat) &
            (df["centroid_lon"].round(2) == target_lon)
        ]

        if len(match) > 0:
            last_obs  = pd.to_datetime(match.iloc[0]["time"])
            storm_id  = match.iloc[0]["storm_id"]
            seq_len   = X[seq_idx].shape[0]
            obs_times = [
                last_obs - timedelta(minutes=30 * (seq_len - 1 - i))
                for i in range(seq_len)
            ]
            return obs_times, last_obs + timedelta(minutes=30), int(storm_id)

    except Exception as e:
        print(f"  Timestamps unavailable: {e}")

    return None, None, None


def tb_color(tb, tb_min, tb_max):
    t = (tb - tb_min) / (tb_max - tb_min + 1e-6)
    r = int(255 * t)
    g = int(100 * (1 - abs(t - 0.5) * 2))
    b = int(255 * (1 - t))
    return f"rgb({r},{g},{b})"


def make_geo_layout(domain=None):
    base = dict(
        showland=True,        landcolor="#e8dfc8",
        showocean=True,       oceancolor="#b8d4e8",
        showcoastlines=True,  coastlinecolor="#444",
        coastlinewidth=1.5,
        showlakes=True,       lakecolor="#b8d4e8",
        showcountries=True,   countrycolor="#666",
        countrywidth=0.8,
        showframe=True,       framecolor="#888",
        lataxis=dict(range=[-10, 35], showgrid=True,
                     gridcolor="rgba(100,100,100,0.25)", dtick=5),
        lonaxis=dict(range=[55, 110], showgrid=True,
                     gridcolor="rgba(100,100,100,0.25)", dtick=5),
        projection_type="mercator",
        bgcolor="#b8d4e8",
        resolution=50,
    )
    if domain:
        base["domain"] = domain
    return base


def make_zoom_buttons(all_lats, all_lons, lat_pad, lon_pad, x=0.0, y=0.99):
    clat = sum(all_lats) / len(all_lats)
    clon = sum(all_lons) / len(all_lons)
    return dict(
        type="buttons", showactive=False,
        x=x, y=y, xanchor="left", yanchor="top",
        bgcolor="white", bordercolor="#ccc", borderwidth=1,
        buttons=[
            dict(label="+ Zoom in", method="relayout",
                 args=[{"geo.lataxis.range": [clat-3,   clat+3],
                        "geo.lonaxis.range": [clon-3,   clon+3]}]),
            dict(label="++ Close up", method="relayout",
                 args=[{"geo.lataxis.range": [clat-1.5, clat+1.5],
                        "geo.lonaxis.range": [clon-1.5, clon+1.5]}]),
            dict(label="Reset view", method="relayout",
                 args=[{"geo.lataxis.range": [-10, 35],
                        "geo.lonaxis.range": [55, 110]}]),
            dict(label="Fit track", method="relayout",
                 args=[{"geo.lataxis.range": [min(all_lats)-lat_pad,
                                              max(all_lats)+lat_pad],
                        "geo.lonaxis.range": [min(all_lons)-lon_pad,
                                              max(all_lons)+lon_pad]}]),
        ],
    )