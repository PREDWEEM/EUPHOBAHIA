# -*- coding: utf-8 -*-
# streamlit_emergencia_app_FIXED_only_history.py
# Requisitos:
# - Lee SOLO meteo_history.csv (sin API)
# - Acepta dos esquemas de columnas:
#     A) date,tmax,tmin,prec[,jd,source,updated_at]
#     B) Fecha,Julian_days,TMAX,TMIN,Prec
# - NO reindexa ni "inventa" fechas faltantes.
# - Ventana de gráficos fija 2025-09-01 → 2026-01-01, pero se dibujan solo fechas con datos.
# - EMERREL eje 0–0.08; EMEAC 0–100 %; incluye MA5 y tabla con iconos.

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ================== Config ==================
st.set_page_config(page_title="Predicción de Emergencia Agrícola (solo history)", layout="wide")
HISTORY_PATH = Path("meteo_history.csv")

VENTANA_MIN = pd.Timestamp("2025-09-01")
VENTANA_MAX = pd.Timestamp("2026-01-01")
BASE = VENTANA_MIN

THR_BAJO_MEDIO = 0.02
THR_MEDIO_ALTO = 0.079
COLOR_MAP = {"Bajo": "#2ca02c", "Medio": "#ff7f0e", "Alto": "#d62728"}
COLOR_FALLBACK = "#808080"

EMEAC_MIN_DEN = 5.0
EMEAC_MAX_DEN = 15.0

# ================== ANN (igual interfaz que la app anterior) ==================
import numpy as np
class PracticalANNModel:
    def __init__(self):
        self.IW = np.array([
            [-2.924160, -7.896739, -0.977000, 0.554961, 9.510761, 8.739410, 10.592497, 21.705275, -2.532038, 7.847811,
             -3.907758, 13.933289, 3.727601, 3.751941, 0.639185, -0.758034, 1.556183, 10.458917, -1.343551, -14.721089],
            [0.115434, 0.615363, -0.241457, 5.478775, -26.598709, -2.316081, 0.545053, -2.924576, -14.629911, -8.916969,
             3.516110, -6.315180, -0.005914, 10.801424, 4.928928, 1.158809, 4.394316, -23.519282, 2.694073, 3.387557],
            [6.210673, -0.666815, 2.923249, -8.329875, 7.029798, 1.202168, -4.650263, 2.243358, 22.006945, 5.118664,
             1.901176, -6.076520, 0.239450, -6.862627, -7.592373, 1.422826, -2.575074, 5.302610, -6.379549, -14.810670],
            [10.220671, 2.665316, 4.119266, 5.812964, -3.848171, 1.472373, -4.829068, -7.422444, 0.862384, 0.001028,
             0.853059, 2.953289, 1.403689, -3.040909, -6.946802, -1.799923, 0.994357, -5.551789, -0.764891, 5.520776]
        ], dtype=float)
        self.bias_IW = np.array([
            7.229977, -2.428431, 2.973525, 1.956296, -1.155897, 0.907013, 0.231416, 5.258464, 3.284862, 5.474901,
            2.971978, 4.302273, 1.650572, -1.768043, -7.693806, -0.010850, 1.497102, -2.799158, -2.366918, -9.754413
        ], dtype=float)
        self.LW = np.array([
            5.508609, -21.909052, -10.648533, -2.939799, 8.192068, -2.157424, -3.373238, -5.932938, -2.680237,
            -3.399422, 5.870659, -1.720078, 7.134293, 3.227154, -5.039080, -10.872101, -6.569051, -8.455429,
            2.703778, 4.776029
        ], dtype=float)
        self.bias_out = -5.394722
        self.input_min = np.array([1.0, 7.7, -3.5, 0.0], dtype=float)
        self.input_max = np.array([148.0, 38.5, 23.5, 59.9], dtype=float)

    def tansig(self, x): return np.tanh(x)
    def normalize_input(self, X_real):
        Xc = np.clip(X_real, self.input_min, self.input_max)
        return 2 * (Xc - self.input_min) / (self.input_max - self.input_min) - 1
    def desnormalize_output(self, y_norm, ymin=-1.0, ymax=1.0):
        return (y_norm - ymin) / (ymax - ymin)
    def _predict_single(self, x_norm):
        z1 = self.IW.T @ x_norm + self.bias_IW
        a1 = self.tansig(z1)
        z2 = self.LW @ a1 + self.bias_out
        return self.tansig(z2)
    def predict(self, X_real):
        X_norm = self.normalize_input(X_real.astype(float))
        emerrel_pred = np.array([self._predict_single(x) for x in X_norm], dtype=float)
        emerrel_desnorm = self.desnormalize_output(emerrel_pred)
        emerrel_cumsum = np.cumsum(emerrel_desnorm)
        valor_max_emeac = 8.05
        emer_ac = emerrel_cumsum / valor_max_emeac
        emerrel_diff = np.diff(emer_ac, prepend=0.0)

        def clasificar(v):
            if v < THR_BAJO_MEDIO: return "Bajo"
            elif v <= THR_MEDIO_ALTO: return "Medio"
            else: return "Alto"
        riesgo = np.array([clasificar(v) for v in emerrel_diff], dtype=object)
        return pd.DataFrame({"EMERREL(0-1)": emerrel_diff, "Nivel_Emergencia_relativa": riesgo})

@st.cache_resource
def get_model():
    return PracticalANNModel()

modelo = get_model()

# ================== Load history (ONLY) ==================
def load_history_only(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["Fecha","Julian_days","TMAX","TMIN","Prec"])

    # Intentar ambos esquemas
    try:
        df = pd.read_csv(path, parse_dates=["Fecha"])
        if {"Fecha","Julian_days","TMAX","TMIN","Prec"}.issubset(df.columns):
            df = df.copy()
        else:
            raise Exception("Schema B not present")
    except Exception:
        df = pd.read_csv(path, parse_dates=["date"])
        if "date" not in df.columns or "tmax" not in df.columns or "tmin" not in df.columns or "prec" not in df.columns:
            return pd.DataFrame(columns=["Fecha","Julian_days","TMAX","TMIN","Prec"])
        df = df.rename(columns={"date":"Fecha","tmax":"TMAX","tmin":"TMIN","prec":"Prec"})
        # Si no hay Julian_days lo calculamos
        df["Fecha"] = pd.to_datetime(df["Fecha"]).dt.normalize()
        df["Julian_days"] = (df["Fecha"] - BASE).dt.days + 1

    # Sanitizar y filtrar ventana, sin reindex
    for c in ["TMAX","TMIN","Prec"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Prec"] = df["Prec"].fillna(0).clip(lower=0)
    df = (df.dropna(subset=["Fecha"])
            .drop_duplicates("Fecha")
            .sort_values("Fecha")
            .reset_index(drop=True))
    m_vis = (df["Fecha"] >= VENTANA_MIN) & (df["Fecha"] <= VENTANA_MAX)
    return df.loc[m_vis].copy()

# ================== UI ==================
st.title("Predicción de Emergencia Agrícola — Solo con meteo_history.csv")
st.caption(f"Ventana fija para gráficos: {VENTANA_MIN.date()} → {VENTANA_MAX.date()} (solo fechas existentes).")

umbral_usuario = st.sidebar.number_input(
    "Umbral ajustable de EMEAC para 100%", 5.0, 15.0, 14.0, 0.01, format="%.2f"
)

dfh = load_history_only(HISTORY_PATH)
if dfh.empty:
    st.error("meteo_history.csv no tiene filas utilizables en la ventana definida.")
    st.stop()

st.success(f"History detectado: {dfh['Fecha'].min().date()} → {dfh['Fecha'].max().date()} · {len(dfh)} día(s)")

# ====== Modelo ======
X_real = dfh[["Julian_days","TMAX","TMIN","Prec"]].to_numpy(float)
fechas = dfh["Fecha"]
pred = modelo.predict(X_real)

pred["Fecha"] = fechas
pred["Julian_days"] = dfh["Julian_days"].to_numpy()
pred["EMERREL acumulado"] = pred["EMERREL(0-1)"].cumsum()

# Bandas EMEAC
pred["EMEAC (0-1) - mínimo"]    = pred["EMERREL acumulado"] / EMEAC_MIN_DEN
pred["EMEAC (0-1) - máximo"]    = pred["EMERREL acumulado"] / EMEAC_MAX_DEN
pred["EMEAC (0-1) - ajustable"] = pred["EMERREL acumulado"] / umbral_usuario
for col in ["EMEAC (0-1) - mínimo","EMEAC (0-1) - máximo","EMEAC (0-1) - ajustable"]:
    pred[col.replace("(0-1)","(%)")] = (pred[col]*100).clip(0,100)

# Forzar "Bajo" cuando EMEAC ajustable < 10%
pred.loc[pred["EMEAC (%) - ajustable"] < 10.0, "Nivel_Emergencia_relativa"] = "Bajo"

pred["EMERREL_MA5"] = pred["EMERREL(0-1)"].rolling(5, 1).mean()

# ====== Figuras ======
st.subheader("EMERGENCIA RELATIVA DIARIA")
colores = pred["Nivel_Emergencia_relativa"].map(COLOR_MAP).fillna(COLOR_FALLBACK).tolist()
fig_er = go.Figure()
fig_er.add_bar(
    x=pred["Fecha"],
    y=pred["EMERREL(0-1)"],
    marker=dict(color=colores),
    customdata=pred["Nivel_Emergencia_relativa"],
    hovertemplate="Fecha: %{x|%d-%b-%Y}<br>EMERREL: %{y:.3f}<br>Nivel: %{customdata}<extra></extra>",
    name="EMERREL"
)
fig_er.add_scatter(x=pred["Fecha"], y=pred["EMERREL_MA5"], mode="lines", name="MA5")
fig_er.update_xaxes(range=[str(VENTANA_MIN.date()), str(VENTANA_MAX.date())], dtick="M1", tickformat="%b")
fig_er.update_yaxes(range=[0, 0.08])
st.plotly_chart(fig_er, use_container_width=True)

st.subheader("EMERGENCIA ACUMULADA DIARIA")
fig_acc = go.Figure()
fig_acc.add_scatter(x=pred["Fecha"], y=pred["EMEAC (%) - mínimo"], mode="lines", line=dict(width=0), name="EMEAC mín")
fig_acc.add_scatter(x=pred["Fecha"], y=pred["EMEAC (%) - máximo"], mode="lines", line=dict(width=0), fill="tonexty", name="EMEAC máx")
fig_acc.add_scatter(x=pred["Fecha"], y=pred["EMEAC (%) - ajustable"], mode="lines", line=dict(width=2.5), name=f"Ajustable /{umbral_usuario:.2f}")
fig_acc.update_yaxes(range=[0, 100])
fig_acc.update_xaxes(range=[str(VENTANA_MIN.date()), str(VENTANA_MAX.date())], dtick="M1", tickformat="%b")
st.plotly_chart(fig_acc, use_container_width=True)

# ====== Tabla ======
st.subheader("Resultados (solo fechas existentes)")
tabla = pred[["Fecha","Julian_days","EMERREL(0-1)","EMERREL_MA5","EMEAC (%) - ajustable","Nivel_Emergencia_relativa"]].copy()
tabla = tabla.rename(columns={"EMEAC (%) - ajustable":"EMEAC (%)","Nivel_Emergencia_relativa":"Nivel"})
iconos = {"Bajo":"🟢 Bajo","Medio":"🟠 Medio","Alto":"🔴 Alto"}
tabla["Nivel"] = tabla["Nivel"].map(iconos)
st.dataframe(tabla, use_container_width=True)

st.download_button(
    "Descargar CSV",
    tabla.to_csv(index=False).encode("utf-8"),
    "resultados_only_history.csv",
    "text/csv"
)