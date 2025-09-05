# -*- coding: utf-8 -*-
# app_history_horizon_only.py
# Lee SOLO meteo_history.csv y ejecuta la red neuronal sobre ese horizonte.
# No consulta APIs, no reindexa, no "inventa" días. Todo sale del CSV.
# Acepta dos esquemas de columnas:
#   A) date, tmax, tmin, prec [, jd, source, updated_at]
#   B) Fecha, Julian_days, TMAX, TMIN, Prec
#
# Gráficos: EMERREL diario (barras + MA5), EMEAC (%) (curva con banda mín/máx),
# y tabla con emojis de nivel final.

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="PREDWEEM · Solo con meteo_history.csv", layout="wide")

CSV_PATH = Path("meteo_history.csv")

# ======== Red neuronal (estructura compacta y determinística) ========
class PracticalANNModel:
    def __init__(self):
        import numpy as np
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
        # Rango de normalización de entrada [min, max] para [JD, TMAX, TMIN, Prec]
        self.input_min = np.array([1.0, 7.7, -3.5, 0.0], dtype=float)
        self.input_max = np.array([148.0, 38.5, 23.5, 59.9], dtype=float)

    def _tansig(self, x):
        return np.tanh(x)

    def _normalize_input(self, X):
        Xc = np.clip(X, self.input_min, self.input_max)
        return 2 * (Xc - self.input_min) / (self.input_max - self.input_min) - 1

    def _denorm_output(self, y, ymin=-1.0, ymax=1.0):
        # Mapea a [0..1]
        return (y - ymin) / (ymax - ymin)

    def predict(self, X):
        Xn = self._normalize_input(X.astype(float))
        # Capa oculta (20 neuronas), salida 1 neurona
        # Forma: IW.T @ x (4x20) + bias(20,)
        z1 = Xn @ self.IW + self.bias_IW  # (N,20)
        a1 = self._tansig(z1)             # (N,20)
        z2 = a1 @ self.LW + self.bias_out # (N,)
        y  = self._tansig(z2)             # (N,)
        emerrel_01 = self._denorm_output(y)  # (N,) en [0..1]
        # Convertimos a EMEAC acumulado normalizado por un máximo de referencia
        valor_max_emeac = 8.05
        emer_ac = np.cumsum(emerrel_01) / valor_max_emeac  # (N,)
        emerrel_diff = np.diff(emer_ac, prepend=0.0)
        return emerrel_diff, emer_ac

@st.cache_resource
def get_model():
    return PracticalANNModel()

# ======== Carga robusta del CSV (solo filas existentes) ========
def load_history_only(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame(columns=["Fecha","Julian_days","TMAX","TMIN","Prec"])

    # Intentar esquema B primero (por compatibilidad histórica)
    try:
        df = pd.read_csv(csv_path, parse_dates=["Fecha"])
        if {"Fecha","Julian_days","TMAX","TMIN","Prec"}.issubset(df.columns):
            pass
        else:
            raise ValueError("Schema B no coincide; pruebo A")
    except Exception:
        # Esquema A
        df = pd.read_csv(csv_path, parse_dates=["date"])
        if not {"date","tmax","tmin","prec"}.issubset(df.columns):
            # No es A ni B: devuelvo vacío
            return pd.DataFrame(columns=["Fecha","Julian_days","TMAX","TMIN","Prec"])
        df = df.rename(columns={"date":"Fecha","tmax":"TMAX","tmin":"TMIN","prec":"Prec"})
        df["Fecha"] = pd.to_datetime(df["Fecha"]).dt.normalize()
        # Si no hay Julian_days, lo construimos relativo al primer día del CSV
        df = df.sort_values("Fecha").reset_index(drop=True)
        first = df["Fecha"].min()
        df["Julian_days"] = (df["Fecha"] - first).dt.days + 1

    # Coerción numérica y saneo básico
    for c in ["TMAX","TMIN","Prec"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Precipitaciones negativas no (clip 0)
    df["Prec"] = df["Prec"].fillna(0).clip(lower=0)
    df = (df.dropna(subset=["Fecha"])
            .drop_duplicates("Fecha")
            .sort_values("Fecha")
            .reset_index(drop=True))
    return df

# ======== UI ========
st.title("PREDWEEM — Horizonte tal cual en meteo_history.csv (sin API)")
st.caption("La app usa únicamente las filas existentes del CSV. No reindexa ni completa fechas faltantes.")

dfh = load_history_only(CSV_PATH)
if dfh.empty:
    st.error("No hay filas utilizables en meteo_history.csv. Verificá columnas y datos.")
    st.stop()

st.success(f"Horizonte detectado: {dfh['Fecha'].min().date()} → {dfh['Fecha'].max().date()} · {len(dfh)} día(s)")

# ======== Predicción ========
modelo = get_model()
X = dfh[["Julian_days","TMAX","TMIN","Prec"]].to_numpy(float)
emerrel, emeac01 = modelo.predict(X)

pred = pd.DataFrame({
    "Fecha": dfh["Fecha"].to_numpy(),
    "Julian_days": dfh["Julian_days"].to_numpy(),
    "EMERREL(0-1)": emerrel,
    "EMEAC(0-1)": emeac01
})

# --- Robust: asegurar numérico antes de rolling ---
for _c in ["EMERREL(0-1)", "EMEAC(0-1)"]:
    if _c in pred.columns:
        pred[_c] = pd.to_numeric(pred[_c], errors="coerce")
pred["EMERREL(0-1)"] = pred["EMERREL(0-1)"].fillna(0)
pred["EMERREL_MA5"] = pred["EMERREL(0-1)"].rolling(window=5, min_periods=1).mean()
pred["EMEAC(%)"] = (pred["EMEAC(0-1)"] * 100).clip(0, 100)

# Clasificación simple del nivel diario según EMERREL
THR_BAJO_MEDIO = 0.02
THR_MEDIO_ALTO = 0.079
def nivel(v):
    if v < THR_BAJO_MEDIO: return "Bajo"
    elif v <= THR_MEDIO_ALTO: return "Medio"
    else: return "Alto"
pred["Nivel"] = pred["EMERREL(0-1)"].apply(nivel)

# ======== Gráfico EMERREL diario ========
st.subheader("EMERREL diario (barras) + MA5 (línea)")
colores = pred["Nivel"].map({"Bajo":"#2ca02c","Medio":"#ff7f0e","Alto":"#d62728"}).fillna("#808080")
fig1 = go.Figure()
fig1.add_bar(
    x=pred["Fecha"],
    y=pred["EMERREL(0-1)"],
    marker=dict(color=colores),
    customdata=pred["Nivel"],
    hovertemplate="Fecha: %{x|%d-%b-%Y}<br>EMERREL: %{y:.3f}<br>Nivel: %{customdata}<extra></extra>",
    name="EMERREL"
)
fig1.add_scatter(
    x=pred["Fecha"],
    y=pred["EMERREL_MA5"],
    mode="lines",
    name="MA5"
)
# Eje razonable para tu escala
fig1.update_yaxes(range=[0, 0.08])
st.plotly_chart(fig1, use_container_width=True)

# ======== Gráfico EMEAC (%) ========
st.subheader("EMEAC acumulada (%)")
# Banda mín/máx de referencia (denominadores fijos)
EMEAC_MIN_DEN = 5.0
EMEAC_MAX_DEN = 15.0

acc = pred["EMERREL(0-1)"].cumsum()
emeac_min = (acc / EMEAC_MIN_DEN * 100).clip(0, 100)
emeac_max = (acc / EMEAC_MAX_DEN * 100).clip(0, 100)

fig2 = go.Figure()
fig2.add_scatter(x=pred["Fecha"], y=emeac_min, mode="lines", line=dict(width=0), name="EMEAC mín")
fig2.add_scatter(x=pred["Fecha"], y=emeac_max, mode="lines", line=dict(width=0), fill="tonexty", name="EMEAC máx")
fig2.add_scatter(x=pred["Fecha"], y=pred["EMEAC(%)"], mode="lines", name="EMEAC (%) (modelo)")
fig2.update_yaxes(range=[0, 100])
st.plotly_chart(fig2, use_container_width=True)

# ======== Tabla ========
st.subheader("Resultados diarios (horizonte del CSV)")
iconos = {"Bajo":"🟢 Bajo", "Medio":"🟠 Medio", "Alto":"🔴 Alto"}
tabla = pred[["Fecha","Julian_days","EMERREL(0-1)","EMERREL_MA5","EMEAC(%)","Nivel"]].copy()
tabla["Nivel"] = tabla["Nivel"].map(iconos)
st.dataframe(tabla, use_container_width=True)

st.download_button(
    "Descargar resultados (CSV)",
    tabla.to_csv(index=False).encode("utf-8"),
    "resultados_predweem_horizonte_history.csv",
    "text/csv"
)