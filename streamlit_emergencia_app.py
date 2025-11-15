# -*- coding: utf-8 -*-
# app_history_horizon_only.py
# PREDWEEM · Solo con meteo_history.csv
# Versión simplificada · SOLO EMERREL diaria

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ------------------ CONFIGURACIÓN ------------------
st.set_page_config(page_title="PREDWEEM · Solo meteo_history.csv", layout="wide")
st.title("🌱 PREDWEEM — EUPHORBIA DAVIDII · Bahía Blanca 2025")
st.caption("Usa únicamente las filas disponibles en meteo_history.csv (sin completar ni reindexar).")

CSV_PATH = Path("meteo_history.csv")

# ------------------ SIDEBAR ------------------
st.sidebar.header("⚙️ Parámetros del modelo")
st.sidebar.caption("Versión reducida sin EMEAC (solo EMERREL).")


# ------------------ MODELO (ANN) ------------------
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
        ])
        self.bias_IW = np.array([
            7.229977, -2.428431, 2.973525, 1.956296, -1.155897, 0.907013, 0.231416, 5.258464, 3.284862, 5.474901,
            2.971978, 4.302273, 1.650572, -1.768043, -7.693806, -0.010850, 1.497102, -2.799158, -2.366918, -9.754413
        ])
        self.LW = np.array([
            5.508609, -21.909052, -10.648533, -2.939799, 8.192068, -2.157424, -3.373238, -5.932938, -2.680237,
            -3.399422, 5.870659, -1.720078, 7.134293, 3.227154, -5.039080, -10.872101, -6.569051, -8.455429,
            2.703778, 4.776029
        ])
        self.bias_out = -5.394722

        self.input_min = np.array([1.0, 7.7, -3.5, 0.0])
        self.input_max = np.array([148.0, 38.5, 23.5, 59.9])

    def _tansig(self, x):
        return np.tanh(x)

    def _normalize_input(self, X):
        Xc = np.clip(X, self.input_min, self.input_max)
        return 2 * (Xc - self.input_min) / (self.input_max - self.input_min) - 1

    def _denorm_output(self, y, ymin=-1.0, ymax=1.0):
        return (y - ymin) / (ymax - ymin)

    def predict(self, X):
        Xn = self._normalize_input(X.astype(float))
        z1 = Xn @ self.IW + self.bias_IW
        a1 = self._tansig(z1)
        z2 = a1 @ self.LW + self.bias_out
        y = self._tansig(z2)
        emerrel = self._denorm_output(y).clip(0, 1)
        return emerrel


def get_model():
    return PracticalANNModel()


# ------------------ COLORACIÓN ------------------
HEX_GREEN, HEX_YELLOW, HEX_RED = "#00A651", "#FFC000", "#E53935"
COLOR_MAP_HEX = {"Bajo": HEX_GREEN, "Medio": HEX_YELLOW, "Alto": HEX_RED}
MAP_NIVEL_ICONO = {"Bajo": "🟢 Bajo", "Medio": "🟡 Medio", "Alto": "🔴 Alto"}

def rgba(hex_color: str, alpha: float):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ------------------ CARGA DEL CSV ------------------
def load_history_only(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame(columns=["Fecha","Julian_days","TMAX","TMIN","Prec"])

    try:
        df = pd.read_csv(csv_path, parse_dates=["Fecha"])
        if not {"Fecha","Julian_days","TMAX","TMIN","Prec"}.issubset(df.columns):
            raise ValueError
    except Exception:
        df = pd.read_csv(csv_path, parse_dates=["date"])
        df = df.rename(columns={"date":"Fecha","tmax":"TMAX","tmin":"TMIN","prec":"Prec"})
        first = df["Fecha"].min()
        df["Julian_days"] = (df["Fecha"] - first).dt.days + 1

    for c in ["TMAX","TMIN","Prec"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["Prec"] = df["Prec"].fillna(0).clip(lower=0)

    df = df.dropna(subset=["Fecha"]).drop_duplicates("Fecha").sort_values("Fecha").reset_index(drop=True)
    return df


# ------------------ PROCESAMIENTO ------------------
dfh = load_history_only(CSV_PATH)

# Filtro de horizonte fijo
fecha_ini_fija = pd.to_datetime("2025-09-04")
dfh = dfh[dfh["Fecha"] >= fecha_ini_fija].reset_index(drop=True)

if dfh.empty:
    st.error("No hay filas utilizables en meteo_history.csv.")
    st.stop()

base = dfh["Fecha"].min()
dfh["Julian_days"] = (dfh["Fecha"] - base).dt.days + 1

st.success(f"Horizonte detectado: {dfh['Fecha'].min().date()} → {dfh['Fecha'].max().date()} · {len(dfh)} día(s)")


# ------------------ PREDICCIÓN ------------------
modelo = get_model()
X = dfh[["Julian_days","TMAX","TMIN","Prec"]].to_numpy(float)
emerrel = modelo.predict(X)

pred = pd.DataFrame({
    "Fecha": dfh["Fecha"],
    "Julian_days": dfh["Julian_days"],
    "EMERREL": emerrel
})
pred["EMERREL_MA5"] = pred["EMERREL"].rolling(5, min_periods=1).mean()


# ------------------ CLASIFICACIÓN ------------------
THR_BAJO_MEDIO, THR_MEDIO_ALTO = 0.02, 0.07
def nivel(v):
    return "Bajo" if v < THR_BAJO_MEDIO else ("Medio" if v <= THR_MEDIO_ALTO else "Alto")

pred["Nivel"] = pred["EMERREL"].apply(nivel)


# ------------------ GRÁFICO EMERREL ------------------
st.subheader("🌾 EMERGENCIA RELATIVA (EMERREL)")
bar_colors = pred["Nivel"].map(COLOR_MAP_HEX)

fig1 = go.Figure()
fig1.add_bar(
    x=pred["Fecha"], y=pred["EMERREL"],
    marker=dict(color=bar_colors),
    customdata=pred["Nivel"],
    hovertemplate="Fecha: %{x|%d-%b-%Y}<br>EMERREL: %{y:.3f}<br>Nivel: %{customdata}<extra></extra>"
)

fig1.add_scatter(
    x=pred["Fecha"], y=pred["EMERREL_MA5"],
    mode="lines", line=dict(color="black", width=2),
    name="MA5"
)

fig1.update_yaxes(range=[0, 0.15])
fig1.update_layout(xaxis_title="Fecha", yaxis_title="EMERREL (0-1)",
                   hovermode="x unified", height=560)

st.plotly_chart(fig1, use_container_width=True)


# ------------------ TABLA ------------------
st.subheader("📋 Resultados diarios")

tabla = pred.copy()
tabla["Nivel"] = tabla["Nivel"].map(MAP_NIVEL_ICONO)

st.dataframe(
    tabla[["Fecha","Julian_days","EMERREL","EMERREL_MA5","Nivel"]],
    use_container_width=True
)

st.download_button(
    "💾 Descargar resultados (CSV)",
    tabla.to_csv(index=False).encode("utf-8"),
    "resultados_predweem_sin_emeac.csv",
    "text/csv"
)

