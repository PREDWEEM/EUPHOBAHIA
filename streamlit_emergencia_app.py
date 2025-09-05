# -*- coding: utf-8 -*-
# streamlit_emergencia_app.py — MODO HISTÓRICO EXACTO (sin relleno de fechas)
# Reglas:
# - Usa EXCLUSIVAMENTE las filas existentes de meteo_history.csv.
# - NO crea filas nuevas ni extiende hasta ninguna fecha fija.
# - Ejecuta el modelo sólo sobre esas filas; no hay "fallback" de resultados.

import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# Plotly (opcional)
try:
    import plotly.graph_objects as go
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

# Intentar importar el modelo del usuario
MODEL_OK = False
try:
    from modelo_emerrel import ejecutar_modelo  # tu función
    MODEL_OK = True
except Exception:
    MODEL_OK = False

st.set_page_config(
    page_title="PREDWEEM — Histórico exacto (sin relleno)",
    layout="wide",
    menu_items={"Get help": None, "Report a bug": None, "About": None},
)
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header [data-testid="stToolbar"] {display: none;}
    </style>
    """,
    unsafe_allow_html=True,
)

CSV_PATH_DEFAULT = Path("meteo_history.csv")
ICON_MAP = {"Bajo": "🟢 Bajo", "Medio": "🟠 Medio", "Alto": "🔴 Alto"}

# Sidebar
st.sidebar.header("Fuente única: meteo_history.csv (sin API)")
csv_file = st.sidebar.text_input("Ruta del CSV histórico", str(CSV_PATH_DEFAULT))
filter_from_today = st.sidebar.checkbox("Mostrar sólo desde hoy", value=False)
btn_reload = st.sidebar.button("🔄 Recargar archivo")

@st.cache_data(show_spinner=False)
def load_history(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path.resolve()}")
    df = pd.read_csv(path, parse_dates=["date"], dayfirst=False)
    df.columns = [c.lower() for c in df.columns]
    # asegurar columnas mínimas (si faltan, se crean NA; no se rellenan)
    for c in ["tmax","tmin","prec"]:
        if c not in df.columns:
            df[c] = np.nan
    keep = [c for c in ["date","tmax","tmin","prec","source","updated_at","jd"] if c in df.columns]
    df = df[keep].copy().sort_values("date")
    return df

try:
    if btn_reload:
        st.cache_data.clear()
    raw = load_history(csv_file)
    st.success(f"Histórico cargado: {len(raw)} filas · rango {raw['date'].min().date()} → {raw['date'].max().date()}")
except Exception as e:
    st.error(f"Error al cargar histórico: {e}")
    st.stop()

# Filtro "desde hoy" opcional (no se crean fechas, sólo se filtran)
if filter_from_today:
    raw = raw[raw["date"].dt.date >= pd.Timestamp.today().date()].copy()

# Prepara entrada para el modelo SIN rellenar valores faltantes
df_in = raw.copy()
# Asegurar tipo numérico (sin imputar)
for c in ["tmax","tmin","prec"]:
    df_in[c] = pd.to_numeric(df_in[c], errors="coerce")

# Ejecutar modelo si está disponible (sin fallback)
model_ok = False
res = None
if MODEL_OK:
    try:
        # Ajustá columnas a lo que exija tu modelo
        modelo_in = df_in[["date","tmax","tmin","prec"]].copy()
        # incluir jd si tu modelo lo usa
        if "jd" in df_in.columns:
            modelo_in["jd"] = df_in["jd"]
        res = ejecutar_modelo(modelo_in)
        if isinstance(res, dict):
            res = pd.DataFrame(res)
        if "date" not in res.columns:
            res = res.reset_index().rename(columns={"index":"date"})
        res["date"] = pd.to_datetime(res["date"])
        model_ok = True
        st.success("✅ Modelo ejecutado sobre las filas existentes del CSV (sin relleno).")
    except Exception as e:
        st.error(f"Fallo del modelo: {e}")
else:
    st.info("No se encontró `ejecutar_modelo`. Se mostrará sólo la serie meteorológica.")

# Merge sólo por fechas que EXISTEN en resultados (no se generan huecos nuevos)
if model_ok and res is not None:
    pred_cols = [c for c in res.columns if c.lower() in {"emerrel","emeac","nivel_final"} or c in {"EMERREL","EMEAC","nivel_final"}]
    # Normalizar nombres comunes
    cols_map = {c: ("EMERREL" if c.lower()=="emerrel" else "EMEAC" if c.lower()=="emeac" else c) for c in pred_cols}
    res = res.rename(columns=cols_map)
    out = pd.merge(df_in, res[["date"] + list(cols_map.values())], on="date", how="inner")
else:
    out = df_in.copy()

# nivel_icon si existe nivel_final
if "nivel_final" in out.columns:
    out["nivel_icon"] = out["nivel_final"].map(ICON_MAP).fillna("")

# Métricas de preparación
rows_total = len(df_in)
rows_with_complete_inputs = df_in.dropna(subset=["tmax","tmin","prec"]).shape[0]
st.caption(f"Filas en CSV mostradas: {rows_total} · Filas con meteo completa: {rows_with_complete_inputs}")

# Gráfico
st.subheader("Serie diaria (exacta, sin creación de fechas)")
if PLOTLY_OK:
    fig = go.Figure()
    if "EMERREL" in out.columns:
        fig.add_trace(go.Scatter(x=out["date"], y=out["EMERREL"], mode="lines", name="EMERREL"))
    if "EMEAC" in out.columns:
        fig.add_trace(go.Scatter(x=out["date"], y=out["EMEAC"], mode="lines", name="EMEAC (0–100)", yaxis="y2"))
    fig.update_layout(
        height=460, margin=dict(l=10,r=10,t=30,b=10),
        xaxis=dict(title="Fecha"),
        yaxis=dict(title="EMERREL", rangemode="tozero"),
        yaxis2=dict(title="EMEAC (%)", overlaying="y", side="right", rangemode="tozero"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Instalá `plotly` para ver el gráfico interactivo.")

# Tabla
st.subheader("Tabla (sin relleno de fechas ni valores)")
cols = ["date","tmax","tmin","prec"]
for c in ["EMERREL","EMEAC","nivel_final","nivel_icon"]:
    if c in out.columns: cols.append(c)
tbl = out[cols].copy()
for c, nd in [("EMERREL",4),("EMEAC",1)]:
    if c in tbl.columns:
        tbl[c] = tbl[c].map(lambda x: None if pd.isna(x) else round(float(x), nd))
st.dataframe(tbl, use_container_width=True)

# Descarga
@st.cache_data
def to_csv_bytes(dfout: pd.DataFrame) -> bytes:
    return dfout.to_csv(index=False).encode("utf-8")

st.download_button(
    "⬇️ Descargar resultados (CSV)",
    data=to_csv_bytes(out),
    file_name="resultados_historico_exacto.csv",
    mime="text/csv",
)
