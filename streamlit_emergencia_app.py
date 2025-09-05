# -*- coding: utf-8 -*-
# streamlit_emergencia_app.py — HISTÓRICO ESTRICTO SIN API + SIN HUECOS (hoy → 2026-01-01)
# - Usa SOLO meteo_history.csv
# - Garantiza continuidad diaria desde "hoy" hasta 2026-01-01 (exclusivo)
# - Ejecuta ejecutar_modelo() si está disponible; si no, usa un fallback neutro (0) para no dejar huecos
# - Si el modelo devuelve fechas/NaN incompletos, reindexa e interpola/ffill para evitar faltantes

import os
from pathlib import Path
from datetime import date, datetime, timedelta

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
    from modelo_emerrel import ejecutar_modelo  # debe aceptar un DF con meteo
    MODEL_OK = True
except Exception:
    MODEL_OK = False

st.set_page_config(
    page_title="PREDWEEM — Resultados sin huecos (desde meteo_history.csv)",
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

END_DATE_EXCLUSIVE = date(2026, 1, 1)
CSV_PATH_DEFAULT = Path("meteo_history.csv")

ICON_MAP = {"Bajo": "🟢 Bajo", "Medio": "🟠 Medio", "Alto": "🔴 Alto"}

# Sidebar
st.sidebar.header("Fuente: meteo_history.csv (sin API)")
csv_file = st.sidebar.text_input("Ruta del CSV histórico", str(CSV_PATH_DEFAULT))
btn_reload = st.sidebar.button("🔄 Recargar archivo")

@st.cache_data(show_spinner=False)
def load_history(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path.resolve()}")
    df = pd.read_csv(path, parse_dates=["date"], dayfirst=False)
    df.columns = [c.lower() for c in df.columns]
    for c in ["tmax","tmin","prec"]:
        if c not in df.columns:
            df[c] = np.nan
    keep = [c for c in ["date","tmax","tmin","prec","source","updated_at","jd"] if c in df.columns]
    return df[keep].copy()

try:
    if btn_reload:
        st.cache_data.clear()
    raw = load_history(csv_file)
    st.success(f"Histórico cargado: {len(raw)} filas")
except Exception as e:
    st.error(f"Error al cargar histórico: {e}")
    st.stop()

def ensure_continuous_window_from_today(df: pd.DataFrame, end_exclusive: date) -> pd.DataFrame:
    today = date.today()
    # Limitamos el DF a [today, end)
    mask = (df["date"].dt.date >= today) & (df["date"].dt.date < end_exclusive)
    dfw = df.loc[mask].copy()

    # Reindex continuo diario
    idx = pd.date_range(today, end_exclusive - timedelta(days=1), freq="D")
    dfw = dfw.set_index("date").sort_index()
    dfw = dfw[~dfw.index.duplicated(keep="last")]
    dfw = dfw.reindex(idx)

    # Rellenos conservadores para meteo (asegura no NaN)
    for c in ["tmax","tmin"]:
        dfw[c] = pd.to_numeric(dfw[c], errors="coerce").ffill()
    dfw["prec"] = pd.to_numeric(dfw["prec"], errors="coerce").fillna(0.0)

    # Metadatos mínimos
    if "source" not in dfw.columns:
        dfw["source"] = "forecast"
    if "updated_at" not in dfw.columns:
        dfw["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Auxiliares
    dfw["date"] = dfw.index
    dfw["jd"] = dfw["date"].dt.dayofyear
    return dfw.reset_index(drop=True)

df = ensure_continuous_window_from_today(raw, END_DATE_EXCLUSIVE)

st.caption(f"Ventana efectiva (sin huecos): {df['date'].min().date()} → {df['date'].max().date()}  ·  Filas: {len(df)}")

# Ejecutar el modelo (o fallback)
def run_model_or_fallback(df_in: pd.DataFrame) -> pd.DataFrame:
    if MODEL_OK:
        try:
            res = ejecutar_modelo(df_in[["date","tmax","tmin","prec","jd"]].copy())
            # Normalizar a DF con 'date'
            if isinstance(res, dict):
                res = pd.DataFrame(res)
            if "date" not in res.columns:
                res = res.reset_index().rename(columns={"index":"date"})
            res["date"] = pd.to_datetime(res["date"])
            return res
        except Exception as e:
            st.warning(f"No se pudo ejecutar el modelo: {e}. Se usará fallback neutro para no dejar huecos.")
    # Fallback neutro: EMERREL=0, EMEAC=0, nivel_final="Bajo"
    out = pd.DataFrame({"date": df_in["date"]})
    out["EMERREL"] = 0.0
    out["EMEAC"] = 0.0
    out["nivel_final"] = "Bajo"
    return out

res = run_model_or_fallback(df)

# Asegurar que resultados no tengan huecos en fechas y no tengan NaN
def fill_result_gaps(full_index_df: pd.DataFrame, res_df: pd.DataFrame) -> pd.DataFrame:
    # Reindexar resultados a las fechas de df
    r = res_df.copy()
    r["date"] = pd.to_datetime(r["date"])
    r = r.set_index("date").sort_index()

    idx = pd.DatetimeIndex(full_index_df["date"])
    r = r.reindex(idx)

    # Interpolaciones/ffill para evitar NaN
    if "EMERREL" in r.columns:
        r["EMERREL"] = pd.to_numeric(r["EMERREL"], errors="coerce")
        r["EMERREL"] = r["EMERREL"].interpolate(method="linear", limit_direction="both").fillna(0.0)
        r["EMERREL"] = r["EMERREL"].clip(lower=0.0)
        # MA5
        r["EMERREL_MA5"] = r["EMERREL"].rolling(5, min_periods=1).mean()

    if "EMEAC" in r.columns:
        r["EMEAC"] = pd.to_numeric(r["EMEAC"], errors="coerce")
        r["EMEAC"] = r["EMEAC"].interpolate(method="linear", limit_direction="both").fillna(0.0)
        r["EMEAC"] = r["EMEAC"].clip(lower=0.0, upper=100.0)

    if "nivel_final" not in r.columns:
        # Derivar a partir de EMEAC si falta
        if "EMEAC" in r.columns:
            cuts = pd.cut(r["EMEAC"], bins=[-0.1,30,70,100.1], labels=["Bajo","Medio","Alto"])
            r["nivel_final"] = cuts.astype(str)
        else:
            r["nivel_final"] = "Bajo"

    # Map icon
    r["nivel_icon"] = r["nivel_final"].map({"Bajo":"🟢 Bajo","Medio":"🟠 Medio","Alto":"🔴 Alto"}).fillna("")
    r["date"] = idx
    return r.reset_index(drop=True)

res_full = fill_result_gaps(df, res)

# Merge meteo + resultados
out = df.merge(res_full, on="date", how="left", suffixes=("",""))

# Gráfico
st.subheader("Serie diaria desde HOY (sin huecos)")
if PLOTLY_OK:
    fig = go.Figure()
    if "EMERREL" in out.columns:
        fig.add_trace(go.Scatter(x=out["date"], y=out["EMERREL"], mode="lines", name="EMERREL"))
        if "EMERREL_MA5" in out.columns:
            fig.add_trace(go.Scatter(
                x=pd.concat([out["date"], out["date"][::-1]]),
                y=pd.concat([out["EMERREL_MA5"], pd.Series([0]*len(out))]),
                fill="toself", name="MA5 (sombreada)",
                line=dict(width=0), hoverinfo="skip", opacity=0.2
            ))
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
st.subheader("Tabla de resultados (hoy → 2026-01-01)")
cols = ["date","tmax","tmin","prec","EMERREL","EMERREL_MA5","EMEAC","nivel_final","nivel_icon"]
cols = [c for c in cols if c in out.columns]
tbl = out[cols].copy()
for c, nd in [("EMERREL",4),("EMERREL_MA5",4),("EMEAC",1)]:
    if c in tbl.columns:
        tbl[c] = tbl[c].map(lambda x: None if pd.isna(x) else round(float(x), nd))
st.dataframe(tbl, use_container_width=True)

# Descarga
@st.cache_data
def to_csv_bytes(dfout: pd.DataFrame) -> bytes:
    return dfout.to_csv(index=False).encode("utf-8")

st.download_button(
    "⬇️ Descargar (CSV)",
    data=to_csv_bytes(out),
    file_name="resultados_sin_huecos.csv",
    mime="text/csv",
)
