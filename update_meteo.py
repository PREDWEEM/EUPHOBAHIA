# update_meteo.py
# Consolida y ARCHIVA el horizonte fijo 2025-09-01 → 2026-01-01 (exclusivo)
# - Congela pronósticos pasados
# - Actualiza sólo futuros
# - Asegura continuidad diaria y genera snapshots

import os
import io
import sys
import csv
import math
import json
import time
import pytz
import shutil
import zipfile
import logging
import datetime as dt
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd

import xml.etree.ElementTree as ET
import requests

# ================== CONFIG ==================
TZ = os.getenv("TIMEZONE", "America/Argentina/Buenos_Aires")
LOCAL_TZ = pytz.timezone(TZ)

WINDOW_START = dt.date(2025, 9, 1)
WINDOW_END   = dt.date(2026, 1, 1)   # EXCLUSIVO (no se incluye el 01-ene-2026)

# URL por defecto (Bordenave, 'bd'); podés cambiar a 'bb' si querés Bahía Blanca
METEOBAHIA_URL = os.getenv(
    "METEOBAHIA_URL",
    "https://meteobahia.com.ar/scripts/forecast/for-bd.xml"
)

# Días de pronóstico que se usan de la API
PRON_DIAS_API = int(os.getenv("PRON_DIAS_API", "8"))

# Rutas de salida
APP_HISTORY_PATH = Path(os.getenv("APP_HISTORY_PATH", "meteo_history.csv"))
GH_PATH = Path(os.getenv("GH_PATH", "data/meteo_daily.csv"))

ARCHIVES_DIR = Path("data/archives")
ARCHIVES_DIR.mkdir(parents=True, exist_ok=True)
GH_PATH.parent.mkdir(parents=True, exist_ok=True)

# ================== LOGGING ==================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("update_meteo")

# ================== HELPERS ==================
def now_local_iso() -> str:
    return dt.datetime.now(tz=LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S%z")

def daterange(start: dt.date, end_exclusive: dt.date):
    for n in range((end_exclusive - start).days):
        yield start + dt.timedelta(days=n)

def _num(x) -> Optional[float]:
    if x is None:
        return None
    try:
        s = str(x).strip().replace(",", ".")
        if s == "":
            return None
        return float(s)
    except:
        return None

def _text(elem: ET.Element) -> str:
    if elem is None:
        return ""
    if elem.text is not None and elem.text.strip() != "":
        return elem.text.strip()
    # value en atributo
    v = elem.attrib.get("value")
    return (v or "").strip()

def _find_first(elem: ET.Element, names: List[str]) -> Optional[ET.Element]:
    """Busca primer subtag cuyo tag en minúsculas esté en names."""
    if elem is None:
        return None
    low = set(n.lower() for n in names)
    for child in list(elem):
        if child.tag.lower() in low:
            return child
    # búsqueda recursiva superficial
    for child in list(elem):
        for gchild in list(child):
            if gchild.tag.lower() in low:
                return gchild
    return None

def _get_tag_text(elem: ET.Element, names: List[str]) -> str:
    sub = _find_first(elem, names)
    return _text(sub)

def parse_meteobahia_xml(xml_bytes: bytes, limit_days: int) -> pd.DataFrame:
    """
    Intenta parsear varios esquemas posibles.
    Debe devolver columnas: date (YYYY-MM-DD), tmin, tmax, prec.
    """
    root = ET.fromstring(xml_bytes)

    # Heurística: recolectar nodos candidatos a "día"
    day_candidates = []
    for node in root.iter():
        tag = node.tag.lower()
        # nombres típicos
        if tag in {"dia", "day", "item", "pronostico", "forecastday"}:
            # que tengan algún rastro de fecha o variables
            txt = ET.tostring(node, encoding="unicode").lower()
            if any(k in txt for k in ["fecha", "date", "tmin", "tmax", "min", "max", "prec", "lluv"]):
                day_candidates.append(node)

    rows = []
    for node in day_candidates:
        # fecha en distintos campos
        fecha_txt = (
            _get_tag_text(node, ["fecha", "date", "dia"]) or
            node.attrib.get("fecha", "") or
            node.attrib.get("date", "")
        ).strip()

        # Normalizar fecha
        fdate = None
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"):
            try:
                fdate = dt.datetime.strptime(fecha_txt, fmt).date()
                break
            except:
                pass

        # Si no encontramos fecha, intentar si el nodo tiene índice relativo (poco común)
        if fdate is None:
            # saltar si no hay fecha clara
            continue

        tmin = _num(_get_tag_text(node, ["tmin", "min", "mintemp"]))
        tmax = _num(_get_tag_text(node, ["tmax", "max", "maxtemp"]))
        # precipitación
        prec_raw = _get_tag_text(node, ["prec", "lluvia", "pp", "rain", "precipitacion"])
        # algunos esquemas devuelven "12.3 mm"
        prec = None
        if prec_raw:
            prec = _num(prec_raw.replace("mm", "").strip())

        rows.append({"date": fdate, "tmin": tmin, "tmax": tmax, "prec": prec})

    if not rows:
        # fallback: sin datos
        return pd.DataFrame(columns=["date", "tmin", "tmax", "prec"]).astype(
            {"date": "datetime64[ns]"}
        )

    df = pd.DataFrame(rows).sort_values("date").drop_duplicates("date", keep="last")
    # recortar a los primeros limit_days desde hoy si fuera necesario
    today = dt.date.today()
    horizon_end = today + dt.timedelta(days=max(0, limit_days - 1))
    df = df[(df["date"] >= pd.Timestamp(today)) & (df["date"] <= pd.Timestamp(horizon_end))]
    return df

def load_history(path: Path) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path, parse_dates=["date"], dayfirst=False)
        # normalizar nombres
        cols = {c.lower(): c for c in df.columns}
        df.columns = [c.lower() for c in df.columns]
        # asegurar columnas mínimas
        for c in ["tmax", "tmin", "prec", "source", "updated_at"]:
            if c not in df.columns:
                df[c] = None
        return df[["date", "tmax", "tmin", "prec", "source", "updated_at"]].copy()
    else:
        return pd.DataFrame(columns=["date", "tmax", "tmin", "prec", "source", "updated_at"])

def save_snapshot(current_df: pd.DataFrame):
    ts = dt.datetime.now(LOCAL_TZ).strftime("%Y%m%d_%H%M%S")
    snap_path = ARCHIVES_DIR / f"meteo_history_{ts}.csv"
    current_df.to_csv(snap_path, index=False)

# ================== MAIN PIPELINE ==================
def main():
    log.info("=== UPDATE METEO (Ventana fija 2025-09-01 → 2026-01-01 exclusivo) ===")
    log.info(f"TZ={TZ} | URL={METEOBAHIA_URL} | PRON_DIAS_API={PRON_DIAS_API}")
    today = dt.date.today()

    # 1) Leer histórico previo
    hist = load_history(APP_HISTORY_PATH)
    log.info(f"Histórico cargado: {len(hist)} filas" if not hist.empty else "Sin histórico previo")

    # 2) Descargar pronóstico
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; MeteoCollector/1.0; +https://example.org)"
        }
        resp = requests.get(METEOBAHIA_URL, headers=headers, timeout=30)
        resp.raise_for_status()
        api_df = parse_meteobahia_xml(resp.content, PRON_DIAS_API)
        log.info(f"API parseada: {len(api_df)} días (desde hoy)")
    except Exception as e:
        log.error(f"No se pudo descargar/parsear API: {e}")
        api_df = pd.DataFrame(columns=["date", "tmin", "tmax", "prec"])

    # 3) Construir índice diario completo para la ventana fija
    idx = pd.to_datetime([d for d in daterange(WINDOW_START, WINDOW_END)])
    base = pd.DataFrame({"date": idx})

    # 4) Unir histórico congelado + pronóstico futuro (sin pisar pasado)
    # Normalizar tipos
    for df in (hist, api_df):
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"]).dt.date

    base["date"] = base["date"].dt.date

    # a) Usar histórico existente dentro de la ventana (congelado)
    if not hist.empty:
        hist_win = hist[hist["date"].between(WINDOW_START, WINDOW_END - dt.timedelta(days=1))]
    else:
        hist_win = pd.DataFrame(columns=["date", "tmax", "tmin", "prec", "source", "updated_at"])

    # b) Preparar el pronóstico para fechas futuras (>= hoy)
    if not api_df.empty:
        api_df["source"] = "forecast"
        api_df["updated_at"] = now_local_iso()
        api_future = api_df[api_df["date"] >= today]
    else:
        api_future = pd.DataFrame(columns=["date", "tmax", "tmin", "prec", "source", "updated_at"])

    # c) Merge: primero histórico (pasado congelado), luego insertamos futuros de API
    merged = base.merge(
        hist_win, on="date", how="left", suffixes=("", "_hist")
    )

    # Para cada fecha futura, si hay API, reemplazar/poner valores; para pasada, conservar histórico
    if not api_future.empty:
        fut = merged["date"] >= today
        merged = merged.merge(
            api_future, on="date", how="left", suffixes=("", "_api")
        )

        for col in ["tmax", "tmin", "prec", "source", "updated_at"]:
            # si es futuro y hay valor de API, usarlo; si no, dejar lo existente
            merged[col] = merged[col].where(~fut | merged[f"{col}_api"].isna(), merged[f"{col}_api"])

        # limpiar columnas *_api
        drop_cols = [c for c in merged.columns if c.endswith("_api") or c.endswith("_hist")]
        merged = merged.drop(columns=drop_cols, errors="ignore")

    # 5) Rellenos conservadores para continuidad (sin pisar valores existentes)
    merged = merged.sort_values("date")
    for col in ["tmax", "tmin"]:
        # ffill sólo donde esté NaN
        merged[col] = merged[col].astype(float)
        merged[col] = merged[col].fillna(method="ffill")

    # Precipitación: NaN -> 0.0
    merged["prec"] = merged["prec"].astype(float)
    merged["prec"] = merged["prec"].fillna(0.0)

    # Completar metadatos mínimos
    merged["source"] = merged["source"].fillna("forecast")
    merged["updated_at"] = merged["updated_at"].fillna(now_local_iso())

    # 6) Guardar salidas
    out_df = merged.copy()
    out_df.to_csv(APP_HISTORY_PATH, index=False)
    out_df.to_csv(GH_PA_
