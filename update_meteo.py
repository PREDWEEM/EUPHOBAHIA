# -*- coding: utf-8 -*-
# update_meteo.py — asegura ventana COMPLETA 2025-09-01 → 2026-01-01 y no pisa pasado
# Úsalo para mantener meteo_history.csv siempre completo. La app luego usa SOLO este archivo.
import os
import logging
from pathlib import Path
from datetime import date, datetime, timedelta

import pandas as pd
import pytz
import requests
import xml.etree.ElementTree as ET

TZ = os.getenv("TIMEZONE","America/Argentina/Buenos_Aires")
LOCAL_TZ = pytz.timezone(TZ)

WINDOW_START = date(2025,9,1)
WINDOW_END_EXCL = date(2026,1,1)  # exclusivo
PRON_DIAS_API = int(os.getenv("PRON_DIAS_API","8"))
METEOBAHIA_URL = os.getenv("METEOBAHIA_URL","https://meteobahia.com.ar/scripts/forecast/for-bd.xml")
APP_HISTORY_PATH = Path(os.getenv("APP_HISTORY_PATH","meteo_history.csv"))
GH_PATH = Path(os.getenv("GH_PATH","data/meteo_daily.csv"))
GH_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("update_meteo")

def _num(x):
    try:
        if x is None: return None
        s = str(x).strip().replace(",", ".")
        return float(s) if s != "" else None
    except: return None

def parse_xml(xml_bytes: bytes, limit_days: int) -> pd.DataFrame:
    root = ET.fromstring(xml_bytes)
    rows = []
    for node in root.iter():
        tag = node.tag.lower()
        if tag in {"dia","day","item","pronostico","forecastday"}:
            txt = ET.tostring(node, encoding="unicode").lower()
            if any(k in txt for k in ["fecha","date","tmin","tmax","min","max","prec","lluv"]):
                # fecha
                fecha_txt = ""
                if node.attrib.get("fecha"): fecha_txt = node.attrib["fecha"]
                if node.attrib.get("date"):  fecha_txt = node.attrib["date"] or fecha_txt
                if not fecha_txt:
                    for child in list(node):
                        if child.tag.lower() in {"fecha","date","dia"} and (child.text or "").strip():
                            fecha_txt = child.text.strip()
                            break
                fdate = None
                for fmt in ("%Y-%m-%d","%d/%m/%Y","%d-%m-%Y","%Y/%m/%d"):
                    try:
                        fdate = datetime.strptime(fecha_txt, fmt).date(); break
                    except: pass
                if fdate is None: continue

                def get_text(names):
                    for child in list(node):
                        if child.tag.lower() in names:
                            if (child.text or "").strip(): return child.text.strip()
                            v = child.attrib.get("value")
                            if v: return v.strip()
                    # recursivo superficial
                    for child in list(node):
                        for g in list(child):
                            if g.tag.lower() in names:
                                if (g.text or "").strip(): return g.text.strip()
                                v = g.attrib.get("value")
                                if v: return v.strip()
                    return ""

                tmin = _num(get_text({"tmin","min","mintemp"}))
                tmax = _num(get_text({"tmax","max","maxtemp"}))
                pr = get_text({"prec","lluvia","pp","rain","precipitacion"})
                prec = _num(pr.replace("mm","")) if pr else None
                rows.append({"date": fdate, "tmin": tmin, "tmax": tmax, "prec": prec})

    df = pd.DataFrame(rows).drop_duplicates("date").sort_values("date")
    if df.empty: return df
    today = date.today()
    horizon_end = today + timedelta(days=max(0, limit_days-1))
    return df[(df["date"] >= pd.Timestamp(today)) & (df["date"] <= pd.Timestamp(horizon_end))]

def load_history(path: Path) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path, parse_dates=["date"])
        df.columns = [c.lower() for c in df.columns]
        for c in ["tmax","tmin","prec","source","updated_at"]:
            if c not in df.columns: df[c] = None
        return df[["date","tmax","tmin","prec","source","updated_at"]].copy()
    return pd.DataFrame(columns=["date","tmax","tmin","prec","source","updated_at"])

def now_local_iso():
    return datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S%z")

def main():
    log.info("=== UPDATE METEO (ventana fija 2025-09-01 → 2026-01-01 excl.) ===")
    hist = load_history(APP_HISTORY_PATH)
    # descargar api
    try:
        resp = requests.get(METEOBAHIA_URL, headers={"User-Agent":"Mozilla/5.0"}, timeout=30)
        resp.raise_for_status()
        api_df = parse_xml(resp.content, PRON_DIAS_API)
        log.info(f"API días: {len(api_df)}")
    except Exception as e:
        log.warning(f"No se pudo obtener API: {e}")
        api_df = pd.DataFrame(columns=["date","tmin","tmax","prec"])

    # índice fijo
    idx = pd.date_range(WINDOW_START, WINDOW_END_EXCL - timedelta(days=1), freq="D").date
    base = pd.DataFrame({"date": idx})

    for d in (hist, api_df):
        if not d.empty:
            d["date"] = pd.to_datetime(d["date"]).dt.date

    # conservar pasado (congelado) y usar API en futuro
    today = date.today()
    if not hist.empty:
        hist = hist[hist["date"].between(WINDOW_START, WINDOW_END_EXCL - timedelta(days=1))]
    fut_api = pd.DataFrame(columns=["date","tmax","tmin","prec","source","updated_at"])
    if not api_df.empty:
        fut_api = api_df.copy()
        fut_api["source"] = "forecast"
        fut_api["updated_at"] = now_local_iso()
        fut_api = fut_api[fut_api["date"] >= today]

    merged = base.merge(hist, on="date", how="left")
    if not fut_api.empty:
        merged = merged.merge(fut_api, on="date", how="left", suffixes=("","_api"))
        for c in ["tmax","tmin","prec","source","updated_at"]:
            is_future = merged["date"] >= today
            merged[c] = merged[c].where(~is_future | merged[f"{c}_api"].isna(), merged[f"{c}_api"])
        merged = merged.drop(columns=[c for c in merged.columns if c.endswith("_api")], errors="ignore")

    merged = merged.sort_values("date")
    for c in ["tmax","tmin"]:
        merged[c] = pd.to_numeric(merged[c], errors="coerce").ffill()
    merged["prec"] = pd.to_numeric(merged["prec"], errors="coerce").fillna(0.0)
    merged["source"] = merged["source"].fillna("forecast")
    merged["updated_at"] = merged["updated_at"].fillna(now_local_iso())

    APP_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(APP_HISTORY_PATH, index=False)
    GH_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(GH_PATH, index=False)

    log.info(f"[OK] meteo_history.csv actualizado: {merged['date'].min()} → {merged['date'].max()} | filas={len(merged)}")
    log.info(f"NaN tmax={int(merged['tmax'].isna().sum())} tmin={int(merged['tmin'].isna().sum())} prec={int(merged['prec'].isna().sum())}")

if __name__ == "__main__":
    main()
