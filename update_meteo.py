# -*- coding: utf-8 -*-
# update_meteo.py — MERGE SIN VENTANA FIJA (no crea fechas artificiales)
# Reglas:
# - Conserva todas las filas existentes en meteo_history.csv tal como están.
# - Agrega/actualiza SOLO fechas provistas por la API (p. ej., próximos 8 días).
# - NO reindexa ni completa rangos; no extiende hasta una fecha fija.

import os
from pathlib import Path
from datetime import date, datetime, timedelta

import pandas as pd
import pytz
import requests
import xml.etree.ElementTree as ET

TZ = os.getenv("TIMEZONE","America/Argentina/Buenos_Aires")
LOCAL_TZ = pytz.timezone(TZ)

PRON_DIAS_API = int(os.getenv("PRON_DIAS_API","8"))
METEOBAHIA_URL = os.getenv("METEOBAHIA_URL","https://meteobahia.com.ar/scripts/forecast/for-bd.xml")
APP_HISTORY_PATH = Path(os.getenv("APP_HISTORY_PATH","meteo_history.csv"))
GH_PATH = Path(os.getenv("GH_PATH","data/meteo_daily.csv"))
GH_PATH.parent.mkdir(parents=True, exist_ok=True)

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
    hist = load_history(APP_HISTORY_PATH)
    # Obtener API (si falla, no tocamos nada)
    try:
        resp = requests.get(METEOBAHIA_URL, headers={"User-Agent":"Mozilla/5.0"}, timeout=30)
        resp.raise_for_status()
        api_df = parse_xml(resp.content, PRON_DIAS_API)
    except Exception:
        api_df = pd.DataFrame(columns=["date","tmin","tmax","prec"])

    if not hist.empty:
        hist["date"] = pd.to_datetime(hist["date"]).dt.date
    if not api_df.empty:
        api_df["date"] = pd.to_datetime(api_df["date"]).dt.date
        api_df["source"] = "forecast"
        api_df["updated_at"] = now_local_iso()

    # Merge SÓLO en fechas de la API (no se crean fechas fuera de lo que existe + API)
    out = hist.copy()
    if not api_df.empty:
        # Para cada fecha de API, si existe en hist → actualizar; si no existe → agregar (porque ES dato provisto)
        out = pd.concat([out[~out["date"].isin(api_df["date"])], api_df], ignore_index=True)
        out = out.sort_values("date")

    # Guardar
    APP_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(APP_HISTORY_PATH, index=False)
    GH_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(GH_PATH, index=False)

if __name__ == "__main__":
    main()
