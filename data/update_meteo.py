# update_meteo.py
import os
import sys
import pytz
import datetime as dt
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

# ================== Config ==================
API_URL = os.getenv("METEOBAHIA_URL", "https://meteobahia.com.ar/scripts/forecast/for-bb.xml")
PRON_DIAS_API = int(os.getenv("PRON_DIAS_API", "8"))  # hoy + 7
CSV_PATH = "data/meteo_daily.csv"
TZ = pytz.timezone("America/Argentina/Buenos_Aires")

# ================== Utils ==================
def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def today_local():
    return dt.datetime.now(TZ).date()

def parse_api_xml(url: str):
    """Parsea un XML diario de MeteoBahía a [{'date', 'tmax', 'tmin', 'prec', 'source'}]. Ajustá si tu XML difiere."""
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=30) as r:
        xml_bytes = r.read()
    root = ET.fromstring(xml_bytes)

    out = []
    # Suponemos nodos <day> con sub-entradas fecha/tmax/tmin/prec. Adaptar si es necesario.
    for day in root.iter():
        if day.tag.lower().endswith("day"):
            dstr, tmax, tmin, prec = None, None, None, None
            for child in day:
                tag = child.tag.lower()
                txt = (child.text or "").strip()
                if "date" in tag or "fecha" in tag or tag.endswith("daydate"):
                    dstr = txt
                elif "tmax" in tag or "max" in tag:
                    try: tmax = float(txt.replace(",", ".")); except: pass
                elif "tmin" in tag or "min" in tag:
                    try: tmin = float(txt.replace(",", ".")); except: pass
                elif "prec" in tag or "rain" in tag or "pp" in tag:
                    try: prec = float(txt.replace(",", ".")); except: pass

            d = None
            for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
                if dstr:
                    try:
                        d = dt.datetime.strptime(dstr, fmt).date()
                        break
                    except:
                        continue
            if d is None or tmax is None or tmin is None or prec is None:
                continue

            out.append({"date": d, "tmax": tmax, "tmin": tmin, "prec": prec, "source": "forecast"})
    return out

def load_existing(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=["date","jd","tmax","tmin","prec","source","updated_at"])
    df = pd.read_csv(csv_path, parse_dates=["date","updated_at"])
    df["date"] = df["date"].dt.date
    return df

def save_csv(df: pd.DataFrame, path: str):
    ensure_dir(path)
    df2 = df.sort_values("date").copy()
    df2.to_csv(path, index=False)

# ================== Main ==================
def main():
    today = today_local()
    horizon_end = today + dt.timedelta(days=PRON_DIAS_API - 1)

    df = load_existing(CSV_PATH)

    try:
        forecast = parse_api_xml(API_URL)
    except (HTTPError, URLError, ET.ParseError) as e:
        print(f"[WARN] No se pudo leer la API: {e}", file=sys.stderr)
        forecast = []

    fdf = pd.DataFrame(forecast)
    if not fdf.empty:
        fdf = fdf[(fdf["date"] >= today) & (fdf["date"] <= horizon_end)]
        fdf["jd"] = fdf["date"].apply(lambda d: d.timetuple().tm_yday)
        fdf["updated_at"] = pd.Timestamp.now(TZ)
    else:
        # Sin datos nuevos: crear el índice de fechas y rellenar luego
        dates = pd.date_range(today, horizon_end, freq="D").date
        fdf = pd.DataFrame({
            "date": dates,
            "tmax": np.nan, "tmin": np.nan, "prec": np.nan,
            "source": "forecast",
            "jd": [d.timetuple().tm_yday for d in dates],
            "updated_at": pd.Timestamp.now(TZ)
        })

    if df.empty:
        merged = fdf.copy()
    else:
        if not fdf.empty:
            lo, hi = fdf["date"].min(), fdf["date"].max()
            mask = (df["date"] >= lo) & (df["date"] <= hi)
            merged = pd.concat([df.loc[~mask].copy(), fdf.copy()], ignore_index=True)
        else:
            merged = df.copy()

    # Asegurar continuidad entre el mínimo del dataset y horizon_end
    min_date = merged["date"].min() if not merged.empty else today
    full_idx = pd.date_range(min_date, horizon_end, freq="D").date
    merged = merged.set_index("date").reindex(full_idx).reset_index().rename(columns={"index":"date"})

    # Completar columnas clave
    merged["jd"] = merged["date"].apply(lambda d: d.timetuple().tm_yday)
    merged["updated_at"] = merged["updated_at"].fillna(pd.Timestamp.now(TZ))
    # Imputación conservadora para no dejar NaN (ajustable a tu preferencia)
    merged["tmax"] = merged["tmax"].fillna(method="ffill")
    merged["tmin"] = merged["tmin"].fillna(method="ffill")
    merged["prec"] = merged["prec"].fillna(0.0)
    merged["source"] = merged["source"].fillna("forecast")

    save_csv(merged, CSV_PATH)
    print(f"[OK] Actualizado {CSV_PATH} hasta {horizon_end}")

if __name__ == "__main__":
    main()

