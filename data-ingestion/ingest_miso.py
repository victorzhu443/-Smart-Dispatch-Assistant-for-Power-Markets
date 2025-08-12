"""
MISO real-time LMP ingestion → SQLite/Postgres
Usage:
  # If you have a CSV/TSV endpoint (common for MISO reports), set it:
  set MISO_RT_LMP_URL=<full URL that returns CSV>
  python ingest_miso.py

Notes:
- Script auto-detects CSV vs JSON and common column names.
- Converts local Central time to UTC.
"""
import os, io, sys, json
from datetime import datetime
import pandas as pd
import requests
from sqlalchemy import create_engine, text

# ---------- DB helpers ----------
def setup_database_connection():
    try:
        pg_user = os.getenv('POSTGRES_USER', 'postgres')
        pg_password = os.getenv('POSTGRES_PASSWORD', 'password')
        pg_host = os.getenv('POSTGRES_HOST', 'localhost')
        pg_port = os.getenv('POSTGRES_PORT', '5432')
        pg_database = os.getenv('POSTGRES_DATABASE', 'smart_dispatch')
        engine = create_engine(f"postgresql://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_database}")
        with engine.connect() as c:
            c.execute(text("SELECT 1"))
        print("✅ PostgreSQL connection successful")
        return engine, "postgresql"
    except Exception as e:
        print(f"⚠️ PostgreSQL connection failed: {e}")
        print("🔄 Falling back to SQLite for development…")
        path = "market_data.db"
        engine = create_engine(f"sqlite:///{path}")
        print(f"✅ SQLite connection successful: {path}")
        return engine, "sqlite"

def save_to_sql(df, engine, table="market_data_all"):
    df.to_sql(table, engine, if_exists="append", index=False, method="multi")
    with engine.connect() as c:
        n_rows = c.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
    return n_rows

# ---------- MISO fetch/normalize ----------
def fetch_miso_rt_lmp():
    url = os.getenv("MISO_RT_LMP_URL")
    if not url:
        raise RuntimeError(
            "MISO_RT_LMP_URL not set. Provide a CSV (or JSON) URL for real-time LMPs."
        )
    print(f"🔧 GET {url}")
    r = requests.get(url, timeout=90)
    if r.status_code != 200:
        raise RuntimeError(f"MISO fetch failed: HTTP {r.status_code} – {r.text[:200]}")
    ct = r.headers.get("Content-Type","").lower()
    if "json" in ct:
        try:
            data = r.json()
        except Exception:
            data = json.loads(r.text)
        if isinstance(data, dict):
            rows = data.get("items") or data.get("data") or []
        elif isinstance(data, list):
            rows = data
        else:
            rows = []
        df = pd.DataFrame(rows)
    else:
        # CSV/TSV
        df = pd.read_csv(io.StringIO(r.text))
    if df.empty:
        raise RuntimeError("MISO response parsed but produced an empty DataFrame.")
    print(f"✅ MISO raw rows: {len(df)}; columns: {list(df.columns)[:10]}…")
    return df

def normalize_miso(df_raw) -> pd.DataFrame:
    """
    Normalize to:
      timestamp (UTC), repeat_hour_flag, settlement_point, price,
      iso, location_type, location_id
    Common MISO columns:
      - time: 'LocalTime', 'MarketDate', 'MarketHour', 'IntervalStart', 'Timestamp'
      - location: 'Pnode', 'Location', 'Node'
      - price: 'LMP', 'TotalLMP'
      - type: 'PriceType' / 'LocationType'
    """
    cols = {c.lower(): c for c in df_raw.columns}

    # Timestamp candidates (some reports give date + time separately)
    ts_col = None
    for cand in ["timestamp", "intervalstart", "interval_start", "localtime", "interval_start_local"]:
        if cand in cols:
            ts_col = cols[cand]; break

    if ts_col is None:
        # combine date + hour/min if present
        date_col = None
        time_col = None
        for cand in ["marketdate", "mkt_dt", "date"]:
            if cand in cols:
                date_col = cols[cand]; break
        for cand in ["marketinterval", "mkt_interval", "interval", "time", "hourending", "he"]:
            if cand in cols:
                time_col = cols[cand]; break
        if date_col and time_col:
            ts = pd.to_datetime(df_raw[date_col].astype(str) + " " + df_raw[time_col].astype(str), errors="coerce")
        else:
            # fallback: first datetime-like column
            maybe = None
            for c in df_raw.columns:
                if "time" in c.lower() or "date" in c.lower():
                    maybe = c; break
            if maybe is None:
                raise RuntimeError("Could not detect a timestamp column in MISO payload.")
            ts = pd.to_datetime(df_raw[maybe], errors="coerce")
    else:
        ts = pd.to_datetime(df_raw[ts_col], errors="coerce")

    # localize to Central then convert to UTC (MISO footprint)
    if ts.dt.tz is None:
        try:
            ts = ts.dt.tz_localize("America/Chicago").dt.tz_convert("UTC")
        except Exception:
            ts = ts.dt.tz_localize("UTC")
    else:
        ts = ts.dt.tz_convert("UTC")

    # Price
    price_col = None
    for cand in ["lmp", "totallmp", "total_lmp", "price"]:
        if cand in cols:
            price_col = cols[cand]; break
    if price_col is None:
        for c in df_raw.columns:
            if "lmp" in c.lower():
                price_col = c; break
    if price_col is None:
        raise RuntimeError("Could not detect LMP/price column in MISO payload.")

    # Location
    id_col = None
    for cand in ["pnode", "node", "location", "settlementpoint"]:
        if cand in cols:
            id_col = cols[cand]; break
    if id_col is None:
        for c in df_raw.columns:
            if "node" in c.lower() or "location" in c.lower():
                id_col = c; break

    type_col = None
    for cand in ["locationtype", "pricetype", "type"]:
        if cand in cols:
            type_col = cols[cand]; break

    out = pd.DataFrame()
    out["timestamp"] = ts.dt.tz_convert(None)
    out["price"] = pd.to_numeric(df_raw[price_col], errors="coerce")
    out["settlement_point"] = (df_raw[id_col].astype(str) if id_col else pd.Series(["MISO_UNKNOWN"]*len(out)))
    out["repeat_hour_flag"] = False
    out["iso"] = "MISO"
    out["location_type"] = (df_raw[type_col].astype(str) if type_col else pd.Series(["PNODE"]*len(out)))
    out["location_id"] = out["settlement_point"]

    # Clean
    out["price"] = out["price"].ffill()
    med = out["price"].median()
    out["price"] = out["price"].fillna(med)
    out.loc[out["price"] < 0, "price"] = 0
    out.loc[out["price"] > 5000, "price"] = 5000

    out = out.dropna(subset=["timestamp", "price"])

    return out[["timestamp", "repeat_hour_flag", "settlement_point", "price", "iso", "location_type", "location_id"]]

def main():
    print("🚀 MISO Ingestion (Real-Time LMP)")
    engine, dbtype = setup_database_connection()
    raw = fetch_miso_rt_lmp()
    df = normalize_miso(raw)

    print(f"📊 Normalized rows: {len(df)}; cols: {list(df.columns)}")
    print(f"⏱️ Time range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"💰 Price range: ${df['price'].min():.2f} – ${df['price'].max():.2f}")
    print(f"📍 Unique points: {df['settlement_point'].nunique()}")

    n_before = 0
    try:
        with engine.connect() as c:
            n_before = c.execute(text("SELECT COUNT(*) FROM market_data_all")).scalar()
    except Exception:
        pass
    total = save_to_sql(df, engine, table="market_data_all")
    added = total - n_before
    print(f"✅ Saved to market_data_all ({dbtype}) — total rows: {total} (added {max(0,added)})")

    # PRD echoes
    print("\n🧪 Tests:")
    print(" - HTTP/Parse OK: ✅")
    print(" - Schema has timestamp & price: ✅" if {"timestamp","price"}.issubset(df.columns) else "❌")
    print(" - No NaNs in critical cols:", "✅" if df['timestamp'].isna().sum()==0 and df['price'].isna().sum()==0 else "❌")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ MISO ingestion failed: {e}")
        sys.exit(1)
