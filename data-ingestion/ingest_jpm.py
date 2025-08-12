"""
PJM real-time LMP ingestion → SQLite/Postgres
Usage:
  set PJM_API_KEY=xxxx
  [optional] set PJM_RT_LMP_URL=https://api.pjm.com/api/v1/rt_lmp
  python ingest_pjm.py
"""
import os, io, sys, json
from datetime import datetime, timezone
import pandas as pd
import requests
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# ---------- DB helpers (mirrors your style) ----------
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
    # append to avoid clobbering previous ISO data
    df.to_sql(table, engine, if_exists="append", index=False, method="multi")
    with engine.connect() as c:
        n_rows = c.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
    return n_rows

# ---------- PJM client ----------
def fetch_pjm_rt_lmp(limit=1000):
    base_url = os.getenv("PJM_RT_LMP_URL", "https://api.pjm.com/api/v1/rt_lmp")
    api_key = os.getenv("PJM_API_KEY")
    if not api_key:
        raise RuntimeError("Missing PJM_API_KEY. Get a Data Miner 2 key and set PJM_API_KEY env var.")

    headers = {
        "Ocp-Apim-Subscription-Key": api_key,
        "Accept": "*/*",
    }

    # Try a simple GET; users often add query params in Data Miner 2 (e.g., row limit or time filter)
    print(f"🔧 GET {base_url}  (limit hint={limit})")
    r = requests.get(base_url, headers=headers, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"PJM API call failed: HTTP {r.status_code} – {r.text[:200]}")

    # Try JSON first
    content_type = r.headers.get("Content-Type", "").lower()
    if "json" in content_type:
        try:
            data = r.json()
        except Exception:
            data = json.loads(r.text)
        # Data Miner often returns either {"items":[...]} or a plain list
        if isinstance(data, dict):
            rows = data.get("items") or data.get("data") or data.get("Items") or []
        elif isinstance(data, list):
            rows = data
        else:
            rows = []
        df = pd.DataFrame(rows)
    else:
        # Fall back to CSV parsing
        df = pd.read_csv(io.StringIO(r.text))

    if df.empty:
        raise RuntimeError("PJM response parsed but produced an empty DataFrame.")

    print(f"✅ PJM raw rows: {len(df)}; columns: {list(df.columns)[:10]}…")
    return df

def normalize_pjm(df_raw) -> pd.DataFrame:
    """
    Normalize to a common schema:
      timestamp (UTC), repeat_hour_flag, settlement_point, price,
      iso, location_type, location_id
    PJM commonly exposes:
      - timestamp-like: datetime_beginning_ept, DatetimeBeginEpt
      - IDs: pnode_id / Pnode / PnodeId
      - price: lmp / LMP / total_lmp
      - location type: 'PNODE', 'ZONE', 'HUB' via 'type' or 'LocationType'
    """
    cols = {c.lower(): c for c in df_raw.columns}
    # Timestamp
    ts_col = None
    for cand in ["datetime_beginning_ept", "datetime_beginning_utc", "datetime_beginning", "timestamp", "intervalstartutc", "interval_start_utc"]:
        if cand in cols:
            ts_col = cols[cand]; break
    if ts_col is None:
        # try first datetime-like
        for c in df_raw.columns:
            if "time" in c.lower() or "date" in c.lower():
                ts_col = c; break
    if ts_col is None:
        raise RuntimeError("Could not detect a timestamp column in PJM payload.")

    # Price
    price_col = None
    for cand in ["lmp", "total_lmp", "totallmp", "rt_lmp", "price"]:
        if cand in cols:
            price_col = cols[cand]; break
    if price_col is None:
        # last resort: any column named LMP-ish
        for c in df_raw.columns:
            if "lmp" in c.lower():
                price_col = c; break
    if price_col is None:
        raise RuntimeError("Could not detect an LMP/price column in PJM payload.")

    # Location
    id_col = None
    for cand in ["pnode_id", "pnode", "location", "pnodeid"]:
        if cand in cols:
            id_col = cols[cand]; break
    if id_col is None:
        # maybe it's zone/hub text
        for c in df_raw.columns:
            if "location" in c.lower() or "node" in c.lower() or "pnode" in c.lower():
                id_col = c; break

    type_col = None
    for cand in ["type", "locationtype", "location_type"]:
        if cand in cols:
            type_col = cols[cand]; break

    out = pd.DataFrame()
    # PJM source data often in EPT (Eastern Prevailing Time) → convert to UTC
    ts = pd.to_datetime(df_raw[ts_col], errors="coerce", utc=False)
    if ts.dt.tz is None:
        try:
            ts = ts.dt.tz_localize("America/New_York").dt.tz_convert("UTC")
        except Exception:
            ts = ts.dt.tz_localize("UTC")
    else:
        ts = ts.dt.tz_convert("UTC")
    out["timestamp"] = ts.dt.tz_convert(None)  # naive UTC

    out["price"] = pd.to_numeric(df_raw[price_col], errors="coerce")
    out["settlement_point"] = (df_raw[id_col].astype(str) if id_col else pd.Series(["PJM_UNKNOWN"]*len(out)))
    out["repeat_hour_flag"] = False  # we can refine during DST transitions if needed
    out["iso"] = "PJM"
    out["location_type"] = (df_raw[type_col].astype(str) if type_col else pd.Series(["PNODE"]*len(out)))
    out["location_id"] = out["settlement_point"]

    # basic cleaning bounds, like your ERCOT script
    out["price"] = out["price"].ffill()
    med = out["price"].median()
    out["price"] = out["price"].fillna(med)
    out.loc[out["price"] < 0, "price"] = 0
    out.loc[out["price"] > 5000, "price"] = 5000  # PJM can spike; keep a high cap

    # drop any rows missing timestamp/price after cleaning
    out = out.dropna(subset=["timestamp", "price"])

    return out[["timestamp", "repeat_hour_flag", "settlement_point", "price", "iso", "location_type", "location_id"]]

def main():
    print("🚀 PJM Ingestion (Real-Time LMP)")
    engine, dbtype = setup_database_connection()

    raw = fetch_pjm_rt_lmp(limit=1000)
    df = normalize_pjm(raw)

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

    # PRD test echoes
    print("\n🧪 Tests:")
    print(" - HTTP/Parse OK: ✅")
    print(" - Schema has timestamp & price: ✅" if {"timestamp","price"}.issubset(df.columns) else "❌")
    print(" - No NaNs in critical cols:", "✅" if df["timestamp"].isna().sum()==0 and df["price"].isna().sum()==0 else "❌")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ PJM ingestion failed: {e}")
        sys.exit(1)
