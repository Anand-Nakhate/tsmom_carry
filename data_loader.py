"""
Contract-level futures data loader for Databento.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import pandas as pd
import databento as db


@dataclass(frozen=True)
class RootContract:
    dataset: str
    parent: str
    asset_class: str
    region: str


GLBX_UNIVERSE: List[RootContract] = [
    # Equity
    RootContract("GLBX.MDP3", "ES.FUT",  "Equity",    "US"),
    RootContract("GLBX.MDP3", "NQ.FUT",  "Equity",    "US"),
    RootContract("GLBX.MDP3", "RTY.FUT", "Equity",    "US"),
    RootContract("GLBX.MDP3", "NKD.FUT", "Equity",    "Japan"),

    # Rates
    RootContract("GLBX.MDP3", "ZT.FUT", "Rates", "US"),
    RootContract("GLBX.MDP3", "ZF.FUT", "Rates", "US"),
    RootContract("GLBX.MDP3", "ZN.FUT", "Rates", "US"),
    RootContract("GLBX.MDP3", "ZB.FUT", "Rates", "US"),
    RootContract("GLBX.MDP3", "UB.FUT", "Rates", "US"),

    # FX
    RootContract("GLBX.MDP3", "6E.FUT", "FX", "Global"),
    RootContract("GLBX.MDP3", "6J.FUT", "FX", "Global"),
    RootContract("GLBX.MDP3", "6B.FUT", "FX", "Global"),
    RootContract("GLBX.MDP3", "6A.FUT", "FX", "Global"),
    RootContract("GLBX.MDP3", "6C.FUT", "FX", "Global"),
    RootContract("GLBX.MDP3", "6S.FUT", "FX", "Global"),
    RootContract("GLBX.MDP3", "6N.FUT", "FX", "Global"),

    # Energy
    RootContract("GLBX.MDP3", "CL.FUT", "Commodity", "Global"),
    RootContract("GLBX.MDP3", "NG.FUT", "Commodity", "US"),
    RootContract("GLBX.MDP3", "HO.FUT", "Commodity", "US"),
    RootContract("GLBX.MDP3", "RB.FUT", "Commodity", "US"),

    # Metals
    RootContract("GLBX.MDP3", "GC.FUT", "Commodity", "Global"),
    RootContract("GLBX.MDP3", "SI.FUT", "Commodity", "Global"),
    RootContract("GLBX.MDP3", "HG.FUT", "Commodity", "Global"),

    # Grains / oilseeds
    RootContract("GLBX.MDP3", "ZC.FUT", "Commodity", "US"),
    RootContract("GLBX.MDP3", "ZW.FUT", "Commodity", "US"),
    RootContract("GLBX.MDP3", "ZS.FUT", "Commodity", "US"),
    RootContract("GLBX.MDP3", "ZL.FUT", "Commodity", "US"),
    RootContract("GLBX.MDP3", "ZM.FUT", "Commodity", "US"),
]


def make_historical_client(api_key=None):
    key = api_key or os.environ.get("DATABENTO_API_KEY")
    if not key:
        raise RuntimeError("Set DATABENTO_API_KEY or pass api_key.")
    return db.Historical(key)


def _chunked(iterable, chunk_size):
    buf = []
    for item in iterable:
        buf.append(item)
        if len(buf) >= chunk_size:
            yield buf
            buf = []
    if buf:
        yield buf


def _normalize_dates(start, end):
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    else:
        start_ts = start_ts.tz_convert("UTC")
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    else:
        end_ts = end_ts.tz_convert("UTC")
    return start_ts, end_ts


def _ensure_ts_event_column(df):
    if "ts_event" in df.columns:
        return df
    df = df.reset_index()
    if "ts_event" not in df.columns:
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "ts_event"})
    return df


def fetch_definitions_for_roots(client, roots, as_of):
    as_of_date = pd.to_datetime(as_of).date()
    start_str = as_of_date.strftime("%Y-%m-%d")

    ds_to_assets: Dict[str, set] = {}
    meta_rows = []
    for r in roots:
        root_code = r.parent.split(".")[0]
        ds_to_assets.setdefault(r.dataset, set()).add(root_code)
        meta_rows.append(
            {
                "dataset": r.dataset,
                "asset": root_code,
                "asset_class": r.asset_class,
                "region": r.region,
                "parent": r.parent,
            }
        )
    meta_df = pd.DataFrame(meta_rows).drop_duplicates(subset=["dataset", "asset"])

    frames = []
    for dataset, wanted_assets in ds_to_assets.items():
        print(f"[INFO] Fetching definitions for dataset={dataset} roots={sorted(wanted_assets)}")
        try:
            store = client.timeseries.get_range(
                dataset=dataset,
                schema="definition",
                symbols="ALL_SYMBOLS",
                start=start_str,
            )
        except Exception as e:
            print(f"[WARN] Definition query failed for dataset={dataset}: {e}")
            continue

        df = store.to_df()
        if df.empty:
            print(f"[WARN] Empty definition response for dataset={dataset}")
            continue

        if "asset" not in df.columns:
            raise RuntimeError(
                f"Definition schema for dataset={dataset} has no 'asset' column."
            )

        df = df[df["asset"].isin(wanted_assets)].copy()
        if df.empty:
            print(f"[WARN] No instruments in dataset={dataset} for assets={wanted_assets}")
            continue

        df = df.sort_values("ts_event").groupby("instrument_id", as_index=False).tail(1)

        if "instrument_class" in df.columns:
            df = df[df["instrument_class"] == db.InstrumentClass.FUTURE].copy()

        df["dataset"] = dataset
        frames.append(df)

    if not frames:
        raise RuntimeError("No instrument definitions returned for any dataset/roots.")

    defs = pd.concat(frames, ignore_index=True)

    defs = defs.merge(
        meta_df,
        on=["dataset", "asset"],
        how="left",
        validate="many_to_one",
    )
    defs = defs[~defs["asset_class"].isna()].reset_index(drop=True)

    print(f"[INFO] Definitions fetched: {defs['instrument_id'].nunique()} instruments")
    return defs


def fetch_ohlcv_1d(client, dataset, instrument_ids, start, end,
                   chunk_size=8, days_per_chunk=180):
    if not instrument_ids:
        return pd.DataFrame()

    start_ts, end_ts = _normalize_dates(start, end)
    frames = []

    for chunk_idx, chunk in enumerate(_chunked(instrument_ids, chunk_size), start=1):
        current = start_ts
        while current < end_ts:
            window_end = min(current + pd.Timedelta(days=days_per_chunk), end_ts)
            print(
                f"[INFO] OHLCV dataset={dataset} chunk={chunk_idx} "
                f"ids={len(chunk)} window={current.date()}->{window_end.date()}"
            )
            try:
                store = client.timeseries.get_range(
                    dataset=dataset,
                    schema="ohlcv-1d",
                    stype_in="instrument_id",
                    symbols=chunk,
                    start=current,
                    end=window_end,
                )
                df = store.to_df()
            except Exception as e:
                print(
                    f"[WARN] OHLCV query failed for dataset={dataset}, "
                    f"chunk={chunk_idx}, window={current.date()}->{window_end.date()}: {e}"
                )
                current = window_end
                continue

            if not df.empty:
                df = _ensure_ts_event_column(df)
                frames.append(df)

            current = window_end

    if not frames:
        print(f"[WARN] No OHLCV data returned for dataset={dataset}.")
        return pd.DataFrame()

    ohlcv = pd.concat(frames, ignore_index=True)
    ohlcv["ts_event"] = pd.to_datetime(ohlcv["ts_event"], utc=True)
    return ohlcv


def fetch_statistics(client, dataset, instrument_ids, start, end,
                     chunk_size=8, days_per_chunk=180):
    if not instrument_ids:
        return pd.DataFrame()

    start_ts, end_ts = _normalize_dates(start, end)
    frames = []

    for chunk_idx, chunk in enumerate(_chunked(instrument_ids, chunk_size), start=1):
        current = start_ts
        while current < end_ts:
            window_end = min(current + pd.Timedelta(days=days_per_chunk), end_ts)
            print(
                f"[INFO] STATS dataset={dataset} chunk={chunk_idx} "
                f"ids={len(chunk)} window={current.date()}->{window_end.date()}"
            )
            try:
                store = client.timeseries.get_range(
                    dataset=dataset,
                    schema="statistics",
                    stype_in="instrument_id",
                    symbols=chunk,
                    start=current,
                    end=window_end,
                )
                df = store.to_df()
            except Exception as e:
                print(
                    f"[WARN] Statistics query failed for dataset={dataset}, "
                    f"chunk={chunk_idx}, window={current.date()}->{window_end.date()}: {e}"
                )
                current = window_end
                continue

            if not df.empty:
                df = _ensure_ts_event_column(df)
                frames.append(df)

            current = window_end

    if not frames:
        print(f"[INFO] No statistics data returned for dataset={dataset}.")
        return pd.DataFrame()

    stats = pd.concat(frames, ignore_index=True)
    stats["ts_event"] = pd.to_datetime(stats["ts_event"], utc=True)
    return stats


def build_daily_panel(defs, ohlcv, stats):
    if ohlcv.empty:
        raise ValueError("OHLCV input is empty; nothing to build.")

    settlement_code = db.StatType.SETTLEMENT_PRICE
    vol_code = db.StatType.CLEARED_VOLUME
    oi_code = db.StatType.OPEN_INTEREST

    panel = ohlcv.copy()
    panel["date"] = panel["ts_event"].dt.normalize()

    if not stats.empty:
        stats = stats.copy()
        stats["date"] = stats["ts_event"].dt.normalize()

        def _extract_price_stat(code, colname):
            sub = stats[stats["stat_type"] == code].copy()
            if sub.empty:
                return pd.DataFrame(columns=["instrument_id", "date", colname])
            sub = sub.sort_values("ts_event").groupby(
                ["instrument_id", "date"], as_index=False
            ).tail(1)
            return sub[["instrument_id", "date", "price"]].rename(
                columns={"price": colname}
            )

        def _extract_qty_stat(code, colname):
            sub = stats[stats["stat_type"] == code].copy()
            if sub.empty:
                return pd.DataFrame(columns=["instrument_id", "date", colname])
            sub = sub.sort_values("ts_event").groupby(
                ["instrument_id", "date"], as_index=False
            ).tail(1)
            return sub[["instrument_id", "date", "quantity"]].rename(
                columns={"quantity": colname}
            )

        settle = _extract_price_stat(settlement_code, "settlement")
        cleared_vol = _extract_qty_stat(vol_code, "cleared_volume_stat")
        oi = _extract_qty_stat(oi_code, "open_interest_stat")

        panel = panel.merge(settle, on=["instrument_id", "date"], how="left")
        panel = panel.merge(cleared_vol, on=["instrument_id", "date"], how="left")
        panel = panel.merge(oi, on=["instrument_id", "date"], how="left")
    else:
        panel["settlement"] = np.nan
        panel["cleared_volume_stat"] = np.nan
        panel["open_interest_stat"] = np.nan

    meta_cols = [
        "dataset", "parent", "asset_class", "region",
        "instrument_id", "raw_symbol", "symbol", "expiration",
        "asset", "min_price_increment",
    ]
    meta_cols = [c for c in meta_cols if c in defs.columns]
    meta = defs[meta_cols].drop_duplicates("instrument_id")

    panel = panel.merge(meta, on="instrument_id", how="left")

    sort_cols = [c for c in ["dataset", "parent", "instrument_id", "date"] if c in panel.columns]
    panel = panel.sort_values(sort_cols)

    return panel.reset_index(drop=True)


def download_universe_daily_panel(client, roots, start, end, as_of_for_definitions=None):
    if as_of_for_definitions is None:
        as_of_for_definitions = start

    defs = fetch_definitions_for_roots(client, roots, as_of=as_of_for_definitions)

    panels = []
    for dataset in sorted(defs["dataset"].unique()):
        defs_subset = defs[defs["dataset"] == dataset].copy()
        instrument_ids = defs_subset["instrument_id"].tolist()
        if not instrument_ids:
            continue

        print(f"[INFO] Downloading OHLCV for dataset={dataset}, instruments={len(instrument_ids)}")
        ohlcv = fetch_ohlcv_1d(
            client=client,
            dataset=dataset,
            instrument_ids=instrument_ids,
            start=start,
            end=end,
        )
        if ohlcv.empty:
            print(f"[WARN] Skipping dataset={dataset} due to empty OHLCV.")
            continue

        print(f"[INFO] Downloading statistics for dataset={dataset}, instruments={len(instrument_ids)}")
        stats = fetch_statistics(
            client=client,
            dataset=dataset,
            instrument_ids=instrument_ids,
            start=start,
            end=end,
        )

        panel = build_daily_panel(defs_subset, ohlcv, stats)
        panels.append(panel)

    if not panels:
        raise RuntimeError("No data returned for any dataset in the universe.")

    full_panel = pd.concat(panels, ignore_index=True)
    full_panel = full_panel.sort_values(
        ["date", "asset_class", "parent", "instrument_id"]
    )

    return full_panel.reset_index(drop=True)


if __name__ == "__main__":
    client = make_historical_client("db-TDpqeHtULMkRBM6sRtkTYKJgYrp6h")

    start_date = "2015-01-01"
    end_date = "2025-01-01"

    print(f"[INFO] Starting download for {start_date} -> {end_date}")
    panel = download_universe_daily_panel(
        client=client,
        roots=GLBX_UNIVERSE,
        start=start_date,
        end=end_date,
    )

    out_path = "futures_glbx_daily.csv"
    panel.to_csv(out_path, index=False)
    print(f"[INFO] Saved {len(panel):,} rows to {out_path}")
