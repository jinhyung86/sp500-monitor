"""Free fundamentals: SEC EDGAR filings + Yahoo Finance fallback, persisted cache."""
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

CACHE_FILE = Path("fundamentals_cache.json")
SEC_HEADERS = {
    "User-Agent": os.getenv("SEC_USER_AGENT", "sp500-monitor/1.0 jinhyung86@users.noreply.github.com"),
    "Accept-Encoding": "gzip, deflate",
}
SEC_TAGS = {
    "revenue_usd": ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues",
                    "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax"),
    "operating_income_usd": ("OperatingIncomeLoss",),
    "net_income_usd": ("NetIncomeLoss", "ProfitLoss"),
}
DAY = 86400


def load_cache():
    try:
        value = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def save_cache(cache):
    CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def sec_annual(facts, tags):
    for tag in tags:
        units = facts.get("facts", {}).get("us-gaap", {}).get(tag, {}).get("units", {}).get("USD", [])
        valid = [x for x in units if x.get("form") in ("10-K", "10-K/A") and
                 x.get("fp") == "FY" and x.get("fy") and x.get("start") and x.get("end") and
                 300 <= (datetime.fromisoformat(x["end"]) - datetime.fromisoformat(x["start"])).days <= 400
                 and isinstance(x.get("val"), (int, float))]
        if valid:
            # Filing date selects latest restatement for the latest fiscal period.
            item = max(valid, key=lambda x: (x["end"], x.get("filed", "")))
            return item["val"], item["end"]
    return None, None


def sec_companyfacts(cik):
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{int(cik):010d}.json"
    response = requests.get(url, headers=SEC_HEADERS, timeout=18)
    response.raise_for_status()
    facts = response.json()
    out = {}
    dates = {}
    for key, tags in SEC_TAGS.items():
        value, date = sec_annual(facts, tags)
        if value is not None:
            out[key] = value
            dates[key] = date
    # Only compare values from the same fiscal year; never mix different years.
    if dates:
        anchor = dates.get("revenue_usd") or max(dates.values())
        for key, date in dates.items():
            if date != anchor:
                out.pop(key, None)
        out["statement_period"] = anchor
    return out


def yahoo_info(symbol):
    import yfinance as yf
    info = yf.Ticker(symbol.replace(".", "-")).info
    if not isinstance(info, dict):
        return {}
    out = {}
    for src, dest in (("trailingPE", "pe"), ("targetMeanPrice", "price_target_usd"),
                      ("sector", "sector"), ("industry", "industry"),
                      ("numberOfAnalystOpinions", "analyst_total")):
        value = info.get(src)
        if value is not None:
            out[dest] = value
    # Yahoo does not consistently provide analyst buy counts. Do not invent them.
    rec = info.get("recommendationKey")
    if isinstance(rec, str) and rec.lower() not in ("none", ""):
        out["analyst_consensus"] = rec.capitalize()
    return out


def enrich(result, constituents):
    symbols = sorted({item["symbol"] for group in (
        result["upper_box_watch_candidates"], result["lower_box_candidates"],
        result["new_breakout_signals"], *result["uptrend_candidates"].values())
        for item in group})
    cache = load_cache()
    now = time.time()
    records = cache.setdefault("stocks", {})
    old = {}
    try:
        old = json.loads(Path("result.json").read_text(encoding="utf-8")).get("fundamentals", {})
    except (OSError, ValueError, AttributeError):
        pass
    for symbol in symbols:
        if symbol not in records:
            records[symbol] = {"data": old.get(symbol, {}), "sec_at": 0, "yahoo_at": 0}
    index = {x["symbol"]: x for x in constituents}
    cik_map = cache.get("cik_map", {})
    sec_requests = 0
    yahoo_requests = 0
    errors = []
    # SEC ticker mapping is cached for 30 days.
    if now - cache.get("cik_at", 0) > 30 * DAY:
        try:
            response = requests.get("https://www.sec.gov/files/company_tickers.json",
                                    headers=SEC_HEADERS, timeout=18)
            response.raise_for_status()
            cik_map = {row["ticker"].replace("-", "."): row["cik_str"]
                       for row in response.json().values()}
            cache["cik_map"] = cik_map
            cache["cik_at"] = now
        except (requests.RequestException, ValueError, TypeError, KeyError) as exc:
            errors.append(f"SEC ticker map: {type(exc).__name__}")
    # SEC official annual financial statements, 7-day TTL, at most 20 stocks per run.
    for symbol in symbols:
        record = records[symbol]
        data = record.setdefault("data", {})
        fallback = index.get(symbol, {})
        data.setdefault("sector", fallback.get("sector"))
        data.setdefault("industry", fallback.get("industry"))
        if sec_requests >= 20:
            break
        if now - record.get("sec_at", 0) < 7 * DAY:
            continue
        cik = cik_map.get(symbol)
        if not cik:
            record["sec_at"] = now
            continue
        sec_requests += 1
        try:
            fresh = sec_companyfacts(cik)
            if fresh:
                data.update(fresh)
                data["financial_source"] = "SEC EDGAR"
            record["sec_at"] = now
        except (requests.RequestException, ValueError, TypeError) as exc:
            errors.append(f"SEC {symbol}: {type(exc).__name__}")
            record["sec_at"] = now - 6 * DAY  # retry tomorrow, not every hour
        time.sleep(0.13)
    # Unofficial Yahoo fields: small batch per run, 7-day TTL. Prioritize missing PER.
    ordered = sorted(symbols, key=lambda sym: (records[sym]["data"].get("pe") is not None,
                                                records[sym].get("yahoo_at", 0)))
    for symbol in ordered:
        if yahoo_requests >= 8:
            break
        record = records[symbol]
        if now - record.get("yahoo_at", 0) < 7 * DAY:
            continue
        yahoo_requests += 1
        try:
            fresh = yahoo_info(symbol)
            if fresh:
                record["data"].update({k: v for k, v in fresh.items() if v is not None})
                record["data"]["market_source"] = "Yahoo Finance (unofficial)"
            record["yahoo_at"] = now
        except Exception as exc:
            errors.append(f"Yahoo {symbol}: {type(exc).__name__}")
            record["yahoo_at"] = now - 6 * DAY
        time.sleep(0.3)
    for symbol in symbols:
        records[symbol]["data"].setdefault("source", "SEC EDGAR / Yahoo Finance")
    result["fundamentals"] = {symbol: records[symbol]["data"] for symbol in symbols}
    result["fundamentals_source"] = "SEC EDGAR + Yahoo Finance (unofficial), cached"
    result["fundamentals_api_usage"] = {
        "sec_companyfacts_requests_this_run": sec_requests,
        "yahoo_symbol_requests_this_run": yahoo_requests,
        "fmp_requests_this_run": 0,
        "cache_stocks": len(records),
        "errors": errors[:15],
    }
    save_cache(cache)
