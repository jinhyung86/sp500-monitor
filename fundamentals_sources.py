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
    ticker = yf.Ticker(symbol.replace(".", "-"))
    out = {}
    try:
        info = ticker.info
        if not isinstance(info, dict):
            info = {}
    except Exception:
        info = {}
    for src, dest in (("trailingPE", "pe"), ("targetMeanPrice", "price_target_usd"),
                      ("sector", "sector"), ("industry", "industry"),
                      ("numberOfAnalystOpinions", "analyst_total")):
        value = info.get(src)
        if value is not None:
            out[dest] = value
    # Annual statement fallback when SEC ticker mapping is unavailable.
    try:
        statement = ticker.income_stmt
        if statement is not None and not statement.empty:
            col = sorted(statement.columns, reverse=True)[0]
            values = statement[col]
            for key, aliases in (
                ("revenue_usd", ("Total Revenue", "Operating Revenue")),
                ("operating_income_usd", ("Operating Income",)),
                ("net_income_usd", ("Net Income", "Net Income Common Stockholders")),
            ):
                for alias in aliases:
                    if alias in values.index:
                        value = values.loc[alias]
                        if value is not None and str(value).lower() not in ("nan", "nat", "none"):
                            out[key] = float(value)
                            break
            if any(key in out for key in ("revenue_usd", "operating_income_usd", "net_income_usd")):
                out["statement_period"] = str(col.date()) if hasattr(col, "date") else str(col)[:10]
                out["financial_source"] = "Yahoo Finance annual income statement (unofficial)"
    except Exception:
        pass
    # Yahoo analyst annual revenue / EPS estimates. These are forecasts, not actuals.
    try:
        estimates = ticker.revenue_estimate
        if estimates is not None and not estimates.empty:
            for period in ("+1y", "0y"):
                if period in estimates.index:
                    value = estimates.loc[period].get("avg")
                    if value is not None and str(value).lower() not in ("nan", "none"):
                        out["estimated_revenue_usd"] = float(value)
                        out["estimate_period"] = period
                        out["estimate_source"] = "Yahoo Finance analyst revenue estimates (unofficial)"
                        break
    except Exception:
        pass
    try:
        earnings = ticker.earnings_estimate
        shares = info.get("sharesOutstanding")
        if earnings is not None and not earnings.empty and shares:
            period = out.get("estimate_period", "+1y")
            if period in earnings.index:
                eps = earnings.loc[period].get("avg")
                if eps is not None and str(eps).lower() not in ("nan", "none"):
                    # EPS times current shares is an approximation, NOT an analyst
                    # consensus forecast for aggregate net income.
                    out["estimated_net_income_usd"] = float(eps) * float(shares)
                    out["estimated_net_income_method"] = "EPS consensus x current shares (approximation)"
                    out.setdefault("estimate_period", period)
    except Exception:
        pass
    # Reconcile recommendation counts from one Yahoo snapshot only.
    try:
        trend = ticker.recommendations
        if trend is not None and not trend.empty:
            row = trend.iloc[0]
            strong = int(row.get("strongBuy", 0) or 0)
            buy = int(row.get("buy", 0) or 0)
            hold = int(row.get("hold", 0) or 0)
            sell = int(row.get("sell", 0) or 0)
            strong_sell = int(row.get("strongSell", 0) or 0)
            total = strong + buy + hold + sell + strong_sell
            if total > 0:
                out["analyst_buy_count"] = strong + buy
                out["analyst_total"] = total
                out["analyst_buy_pct"] = round((strong + buy) / total * 100, 1)
                out["analyst_count_source"] = "Yahoo Finance recommendation trend (unofficial)"
    except Exception:
        pass
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
    # Try alternate SEC host if www.sec.gov blocks GitHub Actions.
    # SEC publishes the same mapping on its data host.
    # SEC ticker mapping is cached for 30 days.
    if now - cache.get("cik_at", 0) > 30 * DAY:
        try:
            response = None
            last_error = None
            for url in ("https://www.sec.gov/files/company_tickers.json",
                        "https://www.sec.gov/files/company_tickers_exchange.json"):
                try:
                    response = requests.get(url, headers=SEC_HEADERS, timeout=18)
                    response.raise_for_status()
                    break
                except requests.RequestException as exc:
                    last_error = exc
                    response = None
            if response is None:
                raise last_error
            payload = response.json()
            if isinstance(payload, dict) and "fields" in payload and "data" in payload:
                fields = payload["fields"]
                ti, ci = fields.index("ticker"), fields.index("cik")
                cik_map = {str(row[ti]).replace("-", "."): row[ci] for row in payload["data"]}
            else:
                cik_map = {row["ticker"].replace("-", "."): row["cik_str"]
                           for row in payload.values()}
            cache["cik_map"] = cik_map
            cache["cik_at"] = now
        except (requests.RequestException, ValueError, TypeError, KeyError) as exc:
            errors.append(f"SEC ticker map: {type(exc).__name__}")
    # SEC official annual financial statements, 7-day TTL, at most 20 stocks per run.
    for symbol in symbols:
        record = records[symbol]
        data = record.setdefault("data", {})
        fallback = index.get(symbol, {})
        if not data.get("sector") and fallback.get("sector"):
            data["sector"] = fallback["sector"]
        if not data.get("industry") and fallback.get("industry"):
            data["industry"] = fallback["industry"]
        if sec_requests >= 20:
            break
        if now - record.get("sec_at", 0) < 7 * DAY:
            continue
        cik = cik_map.get(symbol)
        if not cik:
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
    # Prioritize missing actuals, forecasts, and analyst recommendation counts.
    ordered = sorted(symbols, key=lambda sym: (
        all(records[sym]["data"].get(k) is not None
            for k in ("revenue_usd", "operating_income_usd", "net_income_usd")),
        records[sym]["data"].get("estimated_revenue_usd") is not None,
        records[sym]["data"].get("analyst_buy_count") is not None,
        records[sym].get("yahoo_at", 0)))
    for symbol in ordered:
        if yahoo_requests >= 24:
            break
        record = records[symbol]
        # Retry incomplete financial statements sooner than complete records.
        has_financials = all(record["data"].get(k) is not None
                             for k in ("revenue_usd", "operating_income_usd", "net_income_usd"))
        has_forecast = record["data"].get("estimated_revenue_usd") is not None
        has_recommendations = record["data"].get("analyst_buy_count") is not None
        yahoo_ttl = 7 * DAY if (has_financials and has_forecast and has_recommendations) else 15 * 60
        if now - record.get("yahoo_at", 0) < yahoo_ttl:
            continue
        yahoo_requests += 1
        try:
            fresh = yahoo_info(symbol)
            if fresh:
                data = record["data"]
                new_period = fresh.get("statement_period")
                old_period = data.get("statement_period")
                if old_period and new_period and new_period < old_period:
                    fresh = {k: v for k, v in fresh.items()
                             if k not in ("revenue_usd", "operating_income_usd",
                                          "net_income_usd", "statement_period", "financial_source")}
                # Only update counts as a consistent snapshot. Keep previous
                # Yahoo snapshot if this request returns no recommendations.
                if "analyst_count_source" in fresh:
                    for key in ("analyst_buy_count", "analyst_buy_pct", "analyst_total"):
                        data[key] = None
                data.update({k: v for k, v in fresh.items() if v is not None})
                record["data"]["market_source"] = "Yahoo Finance (unofficial)"
            # Incomplete results remain eligible for an earlier retry.
            record["yahoo_at"] = now
        except Exception as exc:
            errors.append(f"Yahoo {symbol}: {type(exc).__name__}")
            record["yahoo_at"] = now - 6 * DAY
        time.sleep(0.3)
    for symbol in symbols:
        data = records[symbol]["data"]
        if data.get("analyst_count_source") != "Yahoo Finance recommendation trend (unofficial)":
            data["analyst_buy_count"] = None
            data["analyst_buy_pct"] = None
            data["analyst_total"] = None
        forecast_date = str(data.get("estimate_period") or "")
        actual_date = str(data.get("statement_period") or "")
        if forecast_date in ("0y", "+1y"):
            forecast_date = ""
        if forecast_date and actual_date and forecast_date[:4].isdigit() and actual_date[:4].isdigit():
            if int(forecast_date[:4]) - int(actual_date[:4]) > 2:
                for key in ("estimated_revenue_usd", "estimated_operating_income_usd",
                            "estimated_net_income_usd", "estimate_period"):
                    data[key] = None
        data.setdefault("source", "SEC EDGAR / Yahoo Finance")
    result["fundamentals"] = {symbol: records[symbol]["data"] for symbol in symbols}
    result["fundamentals_source"] = "SEC EDGAR + Yahoo Finance (unofficial), cached"
    missing_actuals = sum(not all(records[s]["data"].get(k) is not None
                                  for k in ("revenue_usd", "operating_income_usd", "net_income_usd"))
                          for s in symbols)
    missing_forecasts = sum(not any(records[s]["data"].get(k) is not None
                                    for k in ("estimated_revenue_usd", "estimated_operating_income_usd",
                                              "estimated_net_income_usd"))
                            for s in symbols)
    result["fundamentals_api_usage"] = {
        "missing_actuals": missing_actuals,
        "missing_forecasts": missing_forecasts,
        "sec_companyfacts_requests_this_run": sec_requests,
        "yahoo_symbol_requests_this_run": yahoo_requests,
        "fmp_requests_this_run": 0,
        "cache_stocks": len(records),
        "errors": errors[:15],
    }
    save_cache(cache)
