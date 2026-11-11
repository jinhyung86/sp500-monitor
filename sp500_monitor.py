import os
import json
import math
import time
from io import StringIO
from html import escape
from pathlib import Path
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from collections import defaultdict

import requests
import pandas as pd

# =========================
# Configuration
# =========================
ALPACA_BASE = "https://data.alpaca.markets"
SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
RESULT_FILE = Path("result.json")
REPORT_FILE = Path("report.md")
SYMBOL_FILE = Path("sp500_symbols.json")

BATCH_SIZE = 75
REQUEST_TIMEOUT = 20
MAX_RETRIES = 1
DAILY_LOOKBACK_DAYS = 300
BOX_PERIODS = (21, 42, 63, 126)
UPTREND_PERIODS = (5, 10, 20)
UPTREND_MIN_R2 = 0.85
UPTREND_MIN_SLOPE_PCT_PER_DAY = 0.70
UPTREND_MIN_RETURN_PCT = {5: 3.0, 10: 7.0, 20: 12.0}
# Second-stage quality filters (apply only to uptrend candidates).
UPTREND_VOLUME_GROWTH_MIN = 1.20
UPTREND_UP_DOWN_VOLUME_MIN = 1.20
UPTREND_MAX_DRAWDOWN_FROM_20D_CLOSE_HIGH = 0.05
UPTREND_MAX_SINGLE_DAY_GAIN = 0.10
UPTREND_SCORE_MIN = 70.0
BOX_MAX_WIDTH = 0.20
BOX_FIRST_LAST_MAX_MOVE = 0.07
DAILY_VOLUME_RATIO_MIN = 1.50
STRONG_VOLUME_RATIO = 2.00
LOWER_BOX_MAX_POSITION = 0.30
UPPER_BOX_WATCH_MIN_POSITION = 0.70
UPPER_BOX_WATCH_MAX_POSITION = 1.00
UPPER_BOX_WATCH_MIN_SCORE = 85.0
UP_DOWN_VOLUME_RATIO_MIN = 1.30
UP_DOWN_VOLUME_RATIO_EX_MAX_DAY_MIN = 1.10

NY = ZoneInfo("America/New_York")
KST = ZoneInfo("Asia/Seoul")
UTC = timezone.utc

API_KEY = os.environ.get("ALPACA_API_KEY")
SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")
if not API_KEY or not SECRET_KEY:
    raise RuntimeError("ALPACA_API_KEY / ALPACA_SECRET_KEY environment variables are required")

HEADERS = {
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": SECRET_KEY,
    "User-Agent": "sp500-monitor/1.0",
}

session = requests.Session()
session.headers.update(HEADERS)


# =========================
# Helpers
# =========================
def now_utc():
    return datetime.now(UTC)


def iso(dt):
    return dt.isoformat(timespec="seconds")


def chunks(items, n):
    for i in range(0, len(items), n):
        yield items[i:i+n]


def safe_float(v):
    try:
        if v is None:
            return None
        x = float(v)
        if math.isnan(x):
            return None
        return x
    except Exception:
        return None


def request_json(url, params=None, *, context="request"):
    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            r = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                return r.json(), None

            retry_after = r.headers.get("Retry-After")
            msg = f"HTTP {r.status_code}"
            if retry_after:
                msg += f" Retry-After={retry_after}"

            # Do not bypass 403/429. Retry at most once only if Retry-After is short.
            if r.status_code in (403, 429):
                if attempt < MAX_RETRIES and retry_after:
                    try:
                        wait_s = min(float(retry_after), 15.0)
                        time.sleep(max(wait_s, 0.0))
                        continue
                    except Exception:
                        pass
                return None, f"{context}: {msg}"

            if 500 <= r.status_code < 600 and attempt < MAX_RETRIES:
                time.sleep(1.5)
                continue

            body = r.text[:300].replace("\n", " ")
            return None, f"{context}: {msg} {body}"
        except requests.RequestException as e:
            last_error = str(e)
            if attempt < MAX_RETRIES:
                time.sleep(1.0)
                continue
    return None, f"{context}: {last_error or 'unknown error'}"


def normalize_symbol(symbol):
    # Alpaca accepts common class-share symbols such as BRK.B / BF.B.
    return str(symbol).strip().upper()


# =========================
# S&P 500 constituents
# =========================
def load_sp500_constituents():
    source = "wikipedia"
    try:
        r = requests.get(
            SP500_URL,
            timeout=20,
            headers={"User-Agent": "Mozilla/5.0 sp500-monitor/1.0"},
        )
        r.raise_for_status()
        tables = pd.read_html(StringIO(r.text))
        df = tables[0]
        symbol_col = "Symbol"
        name_col = "Security"
        rows = []
        for _, row in df.iterrows():
            symbol = normalize_symbol(row[symbol_col])
            name = str(row[name_col]).strip()
            rows.append({"symbol": symbol, "name": name, "sector": str(row.get("GICS Sector", "")).strip(), "industry": str(row.get("GICS Sub-Industry", "")).strip()})
        if len(rows) < 490:
            raise RuntimeError(f"unexpected constituent count: {len(rows)}")
        SYMBOL_FILE.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        return rows, source, None
    except Exception as e:
        if SYMBOL_FILE.exists():
            rows = json.loads(SYMBOL_FILE.read_text(encoding="utf-8"))
            return rows, "cached sp500_symbols.json", f"Wikipedia fetch failed: {e}"
        raise


# =========================
# Alpaca data fetch
# =========================
def fetch_daily_bars(symbols, start_dt, end_dt, stats, failures):
    all_bars = defaultdict(list)
    url = f"{ALPACA_BASE}/v2/stocks/bars"

    for batch_no, batch in enumerate(chunks(symbols, BATCH_SIZE), start=1):
        stats["requested_symbols"].update(batch)
        page_token = None
        while True:
            params = {
                "symbols": ",".join(batch),
                "timeframe": "1Day",
                "start": start_dt.date().isoformat(),
                "end": end_dt.isoformat().replace("+00:00", "Z"),
                "limit": 10000,
                "adjustment": "split",
                "feed": "sip",
                "sort": "asc",
            }
            if page_token:
                params["page_token"] = page_token

            data, err = request_json(url, params, context=f"daily batch {batch_no}")
            stats["daily_http_requests"] += 1
            if err:
                for s in batch:
                    failures.setdefault(s, []).append(err)
                break

            bars_map = data.get("bars") or {}
            for symbol, bars in bars_map.items():
                all_bars[normalize_symbol(symbol)].extend(bars or [])

            page_token = data.get("next_page_token")
            if not page_token:
                break

    for s in symbols:
        all_bars[s] = sorted(all_bars.get(s, []), key=lambda x: x.get("t", ""))
    return all_bars


def fetch_snapshots(symbols, stats, failures):
    snapshots = {}
    url = f"{ALPACA_BASE}/v2/stocks/snapshots"
    for batch_no, batch in enumerate(chunks(symbols, BATCH_SIZE), start=1):
        params = {
            "symbols": ",".join(batch),
            "feed": "delayed_sip",
        }
        data, err = request_json(url, params, context=f"snapshot batch {batch_no}")
        stats["snapshot_http_requests"] += 1
        if err:
            # Snapshot failure does not invalidate daily history.
            for s in batch:
                failures.setdefault(s, []).append(err)
            continue
        if isinstance(data, dict):
            for symbol, snap in data.items():
                snapshots[normalize_symbol(symbol)] = snap
    return snapshots


def fetch_5m_bars(symbol, start_dt, end_dt, stats):
    url = f"{ALPACA_BASE}/v2/stocks/{symbol}/bars"
    bars = []
    page_token = None
    while True:
        params = {
            "timeframe": "5Min",
            "start": start_dt.isoformat().replace("+00:00", "Z"),
            "end": end_dt.isoformat().replace("+00:00", "Z"),
            "limit": 10000,
            "adjustment": "split",
            "feed": "sip",
            "sort": "asc",
        }
        if page_token:
            params["page_token"] = page_token
        data, err = request_json(url, params, context=f"5m {symbol}")
        stats["intraday_http_requests"] += 1
        if err:
            return None, err
        bars.extend(data.get("bars") or [])
        page_token = data.get("next_page_token")
        if not page_token:
            break
    bars.sort(key=lambda x: x.get("t", ""))
    return bars, None


# =========================
# Market / calculations
# =========================
def parse_bar_date(bar):
    t = bar.get("t")
    if not t:
        return None
    dt = datetime.fromisoformat(t.replace("Z", "+00:00"))
    return dt.astimezone(NY).date()


def regular_market_open(now_ny):
    if now_ny.weekday() >= 5:
        return False
    mins = now_ny.hour * 60 + now_ny.minute
    return 9 * 60 + 30 <= mins < 16 * 60


def completed_daily_bars(bars, now_ny):
    """Exclude today's possibly-partial daily bar while regular session is open."""
    if not bars:
        return []
    if regular_market_open(now_ny):
        today = now_ny.date()
        return [b for b in bars if parse_bar_date(b) != today]
    return list(bars)


def box_metrics(history, period):
    if len(history) < period:
        return None
    w = history[-period:]
    highs = [safe_float(x.get("h")) for x in w]
    lows = [safe_float(x.get("l")) for x in w]
    closes = [safe_float(x.get("c")) for x in w]
    if any(x is None for x in highs + lows + closes):
        return None
    box_high = max(highs)
    box_low = min(lows)
    if box_low <= 0:
        return None
    width = box_high / box_low - 1.0
    first_close = closes[0]
    last_close = closes[-1]
    move = last_close / first_close - 1.0 if first_close else None
    valid = (
        width <= BOX_MAX_WIDTH
        and move is not None
        and abs(move) <= BOX_FIRST_LAST_MAX_MOVE
    )
    return {
        "period": period,
        "box_high": box_high,
        "box_low": box_low,
        "box_width": width,
        "first_close": first_close,
        "last_close": last_close,
        "first_last_move": move,
        "valid_box": valid,
    }


def mean(values):
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def daily_breakout_for_period(completed, period):
    # completed includes the breakout day as the final bar.
    if len(completed) < period + 1:
        return None
    breakout = completed[-1]
    history = completed[:-1]
    metrics = box_metrics(history, period)
    if not metrics:
        return None

    close = safe_float(breakout.get("c"))
    volume = safe_float(breakout.get("v"))
    prev_close = safe_float(history[-1].get("c"))
    prior20_vol = [safe_float(x.get("v")) for x in history[-20:]]
    avg20 = mean(prior20_vol)
    volume_ratio = (volume / avg20) if volume is not None and avg20 else None

    confirmed = bool(
        metrics["valid_box"]
        and close is not None
        and prev_close is not None
        and close > metrics["box_high"]
        and close > prev_close
        and volume_ratio is not None
        and volume_ratio >= DAILY_VOLUME_RATIO_MIN
    )
    return {
        **metrics,
        "breakout_date": str(parse_bar_date(breakout)),
        "close": close,
        "prev_close": prev_close,
        "volume": volume,
        "avg20_volume": avg20,
        "volume_ratio": volume_ratio,
        "confirmed": confirmed,
        "strong": bool(volume_ratio is not None and volume_ratio >= STRONG_VOLUME_RATIO),
    }


def upper_box_watch_for_period(completed, period):
    """Find valid sideways boxes whose latest close sits at 70-100% of box height.

    Momentum/volume metrics rank candidates; they do not exclude early setups.
    """
    metrics = box_metrics(completed, period)
    if not metrics or not metrics["valid_box"]:
        return None
    span = metrics["box_high"] - metrics["box_low"]
    if span <= 0:
        return None
    closes = [safe_float(b.get("c")) for b in completed[-max(period, 25):]]
    volumes = [safe_float(b.get("v")) for b in completed[-max(period, 25):]]
    if any(v is None for v in closes + volumes):
        return None
    close = closes[-1]
    position = (close - metrics["box_low"]) / span
    if not UPPER_BOX_WATCH_MIN_POSITION <= position <= UPPER_BOX_WATCH_MAX_POSITION:
        return None
    recent_5_return_pct = ((close / closes[-5] - 1) * 100) if len(closes) >= 5 and closes[-5] > 0 else None
    ma5 = mean(closes[-5:]) if len(closes) >= 5 else None
    ma20 = mean(closes[-20:]) if len(closes) >= 20 else None
    ma5_rising = len(closes) >= 6 and mean(closes[-5:]) > mean(closes[-6:-1])
    recent5_volume = mean(volumes[-5:]) if len(volumes) >= 5 else None
    previous20_volume = mean(volumes[-25:-5]) if len(volumes) >= 25 else None
    volume_ratio = recent5_volume / previous20_volume if previous20_volume and previous20_volume > 0 else None
    # Score ranks early setups; none of these indicators is a mandatory filter.
    score = round(
        position * 40
        + (20 if recent_5_return_pct is not None and recent_5_return_pct > 0 else 0)
        + (15 if ma5 is not None and ma20 is not None and ma5 > ma20 else 0)
        + (10 if ma5_rising else 0)
        + (15 if volume_ratio is not None and volume_ratio >= 1.1 else 0), 1
    )
    tier = "A" if position >= .95 else ("B" if position >= .85 else "C")
    return {
        **metrics, "as_of_date": str(parse_bar_date(completed[-1])),
        "close": close, "box_position": position, "tier": tier,
        "recent_5_return_pct": round(recent_5_return_pct, 3) if recent_5_return_pct is not None else None,
        "ma5_above_ma20": bool(ma5 is not None and ma20 is not None and ma5 > ma20),
        "ma5_rising": ma5_rising,
        "volume_growth_ratio": round(volume_ratio, 3) if volume_ratio is not None else None,
        "watch_score": score,
    }


def lower_box_for_period(completed, period):
    """Find a sideways box currently sitting in the lower 30% with constructive volume."""
    if len(completed) < period:
        return None
    w = completed[-period:]
    metrics = box_metrics(completed, period)
    if not metrics or not metrics["valid_box"]:
        return None

    closes = [safe_float(x.get("c")) for x in w]
    volumes = [safe_float(x.get("v")) for x in w]
    if any(x is None for x in closes + volumes):
        return None

    last_close = closes[-1]
    span = metrics["box_high"] - metrics["box_low"]
    if span <= 0:
        return None
    box_position = (last_close - metrics["box_low"]) / span

    # Classify each day by close vs prior close. The first day has no in-window prior close.
    classified = []
    for i in range(1, len(w)):
        c0 = closes[i - 1]
        c1 = closes[i]
        v = volumes[i]
        if c1 > c0:
            classified.append(("up", v, i))
        elif c1 < c0:
            classified.append(("down", v, i))

    up_vols = [v for side, v, _ in classified if side == "up"]
    down_vols = [v for side, v, _ in classified if side == "down"]
    if not up_vols or not down_vols:
        return None

    up_avg = mean(up_vols)
    down_avg = mean(down_vols)
    up_down_ratio = (up_avg / down_avg) if down_avg else None

    # Remove the single largest-volume classified day, then recalculate.
    max_entry = max(classified, key=lambda x: x[1]) if classified else None
    filtered = [x for x in classified if x is not max_entry]
    up2 = [v for side, v, _ in filtered if side == "up"]
    down2 = [v for side, v, _ in filtered if side == "down"]
    up_avg_ex = mean(up2)
    down_avg_ex = mean(down2)
    ratio_ex = (up_avg_ex / down_avg_ex) if up_avg_ex is not None and down_avg_ex else None

    # Guard against a fresh support break or a persistent short-term slide being mislabeled as "lower box".
    prior_support_window = closes[max(0, len(closes) - 11):-1]
    prior_support = min(prior_support_window) if prior_support_window else None
    fresh_support_break = bool(prior_support is not None and last_close < prior_support * 0.995)

    recent = closes[-10:] if len(closes) >= 10 else closes
    recent_move = (recent[-1] / recent[0] - 1.0) if len(recent) >= 2 and recent[0] else 0.0
    down_steps = sum(1 for i in range(1, len(recent)) if recent[i] < recent[i - 1])
    persistent_downtrend = bool(len(recent) >= 8 and recent_move <= -0.04 and down_steps >= 6)
    trend_guard_pass = not fresh_support_break and not persistent_downtrend

    qualifies = bool(
        box_position <= LOWER_BOX_MAX_POSITION
        and up_down_ratio is not None
        and up_down_ratio >= UP_DOWN_VOLUME_RATIO_MIN
        and ratio_ex is not None
        and ratio_ex >= UP_DOWN_VOLUME_RATIO_EX_MAX_DAY_MIN
        and trend_guard_pass
    )

    return {
        **metrics,
        "as_of_date": str(parse_bar_date(w[-1])),
        "close": last_close,
        "box_position": box_position,
        "up_avg_volume": up_avg,
        "down_avg_volume": down_avg,
        "up_down_volume_ratio": up_down_ratio,
        "up_down_volume_ratio_ex_max_day": ratio_ex,
        "fresh_support_break": fresh_support_break,
        "persistent_downtrend": persistent_downtrend,
        "trend_guard_pass": trend_guard_pass,
        "qualifies": qualifies,
    }


def uptrend_for_period(completed, period):
    """Detect a rising close-price trend in completed daily bars only.

    A down day is allowed. Regression R² measures trend consistency.
    """
    if len(completed) < period:
        return None
    window = completed[-period:]
    closes = [safe_float(bar.get("c")) for bar in window]
    if any(price is None or not math.isfinite(price) or price <= 0 for price in closes):
        return None

    n = len(closes)
    x_mean = (n - 1) / 2.0
    y_mean = sum(closes) / n
    ss_x = sum((i - x_mean) ** 2 for i in range(n))
    ss_y = sum((price - y_mean) ** 2 for price in closes)
    if ss_x <= 0 or ss_y <= 0:
        return None
    covariance = sum((i - x_mean) * (price - y_mean) for i, price in enumerate(closes))
    slope = covariance / ss_x
    r_squared = max(0.0, min(1.0, covariance ** 2 / (ss_x * ss_y)))
    return_pct = (closes[-1] / closes[0] - 1.0) * 100.0
    slope_pct_per_day = slope / y_mean * 100.0
    if (
        slope <= 0
        or r_squared < UPTREND_MIN_R2
        or slope_pct_per_day < UPTREND_MIN_SLOPE_PCT_PER_DAY
        or return_pct < UPTREND_MIN_RETURN_PCT[period]
    ):
        return None

    return {
        "period": period,
        "as_of_date": str(parse_bar_date(window[-1])),
        "start_date": str(parse_bar_date(window[0])),
        "start_close": closes[0],
        "close": closes[-1],
        "return_pct": round(return_pct, 3),
        "r2": round(r_squared, 4),
        "slope_pct_per_day": round(slope_pct_per_day, 4),
        "up_days": sum(closes[i] > closes[i - 1] for i in range(1, n)),
        "down_days": sum(closes[i] < closes[i - 1] for i in range(1, n)),
    }


def uptrend_quality(completed, period):
    """Score a qualifying uptrend using completed bars, without extra API calls.

    Requires 25 completed bars: 5-day volume compared with preceding 20 days.
    All volume checks use daily bars, and flat days are excluded from up/down ratio.
    """
    if len(completed) < max(25, period):
        return None
    window = completed[-25:]
    closes = [safe_float(b.get("c")) for b in window]
    volumes = [safe_float(b.get("v")) for b in window]
    if any(v is None or not math.isfinite(v) or v <= 0 for v in closes + volumes):
        return None

    ma5 = mean(closes[-5:])
    ma10 = mean(closes[-10:])
    ma20 = mean(closes[-20:])
    ma_aligned = ma5 > ma10 > ma20
    volume_growth = mean(volumes[-5:]) / mean(volumes[-25:-5])
    high20 = max(closes[-20:])
    drawdown = 1.0 - closes[-1] / high20

    # Use the same evaluation window as the selected 5/10/20-day trend.
    period_closes = closes[-period:]
    period_volumes = volumes[-period:]
    up_vol = [period_volumes[i] for i in range(1, period)
              if period_closes[i] > period_closes[i - 1]]
    down_vol = [period_volumes[i] for i in range(1, period)
                if period_closes[i] < period_closes[i - 1]]
    volume_ratio = mean(up_vol) / mean(down_vol) if up_vol and down_vol else None
    # All-up periods have no down-day comparator: don't fabricate a ratio.
    volume_balance_ok = bool(volume_ratio is not None and volume_ratio >= UPTREND_UP_DOWN_VOLUME_MIN)
    max_day_gain = max(period_closes[i] / period_closes[i-1] - 1
                       for i in range(1, period))
    no_spike = max_day_gain <= UPTREND_MAX_SINGLE_DAY_GAIN
    passes = (ma_aligned and volume_growth >= UPTREND_VOLUME_GROWTH_MIN
              and volume_balance_ok
              and drawdown <= UPTREND_MAX_DRAWDOWN_FROM_20D_CLOSE_HIGH
              and no_spike)
    return {
        "ma5": round(ma5, 4), "ma10": round(ma10, 4), "ma20": round(ma20, 4),
        "ma_aligned": ma_aligned,
        "volume_growth_ratio": round(volume_growth, 4),
        "up_down_volume_ratio": round(volume_ratio, 4) if volume_ratio is not None else None,
        "drawdown_from_20d_close_high_pct": round(drawdown * 100, 3),
        "max_single_day_gain_pct": round(max_day_gain * 100, 3),
        "quality_filters_pass": passes,
    }


def uptrend_quality_score(trend, quality):
    """Transparent 100-point ranking, not a forecast of future returns."""
    trend_score = 30 * min(1.0, max(0.0, (trend["r2"] - 0.60) / 0.40))
    vol_growth = quality["volume_growth_ratio"]
    vol_balance = quality["up_down_volume_ratio"] or 0.0
    volume_score = 15 * min(1.0, max(0.0, (vol_growth - 1.0) / 0.50))
    volume_score += 15 * min(1.0, max(0.0, (vol_balance - 1.0) / 0.50))
    ma_score = 20.0 if quality["ma_aligned"] else 0.0
    drawdown = quality["drawdown_from_20d_close_high_pct"]
    high_score = 20 * min(1.0, max(0.0, (10.0 - drawdown) / 10.0))
    return round(trend_score + volume_score + ma_score + high_score, 1)


def snapshot_price(snapshot):
    if not snapshot:
        return None
    for key, field in (("minuteBar", "c"), ("latestTrade", "p"), ("dailyBar", "c")):
        obj = snapshot.get(key) or {}
        p = safe_float(obj.get(field))
        if p is not None:
            return p
    return None


def snapshot_day_high(snapshot):
    return safe_float((snapshot or {}).get("dailyBar", {}).get("h"))


def intraday_candidate(completed_history, snapshot, period):
    if len(completed_history) < period:
        return None
    metrics = box_metrics(completed_history, period)
    if not metrics or not metrics["valid_box"]:
        return None
    price = snapshot_price(snapshot)
    prev_close = safe_float(completed_history[-1].get("c"))
    if price is None or prev_close is None:
        return None
    if price > metrics["box_high"] and price > prev_close:
        return {**metrics, "price": price, "prev_close": prev_close}
    return None


def filter_regular_5m(bars):
    sessions = defaultdict(list)
    for b in bars:
        t = b.get("t")
        if not t:
            continue
        dt = datetime.fromisoformat(t.replace("Z", "+00:00")).astimezone(NY)
        mins = dt.hour * 60 + dt.minute
        if dt.weekday() < 5 and 9 * 60 + 30 <= mins < 16 * 60:
            sessions[dt.date()].append((dt, b))
    for d in sessions:
        sessions[d].sort(key=lambda x: x[0])
    return sessions


def validate_intraday_volume(symbol, bars, now_ny):
    sessions = filter_regular_5m(bars)
    today = now_ny.date()
    current = sessions.get(today, [])
    if not current:
        # delayed SIP can still have current-session bars; if not, use latest session only
        if not sessions:
            return None, "no regular-session 5m bars"
        latest_day = max(sessions)
        if latest_day != today:
            return None, f"no current-session 5m bars (latest={latest_day})"
        current = sessions[latest_day]

    last_dt = current[-1][0]
    cutoff_minute = last_dt.hour * 60 + last_dt.minute
    current_cum = sum(safe_float(b.get("v")) or 0.0 for _, b in current)

    prior_days = sorted(d for d in sessions if d < today)[-20:]
    prior_cums = []
    for d in prior_days:
        cum = 0.0
        has = False
        for dt, b in sessions[d]:
            minute = dt.hour * 60 + dt.minute
            if minute <= cutoff_minute:
                cum += safe_float(b.get("v")) or 0.0
                has = True
        if has:
            prior_cums.append(cum)

    if len(prior_cums) < 10:
        return None, f"insufficient comparable sessions: {len(prior_cums)}"

    avg = mean(prior_cums)
    ratio = current_cum / avg if avg else None
    return {
        "symbol": symbol,
        "last_5m_time_et": last_dt.isoformat(timespec="minutes"),
        "current_cum_volume": current_cum,
        "prior_session_count": len(prior_cums),
        "same_time_avg_volume": avg,
        "volume_ratio": ratio,
        "confirmed": bool(ratio is not None and ratio >= DAILY_VOLUME_RATIO_MIN),
        "strong": bool(ratio is not None and ratio >= STRONG_VOLUME_RATIO),
    }, None


# =========================
# Report generation
# =========================
def load_previous_state():
    if not RESULT_FILE.exists():
        return {"seen_signal_keys": []}
    try:
        d = json.loads(RESULT_FILE.read_text(encoding="utf-8"))
        return {"seen_signal_keys": d.get("seen_signal_keys", [])}
    except Exception:
        return {"seen_signal_keys": []}


def signal_key(sig):
    return f"{sig['symbol']}:{sig['period']}:{sig['box_high']:.4f}:{sig.get('breakout_date') or sig.get('last_5m_time_et','intraday')}"


# Optional fundamentals/analyst enrichment. Requires FMP_API_KEY GitHub secret.
# Only selected stocks are queried; missing/unauthorized data remain N/A.
FMP_API_KEY = os.environ.get("FMP_API_KEY", "").strip()
FMP_URL = "https://financialmodelingprep.com/stable"
FMP_CACHE = Path("fmp_cache.json")
FMP_DAILY_BUDGET = 6  # Conservative cap; 250/day free-plan quota is shared.
FMP_CACHE_TTL_DAYS = 30
FMP_ENDPOINTS = ("profile", "ratios-ttm", "grades-consensus", "price-target-consensus", "income-statement", "analyst-estimates")

THEME_KO = {
    "Utilities": "유틸리티", "Financial Services": "금융", "Technology": "기술",
    "Healthcare": "헬스케어", "Consumer Cyclical": "경기소비재",
    "Consumer Defensive": "필수소비재", "Industrials": "산업재",
    "Real Estate": "부동산", "Energy": "에너지",
    "Communication Services": "통신·미디어", "Basic Materials": "소재",
    "Regulated Electric": "규제 전력", "Utilities - Regulated Electric": "규제 전력",
    "Utilities - Renewable": "재생에너지", "Renewable Utilities": "재생에너지",
    "Insurance - Life": "생명보험", "Insurance - Property & Casualty": "손해보험",
    "REIT - Retail": "리테일 리츠", "REIT - Residential": "주거용 리츠",
    "REIT - Specialty": "특수목적 리츠", "REIT - Office": "오피스 리츠",
    "REIT - Industrial": "물류·산업 리츠", "REIT - Healthcare Facilities": "헬스케어 리츠",
    "Semiconductors": "반도체", "Software - Infrastructure": "인프라 소프트웨어",
    "Software - Application": "응용 소프트웨어", "Information Technology Services": "IT 서비스",
    "Biotechnology": "바이오", "Medical Devices": "의료기기",
    "Medical Instruments & Supplies": "의료장비·소모품",
    "Drug Manufacturers - General": "제약", "Drug Manufacturers - Specialty & Generic": "전문·제네릭 의약품",
    "Aerospace & Defense": "항공우주·방산", "Farm & Heavy Construction Machinery": "농기계·중장비",
    "Entertainment": "엔터테인먼트", "Internet Content & Information": "인터넷 플랫폼",
    "Internet Retail": "온라인 유통", "Auto Manufacturers": "자동차 제조",
    "Banks - Diversified": "대형은행", "Capital Markets": "자본시장",
    "Oil & Gas Integrated": "종합 석유·가스", "Oil & Gas E&P": "석유·가스 탐사·생산",
    "Discount Stores": "할인 유통", "Agricultural - Machinery": "농기계",
    "Medical - Devices": "의료기기", "Home Improvement": "주택개선·건자재 유통",
    "Medical - Distribution": "의료 유통", "Medical - Healthcare Plans": "건강보험",
    "Communication Equipment": "통신장비", "Medical - Instruments & Supplies": "의료기기·소모품",
    "Specialty Business Services": "전문 기업서비스",
    "Healthcare Plans": "건강보험", "Diagnostics & Research": "진단·연구",
    "Medical Distribution": "의료 유통", "Electronic Components": "전자부품",
}

def theme_korean(industry, sector):
    for name in (industry, sector):
        if isinstance(name, str) and name.strip() in THEME_KO:
            return THEME_KO[name.strip()]
    return ""

def fmp_number(data, *keys):
    if not isinstance(data, dict):
        return None
    for key in keys:
        v = safe_float(data.get(key))
        if v is not None and math.isfinite(v):
            return v
    return None

def fmp_enrich(result):
    symbols = sorted({
        item["symbol"] for group in (
            result["upper_box_watch_candidates"], result["lower_box_candidates"],
            *result["uptrend_candidates"].values(), result["new_breakout_signals"]
        ) for item in group
    })
    cache = {"stocks": {}, "daily_usage": {}, "blocked": {}}
    if FMP_CACHE.exists():
        try:
            stored = json.loads(FMP_CACHE.read_text(encoding="utf-8"))
            if isinstance(stored, dict) and isinstance(stored.get("stocks"), dict):
                cache.update(stored)
        except (OSError, ValueError):
            pass
    # Preserve existing results when migrating from the old uncached version.
    if RESULT_FILE.exists():
        try:
            old = json.loads(RESULT_FILE.read_text(encoding="utf-8")).get("fundamentals", {})
            for symbol, data in old.items():
                if symbol not in cache["stocks"] and isinstance(data, dict):
                    cache["stocks"][symbol] = {"data": data, "checked": {}}
        except (OSError, ValueError, AttributeError):
            pass
    today = datetime.now(KST).date().isoformat()
    now = time.time()
    usage = cache["daily_usage"]
    usage = {day: n for day, n in usage.items() if day >= today}
    cache["daily_usage"] = usage
    used = int(usage.get(today, 0))
    blocked = cache["blocked"]
    stocks = cache["stocks"]
    # All HTTP attempts count toward the budget; 401/402/403 endpoints are disabled
    # for 30 days, avoiding repeated paid-plan errors.
    for symbol in symbols:
        entry = stocks.setdefault(symbol, {"data": {}, "checked": {}})
        entry.setdefault("data", {})
        entry.setdefault("checked", {})
        if not FMP_API_KEY:
            continue
        for endpoint in FMP_ENDPOINTS:
            if used >= FMP_DAILY_BUDGET:
                break
            if now - entry["checked"].get(endpoint, 0) < FMP_CACHE_TTL_DAYS * 86400:
                continue
            if now - blocked.get(endpoint, 0) < 30 * 86400:
                continue
            params = {"symbol": symbol, "apikey": FMP_API_KEY}
            if endpoint == "analyst-estimates":
                params.update({"period": "annual", "page": 0, "limit": 10})
            used += 1
            usage[today] = used
            entry["checked"][endpoint] = now
            try:
                response = requests.get(FMP_URL + "/" + endpoint, params=params, timeout=12)
                if response.status_code in (401, 402, 403):
                    blocked[endpoint] = now
                    continue
                if response.status_code == 429:
                    used = FMP_DAILY_BUDGET
                    usage[today] = used
                    break
                if response.status_code != 200:
                    continue
                rows = response.json()
                data = rows[0] if isinstance(rows, list) and rows else (rows if isinstance(rows, dict) else None)
                if endpoint == "analyst-estimates" and isinstance(rows, list):
                    data = next((x for x in rows if isinstance(x, dict) and
                                 str(x.get("date", "")) >= today), None)
                if not isinstance(data, dict):
                    continue
                out = entry["data"]
                if endpoint == "profile":
                    out.update({"sector": data.get("sector"), "industry": data.get("industry")})
                elif endpoint == "ratios-ttm":
                    out.update({"pe": fmp_number(data, "priceToEarningsRatioTTM", "peRatioTTM"),
                                "pb": fmp_number(data, "priceToBookRatioTTM", "pbRatioTTM"),
                                "ps": fmp_number(data, "priceToSalesRatioTTM", "priceToSalesRatio")})
                elif endpoint == "grades-consensus":
                    counts = [fmp_number(data, k) for k in ("strongBuy", "buy", "hold", "sell", "strongSell")]
                    total = sum(counts) if all(x is not None for x in counts) else None
                    buys = counts[0] + counts[1] if counts[0] is not None and counts[1] is not None else None
                    out.update({"analyst_consensus": data.get("consensus"),
                                "analyst_buy_count": int(buys) if buys is not None else None,
                                "analyst_total": int(total) if total else None,
                                "analyst_buy_pct": round(buys / total * 100, 1) if total and buys is not None else None})
                elif endpoint == "price-target-consensus":
                    out["price_target_usd"] = fmp_number(data, "targetConsensus")
                elif endpoint == "income-statement":
                    out.update({"revenue_usd": fmp_number(data, "revenue"),
                                "operating_income_usd": fmp_number(data, "operatingIncome"),
                                "net_income_usd": fmp_number(data, "netIncome"),
                                "statement_period": data.get("date")})
                elif endpoint == "analyst-estimates":
                    out.update({"estimated_revenue_usd": fmp_number(data, "revenueAvg"),
                                "estimated_operating_income_usd": fmp_number(data, "ebitAvg"),
                                "estimated_net_income_usd": fmp_number(data, "netIncomeAvg"),
                                "estimate_period": data.get("date")})
                out["source"] = "Financial Modeling Prep"
                out["fetched_utc"] = iso(now_utc())
            except (requests.RequestException, ValueError, TypeError):
                pass
    result["fundamentals"] = {symbol: stocks.get(symbol, {}).get("data", {}) for symbol in symbols}
    result["fundamentals_source"] = "Financial Modeling Prep (cached)" if FMP_API_KEY else "unavailable: FMP_API_KEY not configured"
    result["fundamentals_api_usage"] = {"date_kst": today, "requests_used_today_by_monitor": used, "daily_budget": FMP_DAILY_BUDGET}
    # Commit this file with result.json so the cache persists across GitHub runners.
    FMP_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

def fmt_fund(v, decimals=1, suffix=""):
    n = safe_float(v)
    return f"{n:,.{decimals}f}{suffix}" if n is not None else ""

def fmt_billions(v):
    n = safe_float(v)
    return f"{n / 1_000_000_000:,.2f}B" if n is not None else ""

def append_fundamental_report(lines, result):
    data = result.get("fundamentals", {})
    if not data:
        return
    lines.append("")
    lines.append("## 종목별 테마·애널리스트·가치평가 (USD)")
    lines.append("| 티커 | 업종/테마 | 투자의견 | 매수/전체 | 매수비율 | 목표가($) | 상승여력 | PER |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    price_map = {}
    for group in (result["upper_box_watch_candidates"], result["lower_box_candidates"], *result["uptrend_candidates"].values()):
        for item in group:
            price_map[item["symbol"]] = item.get("close", price_map.get(item["symbol"]))
    for symbol, x in sorted(data.items()):
        if not x:
            x = {}
        price = safe_float(price_map.get(symbol))
        target = safe_float(x.get("price_target_usd"))
        upside = (target / price - 1) * 100 if price and price > 0 and target is not None else None
        count = f"{x['analyst_buy_count']}/{x['analyst_total']}" if x.get("analyst_buy_count") is not None and x.get("analyst_total") else "N/A"
        theme = theme_korean(x.get("industry"), x.get("sector"))
        lines.append(f"| {symbol} | {theme} | {x.get('analyst_consensus') or 'N/A'} | {count} | {fmt_fund(x.get('analyst_buy_pct'), 1, '%')} | {fmt_fund(target, 2)} | {fmt_fund(upside, 1, '%')} | {fmt_fund(x.get('pe'))} |")
    lines.append("")
    lines.append("### 연간 재무실적 및 예상치 (십억 USD)")
    lines.append("| 티커 | 실적기준일 | 매출 | 영업이익 | 순이익 | 예상기준일 | 예상매출 | 예상영업이익 | 예상순이익 |")
    lines.append("|---|---|---:|---:|---:|---|---:|---:|---:|")
    for symbol, x in sorted(data.items()):
        lines.append(f"| {symbol} | {x.get('statement_period') or 'N/A'} | {fmt_billions(x.get('revenue_usd'))} | {fmt_billions(x.get('operating_income_usd'))} | {fmt_billions(x.get('net_income_usd'))} | {x.get('estimate_period') or 'N/A'} | {fmt_billions(x.get('estimated_revenue_usd'))} | {fmt_billions(x.get('estimated_operating_income_usd'))} | {fmt_billions(x.get('estimated_net_income_usd'))} |")
    lines.append("")
    lines.append("- 출처: Financial Modeling Prep. 데이터 제공 범위/요금제에 따라 N/A 표시. 목표가 상승여력은 보고서 종가 기준이며 실시간 수익률이 아닙니다.")

def fmt_margin(revenue, profit):
    revenue = safe_float(revenue)
    profit = safe_float(profit)
    return f" ({profit / revenue * 100:+.1f}%)" if revenue is not None and revenue != 0 and profit is not None else ""


def fmt_financial_triplet(data, keys):
    revenue, operating, net = (data.get(k) for k in keys)
    return (f"{fmt_billions(revenue)} / "
            f"{fmt_billions(operating)}{fmt_margin(revenue, operating)} / "
            f"{fmt_billions(net)}{fmt_margin(revenue, net)}")


def candidate_fund_line(result, item):
    x = result.get("fundamentals", {}).get(item["symbol"]) or {}
    theme = theme_korean(x.get("industry") or item.get("industry"), x.get("sector") or item.get("sector"))
    opinion = str(x.get("analyst_consensus") or "").replace("|", "/")
    buy = (f"{x['analyst_buy_count']}/{x['analyst_total']}"
           if x.get("analyst_buy_count") is not None and x.get("analyst_total") else "")
    buy_pct = fmt_fund(x.get("analyst_buy_pct"), 1, "%")
    opinion_info = " · ".join(filter(None, (opinion, f"매수 {buy}" if buy else "",
                                               f"매수비율 {buy_pct}" if buy_pct else "")))
    target = safe_float(x.get("price_target_usd"))
    price = safe_float(item.get("close", item.get("price")))
    upside = (100 * (target / price - 1) if target is not None and price and price > 0 else None)
    target_info = " · ".join(filter(None, (
        f"목표 USD {fmt_fund(target, 2)}" if target is not None else "",
        f"여력 {fmt_fund(upside, 1, '%')}" if upside is not None else "")))
    pe = fmt_fund(x.get("pe"), 1)
    actual = fmt_financial_triplet(x, ("revenue_usd", "operating_income_usd", "net_income_usd"))
    forecast = fmt_financial_triplet(x, ("estimated_revenue_usd", "estimated_operating_income_usd", "estimated_net_income_usd"))
    return (f"| 정보 | {theme} | {opinion_info} | {target_info} | "
            f"{pe + '배' if pe else ''} | {actual} | {forecast} | "
            f"{x.get('statement_period') or ''} |")


def build_markdown(result):
    lines = ["# S&P 500 전수검사", ""]
    header = ("<table><thead><tr>"
              "<th>종목</th><th>회사명<br>테마</th><th>점수<br>투자의견</th>"
              "<th>가격<br>목표가</th><th>기술지표<br>PER</th>"
              "<th>거래량<br>실적(십억 USD)</th><th>추세<br>예상(십억 USD)</th>"
              "<th>기간<br>실적기준일</th></tr></thead><tbody>")

    def begin_section(title):
        lines.extend(["", "## " + title, "", header])

    def add_stock(item, technical):
        # GitHub Markdown tables cannot merge cells; HTML rowspan merges ticker vertically.
        info = candidate_fund_line(result, item).strip("|").split(" | ")
        if len(info) != 8:
            raise ValueError(f"Unexpected fundamentals columns for {item['symbol']}: {len(info)}")
        def td(value, rowspan=False):
            attr = ' rowspan="2"' if rowspan else ""
            return f"<td{attr}>{escape(str(value))}</td>"
        lines.append("<tr>" + td(technical[0], rowspan=True) +
                     "".join(td(value) for value in technical[1:]) + "</tr>")
        lines.append("<tr>" + "".join(td(value) for value in info[1:]) + "</tr>")

    begin_section("신규 상단 돌파 신호")
    if result["new_breakout_signals"]:
        for x in result["new_breakout_signals"]:
            add_stock(x, [
                x["symbol"], x.get("name", ""), "상단돌파" + (" · 강함" if x.get("strong") else ""),
                fmt_fund(x.get("close", x.get("price")), 2), "상단 " + fmt_fund(x.get("box_high"), 2),
                "거래량 " + fmt_fund(x.get("volume_ratio"), 2, "x"), "-", str(x["period"]) + "일"])
    else:
        lines.append('<tr><td colspan="8">없음</td></tr>')

    lines.append("</tbody></table>")
    begin_section("박스권 하단 후보")
    if result["lower_box_candidates"]:
        for x in result["lower_box_candidates"]:
            add_stock(x, [
                x["symbol"], x.get("name", ""), "하단 감시", fmt_fund(x.get("close"), 2),
                "위치 " + fmt_fund(100*x["box_position"], 1, "%"),
                "상승/하락 " + fmt_fund(x.get("up_down_volume_ratio"), 2, "x"),
                "최대일 제외 " + fmt_fund(x.get("up_down_volume_ratio_ex_max_day"), 2, "x"),
                str(x["period"]) + "일"])
    else:
        lines.append("| 없음 | - | - | - | - | - | - | - |")

    lines.append("</tbody></table>")
    begin_section("박스권 상단 감시 후보 (85점 이상)")
    if result["upper_box_watch_candidates"]:
        for x in result["upper_box_watch_candidates"]:
            add_stock(x, [
                x["symbol"], x.get("name", ""), fmt_fund(x.get("watch_score"), 1) + "점 · " + x["tier"] + "등급",
                fmt_fund(x.get("close"), 2),
                "위치 " + fmt_fund(100*x["box_position"], 1, "%") + " · 상단 " + fmt_fund(x.get("box_high"), 2),
                "거래량 " + fmt_fund(x.get("volume_growth_ratio"), 2, "x"),
                "5일 " + fmt_fund(x.get("recent_5_return_pct"), 2, "%"),
                str(x["period"]) + "일"])
    else:
        lines.append("| 없음 | - | - | - | - | - | - | - |")

    lines.append("</tbody></table>")
    lines.extend(["", "## 우상향 종목 (완료된 일봉 기준)"])
    for period in UPTREND_PERIODS:
        candidates = result["uptrend_candidates"][f"{period}d"]
        lines.extend(["", f"### 최근 {period}거래일: {len(candidates)}종목", "", header])
        if not candidates:
            lines.append("| 없음 | - | - | - | - | - | - | - |")
        for x in candidates:
            add_stock(x, [
                x["symbol"], x.get("name", ""), fmt_fund(x.get("quality_score"), 1) + "점",
                "수익률 " + fmt_fund(x.get("return_pct"), 2, "%"),
                "R² " + fmt_fund(x.get("r2"), 3),
                "거래량 " + fmt_fund(x.get("volume_growth_ratio"), 2, "x"),
                "기울기 " + fmt_fund(x.get("slope_pct_per_day"), 3, "%"),
                str(period) + "일"])
        lines.append("</tbody></table>")

    lines.extend(["", "※ 각 종목 첫 줄은 기술적 지표, 둘째 줄은 테마·애널리스트·재무정보입니다. "
                  "실적/예상 값은 매출·영업이익·순이익 순서이며 단위는 십억 USD입니다. "
                  "목표가 상승여력은 해당 후보 종가 기준입니다. 데이터가 없으면 공백으로 표시합니다."])
    if result["failures"]:
        lines.extend(["", "## 실패/누락"])
        for item in result["failures"][:100]:
            lines.append(f"- {item['symbol']}: {item['reason']}")
        if len(result["failures"]) > 100:
            lines.append(f"- ... 추가 {len(result['failures']) - 100}건은 result.json 참고")

    lines.extend(["", "## 실행 결과 요약",
        f"- 실행 시작(KST): {result['run']['started_kst']}",
        f"- 실행 종료(KST): {result['run']['finished_kst']}",
        f"- 모드: {result['run']['mode']}",
        f"- 전체 대상: {result['coverage']['target_count']}",
        f"- 고유 요청: {result['coverage']['unique_requested_count']}"])
    period_summary = ' / '.join(str(p) for p in result['criteria']['box_periods'])
    breakout_complete = ' / '.join(str(result['coverage']['breakout_calc_complete'][str(p)]) for p in result['criteria']['box_periods'])
    lower_complete = ' / '.join(str(result['coverage']['lower_box_calc_complete'][str(p)]) for p in result['criteria']['box_periods'])
    lines.extend([
        f"- 상단돌파 계산 완료({period_summary}): {breakout_complete}",
        f"- 하단후보 계산 완료({period_summary}): {lower_complete}",
        f"- 후보 5분봉 검증: {result['coverage']['candidate_volume_validations']}",
        f"- 자료부족: {result['coverage']['data_shortage_count']}",
        f"- 미요청: {result['coverage']['unrequested_count']}", ""])
    return "\n".join(lines) + "\n"


# 보고서 회사명 한글 표기 (미등록 종목은 원문 유지)
KOREAN_COMPANY_NAMES = {
    "SPG": "사이먼 프로퍼티 그룹",
    "TKO": "TKO 그룹 홀딩스",
    "DTE": "DTE 에너지",
    "JNJ": "존슨앤드존슨",
    "TECH": "바이오테크네",
    "HSIC": "헨리 샤인",
    "AEE": "아메렌",
    "EVRG": "에버지",
    "VMRK": "비브마크 레지덴셜",
    "DE": "디어",
    "TDG": "트랜스다임 그룹",
    "HUM": "휴매나",
    "AFL": "애플랙",
    "SOLV": "솔벤텀",
    "UBER": "우버",
    "EW": "에드워즈 라이프사이언시스",
    "CPAY": "코페이",
    "TPR": "태피스트리",
    "MSI": "모토로라 솔루션스",
    "AIZ": "어슈어런트",
    "COST": "코스트코",
    "MCK": "맥케슨",
    "HD": "홈디포",
    "DXCM": "덱스컴",
    "NVDA": "엔비디아",
    "TSLA": "테슬라",
    "PFE": "화이자",
    "VST": "비스트라",
    "BEN": "프랭클린 리소시스",
    "CCI": "크라운 캐슬",
    "NDAQ": "나스닥",
    "NFLX": "넷플릭스",
    "UPS": "UPS",
    "FE": "퍼스트에너지",
    "BIIB": "바이오젠",
    "PPG": "PPG 인더스트리스",
    "GE": "GE 에어로스페이스",
    "GD": "제너럴 다이내믹스",
    "PSA": "퍼블릭 스토리지",
    "MGM": "MGM 리조트",
    "UDR": "UDR",
    "LNT": "얼라이언트 에너지",
    "NEE": "넥스트에라 에너지",
    "AMT": "아메리칸 타워",
    "AAPL": "애플",
    "MSFT": "마이크로소프트",
    "GOOGL": "알파벳",
    "GOOG": "알파벳",
    "AMZN": "아마존",
    "META": "메타 플랫폼스",
    "AVGO": "브로드컴",
    "BRK_B": "버크셔 해서웨이",
    "JPM": "JP모건 체이스",
    "V": "비자",
    "MA": "마스터카드",
    "WMT": "월마트",
    "XOM": "엑슨모빌",
    "ORCL": "오라클",
    "LLY": "일라이 릴리",
    "ABBV": "애브비",
    "KO": "코카콜라",
    "PEP": "펩시코",
    "BAC": "뱅크오브아메리카",
    "CRM": "세일즈포스",
    "AMD": "AMD",
    "INTC": "인텔",
    "IBM": "IBM",
    "QCOM": "퀄컴",
    "CSCO": "시스코 시스템즈",
    "DIS": "월트 디즈니",
    "NKE": "나이키",
    "BA": "보잉",
    "CAT": "캐터필러",
    "UNH": "유나이티드헬스 그룹",
    "T": "AT&T",
    "VZ": "버라이즌",
    "CVX": "셰브론",
    "MCD": "맥도날드",
    "SBUX": "스타벅스",
    "ADBE": "어도비",
    "NOW": "서비스나우",
    "PLTR": "팔란티어",
    "PYPL": "페이팔",
    "ABT": "애보트 래버러토리스",
    "TMO": "써모 피셔 사이언티픽",
    "DHR": "다나허",
    "ISRG": "인튜이티브 서지컬",
    "GS": "골드만삭스",
    "MS": "모건스탠리",
    "BLK": "블랙록",
    "C": "씨티그룹",
    "SCHW": "찰스 슈왑",
    "AXP": "아메리칸 익스프레스",
    "BKNG": "부킹 홀딩스",
    "CMCSA": "컴캐스트",
    "PGR": "프로그레시브",
    "CB": "처브",
    "SO": "서던 컴퍼니",
    "DUK": "듀크 에너지",
    "EXC": "엑셀론",
    "KIM": "킴코 리얼티",
    "VRSN": "베리사인",
    "CRL": "찰스 리버 래버러토리스",
    "POOL": "풀 코퍼레이션",
    "EBAY": "이베이",
    "J": "제이콥스 솔루션스",
    "FDX": "페덱스",
    "GM": "제너럴 모터스",
    "F": "포드 모터",
    "LOW": "로우스",
    "CME": "CME 그룹",
    "ICE": "인터컨티넨털 익스체인지",
    "LMT": "록히드 마틴",
    "RTX": "RTX",
    "NOC": "노스럽 그러먼",
    "HON": "허니웰",
    "MMM": "3M",
    "WFC": "웰스파고",
}

def main():
    started = now_utc()
    started_kst = started.astimezone(KST)
    now_ny = started.astimezone(NY)
    is_open = regular_market_open(now_ny)
    mode = "intraday_delayed_sip" if is_open else "after_close_daily"

    previous = load_previous_state()
    seen_signal_keys = set(previous.get("seen_signal_keys", []))

    constituents, constituent_source, constituent_warning = load_sp500_constituents()
    symbol_to_name = {x["symbol"]: KOREAN_COMPANY_NAMES.get(x["symbol"], x["name"]) for x in constituents}
    symbol_to_theme = {x["symbol"]: {"sector": x.get("sector"), "industry": x.get("industry")} for x in constituents}
    symbols = list(symbol_to_name)

    stats = {
        "requested_symbols": set(),
        "daily_http_requests": 0,
        "snapshot_http_requests": 0,
        "intraday_http_requests": 0,
    }
    failures = {}

    # Free historical SIP is available with delay, so keep end safely >15m behind now.
    historical_end = started - timedelta(minutes=16)
    historical_start = historical_end - timedelta(days=DAILY_LOOKBACK_DAYS)
    daily_bars = fetch_daily_bars(symbols, historical_start, historical_end, stats, failures)

    snapshots = {}
    if is_open:
        snapshots = fetch_snapshots(symbols, stats, failures)

    breakout_calc_complete = {str(p): 0 for p in BOX_PERIODS}
    lower_box_calc_complete = {str(p): 0 for p in BOX_PERIODS}
    data_shortage_symbols = []
    all_confirmed = []
    lower_box_candidates = []
    upper_box_watch_by_symbol = {}
    upper_box_watch_calc_complete = {str(p): 0 for p in BOX_PERIODS}
    price_candidates = []
    uptrend_candidates = {f"{p}d": [] for p in UPTREND_PERIODS}
    uptrend_calc_complete = {str(p): 0 for p in UPTREND_PERIODS}

    for symbol in symbols:
        raw = daily_bars.get(symbol, [])
        completed = completed_daily_bars(raw, now_ny)

        for p in UPTREND_PERIODS:
            if len(completed) >= p:
                uptrend_calc_complete[str(p)] += 1
                trend = uptrend_for_period(completed, p)
                if trend:
                    quality = uptrend_quality(completed, p)
                    if quality and quality["quality_filters_pass"]:
                        trend.update(quality)
                        trend["quality_score"] = uptrend_quality_score(trend, quality)
                        if trend["quality_score"] >= UPTREND_SCORE_MIN:
                            trend.update({"symbol": symbol, "name": symbol_to_name[symbol]})
                            uptrend_candidates[f"{p}d"].append(trend)

        for p in BOX_PERIODS:
            if len(completed) >= p:
                upper_box_watch_calc_complete[str(p)] += 1
                watch = upper_box_watch_for_period(completed, p)
                if watch:
                    watch.update({"symbol": symbol, "name": symbol_to_name[symbol]})
                    old = upper_box_watch_by_symbol.get(symbol)
                    if old is None or (watch["watch_score"], watch["box_position"], watch["period"]) > (old["watch_score"], old["box_position"], old["period"]):
                        upper_box_watch_by_symbol[symbol] = watch
                lower_box_calc_complete[str(p)] += 1
                low = lower_box_for_period(completed, p)
                if low and low["qualifies"]:
                    low.update({"symbol": symbol, "name": symbol_to_name[symbol]})
                    lower_box_candidates.append(low)

            # Need p historical sessions plus one breakout day for after-close scan.
            need = p + (0 if is_open else 1)
            if len(completed) >= need:
                breakout_calc_complete[str(p)] += 1

        min_required = max(BOX_PERIODS) + (0 if is_open else 1)
        if len(completed) < min_required:
            data_shortage_symbols.append(symbol)
            failures.setdefault(symbol, []).append(
                f"daily bars insufficient: {len(completed)} (need {min_required} for {max(BOX_PERIODS)}-day scan)"
            )

        if is_open:
            snap = snapshots.get(symbol)
            for p in BOX_PERIODS:
                cand = intraday_candidate(completed, snap, p)
                if cand:
                    cand.update({"symbol": symbol, "name": symbol_to_name[symbol], "mode": "intraday"})
                    price_candidates.append(cand)
        else:
            for p in BOX_PERIODS:
                sig = daily_breakout_for_period(completed, p)
                if sig and sig["confirmed"]:
                    sig.update({"symbol": symbol, "name": symbol_to_name[symbol], "mode": "after_close"})
                    all_confirmed.append(sig)

    upper_box_watch_candidates = sorted(
        (x for x in upper_box_watch_by_symbol.values() if x["watch_score"] >= UPPER_BOX_WATCH_MIN_SCORE),
        key=lambda x: (-x["watch_score"], -x["box_position"], x["symbol"])
    )
    upper_box_watch_tier_counts = {tier: sum(x["tier"] == tier for x in upper_box_watch_candidates) for tier in ("A", "B", "C")}

    for period_key in uptrend_candidates:
        uptrend_candidates[period_key].sort(
            key=lambda item: (-item["quality_score"], -item["r2"], item["symbol"])
        )

    candidate_volume_validations = 0
    if is_open and price_candidates:
        # A symbol can qualify for more than one box period; fetch 5m only once per symbol.
        by_symbol = defaultdict(list)
        for c in price_candidates:
            by_symbol[c["symbol"]].append(c)

        intraday_end = historical_end
        intraday_start = intraday_end - timedelta(days=45)
        for symbol, candidates in by_symbol.items():
            bars5, err = fetch_5m_bars(symbol, intraday_start, intraday_end, stats)
            candidate_volume_validations += 1
            if err:
                failures.setdefault(symbol, []).append(err)
                continue
            volcheck, verr = validate_intraday_volume(symbol, bars5, now_ny)
            if verr:
                failures.setdefault(symbol, []).append(f"5m validation: {verr}")
                continue
            for c in candidates:
                merged = {**c, **volcheck}
                if merged.get("confirmed"):
                    all_confirmed.append(merged)

    # Deduplicate signals by symbol/period/box/date-or-time.
    unique_confirmed = []
    current_keys = set()
    for s in all_confirmed:
        key = signal_key(s)
        if key in current_keys:
            continue
        current_keys.add(key)
        s["signal_key"] = key
        unique_confirmed.append(s)

    new_signals = [s for s in unique_confirmed if s["signal_key"] not in seen_signal_keys]
    seen_signal_keys.update(s["signal_key"] for s in unique_confirmed)


    target_count = len(symbols)
    unique_requested_count = len(stats["requested_symbols"])
    unrequested = sorted(set(symbols) - stats["requested_symbols"])

    failure_list = []
    for symbol in sorted(failures):
        # Collapse duplicate reasons.
        reasons = []
        for reason in failures[symbol]:
            if reason not in reasons:
                reasons.append(reason)
        failure_list.append({"symbol": symbol, "reason": " | ".join(reasons)})

    finished = now_utc()
    result = {
        "run": {
            "started_utc": iso(started),
            "finished_utc": iso(finished),
            "started_kst": iso(started_kst),
            "finished_kst": iso(finished.astimezone(KST)),
            "started_et": iso(started.astimezone(NY)),
            "mode": mode,
            "market_open_estimate": is_open,
            "historical_data_end_utc": iso(historical_end),
        },
        "sources": {
            "constituents": constituent_source,
            "constituent_warning": constituent_warning,
            "market_data": "Alpaca Market Data",
            "historical_feed": "sip",
            "snapshot_feed": "delayed_sip" if is_open else None,
            "adjustment": "split",
        },
        "criteria": {
            "box_periods": list(BOX_PERIODS),
            "uptrend_periods": list(UPTREND_PERIODS),
            "uptrend_min_r2": UPTREND_MIN_R2,
            "uptrend_min_slope_pct_per_day": UPTREND_MIN_SLOPE_PCT_PER_DAY,
            "uptrend_min_return_pct": {str(k): v for k, v in UPTREND_MIN_RETURN_PCT.items()},
            "uptrend_quality_filters": {
                "volume_growth_min": UPTREND_VOLUME_GROWTH_MIN,
                "up_down_volume_ratio_min": UPTREND_UP_DOWN_VOLUME_MIN,
                "ma_order": "MA5 > MA10 > MA20",
                "max_drawdown_from_20d_close_high": UPTREND_MAX_DRAWDOWN_FROM_20D_CLOSE_HIGH,
                "max_single_day_gain": UPTREND_MAX_SINGLE_DAY_GAIN,
                "min_quality_score": UPTREND_SCORE_MIN,
                "min_completed_daily_bars": 25,
            },
            "upper_box_watch_min_position": UPPER_BOX_WATCH_MIN_POSITION,
            "upper_box_watch_max_position": UPPER_BOX_WATCH_MAX_POSITION,
            "upper_box_watch_min_score": UPPER_BOX_WATCH_MIN_SCORE,
            "upper_box_watch_tiers": {"A": [0.95, 1.00], "B": [0.85, 0.95], "C": [0.70, 0.85]},
            "upper_box_watch_deduplicate_by_symbol": True,
            "box_max_width": BOX_MAX_WIDTH,
            "first_to_last_close_max_abs_move": BOX_FIRST_LAST_MAX_MOVE,
            "volume_ratio_min": DAILY_VOLUME_RATIO_MIN,
            "strong_volume_ratio": STRONG_VOLUME_RATIO,
            "breakout_day_excluded_from_box": True,
            "lower_box_max_position": LOWER_BOX_MAX_POSITION,
            "up_down_volume_ratio_min": UP_DOWN_VOLUME_RATIO_MIN,
            "up_down_volume_ratio_ex_max_day_min": UP_DOWN_VOLUME_RATIO_EX_MAX_DAY_MIN,
        },
        "coverage": {
            "target_count": target_count,
            "unique_requested_count": unique_requested_count,
            "breakout_calc_complete": breakout_calc_complete,
            "lower_box_calc_complete": lower_box_calc_complete,
            "upper_box_watch_calc_complete": upper_box_watch_calc_complete,
            "upper_box_watch_candidate_count": len(upper_box_watch_candidates),
            "upper_box_watch_tier_counts": upper_box_watch_tier_counts,
            "uptrend_calc_complete": uptrend_calc_complete,
            "uptrend_candidate_counts": {key: len(value) for key, value in uptrend_candidates.items()},
            "candidate_price_breakouts": len(price_candidates),
            "candidate_volume_validations": candidate_volume_validations,
            "data_shortage_count": len(set(data_shortage_symbols)),
            "data_shortage_symbols": sorted(set(data_shortage_symbols)),
            "unrequested_count": len(unrequested),
            "unrequested_symbols": unrequested,
            "http_requests": {
                "daily": stats["daily_http_requests"],
                "snapshots": stats["snapshot_http_requests"],
                "intraday_5m": stats["intraday_http_requests"],
            },
        },
        "confirmed_signals": unique_confirmed,
        "new_breakout_signals": new_signals,
        "upper_box_watch_candidates": upper_box_watch_candidates,
        "lower_box_candidates": sorted(lower_box_candidates, key=lambda x: (x["box_position"], -x["up_down_volume_ratio"])),
        "uptrend_candidates": uptrend_candidates,
        "failures": failure_list,
        "seen_signal_keys": sorted(seen_signal_keys)[-5000:],
    }

    fmp_enrich(result)
    for group in (result["upper_box_watch_candidates"], result["lower_box_candidates"], result["new_breakout_signals"], *result["uptrend_candidates"].values()):
        for item in group:
            item.update({k: v for k, v in symbol_to_theme.get(item["symbol"], {}).items() if k not in item})
    RESULT_FILE.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_FILE.write_text(build_markdown(result), encoding="utf-8")

    print(json.dumps({
        "status": "ok",
        "mode": mode,
        "target": target_count,
        "requested": unique_requested_count,
        "breakout_calc_complete": breakout_calc_complete,
        "lower_box_calc_complete": lower_box_calc_complete,
        "lower_box_candidates": len(lower_box_candidates),
        "upper_box_watch_candidates": len(upper_box_watch_candidates),
        "upper_box_watch_tier_counts": upper_box_watch_tier_counts,
        "uptrend_candidates": {key: len(value) for key, value in uptrend_candidates.items()},
        "price_candidates": len(price_candidates),
        "volume_validations": candidate_volume_validations,
        "new_signals": len(new_signals),
        "failures": len(failure_list),
        "result_file": str(RESULT_FILE),
        "report_file": str(REPORT_FILE),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
