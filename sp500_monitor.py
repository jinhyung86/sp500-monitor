import os
import json
import math
import time
from io import StringIO
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
            rows.append({"symbol": symbol, "name": name})
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


def build_markdown(result):
    lines = []
    lines.append("# S&P 500 전수검사")
    lines.append("")
    lines.append(f"- 실행 시작(KST): {result['run']['started_kst']}")
    lines.append(f"- 실행 종료(KST): {result['run']['finished_kst']}")
    lines.append(f"- 모드: {result['run']['mode']}")
    lines.append(f"- 전체 대상: {result['coverage']['target_count']}")
    lines.append(f"- 고유 요청: {result['coverage']['unique_requested_count']}")
    period_summary = ' / '.join(str(p) for p in result['criteria']['box_periods'])
    breakout_complete = ' / '.join(str(result['coverage']['breakout_calc_complete'][str(p)]) for p in result['criteria']['box_periods'])
    lower_complete = ' / '.join(str(result['coverage']['lower_box_calc_complete'][str(p)]) for p in result['criteria']['box_periods'])
    lines.append(f"- 상단돌파 계산 완료({period_summary}): {breakout_complete}")
    lines.append(f"- 하단후보 계산 완료({period_summary}): {lower_complete}")
    lines.append(f"- 후보 5분봉 검증: {result['coverage']['candidate_volume_validations']}")
    lines.append(f"- 자료부족: {result['coverage']['data_shortage_count']}")
    lines.append(f"- 미요청: {result['coverage']['unrequested_count']}")
    lines.append("")

    lines.append("## 신규 상단 돌파 신호")
    if result["new_breakout_signals"]:
        lines.append("")
        lines.append("| 티커 | 기간 | 가격/종가 | 박스상단 | 거래량비율 | 강한신호 |")
        lines.append("|---|---:|---:|---:|---:|---|")
        for s in result["new_breakout_signals"]:
            px = s.get("close", s.get("price"))
            vr = s.get("volume_ratio")
            lines.append(f"| {s['symbol']} | {s['period']} | {px:.2f} | {s['box_high']:.2f} | {vr:.2f}x | {'예' if s.get('strong') else '아니오'} |")
    else:
        lines.append("- 없음")

    lines.append("")
    lines.append("## 박스권 하단 후보")
    if result["lower_box_candidates"]:
        lines.append("")
        lines.append("| 티커 | 기간 | 종가 | 박스하단 | 박스상단 | 위치 | 상승/하락 거래량 | 최대거래량일 제외 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for s in result["lower_box_candidates"]:
            lines.append(
                f"| {s['symbol']} | {s['period']} | {s['close']:.2f} | {s['box_low']:.2f} | {s['box_high']:.2f} | "
                f"{s['box_position']*100:.1f}% | {s['up_down_volume_ratio']:.2f}x | {s['up_down_volume_ratio_ex_max_day']:.2f}x |"
            )
    else:
        lines.append("- 없음")

    lines.append("")
    lines.append("## 박스권 상단 접근 후보 (위치 70~100%)")
    lines.append("- 박스 위치 = (종가 - 박스 하단) / (박스 상단 - 박스 하단) × 100")
    lines.append("- A: 95~100% / B: 85~95% / C: 70~85%")
    lines.append("- 기존 유효 박스 조건 유지. 상승·거래량은 가점만 부여하며 필수 조건이 아닙니다.")
    counts = result["coverage"]["upper_box_watch_tier_counts"]
    lines.append(f"- 중복 제거: {len(result['upper_box_watch_candidates'])}종목 (A {counts['A']} / B {counts['B']} / C {counts['C']})")
    lines.append(f"- 감시점수 {UPPER_BOX_WATCH_MIN_SCORE:.0f}점 이상만 표시하며, 점수 높은 순으로 정렬합니다.")
    items = result["upper_box_watch_candidates"]
    lines.append("")
    if not items:
        lines.append("- 없음")
    else:
        lines.append("| 티커 | 회사명 | 등급 | 기간 | 종가 | 박스상단 | 박스위치 | 5일 등락 | 거래량 변화 | 감시점수 |")
        lines.append("|---|---|:---:|---:|---:|---:|---:|---:|---:|---:|")
        for x in items:
            r = f"{x['recent_5_return_pct']:+.2f}%" if x['recent_5_return_pct'] is not None else "-"
            v = f"{x['volume_growth_ratio']:.2f}x" if x['volume_growth_ratio'] is not None else "-"
            lines.append(f"| {x['symbol']} | {x['name']} | {x['tier']} | {x['period']} | {x['close']:.2f} | {x['box_high']:.2f} | {x['box_position']*100:.1f}% | {r} | {v} | {x['watch_score']:.1f} |")
    lines.append("")

    lines.append("## 우상향 종목 (완료된 일봉 기준)")
    lines.append(f"- 공통 기준: R² ≥ {UPTREND_MIN_R2:.2f}, 일평균 추세 기울기 ≥ +{UPTREND_MIN_SLOPE_PCT_PER_DAY:.2f}%")
    lines.append("- 기간별 최소 수익률: " + " / ".join(f"{p}거래일 +{UPTREND_MIN_RETURN_PCT[p]:.0f}%" for p in UPTREND_PERIODS))
    lines.append("- 2차 필터: 5일 평균 거래량/직전 20일 ≥ 1.20배, 5일선 > 10일선 > 20일선, 상승/하락일 거래량 ≥ 1.20배")
    lines.append("- 20일 최고 종가 대비 -5% 이내, 기간 중 하루 +10% 초과 급등 제외, 품질 점수 70점 이상")
    lines.append("- 점수 구성: 추세 일관성 30 + 거래량 30 + 이동평균선 20 + 고점 유지 20 (투자 수익 예측 아님)")
    for period in UPTREND_PERIODS:
        candidates = result["uptrend_candidates"][f"{period}d"]
        lines.append("")
        lines.append(f"### 최근 {period}거래일: {len(candidates)}종목")
        if candidates:
            lines.append("| 티커 | 회사명 | 점수 | 기간 수익률 | R² | 일평균 추세기울기 | 거래량증가 | 상승/하락 거래량 | 고점대비 | 기준일 |")
            lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
            for item in candidates:
                lines.append(
                    f"| {item['symbol']} | {item['name']} | {item['quality_score']:.1f} | {item['return_pct']:+.2f}% | "
                    f"{item['r2']:.3f} | {item['slope_pct_per_day']:+.3f}% | "
                    f"{item['volume_growth_ratio']:.2f}x | {item['up_down_volume_ratio']:.2f}x | "
                    f"-{item['drawdown_from_20d_close_high_pct']:.2f}% | {item['as_of_date']} |"
                )
        else:
            lines.append("- 없음")
    lines.append("")

    if result["failures"]:
        lines.append("")
        lines.append("## 실패/누락")
        for item in result["failures"][:100]:
            lines.append(f"- {item['symbol']}: {item['reason']}")
        if len(result["failures"]) > 100:
            lines.append(f"- ... 추가 {len(result['failures']) - 100}건은 result.json 참고")

    return "\n".join(lines) + "\n"

def main():
    started = now_utc()
    started_kst = started.astimezone(KST)
    now_ny = started.astimezone(NY)
    is_open = regular_market_open(now_ny)
    mode = "intraday_delayed_sip" if is_open else "after_close_daily"

    previous = load_previous_state()
    seen_signal_keys = set(previous.get("seen_signal_keys", []))

    constituents, constituent_source, constituent_warning = load_sp500_constituents()
    symbol_to_name = {x["symbol"]: x["name"] for x in constituents}
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
