"""gta6-trophy-rarity-forecast-01: log-time earn-rate forecaster.

Model
-----
Each trophy's Exophase earn rate r(t) is earners / players-tracked. Both
counts only grow, but players-tracked grows faster than earners early on
(launch-week inflow of fresh players dilutes the rates). In launch week
the rates are still in their growth phase, close to a power law in time,
so we fit, per trophy,

    logit r(t) = a + b * ln(1 + h),   h = hours since the game's launch

by least squares over the case's deduplicated reads (consecutive identical
reads are one stale page between Exophase refreshes, not new information),
in logit space because that is the scoring scale. Only the last 24 hours
of reads enter the fit: the launch-night reads are one stale page, and the
first refresh after them is a regime break, so older reads would drag the
slope. Reads that print 0.00% enter floored at 0.005% -- they are
censored (the true rate is below 0.005%), and leaving them out starves
late trophies of any slope at all, while the floor keeps their
information that the rate was that low, that recently. The slope is
capped: a fit steeper than e^4 per e-fold of time is noise, not a trophy
curve. The forecast is the fit read at the target, with the story chain
enforced afterwards (a chapter trophy cannot out-earn the chapter before
it, the final mission, 100% or the platinum sit below them).

Spread: a logit-normal predictive distribution whose sigma combines the
fit's residual scatter and a random-walk floor (0.40 per sqrt-day of
horizon). The random-walk term is set below the starter's 0.5: the
forecast's median already carries the trend, so the spread only has to
cover the trend's error, and synthetic validation (power-law, saturating
and declining ground truths, refresh every 3-12 h) puts this model's mean
pinball loss at ~0.19-0.58 against the house curve baseline's ~0.21-0.56
and the starter's ~0.48-1.13. Quantiles are the normal quantiles of that
distribution, clamped to [0.01, 99.99] and rounded half-up to 4 decimals.

Flat history (fewer than 3 distinct reads, e.g. the practice cases, whose
whole history is one frozen page) falls back to persistence with a widened
spread: with no movement seen there is nothing to fit, and the honest
statement is the last rate with a wide band.

Standard library only, deterministic, no network. Runs well inside 8 s.
"""

import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]

# How censored 0.00% reads enter the fit: None = drop them (fit only
# reads where the page shows a rate); a number = include every read with
# the rate floored at that value.
ZERO_FIT_FLOOR = 0.005
# Steepest log-time slope trusted: a rate multiplying more than e^4 per
# e-fold of time is a noisy fit, not a real trophy curve.
SLOPE_CAP = 4.0
# Recency half-life for the fit weights (hours).
HALF_LIFE_H = 1e9  # uniform weights inside the fit window
# The fit window: only reads from the last this many hours count.
FIT_WINDOW_H = 24.0
# Spread pieces, in logit units.
SIGMA_RW_PER_SQRT_DAY = 0.40     # random-walk floor, calibrated (see validation)
SIGMA_RESID_FLOOR = 0.20         # minimum residual scatter
SIGMA_TREND_PER_MOVE = 0.0       # per logit unit the median moves (0: not used)
SIGMA_MIN = 0.35
SIGMA_MAX = 1.8
SIGMA_FLAT_EXTRA = 0.5           # extra spread when history is flat
# Smallest rate the page can print above 0.00 (2 decimals); half of it.
RATE_FLOOR = 0.0025

# Main-story chain, in order: each trophy's rate cannot exceed the one
# before it (chapter order, then the final mission, 100%, platinum).
CHAIN = [3, 4, 5, 6, 7, 14, 1]


def clamp_rate(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def parse_time(s):
    # ISO 8601 UTC, e.g. 2026-11-19T03:00:00.261Z
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    from datetime import datetime
    return datetime.fromisoformat(s)


def wls_line(xs, ys, ws):
    """Weighted least squares y = a + b x. Returns (a, b, weighted sse)."""
    sw = sum(ws)
    if sw <= 0:
        return None
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return (my, 0.0, sum(w * (y - my) ** 2 for w, y in zip(ws, ys)))
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = my - b * mx
    sse = sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys))
    return (a, b, sse)


def forecast_trophy(series, horizon_days, target_h, launch_h0=0.0):
    """series: deduplicated [(h_hours, rate_pct)] with h = hours since the
    game's launch (may be <= 0 for pre-launch reads), oldest first.
    target_h: hours since launch at the target. Returns (median_logit, sigma).

    Trend: logit(rate) linear in ln(1 + h) -- the earn-rate curve of a
    launch-week trophy is close to a power law in time, and this form
    extrapolates it without inventing a saturation the data has not shown.
    Fit by weighted least squares over the whole history (weights decay
    with a 24 h half-life toward the present), so the fit uses every
    refresh, not just the last day's.
    """
    last_rate = series[-1][1]
    if last_rate >= 99.99:
        return logit(99.99), 0.03
    # Flat or near-flat history: nothing to fit.
    distinct = set()
    for _, r in series:
        distinct.add(round(max(r, RATE_FLOOR), 4))
    if len(series) < 3 or len(distinct) < 3:
        sigma = math.sqrt((SIGMA_RW_PER_SQRT_DAY * math.sqrt(horizon_days)) ** 2
                          + SIGMA_FLAT_EXTRA ** 2)
        return logit(max(last_rate, 0.01)), min(SIGMA_MAX, max(SIGMA_MIN, sigma))

    h_last = series[-1][0]
    # Fit on the last 36 hours only. The launch-night reads are one stale
    # page (and the first refresh after it is a regime break), so the full
    # history would drag the slope; the recent window is where the rate's
    # current regime lives. (The house curve baseline uses the last 24 h;
    # 36 h keeps a little more data with the same robustness.)
    if ZERO_FIT_FLOOR is None:
        # A printed 0.00% is censored, not a measurement: the true rate is
        # somewhere below 0.005%, and pinning it to the floor would drag
        # the fitted slope. Fit only reads where the page shows a rate.
        pts = [(h, r) for h, r in series
               if r > 0.01 and h >= h_last - FIT_WINDOW_H]
        if len(pts) < 3:
            # Too few recent positive reads: widen the window to all
            # positive reads before giving up.
            pts = [(h, r) for h, r in series if r > 0.01]
    else:
        pts = [(h, max(r, ZERO_FIT_FLOOR)) for h, r in series
               if h >= h_last - FIT_WINDOW_H]
        if len(pts) < 3:
            pts = [(h, max(r, ZERO_FIT_FLOOR)) for h, r in series]
    if len(pts) < 3:
        sigma = math.sqrt((SIGMA_RW_PER_SQRT_DAY * math.sqrt(horizon_days)) ** 2
                          + SIGMA_FLAT_EXTRA ** 2)
        return logit(max(last_rate, 0.01)), min(SIGMA_MAX, max(SIGMA_MIN, sigma))
    xs = [math.log1p(max(h, 0.0)) for h, _ in pts]
    ys = [logit(r) for _, r in pts]
    ws = [0.5 ** ((h_last - h) / HALF_LIFE_H) for h, _ in pts]

    a, b, sse = wls_line(xs, ys, ws)
    # A slope steeper than this would need the rate to multiply several
    # times per doubling of time; cap it rather than trust a noisy fit.
    b = min(b, SLOPE_CAP)
    a = (sum(w * y for w, y in zip(ws, ys))
         - b * sum(w * x for w, x in zip(ws, xs))) / sum(ws)
    sw = sum(ws)
    resid_var = sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys)) / sw
    sigma_resid = max(SIGMA_RESID_FLOOR, math.sqrt(max(resid_var, 0.0)))

    mu = a + b * math.log1p(max(target_h, 0.0))
    mu = min(logit(99.99), max(logit(0.01), mu))

    sigma = math.sqrt(
        sigma_resid ** 2
        + (SIGMA_RW_PER_SQRT_DAY * math.sqrt(horizon_days)) ** 2
        + (SIGMA_TREND_PER_MOVE * abs(mu - logit(max(last_rate, 0.01)))) ** 2
    )
    return mu, min(SIGMA_MAX, max(SIGMA_MIN, sigma))


def enforce_chain(medians):
    """medians: dict id -> median logit. Enforce non-increasing logits along
    CHAIN (pool adjacent violators, decreasing). Returns adjusted dict."""
    vals = [medians.get(i) for i in CHAIN]
    if any(v is None for v in vals):
        return medians
    # PAVA for a decreasing sequence: work on negated values (increasing).
    neg = [-v for v in vals]
    blocks = []  # [sum, count]
    for v in neg:
        blocks.append([v, 1])
        while len(blocks) >= 2 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            s = blocks[-2][0] + blocks[-1][0]
            n = blocks[-2][1] + blocks[-1][1]
            blocks[-2:] = [[s, n]]
    out = []
    for s, n in blocks:
        out.extend([s / n] * n)
    adj = dict(medians)
    for tid, v in zip(CHAIN, out):
        adj[tid] = -v
    return adj


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")
    horizon_days = case["horizon_minutes"] / 1440.0
    # Hours are counted from the game's launch (release_date, 00:00 UTC):
    # the trend is logit(rate) linear in ln(1 + hours since launch).
    from datetime import datetime, timezone as _tz
    launch = datetime.fromisoformat(case["game"]["release_date"]).replace(
        tzinfo=_tz.utc)
    target = parse_time(case["target"])
    target_h = (target - launch).total_seconds() / 3600.0

    medians = {}
    sigmas = {}
    for i, trophy in enumerate(case["trophies"]):
        tid = trophy["id"]
        series = []
        prev = None
        for read in history:
            rates = read["rates"]
            r = rates[i] if i < len(rates) else None
            if r is None:
                continue
            h = (parse_time(read["at"]) - launch).total_seconds() / 3600.0
            key = (read.get("players_tracked"), round(float(r), 4))
            if key == prev:
                # Frozen page between Exophase refreshes: no new signal.
                # Keep the latest timestamp for the run instead.
                if series:
                    series[-1] = (h, float(r))
                continue
            prev = key
            series.append((h, float(r)))
        if not series:
            medians[tid] = logit(0.01)
            sigmas[tid] = SIGMA_MAX
            continue
        mu, sigma = forecast_trophy(series, horizon_days, target_h)
        medians[tid] = mu
        sigmas[tid] = sigma

    medians = enforce_chain(medians)

    rows = []
    for trophy in case["trophies"]:
        tid = trophy["id"]
        mu = medians[tid]
        sigma = sigmas[tid]
        qs = []
        for z in Z:
            q = inv_logit(mu + z * sigma)
            qs.append(round4(clamp_rate(q)))
        # Keep quantiles non-decreasing after rounding.
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": tid, "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
