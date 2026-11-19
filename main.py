"""gta6-trophy-rarity-forecast-01 entry: ceiling-saturation + earner-trend forecaster.

The final case asks, from every read the house recorded up to 2026-11-21T11:30Z, for
each GTA VI trophy's Exophase earn rate at 2026-11-26T00:00Z as five quantiles, scored
by pinball loss on the logit scale. Persistence (the house baseline) loses badly over a
~4.5-day horizon because launch-week rates move a lot. Our median blends two views:

  A. Ceiling saturation, per trophy class. Every mandatory story trophy's rate climbs
     toward a high ceiling (most launch-week players finish early chapters); hidden
     collectibles and stunts toward a low one; the platinum stays rare. In logit space
     the approach to the ceiling is exponential with a ~2.5-day time constant:
         lg_target = lg(C) + (lg_now - lg(C)) * exp(-h / T)
  B. Earner-count trend, data-driven. earners = rate * players_tracked / 100 rebuilt
     from every read; a recency-weighted least-squares slope over the last 24 h,
     damped by exp(-h/3), projected onto the tracked-player count extrapolated with
     its own damped slope (both floored at their last values; neither ever falls).

Median logit = 0.6 * A + 0.4 * B, never more than 0.35 logits below the current rate
(a refresh can cut a rate, but a week out the median should not sit below today's).
Never-seen trophies hug 0.01 with a wide band.

Spread, on the logit scale: the disagreement between the two views, the fit's residual
scale carried through the logit derivative, and a random-walk floor that widens with
sqrt(horizon); clipped so every quantile stays inside (0, 100).

q_tau = inv_logit(lg_med + z_tau * sigma), rounded to 4 decimals, half up.
Deterministic: no randomness, no network, standard library only. Any unexpected case
shape falls back to the house baseline's numbers so the case stays valid.
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
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

EPOCH = "2026-11-19T00:00:00Z"  # release day, for the earner-trend time axis
T_SAT = 2.5          # days, ceiling-model time constant
TREND_H = 86400.0    # seconds, recent window for the earner trend
W_CEIL = 0.6         # weight of the ceiling view in the median
W_TREND = 0.4        # weight of the earner-trend view
MEDIAN_FLOOR_DROP = 0.35  # logits the median may sit below the current rate

# Ceiling percent of tracked players by class: 300 points = platinum, 90 gold, 30 silver, 15 bronze.
CEIL_PLATINUM = 2.0
CEIL_GOLD = 30.0
CEIL_SILVER = 45.0
CEIL_BRONZE = 55.0
CEIL_BRONZE_SECRET = 8.0
CEIL_GOLD_SECRET = 20.0
CEIL_SILVER_SECRET = 25.0

# Spread constants (logit units).
SIGMA_WALK_FLOOR = 0.20
SIGMA_WALK = 0.28      # per sqrt-day, like the house random walk but a touch wider
SIGMA_DISAGREE = 0.35  # per logit of median disagreement between the two views
SIGMA_MIN = 0.45
SIGMA_MAX = 3.6

# House baseline constants, used verbatim in the fallback path.
BASE_SIGMA = 0.5


def clamp(pct: float) -> float:
    return min(99.99, max(0.01, pct))


def logit(pct: float) -> float:
    return math.log(clamp(pct) / (100 - clamp(pct)))


def inv_logit(lg: float) -> float:
    lg = min(12.0, max(-12.0, lg))
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x: float) -> float | int:
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_ts(ts: str) -> float:
    """ISO 8601 UTC timestamp -> seconds since EPOCH (naive parser, stdlib only)."""
    t = ts.strip().rstrip("Z")
    date, _, clock = t.partition("T")
    y, mo, d = (int(x) for x in date.split("-"))
    clock = clock.split(".")[0]
    parts = [int(x) for x in clock.split(":")] if clock else [0, 0, 0]
    while len(parts) < 3:
        parts.append(0)
    hh, mi, ss = parts
    y0, mo0, d0 = (int(x) for x in EPOCH[:10].split("-"))
    days = (y - y0) * 365.25 + (mo - mo0) * 30.44 + (d - d0)
    return days * 86400 + hh * 3600 + mi * 60 + ss


def wls_slope(xs: list[float], ys: list[float], ws: list[float]) -> tuple[float, float]:
    """Recency-weighted least-squares slope + intercept; (0, mean) when degenerate."""
    n = len(xs)
    if n < 2 or max(xs) - min(xs) <= 0:
        return 0.0, (sum(ys) / n if n else 0.0)
    wsum = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / wsum
    my = sum(w * y for w, y in zip(ws, ys)) / wsum
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    slope = sxy / sxx if sxx > 0 else 0.0
    return slope, my - slope * mx


def ceiling_for(points, secret) -> float:
    if points >= 300:
        return CEIL_PLATINUM
    if points >= 90:
        return CEIL_GOLD_SECRET if secret else CEIL_GOLD
    if points >= 30:
        return CEIL_SILVER_SECRET if secret else CEIL_SILVER
    return CEIL_BRONZE_SECRET if secret else CEIL_BRONZE


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")

    horizon_days = max(0.0, case["horizon_minutes"] / 1440.0)
    history = case["history"]
    slots = [parse_ts(h["slot"]) for h in history]
    tracked = [float(h["players_tracked"]) for h in history]
    n = len(history)

    # Tracked-player growth: whole-history slope, damped hard over the horizon (new-player
    # arrivals fall off quickly after launch, faster than trophies accrue).
    ws_t = [1.0 + 2.0 * k / max(1, n - 1) for k in range(n)]
    t_slope, _ = wls_slope([s / 86400.0 for s in slots], tracked, ws_t)
    tracked_last = tracked[-1]
    tracked_pred = max(tracked_last, tracked_last + t_slope * horizon_days * math.exp(-horizon_days / 2.0))

    # Recent window for the earner trend: last 24 h (all reads if history is shorter).
    window_start = slots[-1] - TREND_H
    idx = [k for k in range(n) if slots[k] >= window_start]
    if len(idx) < 3:
        idx = list(range(n))
    xs = [slots[k] / 86400.0 for k in idx]
    ws = [1.0 + 2.0 * j / max(1, len(idx) - 1) for j in range(len(idx))]

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rates = [h["rates"][i] for h in history]
        seen = [r for r in rates if r is not None]
        if not seen or max(seen) <= 0.0:
            med_pct, sigma = 0.01, 1.45
        else:
            earners = [0.0 if r is None else r * tracked[k] / 100.0 for k, r in enumerate(rates)]
            ys = [earners[k] for k in idx]
            slope, intercept = wls_slope(xs, ys, ws)
            fit_vals = [slope * x + intercept for x in xs]
            resid = [y - f for y, f in zip(ys, fit_vals)]
            var = sum(r * r for r in resid) / max(1, len(resid) - 1)

            earners_last = earners[-1]
            earners_pred = max(earners_last, earners_last + slope * horizon_days * math.exp(-horizon_days / 3.0))
            rate_b = 100.0 * earners_pred / tracked_pred if tracked_pred > 0 else 0.0

            rate_now = seen[-1]
            lg_now = logit(rate_now)
            ceil_pct = ceiling_for(trophy.get("points"), trophy.get("secret"))
            if rate_now > ceil_pct:
                ceil_pct = max(ceil_pct, rate_now * 1.15)
            lg_ceil = logit(ceil_pct)
            lg_a = lg_ceil + (lg_now - lg_ceil) * math.exp(-horizon_days / T_SAT)
            if lg_now >= lg_ceil:  # already at/above its class ceiling: never predict below
                lg_a = max(lg_a, lg_now)

            lg_b = logit(clamp(rate_b)) if rate_b > 0 else lg_now
            lg_med = W_CEIL * lg_a + W_TREND * lg_b
            lg_med = max(lg_med, lg_now - MEDIAN_FLOOR_DROP)
            med_pct = inv_logit(lg_med)

            # Spread: view disagreement + fit residual through the logit derivative + walk floor.
            disagree = abs(lg_a - lg_b)
            p = med_pct / 100.0
            deriv = 100.0 / max(1e-9, p * (1 - p) * tracked_pred)
            sigma_fit = math.sqrt(max(var, 0.0)) * deriv * math.sqrt(1.0 + horizon_days)
            walk = SIGMA_WALK_FLOOR + SIGMA_WALK * math.sqrt(horizon_days)
            sigma = walk + SIGMA_DISAGREE * disagree + 0.25 * sigma_fit
            edge = min(logit(med_pct) - logit(0.02), logit(99.98) - logit(med_pct)) \
                if med_pct not in (0.01, 99.99) else 9.0
            sigma = min(max(sigma, SIGMA_MIN), SIGMA_MAX, max(0.15, edge / 1.70))

        lg = logit(med_pct)
        q = [round4(min(100.0, max(0.0, inv_logit(lg + z * sigma)))) for z in Z]
        for k in range(1, 5):
            if q[k] < q[k - 1]:
                q[k] = q[k - 1]
        rows.append({"id": trophy["id"], "q": q})
    return {"trophies": rows}


def fallback_row(rate: float, days: float) -> list:
    spread = BASE_SIGMA * math.sqrt(days)
    lg = logit(rate if rate is not None else 0)
    return [round4(100 / (1 + math.exp(-(lg + z * spread)))) for z in Z]


def house_baseline(case: dict, name: str) -> dict:
    last = case["history"][-1]
    days = case["horizon_minutes"] / 1440.0
    return {"trophies": [{"id": t["id"], "q": fallback_row(last["rates"][i], days)}
                         for i, t in enumerate(case["trophies"])]}


def forecast(case: dict, name: str) -> dict:
    try:
        return forecast_case(case, name)
    except Exception:
        return house_baseline(case, name)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        out = forecast(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
