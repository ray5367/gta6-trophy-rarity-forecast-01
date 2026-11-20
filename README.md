# GTA VI trophy rarity forecaster

Forecasts each PlayStation 5 trophy's Exophase earn rate at 2026-11-26
00:00 UTC, as five quantiles (5%, 25%, 50%, 75%, 95%) per trophy.

## Model

`main.py` (Python 3.13, standard library only, no network) fits, per
trophy, `logit(rate) = a + b * ln(1 + h)` with `h` hours since the game's
launch, by least squares over the case's deduplicated reads from the last
24 hours. Reads that print `0.00%` enter floored at `0.005%` (censored,
not zero). The slope is capped at 4. The forecast is the fit read at the
target; medians are then forced non-increasing along the story chain
(chapter 1 → 2 → 3 → 4 → final mission → 100% → platinum). Quantiles are
normal quantiles of a logit-normal predictive distribution whose sigma
combines the fit's residual scatter and a random-walk floor of 0.40 per
sqrt-day of horizon.

Flat history (e.g. the practice cases, whose whole history is one frozen
page) falls back to persistence with a widened spread.

## Validation

Synthetic ground truths (power-law, saturating and declining earn-rate
curves; Exophase refresh every 3–12 h; rates printed to 2 decimals) scored
by the competition's pinball loss on the logit scale:

| scenario | this model | house curve baseline | starter |
|---|---|---|---|
| power-law, refresh 6 h | 0.49 | 0.51 | 1.05 |
| power-law, refresh 3 h | 0.47 | 0.49 | 1.00 |
| power-law, refresh 12 h | 0.58 | 0.56 | 1.13 |
| saturating | 0.33 | 0.34 | 0.48 |
| declining (dilution wins) | 0.19 | 0.21 | 0.52 |

The validation harness lives outside this repository
(`trophy-validation/sim.py` in the build workspace): it generates the
ground truth, the house recorder's reads, and scores this program, the
starter and a re-implementation of the house curve baseline side by side.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` in the input directory, writes one `<id>.out` per case.
Deterministic; runs in ~0.3 s for the two practice cases.

## License

MIT (see LICENSE).
