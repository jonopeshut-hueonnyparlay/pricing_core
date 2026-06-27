# pricing_core

The single shared **distribution → probability + game-line pricing** library for the
EdgeModel ↔ JonnyParlay betting system. Extracted so the producer (EdgeModel) and
consumer (JonnyParlay) can never drift on the pricing math ("lockstep drift").

## What's in here
- `quant/` — `distributions` (Normal / Poisson / Negative-Binomial PMFs/CDFs),
  `derived` (`mlb_ml_from_nb`, total-bases convolution, no-vig edge), `odds`
  (devig helpers), `copula` (same-game correlation).
- `game_line_pricing.py` — the canonical game-line engine: anchored Normal/NB
  pricers with a mandatory market-anchoring `trust` parameter (`trust==1.0`
  fast-path = raw; `0.25` = market-anchored).

These are shipped as **top-level** modules (`import quant`, `import game_line_pricing`)
so existing call sites in the consuming repos resolve from the installed package
**without any import changes**.

## Install (single box, single operator)
```
pip install -e C:\Dev\pricing_core
```
Editable install — edits here are immediately live in both repos. If a second
machine ever appears: `python -m build` and pin `pricing-core==X.Y.Z` in each
repo's `requirements.txt`; imports and tests don't change.

## Determinism
Pure `math`-based closed-form functions (no RNG); byte-identical across runs.
Pin this package's version alongside Python/NumPy when reproducibility matters.
