"""pricing_core -- the single shared distribution->probability + game-line pricing
library for the EdgeModel <-> JonnyParlay system.

The canonical math is shipped as TOP-LEVEL modules so existing call sites resolve
unchanged once the in-repo copies are deleted and this is editable-installed:
  * ``quant``               -- distributions / derived / odds / copula (Normal, Poisson,
                               Negative Binomial CDFs; mlb_ml_from_nb; devig helpers)
  * ``game_line_pricing``   -- the canonical game-line engine (anchored Normal/NB pricers)

Single-box, single-operator: editable-installed (`pip install -e`). If a second
machine ever appears, build a wheel and pin `pricing-core==X.Y.Z` -- imports and
tests are unchanged.
"""
__version__ = "0.1.0"
