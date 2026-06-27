"""Shared calibrated constants -- single source of truth across EdgeModel + JonnyParlay.

These values were previously duplicated as literals in BOTH repos and kept in
manual lockstep (e.g. the WNBA 3PM NB-r refit had to be applied in JonnyParlay
calibrated.py AND EdgeModel backtest_calibration.py separately). Centralising
them here makes a refit a one-line change both consumers pick up automatically.

WNBA Negative-Binomial dispersion `r` per stat. Keys are lowercase stat ids:
  fg3m  == JonnyParlay "3PM" (threes)
  reb   == JonnyParlay "REB"
  ast   == JonnyParlay "AST"
Calibrated 2026-06-04 (reb/ast: 202 players / 13,322 games, 2023-2026 WNBA RS,
min>=8) and 2026-06-26 (fg3m within-player refit r=5.0; prior 1.342 was pooled /
over-dispersed). Change values here, not in the consumers.
"""
WNBA_NB_R = {
    "reb": 10.74,
    "ast": 11.37,
    "fg3m": 5.0,
}
