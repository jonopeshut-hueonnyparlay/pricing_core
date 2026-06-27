"""quant — pure math layer for JonnyParlay.

Leaf package: imports only the stdlib. Contains zero I/O, zero side effects, and
reads no module globals — every function's output depends solely on its inputs, so
each is property-testable in isolation. Named ``quant`` (not ``math``) deliberately:
``engine/`` sits on ``sys.path[0]``, so a package named ``math`` would shadow the
stdlib ``math`` module that these functions depend on.

run_picks.py / sgp_builder.py / mlb_sgp_builder.py import from here and re-export the
names, so existing call sites and `from run_picks import ...` test imports keep working.
"""
