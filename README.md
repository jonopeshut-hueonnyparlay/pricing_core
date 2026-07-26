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

## Releasing (ADR-002)

Both consumers pin this package by version tag, not a raw git SHA. Cutting a
release is the repo owner's explicit responsibility -- it is a deliberate act,
not automated:

1. Bump `version` in `pyproject.toml` (semver: patch for non-breaking fixes,
   minor for additive API, major for breaking changes to `quant`/
   `game_line_pricing`'s public functions).
2. `pip install -e . --no-deps` locally to refresh the installed metadata
   (`pricing_core.__version__` reads it dynamically via
   `importlib.metadata` -- there is no second hardcoded version string to
   forget).
3. `git tag -a vX.Y.Z -m "..."` at the commit to release, then
   `git push origin vX.Y.Z`. **Push the tag before** re-pinning either
   consumer -- a pin referencing a tag that isn't pushed yet breaks
   `pip install -r requirements.txt` anywhere that resolves it fresh (CI, a
   new machine, a deploy).
4. From EdgeModel: `python scripts/bump_pricing_core.py vX.Y.Z` -- re-pins
   EdgeModel's `requirements.txt` *and* `requirements.lock.txt`, plus
   JonnyParlay's `requirements.txt`, atomically.
5. `python scripts/bump_pricing_core.py vX.Y.Z --check` -- confirms all three
   pins agree (this doubles as the cross-repo compatibility validation ADR-002
   calls for; a genuine blocking CI check across the two separate GitHub
   repos is a separate, not-yet-decided piece of work -- see ADR-002's Open
   Questions).
6. Run each repo's full test suite against the new pin before committing.
