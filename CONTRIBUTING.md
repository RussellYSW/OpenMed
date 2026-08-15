# Contributing to OpenMed

Thanks for your interest! **OpenMed** is the ecosystem; **`trustfed`** is the Python
package that implements it -- the two names refer to the same project. Contributions
of all sizes are welcome, and they do not have to be framework code: returning a
model, or an evaluation of someone else's model, is a first-class contribution.

## Getting set up

```bash
git clone https://github.com/RussellYSW/Trusted-TEE-for-Data-Sharing.git
cd Trusted-TEE-for-Data-Sharing
python -m pip install -e ".[dev]"
python -m pytest -q
python examples/parkinson_decline/run_demo.py
```

## Ground rules

1. **Never commit data.** Only synthetic data belongs in this repo. See
   [`docs/DATA.md`](docs/DATA.md). PRs that add `.csv`/`.xlsx`/imaging files will
   be rejected.
2. **Keep the core dependency-light.** The runtime should stay numpy-only where
   practical; heavier deps go behind optional extras.
3. **Add a test.** New aggregators, attacks, or attestor backends should come
   with unit tests under `tests/`.
4. **Document the guarantee.** For a new aggregator, state its Byzantine
   tolerance condition; for an attack, state the threat it models.

## Good first contributions

- A new robust aggregator (e.g. Bulyan) + tests, wired into `AGGREGATORS`
  and the benchmark grid.
- A Flower or NVIDIA FLARE adapter that reuses `trustfed.aggregation`.
- A real TEE `Attestor` backend (SGX/TDX/SEV-SNP).
- Improvements to the synthetic data generator (more realistic heterogeneity).

## Pull requests

- Branch from `main`, keep PRs focused, and describe the change and its
  motivation.
- Ensure `pytest` passes and the demo still runs.
- By contributing you agree your contribution is licensed under Apache-2.0.

## Code of conduct

Participation is governed by [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).
