## What this change does

## Checklist

- [ ] `ruff check src tests scripts` and `ruff format --check src tests scripts` are clean
- [ ] `python -m pytest -q` passes
- [ ] `python -m dynamic_trading_engine experiment --quick --workers 2` runs, and `quickhash` still matches on the same platform
- [ ] Tests added or updated for the changed behavior
- [ ] If the change moves a result: tuned, frozen, and evaluated by the protocol, with result files and manifests committed
- [ ] Documentation updated; every new number names its result file
