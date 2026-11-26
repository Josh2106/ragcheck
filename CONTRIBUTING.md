# Contributing

Thanks for your interest in ragcheck! Bug reports, docs fixes and new components are all
welcome.

## Setup

```bash
git clone https://github.com/Josh2106/ragcheck && cd ragcheck
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Before opening a pull request

```bash
ruff format src tests
ruff check src tests
mypy                          # strict mode
pytest --cov=ragcheck         # must stay >= 90% (CI enforces it)
```

Guidelines:

- **Tests must run offline.** Never call a real API in tests. Use `FakeProvider`,
  `httpx.MockTransport` or `respx` for HTTP, and `HashingEmbedder` for embeddings.
- **Keep results deterministic.** Break ties explicitly; avoid Python's salted `hash()`.
- **Public functions are typed and documented.** Metric docstrings should state the formula
  and cite the source when one exists.
- **Property tests for invariants.** If you add a chunker, add it to
  `tests/test_properties.py` so the span/coverage/size invariants are checked.
- Use [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`,
  `docs:`, `test:`, `refactor:`, `chore:`, `ci:`).
- Add an entry under *Unreleased* in `CHANGELOG.md` for user-visible changes.

## Adding a component

See the *Extending* section of [ARCHITECTURE.md](ARCHITECTURE.md). In short: implement the
protocol from `ragcheck.protocols`, add unit tests for edge cases (empty input, Unicode,
oversize input), and register a `type:` name in `ragcheck/config.py` if it should be usable
from YAML.

## Releasing (maintainers)

1. Bump `version` in `pyproject.toml` and move *Unreleased* notes to a new section.
2. Commit, then tag: `git tag v0.2.0 && git push --tags`.
3. The `Release` workflow builds and publishes to PyPI via trusted publishing.
