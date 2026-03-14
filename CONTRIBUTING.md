# Contributing to syndatakit

Thank you for your interest. Contributions are welcome — especially new datasets and generators.

---

## Two most impactful contributions

### 1. Add a new dataset

A dataset is a named source with a seed builder and priors. It takes about 30–60 minutes.

**Step 1 — Add metadata to `syndatakit/catalog/loader.py`**

```python
DATASETS["my_dataset"] = {
    "name":        "My Dataset Name",
    "vertical":    "Capital Markets",          # one of the four verticals
    "source":      "Source Agency Year",
    "description": "One-sentence description.",
    "columns":     ["col_a", "col_b", "col_c"],
    "col_count":   3,
    "tags":        ["CSV", "GDPR safe"],
    "fidelity":    None,                       # fill in after evaluation
    "status":      "live",
    "use_cases":   ["Use case 1", "Use case 2"],
}
```

**Step 2 — Write the seed builder**

```python
def _build_my_dataset(n: int = 2000) -> pd.DataFrame:
    """
    Build seed data from published statistics.
    Use publicly available aggregate statistics — NOT individual records.
    Cite your source in the docstring.
    """
    rng = _rng()
    return pd.DataFrame({
        "col_a": _lognorm(rng, mu=10.0, sigma=0.8, lo=100, hi=1e6, n=n).astype(int),
        "col_b": np.clip(rng.normal(3.5, 1.2, n), 0, 10).round(2),
        "col_c": _weighted(rng, ["A", "B", "C"], [50, 30, 20], n),
    })
```

Register it in `load_seed()`:
```python
builders = {
    ...,
    "my_dataset": _build_my_dataset,
}
```

**Step 3 — Add priors to `syndatakit/calibration/priors.py`**

```python
DATASET_PRIORS["my_dataset"] = PriorSet({
    "col_a": Prior("lognormal", mu=10.0, sigma=0.8, strength=2.0),
    "col_b": Prior("normal",   mu=3.5,  sigma=1.2, strength=1.5),
})
```

**Step 4 — Add tests**

In `tests/test_v2.py`, add your dataset ID to the `test_all_seeds_build` parametrize list and the `test_all_cross_sectional_datasets` list (or write a custom test for time series / panel datasets).

**Step 5 — Run the test suite**

```bash
pytest tests/ -v
```

All 130+ tests should pass. Then open a PR.

---

### 2. Add a new generator

All generators inherit from `BaseGenerator`. You must implement two methods: `fit()` and `sample()`.

```python
# syndatakit/generators/cross_sectional/my_generator.py
from ..base import BaseGenerator

class MyGenerator(BaseGenerator):
    supported_types = ["cross_sectional"]

    def _init(self, my_param=1.0, **kwargs):
        self._my_param = my_param
        # initialise your model state here

    def fit(self, data: pd.DataFrame) -> "MyGenerator":
        self._record_schema(data)
        # learn your model from data
        self._fitted = True
        return self

    def sample(self, n, filters=None, seed=None) -> pd.DataFrame:
        self._require_fitted()
        # generate n rows
        df = ...
        return self._add_syn_id(df.head(n))
```

Export it from `syndatakit/generators/cross_sectional/__init__.py` and add tests.

---

## Development setup

```bash
git clone https://github.com/yourhandle/syndatakit.git
cd syndatakit
pip install -e ".[dev]"
pytest tests/ -v
```

## Code style

- Type hints on all public functions
- Docstrings on all public classes and methods
- No external dependencies beyond `pandas`, `numpy`, `scipy` for core modules
- Optional dependencies go in `pyproject.toml` extras

## Pull request checklist

- [ ] Tests pass: `pytest tests/ -v`
- [ ] New dataset: metadata + seed builder + priors + test coverage
- [ ] New generator: subclasses `BaseGenerator` + implements `fit()` + `sample()` + tests
- [ ] No new required dependencies added to core
