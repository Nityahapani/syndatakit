# Publishing syndatakit

Two targets: GitHub (free, always first) then PyPI.

---

## Step 1 — Push to GitHub

```bash
cd syndatakit-v2

# Initialise repo
git init
git add .
git commit -m "feat: syndatakit v2.0.0 — research-grade synthetic data for finance & econometrics"

# Create repo on GitHub (github.com → New repository → name: syndatakit)
# Then connect and push:
git remote add origin https://github.com/<your-username>/syndatakit.git
git branch -M main
git push -u origin main
```

That's it. GitHub Actions CI will run automatically on push (`.github/workflows/ci.yml`).
Check: `github.com/<you>/syndatakit/actions`

---

## Step 2 — Publish to PyPI

### One-time setup (do this once, ~10 minutes)

**2a. Create a PyPI account**
- https://pypi.org/account/register/
- Verify your email

**2b. Enable two-factor authentication**
- https://pypi.org/manage/account/two-factor/
- Use an authenticator app (Google Authenticator, 1Password, etc.)
- PyPI requires 2FA before you can publish

**2c. Configure Trusted Publishing (no API key needed)**
- Go to: https://pypi.org/manage/account/publishing/
- Click **"Add a new pending publisher"**
- Fill in:
  - PyPI Project Name: `syndatakit`
  - GitHub owner: `<your GitHub username>`
  - Repository name: `syndatakit`
  - Workflow filename: `publish.yml`
  - Environment name: `pypi`
- Click **Add**

**2d. Create the `pypi` environment in GitHub**
- Go to: `github.com/<you>/syndatakit` → Settings → Environments → New environment
- Name: `pypi`
- (Optional) Add yourself as a required reviewer for production protection

### Publish a release

```bash
# Tag the release — this triggers GitHub Actions automatically
git tag v2.0.0
git push --tags
```

GitHub Actions will:
1. Run all 130 tests on Python 3.9, 3.10, 3.11, 3.12
2. If all pass → build the distribution
3. Run `twine check dist/*` to verify the package
4. Publish to PyPI via OIDC (no password or API key needed)

Check the release: https://pypi.org/project/syndatakit/

### Verify the published package

```bash
pip install syndatakit
syndatakit list
syndatakit generate fred_macro --rows 1000 --output macro.csv
```

---

## Future releases

```bash
# Bump version in two places:
# 1. pyproject.toml  → version = "2.1.0"
# 2. syndatakit/__init__.py → __version__ = "2.1.0"

git add pyproject.toml syndatakit/__init__.py
git commit -m "bump: v2.1.0"
git tag v2.1.0
git push && git push --tags
# GitHub Actions handles the rest
```

---

## Manual publish (alternative to GitHub Actions)

If you prefer to publish without GitHub Actions:

```bash
pip install build twine

# Build
python -m build

# Check
twine check dist/*

# Upload to TestPyPI first (always recommended)
twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ syndatakit
syndatakit list   # verify it works

# Upload to real PyPI
twine upload dist/*
```

For credentials, create `~/.pypirc`:
```ini
[pypi]
username = __token__
password = pypi-<your-api-token>

[testpypi]
username = __token__
password = pypi-<your-testpypi-token>
```

---

## If `syndatakit` is already taken on PyPI

Check: https://pypi.org/project/syndatakit/

If taken, rename in `pyproject.toml`:
```toml
name = "syndatakit-finance"   # or "opensyndata", "finsyndata"
```

The CLI entrypoint (`syndatakit` command) stays the same regardless of package name.
