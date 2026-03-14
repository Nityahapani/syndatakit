.PHONY: install test build clean release

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short

test-fast:
	pytest tests/ -q --tb=line

build: clean
	python -m build
	twine check dist/*

clean:
	rm -rf dist/ build/ *.egg-info syndatakit.egg-info

release:
	@read -p "Version (e.g. 2.1.0): " v; \
	sed -i "s/^version .*/version         = \"$$v\"/" pyproject.toml; \
	sed -i "s/__version__ = .*/__version__ = \"$$v\"/" syndatakit/__init__.py; \
	git add pyproject.toml syndatakit/__init__.py; \
	git commit -m "bump: v$$v"; \
	git tag v$$v; \
	git push && git push --tags; \
	echo "Tagged v$$v — GitHub Actions will publish to PyPI automatically."
