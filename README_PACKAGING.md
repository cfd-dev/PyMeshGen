# PyMeshGen Packaging and Publishing Guide

## Preferred distribution path

PyMeshGen now treats **Python packaging as the primary release path**:

1. build `sdist` + `wheel`
2. validate the distributions
3. publish to **PyPI**
4. keep Windows executable packaging as an optional secondary path

For most users, the intended installation target is:

```bash
pip install pymeshgen
```

## Standard Python package build

```bash
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
```

Artifacts are written to `dist/`:

| File | Description |
|------|-------------|
| `dist/pymeshgen-<version>.tar.gz` | Source distribution |
| `dist/pymeshgen-<version>-py3-none-any.whl` | Wheel |

Versioning is sourced from the repository `VERSION` file. Update `VERSION` before creating a release tag.

## PyPI publishing

### Recommended: GitHub Actions + Trusted Publisher

This repository now includes:

- `.github/workflows/python-package.yml` — builds and validates the Python package on push / pull request
- `.github/workflows/publish-pypi.yml` — publishes to TestPyPI or PyPI through GitHub Actions

### One-time PyPI setup

In **PyPI** and **TestPyPI**, configure a Trusted Publisher for this repository:

- **Owner**: `cfd-dev`
- **Repository**: `PyMeshGen`
- **Workflow**: `publish-pypi.yml`
- **Environment**:
  - `testpypi` for TestPyPI
  - `pypi` for PyPI

After that:

- **Manual TestPyPI publish**: run the `Publish Python package` workflow with `repository = testpypi`
- **Official PyPI publish**: create a GitHub Release from a version tag such as `v1.0.1`

### Release flow

1. Update `VERSION`
2. Commit changes
3. Tag the release
   ```bash
   git tag v1.0.1
   git push origin master --tags
   ```
4. Draft / publish the GitHub Release
5. GitHub Actions publishes the wheel and sdist to PyPI

## Manual fallback publishing

If you need to publish from a local machine instead of GitHub Actions:

```bash
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
python -m twine upload dist/*
```

For TestPyPI:

```bash
python -m twine upload --repository testpypi dist/*
```

If you use a local `.pypirc`, keep it out of the repository. `.gitignore` already excludes it.

## Installing the published package

### CLI

```bash
pip install pymeshgen
pymeshgen --case ".\config\30p30n.json"
```

### GUI

```bash
pip install pymeshgen
pymeshgen-gui
```

GUI usage still depends on GUI/runtime dependencies such as `PyQt5`, `vtk`, and `pythonocc-core`.

## Optional Windows application packaging

Windows executable packaging is still available, but it is no longer the primary distribution mechanism.

```bash
python packaging\windows\build_app.py
python packaging\windows\build_app.py --all
```

Typical outputs:

| File | Location |
|------|----------|
| Executable | `dist/PyMeshGen.exe` |
| Portable ZIP | `releases/PyMeshGen-v*.zip` |
| Windows installer | `installer/PyMeshGen-Setup-*.exe` |

This path depends on a fully prepared local Python environment plus PyInstaller, and is best treated as an optional desktop-delivery workflow rather than the main release channel.

The optional Windows packaging files now live under `packaging/windows/`:

- `packaging/windows/build_app.py`
- `packaging/windows/build.bat`
- `packaging/windows/PyMeshGen.spec`
- `packaging/windows/PyMeshGen.iss`

You can also invoke the optional Windows wrapper directly:

```bash
packaging\windows\build.bat
```
