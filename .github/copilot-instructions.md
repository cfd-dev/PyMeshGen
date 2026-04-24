# Copilot Instructions for PyMeshGen

## Setup, run, test, and packaging commands
- Install core dependencies: `pip install -r requirements.txt`
- Install optional OCC/ML dependencies when needed: `pip install -r requirements-optional.txt`
- Install in editable mode: `pip install -e .`
- CLI mesh generation (config-driven): `python PyMeshGen.py --case ".\\config\\30p30n.json"`
- Launch GUI: `python start_gui.py`
- Run all tests: `cd unittests; python run_tests.py`
- Run quick tests (skips heavy mesh-generation regressions): `cd unittests; python run_quick_tests.py`
- Run a named test module: `cd unittests; python run_tests.py cas_file_io`
- Run one unittest target from repo root: `python -m unittest unittests.test_bowyer_watson.TestBowyerWatsonJSONConfig.test_anw_bowyer_watson`
- Build source and wheel distributions: `python -m pip install build && python -m build`
- Preferred distribution path is Python packaging to PyPI; `packaging/windows/build_app.py` is only for optional Windows desktop packaging
- There is no repo-root lint configuration checked in; do not invent Ruff/Flake8/Pylint commands. Prefer the existing unittest scripts and focused unittest targets.

## High-level architecture
- **Entry points**: `PyMeshGen.py` is the CLI/library wrapper and `start_gui.py` is the GUI launcher. The GUI bootstraps PyQt5 and prepends `3rd_party\\meshio\\src` to `sys.path`.
- **Pipeline orchestrator**: `core.generate_mesh()` is the main hub. It resolves the input grid/front data, builds `meshsize.QuadtreeSizing`, optionally grows boundary layers with `adfront2.adlayers2.Adlayers2`, runs the selected interior meshing path, then merges/optimizes/exports the result.
- **Interior meshing branches**:
  - `mesh_type == 1` or `2`: triangular advancing-front interior meshing via `adfront2.adfront2.Adfront2`
  - `mesh_type == 3`: mixed mesh path; `triangle_to_quad_method == "q_morph"` uses a triangle-first pipeline, otherwise `adfront2.adfront2_hybrid.Adfront2Hybrid`
  - `mesh_type == 4`: Delaunay path via `delaunay.create_bowyer_watson_mesh()`, which dispatches to the Bowyer-Watson core or `delaunay.triangle_backend`
- **Delaunay package layout**: `bw_utils.py` prepares boundary input and backend dispatch; `bw_core_stable.py` contains the active Bowyer-Watson implementation; `bw_cavity.py`, `bw_predicates.py`, and `bw_types.py` hold supporting geometry/topology logic; `triangle_backend.py` wraps the Triangle executable path; `postprocess.py` and `validation.py` handle boundary recovery and topology checks.
- **Mesh container**: `data_structure.unstructured_grid.Unstructured_Grid` is the canonical mesh object shared across the pipeline.

## Configuration and data conventions
- `Parameters` loads either `config\\main.json` (which points to a case file) or a direct case JSON passed with `--case`. Automation should usually work with the case JSON under `config\\`.
- Each part requires `part_name`. Optional part fields such as `max_size`, `PRISM_SWITCH`, `first_height`, `growth_rate`, `max_layers`, `full_layers`, and `multi_direction` are defaulted in `Parameters._create_part_params()`.
- If imported mesh data contains parts missing from the config, `Parameters.update_part_params_from_mesh()` adds default part settings instead of failing.
- `mesh_type == 4` is the Delaunay/Bowyer-Watson workflow. `delaunay_backend` defaults to `"bowyer_watson"` and `triangle_point_strategy` defaults to `"equilateral"`.
- Boundary-layer cases on `mesh_type == 4` force the effective inner Delaunay backend to `triangle` through `utils.core_helpers.select_delaunay_backend()`, even if the config requested Bowyer-Watson.
- `gui.config_manager.ConfigManager` preserves part-level settings plus non-default `curves` entries when importing/exporting GUI JSON config. Curve entries with `PRISM_SWITCH == "match"` are meaningful and should not be dropped accidentally.
- `utils.message` is the shared message channel. `core.generate_mesh()` registers the GUI instance so `info`, `warning`, and `error` output is mirrored into the GUI log.

## Unstructured_Grid invariants
- `cell_container` is the only writable cell source of truth.
- `cells` is a derived read-only node-id view; update cell topology through `set_cells()` or `Unstructured_Grid.from_cells()`, not by mutating `cells` directly.
- Updating `node_coords` invalidates the cached `bbox`; it is recomputed lazily on access.

## Test guidance
- `unittests/run_quick_tests.py` is the fastest broad regression pass and skips the slow mesh-generation scenarios.
- Delaunay, boundary recovery, and boundary-layer regressions are concentrated in `unittests/test_bowyer_watson.py`; use focused unittest targets there when changing `delaunay/`, `core.py`, `meshsize/`, or `adfront2/`.
- GUI-related tests exist, but they are not the default smoke tests for algorithmic meshing changes.
