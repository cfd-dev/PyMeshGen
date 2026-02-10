# Copilot Instructions for PyMeshGen

## Build, test, and lint commands
- Install dependencies: `pip install -r requirements.txt`
- CLI mesh generation (config-driven): `python PyMeshGen.py --case "./config/30p30n.json"`
- Launch GUI: `python start_gui.py`
- Run all tests: `cd unittests; python run_tests.py`
- Run quick tests (skips heavy mesh generation): `cd unittests; python run_quick_tests.py`
- Run a specific test module: `cd unittests; python run_tests.py <test_name>` (e.g., `python run_tests.py cas_file_io`)
- Run a single test file via unittest: `cd unittests; python -m unittest test_cas_file_io.py`

## High-level architecture
- **Entry points**: `PyMeshGen.py` (CLI and library wrapper) and `start_gui.py` (GUI launcher). Both construct `Parameters` and call `core.generate_mesh`.
- **Core pipeline (`core.py`)**: parse input mesh (typically Fluent `.cas`) → construct initial front (`data_structure.front2d`) → build sizing field (`meshsize.QuadtreeSizing`) → generate boundary layers (`adfront2.adlayers2`) → generate interior mesh (`adfront2.adfront2` or `adfront2.adfront2_hybrid`) → optimize (`optimize.edge_swap`, `optimize.optimize_hybrid_grid`, `optimize.laplacian_smooth`) → merge and export via `Unstructured_Grid`.
- **Data model**: `data_structure.unstructured_grid.Unstructured_Grid` is the central mesh container; `cell_container` is the authoritative store and `cells` is a derived read-only view.
- **I/O**: `fileIO.read_cas` for Fluent input, `fileIO.vtk_io` / `fileIO.stl_io` for output and conversions; OCC/VTK conversion lives in `fileIO.occ_to_vtk`.
- **GUI**: `gui.gui_main` hosts the PyQt5 app; `gui.mesh_display` handles VTK rendering; `gui.config_manager` maps GUI parameters to `Parameters` and config JSON; import runs asynchronously in `gui.import_thread`.
- **AI/optimization modules**: `neural/` contains NN/DRL smoothing, used for advanced mesh optimization workflows.

## Key conventions
- **Configuration source**: `Parameters` loads either `config/main.json` (which points to a case file) or a direct case JSON (via `--case`). Case files live under `config/` and specify parts, sizes, and prism-layer settings.
- **Part parameters**: each part requires at least `part_name`; optional fields (e.g., `PRISM_SWITCH`, `first_height`, `max_layers`, `full_layers`) default in `Parameters._create_part_params` and are merged with mesh-derived parts via `update_part_params_from_mesh`.
- **Mesh type selection**: `Parameters.mesh_type` drives algorithm choice in `core.generate_mesh` (1 = triangular, 3 = hybrid tri/quad).
- **Unstructured grid invariants**: update mesh cells via `Unstructured_Grid.set_cells`; do not write to `cells` directly (it is derived from `cell_container`). Updating `node_coords` invalidates `bbox` and is recomputed lazily.
- **GUI/CLI messaging**: `utils.message` is the centralized message system; GUI passes its instance to `core.generate_mesh` so status updates are routed to the UI.
- **Third-party meshio**: GUI startup adds `3rd_party/meshio/src` to `sys.path` for compatibility with bundled meshio.
