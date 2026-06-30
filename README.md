# Surface confidence assessment & stratigraphic boundaries and faults overlap for 3D geological models

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DanieleBuson/GeoSurface_Accuracy_custom/blob/main/GeoSurface_Accuracy_Custom_original.ipynb)

Self-contained accuracy/confidence evaluation for the surfaces of a 3D geological model,
with dedicated **fault-trace validation** against an independent GIS reference. For each
surface it computes:

- **horizontal confidence** (IDW distance to section/map controls),
- **vertical confidence** (|ΔZ| against topography-intersection checkpoints),
- **combined confidence** (horizontal + vertical), and
- a **symmetric (bidirectional) boundary-overlap** metric comparing the model's computed
  outcrop trace against an independently mapped geological contact, plus an equivalent
  fault-validation metric.

## Origin and purpose

This repository was originally built to validate the **faults of a specific 3D geological
model** — checking that the fault traces and horizon outcrops computed by the 3D modeling
software (MOVE) agree with what was independently mapped in the field/GIS, within a stated
tolerance. The dataset-specific pieces (the `.ts`/shapefile inputs in
[`working_files_folder/`](./working_files_folder/), the `STRAT_MAP` stratigraphic
translation table, the specific field names like `Nome_fagli` or `Base_di`) live in
[`custom_validation.py`](./custom_validation.py) and `run_accuracy_original.py`.

The underlying **validation method is generic**, not tied to that original dataset. It is
based on [BaterHub/GeoSurface_Accuracy](https://github.com/BaterHub/GeoSurface_Accuracy)
(the horizontal/vertical/combined confidence logic in [`files_utils.py`](./files_utils.py),
and the IDW-based confidence-grid approach in general) and on the accuracy-evaluation
workflow described in the ISPRA national 3D geological model guidelines,
[`Petricca_etal2025_LineeGuida_StrutturaBD_Nazionale_ModGeo3D.pdf`](./Petricca_etal2025_LineeGuida_StrutturaBD_Nazionale_ModGeo3D.pdf)
(§4.4 explicitly names `GeoSurface_Accuracy` as ISPRA's reference tool for this purpose).
This repository's focus is a little different — it adds a *symmetric boundary/fault-overlap*
metric and a dedicated fault-validation path that the upstream tool didn't have — but the
core validation logic (interpolate confidence from known control points, compare a
modeled trace against an independent reference trace) is the same. It is meant to be
**reusable**: anyone with a similar 3D-model export (GOCAD `.ts` surfaces + GIS reference
shapefiles) can point this pipeline at their own dataset, or adapt
`custom_validation.py` the way this repo adapted the upstream tool. See
[CONTEXT.md](./CONTEXT.md) for how every THEORY.md concept maps onto the original
case-study dataset specifically — that mapping is the template to follow for a new dataset.

See **[THEORY.md](./THEORY.md)** for the full method and a description of every output
file, and **[CONTEXT.md](./CONTEXT.md)** for background on this dataset.

## Repository contents

| Path | Content |
| --- | --- |
| `working_files_folder/` | Input data: GOCAD `.ts` model + reference shapefiles (sections, mapped geological contacts, topo intersections) |
| `files_utils.py` | All reading/IDW/plotting/boundary-overlap logic (the generic, dataset-agnostic part) |
| `custom_utils.py` | Re-export of `files_utils.py` (kept for backward-compat imports) |
| `custom_validation.py` | Dataset-specific stratigraphic matching (`STRAT_MAP`), symmetric boundary/fault-overlap metric, and acceptance logic |
| `run_accuracy_original.py` | Pipeline entry point → `output_results_original/` |
| `GeoSurface_Accuracy_Custom_original.ipynb` | Notebook version of the script above, runnable locally or on Colab |
| `output_results_original/` | Reference outputs from a previous run (CSV/PNG/HTML) — see [THEORY.md §9](./THEORY.md#9-output-file-reference) for what each file means |
| `THEORY.md` | Method + per-output explanation |
| `CONTEXT.md` | Dataset background, CRS notes, ISPRA "Linee Guida" references |
| `Petricca_etal2025_LineeGuida_StrutturaBD_Nazionale_ModGeo3D.pdf` | ISPRA national 3D model guidelines (background reading) |

## Requirements

- Python 3.10+
- See [`requirements.txt`](./requirements.txt) (pandas, numpy, matplotlib, scipy,
  scikit-learn, pyproj, plotly, networkx, geopandas, shapely, fiona, gdal)

## Run locally

```bash
git clone https://github.com/DanieleBuson/GeoSurface_Accuracy_custom.git
cd GeoSurface_Accuracy_custom

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python run_accuracy_original.py
```

Or open `GeoSurface_Accuracy_Custom_original.ipynb` with Jupyter and run all cells.
Outputs are written to `output_results_original/` (existing files in that folder are
reference outputs from a previous run and will be overwritten).

## Run on Google Colab

1. Click the **"Open in Colab"** badge above (or open
   `GeoSurface_Accuracy_Custom_original.ipynb` from
   `https://github.com/DanieleBuson/GeoSurface_Accuracy_custom` directly in Colab).
2. Run the first code cell (**"LOAD WORKSPACE"**) — it clones this repository into
   `/content/GeoSurface_Accuracy_custom`, installs `requirements.txt`, and `cd`s into it.
   The repo already includes `working_files_folder/` with all the data, so **no manual
   upload is needed**.
3. Run the remaining cells (*Runtime ▸ Run after*, or one by one). Outputs are written to
   `output_results_original/` inside the cloned repo on the Colab VM.
4. To download results, either use the Colab file browser (left sidebar) or zip the
   output folder, e.g.:
   ```python
   import shutil
   shutil.make_archive('/content/output_results_original', 'zip', 'output_results_original')
   ```
   then download the resulting `.zip` from the file browser.

## Outputs

Each run produces, per surface: `horizontal_confidence_grid_*.csv` (with
`dist_maps`/`weight_maps` populated), `vertical_confidence_grid_*.csv`,
`combined_confidence_grid_*.csv`, heatmaps/HTML interactive maps, and (for surfaces with a
`Base_di` match) `boundary_overlap_<surface>.csv`/`.png` with symmetric
`overlap_pct_A_to_B`/`overlap_pct_B_to_A`/`overlap_pct_mean` columns, plus project-level
`boundary_overlap_summary.csv`, `fault_validation_per_fault.csv`,
`fault_validation_aggregate.csv`, `acceptance_table.csv`, and `validation_summary.csv`.
**Every file is explained in [THEORY.md §9](./THEORY.md#9-output-file-reference).**

## Reusing this on another dataset

The method does not depend on anything specific to the original case study — only on
having the following kinds of input, in the formats below. To adapt this repo to a new 3D
geological model:

| Need | Required format | Used for |
| --- | --- | --- |
| **Horizon surfaces** | GOCAD ASCII `.ts` file (one or more `TSURF` blocks), vertices + triangles, optionally a `Thickness` per-vertex property | The 3D surfaces being validated (`read_gocad_ts_multi` in `custom_utils.py`) |
| **Fault surfaces** (optional) | Same GOCAD ASCII `.ts` format, separate file or mixed into the horizons file | Fault-throw impact layer per horizon |
| **Control points/lines** (at least one of): wells, section traces, mapped contacts | Point/line shapefile, projected CRS (meters), one geometry per control; depth/Z either in the geometry or in a companion CSV keyed by point/well id | Horizontal-confidence (IDW) controls — wells get highest priority (`p=1`), sections next (`p=2`), maps last (`p=3`) |
| **Topography intersections** | Line shapefile, one feature (or feature group) per surface, attribute identifying which surface it belongs to, geometry Z = ground-truth elevation (e.g. DEM) | Vertical-confidence checkpoints — the most direct ground truth available |
| **Independently mapped geological contacts** (for boundary-overlap) | Line shapefile, one feature (or feature group) per contact, attribute linking each contact to a surface (e.g. a "base of" field), optional reliability/observation-method attribute | Symmetric `overlap_pct_*` metric: model trace vs. independently mapped trace |
| **Independently mapped faults** (for fault validation) | Line shapefile, one feature (or feature group) per fault, attribute holding a fault id/name comparable to the model's fault names | Symmetric `fault_overlap_*` metric: model fault trace vs. GIS-mapped fault trace |
| **CRS** | A single consistent **projected** CRS (meters) across every input — reproject everything to one EPSG code before running, the way `files_utils.standardize_crs` does here | All distance/length/IDW math assumes planar meters |

What to change when porting to a new dataset:
1. Point `WORKING_DIR`/`*_FILE` constants in a `run_accuracy_*.py`-style entry point at
   your files.
2. Replace `STRAT_MAP` and `select_map_lines_strat` in `custom_validation.py` with however
   your contacts attribute links to a surface name (here it's a `Base_di` string translated
   through a lookup table — yours might be a direct match, a numeric code, anything).
3. Replace the fault-id normalization (`_normalize_fault_id`) if your model/GIS fault names
   don't already share a common key.
4. Leave `files_utils.py` alone — the IDW grid, vertical-checkpoint, combined-confidence,
   and dissolve/symmetric-overlap logic are all dataset-agnostic.
5. Pick a `BUFFER_DIST_M` appropriate to your data's expected positional accuracy (this
   dataset currently uses 20 m).

See [CONTEXT.md](./CONTEXT.md) for a concrete worked example of every one of these mapping
decisions on the original case-study dataset, and [THEORY.md](./THEORY.md) for the full
math behind each metric.
