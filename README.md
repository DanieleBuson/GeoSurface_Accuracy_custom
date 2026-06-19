# GeoSurface_Accuracy_custom — Trentino 7-Surface Model

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)

[![Open Original in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DanieleBuson/GeoSurface_Accuracy_custom/blob/main/GeoSurface_Accuracy_Custom_original.ipynb)
[![Open Resampled in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DanieleBuson/GeoSurface_Accuracy_custom/blob/main/GeoSurface_Accuracy_Custom_resampled.ipynb)

Self-contained accuracy/confidence evaluation for a 7-surface 3D geological model
(Trentino), based on the [GeoSurface_Accuracy](https://github.com/BaterHub/GeoSurface_Accuracy)
method. For each surface, it computes:

- **horizontal confidence** (IDW distance to section/map controls),
- **vertical confidence** (|ΔZ| against topography-intersection checkpoints),
- **combined confidence** (horizontal + vertical), and
- a **symmetric (bidirectional) boundary-overlap** metric comparing the model's computed
  outcrop trace against an independently mapped geological contact, plus an equivalent
  fault-validation metric.

See **[THEORY.md](./THEORY.md)** for the full method and a description of every output
file, and **[CONTEXT.md](./CONTEXT.md)** for background on the dataset.

## Repository contents

| Path | Content |
| --- | --- |
| `working_files_folder/` | Input data: GOCAD `.ts` model + shapefiles (sections, mapped geological contacts, topo intersections, original and resampled) |
| `files_utils.py` | All reading/IDW/plotting/boundary-overlap logic |
| `custom_utils.py` | Re-export of `files_utils.py` (kept for backward-compat imports) |
| `run_accuracy_original.py` | Pipeline using the **original** (non-resampled) shapefiles → `output_results_original/` |
| `run_accuracy_resampled.py` | Pipeline using the **resampled** (0.1 m step) shapefiles → `output_results_resampled/` |
| `GeoSurface_Accuracy_Custom_original.ipynb` / `_resampled.ipynb` | Notebook versions of the two scripts above, runnable locally or on Colab |
| `output_results_original/`, `output_results_resampled/` | Reference outputs from a previous run (CSV/PNG/HTML) — see [THEORY.md §9](./THEORY.md#9-output-file-reference) for what each file means |
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

# Original (non-resampled) topo-intersections + map contacts
python run_accuracy_original.py

# Resampled (0.1 m step) topo-intersections + map contacts
python run_accuracy_resampled.py
```

Or open either `.ipynb` with Jupyter and run all cells. Outputs are written to
`output_results_original/` and `output_results_resampled/` respectively (existing
files in those folders are reference outputs from a previous run and will be
overwritten).

## Run on Google Colab

1. Click one of the **"Open in Colab"** badges above (or open
   `GeoSurface_Accuracy_Custom_original.ipynb` / `_resampled.ipynb` from
   `https://github.com/DanieleBuson/GeoSurface_Accuracy_custom` directly in Colab).
2. Run the first code cell (**"LOAD WORKSPACE"**) — it clones this repository into
   `/content/GeoSurface_Accuracy_custom`, installs `requirements.txt`, and `cd`s into it.
   The repo already includes `working_files_folder/` with all the data, so **no manual
   upload is needed**.
3. Run the remaining cells (*Runtime ▸ Run after*, or one by one). Outputs are written to
   `output_results_original/` or `output_results_resampled/` inside the cloned repo on
   the Colab VM.
4. To download results, either use the Colab file browser (left sidebar) or zip the
   output folder, e.g.:
   ```python
   import shutil
   shutil.make_archive('/content/output_results_original', 'zip', 'output_results_original')
   ```
   then download the resulting `.zip` from the file browser.

> Note: the cloned repo (data + reference outputs) is ~300 MB; the first cell may take a
> minute or two on Colab.

## Outputs

Each run produces, per surface: `horizontal_confidence_grid_*.csv` (with
`dist_maps`/`weight_maps` populated), `vertical_confidence_grid_*.csv`,
`combined_confidence_grid_*.csv`, heatmaps/HTML interactive maps, and (for surfaces with a
`Base_di` match) `boundary_overlap_<surface>.csv`/`.png` with symmetric
`overlap_pct_A_to_B`/`overlap_pct_B_to_A`/`overlap_pct_mean` columns, plus project-level
`boundary_overlap_summary.csv`, `fault_validation_per_fault.csv`,
`fault_validation_aggregate.csv`, `acceptance_table.csv`, and `validation_summary.csv`.
**Every file is explained in [THEORY.md §9](./THEORY.md#9-output-file-reference).**
