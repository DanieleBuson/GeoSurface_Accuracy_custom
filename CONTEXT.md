# GeoSurface_Accuracy_custom — context

This repository applies the **GeoSurface_Accuracy** accuracy-evaluation method
(originally [BaterHub/GeoSurface_Accuracy](https://github.com/BaterHub/GeoSurface_Accuracy);
see [`THEORY.md`](./THEORY.md) for the full method, fully self-contained in this repo) to
a different dataset: a 7-surface 3D geological model of an area in Trentino, with
**no wells** and a different set of GIS inputs. All shared logic (readers, IDW, plotting,
boundary-overlap) lives in [`files_utils.py`](./files_utils.py) (vendored here so this
folder is self-contained); [`custom_utils.py`](./custom_utils.py) is a thin wrapper that
re-exports it.

See [`README.md`](./README.md) for setup/run instructions (local and Google Colab).

## Dataset

All inputs live in [`working_files_folder/`](./working_files_folder/):

| File | Content |
| --- | --- |
| `GOCAD_ASCII_All.ts` | 7 surfaces (`TOP_ARV`, `TOP_DPR`, `TOP_LOP`, `TOP_RTZinf`, `TOP_FMZ`, `TOP_RTZsup`, `TOP_OSV`), 343,861 vertices total |
| `3D_SectionsGrid.shp` | 22 generic 3D section-trace lines (field `Name`), used as a "sections" checkpoint source for **every** surface |
| `3D_Topo_Intersections.shp` / `3D_Topo_Intersections_rsmpl_0.1.shp` | 7 / 231 features — per surface, the 3D line where that surface **as computed by the modeling software** crops out at the topography (field `Horizon` = `TOP_ARV`, `TOP_DPR`, ...). Geometry Z = DEM elevation. |
| `Limiti_CartaGeol.shp` / `Limiti_CartaGeol_rsmpl_0.1.shp` | 417 / 194 features — **independently, explicitly digitized** geological-map contact lines (from field mapping / GIS), with `Base_di` (links a contact to a surface, e.g. `Base ARV`, `Base RTZ inf`), `Tipo` (tectonic vs non-tectonic/stratigraphic contact), and `Affidabili` (reliability: `Osservato`/`Dedotto`/`Ipotizzato`/`Coperto`) |

There are **no well shapefiles** in this dataset, so `use_wells=False` everywhere.

## CRS

All shapefiles and the `.ts` model use the same projected meters, treated as
**`EPSG:7791`** (RDN2008 / UTM zone 32N) throughout both pipelines — chosen because
it is numerically consistent across `.ts`, `3D_SectionsGrid`, `3D_Topo_Intersections*`
and `Limiti_CartaGeol*`, even though a couple of `.prj` files resolve to `EPSG:6707`
at only ~70% confidence (same underlying UTM32N projected meters either way).

Per the "Linee Guida" PDF (§5, production workflow step 1), the **national GeoIT3D
submission standard** re-projects models to **`EPSG:6708` (RDN2008 / UTM zone 33N)**.
This dataset is **not** reprojected here — this is a local accuracy test, not a
submission package — but anyone preparing a real GeoIT3D delivery from this model
would need that reprojection step.

## How THEORY.md concepts map onto this dataset

(See [`THEORY.md`](./THEORY.md) §5 for the full explanation; summary below.)

| THEORY.md concept | This dataset |
| --- | --- |
| Wells (`p=1`) | Not present — `use_wells=False` for all surfaces |
| Sections (`p=2`) | `3D_SectionsGrid.shp`, applied unfiltered to **all 7 surfaces** (it's a generic section grid, not surface-specific) |
| Maps (`p=3`) | `Limiti_CartaGeol*.shp`, non-tectonic contacts matched to a surface via `Base_di` → `_normalize_map_code` == `_surface_short_code(surface)`, each sampled point weighted by `Affidabili` reliability (`AFFIDABILI_WEIGHTS`) |
| Vertical Z checkpoints | `3D_Topo_Intersections*.shp`, matched via `Horizon` → `_normalize_horizon_code`. Geometry Z is the DEM elevation at the modeled outcrop — the most direct ground truth available |
| Mapping CSVs (`surface_data_mapping.csv` / `surface_checkpoint_edges.csv`) | **Not used.** Maps/topo matching is fully automatic via short codes; there's nothing to configure per surface |

`Base_di` → surface matching covers 6 of the 7 surfaces. **`TOP_DPR` has no
matching `Base_di` value** (`"Base DPR"` does not exist in `Limiti_CartaGeol`), so
for `TOP_DPR` the maps tier contributes nothing and the boundary-overlap metric is
skipped — this is expected, not a bug.

## Boundary-overlap metric (new)

In addition to the horizontal/vertical/combined confidence outputs described in
THEORY.md, this dataset lets us compare two **independent** representations of each
surface's outcrop trace:

- **`3D_Topo_Intersections`** — the trace **interpolated by the modeling software**
  (surface ∩ DEM).
- **`Limiti_CartaGeol`** (non-tectonic contacts only) — the trace **explicitly mapped
  in the field/GIS**, independent of the 3D model.

`generate_boundary_overlap_outputs` buffers the mapped contact by a fixed
**50 m tolerance**, then computes the **% of the topo-intersection line's length**
that falls inside that buffer. A high % means the model's computed outcrop agrees
with the independently mapped geology within 50 m; a low % flags a real divergence
between the 3D model and the field mapping for that horizon — that is the whole
point of the metric, it is meant to surface exactly this kind of mismatch.

Results from the two test runs (identical between original and resampled inputs,
as expected — the metric is a length-ratio property of the geometry, not sensitive
to vertex density):

| Surface | Overlap % (50 m buffer) |
| --- | --- |
| TOP_ARV | 32.4% |
| TOP_DPR | — (no `Base_di` match, skipped) |
| TOP_LOP | 16.0% |
| TOP_RTZinf | 0.0% |
| TOP_FMZ | 0.0% |
| TOP_RTZsup | 0.0% |
| TOP_OSV | 24.8% |

The three 0% surfaces (`RTZinf`, `FMZ`, `RTZsup`) have their closest topo↔map
features 65–96 m apart — everywhere outside the 50 m buffer — i.e. a genuine
spatial disagreement between the modeled outcrop and the mapped contact for those
horizons, not a CRS or matching error.

## Relevant sections of the "Linee Guida" PDF

[`Petricca_etal2025_LineeGuida_StrutturaBD_Nazionale_ModGeo3D.pdf`](./Petricca_etal2025_LineeGuida_StrutturaBD_Nazionale_ModGeo3D.pdf)
(ISPRA, PNRR GeoSciencesIR) defines the national 3D geological model database
structure. The parts most relevant here:

- **§3.2, Tabella 3 (`evaluation_method`)** and **Tabella 4 (`observation_method`)**
  define controlled vocabularies for *how* a surface/contact was determined
  (compilation, direct observation, indirect method, inferred, surveyed; resistivity/
  seismic/borehole/outcrop surveys, etc.). The `Affidabili` field in
  `Limiti_CartaGeol` (`Osservato`/`Dedotto`/`Ipotizzato`/`Coperto`) is this
  dataset's local analogue of that evaluation-method/reliability concept, and is
  what drives the maps-tier reliability weighting (`AFFIDABILI_WEIGHTS`).
- **§5 (production workflow)** specifies re-projection to `EPSG:6708` (RDN2008/UTM33N)
  for the national submission package — see the CRS note above.
- **§4.4 ("Tool a supporto della valutazione dell'accuratezza")** explicitly points
  at the `GeoSurface_Accuracy` tool (this repository) as ISPRA's reference
  implementation for evaluating the accuracy of 3D model surfaces before submission
  — i.e. the additions in this folder extend the very tool the guidelines describe.

## How to run

See [`README.md`](./README.md) for full setup instructions, including how to run this on
Google Colab. Short version, from a Python environment with
[`requirements.txt`](./requirements.txt) installed:

```bash
# Original (non-resampled) topo-intersections + map contacts
python run_accuracy_original.py
# or: jupyter nbconvert --to notebook --execute --inplace GeoSurface_Accuracy_Custom_original.ipynb

# Resampled (0.1 m step) topo-intersections + map contacts
python run_accuracy_resampled.py
# or: jupyter nbconvert --to notebook --execute --inplace GeoSurface_Accuracy_Custom_resampled.ipynb
```

- **Original** uses `3D_Topo_Intersections.shp` (7 features) and `Limiti_CartaGeol.shp`
  (417 features) → outputs in `output_results_original/`.
- **Resampled** uses `3D_Topo_Intersections_rsmpl_0.1.shp` (231 features) and
  `Limiti_CartaGeol_rsmpl_0.1.shp` (194 features) → outputs in
  `output_results_resampled/`. The denser vertex sampling gives many more vertical
  checkpoints (tens of thousands vs thousands per surface) but the same
  `overlap_pct` and IDW results, which is a useful consistency check.

Both runs produce, per surface: `horizontal_confidence_grid_*.csv` (with populated
`dist_maps`/`weight_maps`), `vertical_confidence_grid_*.csv`,
`combined_confidence_grid_*.csv`, heatmaps/HTML maps, and (for the 6 surfaces with
a `Base_di` match) `boundary_overlap_<surface>.csv`/`.png`, plus a single
`boundary_overlap_summary.csv` (6 rows — `TOP_DPR` excluded).
