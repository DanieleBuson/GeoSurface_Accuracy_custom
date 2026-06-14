# THEORY — Surface Accuracy & Boundary-Overlap for the Trentino 7-Surface Model

This document explains **what the pipeline in this repository computes, why, and what every
generated file means**. It is self-contained: you do not need access to any other repository
to understand it.

## 1. Where this comes from

This pipeline is a dataset-specific application of the **GeoSurface_Accuracy** tool
(originally [BaterHub/GeoSurface_Accuracy](https://github.com/BaterHub/GeoSurface_Accuracy)),
which estimates *confidence maps* for geological surfaces stored in GOCAD `.ts` format —
i.e. "how much do we trust the modeled surface elevation/position at each location?",
based on how close that location is to actual control data (wells, sections, mapped
contacts, topography intersections).

All reading/IDW/plotting logic lives in [`files_utils.py`](./files_utils.py) (a verbatim
copy of the upstream tool's implementation, vendored here so this folder can be cloned and
run on its own). [`custom_utils.py`](./custom_utils.py) is just a re-export of
`files_utils.py`, kept so the existing notebooks/scripts don't need to change their imports.

Compared to the original tool's reference dataset (which has wells + seismic sections),
**this dataset has no wells**, but adds two control sources the original tool didn't use:
explicitly-mapped geological contacts (`Limiti_CartaGeol*.shp`) and software-computed
topography intersections (`3D_Topo_Intersections*.shp`). Supporting those required two
additions to the shared logic, both used here:

- a **"maps" horizontal-confidence tier** (geological-map contact lines, `order_p=3`,
  weighted by field-reliability), and
- a **boundary-overlap metric** comparing the model's computed outcrop trace against the
  independently-mapped contact (new — not part of the original tool's outputs).

## 2. The dataset

All inputs live in [`working_files_folder/`](./working_files_folder/):

| File | Content |
| --- | --- |
| `GOCAD_ASCII_All.ts` | 7 surfaces (`TOP_ARV`, `TOP_DPR`, `TOP_LOP`, `TOP_RTZinf`, `TOP_FMZ`, `TOP_RTZsup`, `TOP_OSV`) of a 3D geological model of an area in Trentino, 343,861 vertices total |
| `3D_SectionsGrid.shp` | 22 generic 3D section-trace lines (field `Name`), used as a "sections" checkpoint source for **every** surface |
| `3D_Topo_Intersections.shp` / `3D_Topo_Intersections_rsmpl_0.1.shp` | 7 / 231 features — per surface, the 3D line where that surface **as computed by the modeling software** crops out at the topography (field `Horizon` = `TOP_ARV`, `TOP_DPR`, ...). Geometry Z = DEM elevation. |
| `Limiti_CartaGeol.shp` / `Limiti_CartaGeol_rsmpl_0.1.shp` | 417 / 194 features — **independently, explicitly digitized** geological-map contact lines (from field mapping / GIS), with `Base_di` (links a contact to a surface, e.g. `Base ARV`, `Base RTZ inf`), `Tipo` (tectonic vs non-tectonic/stratigraphic contact), and `Affidabili` (reliability: `Osservato`/`Dedotto`/`Ipotizzato`/`Coperto`) |

There are **no well shapefiles** in this dataset, so `use_wells=False` everywhere.

Two variants of the same run exist: **original** (`3D_Topo_Intersections.shp`,
`Limiti_CartaGeol.shp` → `output_results_original/`) and **resampled**
(`*_rsmpl_0.1.shp`, densified to a 0.1 m vertex step → `output_results_resampled/`).
The resampling only changes how densely the topo/map lines are sampled — the IDW
results and `overlap_pct` values are (by design) the same between the two runs; the
resampled run mainly produces many more vertical checkpoints.

## 3. Coordinate reference system

All shapefiles and the `.ts` model use the same projected meters, treated as
**`EPSG:7791`** (RDN2008 / UTM zone 32N) throughout — chosen because it is numerically
consistent across `.ts`, `3D_SectionsGrid`, `3D_Topo_Intersections*` and
`Limiti_CartaGeol*`, even though a couple of `.prj` files resolve to `EPSG:6707` at only
~70% confidence (same underlying UTM32N projected meters either way).

Per the "Linee Guida" PDF (§5, production workflow step 1, see §7 below), the **national
GeoIT3D submission standard** re-projects models to **`EPSG:6708` (RDN2008 / UTM zone
33N)**. This dataset is **not** reprojected here — this is a local accuracy test, not a
submission package — but anyone preparing a real GeoIT3D delivery from this model would
need that reprojection step.

## 4. Method

### 4.1 Grid + hull
For each surface, build a 2D evaluation grid at a fixed spacing (`GRID_SPACING`, 200 m in
both runs), clipped to the convex hull of that surface's footprint
(`build_grid` in `files_utils.py`).

### 4.2 Sample controls
- **Sections** (`3D_SectionsGrid.shp`): resampled along their geometry every
  `LINE_STEP` (200 m) to get dense points (`sample_lines_gdf`).
- **Maps** (`Limiti_CartaGeol*.shp`): for each surface, the subset of contacts whose
  `Base_di` matches that surface (see §5) is resampled every `MAPS_STEP` (100 m).
- **Wells**: not present in this dataset (`use_wells=False`).

### 4.3 Horizontal confidence (IDW)
For each grid node, compute the nearest distance to each available control type
(sections, maps). Convert distance `r` (km) to an inverse-distance score
`ID = 1 / (1 + r^p)`, then min–max normalize per type to `[0, 1]`
(`compute_order_weight`). The exponent `p` differs by control type — **sections use
`p=2`, maps use `p=3`** (wells would use `p=1` if present) — so point-like, higher-order
controls (maps) decay faster with distance than line-like, lower-order controls
(sections).

For the **maps** tier specifically, each sampled point's weight is additionally
multiplied by a **reliability factor** derived from that contact's `Affidabili` field
(`AFFIDABILI_WEIGHTS`, see §5.2) and re-clipped to `[0, 1]`. This means a grid node close
to a contact marked `Ipotizzato` ("hypothesized") gets less horizontal-confidence credit
than the same distance to a contact marked `Osservato` ("observed").

The final `weight_combined` is the **mean of the available per-type normalized weights**
at each grid node (`compute_horizontal_weights`). A ranking plot sorts all grid nodes by
descending `weight_combined`.

### 4.4 Vertical confidence (|ΔZ|)
Wherever checkpoints carry a Z value, the surface's interpolated elevation at that XY
(via Delaunay/linear triangulation of the surface vertices, `LinearNDInterpolator`) is
compared against the checkpoint's Z: `delta_z = z_checkpoint - z_surface`. The absolute
residual `|delta_z|` is IDW-interpolated onto the same evaluation grid
(`idw_power=2`), giving `abs_delta_z`. A normalized version is also produced:
`abs_delta_norm = 1 - (|dZ| - min)/(max - min)`, so **1 = best** (smallest residual) and
**0 = worst** (largest residual) on that surface's grid.

In this dataset, the **only source of Z checkpoints is `3D_Topo_Intersections*`**
(no wells, no depth CSVs, no section Z) — see §5.3. Geometry Z in that shapefile is the
**DEM elevation** at the point where the modeled surface crops out at the topography,
i.e. the most direct ground truth available: if the 3D model is correct, the surface's
elevation at that XY *should* equal the DEM elevation there.

### 4.5 Combined confidence (H + V)
Where both a horizontal weight `H` and a normalized vertical confidence `V` exist on the
same grid, three combinations are computed (`generate_combined_confidence`):
- **arithmetic**: `alpha*H + (1-alpha)*V`
- **geometric**: `H^alpha * V^(1-alpha)`
- **min**: `min(H, V)`

Both run scripts use `alpha=0.5` and select **`mode="min"`** for the headline
PNG/HTML — i.e. *"this location is only as trustworthy as its weakest dimension"*. The
geometric variant is always also saved as a PNG for comparison.

### 4.6 Boundary overlap (model vs. mapped trace) — new for this dataset
This dataset offers **two independent representations of each surface's outcrop trace**:

- **`3D_Topo_Intersections`** — the trace **interpolated by the modeling software**
  (surface ∩ DEM): "what the 3D model thinks the outcrop is".
- **`Limiti_CartaGeol`** (non-tectonic contacts only) — the trace **explicitly mapped in
  the field/GIS**, independent of the 3D model: "what the geologist mapped on the
  ground".

`generate_boundary_overlap_outputs` buffers the mapped contact by a fixed tolerance
(`BUFFER_DIST_M = 50 m`), then computes **what % of the topo-intersection line's total
length falls inside that buffer**. A high % means the model's computed outcrop agrees
with the independently mapped geology within 50 m; a low % flags a **real divergence
between the 3D model and the field mapping** for that horizon — that is the whole point
of the metric, it is meant to surface exactly this kind of mismatch. It is a purely 2D
planar comparison: both linesets are projected to 2D (Z dropped) before buffering/length
calculations.

## 5. How the general method maps onto this dataset

| Concept | This dataset |
| --- | --- |
| Wells (`p=1`) | Not present — `use_wells=False` for all surfaces |
| Sections (`p=2`) | `3D_SectionsGrid.shp`, applied unfiltered to **all 7 surfaces** (it's a generic section grid, not surface-specific) |
| Maps (`p=3`) | `Limiti_CartaGeol*.shp`, non-tectonic contacts matched to a surface via `Base_di` → `_normalize_map_code` == `_surface_short_code(surface)`, each sampled point weighted by `Affidabili` reliability (`AFFIDABILI_WEIGHTS`) |
| Vertical Z checkpoints | `3D_Topo_Intersections*.shp`, matched via `Horizon` → `_normalize_horizon_code`. Geometry Z is the DEM elevation at the modeled outcrop — the most direct ground truth available |
| Mapping CSVs (`surface_data_mapping.csv` / `surface_checkpoint_edges.csv`) | **Not used.** Maps/topo matching is fully automatic via short codes; there's nothing to configure per surface |

### 5.1 Surface ↔ contact/horizon matching (`_surface_short_code` & friends)

Both the maps tier and the vertical-checkpoint tier need to know *which* `Limiti_CartaGeol`
/ `3D_Topo_Intersections` features belong to *which* of the 7 `.ts` surfaces. This is done
purely by string-matching short codes (`files_utils.py`):

- `_surface_short_code("TOP_RTZinf_merged_resampled_original")` → strips the leading
  `TOP_`/`TOP`, drops everything from `_merged` onward, and removes non-alphanumerics →
  `"RTZINF"`.
- `_normalize_map_code("Base RTZ inf")` → strips a leading `Base`, removes
  non-alphanumerics → `"RTZINF"`.
- `_normalize_horizon_code("TOP_RTZinf")` → strips a leading `TOP_`/`TOP `, removes
  non-alphanumerics → `"RTZINF"`.

A feature is selected for a surface when its normalized code equals the surface's short
code. `select_map_lines_for_surface` additionally **excludes tectonic contacts** (rows
where `Tipo` contains "tettonic" but not "non tettonic") — only stratigraphic/
non-tectonic contacts represent a surface's outcrop boundary.

**`Base_di` → surface matching covers 6 of the 7 surfaces. `TOP_DPR` has no matching
`Base_di` value** (`"Base DPR"` does not exist in `Limiti_CartaGeol`), so for `TOP_DPR`
the maps tier contributes nothing (horizontal confidence falls back to the sections-only
average) and the boundary-overlap metric is skipped entirely — this is expected, not a
bug, and shows up as `TOP_DPR` being absent from `boundary_overlap_summary.csv`.

### 5.2 Reliability weighting (`AFFIDABILI_WEIGHTS`)

```python
AFFIDABILI_WEIGHTS = {
    'osservato':  1.0,   # observed directly in the field
    'dedotto':    0.75,  # deduced/inferred
    'ipotizzato': 0.5,   # hypothesized
    'coperto':    0.5,   # covered (not directly visible)
}
AFFIDABILI_DEFAULT_WEIGHT = 0.75  # missing/unknown Affidabili value
```

This is this dataset's local analogue of the ISPRA "Linee Guida" §3.2
`evaluation_method`/`observation_method` controlled vocabulary (see §7): a contact that
was directly observed in the field is worth full reliability, while a hypothesized or
covered contact contributes only half as much to the maps-tier confidence at nearby grid
nodes.

### 5.3 Vertical checkpoints come only from topo intersections

`generate_vertical_outputs` looks for Z in this priority order: well depth CSV → well
geometry, section depth CSV → section geometry, then `3D_Topo_Intersections*` geometry Z.
In this dataset, the first two sources are empty (no wells, no depth CSVs, sections have
no Z), so **every vertical checkpoint comes from `3D_Topo_Intersections*`** — each vertex
of the surface's matched topo-intersection line(s) becomes one `(x, y, z_DEM)` sample.
The **resampled** shapefiles have far more vertices (tens of thousands vs. low thousands
per surface), so the resampled run has many more vertical checkpoints — but because IDW
is driven by the *same underlying line geometry*, the resulting confidence grids and
`overlap_pct` values are essentially unchanged.

## 6. Boundary-overlap results (50 m buffer)

Results are identical between the original and resampled runs, as expected — the metric
is a length-ratio property of the line geometry, not sensitive to vertex density:

| Surface | Overlap % (50 m buffer) |
| --- | --- |
| TOP_ARV | 32.4% |
| TOP_DPR | — (no `Base_di` match, skipped) |
| TOP_LOP | 16.0% |
| TOP_RTZinf | 0.0% |
| TOP_FMZ | 0.0% |
| TOP_RTZsup | 0.0% |
| TOP_OSV | 24.8% |

The three 0% surfaces (`RTZinf`, `FMZ`, `RTZsup`) have their closest topo↔map features
65–96 m apart — everywhere outside the 50 m buffer — i.e. a genuine spatial disagreement
between the modeled outcrop and the mapped contact for those horizons, not a CRS or
matching error.

## 7. Relevant sections of the "Linee Guida" PDF

[`Petricca_etal2025_LineeGuida_StrutturaBD_Nazionale_ModGeo3D.pdf`](./Petricca_etal2025_LineeGuida_StrutturaBD_Nazionale_ModGeo3D.pdf)
(ISPRA, PNRR GeoSciencesIR) defines the national 3D geological model database structure.
The parts most relevant here:

- **§3.2, Tabella 3 (`evaluation_method`)** and **Tabella 4 (`observation_method`)** define
  controlled vocabularies for *how* a surface/contact was determined (compilation, direct
  observation, indirect method, inferred, surveyed; resistivity/seismic/borehole/outcrop
  surveys, etc.). The `Affidabili` field in `Limiti_CartaGeol` is this dataset's local
  analogue, and drives the maps-tier reliability weighting (§5.2).
- **§5 (production workflow)** specifies re-projection to `EPSG:6708` (RDN2008/UTM33N)
  for the national submission package (§3 above).
- **§4.4 ("Tool a supporto della valutazione dell'accuratezza")** explicitly points at the
  GeoSurface_Accuracy tool as ISPRA's reference implementation for evaluating the accuracy
  of 3D model surfaces before submission — i.e. this pipeline extends the very tool the
  guidelines describe.

## 8. Run parameters

Both `run_accuracy_original.py` and `run_accuracy_resampled.py` use:

| Parameter | Value | Meaning |
| --- | --- | --- |
| `GRID_SPACING` | 200 m | evaluation grid node spacing |
| `LINE_STEP` | 200 m | section-line sampling step |
| `MAPS_STEP` | 100 m | geological-map-contact sampling step |
| `BUFFER_DIST_M` | 50 m | boundary-overlap tolerance |
| `idw_power` | 2 | power for the vertical-residual IDW |
| `alpha` | 0.5 | combined-confidence weighting (H vs V) |
| `mode` | `"min"` | combined-confidence layer chosen for the headline PNG/HTML |
| `use_wells` | `False` | no well shapefiles in this dataset |
| `use_sections` | `True` | `3D_SectionsGrid.shp`, applied to every surface |
| `use_maps` | `True` | `Limiti_CartaGeol*.shp`, matched per surface via `Base_di` |

The only difference between the two run scripts is which shapefiles they load:
`run_accuracy_original.py` uses `3D_Topo_Intersections.shp` / `Limiti_CartaGeol.shp`
(→ `output_results_original/`); `run_accuracy_resampled.py` uses the `*_rsmpl_0.1.shp`
densified versions (→ `output_results_resampled/`).

## 9. Output file reference

Every output is written per-surface, with `<surface>` being the `.ts` surface name as
read from the GOCAD file (e.g. `TOP_ARV_merged_resampledoriginal`, `TOP_LOP_merged_resampled_original`,
etc. — the exact suffix comes from how the surfaces were named when the model was
exported). Two project-level outputs (`model_dataset.png`,
`boundary_overlap_summary.csv`) are also written once per run.

### 9.1 Whole-model

- **`model_dataset.png`** — footprint of the *entire* model (all 7 surfaces' vertices
  combined), drawn as a bounding-box rectangle over the shared X/Y extent, with the
  coordinate system noted and a scale bar. Gives a single "where is everything" overview.

### 9.2 Per-surface footprint

- **`model_dataset_<surface>.png`** — same style as above but for one surface only, with
  that surface's **evaluation grid points** overlaid (gray dots) so you can see the
  density/coverage of the grid used for all the confidence calculations below.

### 9.3 Horizontal confidence

- **`horizontal_confidence_grid_<surface>.csv`** — one row per evaluation-grid node
  (inside the surface's convex hull). Columns:
  - `x`, `y` — grid-node coordinates in `EPSG:7791` (projected meters).
  - `lon`, `lat` — same point reprojected to `EPSG:4326` (WGS84), for mapping.
  - `dist_wells`, `dist_wells_km`, `weight_wells` — distance to nearest well and its
    normalized IDW weight; **always empty in this dataset** (no wells).
  - `dist_sections`, `dist_sections_km`, `weight_sections` — distance (m / km) to the
    nearest sampled point on `3D_SectionsGrid.shp`, and its normalized IDW weight
    (`order_p=2`).
  - `dist_maps`, `dist_maps_km`, `weight_maps` — distance (m / km) to the nearest sampled
    point on the surface's matched `Limiti_CartaGeol*` contacts, and its normalized,
    reliability-scaled IDW weight (`order_p=3`); empty for `TOP_DPR` (no `Base_di` match).
  - `weight_combined` — mean of the available `weight_*` columns at this node; this is
    **"horizontal confidence", 0–1, higher = better**. This is the value plotted in the
    heatmap and interactive map below.

- **`horizontal_confidence_idw_<surface>.png`** — heatmap of `weight_combined` over the
  evaluation grid (`viridis_r`, scale fixed 0–1). Dark = high confidence (close to
  controls), light = low confidence (far from controls).

- **`horizontal_confidence_rank_<surface>.png`** — all grid nodes' `weight_combined`
  values sorted in descending order and plotted as a curve. Useful to see, at a glance,
  what fraction of the surface's footprint sits above any given confidence threshold
  (e.g. "what % of the grid has confidence ≥ 0.5?").

- **`distance_histogram_<surface>.png`** — histogram(s) of `dist_*_km` (one outlined
  series per available checkpoint type: sections, maps). Shows how far, in km, grid nodes
  typically are from each type of control — a quick sanity check on control density.

- **`interactive_confidence_<surface>.html`** — Plotly map of `weight_combined` on top of
  an OpenStreetMap basemap (or satellite if `MAPBOX_TOKEN` is set), with gray contour
  lines ("isolines") at every 0.1 confidence level (0.0, 0.1, ... 1.0) and a toggleable
  legend. Pan/zoom/hover to inspect confidence at any location. This is the file you
  opened in the editor (`interactive_confidence_TOP_OSV_*.html`).

### 9.4 Vertical confidence

- **`vertical_confidence_grid_<surface>.csv`** — one row per evaluation-grid node.
  Columns:
  - `x`, `y` — grid-node coordinates (`EPSG:7791`).
  - `abs_delta_z` — IDW-interpolated `|z_checkpoint - z_surface|` (meters) at this node,
    i.e. the estimated absolute vertical error of the modeled surface here, based on
    nearby topo-intersection checkpoints.
  - `abs_delta_norm` — `abs_delta_z` min-max normalized and **inverted** to `[0, 1]`
    (`1 - (val - min)/(max - min)`), so **1 = best (smallest |ΔZ|)**, **0 = worst
    (largest |ΔZ|)** *for this surface's grid*. This is what's plotted/mapped below.
  - `lon`, `lat` — same point reprojected to `EPSG:4326`.

- **`vertical_deltaZ_norm_<surface>.png`** — heatmap of `abs_delta_norm` (`plasma`, fixed
  0–1 scale). Bright = high vertical confidence (small modeled-vs-DEM residual near this
  location), dark = low vertical confidence.

- **`vertical_deltaZ_<surface>.html`** — Plotly interactive map of `abs_delta_norm`, same
  basemap/isoline/legend conventions as §9.3's interactive map, but colored with
  `plasma` and labeled "normalized |dZ| (1 = best)".

  > Note: despite the "deltaZ" name (matching the underlying residual quantity), the
  > value actually plotted is the **normalized** (0–1, inverted) confidence
  > `abs_delta_norm`, not the raw `abs_delta_z` in meters — use the CSV if you need the
  > raw residual.

### 9.5 Combined confidence (horizontal + vertical)

- **`combined_confidence_grid_<surface>.csv`** — one row per evaluation-grid node.
  Columns:
  - `x`, `y`, `lon`, `lat` — as above.
  - `horizontal_weight` — copy of `weight_combined` from §9.3 (H).
  - `vertical_norm` — copy of `abs_delta_norm` from §9.4 (V).
  - `arithmetic_combined` = `0.5*H + 0.5*V`.
  - `geometric_combined` = `H^0.5 * V^0.5` (geometric mean).
  - `min_combined` = `min(H, V)` — **this is the headline metric** (the run scripts use
    `mode="min"`): a location is only as confident as its *weakest* dimension.
  - `has_vertical` — always `True` here (only written when both H and V exist).

- **`combined_confidence_min_<surface>.png`** / **`combined_confidence_min_<surface>.html`**
  — heatmap / interactive map of `min_combined` (`magma_r`, 0–1). This is the
  **primary "overall confidence" output** of the pipeline for this dataset.

- **`combined_confidence_geometric_<surface>.png`** — heatmap of `geometric_combined`
  (`cividis`, 0–1), saved alongside the selected `min` mode purely for visual comparison
  (no separate `.html` for this one).

### 9.6 Boundary overlap (model vs. mapped trace)

Only produced for the 6 surfaces with a `Base_di` match (not `TOP_DPR`):

- **`boundary_overlap_<surface>.csv`** — single-row summary:
  - `surface` — surface name.
  - `topo_length_m` — total length (m) of this surface's `3D_Topo_Intersections*` line(s)
    (the model-computed outcrop trace).
  - `map_length_m` — total length (m) of this surface's matched non-tectonic
    `Limiti_CartaGeol*` contact line(s) (the independently mapped trace).
  - `covered_length_m` — portion of `topo_length_m` that falls inside the 50 m buffer
    around the mapped trace.
  - `overlap_pct` = `100 * covered_length_m / topo_length_m` — **the headline number**:
    % of the model's outcrop trace that agrees with the field mapping within 50 m.
  - `buffer_dist_m` — the buffer tolerance used (50 m).
  - `n_map_features` / `n_topo_features` — number of line features that went into
    `map_length_m` / `topo_length_m` (sanity check on the matching in §5.1).

- **`boundary_overlap_<surface>.png`** — map showing the mapped contact (orange line) and
  its 50 m buffer (shaded orange polygon), the topo-intersection trace (blue line), and
  the resulting `overlap_pct` in the title. Visually, the portion of the blue line inside
  the shaded area is `covered_length_m`.

- **`boundary_overlap_summary.csv`** (project-level, one file per run) — concatenation of
  all per-surface `boundary_overlap_<surface>.csv` rows (6 rows; `TOP_DPR` excluded). This
  is the table reproduced in §6.

## 10. Assumptions and limitations

- Nearest-distance IDW is a simple proxy; it does not model anisotropy, structural trends,
  or true vertical accuracy/uncertainty.
- Quality depends on correct CRS, complete control data, and sensible grid/line-sampling
  parameters (§8).
- The maps/topo matching (`Base_di`/`Horizon` → short code) is purely string-based; if a
  future version of `Limiti_CartaGeol` renames a `Base_di` value, that surface's maps tier
  and boundary-overlap output will silently go empty (as already happens for `TOP_DPR`).
- The boundary-overlap metric is a 2D planar comparison — Z is dropped from both linesets
  before buffering/length calculations, even though both source shapefiles carry Z.
- `overlap_pct` is **not** symmetric: it measures how much of the *topo* trace is covered
  by the *map* buffer, not vice versa. A surface could have `overlap_pct` near 100% while
  the mapped contact is much longer/shorter than the topo trace (compare `map_length_m`
  vs `topo_length_m`).
