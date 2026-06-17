"""
GeoSurface_Accuracy - custom dataset, RESAMPLED inputs.

Uses densified (0.1 m step) topo-intersection and geological-map shapefiles,
plus the sub-TOPO-only split/ horizon and fault surfaces.

Key changes vs the original pipeline:
  - .ts source: split/GOCAD_ASCII_All.ts (horizons + faults, below TOPO only)
  - Stratigraphic contact matching: STRAT_MAP translation (Base_X → correct surface)
  - Enhanced boundary overlap: bidirectional A↔B, mean/median/P95/max distance
  - Fault throw impact layer per horizon (IDW proxy)
  - Fault trace validation vs mapped tectonic contacts
  - Fault throw qualitative comparison (model vs Faglie/Giaciture)
  - Per-surface acceptance classification using PVRTX Thickness property

CRS baseline: EPSG:6707 (ED50/UTM32N) for all new spatial comparisons.
Faglie_carta_geologica_DEF.shp and Giaciture_FB.shp are natively EPSG:6707.
Other vector data (loaded in EPSG:7791) is reprojected to EPSG:6707 before
being passed to custom_validation functions.

Outputs are written to ./output_results_resampled/.
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd

from custom_utils import (
    read_gocad_ts_multi,
    generate_accuracy_outputs,
    generate_vertical_outputs,
    generate_combined_confidence,
    visualize_data,
)
from custom_validation import (
    STRAT_MAP,
    select_map_lines_strat,
    generate_enhanced_boundary_overlap,
    compute_fault_throw_per_horizon,
    generate_fault_validation_outputs,
    generate_fault_throw_comparison,
    compute_unit_thickness_at_grid,
    generate_acceptance_table,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WORKING_DIR = "working_files_folder"
OUTPUT_DIR = "output_results_resampled"

# Horizons + faults in one file (all below TOPO); fault surfaces have no "TOP_" prefix
TS_FILE = os.path.join("split", "GOCAD_ASCII_All.ts")

SECTIONS_FILE = "3D_SectionsGrid.shp"
MAPS_FILE = "Limiti_CartaGeol_rsmpl_0.1.shp"          # stratigraphic contacts (resampled)
TOPO_FILE = "3D_Topo_Intersections_rsmpl_0.1.shp"     # horizon topo-intersections
FAULTS_TOPO_FILE = "3D_Topo_Intersections_Faults_rsmpl_0.1.shp"
LIMITI_ORIG_FILE = "Limiti_CartaGeol.shp"              # tectonic contacts (only non-resampled has them)
FAGLIE_FILE = "Faglie_carta_geologica_DEF.shp"         # native EPSG:6707
GIACITURE_FILE = "Giaciture_FB.shp"                   # native EPSG:6707

CRS_MODEL = "EPSG:7791"   # RDN2008/UTM32N — model coordinate space
CRS_VAL = "EPSG:6707"     # ED50/UTM32N   — CRS baseline for new validation outputs

GRID_SPACING = 200.0
LINE_STEP = 200.0
MAPS_STEP = 100.0
BUFFER_DIST_M = 50.0


def main():
    print("Starting custom geological accuracy analysis (resampled dataset)...")

    if not os.path.exists(WORKING_DIR):
        print(f"The folder {WORKING_DIR} does not exist.")
        return None

    # --- Load surfaces: horizons (TOP_*) and faults (F*) ---
    ts_path = os.path.join(WORKING_DIR, TS_FILE)
    surfaces_data = read_gocad_ts_multi(ts_path, read_thickness=True)
    if not surfaces_data:
        print("No surfaces found in the .ts file.")
        return None

    # TOP_OSV is absent from GOCAD_ASCII_All.ts; load it from its individual file
    osv_path = os.path.join(WORKING_DIR, "split", "GOCAD_ASCII_TOP_OSV.ts")
    if os.path.exists(osv_path):
        osv_data = read_gocad_ts_multi(osv_path, read_thickness=True)
        surfaces_data.update(osv_data)

    # Supplement thickness for surfaces stored as VRTX (no property columns) in All.ts.
    # TOP_DPR is the only affected surface; load from its individual split/ file.
    split_dir = os.path.join(WORKING_DIR, "split")
    for sname_key, sdata in list(surfaces_data.items()):
        if not sname_key.upper().startswith('TOP_'):
            continue
        t = sdata.get('thickness')
        has_thickness = t is not None and len(t) > 0 and not np.all(np.isnan(t))
        if has_thickness:
            continue
        # Find matching individual .ts file in split/
        candidates = [f for f in os.listdir(split_dir)
                      if f.startswith('GOCAD_ASCII_TOP_') and f.endswith('.ts')
                      and f not in ('GOCAD_ASCII_All.ts',)]
        for cand in candidates:
            ind_data = read_gocad_ts_multi(os.path.join(split_dir, cand), read_thickness=True)
            match = next((v for k, v in ind_data.items()
                          if k.upper().replace('_', '') == sname_key.upper().replace('_', '')), None)
            if match is not None and match.get('thickness') is not None:
                surfaces_data[sname_key]['thickness'] = match['thickness']
                print(f"  Supplemented thickness for {sname_key} from {cand}")
                break

    horizon_surfaces = {k: v for k, v in surfaces_data.items() if k.upper().startswith('TOP_')}
    fault_surfaces = {k: v for k, v in surfaces_data.items() if not k.upper().startswith('TOP_')}
    print(f"Horizon surfaces ({len(horizon_surfaces)}): {list(horizon_surfaces.keys())}")
    print(f"Fault surfaces  ({len(fault_surfaces)}): {list(fault_surfaces.keys())}")

    # --- Load vector data ---
    sections_all = gpd.read_file(os.path.join(WORKING_DIR, SECTIONS_FILE))
    maps_all = gpd.read_file(os.path.join(WORKING_DIR, MAPS_FILE))
    topo_all = gpd.read_file(os.path.join(WORKING_DIR, TOPO_FILE))

    faults_topo = gpd.read_file(os.path.join(WORKING_DIR, FAULTS_TOPO_FILE)).to_crs(CRS_VAL)
    limiti_orig = gpd.read_file(os.path.join(WORKING_DIR, LIMITI_ORIG_FILE))
    faglie_gdf = gpd.read_file(os.path.join(WORKING_DIR, FAGLIE_FILE))   # native EPSG:6707
    giaciture_gdf = gpd.read_file(os.path.join(WORKING_DIR, GIACITURE_FILE))  # native EPSG:6707

    # Tectonic contacts for fault validation (in EPSG:6707 for comparison with faults_topo)
    tipo_col = 'Tipo' if 'Tipo' in limiti_orig.columns else None
    if tipo_col:
        tipo = limiti_orig[tipo_col].astype(str).str.lower()
        is_tectonic = tipo.str.contains('tettonic', na=False) & ~tipo.str.contains('non tettonic', na=False)
        tectonic_gdf = limiti_orig[is_tectonic].to_crs(CRS_VAL)
    else:
        tectonic_gdf = gpd.GeoDataFrame()

    print(f"Sections: {len(sections_all)}, Maps: {len(maps_all)}, "
          f"Topo: {len(topo_all)}, FaultTopo: {len(faults_topo)}, "
          f"TectonicContacts: {len(tectonic_gdf)}, "
          f"Faglie: {len(faglie_gdf)}, Giaciture: {len(giaciture_gdf)}")

    # --- Global extents (from all horizon vertices) ---
    all_vert_list = [v for v in horizon_surfaces.values()
                     if v.get('vertices') is not None and len(v['vertices']) > 0]
    all_xyz = np.vstack([v['vertices'] for v in all_vert_list])
    global_xmin, global_ymin = np.min(all_xyz[:, :2], axis=0)
    global_xmax, global_ymax = np.max(all_xyz[:, :2], axis=0)

    for gdf in (sections_all, maps_all, topo_all):
        if gdf is not None and not gdf.empty:
            b = gdf.geometry.bounds
            global_xmin = min(global_xmin, b.minx.min())
            global_xmax = max(global_xmax, b.maxx.max())
            global_ymin = min(global_ymin, b.miny.min())
            global_ymax = max(global_ymax, b.maxy.max())

    global_xlim = (global_xmin, global_xmax)
    global_ylim = (global_ymin, global_ymax)
    study_bbox = (global_xmin, global_ymin, global_xmax, global_ymax)

    # --- Per-horizon loop ---
    results = {}
    all_vertices = []
    enhanced_overlap_results = []

    for sname, data in horizon_surfaces.items():
        print(f"\n--- Surface: {sname} ---")
        vertices = data.get('vertices')
        triangles = data.get('triangles')
        thickness = data.get('thickness')
        if vertices is None or len(vertices) == 0:
            print("  No vertices, skipping.")
            continue
        all_vertices.append(vertices)

        # Correctly matched map contacts using stratigraphic translation table
        maps_for_surface = select_map_lines_strat(maps_all, sname)
        print(f"  Map contacts matched (strat): {len(maps_for_surface)} features")

        acc_outputs = generate_accuracy_outputs(
            vertices, None, sections_all, OUTPUT_DIR,
            use_wells=False, use_sections=True,
            use_maps=True, maps_shp=maps_all, maps_step=MAPS_STEP,
            grid_spacing=GRID_SPACING, line_step=LINE_STEP, surface_name=sname,
            xlim=global_xlim, ylim=global_ylim, crs_proj=CRS_MODEL,
            maps_lines_prefiltered=maps_for_surface,
        )

        vert_outputs = None
        try:
            vert_outputs = generate_vertical_outputs(
                vertices, triangles, None, sections_all,
                acc_outputs.get('grid_points'), acc_outputs.get('GX'), acc_outputs.get('GY'),
                acc_outputs.get('mask'), OUTPUT_DIR, sname, idw_power=2,
                topo_shp=topo_all, xlim=global_xlim, ylim=global_ylim, crs_proj=CRS_MODEL
            )
            if vert_outputs:
                print(f"  Vertical: {vert_outputs.get('samples', 0)} checkpoints "
                      f"({vert_outputs.get('samples_topo', 0)} from topo).")
        except Exception as e:
            print(f"  Error in vertical confidence: {e}")

        combined_outputs = None
        if vert_outputs is not None:
            combined_outputs = generate_combined_confidence(
                acc_outputs, vert_outputs, OUTPUT_DIR, sname,
                crs_proj=CRS_MODEL, alpha=0.5, mode="min"
            )

        # Enhanced boundary overlap (corrected stratigraphic matching + bidirectional + distance stats)
        overlap_result = None
        try:
            overlap_result = generate_enhanced_boundary_overlap(
                topo_all, maps_for_surface, sname, OUTPUT_DIR,
                buffer_dist=BUFFER_DIST_M, xlim=global_xlim, ylim=global_ylim,
            )
            if overlap_result:
                print(f"  Boundary overlap A→B: {overlap_result['overlap_pct_topo_in_map']:.1f}%  "
                      f"B→A: {overlap_result['overlap_pct_map_in_topo']:.1f}%  "
                      f"P95: {overlap_result['p95_distance_m']:.0f} m")
                enhanced_overlap_results.append(overlap_result)
        except Exception as e:
            print(f"  Error in enhanced boundary overlap: {e}")
            import traceback; traceback.print_exc()

        # Fault throw impact layer per horizon
        fault_throw_out = None
        try:
            gp = acc_outputs.get('grid_points')
            GX = acc_outputs.get('GX')
            GY = acc_outputs.get('GY')
            mask = acc_outputs.get('mask')
            if gp is not None and fault_surfaces:
                fault_throw_out = compute_fault_throw_per_horizon(
                    vertices, fault_surfaces, OUTPUT_DIR, sname,
                    gp, GX, GY, mask,
                    offset_m=100, crs_proj=CRS_MODEL,
                    xlim=global_xlim, ylim=global_ylim,
                )
                if fault_throw_out:
                    print(f"  Fault throw impact: {len(fault_throw_out['throw_sample_values'])} "
                          f"sample points, mean={np.nanmean(fault_throw_out['throw_sample_values']):.1f} m")
        except Exception as e:
            print(f"  Error in fault throw per horizon: {e}")
            import traceback; traceback.print_exc()

        # Unit thickness for acceptance criteria
        # acc_outputs['grid_points'] is already the masked subset (grid_points[mask])
        gp_masked = acc_outputs.get('grid_points')
        thickness_at_grid = None
        if gp_masked is not None:
            try:
                thickness_at_grid = compute_unit_thickness_at_grid(
                    vertices, thickness, gp_masked
                )
                valid_t = thickness_at_grid[~np.isnan(thickness_at_grid)]
                if len(valid_t) > 0:
                    print(f"  Unit thickness at grid: mean={np.nanmean(valid_t):.1f} m, "
                          f"valid={len(valid_t)}/{len(thickness_at_grid)} nodes")
            except Exception as e:
                print(f"  Error in thickness interpolation: {e}")

        try:
            visualize_data(
                vertices, triangles, None, sections_all, apply_smoothing=False,
                smoothing_iterations=3, smoothing_factor=0.2, crs=CRS_MODEL,
                output_filename=f'model_dataset_{sname}.png',
                grid_points=acc_outputs.get('grid_points'), surface_name=sname,
                show_plot=False, xlim=global_xlim, ylim=global_ylim, output_dir=OUTPUT_DIR
            )
        except Exception as e:
            print(f"  Visualization error: {e}")

        results[sname] = {
            'vertices': vertices,
            'triangles': triangles,
            'grid_points': acc_outputs.get('grid_points'),
            'horizontal_weights': acc_outputs.get('weights'),
            'vertical_confidence': vert_outputs,
            'combined_confidence': combined_outputs,
            'overlap_result': overlap_result,
            'thickness_at_grid': thickness_at_grid,
            'vert_outputs': vert_outputs,
        }

    # --- Combined model footprint ---
    if all_vertices:
        try:
            visualize_data(
                np.vstack(all_vertices), None, None, sections_all, apply_smoothing=False,
                smoothing_iterations=0, smoothing_factor=0.0, crs=CRS_MODEL,
                output_filename='model_dataset.png', grid_points=None,
                surface_name='model', show_plot=False,
                xlim=global_xlim, ylim=global_ylim, output_dir=OUTPUT_DIR
            )
        except Exception as e:
            print(f"Combined visualization error: {e}")

    # --- Boundary overlap summary ---
    if enhanced_overlap_results:
        summary_path = os.path.join(OUTPUT_DIR, 'boundary_overlap_summary.csv')
        pd.DataFrame(enhanced_overlap_results).to_csv(summary_path, index=False)
        print(f"\nWrote {summary_path} ({len(enhanced_overlap_results)} rows)")

    # --- Fault validation ---
    print("\n--- Fault validation ---")
    try:
        generate_fault_validation_outputs(
            faults_topo, tectonic_gdf, OUTPUT_DIR, buffer_dists=(25, 50, 100)
        )
    except Exception as e:
        print(f"Error in fault validation: {e}")
        import traceback; traceback.print_exc()

    # --- Fault throw qualitative comparison ---
    print("\n--- Fault throw comparison ---")
    try:
        generate_fault_throw_comparison(
            fault_surfaces, faglie_gdf, giaciture_gdf, study_bbox, OUTPUT_DIR
        )
    except Exception as e:
        print(f"Error in fault throw comparison: {e}")
        import traceback; traceback.print_exc()

    # --- Acceptance table ---
    print("\n--- Acceptance table ---")
    try:
        generate_acceptance_table(results, OUTPUT_DIR)
    except Exception as e:
        print(f"Error in acceptance table: {e}")
        import traceback; traceback.print_exc()

    print("\nAnalysis completed (resampled).")
    return results


if __name__ == "__main__":
    data = main()
