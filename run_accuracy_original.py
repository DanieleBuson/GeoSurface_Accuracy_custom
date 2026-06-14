"""
GeoSurface_Accuracy - custom dataset, ORIGINAL (non-resampled) inputs.

Runs the horizontal/vertical/combined confidence pipeline plus the boundary-overlap
metric on the Trentino dataset using the original (non-resampled) topo-intersection
and geological-map-boundary shapefiles. See ../GeoSurface_Accuracy_custom/CONTEXT.md
for the full explanation of how this dataset maps onto THEORY.md.

Outputs are written to ./output_results_original/.
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
    generate_boundary_overlap_outputs,
    visualize_data,
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
WORKING_DIR = "working_files_folder"
OUTPUT_DIR = "output_results_original"

TS_FILE = "GOCAD_ASCII_All.ts"
SECTIONS_FILE = "3D_SectionsGrid.shp"
MAPS_FILE = "Limiti_CartaGeol.shp"
TOPO_FILE = "3D_Topo_Intersections.shp"

CRS = "EPSG:7791"  # RDN2008 / UTM zone 32N (matches all source .prj files)

GRID_SPACING = 200.0  # evaluation grid node spacing (m)
LINE_STEP = 200.0     # section-line sampling step (m)
MAPS_STEP = 100.0     # geological-map-contact sampling step (m)
BUFFER_DIST_M = 50.0  # boundary-overlap tolerance (m)


def main():
    print("Starting custom geological accuracy analysis (original dataset)...")

    if not os.path.exists(WORKING_DIR):
        print(f"The folder {WORKING_DIR} does not exist.")
        return None

    ts_path = os.path.join(WORKING_DIR, TS_FILE)
    surfaces_data = read_gocad_ts_multi(ts_path)
    surface_names = list(surfaces_data.keys())
    if not surface_names:
        print("No surface found in the .ts file.")
        return None
    print(f"Found {len(surface_names)} surfaces: {surface_names}")

    sections_all = gpd.read_file(os.path.join(WORKING_DIR, SECTIONS_FILE))
    maps_all = gpd.read_file(os.path.join(WORKING_DIR, MAPS_FILE))
    topo_all = gpd.read_file(os.path.join(WORKING_DIR, TOPO_FILE))
    print(f"Sections: {len(sections_all)} features, Maps: {len(maps_all)} features, "
          f"Topo intersections: {len(topo_all)} features")

    # Global extents for consistent axes (vertices + sections + maps + topo)
    all_vert_list = [v for v in surfaces_data.values() if v.get('vertices') is not None and len(v.get('vertices')) > 0]
    all_xyz = np.vstack([v['vertices'] for v in all_vert_list])
    global_xmin, global_ymin = np.min(all_xyz[:, :2], axis=0)
    global_xmax, global_ymax = np.max(all_xyz[:, :2], axis=0)

    for gdf in (sections_all, maps_all, topo_all):
        if gdf is not None and not gdf.empty:
            bounds = gdf.geometry.bounds
            global_xmin = min(global_xmin, bounds.minx.min())
            global_xmax = max(global_xmax, bounds.maxx.max())
            global_ymin = min(global_ymin, bounds.miny.min())
            global_ymax = max(global_ymax, bounds.maxy.max())

    global_xlim = (global_xmin, global_xmax)
    global_ylim = (global_ymin, global_ymax)

    results = {}
    all_vertices = []
    overlap_results = []

    for sname, data in surfaces_data.items():
        print(f"\n--- Surface: {sname} ---")
        vertices = data.get('vertices')
        triangles = data.get('triangles')
        if vertices is None or len(vertices) == 0:
            print("No vertices for this surface, skipping.")
            continue
        all_vertices.append(vertices)

        acc_outputs = generate_accuracy_outputs(
            vertices, None, sections_all, OUTPUT_DIR,
            use_wells=False, use_sections=True,
            use_maps=True, maps_shp=maps_all, maps_step=MAPS_STEP,
            grid_spacing=GRID_SPACING, line_step=LINE_STEP, surface_name=sname,
            xlim=global_xlim, ylim=global_ylim, crs_proj=CRS
        )

        vert_outputs = None
        try:
            vert_outputs = generate_vertical_outputs(
                vertices, triangles, None, sections_all,
                acc_outputs.get('grid_points'), acc_outputs.get('GX'), acc_outputs.get('GY'),
                acc_outputs.get('mask'), OUTPUT_DIR, sname, idw_power=2,
                topo_shp=topo_all, xlim=global_xlim, ylim=global_ylim, crs_proj=CRS
            )
            if vert_outputs:
                print(f"Vertical confidence calculated with {vert_outputs.get('samples', 0)} checkpoints "
                      f"({vert_outputs.get('samples_topo', 0)} from topo intersections).")
        except Exception as e:
            print(f"Error during vertical confidence computation: {e}")

        combined_outputs = None
        if vert_outputs is not None:
            combined_outputs = generate_combined_confidence(
                acc_outputs, vert_outputs, OUTPUT_DIR, sname, crs_proj=CRS, alpha=0.5, mode="min"
            )

        overlap = None
        try:
            overlap = generate_boundary_overlap_outputs(
                topo_all, maps_all, sname, OUTPUT_DIR,
                buffer_dist=BUFFER_DIST_M, xlim=global_xlim, ylim=global_ylim
            )
            if overlap:
                print(f"Boundary overlap: {overlap['overlap_pct']:.1f}% of topo-intersection trace "
                      f"within {BUFFER_DIST_M:.0f} m of the mapped boundary.")
                overlap_results.append(overlap)
            else:
                print("Boundary overlap: skipped (no matching topo/map lines for this surface).")
        except Exception as e:
            print(f"Error during boundary overlap computation: {e}")

        try:
            visualize_data(vertices, triangles, None, sections_all, apply_smoothing=False,
                           smoothing_iterations=3, smoothing_factor=0.2, crs=CRS,
                           output_filename=f'model_dataset_{sname}.png',
                           grid_points=acc_outputs.get('grid_points'), surface_name=sname, show_plot=False,
                           xlim=global_xlim, ylim=global_ylim, output_dir=OUTPUT_DIR)
            print("Visualization completed successfully.")
        except Exception as e:
            print(f"Error during visualization: {e}")
            import traceback
            traceback.print_exc()

        results[sname] = {
            'vertices': vertices,
            'triangles': triangles,
            'grid_points': acc_outputs.get('grid_points'),
            'horizontal_weights': acc_outputs.get('weights'),
            'vertical_confidence': vert_outputs,
            'combined_confidence': combined_outputs,
            'boundary_overlap': overlap,
        }

    if all_vertices:
        try:
            all_vertices_arr = np.vstack(all_vertices)
            visualize_data(all_vertices_arr, None, None, sections_all, apply_smoothing=False,
                           smoothing_iterations=0, smoothing_factor=0.0, crs=CRS,
                           output_filename='model_dataset.png', grid_points=None,
                           surface_name='model', show_plot=False,
                           xlim=global_xlim, ylim=global_ylim, output_dir=OUTPUT_DIR)
            print("Combined visualization saved (model_dataset.png).")
        except Exception as e:
            print(f"Error during combined visualization: {e}")
            import traceback
            traceback.print_exc()

    if overlap_results:
        summary_path = os.path.join(OUTPUT_DIR, 'boundary_overlap_summary.csv')
        pd.DataFrame(overlap_results).to_csv(summary_path, index=False)
        print(f"\nWrote {summary_path} with {len(overlap_results)} rows.")

    print("\nAnalysis completed.")
    return results


if __name__ == "__main__":
    data = main()
