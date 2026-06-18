"""
custom_validation.py — extended geological validation for the Trentino 7-surface model.

New functions called by both run_accuracy_original.py and run_accuracy_resampled.py:
  - select_map_lines_strat       : corrected stratigraphic contact matching
  - generate_enhanced_boundary_overlap : single-geometry MOVE vs GIS overlap_pct
  - compute_fault_throw_per_horizon    : IDW fault-throw-impact layer per horizon
  - generate_fault_validation_outputs  : dissolved MOVE faults vs dissolved GIS faults
  - generate_fault_throw_comparison    : qualitative modeled vs observed throw table
  - compute_unit_thickness_at_grid     : interpolate PVRTX Thickness onto eval grid
  - compute_acceptance_class           : single-surface acceptance classification
  - generate_acceptance_table          : aggregate table + model-level summary

CRS baseline: EPSG:6707 (RDN2008/UTM32N) for every layer used by this module — callers
must standardize all GeoDataFrames to EPSG:6707 (see `files_utils.standardize_crs`)
before calling functions here. No layer in this dataset is exempt or kept in a
different CRS.
"""
import os
import re
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from shapely.ops import unary_union, linemerge
from scipy.interpolate import LinearNDInterpolator

from files_utils import (
    _surface_short_code,
    _normalize_map_code,
    select_topo_lines_for_surface,
    _drop_z,
    _line_parts_xy,
    _idw_from_points,
    dissolve_lines_by_key,
)

# ---------------------------------------------------------------------------
# Stratigraphic constants
# ---------------------------------------------------------------------------

# Maps normalized Base_di code → surface short code (base of overlying = top of underlying)
# Stratigraphy oldest→youngest: DPR → FMZ → LOP → RTZinf → RTZsup → OSV → ARV → Maiolica
STRAT_MAP = {
    'FMZ':    'DPR',
    'LOP':    'FMZ',
    'RTZINF': 'LOP',
    'RTZSUP': 'RTZINF',
    'OSV':    'RTZSUP',
    'ARV':    'OSV',
    'MAI':    'ARV',
}

STRAT_SUCCESSION = ['DPR', 'FMZ', 'LOP', 'RTZINF', 'RTZSUP', 'OSV', 'ARV']

THROW_CLASSES = [(5, 20), (20, 50), (50, 80), (80, 120), (120, float('inf'))]


def _throw_class_label(throw_m):
    if throw_m is None or np.isnan(throw_m):
        return 'unknown'
    for lo, hi in THROW_CLASSES:
        if lo <= throw_m < hi:
            hi_str = f'{hi:.0f}' if hi < float('inf') else '+'
            return f'{lo:.0f}–{hi_str} m'
    return f'< {THROW_CLASSES[0][0]:.0f} m'


def _normalize_fault_id(text):
    """
    Normalize a fault identifier to a common key comparable between the GIS dissolve
    field ('Nome_fagli', e.g. '8a') and the MOVE topo-intersection 'Name' field
    (e.g. 'F8a') -> both become '8A'.
    """
    s = str(text).strip().upper()
    s = re.sub(r'^F', '', s)
    s = re.sub(r'[^A-Z0-9]', '', s)
    return s


def _fault_sort_key(fault_id):
    """Natural sort key for fault ids like '1', '8A', '9B', '10' -> numeric-then-letter order."""
    m = re.match(r'^(\d+)([A-Z]*)$', fault_id)
    if not m:
        return (999, fault_id)
    return (int(m.group(1)), m.group(2))


# ---------------------------------------------------------------------------
# 1. Corrected stratigraphic map-contact selection
# ---------------------------------------------------------------------------

def select_map_lines_strat(maps_gdf, surface_name, strat_map=None):
    """
    Return non-tectonic Limiti_CartaGeol features whose 'Base_di' corresponds to the
    UNDERLYING surface `surface_name` according to the stratigraphic translation table.

    In the geological map, a contact is labelled as "Base of the OVERLYING unit",
    while the GOCAD model stores it as "Top of the UNDERLYING unit".  STRAT_MAP
    translates: Base_di normalized code → surface short code.

    Returns an empty GeoDataFrame (never None) if no match is found.
    """
    if strat_map is None:
        strat_map = STRAT_MAP

    if maps_gdf is None or maps_gdf.empty:
        return gpd.GeoDataFrame(columns=maps_gdf.columns if maps_gdf is not None else [])

    surface_code = _surface_short_code(surface_name)

    # Find which Base_di normalized code maps to this surface
    matching_base_codes = [base for base, surf in strat_map.items() if surf == surface_code]
    if not matching_base_codes:
        return maps_gdf.iloc[0:0]

    if 'Tipo' in maps_gdf.columns:
        tipo = maps_gdf['Tipo'].astype(str).str.lower()
        is_tectonic = tipo.str.contains('tettonic', na=False) & ~tipo.str.contains('non tettonic', na=False)
        non_tectonic = ~is_tectonic
    else:
        non_tectonic = pd.Series(True, index=maps_gdf.index)

    if 'Base_di' in maps_gdf.columns:
        base_codes = maps_gdf['Base_di'].apply(_normalize_map_code)
        matches = base_codes.isin(matching_base_codes)
    else:
        matches = pd.Series(False, index=maps_gdf.index)

    return maps_gdf[non_tectonic & matches]


# ---------------------------------------------------------------------------
# 2. Enhanced boundary overlap
# ---------------------------------------------------------------------------

def generate_enhanced_boundary_overlap(topo_shp, maps_lines_prefiltered, surface_name,
                                       output_dir, buffer_dist=50.0, xlim=None, ylim=None):
    """
    Single-geometry boundary-overlap metric: dissolves every MOVE topo-intersection segment
    for this formation into one continuous line, and every GIS stratigraphic-contact segment
    (already formation-filtered by `select_map_lines_strat`) into one continuous line, then
    computes a single one-directional overlap ratio:

        overlap_pct = 100 * length(intersection(gis_line.buffer(buffer_dist), move_line))
                          / length(move_line)

    Inputs:
      topo_shp              : full 3D_Topo_Intersections GeoDataFrame (all surfaces)
      maps_lines_prefiltered: already-filtered map contact GeoDataFrame for this surface
                              (from select_map_lines_strat); must be in the same CRS as topo_shp

    Writes boundary_overlap_<surface>.csv and .png to output_dir.
    Returns a dict of metrics, or None if geometry is missing.
    """
    os.makedirs(output_dir, exist_ok=True)

    topo_lines = select_topo_lines_for_surface(topo_shp, surface_name)
    map_lines = maps_lines_prefiltered

    if topo_lines is None or topo_lines.empty or map_lines is None or map_lines.empty:
        print(f"  Enhanced boundary overlap skipped for {surface_name}: missing lines.")
        return None

    map_geoms = [_drop_z(g) for g in map_lines.geometry if g is not None and not g.is_empty]
    topo_geoms = [_drop_z(g) for g in topo_lines.geometry if g is not None and not g.is_empty]
    if not map_geoms or not topo_geoms:
        print(f"  Enhanced boundary overlap skipped for {surface_name}: empty geometries.")
        return None

    move_line = linemerge(unary_union(topo_geoms))
    gis_line = linemerge(unary_union(map_geoms))

    gis_buffer = gis_line.buffer(buffer_dist)
    move_length = move_line.length
    gis_length = gis_line.length
    covered_length = move_line.intersection(gis_buffer).length
    overlap_pct = 100.0 * covered_length / move_length if move_length > 0 else np.nan

    result = {
        'surface': surface_name,
        'move_length_m': move_length,
        'gis_length_m': gis_length,
        'overlap_pct': overlap_pct,
        'buffer_dist_m': buffer_dist,
        'n_move_segments_merged': len(topo_geoms),
        'n_gis_segments_merged': len(map_geoms),
    }
    pd.DataFrame([result]).to_csv(
        os.path.join(output_dir, f'boundary_overlap_{surface_name}.csv'), index=False
    )

    # Plot
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        buffer_polys = [gis_buffer] if gis_buffer.geom_type == 'Polygon' else list(gis_buffer.geoms)
        for poly in buffer_polys:
            bx, by = poly.exterior.xy
            ax.fill(bx, by, color='orange', alpha=0.15)
            for interior in poly.interiors:
                ix, iy = interior.xy
                ax.fill(ix, iy, color='white', alpha=1.0)
        for i, (x, y) in enumerate(_line_parts_xy(gis_line)):
            ax.plot(x, y, color='darkorange', linewidth=2,
                    label='GIS mapped' if i == 0 else None)
        for i, (x, y) in enumerate(_line_parts_xy(move_line)):
            ax.plot(x, y, color='steelblue', linewidth=2,
                    label='MOVE interpolated' if i == 0 else None)
        ax.set_title(f'Boundary overlap — {surface_name}: {overlap_pct:.1f}%'
                     f' (buffer {buffer_dist:.0f} m)')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.legend(fontsize=9)
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        plt.savefig(
            os.path.join(output_dir, f'boundary_overlap_{surface_name}.png'),
            dpi=300, bbox_inches='tight'
        )
        plt.close(fig)
    except Exception as e:
        warnings.warn(f"Could not save enhanced boundary overlap plot for {surface_name}: {e}")

    return result


# ---------------------------------------------------------------------------
# 3. Fault throw impact layer per horizon
# ---------------------------------------------------------------------------

def compute_fault_throw_per_horizon(horizon_vertices, fault_surfaces_dict,
                                    output_dir, surface_name,
                                    grid_points, GX, GY, mask,
                                    offset_m=100, crs_proj=None,
                                    xlim=None, ylim=None):
    """
    For each fault surface, sample horizon Z on both sides of the fault trace and
    compute a vertical throw proxy.  IDW-interpolate these throw values onto the
    horizon evaluation grid.

    horizon_vertices   : np.array (N, 3) — X, Y, Z of the horizon surface
    fault_surfaces_dict: {fault_name: {'vertices': np.array(M, 3), ...}}
    grid_points        : (K, 2) evaluation grid nodes (from generate_accuracy_outputs)
    GX, GY            : 2D meshgrid arrays (from generate_accuracy_outputs)
    mask               : boolean mask (flat) of valid grid nodes

    Returns a dict with 'grid_throw', 'grid_points' and 'output_paths', or None on failure.
    """
    os.makedirs(output_dir, exist_ok=True)

    if horizon_vertices is None or len(horizon_vertices) == 0:
        return None
    if not fault_surfaces_dict:
        return None

    hz_xy = horizon_vertices[:, :2]
    hz_z = horizon_vertices[:, 2]

    try:
        hz_interp = LinearNDInterpolator(hz_xy, hz_z)
    except Exception as e:
        warnings.warn(f"Could not build horizon interpolator for {surface_name}: {e}")
        return None

    throw_pts = []   # (x, y)
    throw_vals = []  # throw proxy in metres

    for fname, fdata in fault_surfaces_dict.items():
        fv = fdata.get('vertices')
        if fv is None or len(fv) < 2:
            continue
        fxy = fv[:, :2]

        # Estimate mean fault strike from the principal axis of fault XY
        try:
            centroid = fxy.mean(axis=0)
            cov = np.cov((fxy - centroid).T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            strike_vec = eigvecs[:, np.argmax(eigvals)]
            # Perpendicular to strike = dip direction
            perp = np.array([-strike_vec[1], strike_vec[0]])
            perp = perp / (np.linalg.norm(perp) + 1e-12)
        except Exception:
            perp = np.array([1.0, 0.0])

        # Subsample fault vertices to avoid redundant IDW points
        step = max(1, len(fxy) // 500)
        sample_pts = fxy[::step]
        hw_pts = sample_pts + offset_m * perp
        fw_pts = sample_pts - offset_m * perp
        z_hw_arr = hz_interp(hw_pts)
        z_fw_arr = hz_interp(fw_pts)
        for pt, z_hw, z_fw in zip(sample_pts, z_hw_arr, z_fw_arr):
            if not (np.isnan(z_hw) or np.isnan(z_fw)):
                throw_pts.append(pt)
                throw_vals.append(abs(float(z_hw) - float(z_fw)))

    if not throw_pts:
        print(f"  Fault throw impact: no valid samples for {surface_name}, skipping.")
        return None

    throw_pts = np.array(throw_pts)
    throw_vals = np.array(throw_vals)

    # grid_points is already the masked subset (grid_points_use from generate_accuracy_outputs)
    grid_points_use = grid_points
    grid_throw_flat = _idw_from_points(throw_vals, throw_pts, grid_points_use, power=2)

    if grid_throw_flat is None:
        return None

    # CSV output
    df = pd.DataFrame({
        'x': grid_points_use[:, 0],
        'y': grid_points_use[:, 1],
        'fault_throw_m': grid_throw_flat,
    })
    if crs_proj:
        try:
            from pyproj import Transformer
            tr = Transformer.from_crs(crs_proj, "EPSG:4326", always_xy=True)
            df['lon'], df['lat'] = tr.transform(df['x'].values, df['y'].values)
        except Exception:
            pass
    csv_path = os.path.join(output_dir, f'fault_throw_impact_{surface_name}.csv')
    df.to_csv(csv_path, index=False)

    # Heatmap PNG
    png_path = os.path.join(output_dir, f'fault_throw_impact_{surface_name}.png')
    try:
        grid_throw_2d = np.full(GX.shape, np.nan)
        mask_2d = mask.reshape(GX.shape)
        grid_throw_2d[mask_2d] = grid_throw_flat
        fig, ax = plt.subplots(figsize=(10, 8))
        vmax = np.nanpercentile(grid_throw_flat, 98) if len(grid_throw_flat) > 0 else 1
        im = ax.pcolormesh(GX, GY, grid_throw_2d, cmap='plasma', shading='auto',
                           vmin=0, vmax=max(vmax, 1))
        plt.colorbar(im, ax=ax, label='Fault throw proxy (m)')
        ax.set_title(f'Fault throw impact — {surface_name}')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        warnings.warn(f"Could not save fault throw heatmap for {surface_name}: {e}")

    # Interactive HTML
    html_path = os.path.join(output_dir, f'fault_throw_impact_{surface_name}.html')
    try:
        import plotly.graph_objects as go
        valid = ~np.isnan(df['fault_throw_m'])
        fig_html = go.Figure(go.Scatter(
            x=df.loc[valid, 'x'], y=df.loc[valid, 'y'],
            mode='markers',
            marker=dict(
                color=df.loc[valid, 'fault_throw_m'],
                colorscale='Plasma', showscale=True,
                colorbar=dict(title='Throw proxy (m)'), size=4
            ),
            text=[f'{v:.1f} m' for v in df.loc[valid, 'fault_throw_m']],
            hoverinfo='text+x+y',
        ))
        fig_html.update_layout(
            title=f'Fault throw impact — {surface_name}',
            xaxis_title='X (m)', yaxis_title='Y (m)',
            yaxis_scaleanchor='x'
        )
        fig_html.write_html(html_path)
    except Exception as e:
        warnings.warn(f"Could not save fault throw HTML for {surface_name}: {e}")

    return {
        'grid_throw': grid_throw_flat,
        'grid_points': grid_points_use,
        'throw_sample_points': throw_pts,
        'throw_sample_values': throw_vals,
        'output_paths': {'csv': csv_path, 'png': png_path, 'html': html_path},
    }


# ---------------------------------------------------------------------------
# 4. Fault validation (model traces vs mapped tectonic contacts)
# ---------------------------------------------------------------------------

def generate_fault_validation_outputs(topo_faults_shp, faglie_gdf,
                                      output_dir, buffer_dist=50.0):
    """
    Per-fault comparison of MOVE fault traces (3D_Topo_Intersections_Faults, dissolved by
    'Name') against GIS-mapped faults (Faglie_carta_geologica_DEF, dissolved by 'Nome_fagli').

    Each side is first collapsed to one geometry per fault entity via `dissolve_lines_by_key`
    (e.g. all 'Nome_fagli' == '8a' rows -> one merged GIS line for fault 8a), normalized so
    MOVE's 'F8a' and GIS's '8a' resolve to the same key. Faults present in only one source are
    skipped (never guessed) and logged. Both inputs must already be in EPSG:6707.

    overlap_pct = 100 * length(intersection(gis_fault.buffer(buffer_dist), move_fault))
                      / length(move_fault)

    Writes:
      fault_validation_aggregate.csv  — single-row model-level summary
      fault_validation_per_fault.csv  — one row per matched fault entity
      fault_validation_map.png        — one line per GIS fault + one aggregated line per
                                         MOVE fault, legend keyed by fault id
    """
    os.makedirs(output_dir, exist_ok=True)

    if topo_faults_shp is None or topo_faults_shp.empty:
        print("  Fault validation skipped: no model fault topo-intersection lines.")
        return None
    if faglie_gdf is None or faglie_gdf.empty:
        print("  Fault validation skipped: no GIS fault lines (Faglie_carta_geologica_DEF).")
        return None

    gis_faults = dissolve_lines_by_key(faglie_gdf, 'Nome_fagli', normalize_fn=_normalize_fault_id)
    move_faults = dissolve_lines_by_key(topo_faults_shp, 'Name', normalize_fn=_normalize_fault_id)

    if not gis_faults or not move_faults:
        print("  Fault validation skipped: dissolve produced no fault geometries.")
        return None

    common_ids = sorted(set(gis_faults) & set(move_faults), key=_fault_sort_key)
    missing_in_gis = sorted(set(move_faults) - set(gis_faults), key=_fault_sort_key)
    missing_in_move = sorted(set(gis_faults) - set(move_faults), key=_fault_sort_key)
    if missing_in_gis:
        print(f"  Faults present in MOVE but not in GIS (skipped): {missing_in_gis}")
    if missing_in_move:
        print(f"  Faults present in GIS but not in MOVE (skipped): {missing_in_move}")

    if not common_ids:
        print("  Fault validation skipped: no fault id matched between MOVE and GIS.")
        return None

    per_fault_rows = []
    for fid in common_ids:
        move_geom = move_faults[fid]
        gis_geom = gis_faults[fid]
        gis_buf = gis_geom.buffer(buffer_dist)
        covered = move_geom.intersection(gis_buf).length
        overlap_pct = 100 * covered / move_geom.length if move_geom.length > 0 else np.nan
        per_fault_rows.append({
            'fault_id': f'Faglia {fid}',
            'move_length_m': move_geom.length,
            'gis_length_m': gis_geom.length,
            'overlap_pct': overlap_pct,
        })

    per_fault_df = pd.DataFrame(per_fault_rows)
    per_path = os.path.join(output_dir, 'fault_validation_per_fault.csv')
    per_fault_df.to_csv(per_path, index=False)
    print(f"  Wrote {per_path}")

    agg_summary = pd.DataFrame([{
        'n_faults_compared': len(common_ids),
        'mean_overlap_pct': per_fault_df['overlap_pct'].mean(),
        'buffer_dist_m': buffer_dist,
    }])
    agg_path = os.path.join(output_dir, 'fault_validation_aggregate.csv')
    agg_summary.to_csv(agg_path, index=False)
    print(f"  Wrote {agg_path}")

    # --- Map plot: one color per fault id, GIS solid / MOVE dashed, one legend entry per fault ---
    map_png = os.path.join(output_dir, 'fault_validation_map.png')
    try:
        fig, ax = plt.subplots(figsize=(12, 10))
        try:
            colors = matplotlib.colormaps['tab20'].resampled(max(len(common_ids), 1))
        except AttributeError:
            colors = cm.get_cmap('tab20', max(len(common_ids), 1))
        for i, fid in enumerate(common_ids):
            color = colors(i)
            for j, (x, y) in enumerate(_line_parts_xy(gis_faults[fid])):
                ax.plot(x, y, color=color, linewidth=2.0, linestyle='-',
                        label=f'Faglia {fid}' if j == 0 else None)
            for x, y in _line_parts_xy(move_faults[fid]):
                ax.plot(x, y, color=color, linewidth=2.0, linestyle='--')
        ax.set_title('Fault validation — GIS mapped (solid) vs MOVE interpolated (dashed)')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.legend(fontsize=7, ncol=2)
        plt.savefig(map_png, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Wrote {map_png}")
    except Exception as e:
        warnings.warn(f"Could not save fault validation map: {e}")

    return {'aggregate': agg_summary, 'per_fault': per_fault_df}


# ---------------------------------------------------------------------------
# 5. Fault throw comparison (qualitative)
# ---------------------------------------------------------------------------

def generate_fault_throw_comparison(fault_surfaces_dict, faglie_gdf, giaciture_gdf,
                                    study_bbox, output_dir):
    """
    Qualitative comparison of modeled fault throw (Z-range proxy per fault surface) vs
    observed throw classification from Faglie_carta_geologica_DEF.shp.

    fault_surfaces_dict: {name: {'vertices': np.array}} for fault surfaces only
    faglie_gdf         : GeoDataFrame, EPSG:6707 (native), Tipo_fagli field
    giaciture_gdf      : GeoDataFrame, EPSG:6707 (native), TIPO field (faglia/s0)
    study_bbox         : (xmin, ymin, xmax, ymax) in EPSG:6707

    Writes fault_throw_comparison.csv to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    xmin, ymin, xmax, ymax = study_bbox

    # Filter faglie and giaciture to study area
    faglie_study = faglie_gdf.cx[xmin:xmax, ymin:ymax] if faglie_gdf is not None else None
    giac_faults = None
    if giaciture_gdf is not None:
        g_study = giaciture_gdf.cx[xmin:xmax, ymin:ymax]
        giac_faults = g_study[g_study.get('TIPO', g_study.get('Tipo', pd.Series())).str.lower() == 'faglia']

    rows = []
    for fname, fdata in fault_surfaces_dict.items():
        fv = fdata.get('vertices')
        if fv is None or len(fv) == 0:
            continue

        # Modeled throw proxy: Z range of fault surface vertices
        z_vals = fv[:, 2]
        modeled_throw = float(np.nanmax(z_vals) - np.nanmin(z_vals))
        modeled_class = _throw_class_label(modeled_throw)

        # Observed throw from Faglie: check if any Faglie feature is spatially near this fault
        observed_class = 'unknown'
        tipo_fagli_val = ''
        if faglie_study is not None and not faglie_study.empty:
            fault_xy = fv[:, :2]
            fault_centroid_x = float(np.mean(fault_xy[:, 0]))
            fault_centroid_y = float(np.mean(fault_xy[:, 1]))
            from shapely.geometry import Point as ShapelyPoint
            fault_pt = ShapelyPoint(fault_centroid_x, fault_centroid_y)
            # Find nearest Faglie feature within 500 m of fault centroid
            dists = faglie_study.geometry.distance(fault_pt)
            if dists.min() < 500:
                nearest_idx = dists.idxmin()
                tf = str(faglie_study.loc[nearest_idx, 'Tipo_fagli']) if 'Tipo_fagli' in faglie_study.columns else ''
                tipo_fagli_val = tf
                if 'non trascurabile' in tf.lower():
                    observed_class = 'significant (>20 m)'
                elif 'trascurabile' in tf.lower():
                    observed_class = 'negligible (<20 m)'

        # Giaciture fault measurements near this fault
        giac_near = 0
        if giac_faults is not None and not giac_faults.empty:
            from shapely.geometry import Point as ShapelyPoint
            fp = ShapelyPoint(float(np.mean(fv[:, 0])), float(np.mean(fv[:, 1])))
            giac_dists = giac_faults.geometry.distance(fp)
            giac_near = int((giac_dists < 300).sum())

        agreement = 'N/A'
        if observed_class != 'unknown':
            if observed_class == 'significant (>20 m)' and modeled_throw > 20:
                agreement = 'consistent'
            elif observed_class == 'negligible (<20 m)' and modeled_throw <= 20:
                agreement = 'consistent'
            else:
                agreement = 'discrepant'

        rows.append({
            'fault_name': fname,
            'modeled_throw_m': round(modeled_throw, 1),
            'modeled_throw_class': modeled_class,
            'observed_throw_class': observed_class,
            'tipo_fagli': tipo_fagli_val,
            'agreement': agreement,
            'n_giaciture_fault_measurements_near': giac_near,
        })

    if not rows:
        print("  Fault throw comparison: no fault surfaces to compare.")
        return None

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, 'fault_throw_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"  Wrote {csv_path}")
    return df


# ---------------------------------------------------------------------------
# 6. Unit thickness interpolation from PVRTX property
# ---------------------------------------------------------------------------

def compute_unit_thickness_at_grid(horizon_vertices, thickness_arr, grid_points):
    """
    Interpolate PVRTX Thickness property onto the evaluation grid using
    LinearNDInterpolator.

    horizon_vertices : np.array (N, 3) — X, Y, Z
    thickness_arr    : np.array (N,) — thickness in metres (NaN = NO_DATA), or None
    grid_points      : np.array (K, 2) — evaluation grid nodes (already masked)

    Returns np.array (K,) of interpolated thickness values (NaN where outside hull
    or where input thickness is unavailable).
    """
    if thickness_arr is None or len(thickness_arr) == 0:
        return np.full(len(grid_points), np.nan)

    xy = horizon_vertices[:, :2]
    t = np.asarray(thickness_arr, dtype=float)
    valid = ~np.isnan(t) & (t > 0)
    if valid.sum() < 4:
        return np.full(len(grid_points), np.nan)

    try:
        interp = LinearNDInterpolator(xy[valid], t[valid])
        return interp(grid_points)
    except Exception as e:
        warnings.warn(f"Thickness interpolation failed: {e}")
        return np.full(len(grid_points), np.nan)


# ---------------------------------------------------------------------------
# 7. Acceptance classification
# ---------------------------------------------------------------------------

def compute_acceptance_class(overlap_pct):
    """
    Classify a single surface's validation result from `overlap_pct` alone (the only
    surviving boundary-overlap metric).

    Classification rules:
      Accepted              : overlap_pct >= 70
      Conditionally accepted: overlap_pct >= 50
      Rejected               : overlap_pct < 50
      No data                : overlap_pct is NaN
    """
    if overlap_pct is None or np.isnan(overlap_pct):
        return 'No data'
    if overlap_pct >= 70:
        return 'Accepted'
    if overlap_pct >= 50:
        return 'Conditionally accepted'
    return 'Rejected'


# ---------------------------------------------------------------------------
# 8. Acceptance table
# ---------------------------------------------------------------------------

def generate_acceptance_table(per_surface_results, output_dir):
    """
    Build and write the per-horizon acceptance table and a global model summary.

    per_surface_results: dict of {surface_name: {...}} built during the per-surface loop.
      Each entry should have:
        'overlap_result'    : dict from generate_enhanced_boundary_overlap (or None)
        'thickness_at_grid' : np.array from compute_unit_thickness_at_grid (or None)
        'vert_outputs'      : dict from generate_vertical_outputs (or None)

    Writes:
      acceptance_table.csv    — per horizon
      validation_summary.csv  — one-row model aggregate
    """
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    for sname, res in per_surface_results.items():
        ovlp = res.get('overlap_result') or {}
        thick = res.get('thickness_at_grid')
        vert = res.get('vert_outputs') or {}

        overlap_pct = ovlp.get('overlap_pct', np.nan)
        mean_thickness = float(np.nanmean(thick)) if thick is not None and len(thick) > 0 else np.nan

        cls = compute_acceptance_class(overlap_pct)

        rows.append({
            'surface': sname,
            'overlap_pct': round(overlap_pct, 1) if not np.isnan(overlap_pct) else np.nan,
            'mean_unit_thickness_m': round(mean_thickness, 1) if not np.isnan(mean_thickness) else np.nan,
            'acceptance_class': cls,
            'n_vertical_checkpoints': vert.get('samples', 0),
        })

    df = pd.DataFrame(rows)
    tbl_path = os.path.join(output_dir, 'acceptance_table.csv')
    df.to_csv(tbl_path, index=False)
    print(f"  Wrote {tbl_path}")

    # Global summary
    n_total = len(df)
    counts = df['acceptance_class'].value_counts().to_dict()
    valid_overlaps = df['overlap_pct'].dropna()
    summary = {
        'n_surfaces': n_total,
        'n_accepted': counts.get('Accepted', 0),
        'n_conditionally_accepted': counts.get('Conditionally accepted', 0),
        'n_rejected': counts.get('Rejected', 0),
        'n_no_data': counts.get('No data', 0),
        'mean_overlap_pct': round(valid_overlaps.mean(), 1) if len(valid_overlaps) > 0 else np.nan,
        'min_overlap_pct': round(valid_overlaps.min(), 1) if len(valid_overlaps) > 0 else np.nan,
    }
    summary_path = os.path.join(output_dir, 'validation_summary.csv')
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    print(f"  Wrote {summary_path}")

    return df
