# Required libraries
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point, LineString, MultiPoint, MultiLineString
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.tri import Triangulation
import matplotlib.colors as mcolors
import networkx as nx
from scipy.spatial import cKDTree
from scipy import ndimage
import warnings
import pandas as pd
from scipy.interpolate import LinearNDInterpolator


# Read a single-surface GOCAD .ts file
def read_gocad_ts(file_path):
    """
    Read a GOCAD .ts file and return vertices and triangles (single surface).
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            print(f"Cannot decode file {file_path}. Try another encoding.")
            return np.array([]), np.array([])

    print(f"Opened file {file_path} successfully. Reading {len(lines)} lines.")
    vrtx_count = 0
    trgl_count = 0
    vertices = []
    triangles = []
    id_map = {}

    for line in lines:
        line = line.strip()
        if line.startswith('PVRTX') or line.startswith('VRTX'):
            parts = line.split()
            if len(parts) >= 5:
                try:
                    idx = int(parts[1])
                    x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                    id_map[idx] = len(vertices)
                    vertices.append([x, y, z])
                    vrtx_count += 1
                except (ValueError, IndexError):
                    print(f"Error reading vertex: {line}")
        elif line.startswith('TRGL'):
            parts = line.split()
            if len(parts) >= 4:
                try:
                    v1, v2, v3 = id_map.get(int(parts[1])), id_map.get(int(parts[2])), id_map.get(int(parts[3]))
                    if None not in (v1, v2, v3):
                        triangles.append([v1, v2, v3])
                        trgl_count += 1
                except (ValueError, IndexError):
                    print(f"Error reading triangle: {line}")

    print(f"Found {vrtx_count} vertices and {trgl_count} triangles in the file.")
    vertices_array = np.array(vertices)
    triangles_array = np.array(triangles) if triangles else np.array([])

    if len(vertices_array) > 0:
        print(f"Vertices array shape: {vertices_array.shape}")
        if len(vertices_array.shape) == 1:
            print("Warning: vertices array is 1D.")
    else:
        print("No vertices found in the file.")

    return vertices_array, triangles_array


def read_gocad_ts_multi(file_path):
    """
    Read a GOCAD .ts file with multiple surfaces and return a dict:
    {surface_name: {'vertices': np.array, 'triangles': np.array}}
    """
    surfaces = {}
    vertices = []
    triangles = []
    id_map = {}
    surface_name = None

    def commit():
        nonlocal vertices, triangles, id_map, surface_name
        if surface_name and vertices:
            surfaces[surface_name] = {
                'vertices': np.array(vertices),
                'triangles': np.array(triangles) if triangles else np.array([])
            }
        vertices = []
        triangles = []
        id_map = {}

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin-1') as f:
            lines = f.readlines()

    for line in lines:
        s = line.strip()
        if s.lower().startswith('gocad tsurf'):
            if vertices:
                commit()
            surface_name = None
        elif s.lower().startswith('name:'):
            surface_name = s.split(':', 1)[1].strip()
        elif s.startswith('PVRTX') or s.startswith('VRTX'):
            parts = s.split()
            if len(parts) >= 5:
                try:
                    idx = int(parts[1])
                    x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                    id_map[idx] = len(vertices)
                    vertices.append([x, y, z])
                except Exception:
                    continue
        elif s.startswith('TRGL'):
            parts = s.split()
            if len(parts) >= 4:
                try:
                    v1, v2, v3 = id_map.get(int(parts[1])), id_map.get(int(parts[2])), id_map.get(int(parts[3]))
                    if None not in (v1, v2, v3):
                        triangles.append([v1, v2, v3])
                except Exception:
                    continue
    if vertices:
        commit()
    return surfaces


def process_gocad_file(working_dir):
    try:
        ts_files = [f for f in os.listdir(working_dir) if f.endswith('.ts')]

        if not ts_files:
            print("No .ts file found in the working folder.")
            return None, None

        ts_file = ts_files[0]
        ts_path = os.path.join(working_dir, ts_file)
        print(f"Reading GOCAD .ts file: {ts_file}")

        vertices, triangles = read_gocad_ts(ts_path)

        if vertices is not None:
            if len(vertices) == 0:
                print("No vertices found in the GOCAD .ts file.")
                return None, None

            if not isinstance(vertices, np.ndarray):
                vertices = np.array(vertices)

            if len(vertices.shape) == 1:
                print("Warning: vertices array is 1D. Trying to reshape...")
                if len(vertices) % 3 == 0:
                    vertices = vertices.reshape(-1, 3)
                    print(f"Reshaped vertices array: {vertices.shape}")
                else:
                    print(f"Could not reshape vertices array. Size not compatible: {len(vertices)}")

            print(f"Read {len(vertices)} vertices and {len(triangles) if triangles is not None else 0} triangles.")
            print(f"Forma dell'array vertices: {vertices.shape}")

        return vertices, triangles

    except Exception as e:
        print(f"Error while reading GOCAD .ts file: {e}")
        return None, None


def read_wells_shapefile(working_dir):
    try:
        shp_files = [f for f in os.listdir(working_dir) if f.endswith('.shp')]
        well_keywords = ['pozz', 'well', 'pozzo', 'pozzi', 'sondaggio', 'sondaggi', 'borehole', 'boreholes']

        wells_shp = None
        for shp_file in shp_files:
            filename_lower = shp_file.lower()
            if any(keyword in filename_lower for keyword in well_keywords):
                wells_path = os.path.join(working_dir, shp_file)
                print(f"Reading well shapefile: {shp_file}")
                wells_shp = gpd.read_file(wells_path)

                if wells_shp.geom_type.isin(['Point', 'MultiPoint']).any():
                    print("Confirmed: shapefile contains points (wells).")
                else:
                    print(f"Warning: shapefile {shp_file} was identified as wells but has no points.")
                    print(f"Geometry types present: {wells_shp.geom_type.unique()}")
                break

        if wells_shp is None and shp_files:
            for shp_file in shp_files:
                try:
                    temp_shp = gpd.read_file(os.path.join(working_dir, shp_file))
                    if temp_shp.geom_type.isin(['Point', 'MultiPoint']).any():
                        wells_path = os.path.join(working_dir, shp_file)
                        print(f"Found shapefile with points: {shp_file}")
                        wells_shp = temp_shp
                        break
                except Exception as e:
                    print(f"Error reading {shp_file}: {e}")

        if wells_shp is None:
            print("No shapefile found for wells.")
            return None

        print(f"Columns available in wells shapefile: {wells_shp.columns.tolist()}")
        print(f"Letto shapefile dei Wells con {len(wells_shp)} punti.")
        return wells_shp

    except Exception as e:
        print(f"Error reading wells shapefile: {e}")
        return None


def read_sections_shapefile(working_dir):
    try:
        shp_files = [f for f in os.listdir(working_dir) if f.endswith('.shp')]
        section_keywords = ['sez', 'section', 'trac', 'sezione', 'sezioni', 'linea', 'linee', 'sismica', 'sismiche', 'line']

        sections_shp = None
        for shp_file in shp_files:
            filename_lower = shp_file.lower()
            if any(keyword in filename_lower for keyword in section_keywords):
                sections_path = os.path.join(working_dir, shp_file)
                print(f"Reading section shapefile: {shp_file}")
                sections_shp = gpd.read_file(sections_path)

                if sections_shp.geom_type.isin(['LineString', 'MultiLineString']).any():
                    print("Confirmed: shapefile contains lines (section traces).")
                else:
                    print(f"Warning: shapefile {shp_file} was identified as sections but has no lines.")
                    print(f"Geometry types present: {sections_shp.geom_type.unique()}")
                break

        if sections_shp is None and len(shp_files) >= 2:
            for shp_file in shp_files:
                try:
                    temp_shp = gpd.read_file(os.path.join(working_dir, shp_file))
                    if temp_shp.geom_type.isin(['LineString', 'MultiLineString']).any():
                        sections_path = os.path.join(working_dir, shp_file)
                        print(f"Found shapefile with lines: {shp_file}")
                        sections_shp = temp_shp
                        break
                except Exception as e:
                    print(f"Error reading {shp_file}: {e}")

        if sections_shp is None:
            print("No shapefile found for sections.")
            return None

        print(f"Columns available in sections shapefile: {sections_shp.columns.tolist()}")
        print(f"Loaded sections shapefile with {len(sections_shp)} features.")
        return sections_shp

    except Exception as e:
        print(f"Error reading sections shapefile: {e}")
        return None


def read_maps_shapefile(working_dir):
    """
    Read the geological-map contact lineset (e.g. Limiti_CartaGeol*.shp), used as the
    "maps" (order_p=3) horizontal control tier and as the explicitly-mapped boundary
    for the boundary-overlap metric.
    """
    try:
        shp_files = [f for f in os.listdir(working_dir) if f.endswith('.shp')]
        map_keywords = ['limiti', 'cartageol', 'carta_geol', 'geolmap', 'geological_map']

        maps_shp = None
        for shp_file in shp_files:
            filename_lower = shp_file.lower()
            if any(keyword in filename_lower for keyword in map_keywords):
                maps_path = os.path.join(working_dir, shp_file)
                print(f"Reading geological map boundaries shapefile: {shp_file}")
                maps_shp = gpd.read_file(maps_path)

                if maps_shp.geom_type.isin(['LineString', 'MultiLineString']).any():
                    print("Confirmed: shapefile contains lines (geological boundaries).")
                else:
                    print(f"Warning: shapefile {shp_file} was identified as map boundaries but has no lines.")
                    print(f"Geometry types present: {maps_shp.geom_type.unique()}")
                break

        if maps_shp is None:
            print("No shapefile found for geological map boundaries.")
            return None

        print(f"Columns available in map boundaries shapefile: {maps_shp.columns.tolist()}")
        print(f"Loaded map boundaries shapefile with {len(maps_shp)} features.")
        return maps_shp

    except Exception as e:
        print(f"Error reading map boundaries shapefile: {e}")
        return None


def read_topo_intersections_shapefile(working_dir):
    """
    Read the topography/horizon intersection lineset (e.g. 3D_Topo_Intersections*.shp),
    used as direct DEM-tied Z checkpoints for vertical confidence and as the
    model-interpolated trace for the boundary-overlap metric.
    """
    try:
        shp_files = [f for f in os.listdir(working_dir) if f.endswith('.shp')]
        topo_keywords = ['topo_intersection', 'topointersection', '_topo_', 'topo_intersections']

        topo_shp = None
        for shp_file in shp_files:
            filename_lower = shp_file.lower()
            if any(keyword in filename_lower for keyword in topo_keywords):
                topo_path = os.path.join(working_dir, shp_file)
                print(f"Reading topo-intersections shapefile: {shp_file}")
                topo_shp = gpd.read_file(topo_path)

                if topo_shp.geom_type.isin(['LineString', 'MultiLineString']).any():
                    print("Confirmed: shapefile contains lines (topo-horizon intersections).")
                else:
                    print(f"Warning: shapefile {shp_file} was identified as topo intersections but has no lines.")
                    print(f"Geometry types present: {topo_shp.geom_type.unique()}")
                break

        if topo_shp is None:
            print("No shapefile found for topo-horizon intersections.")
            return None

        print(f"Columns available in topo-intersections shapefile: {topo_shp.columns.tolist()}")
        print(f"Loaded topo-intersections shapefile with {len(topo_shp)} features.")
        return topo_shp

    except Exception as e:
        print(f"Error reading topo-intersections shapefile: {e}")
        return None


def _surface_short_code(surface_name):
    """
    Reduce a GOCAD surface name (e.g. 'TOP_RTZinf_merged_resampled_original') to a short
    code (e.g. 'RTZINF') for matching against map/topo shapefile attributes.
    """
    s = str(surface_name).upper().strip()
    if s.startswith('TOP_'):
        s = s[4:]
    elif s.startswith('TOP'):
        s = s[3:]
    s = re.split(r'_MERGED', s)[0]
    s = re.sub(r'[^A-Z0-9]', '', s)
    return s


def _normalize_map_code(text):
    """
    Normalize a Limiti_CartaGeol 'Base_di' value (e.g. 'Base RTZ inf', 'BaseLOP') to a
    short code (e.g. 'RTZINF', 'LOP') comparable with `_surface_short_code`.
    """
    s = str(text).upper().strip()
    s = re.sub(r'^BASE\s*', '', s)
    s = re.sub(r'[^A-Z0-9]', '', s)
    return s


def _normalize_horizon_code(text):
    """
    Normalize a 3D_Topo_Intersections 'Horizon' value (e.g. 'TOP_ARV') to a short code
    (e.g. 'ARV') comparable with `_surface_short_code`.
    """
    s = str(text).upper().strip()
    s = re.sub(r'^TOP[_\s]*', '', s)
    s = re.sub(r'[^A-Z0-9]', '', s)
    return s


# Reliability weights derived from the 'Affidabili' field of Limiti_CartaGeol, following
# the evaluation/observation-method quality concept in the ISPRA "Linee Guida" PDF.
AFFIDABILI_WEIGHTS = {
    'osservato': 1.0,
    'dedotto': 0.75,
    'ipotizzato': 0.5,
    'coperto': 0.5,
}
AFFIDABILI_DEFAULT_WEIGHT = 0.75


def compute_reliability_weights(values):
    """
    Map an iterable of 'Affidabili' strings to reliability weights in [0,1], using
    `AFFIDABILI_WEIGHTS` (default `AFFIDABILI_DEFAULT_WEIGHT` for missing/unknown values).
    """
    out = []
    for v in values:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            out.append(AFFIDABILI_DEFAULT_WEIGHT)
            continue
        key = str(v).strip().lower()
        out.append(AFFIDABILI_WEIGHTS.get(key, AFFIDABILI_DEFAULT_WEIGHT))
    return np.array(out, dtype=float)


def select_map_lines_for_surface(maps_gdf, surface_name):
    """
    Return the non-tectonic Limiti_CartaGeol features whose 'Base_di' matches this
    surface's short code (`_surface_short_code`) — i.e. the geological boundary
    explicitly mapped from GIS for this surface.
    """
    if maps_gdf is None or maps_gdf.empty:
        return maps_gdf
    code = _surface_short_code(surface_name)
    if 'Tipo' in maps_gdf.columns:
        tipo = maps_gdf['Tipo'].astype(str).str.lower()
        # "Contatto non tettonico ..." and "Limite stratigrafico ..." are non-tectonic;
        # only "Contatto tettonico ..." (without "non") denotes a tectonic/fault contact.
        is_tectonic = tipo.str.contains('tettonic', na=False) & ~tipo.str.contains('non tettonic', na=False)
        non_tectonic = ~is_tectonic
    else:
        non_tectonic = pd.Series(True, index=maps_gdf.index)
    if 'Base_di' in maps_gdf.columns:
        base_codes = maps_gdf['Base_di'].apply(_normalize_map_code)
        matches = base_codes == code
    else:
        matches = pd.Series(False, index=maps_gdf.index)
    return maps_gdf[non_tectonic & matches]


def select_topo_lines_for_surface(topo_gdf, surface_name):
    """
    Return the 3D_Topo_Intersections features whose 'Horizon' matches this surface's
    short code (`_surface_short_code`) — i.e. the topography/horizon intersection
    trace as computed/interpolated by the modeling software for this surface.
    """
    if topo_gdf is None or topo_gdf.empty:
        return topo_gdf
    code = _surface_short_code(surface_name)
    if 'Horizon' in topo_gdf.columns:
        horizon_codes = topo_gdf['Horizon'].apply(_normalize_horizon_code)
        matches = horizon_codes == code
    else:
        matches = pd.Series(False, index=topo_gdf.index)
    return topo_gdf[matches]


# UtilitÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â  per accuratezza orizzontale su griglia
def get_surface_name(working_dir):
    ts_files = [f for f in os.listdir(working_dir) if f.endswith('.ts')]
    if not ts_files:
        return "surface_1"
    ts_path = os.path.join(working_dir, ts_files[0])
    try:
        with open(ts_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip().lower().startswith('name:'):
                    return line.split(':', 1)[1].strip()
    except Exception:
        pass
    return "surface_1"


def ensure_mapping_file(working_dir, surface_name):
    """
    Ensure a surface->data mapping file (wells/sections/maps).
    If missing, create a template with wells/sections enabled, maps disabled, vertical enabled.
    """
    import pandas as pd
    map_path = os.path.join(working_dir, 'surface_data_mapping.csv')
    if not os.path.exists(map_path):
        df = pd.DataFrame([{
            'surface': surface_name,
            'use_wells': 1,
            'use_sections': 1,
            'use_maps': 0,
            'use_vertical': 1
        }])
        df.to_csv(map_path, index=False)
        print(f"Created mapping file: {map_path}")
    try:
        df = pd.read_csv(map_path)
    except Exception as e:
        print(f"Could not read {map_path}: {e}. Using default settings.")
        return {'use_wells': True, 'use_sections': True, 'use_maps': False}
    row = df[df['surface'] == surface_name]
    if row.empty:
        print(f"No row for surface {surface_name} in mapping. Using defaults.")
        return {'use_wells': True, 'use_sections': True, 'use_maps': False, 'use_vertical': True}
    def bool_val(col):
        return bool(row.iloc[0].get(col, 1))
    return {
        'use_wells': bool_val('use_wells'),
        'use_sections': bool_val('use_sections'),
        'use_maps': bool_val('use_maps'),
        'use_vertical': bool_val('use_vertical')
    }


def ensure_checkpoint_edges_file(working_dir, surface_names, wells_gdf=None, sections_gdf=None):
    """
    Ensure a surface-checkpoint-type edge list.
    Columns: surface, checkpoint_id, type (well|section|map).
    Template: one 'ALL' row for wells and sections per surface.
    """
    path = os.path.join(working_dir, 'surface_checkpoint_edges.csv')
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            pass
    rows = []
    for s in surface_names:
        rows.append({'surface': s, 'checkpoint_id': 'ALL', 'type': 'well'})
        rows.append({'surface': s, 'checkpoint_id': 'ALL', 'type': 'section'})
        rows.append({'surface': s, 'checkpoint_id': 'NONE', 'type': 'map'})
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"Created checkpoint edge list: {path}")
    return df


def filter_checkpoints_by_edges(edges_df, surface, wells_gdf, sections_gdf):
    wells_out = wells_gdf
    sections_out = sections_gdf
    subset = edges_df[edges_df['surface'] == surface] if edges_df is not None else pd.DataFrame()
    if subset.empty:
        return wells_out, sections_out
    def norm_series(s):
        return s.fillna('').astype(str).str.strip().str.lower()
    # normalizza edges
    subset_ids = norm_series(subset['checkpoint_id'])
    subset_types = norm_series(subset['type'])
    # Wells
    if wells_gdf is not None:
        wells_ids = norm_series(wells_gdf['NOME_POZZO']) if 'NOME_POZZO' in wells_gdf.columns else norm_series(wells_gdf.index.to_series())
        wells_sel = subset[(subset_types == 'well')]
        if not wells_sel.empty and 'all' not in subset_ids[wells_sel.index].unique():
            ids = subset_ids[wells_sel.index].tolist()
            wells_out = wells_gdf[wells_ids.isin(ids)]
    # Sections
    if sections_gdf is not None:
        sec_ids = norm_series(sections_gdf['NOME']) if 'NOME' in sections_gdf.columns else norm_series(sections_gdf.index.to_series())
        sec_sel = subset[(subset_types == 'section')]
        if not sec_sel.empty and 'all' not in subset_ids[sec_sel.index].unique():
            ids = subset_ids[sec_sel.index].tolist()
            sections_out = sections_gdf[sec_ids.isin(ids)]
    return wells_out, sections_out
    # Wells
    wells_edges = subset[subset['type'].str.lower() == 'well']
    if wells_gdf is not None and not wells_edges.empty:
        ids = wells_edges['checkpoint_id'].astype(str).tolist()
        if 'ALL' not in ids:
            ids = [i for i in ids if i and i != 'NONE']
            if 'NOME_POZZO' in wells_gdf.columns:
                wells_out = wells_gdf[wells_gdf['NOME_POZZO'].astype(str).isin(ids)]
            else:
                wells_out = wells_gdf[wells_gdf.index.astype(str).isin(ids)]
    # Sections
    sec_edges = subset[subset['type'].str.lower() == 'section']
    if sections_gdf is not None and not sec_edges.empty:
        ids = sec_edges['checkpoint_id'].astype(str).tolist()
        if 'ALL' not in ids:
            ids = [i for i in ids if i and i != 'NONE']
            if 'NOME' in sections_gdf.columns:
                sections_out = sections_gdf[sections_gdf['NOME'].astype(str).isin(ids)]
            else:
                sections_out = sections_gdf[sections_gdf.index.astype(str).isin(ids)]
    return wells_out, sections_out


def build_grid(vertices, spacing=5000, clip_to_hull=True):
    xs, ys = vertices[:, 0], vertices[:, 1]
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    gx = np.arange(min_x, max_x + spacing, spacing)
    gy = np.arange(min_y, max_y + spacing, spacing)
    GX, GY = np.meshgrid(gx, gy)
    grid_points = np.c_[GX.ravel(), GY.ravel()]
    if clip_to_hull:
        try:
            from shapely.geometry import Polygon, Point as ShPoint
            hull = Polygon(np.c_[xs, ys]).convex_hull
            mask = np.array([hull.contains(ShPoint(p[0], p[1])) for p in grid_points])
        except Exception:
            mask = np.ones(len(grid_points), dtype=bool)
            hull = None
    else:
        mask = np.ones(len(grid_points), dtype=bool)
        hull = None
    return GX, GY, grid_points, mask, hull


def nearest_distance(points, targets):
    tree = cKDTree(targets)
    dist, idx = tree.query(points, k=1)
    return dist, idx


def extract_points_from_wells(wells_gdf):
    xs = wells_gdf.geometry.x.values
    ys = wells_gdf.geometry.y.values
    return xs, ys


def sample_lines_gdf(lines_gdf, step=2000):
    pts_x, pts_y = [], []
    for geom in lines_gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        geoms = [geom] if geom.geom_type == 'LineString' else list(geom.geoms)
        for g in geoms:
            num = max(2, int(max(g.length, step) // step))
            for f in np.linspace(0, 1, num):
                p = g.interpolate(f, normalized=True)
                pts_x.append(p.x)
                pts_y.append(p.y)
    return np.array(pts_x), np.array(pts_y)


def compute_order_weight(distances_m, order_p):
    # r in km
    r = distances_m / 1000.0
    ID = 1 / (1 + np.power(r, order_p))
    ID_min = ID.min()
    ID_max = ID.max()
    if ID_max == ID_min:
        return np.ones_like(ID)
    return (ID - ID_min) / (ID_max - ID_min)


def compute_horizontal_weights(grid_points, wells_points=None, sections_points=None,
                               maps_points=None, maps_reliability=None):
    weights_list = []
    wells_w = None
    sections_w = None
    maps_w = None
    if wells_points is not None and wells_points.shape[0] > 0:
        dists, _ = nearest_distance(grid_points, wells_points)
        wells_w = compute_order_weight(dists, order_p=1)
        weights_list.append(wells_w)
    if sections_points is not None and sections_points.shape[0] > 0:
        dists, _ = nearest_distance(grid_points, sections_points)
        sections_w = compute_order_weight(dists, order_p=2)
        weights_list.append(sections_w)
    if maps_points is not None and maps_points.shape[0] > 0:
        dists, idx = nearest_distance(grid_points, maps_points)
        maps_w = compute_order_weight(dists, order_p=3)
        if maps_reliability is not None:
            maps_w = np.clip(maps_w * maps_reliability[idx], 0, 1)
        weights_list.append(maps_w)
    if not weights_list:
        return None, None, None, None
    stack = np.vstack(weights_list)
    combined = stack.mean(axis=0)
    return combined, wells_w, sections_w, maps_w


def generate_accuracy_outputs(vertices, wells_shp, sections_shp, output_dir,
                              use_wells=True, use_sections=True,
                              grid_spacing=5000, line_step=2000, surface_name='surface',
                              xlim=None, ylim=None, crs_proj=None,
                              use_maps=False, maps_shp=None, maps_step=2000):
    """
    Compute horizontal confidence weights (IDW) and distance distributions.
    Saves CSV/PNG/HTML in output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    GX, GY, grid_points, mask, hull = build_grid(vertices, spacing=grid_spacing, clip_to_hull=True)

    wells_points = None
    sections_points = None
    maps_points = None
    maps_reliability = None
    if use_wells and wells_shp is not None and not wells_shp.empty:
        wx, wy = extract_points_from_wells(wells_shp)
        wells_points = np.c_[wx, wy]
    if use_sections and sections_shp is not None and not sections_shp.empty:
        lx, ly = sample_lines_gdf(sections_shp, step=line_step)
        if len(lx) > 0:
            sections_points = np.c_[lx, ly]
    if use_maps and maps_shp is not None and not maps_shp.empty:
        maps_lines = select_map_lines_for_surface(maps_shp, surface_name)
        if maps_lines is not None and not maps_lines.empty:
            mx, my, reliability_vals = [], [], []
            for _, row in maps_lines.iterrows():
                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue
                geoms = [geom] if geom.geom_type == 'LineString' else list(geom.geoms)
                for g in geoms:
                    num = max(2, int(max(g.length, maps_step) // maps_step))
                    for f in np.linspace(0, 1, num):
                        p = g.interpolate(f, normalized=True)
                        mx.append(p.x)
                        my.append(p.y)
                        reliability_vals.append(row.get('Affidabili', None))
            if mx:
                maps_points = np.c_[mx, my]
                maps_reliability = compute_reliability_weights(reliability_vals)

    grid_points_use = grid_points[mask]
    combined, wells_w, sections_w, maps_w = compute_horizontal_weights(
        grid_points_use, wells_points, sections_points, maps_points, maps_reliability
    )

    df = pd.DataFrame({
        'x': grid_points_use[:, 0],
        'y': grid_points_use[:, 1]
    })
    if crs_proj:
        try:
            from pyproj import Transformer
            transformer = Transformer.from_crs(crs_proj, "EPSG:4326", always_xy=True)
            lon_vals, lat_vals = transformer.transform(grid_points_use[:, 0], grid_points_use[:, 1])
            df['lon'] = lon_vals
            df['lat'] = lat_vals
        except Exception as e:
            warnings.warn(f"Could not compute lon/lat: {e}")
    # initialize columns with NaN to keep structure even if empty
    df['dist_wells'] = np.nan
    df['dist_wells_km'] = np.nan
    df['weight_wells'] = np.nan
    df['dist_sections'] = np.nan
    df['dist_sections_km'] = np.nan
    df['weight_sections'] = np.nan
    df['dist_maps'] = np.nan
    df['dist_maps_km'] = np.nan
    df['weight_maps'] = np.nan
    df['weight_combined'] = np.nan
    if wells_points is not None:
        d_w, _ = nearest_distance(grid_points_use, wells_points)
        df['dist_wells'] = d_w
        df['dist_wells_km'] = d_w / 1000.0
        if wells_w is not None:
            df['weight_wells'] = wells_w
    if sections_points is not None:
        d_s, _ = nearest_distance(grid_points_use, sections_points)
        df['dist_sections'] = d_s
        df['dist_sections_km'] = d_s / 1000.0
        if sections_w is not None:
            df['weight_sections'] = sections_w
    if maps_points is not None:
        d_m, _ = nearest_distance(grid_points_use, maps_points)
        df['dist_maps'] = d_m
        df['dist_maps_km'] = d_m / 1000.0
        if maps_w is not None:
            df['weight_maps'] = maps_w
    if combined is not None:
        df['weight_combined'] = combined
    df.to_csv(os.path.join(output_dir, f'horizontal_confidence_grid_{surface_name}.csv'), index=False)

    # Heatmap
    if combined is not None:
        try:
            grid_weights = np.full(GX.shape, np.nan, dtype=float)
            mask2d = mask.reshape(GX.shape)
            grid_weights[mask2d] = combined
            fig_w = plt.figure(figsize=(10, 8))
            plt.pcolormesh(GX, GY, grid_weights, cmap='viridis_r', shading='auto', vmin=0, vmax=1)
            plt.colorbar(label='Confidence weight (0-1)')
            plt.title(f'Horizontal confidence (IDW) - {surface_name}')
            if xlim:
                plt.xlim(xlim)
            if ylim:
                plt.ylim(ylim)
            plt.savefig(os.path.join(output_dir, f'horizontal_confidence_idw_{surface_name}.png'), dpi=300, bbox_inches='tight')
            plt.close(fig_w)
        except Exception as e:
            warnings.warn(f"Could not save weight heatmap: {e}")

        # Sorted confidence plot
        try:
            sorted_idx = np.argsort(combined)[::-1]
            weights_sorted = combined[sorted_idx]
            x_idx = np.arange(1, len(weights_sorted) + 1)
            fig_rank = plt.figure(figsize=(10, 6))
            plt.plot(x_idx, weights_sorted, color='steelblue', linewidth=1.5)
            plt.fill_between(x_idx, weights_sorted, color='steelblue', alpha=0.1)
            plt.xlabel('Grid node (sorted by confidence desc.)')
            plt.ylabel('Confidence weight (IDW avg)')
            plt.title(f'Horizontal confidence ranking - {surface_name}')
            plt.xlim(1, len(weights_sorted))
            plt.ylim(0, max(1.0, np.nanmax(weights_sorted) * 1.05))
            plt.grid(alpha=0.2)
            plt.savefig(os.path.join(output_dir, f'horizontal_confidence_rank_{surface_name}.png'), dpi=300, bbox_inches='tight')
            plt.close(fig_rank)
        except Exception as e:
            warnings.warn(f"Could not save confidence ranking plot: {e}")

    # Istogrammi distanze (km)
    try:
        fig_h = plt.figure(figsize=(8, 6))
        max_km = 0
        if wells_points is not None:
            max_km = max(max_km, df['dist_wells_km'].max())
            bins_w = np.arange(df['dist_wells_km'].min(), df['dist_wells_km'].max() + 0.1, 5)
            plt.hist(df['dist_wells_km'], bins=bins_w, alpha=0.6, label='Wells', histtype='step', linewidth=2)
        if sections_points is not None:
            max_km = max(max_km, df['dist_sections_km'].max())
            bins_s = np.arange(df['dist_sections_km'].min(), df['dist_sections_km'].max() + 0.1, 5)
            plt.hist(df['dist_sections_km'], bins=bins_s, alpha=0.6, label='Sections', histtype='step', linewidth=2)
        if maps_points is not None:
            max_km = max(max_km, df['dist_maps_km'].max())
            bins_m = np.arange(df['dist_maps_km'].min(), df['dist_maps_km'].max() + 0.1, 5)
            plt.hist(df['dist_maps_km'], bins=bins_m, alpha=0.6, label='Maps', histtype='step', linewidth=2)
        plt.xlabel(f'distance from nearest checkpoint (km) - {surface_name}')
        plt.ylabel('occurrences')
        if max_km > 0:
            plt.xlim([0, max_km * 1.05])
        if (wells_points is not None) or (sections_points is not None) or (maps_points is not None):
            plt.legend()
        plt.savefig(os.path.join(output_dir, f'distance_histogram_{surface_name}.png'), dpi=300, bbox_inches='tight')
        plt.close(fig_h)
    except Exception as e:
        warnings.warn(f"Could not save distance histogram: {e}")

    # Interattivo IDW
    if crs_proj:
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            if combined is not None:
                df_plot = df.copy()
                df_plot['weight_plot'] = df_plot['weight_combined'].fillna(0)
                use_geo = ('lon' in df_plot.columns and 'lat' in df_plot.columns and
                           df_plot['lon'].notna().any() and df_plot['lat'].notna().any())

                def add_iso_lines(fig_obj, grid_vals, to_lonlat=False):
                    try:
                        levels = np.arange(0, 1.01, 0.1)
                        contour_grid = np.full(GX.shape, np.nan, dtype=float)
                        contour_grid[mask.reshape(GX.shape)] = grid_vals
                        cs = plt.contour(GX, GY, contour_grid, levels=levels, colors='gray', linewidths=0.8)
                        for coll, lvl in zip(cs.collections, levels):
                            for path in coll.get_paths():
                                verts = path.vertices
                                xs, ys = verts[:, 0], verts[:, 1]
                                if to_lonlat and crs_proj:
                                    try:
                                        from pyproj import Transformer
                                        transformer = Transformer.from_crs(crs_proj, "EPSG:4326", always_xy=True)
                                        xs, ys = transformer.transform(xs, ys)
                                    except Exception:
                                        pass
                                trace_cls = go.Scattermapbox if use_geo else go.Scatter
                                fig_obj.add_trace(trace_cls(
                                    lon=xs if use_geo else None,
                                    lat=ys if use_geo else None,
                                    x=None if use_geo else xs,
                                    y=None if use_geo else ys,
                                    mode='lines',
                                    line=dict(color='gray', width=1),
                                    name=f'iso {lvl:.1f}',
                                    hoverinfo='skip',
                                    showlegend=True,
                                    legendgroup='isolines'
                                ))
                        plt.close()
                    except Exception:
                        plt.close()

                if use_geo:
                    mapbox_token = os.getenv("MAPBOX_TOKEN", None)
                    fig = px.scatter_mapbox(
                        df_plot, lat='lat', lon='lon', color='weight_plot',
                        color_continuous_scale='viridis_r', range_color=[0, 1],
                        title=f'Horizontal confidence IDW {surface_name}',
                        zoom=6, height=750
                    )
                    add_iso_lines(fig, combined, to_lonlat=True)
                    fig.update_layout(
                        mapbox_style="satellite-streets" if mapbox_token else "open-street-map",
                        mapbox_accesstoken=mapbox_token
                    )
                else:
                    fig = px.scatter(df_plot, x='x', y='y', color='weight_plot',
                                     color_continuous_scale='viridis_r', range_color=[0, 1],
                                     title=f'Horizontal confidence IDW {surface_name}', width=900, height=750)
                    fig.update_layout(xaxis_title='X', yaxis_title='Y', dragmode='zoom')
                    add_iso_lines(fig, combined, to_lonlat=False)
                fig.write_html(os.path.join(output_dir, f'interactive_confidence_{surface_name}.html'),
                               config={"scrollZoom": True})
        except Exception as e:
            warnings.warn(f"Could not save interactive map: {e}")

    return {
        'grid_points': grid_points_use,
        'weights': combined,
        'weights_wells': wells_w,
        'weights_sections': sections_w,
        'weights_maps': maps_w,
        'GX': GX,
        'GY': GY,
        'mask': mask,
        'dist_wells': df.get('dist_wells_km') if 'dist_wells_km' in df else None,
        'dist_sections': df.get('dist_sections_km') if 'dist_sections_km' in df else None,
        'dist_maps': df.get('dist_maps_km') if 'dist_maps_km' in df else None
    }


def visualize_data(vertices, triangles, wells_shp, sections_shp, apply_smoothing=False,
                   smoothing_iterations=3, smoothing_factor=0.2, crs='EPSG:6708',
                   output_filename='model_dataset.png', grid_points=None, show_plot=False, surface_name=None,
                   xlim=None, ylim=None, output_dir='output_results'):
    """
    Styled visualization of the GOCAD surface footprint (only extent), wells, and sections
    with presentation-friendly styling. Optimized version using a bounding polygon for better performance.
    """
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation
    from matplotlib.colors import LinearSegmentedColormap
    from shapely.geometry import Point, MultiPoint, LineString, MultiLineString, Polygon
    import numpy as np
    import matplotlib.patheffects as PathEffects
    from matplotlib.lines import Line2D
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    import matplotlib.font_manager as fm
    from datetime import datetime
    import os
    from scipy.spatial import ConvexHull
    import time
    from shapely.geometry import Polygon

    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    grid_label_added = False

    # Set a style; fallback if the seaborn theme is unavailable
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        try:
            plt.style.use('seaborn-whitegrid')
        except OSError:
            pass

    fig = plt.figure(figsize=(14, 12), dpi=100)
    ax_2d = fig.add_subplot(111)

    if vertices is not None and triangles is not None and len(vertices) > 0 and len(triangles) > 0:
        if apply_smoothing:
            original_vertices = vertices.copy()
            vertices = smooth_surface(vertices, triangles,
                                     iterations=smoothing_iterations,
                                     factor=smoothing_factor)
            print(f"Smoothing applied to surfaces (iterations: {smoothing_iterations}, factor: {smoothing_factor})")

    if vertices is not None and len(vertices) > 0:
        if len(vertices.shape) == 2 and vertices.shape[1] >= 2:
            try:
                points_2d = vertices[:, :2]
                min_x, min_y = np.min(points_2d, axis=0)
                max_x, max_y = np.max(points_2d, axis=0)
                rect_coords = [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]
                rect_polygon = Polygon(rect_coords)
                x, y = rect_polygon.exterior.xy
                ax_2d.fill(x, y, color='steelblue', alpha=0.1, label='Model and data footprint')
                ax_2d.plot(x, y, color='steelblue', linewidth=1.5, alpha=0.7)
                print("Surface footprint shown as rectangle (fast mode)")
            except Exception as e:
                print(f"Error drawing optimized footprint: {e}")
                try:
                    points_2d = vertices[:, :2]
                    min_x, min_y = np.min(points_2d, axis=0)
                    max_x, max_y = np.max(points_2d, axis=0)
                    ax_2d.plot([min_x, max_x, max_x, min_x, min_x],
                               [min_y, min_y, max_y, max_y, min_y],
                               color='steelblue', linewidth=1.5, alpha=0.7,
                               label='Surface footprint (box)')
                except Exception as e2:
                    print(f"Fallback to point visualization: {e2}")
                    if len(vertices) > 500:
                        sampling_rate = max(1, len(vertices) // 500)
                        sampled_vertices = vertices[::sampling_rate]
                        ax_2d.scatter(sampled_vertices[:, 0], sampled_vertices[:, 1], s=2, alpha=0.6,
                                      c='steelblue', label='GOCAD vertices (sampled)', edgecolors='none')
                    else:
                        ax_2d.scatter(vertices[:, 0], vertices[:, 1], s=2, alpha=0.6,
                                      c='steelblue', label='GOCAD vertices', edgecolors='none')
        else:
            print(f"Warning: vertices has an invalid shape for visualization: {vertices.shape}")

    if wells_shp is not None and not wells_shp.empty:
        try:
            wells_shp.plot(ax=ax_2d, color='none', markersize=0, label='_nolegend_')
            for idx, row in wells_shp.iterrows():
                if isinstance(row.geometry, (Point, MultiPoint)):
                    x = row.geometry.x if hasattr(row.geometry, 'x') else row.geometry.geoms[0].x
                    y = row.geometry.y if hasattr(row.geometry, 'y') else row.geometry.geoms[0].y
                    ax_2d.scatter(x, y, s=180, color='lightskyblue', alpha=0.3,
                                  edgecolors='none', zorder=10)
                    ax_2d.scatter(x, y, s=100, color='royalblue', alpha=0.5,
                                  edgecolors='none', zorder=11)
                    ax_2d.scatter(x, y, s=40, color='darkblue', alpha=0.9,
                                  edgecolors='white', linewidths=1, zorder=12)

            legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='darkblue',
                                      markersize=10, label='Wells', markeredgecolor='white')]

            name_columns = [col for col in wells_shp.columns if any(
                keyword in col.lower() for keyword in ['name', 'nome', 'id', 'cod', 'ident', 'label', 'num'])]

            if name_columns:
                label_col = name_columns[0]
                for idx, row in wells_shp.iterrows():
                    if isinstance(row.geometry, (Point, MultiPoint)):
                        x = row.geometry.x if hasattr(row.geometry, 'x') else row.geometry.geoms[0].x
                        y = row.geometry.y if hasattr(row.geometry, 'y') else row.geometry.geoms[0].y

                        txt = ax_2d.annotate(str(row[label_col]), xy=(x, y),
                                             xytext=(7, 7), textcoords='offset points',
                                             fontsize=9, fontweight='bold', color='white',
                                             bbox=dict(boxstyle="round,pad=0.3", fc='royalblue', ec="none", alpha=0.7))
                        txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='navy')])
        except Exception as e:
            print(f"Error during advanced well visualization: {e}")
            wells_shp.plot(ax=ax_2d, color='blue', markersize=50, label='Wells')

    if sections_shp is not None and not sections_shp.empty:
        try:
            section_legend = Line2D([0], [0], color='crimson', lw=2, label='Sections')
            if 'legend_elements' in locals():
                legend_elements.append(section_legend)
            else:
                legend_elements = [section_legend]

            for geom in sections_shp.geometry:
                if geom is None or geom.is_empty:
                    continue
                geoms = [geom] if isinstance(geom, LineString) else list(geom.geoms)
                for g in geoms:
                    x, y = g.xy
                    ax_2d.plot(x, y, color='salmon', linewidth=6, alpha=0.3, zorder=7)
                    ax_2d.plot(x, y, color='crimson', linewidth=2.5, alpha=0.9, zorder=8)
                    ax_2d.plot(x, y, color='white', linewidth=1, linestyle=(0, (5, 5)),
                               alpha=0.7, zorder=9)

            name_columns = [col for col in sections_shp.columns if any(
                keyword in col.lower() for keyword in ['name', 'nome', 'id', 'cod', 'ident', 'label', 'num', 'linea', 'line'])]

            if name_columns:
                label_col = name_columns[0]
                for idx, row in sections_shp.iterrows():
                    if isinstance(row.geometry, (LineString, MultiLineString)):
                        if isinstance(row.geometry, LineString):
                            midpoint = row.geometry.interpolate(0.5, normalized=True)
                        else:
                            midpoint = row.geometry.geoms[0].interpolate(0.5, normalized=True)

                        txt = ax_2d.annotate(str(row[label_col]), xy=(midpoint.x, midpoint.y),
                                             xytext=(7, 7), textcoords='offset points',
                                             fontsize=9, fontweight='bold', color='white',
                                             bbox=dict(boxstyle="round,pad=0.3", fc='crimson', ec="none", alpha=0.7))
                        txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='darkred')])
        except Exception as e:
            print(f"Error during advanced section visualization: {e}")
            sections_shp.plot(ax=ax_2d, color='red', linewidth=2, label='Sections')

    stats_info = []

    if triangles is not None and len(triangles) > 0:
        num_triangles = len(triangles)
        num_vertices = len(vertices) if vertices is not None else 0
        stats_info.append(f"Triangles: {num_triangles:,}")

        if vertices is not None and len(vertices.shape) == 2 and vertices.shape[1] >= 3:
            z_min, z_max = np.min(vertices[:, 2]), np.max(vertices[:, 2])
            z_mean = np.mean(vertices[:, 2])
            stats_info.append(f"Min elevation: {z_min:.2f} m")
            stats_info.append(f"Max elevation: {z_max:.2f} m")

    if wells_shp is not None and not wells_shp.empty:
        num_wells = len(wells_shp)
        stats_info.append(f"Wells: {num_wells}")

    if sections_shp is not None and not sections_shp.empty:
        num_sections = len(sections_shp)
        stats_info.append(f"Sections: {num_sections}")

    ax_2d.set_xlabel('X (m)', fontsize=12, fontweight='bold')
    ax_2d.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
    lower_name = str(surface_name).lower() if surface_name else ''
    is_model = lower_name.startswith('model')
    if is_model:
        title_txt = "Model and data footprint"
        stats_title = "MODEL STATS"
    else:
        title_txt = f"Surface footprint {surface_name}" if surface_name else "Surface footprint"
        stats_title = "SURFACE STATS"
    ax_2d.set_title(title_txt, fontsize=16, fontweight='bold', pad=20)

    if stats_info:
        stats_text = '\n'.join(stats_info)
        ax_2d.text(0.98, 0.98, stats_text,
                   transform=ax_2d.transAxes,
                   fontsize=10,
                   verticalalignment='top',
                   horizontalalignment='right',
                   bbox=dict(boxstyle='round,pad=0.7',
                             facecolor='white',
                             edgecolor='lightgray',
                             alpha=0.9))

        ax_2d.text(0.98, 1.02, stats_title,
                   transform=ax_2d.transAxes,
                   fontsize=11,
                   fontweight='bold',
                   verticalalignment='bottom',
                   horizontalalignment='right',
                   bbox=dict(boxstyle='round,pad=0.3',
                             facecolor='royalblue',
                             edgecolor='none',
                             alpha=0.9),
                   color='white')

    x_min, x_max = ax_2d.get_xlim()
    plot_width = x_max - x_min
    scale_options = [10, 50, 100, 500, 1000, 2000, 5000, 10000]
    scale_size = next((x for x in scale_options if x > plot_width / 10), scale_options[-1])

    fontprops = fm.FontProperties(size=9, weight='bold')
    scalebar = AnchoredSizeBar(ax_2d.transData,
                               scale_size,
                               f'{scale_size} m',
                               'lower left',
                               pad=0.5,
                               color='black',
                               frameon=True,
                               size_vertical=1,
                               fontproperties=fontprops,
                               bbox_to_anchor=(0.05, 0.05),
                               bbox_transform=ax_2d.transAxes,
                               sep=5)

    scalebar.patch.set_facecolor('white')
    scalebar.patch.set_alpha(0.8)
    scalebar.patch.set_edgecolor('lightgray')
    ax_2d.add_artist(scalebar)

    if 'legend_elements' in locals() and legend_elements:
        if grid_label_added:
            legend_elements.append(Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
                                          markersize=6, alpha=0.5, label='Griglia valutazione'))
        ax_2d.legend(handles=legend_elements, loc='upper left', frameon=True, framealpha=0.9, edgecolor='lightgray')

    ax_2d.text(0.02, 0.02, f"Coordinate system: {crs}",
               transform=ax_2d.transAxes,
               fontsize=8,
               verticalalignment='bottom',
               horizontalalignment='left',
               bbox=dict(boxstyle='round,pad=0.3',
                         facecolor='white',
                         edgecolor='lightgray',
                         alpha=0.8))

    fig.text(0.99, 0.01, f"{datetime.now().strftime('%d/%m/%Y %H:%M')} - {surface_name or ''}",
             fontsize=7, color='gray', ha='right', va='bottom')

    # Griglia di valutazione
    grid_label_added = False
    if grid_points is not None:
        try:
            ax_2d.scatter(grid_points[:, 0], grid_points[:, 1], s=3, color='gray', alpha=0.3, label='Evaluation grid')
            grid_label_added = True
        except Exception:
            grid_label_added = False

    if xlim:
        ax_2d.set_xlim(xlim)
    if ylim:
        ax_2d.set_ylim(ylim)

    plt.tight_layout()

    execution_time = time.time() - start_time
    print(f"Execution time: {execution_time:.2f} seconds")

    output_path = os.path.join(output_dir, output_filename)
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Figure saved to: {output_path}")

    if show_plot:
        plt.show()
    plt.close(fig)
    return fig


def _extract_point_z(gdf, preferred_names=None):
    """
    Extract (x, y, z) tuples from a GeoDataFrame of points, using geometry Z when present
    or falling back to a numeric attribute with a Z-like name.
    """
    if gdf is None or gdf.empty:
        return []
    preferred_names = preferred_names or ['z', 'Z', 'quota', 'elev', 'elevation', 'depth', 'quota_m', 'depth_m']
    z_col = None
    cols_lower = {c.lower(): c for c in gdf.columns if c != 'geometry'}
    for name in preferred_names:
        if name.lower() in cols_lower:
            col = cols_lower[name.lower()]
            if pd.api.types.is_numeric_dtype(gdf[col]):
                z_col = col
                break
    points = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        x = getattr(geom, 'x', None) if hasattr(geom, 'x') else None
        y = getattr(geom, 'y', None) if hasattr(geom, 'y') else None
        z_val = None
        if hasattr(geom, 'has_z') and geom.has_z:
            try:
                z_val = geom.z
            except Exception:
                z_val = None
        if z_val is None and z_col is not None:
            try:
                z_val = float(row[z_col])
            except Exception:
                z_val = None
        if x is None or y is None or z_val is None or pd.isna(z_val):
            continue
        points.append((x, y, z_val))
    return points


def _load_depth_csv(working_dir, filename, required_cols):
    """
    Load a depth CSV if present; returns DataFrame or None.
    """
    path = os.path.join(working_dir, filename)
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        for col in required_cols:
            if col not in df.columns:
                print(f"{filename} missing column {col}, ignoring file.")
                return None
        return df
    except Exception as e:
        print(f"Could not read {filename}: {e}")
        return None


def _idw_from_points(values, points_xy, grid_points, power=2):
    """
    Compute IDW on given grid_points from sample points.
    values: array of shape (n,)
    points_xy: array (n,2)
    grid_points: array (m,2)
    returns array (m,)
    """
    if len(values) == 0 or len(grid_points) == 0:
        return None
    pts = np.asarray(points_xy)
    vals = np.asarray(values)
    tree = cKDTree(pts)
    k = min(8, len(pts))
    dists, idxs = tree.query(grid_points, k=k)
    # ensure 2D arrays
    dists = np.atleast_2d(dists)
    idxs = np.atleast_2d(idxs)
    dists = np.where(dists == 0, 1e-9, dists)
    weights = 1.0 / np.power(dists, power)
    vals_sel = vals[idxs]
    weighted = vals_sel * weights
    num = np.nansum(weighted, axis=1)
    denom = np.nansum(weights, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        res = np.divide(num, denom)
    return res




def smooth_surface(vertices, triangles, iterations=3, factor=0.2):
    """
    Apply Laplacian smoothing to the surface
    """
    import numpy as np

    smoothed_vertices = vertices.copy()
    neighbors = [[] for _ in range(len(vertices))]

    for tri in triangles:
        for i in range(3):
            v1 = tri[i]
            v2 = tri[(i + 1) % 3]
            v3 = tri[(i + 2) % 3]

            if v2 not in neighbors[v1]:
                neighbors[v1].append(v2)
            if v3 not in neighbors[v1]:
                neighbors[v1].append(v3)

    for _ in range(iterations):
        new_vertices = smoothed_vertices.copy()

        for i in range(len(smoothed_vertices)):
            if not neighbors[i]:
                continue

            neighbor_sum = np.zeros(3)
            for n in neighbors[i]:
                neighbor_sum += smoothed_vertices[n]

            neighbor_avg = neighbor_sum / len(neighbors[i])
            new_vertices[i] = smoothed_vertices[i] + factor * (neighbor_avg - smoothed_vertices[i])

        smoothed_vertices = new_vertices

    return smoothed_vertices



def generate_vertical_outputs(vertices, triangles, wells_shp, sections_shp, grid_points_use,
                              GX, GY, mask, output_dir, surface_name, idw_power=2,
                              well_depths_df=None, section_depths_df=None,
                              well_id_field='NOME_POZZO', section_id_field='NOME',
                              xlim=None, ylim=None, crs_proj=None, topo_shp=None):
    """
    Compute vertical confidence from checkpoints with Z (primarily wells).
    Produces CSV, heatmaps (|dZ| and normalized |dZ|), and HTML scatter.
    """
    os.makedirs(output_dir, exist_ok=True)
    samples = []

    def _collect_from_depth_df(gdf, depth_df, id_field):
        pts = []
        if depth_df is None or depth_df.empty or gdf is None or gdf.empty:
            return pts
        gdf_ids = gdf[id_field] if id_field in gdf.columns else gdf.index.astype(str)
        gdf_lookup = dict(zip(gdf_ids.astype(str), gdf.geometry))
        for _, row in depth_df.iterrows():
            cid = str(row.get('checkpoint_id', '')).strip()
            if cid in gdf_lookup:
                geom = gdf_lookup[cid]
                x = getattr(geom, 'x', None)
                y = getattr(geom, 'y', None)
                zval = row.get('z', None)
                if x is None or y is None or pd.isna(zval):
                    continue
                pts.append((float(x), float(y), float(zval)))
        return pts

    # Wells: prefer depth CSV, else geometry Z/attribute
    wells_pts = []
    if well_depths_df is not None:
        wells_pts.extend(_collect_from_depth_df(wells_shp, well_depths_df[well_depths_df['surface'] == surface_name], well_id_field))
    if not wells_pts:
        wells_pts = _extract_point_z(wells_shp)
    if wells_pts:
        samples.extend(wells_pts)

    # Sections: prefer depth CSV (with x/y if provided), else geometry Z/attribute
    sections_pts = []
    if section_depths_df is not None:
        df_sec = section_depths_df[section_depths_df['surface'] == surface_name]
        for _, row in df_sec.iterrows():
            x = row.get('x', None)
            y = row.get('y', None)
            zval = row.get('z', None)
            if pd.notna(x) and pd.notna(y) and pd.notna(zval):
                sections_pts.append((float(x), float(y), float(zval)))
        if not sections_pts:
            sections_pts.extend(_collect_from_depth_df(sections_shp, df_sec, section_id_field))
    if not sections_pts:
        sections_pts = _extract_point_z(sections_shp)
    if sections_pts:
        samples.extend(sections_pts)

    # Topo-horizon intersections: direct DEM-tied Z checkpoints (geometry Z is authoritative)
    samples_topo = 0
    if topo_shp is not None and not topo_shp.empty:
        topo_lines = select_topo_lines_for_surface(topo_shp, surface_name)
        topo_pts = []
        if topo_lines is not None and not topo_lines.empty:
            for geom in topo_lines.geometry:
                if geom is None or geom.is_empty:
                    continue
                geoms = [geom] if geom.geom_type == 'LineString' else list(geom.geoms)
                for g in geoms:
                    for x, y, *rest in g.coords:
                        if rest:
                            topo_pts.append((x, y, rest[0]))
        if topo_pts:
            samples.extend(topo_pts)
            samples_topo = len(topo_pts)

    if not samples:
        print("No checkpoints with Z found; skipping vertical confidence.")
        return None

    samples = np.array(samples)
    pts_xy = samples[:, :2]
    pts_z = samples[:, 2]

    # Interpolate surface Z
    if vertices is None or len(vertices) == 0:
        print("No vertices available for vertical confidence.")
        return None
    interp = LinearNDInterpolator(vertices[:, :2], vertices[:, 2])
    z_model = interp(pts_xy)
    mask_valid = ~np.isnan(z_model)
    if not mask_valid.any():
        print("Surface interpolation failed for all checkpoints; skipping vertical confidence.")
        return None
    pts_xy = pts_xy[mask_valid]
    pts_z = pts_z[mask_valid]
    z_model = z_model[mask_valid]
    delta_z = pts_z - z_model
    abs_delta = np.abs(delta_z)

    # IDW over grid using absolute residuals
    abs_delta_grid = _idw_from_points(abs_delta, pts_xy, grid_points_use, power=idw_power)
    if abs_delta_grid is None:
        return None
    if len(abs_delta_grid) != len(grid_points_use):
        warnings.warn(f"Length mismatch in vertical IDW for {surface_name}, skipping.")
        return None

    abs_min = np.nanmin(abs_delta_grid)
    abs_max = np.nanmax(abs_delta_grid)
    if np.isfinite(abs_min) and np.isfinite(abs_max) and abs_max != abs_min:
        abs_delta_norm = 1.0 - ((abs_delta_grid - abs_min) / (abs_max - abs_min))
    else:
        abs_delta_norm = np.ones_like(abs_delta_grid)

    df = pd.DataFrame({
        'x': grid_points_use[:, 0],
        'y': grid_points_use[:, 1],
        'abs_delta_z': abs_delta_grid,
        'abs_delta_norm': abs_delta_norm
    })
    if crs_proj:
        try:
            from pyproj import Transformer
            transformer = Transformer.from_crs(crs_proj, "EPSG:4326", always_xy=True)
            lon_vals, lat_vals = transformer.transform(grid_points_use[:, 0], grid_points_use[:, 1])
            df['lon'] = lon_vals
            df['lat'] = lat_vals
        except Exception as e:
            warnings.warn(f"Could not compute lon/lat for vertical grid: {e}")
    df.to_csv(os.path.join(output_dir, f'vertical_confidence_grid_{surface_name}.csv'), index=False)

    # Heatmaps
    def _plot_heat(data, title, fname, cmap='coolwarm', vmin=None, vmax=None, cbar_label='Value'):
        try:
            grid_vals = np.full(GX.shape, np.nan, dtype=float)
            grid_vals[mask.reshape(GX.shape)] = data
            fig = plt.figure(figsize=(10, 8))
            plt.pcolormesh(GX, GY, grid_vals, cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)
            plt.colorbar(label=cbar_label)
            plt.title(title)
            plt.xlabel('X')
            plt.ylabel('Y')
            if xlim:
                plt.xlim(xlim)
            if ylim:
                plt.ylim(ylim)
            plt.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            warnings.warn(f"Could not save vertical heatmap {fname}: {e}")

    _plot_heat(abs_delta_norm, f'Vertical confidence normalized |dZ| - {surface_name}',
               f'vertical_deltaZ_norm_{surface_name}.png', cmap='plasma',
               vmin=0, vmax=1, cbar_label='normalized |dZ| (1=best)')

    # Interactive scatter
    if crs_proj:
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            df_plot = df.copy()
            df_plot['abs_delta_norm'] = df_plot['abs_delta_norm'].fillna(0)
            use_geo = ('lon' in df_plot.columns and 'lat' in df_plot.columns and
                       df_plot['lon'].notna().any() and df_plot['lat'].notna().any())

            def add_iso_lines(fig_obj, grid_vals, to_lonlat=False):
                try:
                    levels = np.arange(0, 1.01, 0.1)
                    contour_grid = np.full(GX.shape, np.nan, dtype=float)
                    contour_grid[mask.reshape(GX.shape)] = grid_vals
                    cs = plt.contour(GX, GY, contour_grid, levels=levels, colors='gray', linewidths=0.8)
                    for coll, lvl in zip(cs.collections, levels):
                        for path in coll.get_paths():
                            verts = path.vertices
                            xs, ys = verts[:, 0], verts[:, 1]
                            if to_lonlat and crs_proj:
                                try:
                                    from pyproj import Transformer
                                    transformer = Transformer.from_crs(crs_proj, "EPSG:4326", always_xy=True)
                                    xs, ys = transformer.transform(xs, ys)
                                except Exception:
                                    pass
                            trace_cls = go.Scattermapbox if use_geo else go.Scatter
                            fig_obj.add_trace(trace_cls(
                                lon=xs if use_geo else None,
                                lat=ys if use_geo else None,
                                x=None if use_geo else xs,
                                y=None if use_geo else ys,
                                mode='lines',
                                line=dict(color='gray', width=1),
                                name=f'iso {lvl:.1f}',
                                hoverinfo='skip',
                                showlegend=True,
                                legendgroup='isolines'
                            ))
                    plt.close()
                except Exception:
                    plt.close()

            if use_geo:
                mapbox_token = os.getenv("MAPBOX_TOKEN", None)
                fig = px.scatter_mapbox(
                    df_plot, lat='lat', lon='lon', color='abs_delta_norm',
                    color_continuous_scale='plasma', range_color=[0, 1],
                    title=f'Vertical confidence normalized |dZ| - {surface_name}',
                    labels={'abs_delta_norm': 'normalized |dZ| (1 = best)'},
                    zoom=6, height=750
                )
                add_iso_lines(fig, abs_delta_norm, to_lonlat=True)
                fig.update_layout(
                    mapbox_style="satellite-streets" if mapbox_token else "open-street-map",
                    mapbox_accesstoken=mapbox_token
                )
            else:
                fig = px.scatter(df_plot, x='x', y='y', color='abs_delta_norm',
                                 color_continuous_scale='plasma', range_color=[0, 1],
                                 title=f'Vertical confidence normalized |dZ| - {surface_name}',
                                 labels={'abs_delta_norm': 'normalized |dZ| (1 = best)'})
                fig.update_layout(xaxis_title='X', yaxis_title='Y', dragmode='zoom')
                add_iso_lines(fig, abs_delta_norm, to_lonlat=False)
            fig.write_html(os.path.join(output_dir, f'vertical_deltaZ_{surface_name}.html'),
                           config={"scrollZoom": True})
        except Exception as e:
            warnings.warn(f"Could not save vertical interactive map: {e}")

    return {
        'abs_delta_grid': abs_delta_grid,
        'abs_delta_norm': abs_delta_norm,
        'samples': len(delta_z),
        'samples_topo': samples_topo,
        'grid_points': grid_points_use,
        'GX': GX,
        'GY': GY,
        'mask': mask
    }


def generate_combined_confidence(acc_outputs, vert_outputs, output_dir, surface_name,
                                 crs_proj=None, alpha=0.7, mode="geometric"):
    """
    Combine horizontal weight (H) and vertical normalized confidence (V) into two indicators:
      - arithmetic_combined = alpha*H + (1-alpha)*V
      - geometric_combined  = H**alpha * V**(1-alpha)
      - min_combined        = min(H, V)
    Only computed when both H and V exist on the same grid.
    """
    if acc_outputs is None or vert_outputs is None:
        return None
    weights = acc_outputs.get('weights')
    vert_norm = vert_outputs.get('abs_delta_norm')
    grid_points = acc_outputs.get('grid_points')
    GX = acc_outputs.get('GX')
    GY = acc_outputs.get('GY')
    mask = acc_outputs.get('mask')
    if weights is None or vert_norm is None or grid_points is None or mask is None or GX is None or GY is None:
        return None
    if len(weights) != len(vert_norm) or len(weights) != len(grid_points):
        warnings.warn(f"Combined confidence skipped for {surface_name}: length mismatch.")
        return None

    weights = np.asarray(weights, dtype=float)
    vert_norm = np.asarray(vert_norm, dtype=float)

    arithmetic_combined = alpha * weights + (1 - alpha) * vert_norm
    geometric_combined = np.power(weights, alpha) * np.power(vert_norm, 1 - alpha)
    min_combined = np.minimum(weights, vert_norm)

    df = pd.DataFrame({
        'x': grid_points[:, 0],
        'y': grid_points[:, 1],
        'horizontal_weight': weights,
        'vertical_norm': vert_norm,
        'arithmetic_combined': arithmetic_combined,
        'geometric_combined': geometric_combined,
        'min_combined': min_combined,
        'has_vertical': True
    })
    if crs_proj:
        try:
            from pyproj import Transformer
            transformer = Transformer.from_crs(crs_proj, "EPSG:4326", always_xy=True)
            lon_vals, lat_vals = transformer.transform(grid_points[:, 0], grid_points[:, 1])
            df['lon'] = lon_vals
            df['lat'] = lat_vals
        except Exception as e:
            warnings.warn(f"Could not compute lon/lat for combined grid: {e}")
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, f'combined_confidence_grid_{surface_name}.csv'), index=False)

    # Choose which layer to visualize (default mode or fallback to geometric)
    mode = (mode or "geometric").lower()
    if mode not in {"geometric", "arithmetic", "min"}:
        mode = "geometric"
    mode_map = {
        "geometric": ("geometric_combined", geometric_combined, "cividis"),
        "arithmetic": ("arithmetic_combined", arithmetic_combined, "viridis"),
        "min": ("min_combined", min_combined, "magma_r")
    }
    col_name, combined_vals, cmap = mode_map[mode]

    # Heatmaps for selected mode (and keep geometric/min PNG for quick compare)
    try:
        def _save_heat(vals, fname, title, cmap_use):
            grid_vals = np.full(GX.shape, np.nan, dtype=float)
            grid_vals[mask.reshape(GX.shape)] = vals
            fig = plt.figure(figsize=(10, 8))
            plt.pcolormesh(GX, GY, grid_vals, cmap=cmap_use, shading='auto', vmin=0, vmax=1)
            plt.colorbar(label='Combined confidence')
            plt.title(title)
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches='tight')
            plt.close(fig)

        _save_heat(combined_vals, f'combined_confidence_{mode}_{surface_name}.png',
                   f'Combined confidence ({mode}) - {surface_name}', cmap)
        # Also save geometric/min for quick compare
        if mode != "geometric":
            _save_heat(geometric_combined, f'combined_confidence_geometric_{surface_name}.png',
                       f'Combined confidence (geometric) - {surface_name}', 'cividis')
        if mode != "min":
            _save_heat(min_combined, f'combined_confidence_min_{surface_name}.png',
                       f'Combined confidence (min) - {surface_name}', 'magma_r')
    except Exception as e:
        warnings.warn(f"Could not save combined heatmap: {e}")

    # Interactive HTML for selected mode (only if CRS provided for lon/lat)
    if crs_proj:
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            df_plot = df.copy()
            df_plot['combined_plot'] = combined_vals
            use_geo = ('lon' in df_plot.columns and 'lat' in df_plot.columns and
                       df_plot['lon'].notna().any() and df_plot['lat'].notna().any())

            def add_iso_lines(fig_obj, grid_vals, to_lonlat=False):
                try:
                    levels = np.arange(0, 1.01, 0.1)
                    contour_grid = np.full(GX.shape, np.nan, dtype=float)
                    contour_grid[mask.reshape(GX.shape)] = grid_vals
                    cs = plt.contour(GX, GY, contour_grid, levels=levels, colors='gray', linewidths=0.8)
                    for coll, lvl in zip(cs.collections, levels):
                        for path in coll.get_paths():
                            verts = path.vertices
                            xs, ys = verts[:, 0], verts[:, 1]
                            if to_lonlat and crs_proj:
                                try:
                                    from pyproj import Transformer
                                    transformer = Transformer.from_crs(crs_proj, "EPSG:4326", always_xy=True)
                                    xs, ys = transformer.transform(xs, ys)
                                except Exception:
                                    pass
                            trace_cls = go.Scattermapbox if use_geo else go.Scatter
                            fig_obj.add_trace(trace_cls(
                                lon=xs if use_geo else None,
                                lat=ys if use_geo else None,
                                x=None if use_geo else xs,
                                y=None if use_geo else ys,
                                mode='lines',
                                line=dict(color='gray', width=1),
                                name=f'iso {lvl:.1f}',
                                hoverinfo='skip',
                                showlegend=True,
                                legendgroup='isolines'
                            ))
                    plt.close()
                except Exception:
                    plt.close()

            if use_geo:
                mapbox_token = os.getenv("MAPBOX_TOKEN", None)
                fig = px.scatter_mapbox(
                    df_plot, lat='lat', lon='lon', color='combined_plot',
                    color_continuous_scale=cmap, range_color=[0, 1],
                    title=f'Combined confidence ({mode}) - {surface_name}',
                    labels={'combined_plot': f'combined ({mode})'},
                    zoom=6, height=750
                )
                add_iso_lines(fig, combined_vals, to_lonlat=True)
                fig.update_layout(
                    mapbox_style="satellite-streets" if mapbox_token else "open-street-map",
                    mapbox_accesstoken=mapbox_token
                )
            else:
                fig = px.scatter(df_plot, x='x', y='y', color='combined_plot',
                                 color_continuous_scale=cmap, range_color=[0, 1],
                                 title=f'Combined confidence ({mode}) - {surface_name}',
                                 labels={'combined_plot': f'combined ({mode})'})
                fig.update_layout(xaxis_title='X', yaxis_title='Y', dragmode='zoom')
                add_iso_lines(fig, combined_vals, to_lonlat=False)
            fig.write_html(os.path.join(output_dir, f'combined_confidence_{mode}_{surface_name}.html'),
                           config={"scrollZoom": True})
        except Exception as e:
            warnings.warn(f"Could not save combined interactive map: {e}")

    return {
        'arithmetic_combined': arithmetic_combined,
        'geometric_combined': geometric_combined,
        'min_combined': min_combined,
        'selected_mode': mode
    }


def _drop_z(geom):
    """Project a (Multi)LineString to 2D, dropping any Z coordinate."""
    if geom is None or not geom.has_z:
        return geom
    from shapely.ops import transform as shp_transform
    return shp_transform(lambda x, y, z=None: (x, y), geom)


def _line_parts_xy(geom):
    """Yield (x, y) coordinate arrays for each part of a (Multi)LineString."""
    if geom is None or geom.is_empty:
        return
    geoms = [geom] if geom.geom_type == 'LineString' else list(geom.geoms)
    for g in geoms:
        x, y = g.xy
        yield x, y


def generate_boundary_overlap_outputs(topo_shp, maps_shp, surface_name, output_dir,
                                      buffer_dist=50.0, xlim=None, ylim=None):
    """
    Compare the model-interpolated topo-horizon intersection trace (3D_Topo_Intersections,
    matched via 'Horizon' -> `select_topo_lines_for_surface`) against the explicitly-mapped
    geological boundary (Limiti_CartaGeol non-tectonic contacts, matched via 'Base_di' ->
    `select_map_lines_for_surface`) for this surface.

    Buffers the mapped boundary by `buffer_dist` meters and computes the % of the
    topo-intersection line length that falls inside that buffer (buffer-based % length
    overlap). Writes `boundary_overlap_<surface_name>.csv`/`.png` to `output_dir`.
    Returns a dict of metrics, or None if either input is missing/empty for this surface.
    """
    from shapely.ops import unary_union

    os.makedirs(output_dir, exist_ok=True)

    topo_lines = select_topo_lines_for_surface(topo_shp, surface_name) if topo_shp is not None else None
    map_lines = select_map_lines_for_surface(maps_shp, surface_name) if maps_shp is not None else None

    if topo_lines is None or topo_lines.empty or map_lines is None or map_lines.empty:
        print(f"Boundary overlap skipped for {surface_name}: missing topo or map lines.")
        return None

    map_geoms = [_drop_z(g) for g in map_lines.geometry if g is not None and not g.is_empty]
    topo_geoms = [_drop_z(g) for g in topo_lines.geometry if g is not None and not g.is_empty]
    if not map_geoms or not topo_geoms:
        print(f"Boundary overlap skipped for {surface_name}: empty geometries.")
        return None

    map_union = unary_union(map_geoms)
    buffer_poly = map_union.buffer(buffer_dist)

    covered_length = 0.0
    total_length = 0.0
    for geom in topo_geoms:
        total_length += geom.length
        covered_length += geom.intersection(buffer_poly).length
    map_length = sum(g.length for g in map_geoms)

    overlap_pct = 100.0 * covered_length / total_length if total_length > 0 else np.nan

    result = {
        'surface': surface_name,
        'topo_length_m': total_length,
        'map_length_m': map_length,
        'covered_length_m': covered_length,
        'overlap_pct': overlap_pct,
        'buffer_dist_m': buffer_dist,
        'n_map_features': len(map_geoms),
        'n_topo_features': len(topo_geoms),
    }
    pd.DataFrame([result]).to_csv(
        os.path.join(output_dir, f'boundary_overlap_{surface_name}.csv'), index=False
    )

    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        buffer_polys = [buffer_poly] if buffer_poly.geom_type == 'Polygon' else list(buffer_poly.geoms)
        for poly in buffer_polys:
            bx, by = poly.exterior.xy
            ax.fill(bx, by, color='orange', alpha=0.2,
                    label=f'Mapped boundary buffer ({buffer_dist:.0f} m)' if poly is buffer_polys[0] else None)
            for interior in poly.interiors:
                ix, iy = interior.xy
                ax.fill(ix, iy, color='white', alpha=1.0)

        for i, geom in enumerate(map_geoms):
            for x, y in _line_parts_xy(geom):
                ax.plot(x, y, color='darkorange', linewidth=2,
                        label='Mapped boundary (GIS, Limiti_CartaGeol)' if i == 0 else None)
        for i, geom in enumerate(topo_geoms):
            for x, y in _line_parts_xy(geom):
                ax.plot(x, y, color='steelblue', linewidth=2,
                        label='Topo-horizon intersection (model, 3D_Topo_Intersections)' if i == 0 else None)

        ax.set_title(f'Boundary overlap - {surface_name} ({overlap_pct:.1f}% within {buffer_dist:.0f} m)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        plt.savefig(os.path.join(output_dir, f'boundary_overlap_{surface_name}.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        warnings.warn(f"Could not save boundary overlap plot: {e}")

    return result
