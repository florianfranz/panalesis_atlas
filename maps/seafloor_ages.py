import os
from osgeo import gdal
from rasterio.merge import merge
import re
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, LineString, MultiLineString, Polygon, MultiPolygon, LinearRing, GeometryCollection, box
import numpy as np
from scipy.spatial import cKDTree
import rasterio
from rasterio.mask import mask
from rasterio.coords import BoundingBox
from rasterio.transform import from_origin
from shapely.geometry import mapping
from rasterio.warp import calculate_default_transform, reproject, Resampling
from scipy.interpolate import LinearNDInterpolator
from rasterio.transform import from_bounds
from pathlib import Path
import json

project_root = Path(__file__).resolve().parent.parent
config_path = project_root / "config.json"
with open(config_path, "r") as f:
    config = json.load(f)
input_data_folder = config["input_data_folder"]
base_output_folder_path = config["output_folder_path"]
pyproj_data_path = config.get("pyproj_data_path")
if pyproj_data_path:
    os.environ['PROJ_DATA'] = pyproj_data_path

PM_path = os.path.join(input_data_folder, "R_psAbs_All.shp")
PP_path = os.path.join(input_data_folder, "Plate_All.shp")
COB_path = os.path.join(input_data_folder, "Cobgon_All_fixed.shp")
output_folder_path = os.path.join(base_output_folder_path,"Seafloor_ages")
def subset_pm(gdf, appearance, pp_geom):
    filtered_gdf = gdf[
        (gdf["AGE"] < 999) &
        (gdf["TYPE"].isin(["Isochron", "Ridge", "Passive_Margin"])) &
        (gdf["APPEARANCE"] == appearance) &
        (gdf.intersects(pp_geom))
        ]
    return filtered_gdf

def save_gdf(plate, appearance, gdf, geom_type):
    safe_plate_name = plate.replace(" ", "_")
    filename = f"{geom_type}_{safe_plate_name}_{int(appearance)}.geojson"
    filepath = os.path.join(output_folder_path, filename)
    gdf.to_file(filepath, driver="GeoJSON")
    print(f"Saved {len(gdf)} features to {filepath}")

def save_plate_geom(plate_name, appearance_value, geom, crs):
    safe_plate_name = plate_name.replace(" ", "_")
    filename = f"plate_{safe_plate_name}_{int(appearance_value)}.geojson"
    filepath = os.path.join(output_folder_path, filename)
    plate_gdf = gpd.GeoDataFrame({"PLATE": [plate_name], "APPEARANCE": [appearance_value]}, geometry=[geom], crs=crs)
    plate_gdf.to_file(filepath, driver="GeoJSON")
    print(f"Saved plate geometry to {filepath}")

def extract_vertices_within_geom(gdf, geom):
    rows = []
    for idx, row in gdf.iterrows():
        line_geom = row.geometry

        if line_geom.geom_type == 'LineString':
            coords = line_geom.coords
        elif line_geom.geom_type == 'MultiLineString':
            coords = []
            for linestring in line_geom.geoms:
                coords.extend(linestring.coords)
        else:
            continue

        for coord in coords:
            pt = Point(coord)
            if geom.contains(pt):
                row_data = row.drop(labels='geometry').to_dict()
                row_data["geometry"] = pt
                rows.append(row_data)

    if not rows:
        return gpd.GeoDataFrame(geometry=[], crs=gdf.crs)

    points_gdf = gpd.GeoDataFrame(rows, crs=gdf.crs)
    return points_gdf


def extract_polygon_vertices(geom, properties, crs):
    """
    Extracts all vertices from a Polygon or MultiPolygon geometry
    and returns them as a GeoDataFrame of Points with copied properties.
    """
    if geom.is_empty:
        return gpd.GeoDataFrame(columns=list(properties.keys()) + ['geometry'], crs=crs)

    rows = []

    if geom.geom_type == "Polygon":
        rings = [geom.exterior] + list(geom.interiors)
    elif geom.geom_type == "MultiPolygon":
        rings = []
        for poly in geom.geoms:
            rings.extend([poly.exterior] + list(poly.interiors))
    else:
        raise TypeError(f"Unsupported geometry type: {geom.geom_type}")

    for ring in rings:
        for coord in ring.coords:
            pt = Point(coord)
            row_data = properties.copy()
            row_data["geometry"] = pt
            rows.append(row_data)

    polygon_vertices = gpd.GeoDataFrame(rows, crs=crs)

    return polygon_vertices


def get_vertices_from_polygon(polygon):
    """
    Returns a list of shapely Points from the exterior and interior rings of a Polygon.
    """
    pts = [Point(coord) for coord in polygon.exterior.coords]
    for ring in polygon.interiors:
        pts.extend(Point(coord) for coord in ring.coords)
    return pts


def densify_geometry(geometry, distance):
    """Densify any line or polygon geometry by inserting points every `distance` units."""
    if geometry.is_empty:
        return geometry

    if geometry.geom_type == 'LineString':
        return densify_linestring(geometry, distance)

    elif geometry.geom_type == 'MultiLineString':
        return MultiLineString([densify_linestring(ls, distance) for ls in geometry.geoms])

    elif geometry.geom_type == 'Polygon':
        return densify_polygon(geometry, distance)

    elif geometry.geom_type == 'MultiPolygon':
        return MultiPolygon([densify_polygon(p, distance) for p in geometry.geoms])

    else:
        return geometry  # Return unchanged if geometry is unsupported

def densify_linestring(line, distance):
    if not isinstance(line, LineString):
        return line
    length = line.length
    num_points = int(length // distance)
    if num_points <= 1:
        return line
    points = [line.interpolate(distance * i) for i in range(num_points + 1)]
    return LineString(points)

def densify_ring(ring, distance):
    line = LineString(ring)
    densified = densify_linestring(line, distance)
    return LinearRing(densified.coords)

def densify_polygon(polygon, distance):
    exterior = densify_ring(polygon.exterior, distance)
    interiors = [densify_ring(r, distance) for r in polygon.interiors]
    return Polygon(exterior, interiors)

def extract_name_parts(filename):
    parts = filename.replace(".geojson", "").split("_")
    plate = "_".join(parts[2:-1])
    appearance = int(parts[-1])
    return plate, appearance


age_list = []
densify_distance = 1

pm_gdf = gpd.read_file(PM_path)
pp_gdf = gpd.read_file(PP_path)
filtered_pp_gdf = pp_gdf[pp_gdf["PLATE"] != "GAP"]
grouped = filtered_pp_gdf.groupby("APPEARANCE")

COB_gdf = gpd.read_file(COB_path)
geom_types = COB_gdf.geometry.type.unique()
print("Geometry types:", geom_types)

# Step 1: Combine all geometries into one MultiPolygon
cob_geometry = COB_gdf.geometry.union_all()

# Normalize to a list of Polygon geometries
if isinstance(cob_geometry, MultiPolygon):
    geoms = [mapping(geom) for geom in cob_geometry.geoms]
elif isinstance(cob_geometry, Polygon):
    geoms = [mapping(cob_geometry)]
elif isinstance(cob_geometry, GeometryCollection):
    geoms = [mapping(geom) for geom in cob_geometry.geoms if isinstance(geom, (Polygon, MultiPolygon))]
else:
    raise ValueError(f"Unsupported geometry type: {type(cob_geometry)}")

# First part: get all isochron, ridge and passive margin points inside each plate. Save individual plates, plate vertices, and  PM points as GeoJSON.
for appearance_value, group in grouped:
    print(f"\nProcessing group with APPEARANCE = {appearance_value}")
    for index, row in group.iterrows():
        properties = row.drop(labels='geometry').to_dict()
        plate_name = row["PLATE"]
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        geom = row.geometry.buffer(0.05)
        geom = densify_geometry(geom,densify_distance)
        vertices_gdf = extract_polygon_vertices(geom, properties,filtered_pp_gdf.crs)
        filtered_subset = subset_pm(pm_gdf,appearance_value,geom).copy()
        if not filtered_subset.empty:
            filtered_subset["FEAT_AGE"] = filtered_subset["AGE"] - filtered_subset["APPEARANCE"]
            filtered_subset["FEAT_AGE"] = filtered_subset["FEAT_AGE"].where(filtered_subset["FEAT_AGE"] >= 0, 0)
            filtered_subset["geometry"] = filtered_subset.geometry.apply(
                lambda geom: densify_geometry(geom, densify_distance)
            )
            points_gdf = extract_vertices_within_geom(filtered_subset, geom)
            if not points_gdf.empty:
                save_gdf(plate_name, appearance_value, points_gdf, "points")
        else:
            print(f"Skipping feature {index} ({plate_name}) due to no matching features.")
        save_gdf(plate_name, appearance_value, vertices_gdf, "plate_vertices")
        save_plate_geom(plate_name, appearance_value, geom, filtered_pp_gdf.crs)
        print(f" Feature {index} ({plate_name}) has {len(filtered_subset)} matching features.")


# Second part: update plate vertices with nearest neighbour FEAT_AGE from the plate model points layer, merge plate model points and plate vertices layers into a single one.
files = os.listdir(output_folder_path)
plate_vertex_files = [f for f in files if f.startswith("plate_vertices") and f.endswith(".geojson")]
point_files = [f for f in files if f.startswith("points") and f.endswith(".geojson")]



for vertex_file in plate_vertex_files:
    plate_name, appearance = extract_name_parts(vertex_file)
    matching_point_file = f"points_{plate_name}_{appearance}.geojson"
    vertex_path = os.path.join(output_folder_path, vertex_file)
    point_path = os.path.join(output_folder_path, matching_point_file)

    if not os.path.exists(point_path):
        print(f"Skipping {plate_name} (appearance {appearance}) – no matching points file.")
        continue

    try:
        pv = gpd.read_file(vertex_path)
        pts = gpd.read_file(point_path)

        pv_proj = pv.to_crs("EPSG:3395")
        pts_proj = pts.to_crs("EPSG:3395")
        pv_proj = pv_proj[pv_proj.geometry.notnull() & pv_proj.geometry.is_valid]
        pts_proj = pts_proj[pts_proj.geometry.notnull() & pts_proj.geometry.is_valid]

        pv_coords = np.array([(geom.x, geom.y) for geom in pv_proj.geometry if np.isfinite(geom.x) and np.isfinite(geom.y)])
        pts_coords = np.array([(geom.x, geom.y) for geom in pts_proj.geometry if np.isfinite(geom.x) and np.isfinite(geom.y)])

        if len(pv_coords) == 0 or len(pts_coords) == 0:
            print(f"Skipping {plate_name} (appearance {appearance}) – no valid geometries.")
            continue

        tree = cKDTree(pts_coords)
        distances, indices = tree.query(pv_coords, k=1)
        pv_proj = pv_proj.reset_index(drop=True)

        pv_proj["FEAT_AGE"] = pts_proj.iloc[indices]["FEAT_AGE"].values
        pv_proj["TYPE"] = "PL_VERTEX"

        pv_updated = pv_proj.to_crs(pv.crs)
        pv_updated.to_file(vertex_path, driver="GeoJSON")
        print(f"Updated FEAT_AGE for: {vertex_file}")

        pts_reproj = pts.to_crs(pv.crs)
        all_cols = sorted(set(pv_updated.columns).union(set(pts_reproj.columns)))
        pv_standard = pv_updated.reindex(columns=all_cols, fill_value=None)
        pts_standard = pts_reproj.reindex(columns=all_cols, fill_value=None)

        merged = gpd.GeoDataFrame(pd.concat([pv_standard, pts_standard], ignore_index=True), crs=pv.crs)
        merged_filename = f"merged_{plate_name}_{appearance}.geojson"
        merged_path = os.path.join(output_folder_path, merged_filename)
        merged.to_file(merged_path, driver="GeoJSON")
        print(f"Saved merged file: {merged_filename}")
    except Exception as e:
        print(f"Error processing {vertex_file}: {e}")



#Third part: Interpolate + Clip
files = os.listdir(output_folder_path)
merged_points_files = [f for f in files if f.startswith("merged") and f.endswith(".geojson")]
print(len(merged_points_files))
print(merged_points_files)

res = 0.1

for fname in merged_points_files:
    path = os.path.join(output_folder_path, fname)
    gdf = gpd.read_file(path)
    unfilled_raster_path = os.path.join(output_folder_path, fname.replace(".geojson", "_unfilled.tif"))
    raster_path = os.path.join(output_folder_path, fname.replace(".geojson", ".tif"))
    minx, miny, maxx, maxy = gdf.total_bounds
    width = int((maxx - minx) / res)
    height = int((maxy - miny) / res)
    match = re.search(r'merged_(.+)_(\d+)\.geojson', fname)
    if not match:
        print(f"Filename pattern not matched: {fname}")
        continue
    appearance_value = int(match.group(2))
    COB_filtered = COB_gdf[COB_gdf["APPEARANCE"] == appearance_value]

    if COB_filtered.empty:
        print(f"No COB polygons with APPEARANCE = {appearance_value}. Skipping masking.")
        continue

    cob_geometry = COB_filtered.geometry.union_all()
    if isinstance(cob_geometry, MultiPolygon):
        geoms = [mapping(geom) for geom in cob_geometry.geoms]
    elif isinstance(cob_geometry, Polygon):
        geoms = [mapping(cob_geometry)]
    elif isinstance(cob_geometry, GeometryCollection):
        geoms = [mapping(geom) for geom in cob_geometry.geoms if isinstance(geom, (Polygon, MultiPolygon))]
    else:
        raise ValueError(f"Unsupported geometry type for APPEARANCE={appearance_value}: {type(cob_geometry)}")

    # ── SciPy TIN interpolation (replaces gdal.Grid) ─────────────────────────
    points = np.column_stack([gdf.geometry.x, gdf.geometry.y])
    values = gdf["FEAT_AGE"].values.astype(float)

    interp = LinearNDInterpolator(points, values)

    xs = np.linspace(minx, maxx, width)
    ys = np.linspace(maxy, miny, height)  # top → bottom
    grid_x, grid_y = np.meshgrid(xs, ys)
    grid_z = interp(grid_x, grid_y).astype(np.float32)
    grid_z[np.isnan(grid_z)] = -9999       # match your existing nodata value

    transform = from_bounds(minx, miny, maxx, maxy, width, height)
    crs = gdf.crs

    with rasterio.open(
        unfilled_raster_path, "w",
        driver="GTiff", dtype="float32",
        width=width, height=height, count=1,
        crs=crs, transform=transform, nodata=-9999,
    ) as dst:
        dst.write(grid_z, 1)
    # ─────────────────────────────────────────────────────────────────────────

    gdal.Translate(raster_path, unfilled_raster_path)
    ds = gdal.Open(raster_path, gdal.GA_Update)
    band = ds.GetRasterBand(1)
    gdal.FillNodata(targetBand=band, maskBand=None, maxSearchDist=150, smoothingIterations=1)
    ds = None
    print(f"Raster saved: {raster_path}")

    with rasterio.open(raster_path) as src:
        nodata_value = src.nodata if src.nodata is not None else -9999

        print("Raster bounds:", src.bounds)
        print("COB geometry bounds:", cob_geometry.bounds)
        raster_box = box(*src.bounds)
        if not cob_geometry.intersects(raster_box):
            print("No intersection between COB and raster. Skipping.")
            continue
        out_image, out_transform = mask(
            src,
            geoms,
            invert=True,
            crop=False,
            nodata=nodata_value
        )
        out_meta = src.meta.copy()
        out_meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "nodata": nodata_value
        })

    masked_raster_path = raster_path.replace(".tif", "_masked.tif")
    with rasterio.open(masked_raster_path, "w", **out_meta) as dest:
        dest.write(out_image)

    print(f"Inverted clipped raster saved: {masked_raster_path}")
    match = re.search(r'merged_(.+)_(\d+)\.geojson', fname)
    if not match:
        print(f"Filename pattern not matched for plate lookup: {fname}")
        continue

    plate_name = match.group(1)
    appearance_value = int(match.group(2))
    plate_filename = f"plate_{plate_name}_{appearance_value}.geojson"
    plate_path = os.path.join(output_folder_path, plate_filename)

    if not os.path.exists(plate_path):
        print(f" Plate file not found: {plate_filename}")
        continue
    plate_gdf = gpd.read_file(plate_path)
    plate_geom = plate_gdf.geometry.union_all().buffer(-0.05)  # can be Polygon or MultiPolygon
    if isinstance(plate_geom, MultiPolygon):
        plate_geoms = [mapping(geom) for geom in plate_geom.geoms]
    else:
        plate_geoms = [mapping(plate_geom)]

    with rasterio.open(masked_raster_path) as src:
        final_image, final_transform = mask(
            src,
            plate_geoms,
            invert=False,
            crop=False
        )
        final_meta = src.meta.copy()
        final_meta.update({
            "height": final_image.shape[1],
            "width": final_image.shape[2],
            "transform": final_transform
        })
    final_raster_path = masked_raster_path.replace("_masked.tif", "_final.tif")
    with rasterio.open(final_raster_path, "w", **final_meta) as dest:
        dest.write(final_image)
    print(f"Final plate-clipped raster saved: {final_raster_path}")

final_output_dir = os.path.join(output_folder_path, "final_outputs")
os.makedirs(final_output_dir, exist_ok=True)
final_rasters = [f for f in os.listdir(output_folder_path) if f.endswith("_final.tif")]
appearance_groups = {}
filled_paths = []
for f in final_rasters:
    match = re.search(r'_([0-9]+)_final\.tif$', f)
    if not match:
        print(f" Skipping file with unexpected name: {f}")
        continue
    appearance_value = int(match.group(1))
    appearance_groups.setdefault(appearance_value, []).append(os.path.join(output_folder_path, f))

global_bounds = BoundingBox(left=-180, bottom=-90, right=180, top=90)
res = 0.1
nodata_value = -9999

global_width = int((global_bounds.right - global_bounds.left) / res)
global_height = int((global_bounds.top - global_bounds.bottom) / res)
global_transform = from_origin(global_bounds.left, global_bounds.top, res, res)

for appearance_value, raster_paths in appearance_groups.items():
    print(f" Mosaicking {len(raster_paths)} rasters for APPEARANCE = {appearance_value}...")
    src_files_to_mosaic = [rasterio.open(p) for p in raster_paths]
    mosaic, _ = merge(src_files_to_mosaic)
    out_meta = src_files_to_mosaic[0].meta.copy()

    bounds_list = [src.bounds for src in src_files_to_mosaic]
    min_left = min(b.left for b in bounds_list)
    min_bottom = min(b.bottom for b in bounds_list)
    max_right = max(b.right for b in bounds_list)
    max_top = max(b.top for b in bounds_list)
    merged_bounds = BoundingBox(left=min_left, bottom=min_bottom, right=max_right, top=max_top)
    row_offset = int((global_bounds.top - merged_bounds.top) / res)
    col_offset = int((merged_bounds.left - global_bounds.left) / res)
    global_array = np.full((1, global_height, global_width), nodata_value, dtype=mosaic.dtype)
    global_array[:, row_offset:row_offset + mosaic.shape[1], col_offset:col_offset + mosaic.shape[2]] = mosaic

    out_meta.update({
        "height": global_height,
        "width": global_width,
        "transform": global_transform,
        "nodata": nodata_value
    })

    mosaic_path = os.path.join(output_folder_path, f"seafloor_ages_{appearance_value}.tif")
    filled_path = os.path.join(final_output_dir, f"final_seafloor_ages_{appearance_value}.tif")

    with rasterio.open(mosaic_path, "w", **out_meta) as dest:
        dest.write(global_array)


    gdal.Translate(filled_path, mosaic_path)
    ds = gdal.Open(filled_path, gdal.GA_Update)
    band = ds.GetRasterBand(1)
    gdal.FillNodata(targetBand=band, maskBand=None, maxSearchDist=100, smoothingIterations=2)
    ds = None

    print(f" Filled raster saved: {filled_path}")
    filled_paths.append(filled_path)

pattern = r"final_seafloor_ages_(.+?)\.tif"

for filled_raster in filled_paths:
    filename = os.path.basename(filled_raster)
    match = re.search(pattern, filename)
    if match:
        appearance_value = int(match.group(1))
        print(f"Extracted appearance_value: {appearance_value}")
    else:
        print("Pattern not found in:", filename)
    COB_filtered = COB_gdf[COB_gdf["APPEARANCE"] == appearance_value]
    if COB_filtered.empty:
        print(f"No COB polygons with APPEARANCE = {appearance_value}. Skipping masking.")
        continue
    cob_geometry = COB_filtered.geometry.union_all()
    if isinstance(cob_geometry, MultiPolygon):
        geoms = [mapping(geom) for geom in cob_geometry.geoms]
    elif isinstance(cob_geometry, Polygon):
        geoms = [mapping(cob_geometry)]
    elif isinstance(cob_geometry, GeometryCollection):
        geoms = [mapping(geom) for geom in cob_geometry.geoms if isinstance(geom, (Polygon, MultiPolygon))]
    else:
        raise ValueError(f"Unsupported geometry type: {type(cob_geometry)}")
    cob_masked_raster = filled_raster.replace(".tif","_cob_masked.tif")
    with rasterio.open(filled_raster) as src:
        nodata_value = src.nodata if src.nodata is not None else -9999

        print("Raster bounds:", src.bounds, "Normally should be -180,-90,180,90")
        print("COB geometry bounds:", cob_geometry.bounds)
        raster_box = box(*src.bounds)
        if not cob_geometry.intersects(raster_box):
            print("No intersection between COB and raster. Skipping.")
            continue
        out_image, out_transform = mask(
            src,
            geoms,
            invert=True,
            crop=False,
            nodata=nodata_value
        )
        out_meta = src.meta.copy()
        out_meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "nodata": nodata_value
        })
    with rasterio.open(cob_masked_raster, "w", **out_meta) as dest:
        dest.write(out_image)

    dst_crs = "ESRI:54034"
    reprojected_raster = cob_masked_raster.replace("_cob_masked.tif", "_cob_masked_proj.tif")

    with rasterio.open(cob_masked_raster) as src:
        # Calculate transform and dimensions for 10km resolution
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds, resolution=(10000, 10000)
        )

        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height,
            'nodata': src.nodata
        })

        with rasterio.open(reprojected_raster, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest
                )

