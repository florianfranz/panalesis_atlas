import os
import re
import numpy as np
import rasterio
from pathlib import Path
import json
import geopandas as gpd
from shapely.geometry import mapping, Polygon, MultiPolygon, GeometryCollection
from shapely.validation import make_valid
from shapely.ops import unary_union
from rasterio.features import geometry_mask


def calculate_crustal_thickness(elevation):
    if elevation <= -2000:
        return 6.5
    elif -2000 < elevation < 9000:
        # Power law fit through three reference points:
        # (-2000m, 6.5 km), (240m, 37.0 km), (9000m, 80.0 km)
        # thickness = 0.4291976178 * (elevation + 2000)^0.5526881969 + 6.5
        return 0.4291976178 * (elevation + 2000) ** 0.5526881969 + 6.5
    elif elevation >= 9000:
        return 0.002407789353 * elevation + 58.38269138
    else:
        return np.nan


# Load the configuration file
project_root = Path(__file__).resolve().parent.parent
config_path = project_root / "config.json"

with open(config_path, "r") as f:
    config = json.load(f)

input_data_folder = config["input_data_folder"]
COB_path = os.path.join(input_data_folder, "Cobgon_All_fixed.shp")
output_folder_path = config["output_folder_path"]
pyproj_data_path = config.get("pyproj_data_path")
if pyproj_data_path:
    os.environ['PROJ_DATA'] = pyproj_data_path

# Paths
palaeogeographic_maps_path = Path(output_folder_path) / "Palaeogeography"
print(palaeogeographic_maps_path)

COB_gdf = gpd.read_file(COB_path)
pattern = re.compile(r"palaeogeography_(\d+)\.tif$")

def geom_make_valid(geom):
    if geom is None or geom.is_empty:
        return None
    if geom.is_valid:
        return geom
    fixed = make_valid(geom)
    if isinstance(fixed, GeometryCollection):
        polys = [g for g in fixed.geoms if isinstance(g, (Polygon, MultiPolygon)) and not g.is_empty]
        return unary_union(polys) if polys else None
    if isinstance(fixed, (Polygon, MultiPolygon)):
        return fixed
    return None

def clean_geometries(gdf):
    gdf = gdf.copy()
    gdf["geometry"] = gdf["geometry"].apply(geom_make_valid)
    gdf = gdf[gdf["geometry"].notnull()]
    return gdf

for tif_file in palaeogeographic_maps_path.glob("*.tif"):
    match = pattern.search(tif_file.name)
    if match:
        recon_age =int(match.group(1))
        age = int(match.group(1)) - 2000
        with rasterio.open(tif_file) as src:
            data = src.read(1)
            profile = src.profile
            if src.crs != "ESRI:54034":
                print("Setting default CRS to ESRI:54034 for raster")
                src_crs = "ESRI:54034"
            else:
                src_crs = src.crs

            COB_filtered = COB_gdf[COB_gdf["APPEARANCE"] == age]
            if COB_filtered.empty:
                continue
            COB_filtered = clean_geometries(COB_filtered)
            if COB_filtered.empty:
                continue
            COB_filtered = COB_filtered.to_crs(src_crs)
            try:
                for idx, geom in enumerate(COB_filtered.geometry):
                    if not geom.is_valid:
                        print(f" Invalid geometry WKT: {geom.wkt[:300]}...")

                accum = COB_filtered.geometry.iloc[0]
                for i, geom in enumerate(COB_filtered.geometry.iloc[1:], start=1):
                    try:
                        accum = unary_union([accum, geom])
                    except Exception as e:
                        print(f"Union failed at geometry index {i}: {e}")
                        print(f"WKT: {geom.wkt[:300]}...")
                        raise
                cob_geometry = accum
            except Exception as e:
                print(f" Failed to union geometries for age {age}: {e}")
                continue

            if isinstance(cob_geometry, MultiPolygon):
                geoms = [mapping(geom) for geom in cob_geometry.geoms]
            elif isinstance(cob_geometry, Polygon):
                geoms = [mapping(cob_geometry)]
            elif isinstance(cob_geometry, GeometryCollection):
                geoms = [mapping(geom) for geom in cob_geometry.geoms if isinstance(geom, (Polygon, MultiPolygon))]
            else:
                print(f"Unsupported geometry type for APPEARANCE={age}: {type(cob_geometry)}")
                continue

            mask = geometry_mask(geoms, transform=src.transform, invert=False, out_shape=(src.height, src.width))
            new_data = np.where(mask, 6.5, np.vectorize(calculate_crustal_thickness)(data))
            crust_thick_name = f"crustal_thickness_{recon_age}.tif"
            output_crust = os.path.join(output_folder_path, "Crustal_thickness", crust_thick_name)
            os.makedirs(os.path.dirname(output_crust), exist_ok=True)

            with rasterio.open(output_crust, 'w', **profile) as dst:
                dst.write(new_data, 1)