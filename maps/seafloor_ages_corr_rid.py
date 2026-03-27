import os
import re
import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import mapping
from rasterio.features import geometry_mask
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
seafloor_ages_maps_path = Path(base_output_folder_path) / "Seafloor_ages" / "final_outputs"
print(seafloor_ages_maps_path)
pattern = re.compile(r"final_seafloor_ages_(\d+)\_cob_masked.tif$")
pm_gdf = gpd.read_file(PM_path)
rid_pm_gdf = pm_gdf[pm_gdf["TYPE"] == "Ridge"]
buffer_dist = 0.5


for tif_file in seafloor_ages_maps_path.glob("*.tif"):
    match = pattern.search(tif_file.name)
    if match:
        recon_age =int(match.group(1))
        age = recon_age
        with rasterio.open(tif_file,'r+') as src:
            data = src.read(1)
            profile = src.profile
            sf_ages = data[data >= 0]
            valid_mask = data >= 0
            new_data = np.full_like(data, -9999, dtype=np.float32)
            RID_filtered = rid_pm_gdf[rid_pm_gdf["APPEARANCE"] == age]
            RID_filtered = RID_filtered.to_crs(src.crs)
            buffered_polygons = RID_filtered.geometry.buffer(buffer_dist)
            buffered_gdf = gpd.GeoDataFrame(geometry=buffered_polygons, crs=src.crs)
            geoms = [mapping(geom) for geom in buffered_gdf.geometry if geom is not None and geom.is_valid]

            rid_mask = geometry_mask(
                geoms,
                transform=src.transform,
                invert=False,
                out_shape=(src.height, src.width)
            )

            inside_rid = (~rid_mask) & valid_mask
            print(f"{tif_file.name}: inside_rid pixels = {inside_rid.sum()}")

            data[inside_rid] = 0

            src.write(data, 1)