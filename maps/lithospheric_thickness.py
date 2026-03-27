import os
import re
from pathlib import Path
import json
import geopandas as gpd
import rasterio
import numpy as np
from rasterio.features import geometry_mask
from rasterio.warp import reproject, Resampling
from shapely.geometry import mapping, MultiPolygon, Polygon, GeometryCollection
from shapely.validation import make_valid
from shapely.ops import unary_union
import matplotlib.pyplot as plt


def calculate_lithospheric_thickness(crustal_thickness, seafloor_age, cob_mask, cra_mask, rift_age_raster):
    """
    Calculate lithospheric thickness based on crustal thickness, tectonic setting, and rift age.
    Rules:
    - Oceanic: sqrt(age) law
    - Continental normal: multiplier ~3.5
    - Cratonic: multiplier ~6.0
    - Rift: multiplier scales with age (1.5 to 2.5 over 0–100 Myr)
    - Ambiguous: multiplier ~3.0
    - Special case: inside COB with crust thickness <= 6.5 km → lithosphere = 98 km
    - Fallback: unclassified pixels treated as oceanic
    """

    litho_thickness = np.zeros_like(crustal_thickness)
    seafloor_age_clean = np.where(seafloor_age == -9999, np.nan, seafloor_age)
    crustal_thickness_clean = np.where(crustal_thickness < 0, 0, crustal_thickness)

    print(f"COB mask - True: {np.sum(cob_mask)}, False: {np.sum(~cob_mask)}")
    print(f"CRA mask - True: {np.sum(cra_mask)}, False: {np.sum(~cra_mask)}")
    print(f"Rift age raster: NaN count = {np.sum(np.isnan(rift_age_raster))}")
    print(f"Crustal thickness range: {np.min(crustal_thickness_clean)} to {np.max(crustal_thickness_clean)}")
    print(f"Seafloor age range: {np.nanmin(seafloor_age_clean)} to {np.nanmax(seafloor_age_clean)}")

    # Masks
    inside_cob = cob_mask.astype(bool)
    inside_cra = cra_mask.astype(bool)
    oceanic_mask = ~inside_cob

    # Oceanic lithosphere
    valid_oceanic = oceanic_mask & ~np.isnan(seafloor_age_clean) & (crustal_thickness_clean > 0)
    litho_thickness[valid_oceanic] = (crustal_thickness_clean[valid_oceanic] +2 * 1.33810856536126 * np.sqrt(22.9423752 * seafloor_age_clean[valid_oceanic]))

    # Special case: very thin crust inside COB
    special_case_mask = inside_cob & (crustal_thickness_clean <= 30)
    litho_thickness[special_case_mask] = 95.0

    # Continental normal
    continental_normal_mask = inside_cob & ~inside_cra & np.isnan(rift_age_raster) & ~special_case_mask
    valid_continental = continental_normal_mask & (crustal_thickness_clean > 0)
    litho_thickness[valid_continental] = crustal_thickness_clean[valid_continental] * 3

    # Cratonic
    cratonic_mask = inside_cob & inside_cra & np.isnan(rift_age_raster) & ~special_case_mask
    valid_cratonic = cratonic_mask & (crustal_thickness_clean > 0)
    litho_thickness[valid_cratonic] = crustal_thickness_clean[valid_cratonic] * 5

    # Rift (age-dependent multiplier)
    valid_rift = inside_cob & ~inside_cra & ~np.isnan(rift_age_raster) & (crustal_thickness_clean > 0) & ~special_case_mask
    if np.any(valid_rift):
        rift_age = rift_age_raster[valid_rift]
        factor = 0.75 + np.clip(rift_age / 100, 0, 1.0)
        litho_thickness[valid_rift] = crustal_thickness_clean[valid_rift] * factor

    # Ambiguous (COB + CRA + Rift)
    ambiguous_mask = inside_cob & inside_cra & ~np.isnan(rift_age_raster) & ~special_case_mask
    valid_ambiguous = ambiguous_mask & (crustal_thickness_clean > 0)
    litho_thickness[valid_ambiguous] = crustal_thickness_clean[valid_ambiguous] * 2.5

    return litho_thickness
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


project_root = Path(__file__).resolve().parent.parent
config_path = project_root / "config.json"

with open(config_path, "r") as f:
    config = json.load(f)

input_data_folder = config["input_data_folder"]
COB_path = os.path.join(input_data_folder, "Cobgon_All_fixed.shp")
#PM_path = os.path.join(input_data_folder, "R_psAbs_All.shp")
PM_path = r"C:/Users/franzisf/Documents/TopographyMaker_Dev/with_CRA/R_psAbs_2025/brute_force/R_psAbs_All.shp"
PM_RIB_path = os.path.join(input_data_folder, "RIB_lines_buff_clip3.shp")
output_folder_path = config["output_folder_path"]
pyproj_data_path = config.get("pyproj_data_path")
if pyproj_data_path:
    os.environ['PROJ_DATA'] = pyproj_data_path

seafloor_age_maps_path = Path(output_folder_path) / "Seafloor_ages"
crustal_thickness_maps_path = Path(output_folder_path) / "Crustal_thickness"
elevation_maps_path = Path(output_folder_path) / "Palaeogeography"

COB_gdf = gpd.read_file(COB_path)

def convert_to_polygon(geometry):
    if geometry.geom_type == 'LineString':
        coords = list(geometry.coords)
        if len(coords) == 3:
            coords.append(coords[0])
            return Polygon(coords)
        elif len(coords) > 3:
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            return Polygon(coords)
    elif geometry.geom_type == 'MultiLineString':
        polygons = []
        for line in geometry.geoms:
            coords = list(line.coords)
            if len(coords) == 3:
                coords.append(coords[0])
                polygons.append(Polygon(coords))
            elif len(coords) > 3:
                if coords[0] != coords[-1]:
                    coords.append(coords[0])
                polygons.append(Polygon(coords))
        if polygons:
            return MultiPolygon(polygons)
    return None


def convert_geometries(gdf):
    converted = 0
    skipped = 0
    non_geoms = 0
    new_geometries = []

    for geom in gdf.geometry:
        if geom is None:
            non_geoms += 1
            skipped += 1
            new_geometries.append(None)
        else:
            new_geom = convert_to_polygon(geom)
            if new_geom is not None:
                converted += 1
                new_geometries.append(new_geom)
            else:
                skipped += 1
                new_geometries.append(geom)

    gdf.geometry = new_geometries
    return gdf, converted, skipped

pm_gdf = gpd.read_file(PM_path)

cra_filtered_pm_gdf = pm_gdf[pm_gdf["TYPE"] == "Limit_Craton"]
cra_converted_gdf, cra_converted_count, cra_skipped_count = convert_geometries(cra_filtered_pm_gdf)
cra_converted_gdf = cra_converted_gdf[cra_converted_gdf.geom_type.isin(["Polygon", "MultiPolygon"])]

rib_gdf = gpd.read_file(PM_RIB_path)
rib_converted_gdf = rib_gdf[rib_gdf.geom_type.isin(["Polygon", "MultiPolygon"])]




pattern =re.compile(r"palaeogeography_(\d+)\.tif$")
sf_pattern = re.compile(r"seafloor_age_(\d+)\.tif$")

for elevation_file in elevation_maps_path.glob("*.tif"):
    match = pattern.search(elevation_file.name)
    if match:
        print(elevation_file)
        recon_age = int(match.group(1))
        age = recon_age - 2000
        print(f"processing age: {age}")
        seafloor_age_file = seafloor_age_maps_path / f"seafloor_age_{recon_age}.tif"
        crustal_thickness_file = crustal_thickness_maps_path / f"crustal_thickness_{recon_age}.tif"
        try:
            with rasterio.open(elevation_file) as elev_src, \
                 rasterio.open(seafloor_age_file) as sea_age_src, \
                 rasterio.open(crustal_thickness_file) as crust_src:
                elevation = elev_src.read(1)
                if elev_src.crs != "EPSG:4326":
                    print("Setting default CRS to EPSG:4326 for raster")
                    elev_src_crs = "EPSG:4326"
                else:
                    elev_src_crs = elev_src.crs

                seafloor_age = np.empty_like(elevation)
                reproject(
                    source=rasterio.band(sea_age_src, 1),
                    destination=seafloor_age,
                    src_transform=sea_age_src.transform,
                    src_crs=sea_age_src.crs,
                    dst_transform=elev_src.transform,
                    dst_crs=elev_src_crs,
                    resampling=Resampling.nearest
                )

                crustal_thickness = np.empty_like(elevation)
                reproject(
                    source=rasterio.band(crust_src, 1),
                    destination=crustal_thickness,
                    src_transform=crust_src.transform,
                    src_crs=crust_src.crs,
                    dst_transform=elev_src.transform,
                    dst_crs=elev_src_crs,
                    resampling=Resampling.nearest
                )
                print(f"Processing COB geometries for age {age}")

                geoms = []
                COB_filtered = COB_gdf[COB_gdf["APPEARANCE"] == age]
                COB_filtered = COB_filtered.to_crs(elev_src_crs)
                try:
                    for idx, geom in enumerate(COB_filtered.geometry):
                        if not geom.is_valid:
                            print(f" COB: Invalid geometry WKT: {geom.wkt[:300]}...")
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
                    raise ValueError(f"Unsupported geometry type for APPEARANCE={age}: {type(cob_geometry)}")

                cob_mask = geometry_mask(
                    geoms,
                    transform=elev_src.transform,
                    invert=True,
                    out_shape=(elev_src.height, elev_src.width)
                )

                from rasterio.features import rasterize
                from shapely.ops import unary_union
                from shapely.geometry import Polygon, MultiPolygon, GeometryCollection, mapping
                import numpy as np

                print(f"Processing RIB geometries for age {age}")

                # Filter rift features for current time slice
                RIB_filtered = rib_converted_gdf[
                    (rib_converted_gdf["APPEARANCE"] == age) &
                    ((rib_converted_gdf["AGE"] - rib_converted_gdf["APPEARANCE"]) <= 100)
                    ]

                print(
                    f"For age {age}, filtered {len(RIB_filtered)} features out of a total of {len(rib_converted_gdf)}")

                # Compute rift age for each feature
                RIB_filtered["RIFT_AGE"] = RIB_filtered["AGE"] - RIB_filtered["APPEARANCE"]

                # Clean geometries and reproject
                RIB_filtered = clean_geometries(RIB_filtered).to_crs(elev_src_crs)

                try:
                    # Validate geometries
                    for idx, geom in enumerate(RIB_filtered.geometry):
                        if not geom.is_valid:
                            print(f"RIB: Invalid geometry WKT: {geom.wkt[:300]}...")

                    valid_geoms = list(RIB_filtered.geometry)
                    if not valid_geoms:
                        print(f"No valid RIB geometries for age {age}. Skipping.")
                        rib_age_raster = np.full((elev_src.height, elev_src.width), np.nan, dtype="float32")
                    else:
                        # Union all geometries
                        accum = valid_geoms[0]
                        for i, geom in enumerate(valid_geoms[1:], start=1):
                            try:
                                accum = unary_union([accum, geom])
                            except Exception as e:
                                print(f"Union failed at geometry index {i}: {e}")
                                print(f"WKT: {geom.wkt[:300]}...")
                                continue

                        rib_geometry = accum

                        # Prepare geometry list for rasterization
                        if isinstance(rib_geometry, MultiPolygon):
                            geoms = [mapping(geom) for geom in rib_geometry.geoms]
                        elif isinstance(rib_geometry, Polygon):
                            geoms = [mapping(rib_geometry)]
                        elif isinstance(rib_geometry, GeometryCollection):
                            geoms = [mapping(geom) for geom in rib_geometry.geoms if
                                     isinstance(geom, (Polygon, MultiPolygon))]
                        else:
                            raise ValueError(
                                f"Rifts and Basins: Unsupported geometry type for APPEARANCE={age}: {type(rib_geometry)}")

                        # Prepare (geometry, age) pairs for rasterization
                        geom_age_pairs = [(geom, age_val) for geom, age_val in
                                          zip(RIB_filtered.geometry, RIB_filtered["RIFT_AGE"])]

                        # Rasterize rift age (NaN outside rifts)
                        rib_age_raster = rasterize(
                            geom_age_pairs,
                            out_shape=(elev_src.height, elev_src.width),
                            transform=elev_src.transform,
                            fill=np.nan,
                            dtype="float32"
                        )

                except Exception as e:
                    print(f"Failed to process RIB geometries for age {age}: {e}")
                    rib_age_raster = np.full((elev_src.height, elev_src.width), np.nan, dtype="float32")

                geoms = []
                print(f"Processing CRA geometries for age {age}")
                CRA_filtered = cra_converted_gdf[cra_converted_gdf["APPEARANCE"] == age]
                CRA_filtered = clean_geometries(CRA_filtered)
                CRA_filtered = CRA_filtered.to_crs(elev_src_crs)

                try:
                    for idx, geom in enumerate(CRA_filtered.geometry):
                        if not geom.is_valid:
                            print(f"RIB: Invalid geometry WKT: {geom.wkt[:300]}...")

                    valid_geoms = list(CRA_filtered.geometry)
                    if not valid_geoms:
                        print(f"No valid geometries for age {age}. Skipping.")
                        continue

                    accum = valid_geoms[0]
                    for i, geom in enumerate(valid_geoms[1:], start=1):
                        try:
                            accum = unary_union([accum, geom])
                        except Exception as e:
                            print(f"Union failed at geometry index {i}: {e}")
                            print(f"WKT: {geom.wkt[:300]}...")
                            continue

                    cra_geometry = accum

                except Exception as e:
                    print(f"Failed to union geometries for age {age}: {e}")
                    continue

                if isinstance(cra_geometry, MultiPolygon):
                    geoms = [mapping(geom) for geom in cra_geometry.geoms]
                elif isinstance(cra_geometry, Polygon):
                    geoms = [mapping(cra_geometry)]
                elif isinstance(cra_geometry, GeometryCollection):
                    geoms = [mapping(geom) for geom in cra_geometry.geoms if isinstance(geom, (Polygon, MultiPolygon))]
                else:
                    raise ValueError(f"Cratons: Unsupported geometry type for APPEARANCE={age}: {type(cra_geometry)}")

                cra_mask = geometry_mask(
                    geoms,
                    transform=elev_src.transform,
                    invert=True,
                    out_shape=(elev_src.height, elev_src.width)
                )

                litho_thickness = calculate_lithospheric_thickness(
                    crustal_thickness, seafloor_age, cob_mask, cra_mask,rib_age_raster
                )

                litho_thick_name = f"lithospheric_thickness_{recon_age}.tif"
                output_litho = os.path.join(output_folder_path, "Lithospheric_thickness", litho_thick_name)
                os.makedirs(os.path.dirname(output_litho), exist_ok=True)

                with rasterio.open(
                    output_litho,
                    'w',
                    driver='GTiff',
                    height=elev_src.height,
                    width=elev_src.width,
                    count=1,
                    dtype=litho_thickness.dtype,
                    crs=elev_src_crs,
                    transform=elev_src.transform
                ) as dst:
                    dst.write(litho_thickness, 1)
        except Exception as e:
            print(f"An error occurred while opening the raster files: {e}")

print("Lithospheric thickness calculation completed.")

