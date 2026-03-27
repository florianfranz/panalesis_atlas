# PANALESIS Atlas

This repository contains the code to create and validate the PANALESIS Atlas products.

Dataset: Franziskakis, F., Werner, N., Vérard, C., Castelltort, S., & Giuliani, G. (2026). A Phanerozoic Atlas of Earth's Atmosphere, Surface, and Interior Derived from the PANALESIS Plate Tectonic Model (v1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.19134591

## Maps

The dataset contains maps covering the entire Phanerozoic (545 - 0 Ma) in 45 time steps, describing the following aspects:
1. Palaeogeography
2. Seafloor ages
3. Crustal thicnkess
4. Lithospheric thickness
5. Precipitation (currently only 38 time-steps)
6. Surface Air Temperature (currently only 38 time-steps)

In this repository, we provide the code used to process seafloor ages, crustal and lithospheric thickness maps in the `maps`. 
The source code to create the palaeogeographic maps is available [on GitHub](https://github.com/florianfranz/topo_chronia).

Franziskakis et al., (2026). TopoChronia: A QGIS plugin for the creation of fully quantified palaeogeographic maps. Journal of Open Source Software, 11(118), 8812, https://doi.org/10.21105/joss.08812

Moreover, the precipication and surface air temperature maps are obtained with the PLASIM-GENIE AOGCM directly.

## Validation

We provide validation code used to compare our outputs with reference data in the `validation` section.





