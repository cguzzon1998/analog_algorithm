# Analog Algorithm

This repository contains the **Analog Algorithm**, a Python-based tool designed for weather forecasting through the comparison of geopotential fields from the Global Forecast System (GFS) with historical analogs from the ERA5 reanalysis. The primary goal is to predict 24-hour cumulative precipitation using analog days that share similar synoptic patterns.

## Overview

The algorithm is divided into three main modules:

1. **GFS Module**:
    - Downloads the most recent 00 UTC run of GFS forecast files.
    - Extracts geopotential height (GPT) fields at 500 hPa (Z500) and 1000 hPa (Z1000).
    - Computes weather types (WTs) based on these geopotential fields.
    - Saves the 24-hour cumulative precipitation field forecasted by GFS as both a NetCDF file and a PNG figure.

2. **Analogs Computation Module**:
    - Identifies and ranks historical analog days using GPT fields from ERA5 reanalysis based on the weather types and patterns computed from the GFS module.
    - Computes a score for each potential analog day and ranks the top 10 based on these scores.
    - Saves the ranked analog days in a CSV file and plots their respective geopotential fields.

3. **Analog Precipitation Module**:
    - For the top-ranked analogs, the 24-hour cumulative precipitation fields are retrieved from the ERA5 dataset.
    - These fields are seasonally standardized (normalized) and saved as NetCDF files.
    - Plots of the precipitation fields for each analog day are saved as PNG figures.

## Directory Structure

The output of each run is stored in the `output/` directory under a subfolder named with the target date (in `YYYYMMDD` format). Each module saves its outputs in the following subfolders:

- `output/YYYYMMDD/`
    - `gfs_precip_field.nc`: NetCDF file containing the 24-hour cumulative precipitation forecasted by GFS.
    - `24h_precip_field_GFS.png`: PNG figure of the GFS precipitation forecast.
    - `top_analogs.csv`: CSV file listing the top 10 analogs and their computed scores.
    - `analog_gpt_field/`: Folder containing PNG figures of the geopotential fields for the top analogs.
    - `analog_nc/`: Folder containing NetCDF files of the 24-hour cumulative precipitation fields for each analog.
    - `analog_p_field/`: Folder containing PNG figures of the precipitation fields for each analog.

## Requirements

The algorithm requires the following Python libraries:
- `xarray`
- `numpy`
- `pandas`
- `tqdm`
- `cfgrib`
- `requests`
- `matplotlib`

## Running the Algorithm

To run the algorithm, simply execute the `main()` function. The script will automatically:
1. Download and process GFS forecast data.
2. Identify and rank analog days using ERA5 reanalysis.
3. Retrieve and process precipitation fields for the top analogs.

The following command can be used to run the script:

```bash
python analog_algorithm.py
```

Make sure the necessary input files (e.g., ERA5 datasets and classification CSVs) are available in the `input/` directory.
