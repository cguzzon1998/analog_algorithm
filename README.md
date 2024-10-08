# Analog Algorithm

This repository hosts the **Analog Algorithm**, a Python-based tool designed to enhance the real-time forecasting of precipitation fields using analogs. The algorithm aims to refine the 24-hour precipitation forecast provided by the Global Forecast System (GFS) by identifying historical weather patterns, defined as *analogs*, that resemble the synoptic conditions of a given target day. By comparing the forecasted geopotential fields at 500 and 1000 hPa from the GFS with those from past events, this method leverages observed precipitation data from similar atmospheric setups to provide more accurate forecasts.

## Table of Contents
- [Installation](#installation)
- [Setup](#setup)
- [Usage](#usage)
- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [NetCDF files format](#netcdf-files-format)
- [References](#references)
- [Contact](#contact)


## Installation

### 1. Clone the repository from GitHub

To get started, clone the repository from GitHub:

```bash
git clone https://github.com/cguzzon1998/analog_algorithm.git
```

Then navigate to the cloned directory:
```bash
cd analog_algorithm
```

### 2. Install python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### 3. Set up of Copernicus ERA5 API:
This procedure is necessary to be able to download the data from ERA5 reanalysis directly from the API. If you already had saved ERA5 data and stored them in the input folder you can skip the set up steps.

**Step 1. Install the `cdsapi` Python package**

First, you need to install the `cdsapi` Python library, which is required to interact with the Copernicus Climate Data Store API. You can install it using `pip`:

```bash
pip install cdsapi
```

Alternatively, if you are using a `conda` environment:

```bash
conda install -c conda-forge cdsapi
```

**Step 2: Obtain your API Key from Copernicus Climate Data Store (CDS)**

1. Register at the [Copernicus Climate Data Store](https://cds.climate.copernicus.eu).
2. Log in to your account.
3. Go to your [API key page](https://cds.climate.copernicus.eu/api-how-to), and copy your unique API key.

Your API key will be in the format:

```
url: https://cds.climate.copernicus.eu/api/v2
key: <your-UID>:<your-API-key>
```

**Step 3: Configure the API Key**

To configure your system to use the API key, create a file named `.cdsapirc` in your home directory (or directly write it to the configuration if it's already set up).

```bash
nano ~/.cdsapirc
```

Paste the following content into the file:

```plaintext
url: https://cds.climate.copernicus.eu/api/v2
key: <your-UID>:<your-API-key>
```

Replace `<your-UID>` and `<your-API-key>` with the values from your Copernicus account.

Save the file (`Ctrl + X`, then `Y`, then `Enter`).


## Setup
The `analog_algorithm.py` requires a few setup steps before execution, all of which are managed within the `main()` function. These steps include:

### 1. Definition of the Target Date
The script allows you to define the date for which the analysis will be performed. By default, the target date is set to the current day (today). It is possible to analyze up to 9 days prior to the current date, which are available online in the *NOAA Operational Model Archive and Distribution System*. Check the following link for details: [https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/](https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/)
The script will automatically create an output folder for the results, named after the target date (e.g., `output/20231003` for October 3rd, 2024).

### 2. Definition of the Spatial Domain
You need to define two spatial domains:
- **Synoptic Domain:** This is the larger domain used to evaluate the geopotential (GPT) fields for analog selection on a synoptic scale. The latitude and longitude bounds are set manually within the script. By default they are set for the **Western Mediterranean Region (D09)** ([Philipp et al., 2010](#philipp2010))
 
- **Mesoscale Domain:** This smaller domain is used for evaluating precipitation fields at mesoscale level. It is specific to the region of interest. By default is set up for Catalonia region (coords = [lon(0, 3.5); lat(40, 43)]).

### 3. Download ERA5 Reanalysis Data

The algorithm can automatically download ERA5 reanalysis data for the defined spatial domains. By default, this is set to False to avoid unnecessary downloads, as this step can be time- and memory-intensive. You can change the era5_download variable to True to enable the download.
To download ERA5 data, the script relies on the Copernicus Climate Data Store (CDS) API, which requires that you have the API key properly configured. (See the ERA5 API Setup section for details).

ERA5 API uses for download the same coordinates specified in the previous section of the setup (Synoptic domains for GPTs and mesosclae domains for precipitation). Remember to download again reanalysis data in case of change coordinates setup, by imposing **era5_download=True**

### Example of default setup of the algorithm

```python
    ############### Definition of the target date ################
    today = datetime.today() # Get today's date

    ############## Definition of the spatial domain ##############
    # Synoptic domain: to extract GPT field and compute WTs
    syn_lat_s = 31
    syn_lat_n = 48
    syn_lon_w = -17
    syn_lon_e = 9

    # Mesoscale domain: to extract precipitation field
    mes_lat_s = 40
    mes_lat_n = 43
    mes_lon_w = 0
    mes_lon_e = 3.5

    ############# Download of ERA5 reanalysis fields ##############
    era5_download = False  # True
```

## Usage:
The following command can be used to run the script:

```bash
python analog_algorithm.py
```

Make sure the necessary input files (e.g., ERA5 datasets and classification CSVs) are available in the `input/` directory.


## Overview:
The **Analog algorithm**, contained in the script *analog_algorithm.py* is divided into 4 modules, that are called individually from the *main()* function. Here the 4 modules are presented briefly:

1. **Download of ERA5 data**
    Call to the Copernicus ERA5 API to download reanalysis data:
    - 500 and 1000 hPa geopotential (GPT) height databases (saved in the folder *'input/ERA5_gpt_ds'*)
    - Hourly cumulated precipitation databases by year, from 1940 to 2023 (saved in the folder *'input/ERA5_precip_ds'*)

    **!** **WARNING**: this module is run only when the variable **era5_download** is set equal to **True** in the Setup section of the *main()* function. The download of reanalysis data required large amount of time and memory space, so set the variable equal to True only if you need to download reanalysis data **!**

2. **GFS Module**:
    - Downloads the most recent 00 UTC run of GFS forecast files for *today* (day of algorithm run)
    - Extracts GPT fields at 500 hPa (Z500) and 1000 hPa (Z1000)
    - Computes weather types (WTs) based on GPT fields
    - Saves the hourly precipitation field forecasted by GFS from +001h to +024h valid_time as NetCDF files in the file *'output/yyyymmdd/gfs_forecast.nc'*, where yyyymmdd is the target date analyzed

3. **Analogs Computation Module**:
    - Identifies and ranks analogs comparing GPT fields forecasted by GFS at 500 and 1000 hPa with ERA5 reanalysis GPT fields, for historical days with the same WTs of the target day
    - Computes a score for each potential analog day and ranks the top 10 based on these scores.
    - Saves the ranked analog days in a CSV file and plots their respective geopotential fields in the folder *'output/yyyymmdd/analog_gpt_field*.

4. **Analog Precipitation Module**:
    - For the 10 best analogs, the hourly precipitation fields are retrieved from the ERA5 dataset for a +24h time range
    - These fields are seasonally standardized (normalized) and saved as NetCDF files in the folder *'output/yyyymmdd/analog_nc*.

All the addtional functions needed to run the algorithm are contained in the script *functions.py*

## Directory Structure

### *input/*
After the clone of the repository from GitHub the input folder will contain:
- *era5_classification.csv*: csv file which contains the classification of the Weather Types for each day from *1940-01-01* to *2023-12-31*, computed using **Beck** WTs classifiation method ([Beck, 2000](#Beck2000))
- *seasonal_precip_standardization.nc*: NetCDF file contains the mean and standard deviation values of the precipitation fields for the dafault setting of the algorithm (synoptic scale: **D09**, mesoscale: **Catalonia**) to compute **seasonal standardization** of the analog's precipitation fields
After the download of ERA5 reanalysis data the following folders will be add to the *input/* folder:
- *ERA5_gpt_ds*: containing the GPT fields at the specified pressure levels (default 500 and 1000) for the *synoptic domain* specified in the settings
- *ERA5_precip_ds*: containing precipitation fields databases, year by year, for the *mesoscale domain* specified in the settings

### *output/*
It contains a folder for each day for which the algorithm has been run in the format *yyyymmdd* (e.g. *20241004/*). Each day folder contains the following items:
- *analog_gpt_field/*: plots of the GPT fields at 500 and 1000 hPa with the comparison between the target date and the analogs
- *analog_nc/*: NetCDF files with the hourly precipitation fields for the best ten analogs up to +024h (from *Analog_1.nc* to *Analog_10.nc*)
- *precip_field_plot/*: plots of the hourly cumulated precipitation fields for the 24h-period analyzed, for the 10 best analogs (*analog_i.png*) and for the GFS forecast (*gfs.png*)
- *gfs_forecast.nc*: netCDF file containing the GFS forecast of the hourly precipitation field for the target date, from +001 to +024h forecasting time
- *top_analogs.csv*: csv file with the dates of the 10 best analogs computed for the target date and their score


## NetCDF files format
### gfs_forecast.nc**:
```plaintext
<xarray.DataArray 'tp' (time: 24, latitude: 13, longitude: 15)> Size: 19kB
[4680 values with dtype=float32]
Coordinates:
  * latitude        (latitude) float64 104B 43.0 42.75 42.5 ... 40.5 40.25 40.0
  * longitude       (longitude) float64 120B 0.0 0.25 0.5 0.75 ... 3.0 3.25 3.5
    init_time       datetime64[ns] 8B ...
    valid_time      (time) datetime64[ns] 192B ...
    start_acc_time  (time) datetime64[ns] 192B ...
Dimensions without coordinates: time
Attributes: (12/30)
    GRIB_typeOfLevel:                         surface
    GRIB_stepUnits:                           1
    GRIB_stepType:                            accum
    GRIB_name:                                Total Precipitation
    GRIB_shortName:                           tp
    GRIB_units:                               kg m**-2
    long_name:                                Total Precipitation
    units:                                    kg m**-2 
```
**Coordinates:**
- *latitude*: resolution of 0.25º
- *longitude*: resolution of 0.25º
- *init_time* is the time of initialization of the GFS model run, i.e. the target day at 00 UTC
- *valid_time* is the current time of hourly precipitation accumulation forecast, so it is the end acuumulation time (e.g: valid_time = 01:00 represents the accumulation period from 00:00 UTC to 01:00 UTC)
- *start_acc_time* is the initial time of the accumulation period (e.g: 00:00 UTC for the accumulation period from 00:00 UTC to 01:00 UTC)

**Attributes:**
- *units*: kg m**-2 which correspond to mm (millimiters of rainfall fallen in the pixel in 1 hour period)

### **Analog_i.nc**:

```plaintext
<xarray.DataArray 'tp' (valid_time: 24, latitude: 13, longitude: 15)> Size: 37kB
[4680 values with dtype=float64]
Coordinates:
  * latitude        (latitude) float64 104B 43.0 42.75 42.5 ... 40.5 40.25 40.0
  * longitude       (longitude) float64 120B 0.0 0.25 0.5 0.75 ... 3.0 3.25 3.5
    an_day          datetime64[ns] 8B ...
  * valid_time      (valid_time) datetime64[ns] 192B 2002-08-10T01:00:00 ... ...
    start_acc_time  (valid_time) datetime64[ns] 192B ...
Attributes:
    GRIB_edition:            1
    GRIB_centre:             ecmf
    GRIB_centreDescription:  European Centre for Medium-Range Weather Forecasts
    GRIB_subCentre:          0
    Conventions:             CF-1.7
    institution:             European Centre for Medium-Range Weather Forecasts
    history:                 2024-10-03T13:31 GRIB to CDM+CF via cfgrib-0.9.1
    units:                   mm
```

**Coordinates:**
- *latitude*: resolution of 0.25º
- *longitude*: resolution of 0.25º
- *an_date* is the analog date at 00:00 UTC
- *valid_time* is the current time of hourly precipitation accumulation forecast, so it is the end acuumulation time (e.g: valid_time = 01:00 represents the accumulation period from 00:00 UTC to 01:00 UTC)
- *start_acc_time* is the initial time of the accumulation period (e.g: 00:00 UTC for the accumulation period from 00:00 UTC to 01:00 UTC)

**Attributes:**
- *units*: mm (millimiters of rainfall fallen in the pixel in 1 hour period)


## References

- **Philipp, A.**, Bartholy, J., Beck, C., Erpicum, M., Esteban, P., Fettweis, X., Huth, R., James, P., Jourdain, S., Kreienkamp, F., Krennert, T., Lykoudis, S., Michalides, S. C., Pianko-Kluczy´nska, K., Post, P.,´Alvarez, D. F. R., Schiemann, R. K. H., Spekat, A., and Tymvios, F. (2010).
Cost733cat - a database of weather and circulation type classifications. Physics and Chemistry of The Earth, 35:360–373

- **Beck, C.** (2000). Zirkulationsdynamische Variabilit¨at
im Bereich Nordatlantik-Europa seit 1780

## Contact

**Carlo Guzzon**  
**Email**: [cguzzon@meteo.ub.edu](mailto:cguzzon@meteo.ub.edu)

GAMA Team (Meteorological Hazards Analysis Team)  
Dept. Applied Physics  
Faculty of Physics, University of Barcelona  
Martí i Franqués, 1, 08028 Barcelona  
[http://gamariesgos.wordpress.com/](http://gamariesgos.wordpress.com/)

Follow us on Twitter: [@GAMA_UB](https://twitter.com/GAMA_UB)  
Join us on Facebook: [GAMA - Riesgos Naturales - UB](https://www.facebook.com/GAMARiesgosNaturalesUB)
PARTICIPA EN [WWW.FLOODUP.UB.EDU](http://www.floodup.ub.edu)

