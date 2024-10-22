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

The algorithm is designed to evaluate analogs based on a specified spatial domain. It has been primarily developed and tested for evaluating analogs over a synoptic domain, with the **Western Mediterranean Region (D09)** as the default domain ([Philipp et al., 2010](#philipp2010)).

The algorithm can also be run for other regions by modifying the *syn_coords* variables in the script's settings section. In such cases, it is necessary to download the corresponding ERA5 reanalysis data for the new region and compute the climatological fields for that area. This is required to perform seasonal standardization of the precipitation fields. To enable this process, the following flags in the settings section must be set to **True**: *ERA5_download* *ERA5_wts*, and *seasonal_standardization*. By default, the algorithm is configured for the D09 Western Mediterranean region (coordinates: [lon(-17, 9); lat(31, 48)]).

The evaluation of precipitation fields for the analog dates can be conducted over the entire synoptic domain by setting the mesoscale coordinates (*mes_coords*) to match the synoptic coordinates (*syn_coords*). Alternatively, the evaluation can be performed for a smaller region within the synoptic domain. In this case, the spatial domain for precipitation evaluation must be specified by modifying the *mes_coords* variables in the settings section. It is important to ensure that the mesoscale domain is always fully contained within the synoptic domain. By default, the mesoscale domain is set to the Catalonia region (coordinates: [lon(0, 3.5); lat(40, 43)]).


### 3. Flag for ERA5 Reanalysis Data downloading

The algorithm can automatically download ERA5 reanalysis data for the defined spatial domains. By default, this is set to False to avoid unnecessary downloads, as this step can be time- and memory-intensive. You can change the era5_download variable to True to enable the download.
To download ERA5 data, the script relies on the Copernicus Climate Data Store (CDS) API, which requires that you have the API key properly configured. (See the ERA5 API Setup section for details).

ERA5 API uses for download the same coordinates specified in the previous section of the setup (Synoptic domains for GPTs and mesosclae domains for precipitation). Remember to download again reanalysis data in case of change coordinates setup of synoptic spatial domain (*syn_coords*), by imposing **era5_download=True**

### 4. Flag for Weather Types (WTs) computation
If it set equal to **True** the algorithm computes the WTs table from the ERA5 geopotential data using the synoptic spatial domain and saves it in the file *input/era5_classification.csv'*

**! WARNING**: every time you change the synoptic spatial domain (*syn_coords*) definition WTs must be computed again **!**

### 5. Flag for Seasonal Precipitation Standardization Statistics (SPSS) computation
If it is set equal to **True** the algorithm computes the NetCDF file used to compute the seasonal standardization in the analog method and saves it in the file *input/seasonal_precipitation_statistics.nc'*

**! WARNING**:  every time you change the synoptic spatial domain (*syn_coords*)  definition WTs must be computed again **!** """


### Example of default setup of the algorithm

```python
    ############### Definition of the target date ################
    today = datetime.today() # Get today's date

    ############## Definition of the spatial domain ##############
    # Synoptic domain: to evaluate analogs
    syn_lat_s = 31
    syn_lat_n = 48
    syn_lon_w = -17
    syn_lon_e = 9

    # Mesoscale domain: to evaluate and extract precipitation fields
    """
    - cat = (40, 43, 0, 3.5)
    - arga = (42.25, 43.25, -2.75, -0.75)
    - d09 (Weastern Europe) = (31, 48, -17, 9)
    """
    mes_lat_s = 40
    mes_lat_n = 43
    mes_lon_w = 0
    mes_lon_e = 3.5

    ############# Download of ERA5 reanalysis fields ##############
    era5_download = False  # True

    ############# Compute WTs from ERA5 data ##############
    era5_wts = False # True

    ############# Compute Seasonal Precipitation Standardization Statistics (SPSS) ##############
    seasonal_standardization = False # True
    
```

## Usage:
The following command can be used to run the script:

```bash
python analog_algorithm.py
```

Make sure the necessary input files (e.g., ERA5 datasets and classification CSVs) are available in the `input/` directory.


## Overview:
The **Analog algorithm**, contained in the script *analog_algorithm.py* is divided into 4 modules, that are called individually from the *main()* function. Here the 4 modules are presented briefly:

1. **Download of ERA5 data**:
    Call to the Copernicus ERA5 API to download reanalysis data:
    - 500 and 1000 hPa geopotential (GPT) height databases (saved in the folder *'input/ERA5_gpt_ds'*)
    - Hourly cumulated precipitation databases by year, from 1940 to 2023 (saved in the folder *'input/ERA5_precip_ds'*)

    **!** **WARNING**: this module is run only when the variable **era5_download** is set equal to **True** in the Setup section of the *main()* function. The download of reanalysis data required large amount of time and memory space, so set the variable equal to True only if you need to download reanalysis data **!**

2. **Compute WTs**:
    Computation of WTs for all the ERA5 Reanalysis data for the period 1940-2023, at 500 hPa and 1000 hPa Geopotential Height using Beck method (Beck, 2000). The results are saved in the file *input/era5_classification.csv* and used later on in the algorithm

3. **Compute SPSS**:
    Computation of the database containing the average and standard deviation hourly field of precipitation for the 365 days of the year computed using 33 years of ERA5 precipitation data (1990-2023).
    Results are saved in the NetCDF file: *input/seasonal_precipitation_statistics.nc*

4. **GFS Module**:
    - Downloads the most recent 00 UTC run of GFS forecast files for *today* (day of algorithm run)
    - Extracts GPT fields at 500 hPa (Z500) and 1000 hPa (Z1000)
    - Computes weather types (WTs) based on GPT fields
    - Saves the hourly precipitation field forecasted by GFS from +001h to +024h valid_time as NetCDF files in the file *'output/yyyymmdd/gfs_forecast.nc'*, where yyyymmdd is the target date analyzed

5. **Analogs Computation Module**:
    - Identifies and ranks analogs comparing GPT fields forecasted by GFS at 500 and 1000 hPa with ERA5 reanalysis GPT fields, for historical days with the same WTs of the target day
    - Computes a score for each potential analog day and ranks the top 10 based on these scores.
    - Saves the ranked analog days in a CSV file and plots their respective geopotential fields in the folder *output/yyyymmdd/analog_gpt_field*
    -  Computes the statistics about floods events recorded in the past days with the same WTs of the target day analyzed and saves them in a text file in the path *output/yyyymmdd/flood_statistics.txt*. Flood events are extracted from the INUNGAMA database ([Barnolas and Llasat, 2007](#Barnolas2007))


6. **Analog Precipitation Module**:
    - For the 10 best analogs, the hourly precipitation fields are retrieved from the ERA5 dataset for a +24h time range
    - These fields are seasonally standardized (normalized) and saved as NetCDF files in the folder *'output/yyyymmdd/analog_nc*.

All the addtional functions needed to run the algorithm are contained in the script *functions.py*

## Directory Structure

### *input/*
After the clone of the repository from GitHub the input folder will contain:
- *era5_classification.csv*: csv file which contains the classification of the Weather Types for each day from *1940-01-01* to *2023-12-31*, computed using **Beck** WTs classifiation method ([Beck, 2000](#Beck2000))
- *seasonal_precipitation_statistics.nc*: NetCDF file contains the mean and standard deviation values of the precipitation fields for the dafault setting of the algorithm (synoptic scale: **D09**, mesoscale: **Catalonia**) to compute **seasonal standardization** of the analog's precipitation fields
After the download of ERA5 reanalysis data the following folders will be add to the *input/* folder:
- *ERA5_gpt_ds*: containing the GPT fields at the specified pressure levels (default 500 and 1000) for the *synoptic domain* specified in the settings
- *ERA5_precip_ds*: containing precipitation fields databases, year by year, for the *synoptic domain* specified in the settings. 

### *output/*
It contains a folder for each day for which the algorithm has been run in the format *yyyymmdd* (e.g. *20241004/*). Each day folder contains the following items:
- *analog_gpt_field/*: plots of the GPT fields at 500 and 1000 hPa with the comparison between the target date and the analogs
- *analog_nc/*: NetCDF files with the hourly precipitation fields for the best ten analogs up to +024h (from *Analog_1.nc* to *Analog_10.nc*)
- *precip_field_plot/*: plots of the hourly cumulated precipitation fields for the 24h-period analyzed, for the 10 best analogs (*analog_i.png*) and for the GFS forecast (*gfs.png*)
- flood_statistics.txt: text file containing the statistics of the flood events occurred in the past in days with same WTs of the target day analyzed, according to INUNGAMA flood database ([Barnolas and Llasat, 2007](#Barnolas2007))
- *gfs_forecast.nc*: netCDF file containing the GFS forecast of the hourly precipitation field for the target date, from +001 to +024h forecasting time
- *top_analogs.csv*: csv file with the dates of the 10 best analogs computed for the target date and their score


## NetCDF files format
### gfs_forecast.nc:

```plaintext
<xarray.Dataset> Size: 19kB
Dimensions:              (latitude: 13, longitude: 15, valid_time: 24)
Coordinates:
  * latitude             (latitude) float64 104B 43.0 42.75 42.5 ... 40.25 40.0
  * longitude            (longitude) float64 120B 0.0 0.25 0.5 ... 3.0 3.25 3.5
  * valid_time           (valid_time) datetime64[ns] 192B 2024-10-18T01:00:00...
    initialization_time  datetime64[ns] 8B ...
Data variables:
    tp                   (valid_time, latitude, longitude) float32 19kB ...
    units:                              kg m-2
    long_name:                          Total precipitation
    standard_name:                      precipitation_amount
    GRIB_cfVarName:                     tp
    GRIB_stepType:                      1h-accum
    GRIB_iDirectionIncrementInDegrees:  0.25
Attributes:
    title:        GFS hourly Cumulative Precipitation Data
    history:      Created on 2024-10-18 09:03:54.871129 from GFS data
    Conventions:  CF-1.7
    institution:  Universitat de Barcelona, GAMA team
    source:       GFS (NOAA)
    references:   https://www.ncei.noaa.gov/products/weather-climate-models/g...
    
```

**Coordinates:**
- *latitude*: resolution of 0.25 deg
- *longitude*: resolution of 0.25 deg
- *valid_time* is the current time of hourly precipitation accumulation forecast, i.e. the end of acuumulation time (e.g: valid_time = 01:00 represents the accumulation period from 00:00 UTC to 01:00 UTC)
- *init_time* is the time of initialization of the GFS model run, i.e. the target day at 00 UTC

**Data variables:**
- *tp*: total 1h-accumulated precipitation, expressed in *kg m-1* (corresponding to *mm* of rainfall)


### Analog_i.nc:

```plaintext
<xarray.Dataset> Size: 38kB
Dimensions:      (latitude: 13, longitude: 15, valid_time: 24)
Coordinates:
  * latitude     (latitude) float64 104B 43.0 42.75 42.5 ... 40.5 40.25 40.0
  * longitude    (longitude) float64 120B 0.0 0.25 0.5 0.75 ... 3.0 3.25 3.5
  * valid_time   (valid_time) datetime64[ns] 192B 2024-10-18T01:00:00 ... 202...
    analog_date  datetime64[ns] 8B ...
Data variables:
    tp           (valid_time, latitude, longitude) float64 37kB ...
    units:                              kg m-2
    long_name:                          Total precipitation
    standard_name:                      precipitation_amount
    GRIB_cfVarName:                     tp
    GRIB_stepType:                      1h-accum
Attributes:
    title:        ERA5 hourly Cumulative Precipitation Data
    history:      Created on 2024-10-18 09:05:04.802954 from ERA5 Reanalysis ...
    Conventions:  CF-1.7
    institution:  Universitat de Barcelona, GAMA team
    source:       ERA5 Reanalysis
    references:   https://climate.copernicus.eu/climate-reanalysis

```

**Coordinates:**
- *latitude*: resolution of 0.25 deg
- *longitude*: resolution of 0.25 deg
- *valid_time*: is the current time of hourly precipitation accumulation forecast, i.e. the end acuumulation time (e.g: valid_time = 01:00 represents the accumulation period from 00:00 UTC to 01:00 UTC)
- *analog_date*: is the date corresponding to the analog day (at 00:00 UTC)

**Data variables:**
- *tp*: total 1h-accumulated precipitation, expressed in *kg m-1* (corresponding to *mm* of rainfall)


## References

- **Philipp, A.**, Bartholy, J., Beck, C., Erpicum, M., Esteban, P., Fettweis, X., Huth, R., James, P., Jourdain, S., Kreienkamp, F., Krennert, T., Lykoudis, S., Michalides, S. C., Pianko-Kluczy´nska, K., Post, P.,´Alvarez, D. F. R., Schiemann, R. K. H., Spekat, A., and Tymvios, F. (2010).
Cost733cat - a database of weather and circulation type classifications. Physics and Chemistry of The Earth, 35:360–373

- **Beck, C.** (2000). Zirkulationsdynamische Variabilitat im Bereich Nordatlantik-Europa seit 

- **Barnolas, M. and Llasat, M. C**. (2007). A flood geodatabase and its climatological applications: the case of catalonia for the last century. Natural Hazards and Earth System Sciences, 7(2):271–281.

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

