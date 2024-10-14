# %% APIs to download ERA5 reanalysis data
import cdsapi
from tqdm import tqdm
import os

def download_era5_gpt(level, coords):

    """ API of ERA5 to download reanalysis data of geopotential height fields for a specified level
        INPUT: 
            - level: pressure level of the geopotential height field (eg:'500')
            - coords: array of the coordinates forming the spatial box to download data, in the format [lat_n, lon_w, lat_s, lon_e]
                      (eg: for Western Meditarranean region (A. Philipp, 2010) -> coords = [48, -17, 31, 9] )
        OUTPUT: ds of the requested gpt data saved in the folder gpt_fold"""

    gpt_fold = 'test/ERA5_gpt_ds'
    os.makedirs(gpt_fold, exist_ok=True)

    dataset = "reanalysis-era5-pressure-levels"
    request = {
        "product_type": ["reanalysis"],
        "variable": ["geopotential"],
        "year": [
            "1940", "1941", "1942",
            "1943", "1944", "1945",
            "1946", "1947", "1948",
            "1949", "1950", "1951",
            "1952", "1953", "1954",
            "1955", "1956", "1957",
            "1958", "1959", "1960",
            "1961", "1962", "1963",
            "1964", "1965", "1966",
            "1967", "1968", "1969",
            "1970", "1971", "1972",
            "1973", "1974", "1975",
            "1976", "1977", "1978",
            "1979", "1980", "1981",
            "1982", "1983", "1984",
            "1985", "1986", "1987",
            "1988", "1989", "1990",
            "1991", "1992", "1993",
            "1994", "1995", "1996",
            "1997", "1998", "1999",
            "2000", "2001", "2002",
            "2003", "2004", "2005",
            "2006", "2007", "2008",
            "2009", "2010", "2011",
            "2012", "2013", "2014",
            "2015", "2016", "2017",
            "2018", "2019", "2020",
            "2021", "2022", "2023"
        ],
        "month": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12"
        ],
        "day": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12",
            "13", "14", "15",
            "16", "17", "18",
            "19", "20", "21",
            "22", "23", "24",
            "25", "26", "27",
            "28", "29", "30",
            "31"
        ],
        "time": ["00:00"],
        "pressure_level": [level],
        "data_format": "grib",
        "download_format": "unarchived",
        "area": coords
    }

    client = cdsapi.Client()
    client.retrieve(dataset, request).download(f'{gpt_fold}/geopotential_data_{level}.grib')
   
def download_era5_precip(coords):

    """ API of ERA5 to download reanalysis data of hourly cumulated precipitation data for a specified region
        INPUT: 
            - coords: array of the coordinates forming the spatial box to download data, in the format [lat_n, lon_w, lat_s, lon_e]
                      (eg: for Catalonia region -> coords = [43, 0, 40, 3.5] )
        OUTPUT: ds of the requested precipitation data saved in the folder precip_fold"""
    
    precip_fold = 'input/ERA5_precip_ds'
    os.makedirs(precip_fold, exist_ok=True)

    for y in tqdm(range(1940, 2024), desc = 'Downloading ERA5 precipitation reanalysis data', leave = False):

        dataset = "reanalysis-era5-single-levels"
        request = {
            "product_type": ["reanalysis"],
            "variable": ["total_precipitation"],
            "year": [str(y)],
            "month": [
                "01", "02", "03",
                "04", "05", "06",
                "07", "08", "09",
                "10", "11", "12"
            ],
            "day": [
                "01", "02", "03",
                "04", "05", "06",
                "07", "08", "09",
                "10", "11", "12",
                "13", "14", "15",
                "16", "17", "18",
                "19", "20", "21",
                "22", "23", "24",
                "25", "26", "27",
                "28", "29", "30",
                "31"
            ],
            "time": [
                "00:00", "01:00", "02:00",
                "03:00", "04:00", "05:00",
                "06:00", "07:00", "08:00",
                "09:00", "10:00", "11:00",
                "12:00", "13:00", "14:00",
                "15:00", "16:00", "17:00",
                "18:00", "19:00", "20:00",
                "21:00", "22:00", "23:00"
            ],
            "data_format": "grib",
            "download_format": "unarchived",
            "area": coords
        }

        client = cdsapi.Client()
        client.retrieve(dataset, request).download(f'{precip_fold}/{y}_ds.grib')


# %% Compute WTs from ERA5 data (1940-2023)
from tqdm import tqdm
import pandas as pd
import xarray as xr

def compute_era5_wts():
    """ Function to compute WTs for each day of the ERA5 database, going from 1940 to 2023, using Beck method (Beck, 2000)
    
    OUTPUT: WTs classification is saved in a csv file in the path 'input/test_era5_classification.csv'"""
    fp_500 = 'input/ERA5_gpt_ds/geopotential_data_500.grib'
    fp_1000 = 'input/ERA5_gpt_ds/geopotential_data_1000.grib'
    ds_500 = xr.open_dataset(fp_500, engine='cfgrib')
    ds_1000 = xr.open_dataset(fp_1000, engine='cfgrib')

    wt_table = []

    for time in tqdm(ds_500.time, desc = 'Compute ERA5 WTs (1940-2023)', leave = False):
        # Extract geopotential fields at 500 and 1000 hPa
        field_500 = ds_500.sel(time=time)['z'].values
        field_1000 = ds_1000.sel(time=time)['z'].values

        # Define WT with Beck classification
        wt_500 = compute_wt(field_500)
        wt_1000 = compute_wt(field_1000)
        
        wt_table.append([time.values, wt_500, wt_1000])

    df = pd.DataFrame(wt_table, columns=["Date", "wt_500", "wt_1000"])
    df.to_csv("input/era5_classification.csv", index=False)


# %% Compute Seasonal Precipitation Standardization Statistics (SPSS)
from tqdm import tqdm
from datetime import datetime, timedelta

def compute_p_field(date, ds):
    """Compute mean hourly precipitation field """
    start_date = date - pd.Timedelta(days=1)
    end_date = date + pd.Timedelta(days=1)
    ds_cat = ds.sel(time=slice(start_date, end_date))
    tot_precip = 0
    for day in range(1, ds_cat.sizes['time']):
        for tstep in range(0, ds_cat.sizes['step']):
            if (day == 1 and tstep in [0,1,2,3,4,5]) or (day == ds_cat.sizes['time']-1 and tstep in [6,7,8,9,10,11]):
                continue
            field = ds_cat.isel(time=day).isel(step=tstep).tp.values
            tot_precip += field * 1000  # mm
            tot_precip = tot_precip/24 # Divide by 24h to get mean hourly values of precipitation
    return tot_precip

def day_of_year_to_date(day_of_y):
    """Compute day and month starting from the date of the year number (from 1 to 365)"""
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if day_of_y < 1 or day_of_y > 365:
        raise ValueError("Day of the year must be included between 1 and 365")
    month = 1
    while day_of_y > days_in_month[month - 1]:
        day_of_y -= days_in_month[month - 1]
        month += 1
    day = day_of_y
    return day, month

def compute_spss(coords):
    """Function to compute Seasonal Precipitation Standardization Statistics (SPSS):
        - thr to compute mean and std is set equal to 1 mm
        - mean and std are computed over a period that spans from 1990 to 2023
        
        OUTPUT: NetCDF file saved with the path: 'input/seasonal_precip_standardization.nc'
    """
    thr = 1/24 # Set the threshold of precipitation

    # Define the range of years
    start_year = 1990
    end_year = 2023

    # Create the xarray dataset
    lon = np.arange(coords[2], coords[3]+0.25, 0.25)
    lat = np.arange(coords[1], coords[0]+0.25, 0.25)

    # Create an empty array to store the results
    all_dates = pd.date_range(start=datetime(start_year, 1, 1), end=datetime(end_year, 12, 31))
    daily_p_array = np.zeros((len(all_dates), len(lat), len(lon)))

    idx = 0 # Initializate index
    for year in tqdm(range(start_year, end_year+1), desc="Computing SPSS", leave = False):
        filename = f'input/ERA5_precip_ds/{year}_ds.grib'
        with xr.open_dataset(filename, engine='cfgrib') as ds:
            start_date = datetime(year, 1, 1)
            end_date = datetime(year, 12, 31)
            date_list = pd.date_range(start=start_date, end=end_date).tolist()
            
            for date in tqdm(date_list, desc=f"Days {year}", leave=False):
                daily_p = compute_p_field(date, ds)

                daily_p = np.where(daily_p < thr, np.nan, daily_p) # Convert to Nan all the values lower than the threshold
                daily_p_array[idx, :, :] = daily_p
                idx += 1

    daily_p_ds = xr.Dataset({"daily_p": (["time", "lat", "lon"], daily_p_array)},
                             coords={"time": all_dates,'latitude' : lat,'longitude' : lon})

    # Add attributes to the dataset
    daily_p_ds.daily_p.attrs['units'] = 'mm'
    daily_p_ds.daily_p.attrs['long_name'] = 'Daily total precipitation'
    
    # Initialize a list to store all dates
    av_fields = []
    std_fields = []

    # Loop through all days of the year
    for day_of_y in tqdm(range(1, 366), desc = 'Analyzing days of the year'):
        day, month = day_of_year_to_date(day_of_y) # Compute day and month from day of the year (1-365)

        # Compute datelist to compute daily SPSS (30 days mooving window)
        datelist = []
        for year in range(start_year, end_year + 1):
            base_date = pd.to_datetime(f'{year}-{month}-{day}')
            for offset in range(-15, 15):  # -15 to 15 days offset
                date = base_date + timedelta(days=offset)
                datelist.append(pd.to_datetime(date)) # Create datelist to compute precipitation statistics

        # Create the array with the precipitation field to compute mean and sd
        fields_array = []
        for day in datelist:
            fields_array.append(daily_p_ds.sel(time= day,method='nearest').daily_p.values)

        # Compute mean and std field of precipitation
        av =  np.nanmean(fields_array, axis=0)
        std = np.nanstd(fields_array, axis=0)

        av_fields.append(av)
        std_fields.append(std)

    # Create an xarray dataset with the results
    ds = xr.Dataset(
        {
            'av_field': (('day_of_year', 'lat', 'lon'), np.array(av_fields)),
            'std_field': (('day_of_year', 'lat', 'lon'), np.array(std_fields))
        },
        coords={
            'day_of_year': np.arange(1, 366),
            'latitude': daily_p_ds.lat,
            'longitude': daily_p_ds.lon
        }
    )


    ds.to_netcdf('input/seasonal_precipitation_statistics.nc') # Save the dataset to a NetCDF file


# %% WTs computation
""" Functions to compute Weather Types using Beck classification (Beck, 2000)"""
import pandas as pd
import xarray as xr
import numpy as np
from scipy.stats import pearsonr

def create_idealized_matrix(x_len, y_len):
    """
    Creation of the idealized flow matrixes to compute correlation with the GPT field
    """

    # Idealized zonal flow
    id_zonal =  np.zeros((x_len, y_len))
    for i in range(x_len):
        id_zonal[i] = i + 1

    # Idealized meridional flow
    id_meridional = np.zeros((x_len, y_len))
    for j in range(y_len):
        id_meridional[:, j] = j + 1
    pd.DataFrame(id_meridional)

    # Low pressure system
    center_x = x_len // 2 # Compute center of the matrix along x dir
    center_y = y_len // 2 # Compute center of the matrix along y dir

    id_low = np.zeros((x_len, y_len))
    for i in range(x_len): # Compute Euclidean distance
        for j in range(y_len):
            distance = np.sqrt((i - center_x)**2 + (j - center_y)**2)
            id_low[i, j] = distance

    return id_zonal, id_meridional, id_low

def compute_correlation(field, zonal, meridional, low):
    """
    Compute of Pearson spatial correlation coefficient
    """

    # Flatten the input arrays to 1D for correlation computation
    field_flat = field.flatten()
    zonal_flat = zonal.flatten()
    meridional_flat = meridional.flatten()
    low_flat = low.flatten()

    # Compute the Pearson correlation coefficients
    rho_zonal, _ = pearsonr(field_flat, zonal_flat)
    rho_meridional, _ = pearsonr(field_flat, meridional_flat)
    rho_low, _ = pearsonr(field_flat, low_flat)

    return rho_zonal, rho_meridional, rho_low

def compute_wt(field):
    """
    Computation of WT using Beck classification (Beck, 2000)
    
    INPUT: field -> GPT field (matrix)
    OUTPUT: synoptic class string

    """

    x_len = field.shape[0]
    y_len = field.shape[1]
    
    zonal, meridional, low = create_idealized_matrix(x_len, y_len) # Create matrices for idealized flows
    rho_zonal, rho_meridional, rho_low = compute_correlation(field, zonal, meridional, low) # Compute correlation parameters for the actual GPT field with the 3 idealized flows

    ### Definition of the synoptic class ###

    # Pure cyclonic flow:
    if (np.abs(rho_low) > np.abs(rho_zonal)) and (np.abs(rho_low) > np.abs(rho_meridional)):
        if rho_low > 0:
            syn_class = 'C'
        else:
            syn_class = 'A'

    # Directional flows
    else:
        a_vec = [1, 0.7, 0.7, 0, -0.7, -1, -0.7, 0]
        b_vec = [0, 0.7, -0.7, -1, -0.7, 0, 0.7, 1]

        classes = ['W', 'SW', 'NW', 'N', 'NE', 'E', 'SE', 'S']

        D = []

        for aa, bb in zip(a_vec, b_vec):
            euc_dist = np.sqrt((aa - rho_zonal)**2 + (bb - rho_meridional)**2)
            D.append(euc_dist)
        df = pd.DataFrame({'D': D, 'Classes': classes})
        # Extract the direction of the flow with the minimal Euclidean distance from the idealized flows
        dir_class = df.loc[df['D'].idxmin(), 'Classes']
        
        # Analyze flow cyclonality
        if rho_low > 0:
            syn_class = dir_class + 'C'
        else:
            syn_class = dir_class + 'A'

    return syn_class

# %% Compute Analogs

from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr
import pdb

def compute_score(z500_ref, z1000_ref, z500_an, z1000_an):
    """
    Computes the similarity between reference and analog geopotential fields 
    at two 500 and 1000 hPa, combining Euclidean distance and Pearson correlation.

    Input:
    - z500_ref: np.array, geopotential height field at 500 hPa for the reference case.
    - z1000_ref: np.array, geopotential height field at 1000 hPa for the reference case.
    - z500_an: np.array, geopotential height field at 500 hPa for the analog case.
    - z1000_an: np.array, geopotential height field at 1000 hPa for the analog case.

    Output:
    - euclidean_distance_500: float, Euclidean distance between the reference and analog fields at 500 hPa.
    - euclidean_distance_1000: float, Euclidean distance between the reference and analog fields at 1000 hPa.
    - pearson_distance_500: float, Pearson distance (1 - correlation) between the reference and analog fields at 500 hPa.
    - pearson_distance_1000: float, Pearson distance (1 - correlation) between the reference and analog fields at 1000 hPa.
    """
    # Calculate Euclidean distances
    euclidean_distance_500 = euclidean(z500_ref.flatten(), z500_an.flatten())
    euclidean_distance_1000 = euclidean(z1000_ref.flatten(), z1000_an.flatten())
    
    # Calculate Pearson correlations
    pearson_corr_500, _ = pearsonr(z500_ref.flatten(), z500_an.flatten())
    pearson_corr_1000, _ = pearsonr(z1000_ref.flatten(), z1000_an.flatten())
    
    # Calculate Pearson distances (1 - correlation)
    pearson_distance_500 = 1 - pearson_corr_500
    pearson_distance_1000 = 1 - pearson_corr_1000
    
    # Calculate the score
    return euclidean_distance_500,  euclidean_distance_1000, pearson_distance_500, pearson_distance_1000
    

# %% Plot of precipitation field
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from datetime import datetime

def plot_precip_field(ds, savepath, title):

    # Number of hours (assuming 24 time steps)
    n_hours = 24

    # Set up the figure and axes for subplots
    fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(20, 15), subplot_kw={'projection': ccrs.PlateCarree()})
    axes = axes.flatten()  # Flatten for easy indexing

    # Define custom levels and colors
    levels = [0.1, 0.2, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20, 24, 30, 40, 50, 60, 80, 100, 125, 150]
    colors = [
        '#FFFFFF', '#E0FFFF', '#B0E0E6', '#87CEFA', '#4682B4', '#0000FF', '#00008B', '#32CD32', '#228B22', '#006400', '#FFD700', 
        '#FFC300', '#FFA500', '#FF8C00', '#FF4500', '#FF6347', '#FF0000', '#CD5C5C', '#8B0000', '#800080', '#A020F0',
        '#DA70D6', '#FF00FF', '#FFB6C1', '#D3D3D3', '#A9A9A9',
    ]

    # Create a custom colormap
    cmap = plt.matplotlib.colors.ListedColormap(colors)

    # Loop over each hour to plot each field
    for i in range(n_hours):
        ax = axes[i]
        
        # Plot the filled contour (contourf) and contour lines
        cf = ds.isel(valid_time=i).plot.contourf(ax=ax, transform=ccrs.PlateCarree(), levels=levels, cmap=cmap, add_colorbar=False, extend='max')
        ds.isel(valid_time=i).plot.contour(ax=ax, transform=ccrs.PlateCarree(), levels=levels, colors='white', linewidths=0.5)

        # Add coastlines and borders
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=':')

        # Add a title to each subplot for the respective hour
        ax.set_title(f"+{i+1:03}")

    # Add space between subplots
    plt.tight_layout()
    fig.suptitle(title, fontsize=15)

    # Add a single colorbar at the bottom of the figure
    cbar_ax = fig.add_axes([0.2, 0.02, 0.6, 0.02])  # [left, bottom, width, height]
    cbar = fig.colorbar(cf, cax=cbar_ax, orientation='horizontal', label='Total Precipitation (mm)')

    plt.savefig(savepath)

# %% plot GPT fields
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap

def plot_gpt(field1, field2, lon, lat, title1, title2, vmin, vmax, savepath):
    """
    Plot two geopotential fields side by side using both contourf and contour, with borders and coastlines.
    
    Parameters:
    - field1: np.array, first geopotential field (2D array)
    - field2: np.array, second geopotential field (2D array)
    - lon: np.array, longitudes (1D array)
    - lat: np.array, latitudes (1D array)
    - title1: str, title for the first subplot
    - title2: str, title for the second subplot
    - vmin: float, minimum value for contour levels
    - vmax: float, maximum value for contour levels
    - savepath: str, path to save the resulting figure
    """
    
    # Fixed levels from vmin to vmax
    levels = np.linspace(vmin, vmax, 60)

    # Custom colormap based on the image colors
    colors = [
        (0.64, 0, 0.71),   # Purple
        (0.14, 0, 0.51),   # Dark blue
        (0, 0, 1),         # Blue
        (0, 1, 1),         # Cyan
        (0, 1, 0),         # Green
        (1, 1, 0),         # Yellow
        (1, 0.65, 0),      # Orange
        (1, 0, 0),         # Red
    ]
    
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=100)

    # Create figure and subplots with Cartopy projection
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), subplot_kw={'projection': ccrs.PlateCarree()}, constrained_layout=True)
    
    # First subplot (Reference field)
    ax1 = axes[0]
    ax1.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())
    ax1.add_feature(cfeature.COASTLINE)
    ax1.add_feature(cfeature.BORDERS, linestyle=':')
    
    # Gridlines with labels only at the bottom and left
    gl1 = ax1.gridlines(draw_labels=True)
    gl1.top_labels = False
    gl1.right_labels = False
    gl1.xlabel_style = {'fontsize': 14}  # Set larger font size for latitude and longitude labels
    gl1.ylabel_style = {'fontsize': 14}

    cf1 = ax1.contourf(lon, lat, field1, levels=levels, cmap=cmap, transform=ccrs.PlateCarree())
    c1 = ax1.contour(lon, lat, field1, levels=levels, colors='black', linewidths=0.5, transform=ccrs.PlateCarree())
    ax1.clabel(c1, inline=True, fontsize=10)
    ax1.set_title(title1, fontsize=18)  # Increase the title font size

    # Second subplot (Analog field)
    ax2 = axes[1]
    ax2.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())
    ax2.add_feature(cfeature.COASTLINE)
    ax2.add_feature(cfeature.BORDERS, linestyle=':')
    
    # Gridlines with labels only at the bottom and left
    gl2 = ax2.gridlines(draw_labels=True)
    gl2.top_labels = False
    gl2.right_labels = False
    gl2.left_labels = False  # Only show labels on the left for the first plot
    gl2.xlabel_style = {'fontsize': 14}

    cf2 = ax2.contourf(lon, lat, field2, levels=levels, cmap=cmap, transform=ccrs.PlateCarree())
    c2 = ax2.contour(lon, lat, field2, levels=levels, colors='black', linewidths=0.5, transform=ccrs.PlateCarree())
    ax2.clabel(c2, inline=True, fontsize=10)
    ax2.set_title(title2, fontsize=18)

    # Create a single colorbar for both plots
    cbar = fig.colorbar(cf2, ax=axes, orientation='horizontal', fraction=0.05, pad=0.1, label='Geopotential')
    cbar.ax.tick_params(labelsize=14)  # Set colorbar label size
    
    # Save the figure and close to free memory
    plt.savefig(savepath)
    plt.close(fig)


# %% Computation of analog's precipitation fields
def compute_hourly_era5_ds(date, ds, ref_date, coords):
    """
    Computation of the 24h cumulated precipitation field from ERA5 Reanalysis using 'date' as starting time

    INPUT:
    - date: pandas.Timestamp, the starting day for the calculation.
    - ds: xarray.Dataset, dataset containing precipitation data 

    OUTPUT:
    - tot_precip: float, total 24 cumulated precipitation field calculated in millimeters (mm).
    """

    start_date = date - pd.Timedelta(days=1)
    end_date = date + pd.Timedelta(days=1)

    ds = ds.sel(time=slice(start_date, end_date)) # Extract correct time
    ds_domain = ds.sel(latitude=slice(coords[0], coords[1]), longitude=slice(coords[2], coords[3])) # Extract spatial domain

    data_array_list = []
    for day in range(1, ds_domain.sizes['time']):
        for tstep in range(0, ds_domain.sizes['step']):
            if (day == 1 and tstep in [0, 1, 2, 3, 4, 5]) or (day == ds_domain.sizes['time']-1 and tstep in [6, 7, 8, 9, 10, 11]):  # Skip first values belonging to the day before
                continue
            field_ds = ds_domain.isel(time=day).isel(step=tstep)
            field = field_ds.tp.values*1000 # transform in mm of rainfall
            norm_field = normalize_field(field, date, ref_date)

            valid_time = field_ds.valid_time.values
            start_acc_time = valid_time - np.timedelta64(1, 'h')
            
            ds = xr.DataArray(
                data=norm_field,  # precip field
                dims=['latitude', 'longitude'],
                coords={
                    'latitude': field_ds.latitude.values, 
                    'longitude': field_ds.longitude.values,
                    'an_day': date,  
                    'valid_time': valid_time,
                    'start_acc_time': start_acc_time,
                },
                name = 'tp'
            )
            ds.attrs.update(field_ds.attrs) # Add attributes
            data_array_list.append(ds)

    analog_ds = xr.concat(data_array_list, dim='valid_time')
    analog_ds.attrs.update(field_ds.attrs)
    analog_ds.attrs['units'] = 'mm'  # Add new attribute for units

    return analog_ds

def normalize_field(field, an_date, ref_date):
    """
    Seasonal standardization of the precipitation field based on the average and standard 
    deviation from the 'seasonal_precip_standardization' dataset. This file must be included in the 'input' folder

    Input:
    - field: numpy.array, precipitation field to be normalized
    - an_date: pandas.Timestamp, date of the analog
    - ref_date: pandas.Timestamp, reference date

    Output:
    - norm_field: numpy.array, normalized field according to the average and 
                  standard deviation for the corresponding day of the year.
    """
    
    ds = xr.open_dataset('input/seasonal_precipitation_statistics.nc')
    # Analog day
    n_day = (an_date - pd.Timestamp(an_date.year, 1, 1)).days + 1
    an_av = ds['av_field'].sel(day_of_year=n_day, method='nearest').values
    an_sd = ds['std_field'].sel(day_of_year=n_day, method='nearest').values

    # Reference date
    n_day = (ref_date - pd.Timestamp(ref_date.year, 1, 1)).days + 1
    ref_av = ds['av_field'].sel(day_of_year=n_day, method='nearest').values
    ref_sd = ds['std_field'].sel(day_of_year=n_day, method='nearest').values
    norm_field = ref_sd * ((field - an_av) / an_sd) + ref_av
    norm_field[norm_field < 0] = 0

    return norm_field


# %% Mooving Window module
def is_within_window(candidate_date, ref_date):
    candidate_month_day = (candidate_date.month, candidate_date.day)
    
    # Convert to day of the year for easier comparison
    ref_day_of_year = ref_date.timetuple().tm_yday
    candidate_day_of_year = candidate_date.timetuple().tm_yday

    # Calculate days between 2 months before and 2 months after
    # Handle the year boundaries (e.g., December to January)
    start_window = (ref_day_of_year - 61) % 365
    end_window = (ref_day_of_year + 61) % 365
    
    if start_window <= end_window:
        # Simple case: the window is within the same year
        return start_window <= candidate_day_of_year <= end_window
    else:
        # Window spans across the end of the year (e.g., November to February)
        return candidate_day_of_year >= start_window or candidate_day_of_year <= end_window
