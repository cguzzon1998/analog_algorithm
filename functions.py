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

def plot_p_field(p_field, date_str, savepath):
    """
    Function to plot the +24h cumulated precipitation field forecasted by the GFS for the current date analyzed.
    
    INPUT: p_field -> xr.DataArray of the forecasted precipitation field; date_str -> Current day analyzed
    OUTPUT: 24h_precip_field_GFS.png, saved in the folder output
    """

    # Define custom levels and colors
    levels = [0, 0.1, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 125, 150, 200, 300]
    colors = [
        '#FFFFFF', '#E0FFFF', '#B0E0E6', '#87CEFA', '#4682B4', '#0000FF', '#00008B', '#32CD32', '#228B22', '#006400',
        '#FFD700', '#FFC300', '#FFA500', '#FF8C00', '#FF4500', '#FF6347', '#FF0000', '#CD5C5C', '#8B0000', '#800080',
        '#A020F0', '#DA70D6', '#FF00FF', '#FF1493', '#8B008B', '#4B0082',
    ]

    # Main cities to plot (add your own city coordinates)
    cities = {
        'Barcelona': (2.15899, 41.38879 ),
        'Tarragona': (1.25, 41.11667),
        'Lleida': (0.62218, 41.61674 ),
        'Girona': (2.82493, 41.98311 )
    }

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    
    latitudes = p_field['latitude'].values
    longitudes = p_field['longitude'].values

    contour = ax.contourf(longitudes, latitudes, p_field.values, levels=levels[:-1], colors=colors, transform=ccrs.PlateCarree(), extend='both')
    # contour_lines = ax.contour(longitudes, latitudes, p_field.values, levels=levels, colors='black', linewidths=0.5, transform=ccrs.PlateCarree())
    norm = BoundaryNorm(boundaries=levels, ncolors=len(colors)-1)
    
    # Add colorbar
    cbar = plt.colorbar(contour, label='Precipitation (m)', ax=ax, ticks=levels)
    cbar.set_ticks(levels)
    cbar.set_ticklabels([str(i) for i in levels])
    cbar.ax.invert_yaxis()

    # Add labels and title
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Precipitation Field - {datetime.strptime(date_str, "%Y%m%d").date()}')

    # Add coastlines and borders
    ax.coastlines(resolution='10m', color='black', linewidth=1)
    ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1)

    # Plot the cities
    for city, (lon, lat) in cities.items():
        ax.scatter(lon, lat, edgecolor='black', facecolor = 'none', s=30, marker='s', transform=ccrs.PlateCarree())  # Plot city location
        ax.text(lon + 0.03, lat - 0.1, city, fontsize=10, transform=ccrs.PlateCarree())  # Add city name

    # Save the plot
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
def compute_tot_precip(date, ds):
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

    ds_cat = ds.sel(time=slice(start_date, end_date))
    tot_precip = 0

    for day in range(1, ds_cat.sizes['time']):
        for tstep in range(0, ds_cat.sizes['step']):
            if (day == 1 and tstep in [0, 1, 2, 3, 4, 5]) or (day == ds_cat.sizes['time']-1 and tstep in [6, 7, 8, 9, 10, 11]):  # Skip first values belonging to the day before
                continue
            field = ds_cat.isel(time=day).isel(step=tstep).tp.values
            tot_precip += field * 1000  # mm

    return tot_precip


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
    
    ds = xr.open_dataset('input/seasonal_precip_standardization.nc')
    # Analog day
    n_day = (an_date - pd.Timestamp(an_date.year, 1, 1)).days + 1
    an_av = ds['av_field'].sel(day_of_year=n_day, method='nearest').values
    an_sd = ds['std_field'].sel(day_of_year=n_day, method='nearest').values

    # Reference date
    n_day = (ref_date - pd.Timestamp(ref_date.year, 1, 1)).days + 1
    ref_av = ds['av_field'].sel(day_of_year=n_day, method='nearest').values
    ref_sd = ds['std_field'].sel(day_of_year=n_day, method='nearest').values

    norm_field = ref_sd * ((field - an_av) / an_sd) + ref_av

    return norm_field

