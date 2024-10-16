# Main function
def main():
    from datetime import datetime, timedelta
    import os
    import shutil
    import pandas as pd
    from functions import download_era5_gpt, download_era5_precip, compute_era5_wts, compute_spss
    
    ##########################################################################################################################
    """ Setting of the function:
            1. Definition of the Target Day to analyze (Today or previous days)
            2. Definition of the spatial domain: 
                - Synoptic domain: to evaluate analogs based on Geopotential (GPT) fields
                - Mesoscale domain: to evaluate precipitation field (Catalonia region)
            3. Download of ERA5 reanalysis fields:
                - era5_download = true : run of the ERA5 api to download geopotential fields and precipitation fields
                                         for the synoptic and the mesoscale domain respectively
                - era5_download = false : no download"""
    
    ############### Definition of the target date ################
    today = datetime.today() # Get today's date
    date_str = today.strftime('%Y%m%d')
    out_folder = f'output/{date_str}' # Create the output folder
    os.makedirs(out_folder, exist_ok=True)
    
    
    ############## Definition of the spatial domain ##############
    # Synoptic domain: to extract GPT field and compute WTs
    syn_lat_s = 31
    syn_lat_n = 48
    syn_lon_w = -17
    syn_lon_e = 9
    syn_coords = [syn_lat_n, syn_lat_s, syn_lon_w, syn_lon_e]

    # Mesoscale domain: to extract precipitation field
    """
    - cat = (40, 43, 0, 3.5)
    - arga = (42.25, 43.25, -2.75, -0.75)
    - d09 (Weastern Europe) = (31, 48, -17, 9)
    """
    mes_lat_s = 40
    mes_lat_n = 43 
    mes_lon_w = 0
    mes_lon_e = 3.5
    mes_coords = [mes_lat_n, mes_lat_s, mes_lon_w, mes_lon_e]


    ############# Download of ERA5 reanalysis fields ##############
    """ !!! WARNING: Set to True only if you have not downloaded ERA5 data yet, 
            the download requied large amount of time and memory space !!! """
    era5_download = False  # True

    ############# Compute WTs from ERA5 data ##############
    """ Set to True only if you have not computed ERA5 WTs yet, 
            i.e. if you do not have the file 'era5_classification.csv' in the 'input/' folder
            !!! WARNING: every time you change the spatial domain definition WTs must be computed again !!! """
    era5_wts = False # True

    ############# Compute Seasonal Precipitation Standardization Statistics (SPSS) ##############
    """ Set to True only if you have not computed SPSS yet, 
            i.e. if you do not have the file 'seasonal_precipitation_statistics.nc' in the 'input/' folder 
            !!! WARNING:  every time you change the spatial domain definition WTs must be computed again !!! """
    seasonal_standardization = False # True
    
##########################################################################################################################

# %% Analog Algorithm
    """ Running of the modules of the Analog Algorithm
        1. Download of ERA5 reanalysis data: only if era5_download set to True
        2. Compute Weather Types (WTs) for each day from ERA5 Reanalysis data for the whole period 1940-2023
            and save them the csv file era5_classificatiion.csv' in the 'input/' folder
        3. Compute Seasonal Precipitation Standardization Statistics (SPSS) and save the data in the database 
            'seasonal_precip_standardization.nc' in the 'input/' folder
        4. GFS Module: extraction of GPT fields from GFS forecast for the target day;
           Save the predicted 24h cumulated precipitation field
        5. Analog Module: computation of the best 10 analogs
        6. Analog Precipitation Module: save the 24h cumulated precipitation fields for the best 10 analogs
        7. Delate GFS files of the target day (today)"""
        
    # 1. Downaload of ERA5 Reanalysis fields
    if era5_download == True:
        print('Downloading ERA5 reanalysis data:')
        download_era5_gpt(level = "500", coords=[syn_lat_n,syn_lon_w,syn_lat_s,syn_lon_e]) # Download 500 hPa gpt ds
        download_era5_gpt(level = "1000", coords=[syn_lat_n,syn_lon_w,syn_lat_s,syn_lon_e]) # Download 1000 hPa gpt ds
        download_era5_precip(coords=[syn_lat_n,syn_lon_w,syn_lat_s,syn_lon_e]) # Download precipitation ds
        print('\n')

    # 2. Compute WTs from ERA5 data
    if era5_wts == True:
        compute_era5_wts()

    # 3. Compute SSPS
    if seasonal_standardization == True:
        compute_spss(mes_coords)

    # 4. Call GFS Module
    print('Running GFS Module:')
    z500, z1000, z500_wt, z1000_wt = gfs_module(today, syn_coords, mes_coords)
    print('\n')

    # 5. Call Analog Computation Module
    print('Running Analog Module:')
    analog_table = analogs_module(today, z500, z1000, z500_wt, z1000_wt)
    print('\n')

    # 6. Call Analog Precipitation Module
    print('Running Analog Precipitation Module:')
    an_precipitation_module(today, analog_table['Date'], mes_coords)
    print('\n')

    # 7. Delate GFS forecast
    path = 'input/gfs_files'
    try:
       shutil.rmtree(path)
       print('GFS files correcly delated')
    except OSError as e:
       print(f'Error: {e}')


""" Modules """
# 2. GFS Module
def gfs_module(date, syn_coords, mes_coords):
    """
    GFS module: 
        1. Download of the GFS output files for the 00 UTC run of the model
        2. Extract geopotential (GPT) fields Z500 and Z1000
        3. Compute weather types (WTs) for GPT field at 500 and 1000 hPa
        4. Save NetCDF file of the precipitation field forecasted by GFS, for forecast times from +001h to +024h

    INPUT:
        - Date: today date in datetime format
        - syn_coords: coordinate of the synoptic spatial domain to compute WTs
        - mes_coords: coordinate of the mesoscale domain to extract the precipitation field
    
    OUTPUT: all saved in the folder "output/yyyymmdd/"
        - gfs_forecast.nc -> NetCDF file with the precipitation field forecasted by GFS
        - 24h_precip_field_GFS.png -> Map of the forecasted precipitation field
    """
    # %% 1. Download GFS files
    from datetime import datetime, time
    import os
    import requests
    from tqdm import tqdm

    date_str = date.strftime('%Y%m%d')

    base_url = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.{date_str}/00/atmos/" # Construct the URL

    # Create the 'input' directory if it doesn't exist
    path = 'input/gfs_files'
    if os.path.exists(path):
        print("Most recent GFS forecasts already downloaded")
    else:
        os.makedirs(path, exist_ok=True) # Create dir

        # List of files to download
        for i in tqdm(range(0,25), desc = 'Downloading most recent GFS run', leave = False):
            file_name = f'gfs.t00z.pgrb2.0p25.f{i:03}'

            file_url = f'{base_url}{file_name}'
            response = requests.get(file_url)
                
            # Check if the request was successful
            if response.status_code == 200:
                # Save the file in the 'input' directory
                with open(os.path.join(path, file_name), 'wb') as f:
                    f.write(response.content)
            else:
                print(f'Failed to download: {file_name}, Status code: {response.status_code}')

    # %% 2. Extract GPT fields (Z500, Z100)
    import xarray as xr
    import numpy as np

    # Extract Z500 GPT field
    ds_500 = xr.open_dataset(f'{path}/gfs.t00z.pgrb2.0p25.f000', engine='cfgrib', 
                            backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa', 'level': 500}}) # open dataset
    f1 = ds_500['gh'].sel(latitude=slice(syn_coords[0], syn_coords[1]), longitude=slice(360+syn_coords[2], 360)).values 
    f2 = ds_500['gh'].sel(latitude=slice(syn_coords[0], syn_coords[1]), longitude=slice(0, syn_coords[3])).values 
    z500 = np.concatenate((f1, f2), axis=1)

    # Extract Z1000 GPT field
    ds_1000 = xr.open_dataset(f'{path}/gfs.t00z.pgrb2.0p25.f000', engine='cfgrib', 
                            backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa', 'level': 1000}})
    f1 = ds_1000['gh'].sel(latitude=slice(syn_coords[0], syn_coords[1]), longitude=slice(360+syn_coords[2], 360)).values 
    f2 = ds_1000['gh'].sel(latitude=slice(syn_coords[0], syn_coords[1]), longitude=slice(0, syn_coords[3])).values 
    z1000 = np.concatenate((f1, f2), axis=1)

    # %% 3. WTs classification
    from functions import compute_wt

    z500_wt = compute_wt(z500)
    z1000_wt = compute_wt(z1000)

    print(f'\nComputed WTs for {date.date()}:')
    print(f'Z500: {z500_wt}')
    print(f'Z1000: {z1000_wt}')

    # %% 4. Build and save GFS forecasted +24h precipitation field, with hourly step of accumulation
    from functions import plot_precip_field
    import pandas as pd

    data_array_list = []
    init_time = datetime.combine(date.date(), time(0, 0))
    init_time = np.int32(pd.to_datetime(init_time).timestamp())

    for i in tqdm(range(1,25), desc = 'Computing hourly GFS precipitation field', leave = False):

        ds_precip = xr.open_dataset(f'{path}/gfs.t00z.pgrb2.0p25.f{i:03}', engine='cfgrib', 
                                    filter_by_keys={'stepType': 'accum', 'typeOfLevel': 'surface'})['tp']

        if mes_coords[2] < 0 and mes_coords[3] < 0:
            lon_w = 360 + mes_coords[2]
            lon_e = 360 + mes_coords[3]
            p_field = ds_precip.sel(latitude=slice(mes_coords[0], mes_coords[1]), longitude=slice(lon_w, lon_e))

        elif mes_coords[2] < 0 and mes_coords[3] > 0:
            lon_w = 360 + mes_coords[2]
            p1 = ds_precip.sel(latitude=slice(mes_coords[0], mes_coords[1]), longitude=slice(lon_w, 360))
            p2 = ds_precip.sel(latitude=slice(mes_coords[0], mes_coords[1]), longitude=slice(0, mes_coords[3]))
            p_field = xr.concat([p1, p2], dim='longitude')

        else:
            p_field = ds_precip.sel(latitude=slice(mes_coords[0], mes_coords[1]), longitude=slice(mes_coords[2], mes_coords[3]))

        # Extract times
        valid_time = pd.to_datetime(p_field.valid_time.values).timestamp()
        valid_time = np.int32(valid_time)

        # Compute hourly precipitation field:
        if i in[1, 7, 13, 19]:
            hourly_field = p_field.values
        else: 
            hourly_field = p_field.values - p_field_previous

        # Compute correct values of longitude to save into the gfs_ds
        longitude = np.arange(mes_coords[2], mes_coords[3] + 0.25, 0.25)
        ds = xr.DataArray(
            data=hourly_field,  # precip field
            dims=['latitude', 'longitude'],
            coords={
                'latitude': p_field.latitude.values, 
                'longitude': longitude,  
                'valid_time': valid_time,
                'initialization_time': init_time

            },
            name='tp'
        )
        # ds.attrs.update(ds_precip.attrs) # Add attributes
        ds.attrs['units'] = 'kg m-2'
        ds.attrs['long_name'] = 'Total precipitation'
        ds.attrs['standard_name'] = 'precipitation_amount'
        data_array_list.append(ds)

        p_field_previous = p_field.values # Save the precipitation field to compute hourly field for the next step

    gfs_ds = xr.concat(data_array_list, dim='valid_time')
    gfs_ds.attrs.update(ds_precip.attrs)
    gfs_ds.attrs.update({
        'title': 'GFS hourly Cumulative Precipitation Data',
        'history': f'Created on {pd.Timestamp.now()} from GFS data data',
        'Conventions': 'CF-1.7',
        'institution': 'Universitat de Barcelona, GAMA team',
        'source': 'GFS (NOAA)',
        'references': 'https://www.ncei.noaa.gov/products/weather-climate-models/global-forecast'
    })

    # Fix units and add other attributes for coordinates
    gfs_ds.coords['longitude'].attrs['units'] = 'degrees_east'
    gfs_ds.coords['longitude'].attrs['long_name'] = 'Longitude'
    gfs_ds.coords['longitude'].attrs['standard_name'] = 'longitude'

    gfs_ds.coords['latitude'].attrs['units'] = 'degrees_north'
    gfs_ds.coords['latitude'].attrs['long_name'] = 'Latitude'
    gfs_ds.coords['latitude'].attrs['standard_name'] = 'latitude'

    # Add attributes for the new time variables
    gfs_ds.coords['valid_time'].attrs['long_name'] = 'Prediction Time (seconds since 1970-01-01 00:00:00)'
    gfs_ds.coords['valid_time'].attrs['standard_name'] = 'time'
    gfs_ds.coords['valid_time'].attrs['units'] = 'seconds since 1970-01-01 00:00:00'

    gfs_ds.coords['initialization_time'].attrs['long_name'] = 'Initialization Time of GFS model'
    gfs_ds.coords['initialization_time'].attrs['standard_name'] = 'initialization_time'
    gfs_ds.coords['initialization_time'].attrs['units'] = 'seconds since 1970-01-01 00:00:00'

    # Save p_field as netcdf file
    output_fp = f'output/{date_str}/gfs_forecast.nc'
    gfs_ds.to_netcdf(output_fp)

    # Plot precipitation field of GFS forecast
    p_fold = f'output/{date_str}/precip_field_plot'
    os.makedirs(p_fold, exist_ok=True)

    plot_precip_field(gfs_ds, savepath = f'{p_fold}/gfs.png', title = 'GFS forecast') # plot precipitation field

    # %% Return Z500, Z100, WTs
    return z500, z1000, z500_wt, z1000_wt

# 3. Analogs Computation Module
def analogs_module(ref_date, z500, z1000, z500_wt, z1000_wt):
    """
    Identifies and ranks analog dates based on geopotential heights (z500, z1000) and their respective weather types.
    The module computes scores for each analog candidate and saves the top analogs
    along with plots of their geopotential fields.

    Input:
    - date: datetime object, the target date for which analogs are being sought.
    - z500: np.array, geopotential height at 500 hPa for the target date (2D array).
    - z1000: np.array, geopotential height at 1000 hPa for the target date (2D array).
    - z500_wt: str, the weather type classification at 500 hPa for the target date.
    - z1000_wt: str, the weather type classification at 1000 hPa for the target date.

    Output:
    - DataFrame containing the top 10 analog dates ranked by their computed scores, with columns for the date and score.
    - CSV file saved in the output directory containing the top analogs and their scores.
    - PNG files of the geopotential fields for the top 10 analogs saved in the 'analog_gpt_field' folder.
    """
   
    date_str = ref_date.strftime('%Y%m%d')

    # a: Read WTs classification of ERA5 Reanalysis and subsets analogs candidates based on WTs
    import pandas as pd
    import os
    from datetime import timedelta, datetime
    from functions import is_within_window

    filepath = 'input/era5_classification.csv'
    era5_wt = pd.read_csv(filepath, na_filter=False)
    analog_candidate = era5_wt.loc[(era5_wt['wt_500'] == z500_wt) & ((era5_wt['wt_1000'] == z1000_wt))]['Date'].tolist()
    # Exclude analog belonging to the same year of the reference date
    ref_year = ref_date.year
    # analog_datelist = [date for date in analog_candidate if pd.to_datetime(date).year != ref_year]
    analog_datelist = [date for date in analog_candidate if pd.to_datetime(date).year != ref_year
                    and 1940 <= pd.to_datetime(date).year <= 2023]


    # Mooving window module
    # analog_datelist = [date for date in analog_datelist if is_within_window(pd.to_datetime(date), ref_date)]

    # b: Analog's ranking
    from tqdm import tqdm
    import xarray as xr
    from functions import compute_score, plot_gpt

    # Open ERA5 geopotential bases
    ds_500 = xr.open_dataset('input/ERA5_gpt_ds/geopotential_data_500.grib')
    ds_1000 = xr.open_dataset('input/ERA5_gpt_ds/geopotential_data_1000.grib')

    an_list = []
    for date in tqdm(analog_datelist, total=len(analog_datelist), desc='Computing Analog Table', leave = False):
        z500_an = ds_500.sel(time=pd.to_datetime(date))['z'].values / 9.81  # Extract z500 field
        z1000_an = ds_1000.sel(time=pd.to_datetime(date))['z'].values / 9.81  # Extract z1000 field

        # Compute score
        ed_500, ed_1000, r_500, r_1000 = compute_score(z500, z1000, z500_an, z1000_an)
        an_list.append({
            'Date': date,
            'ed_500': ed_500,
            'ed_1000': ed_1000,
            'r_500': r_500,
            'r_1000': r_1000
        })


    df_analogs = pd.DataFrame(an_list)
    
    # Calculate the mean and standard deviation for each column
    means = df_analogs[['ed_500', 'ed_1000', 'r_500', 'r_1000']].mean()
    stds = df_analogs[['ed_500', 'ed_1000', 'r_500', 'r_1000']].std()

    # Normalize each column by subtracting the mean and dividing by the standard deviation
    df_analogs['ed_500_norm'] = (df_analogs['ed_500'] - means['ed_500']) / stds['ed_500']
    df_analogs['ed_1000_norm'] = (df_analogs['ed_1000'] - means['ed_1000']) / stds['ed_1000']
    df_analogs['r_500_norm'] = (df_analogs['r_500'] - means['r_500']) / stds['r_500']
    df_analogs['r_1000_norm'] = (df_analogs['r_1000'] - means['r_1000']) / stds['r_1000']

    # Calculate the score as the sum of the normalized values
    df_analogs['Score'] = df_analogs['ed_500_norm'] + df_analogs['ed_1000_norm'] + df_analogs['r_500_norm'] + df_analogs['r_1000_norm']

    an_list_sorted = df_analogs.sort_values(by='Score') # Sort the analog list by the score
    top_analogs =  an_list_sorted[['Date', 'Score']].head(10) # Keep the top 10 analogs

    # Save the top analogs in a csv file
    top_analogs.to_csv(f'output/{date_str}/top_analogs.csv', columns=['Date', 'Score'], index=False)

    # Plot GPT fields for best 10 analogs
    idx=0
    for an_date in tqdm(top_analogs['Date'], total=len(top_analogs['Date']), desc='Plotting GPT fields', leave = False) :
        idx+=1
        z500_an = ds_500.sel(time=pd.to_datetime(an_date))['z'].values / 9.81  # Extract z500 field
        z1000_an = ds_1000.sel(time=pd.to_datetime(an_date))['z'].values / 9.81  # Extract z1000 field

        fold = f'output/{date_str}/analog_gpt_field'
        os.makedirs(fold, exist_ok=True)

        plot_gpt(z500, z500_an, ds_500.longitude, ds_500.latitude, title1=f'Z500 - Target day: {ref_date.date()}',
                 title2=f'Z500 - Analog {idx} - {an_date}', vmin=4700, vmax=6080, savepath=f'{fold}/z500_analog_{idx}.png')
        plot_gpt(z1000, z1000_an, ds_1000.longitude, ds_1000.latitude, title1=f'Z1000 - Target day: {ref_date.date()}', 
                 title2=f'Z1000 - Analog {idx} - {an_date}', vmin=-300, vmax=300, savepath=f'{fold}/z1000_analog_{idx}.png')

    return top_analogs

# 4. Analog Precipitation Module
def an_precipitation_module(today, an_datelist, coords):
    """
    Processes analog precipitation data for specified dates, computes total precipitation,
    normalizes the field, and saves the results as NetCDF files. Additionally, it plots
    the precipitation fields for each analog date.

    Input:
    - today: str, the current date in 'YYYY-MM-DD' format used as a reference date
    - an_datelist: list of str, list of analog dates in 'YYYY-MM-DD' format to process.

    Output:
    - NetCDF files containing the normalized hourly cumulated precipitation from time +000 to +024, for each analog date,
      saved in the folder 'analog_nc'

    """

    from tqdm import tqdm
    import pandas as pd
    import xarray as xr
    import os
    import numpy as np
    from functions import plot_precip_field, compute_hourly_era5_ds

    ref_date = pd.to_datetime(today).normalize() # trasform today in date format (ref_date)
    date_str = ref_date.strftime('%Y%m%d')
    idx = 0
    for an_date in tqdm(an_datelist, total=len(an_datelist), desc = 'Saving analog precipitation fields', leave = False):
        idx+=1
        an_date = pd.to_datetime(an_date)
        year = an_date.year # Extract year of the analog date
        filepath = f'input/cat_ERA5_precip_ds/{year}_ds.grib'
        with xr.open_dataset(filepath, engine='cfgrib') as ds:
            analog_ds = compute_hourly_era5_ds(an_date, ds, ref_date, coords)

            # Save as NetCDF file
            folder = f"output/{date_str}/analog_nc"
            os.makedirs(folder, exist_ok = True)
            analog_ds.to_netcdf(f'{folder}/Analog_{idx}.nc', mode = 'w')

            # Plot precipitation fields for each analog
            p_fold = f'output/{date_str}/precip_field_plot'
            os.makedirs(p_fold, exist_ok=True)

            plot_precip_field(analog_ds, savepath = f'{p_fold}/analog_{idx}.png', title = f'Analog {idx} - {an_date.date()}') # plot precipitation field


if __name__ == '__main__':
    main()