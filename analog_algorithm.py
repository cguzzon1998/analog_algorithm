import pdb

# 1. GFS Module
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
        - gfs_forecasted_p_field_0_24h.nc -> NetCDF file with the precipitation field forecasted by GFS
        - 24h_precip_field_GFS.png -> Map of the forecasted precipitation field
    """
    # %% 1. Download GFS files
    import datetime
    import os
    import datetime
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
        for i in tqdm(range(0,25), desc = 'Downloading most recent GFS run:', leave = False):
            file_name = f'gfs.t00z.pgrb2.0p25.f{i:03}'

            file_url = f'{base_url}{file_name}'
            response = requests.get(file_url)
                
            # Check if the request was successful
            if response.status_code == 200:
                # Save the file in the 'input' directory
                with open(os.path.join(path, file_name), 'wb') as f:
                    f.write(response.content)
                print(f'Downloaded: {file_name}')
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
    from functions import plot_p_field

    data_array_list = []
    for i in tqdm(range(1,25), desc = 'Computing hourly GFS precipitation field', leave = False):

        ds_precip = xr.open_dataset(f'{path}/gfs.t00z.pgrb2.0p25.f{i:03}', engine='cfgrib', 
                                    filter_by_keys={'stepType': 'accum', 'typeOfLevel': 'surface'})['tp']

        p_field = ds_precip.sel(latitude=slice(mes_coords[0], mes_coords[1]), longitude=slice(mes_coords[2], mes_coords[3]))
        
        # plot_p_field(p_field, valid_time, savepath = f'output/{date_str}/{valid_time}.png') # plot precipitation field
        # Extract times
        time = p_field.time.values
        valid_time = p_field.valid_time.values
        step = p_field.step.values
        
        ds = xr.DataArray(
            data=p_field.values,  # precip field
            dims=['latitude', 'longitude'],
            coords={
                'latitude': p_field.latitude.values, 
                'longitude': p_field.longitude.values,  
                'time': time,  
                'step': step, 
                'valid_time': valid_time
            }
        )
        ds.attrs.update(ds_precip.attrs) # Add attributes
        data_array_list.append(ds)


    gfs_ds = xr.concat(data_array_list, dim='time')
    gfs_ds.attrs.update(ds_precip.attrs)


    # Save p_field as netcdf file
    output_fp = f'output/{date_str}/gfs_forecasted_p_field_0_24h.nc'
    p_field.to_netcdf(output_fp)

    # %% Return Z500, Z100, WTs
    return z500, z1000, z500_wt, z1000_wt

# 2. Analogs Computation Module
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

    filepath = 'input/era5_classification.csv'
    era5_wt = pd.read_csv(filepath, na_filter=False)
    analog_candidate = era5_wt.loc[(era5_wt['wt_500'] == z500_wt) & ((era5_wt['wt_1000'] == z1000_wt))]['Date'].tolist()
    # Exclude analog belonging to the same year of the reference date
    ref_year = ref_date.year
    analog_datelist = [date for date in analog_candidate if pd.to_datetime(date).year != ref_year]

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

# 3. Analog Precipitation Module
def an_precipitation_module(today, an_datelist):
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
    from functions import plot_p_field, compute_hourly_era5_ds

    ref_date = pd.to_datetime(today) # trasform today in date format (ref_date)
    idx = 0
    for an_date in tqdm(an_datelist, total=len(an_datelist), desc = 'Saving analog precipitation fields', leave = False):
        idx+=1
        an_date = pd.to_datetime(an_date)
        year = an_date.year # Extract year of the analog date
        filepath = f'input/ERA5_precip_ds/{year}_ds.grib'
        with xr.open_dataset(filepath, engine='cfgrib') as ds:
            analog_ds = compute_hourly_era5_ds(an_date, ds, ref_date)

            # Save as NetCDF file
            folder = f"output/{ref_date.strftime('%Y%m%d')}/analog_nc"
            os.makedirs(folder, exist_ok = True)  # Specify your output file path
            analog_ds.to_netcdf(f'{folder}/Analog_{idx}.nc')


def main():
    from datetime import datetime, timedelta
    import os
    import shutil
    import pandas as pd
    from functions import download_era5_gpt, download_era5_precip
    
    """ Setting of the function:
            1. Definition of the Target Day to analyze (Today or previous days)
            2. Definition of the spatial domain: 
                - Synoptic domain: to evaluate analogs based on Geopotential (GPT) fields
                - Mesoscale domain: to evaluate precipitation field (Catalonia region)
            3. Download of ERA5 data:
                - era5_download = true : run of the ERA5 api to download geopotential fields and precipitation fields
                                         for the synoptic and the mesoscale domain respectively
                - era5_download = false : no download"""
            

    ############### Definition of the target date ################
    today = datetime.today() # Get today's date
    date_str = today.strftime('%Y%m%d')
    out_folder = f'output/{date_str}' # Create the output folder
    os.makedirs(out_folder, exist_ok=True)
    ##############################################################


    ############## Definition of the spatial domain ##############
    # Synoptic domain: to extract GPT field and compute WTs
    syn_lat_s = 31
    syn_lat_n = 48
    syn_lon_w = -17
    syn_lon_e = 9
    syn_coords = [syn_lat_n, syn_lat_s, syn_lon_w, syn_lon_e]

    # Mesoscale domain: to extract precipitation field
    mes_lat_s = 40
    mes_lat_n = 43
    mes_lon_w = 0
    mes_lon_e = 3.5
    mes_coords = [mes_lat_n, mes_lat_s, mes_lon_w, mes_lon_e]
    ###############################################################

    ############# Download of ERA5 reanalysis fields ##############
    """ !!! WARNING: Set to True only if you have not downloaded ERA5 data yet, 
        the download requied large amount of time and memory space !!! """
    era5_download = False  # True
    ###############################################################

# %% Analog Algorithm
    """ Running of the modules of the Analog Algorithm
        1. Download of ERA5 reanalysis data: only if era5_download set to True
        2. GFS Module: extraction of GPT fields from GFS forecast for the target day;
           Save the predicted 24h cumulated precipitation field
        3. Analog Module: computation of the best 10 analogs
        4. Analog Precipitation Module: save the 24h cumulated precipitation fields for the best 10 analogs"""
    
    # 1. Downaload of ERA5 Reanalysis fields
    if era5_download == True:
        print('Downloading ERA5 reanalysis data:')
        download_era5_gpt(level = "500", coords=[syn_lat_n,syn_lon_w,syn_lat_s,syn_lon_e]) # Download 500 hPa gpt ds
        download_era5_gpt(level = "1000", coords=[syn_lat_n,syn_lon_w,syn_lat_s,syn_lon_e]) # Download 1000 hPa gpt ds
        download_era5_precip(coords=[mes_lat_n,mes_lon_w,mes_lat_s,mes_lon_e]) # Download precipitation ds
        print('\n')

    # 2. Call GFS Module
    print('Running GFS Module:')
    z500, z1000, z500_wt, z1000_wt = gfs_module(today, syn_coords, mes_coords)
    print('\n')

    # 3. Call Analog Computation Module
    print('Running Analog Module:')
    analog_table = analogs_module(today, z500, z1000, z500_wt, z1000_wt)
    print('\n')

    # 4. Call Analog Precipitation Module
    print('Running Analog Precipitation Module:')
    an_precipitation_module(today, analog_table['Date'])
    print('\n')

    # Delate GFS forecast
    path = 'input/gfs_files'
    try:
        shutil.rmtree(path)
        print('GFS files correcly delated')
    except OSError as e:
        print(f'Error: {e}')


if __name__ == '__main__':
    main()