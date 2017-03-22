import pandas as pd
import time
import os
import numpy as np
from datetime import datetime
from datetime import timedelta
from scipy import stats

chicago_bounding_box = None

##########################
# Helper functions start #
##########################
def __read_KDE(kernel, latitude, longitude):
    return kernel([[latitude], [longitude]])[0]

def __datetime_to_kernel(dt, hours_interval=6):
    year = dt.year
    
    year_start = datetime(year,1,1,0,0,0)
    delta = dt-year_start
    
    hours_interval_idx = delta.days*24/hours_interval + delta.seconds/(3600*hours_interval)
        
    return __interval_idx_to_kernel(hours_interval_idx, year)

def __interval_idx_to_kernel(idx, year):
    if year==2013:
        kernel = kernel_2013_intervals[idx]
    elif year==2014:
        kernel = kernel_2014_intervals[idx]
    elif year==2015:
        kernel = kernel_2015_intervals[idx]
    else:
        raise ValueError('Passed year out of time range, '+str(year))
        
    return kernel

def __year_to_kernel(year):
    if year==2013:
        kernel = kernel_2013
    elif year==2014:
        kernel = kernel_2014
    elif year==2015:
        kernel = kernel_2015
    else:
        raise ValueError('Passed year out of time range, '+str(year))
        
    return kernel

def __block_idx_to_bounding_box(block_latitude_idx, block_longitude_idx, latitude_blocks_num, longitude_blocks_num):
    """
    return northwest and southeast bounding box coordinates
    """
    latitude_step = (chicago_bounding_box[1][0] - chicago_bounding_box[0][0])/latitude_blocks_num
    longitude_step = (chicago_bounding_box[1][1] - chicago_bounding_box[0][1])/longitude_blocks_num
    
    nw_latitude = chicago_bounding_box[0][0] + block_latitude_idx*latitude_step
    nw_longitude = chicago_bounding_box[0][1] + block_longitude_idx*longitude_step
    
    se_latitude = nw_latitude+latitude_step
    se_longitude = nw_longitude+longitude_step
    
    return ([nw_latitude, nw_longitude], [se_latitude, se_longitude])

def __generate_samples(nw_corner, se_corner):
    latitude_diff_fourth = (se_corner[0] - nw_corner[0])/4
    longtitude_diff_fourth = (se_corner[1] - nw_corner[1])/4
    
    sample_nw = [nw_corner[0]+latitude_diff_fourth, nw_corner[1]+longtitude_diff_fourth]
    sample_ne = [se_corner[0]-latitude_diff_fourth, sample_nw[1]]
    sample_mid = [sample_nw[0]+latitude_diff_fourth, sample_nw[1]+longtitude_diff_fourth]
    sample_sw = [sample_nw[0], se_corner[1]-longtitude_diff_fourth]
    sample_se = [sample_mid[0]+latitude_diff_fourth, sample_mid[1]+longtitude_diff_fourth]
    
    return (sample_nw, sample_ne, sample_mid, sample_sw, sample_se)

########################
# Helper functions end #
########################

    
def save_chicago_crime_data(year, offset=0):
    '''Collect Chicago crime data and save to disk
    '''
    limit = 1000
    while True:
        query = ("https://data.cityofchicago.org/resource/6zsd-86xi.json?year="+year+"&$limit=1000&$offset="+str(offset))
        df = pd.read_json(query)
        if df.empty:
            break
        
        df.to_csv('data/'+year+'/chicago_crime_data_'+str(offset)+'_'+str(offset+limit-1)+'.csv', sep=',')
        offset += limit
        print offset
        
        time.sleep(1)
    
    
def merge_sort_and_remove_duplicate(year):
    ''' Read saved raw data, then do sort, remote duplicate and drop data without 
        latitude/longitude information
    '''
    path = 'data/'+str(year)+'/'
    filelist = os.listdir(path)
    dfs = []
    for f in filelist:
        df = pd.read_csv(path+os.path.basename(f))
        
        # remove rows without latitude/longitude
        df = df[np.isfinite(df['latitude'])]
        dfs.append(df)
        
    df_total = reduce(lambda df1,df2: df1.append(df2, ignore_index=True), dfs)
    
    # sort by date first, then by case_number
    df_total.sort_values(by=['date', 'case_number'], inplace=True)
    
    # remove duplicate case nu
    df_total.drop_duplicates(subset=['case_number'], inplace=True)
    df_total.to_csv('data/chicago_crime_data_'+str(year)+'_sorted.csv')
    
def split_dataframe(df, year, hours_interval=6):
    '''
    Split yearly data frame into a list of hours interval dataframes
    '''
    year = int(year)
    
    delta = timedelta(hours=hours_interval)
    
    start_datetime = datetime(year, 1, 1, 0, 0, 0)
    end_datetime = datetime(year, 1, 1, hours_interval, 0, 0)
    
    next_year_start = datetime(year+1, 1, 1, 0, 0, 0)
    
    split_dfs = []
    while end_datetime<=next_year_start:
        b1 = df['date']>=start_datetime
        b2 = df['date']<end_datetime
        
        split_dfs.append(df[b1&b2])
        
        start_datetime += delta
        end_datetime += delta
        
    return split_dfs
    
def compute_Gaussian_KDE(data1, data2):
    '''Compute Gaussian KDE
    '''
    values = np.vstack([data1, data2])
    return stats.gaussian_kde(values)
    
#################################################################
# Funnctions get KDE value from pre-computed KDE kernerls start #
#################################################################
def get_KDE_block_dt_block(dt, block_latitude_idx, block_longitude_idx, latitude_blocks_num, longitude_blocks_num, hours_interval=6):
    """
    dt is python datetime.datetime object
    blocks start from nw corner to east as row direction, in row major order
    """
    kernel = __datetime_to_kernel(dt, hours_interval)
    bounding_box = __block_idx_to_bounding_box(block_latitude_idx, block_longitude_idx, latitude_blocks_num, longitude_blocks_num)
    
    return get_KDE_block(kernel, bounding_box[0], bounding_box[1])

def get_KDE_block_dt_corner(dt, nw_corner, se_corner, hours_interval=6):
    """
    dt is python datetime.datetime object
    nw_corner and se_corner are [lat, lon] array
    """
    kernel = __datetime_to_kernel(dt, hours_interval)
    return get_KDE_block(kernel, nw_corner, se_corner)

def get_KDE_block_idx_block(hours_interval_idx, year, block_latitude_idx, block_longitude_idx, latitude_blocks_num, longitude_blocks_num):
    bounding_box = __block_idx_to_bounding_box(block_latitude_idx, block_longitude_idx, latitude_blocks_num, longitude_blocks_num)
    
    kernel = __interval_idx_to_kernel(hours_interval_idx, year)
    return get_KDE_block(kernel, bounding_box[0], bounding_box[1])

def get_KDE_block_idx_corner(hours_interval_idx, year, nw_corner, se_corner):
    kernel = __interval_idx_to_kernel(hours_interval_idx, year)
    return get_KDE_block(kernel, nw_corner, se_corner)
    
def get_KDE_block_year_block(year, block_latitude_idx, block_longitude_idx, latitude_blocks_num, longitude_blocks_num):
    """
    year is four digits year number
    """
    bounding_box = __block_idx_to_bounding_box(block_latitude_idx, block_longitude_idx, latitude_blocks_num, longitude_blocks_num)
    kernel = __year_to_kernel(year)
    return get_KDE_block(kernel, bounding_box[0], bounding_box[1])
    
def get_KDE_block_year_corner(year, nw_corner, se_corner):
    kernel = __year_to_kernel(year)
    return get_KDE_block(kernel, nw_corner, se_corner)

def get_KDE_block(kernel, nw_corner, se_corner):
    (sample_nw, sample_ne, sample_mid, sample_sw, sample_se) = __generate_samples(nw_corner, se_corner)
    
    kde = 0.0
    kde += __read_KDE(kernel, sample_nw[1], sample_nw[0])
    kde += __read_KDE(kernel, sample_ne[1], sample_ne[0])
    kde += __read_KDE(kernel, sample_mid[1], sample_mid[0])
    kde += __read_KDE(kernel, sample_sw[1], sample_sw[0])
    kde += __read_KDE(kernel, sample_se[1], sample_se[0])
    
    return kde/5.0
    
###############################################################
# Funnctions get KDE value from pre-computed KDE kernerls end #
###############################################################

def save_intervals_KDE_to_file(year, hours_interval=6, block_lat=100, block_lon=100):
    ''' Save pre-computed hours interval KDE values into files
    '''
    total_intervals = 365*24/hours_interval
    
    res = np.zeros((total_intervals, block_lat, block_lon))
    
    for t_idx in range(total_intervals):
        print t_idx
        
        for lat_idx in range(block_lat):
            for lon_idx in range(block_lon):
                res[t_idx][lat_idx][lon_idx] = get_KDE_block_idx_block(t_idx, year, lat_idx, lon_idx, block_lat, block_lon)
                
    with file('data/interval_block_KDE_'+str(year)+'_'+str(block_lat)+'_by_'+str(block_lon)+'.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(res.shape))
        for data_slice in res:
            np.savetxt(outfile, data_slice)
            outfile.write('# New slice\n')
            
    return res
    
def save_year_KDE_to_file(year, block_lat=100, block_lon=100):
    ''' Save pre-computed yearly KDE values into files
    '''
    res = np.zeros((block_lat, block_lon))
        
    for lat_idx in range(block_lat):
        print lat_idx
        
        for lon_idx in range(block_lon):
            res[lat_idx][lon_idx] = get_KDE_block_year_block(year, lat_idx, lon_idx, block_lat, block_lon)
                
    with file('data/yearly_KDE_'+str(year)+'_'+str(block_lat)+'_by_'+str(block_lon)+'.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(res.shape))
        for data_slice in res:
            np.savetxt(outfile, data_slice)
            outfile.write('# New slice\n')
            
    return res
    

def crime_counts(dfs, year, block_lat=100, block_lon=100):
    '''
    Compute crime counts of each block in each hours interval, and save pre-computed data into file
    '''
    res = np.zeros((len(dfs), block_lat, block_lon))
    
    start_lat = chicago_bounding_box[0][0]
    start_lon = chicago_bounding_box[0][1]
    
    end_lat = chicago_bounding_box[1][0]
    end_lon = chicago_bounding_box[1][1]
    
    lat_len = (start_lat - end_lat)/block_lat
    lon_len = (start_lon - end_lon)/block_lon
    
    for i,df in enumerate(dfs):
        for r,row in df.iterrows():
            lat = row['latitude']
            lon = row['longitude']
            
            lat_idx = np.int((start_lat - lat)/lat_len)
            lon_idx = np.int((start_lon - lon)/lon_len)
            
            if lat_idx>=block_lat or lon_idx>=block_lon:
                # there is one spot (36.619446395 -91.686565684) that is out of Chicago, seems it's a parking lot
                continue
            
            res[i][lat_idx][lon_idx] += 1
            
    with file('data/crime_counts_'+str(year)+'_'+str(block_lat)+'_by_'+str(block_lon)+'.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(res.shape))
        for data_slice in res:
            np.savetxt(outfile, data_slice)
            outfile.write('# New slice\n')
            
    return res
    

def crime_counts_yearly(df, year, block_lat=100, block_lon=100):
    '''
    Compute yearly crime counts of each block, and save pre-computed data into file
    '''
    res = np.zeros((block_lat, block_lon))
    
    start_lat = chicago_bounding_box[0][0]
    start_lon = chicago_bounding_box[0][1]
    
    end_lat = chicago_bounding_box[1][0]
    end_lon = chicago_bounding_box[1][1]
    
    lat_len = (start_lat - end_lat)/block_lat
    lon_len = (start_lon - end_lon)/block_lon
    
    for r,row in df.iterrows():
        lat = row['latitude']
        lon = row['longitude']

        lat_idx = np.int((start_lat - lat)/lat_len)
        lon_idx = np.int((start_lon - lon)/lon_len)

        if lat_idx>=block_lat or lon_idx>=block_lon:
            # there is one spot (36.619446395 -91.686565684) that is out of Chicago, seems it's a parking lot
            continue

        res[lat_idx][lon_idx] += 1
            
    with file('data/yearly_crime_counts_'+str(year)+'_'+str(block_lat)+'_by_'+str(block_lon)+'.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(res.shape))
        for data_slice in res:
            np.savetxt(outfile, data_slice)
            outfile.write('# New slice\n')
            
    return res
    

if __name__ == "__main__":
    global chicago_bounding_box
    chicago_bounding_box = [[42.025339, -87.950502], [41.633514, -87.515073]]
    
    # Collecting raw crime data
    save_chicago_crime_data('2013', offset=0)
    save_chicago_crime_data('2014', offset=0)
    save_chicago_crime_data('2015', offset=0)
    
    # Process raw data and save processed data
    merge_sort_and_remove_duplicate(2013)
    merge_sort_and_remove_duplicate(2014)
    merge_sort_and_remove_duplicate(2015)
    
    # Read processed data into data frame
    df_2013 = pd.read_csv('data/chicago_crime_data_2013_sorted.csv')
    df_2014 = pd.read_csv('data/chicago_crime_data_2014_sorted.csv')
    df_2015 = pd.read_csv('data/chicago_crime_data_2015_sorted.csv')

    # Set the type of date column to datetime
    df_2013['date'] = pd.to_datetime(df_2013['date'])
    df_2014['date'] = pd.to_datetime(df_2014['date'])
    df_2015['date'] = pd.to_datetime(df_2015['date'])
    
    # Have splitted hours interval dataframes lists
    df_2013_split = split_dataframe(df_2013, 2013)
    df_2014_split = split_dataframe(df_2014, 2014)
    df_2015_split = split_dataframe(df_2015, 2015)
    
    ###############Compute KDEs####################
    
    # Compute yearly KDEs
    kernel_2013 = compute_Gaussian_KDE(df_2013['longitude'], df_2013['latitude'])
    kernel_2014 = compute_Gaussian_KDE(df_2014['longitude'], df_2014['latitude'])
    kernel_2015 = compute_Gaussian_KDE(df_2015['longitude'], df_2015['latitude'])
    
    # Compute hours interval KDEs
    kernel_2013_intervals = [compute_Gaussian_KDE(df['longitude'], df['latitude']) for df in df_2013_split]
    kernel_2014_intervals = [compute_Gaussian_KDE(df['longitude'], df['latitude']) for df in df_2014_split]
    kernel_2015_intervals = [compute_Gaussian_KDE(df['longitude'], df['latitude']) for df in df_2015_split]
    
    # Save KDE values for each block and time period
    save_year_KDE_to_file(2013, block_lat=25, block_lon=25)
    save_year_KDE_to_file(2014, block_lat=25, block_lon=25)
    save_year_KDE_to_file(2015, block_lat=25, block_lon=25)
    
    save_intervals_KDE_to_file(2013, block_lat=25, block_lon=25)
    save_intervals_KDE_to_file(2014, block_lat=25, block_lon=25)
    save_intervals_KDE_to_file(2015, block_lat=25, block_lon=25)
    
    ##################Compute crime counts############
    crime_counts_yearly(df_2013, 2013, block_lat=25, block_lon=25)
    crime_counts_yearly(df_2014, 2014, block_lat=25, block_lon=25)
    crime_counts_yearly(df_2015, 2015, block_lat=25, block_lon=25)

    crime_counts(df_2013_split, 2013, block_lat=25, block_lon=25)
    crime_counts(df_2014_split, 2014, block_lat=25, block_lon=25)
    crime_counts(df_2015_split, 2015, block_lat=25, block_lon=25)