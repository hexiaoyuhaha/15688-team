import pandas as pd
import time
import numpy as np
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


def read_saved_interval_KDE(year, hours_interval=6, block_lat=25, block_lon=25):
    """
    Make sure hours_interval/block_lat/block_lon are the same as those for save_intervals_KDE_to_file
    """
    data = np.loadtxt('data/interval_block_KDE_'+str(year)+'_'+str(block_lat)+'_by_'+str(block_lon)+'.txt')
    total_intervals = 24*365/hours_interval
    
    return data.reshape((total_intervals, block_lat, block_lon))


def get_interval_KDE(dt, block_lat_idx, block_lon_idx, hours_interval=6):
    """
    API function for interval KDE. There are lots of zeros
    """
    year = dt.year
    
    year_start = datetime(year,1,1,0,0,0)
    delta = dt-year_start
    
    hours_interval_idx = delta.days*4 + delta.seconds/(3600*hours_interval)
    
    if year==2013:
        return interval_KDE_2013[hours_interval_idx][block_lat_idx][block_lon_idx]
    elif year==2014:
        return interval_KDE_2014[hours_interval_idx][block_lat_idx][block_lon_idx]
    elif year==2015:
        return interval_KDE_2015[hours_interval_idx][block_lat_idx][block_lon_idx]
    else:
        print 'Year out of range: '+year
        return -1


def read_saved_yearly_KDE(year, block_lat=25, block_lon=25):
    """
    Make sure block_lat/block_lon are the same as those for save_yearly_KDE_to_file
    """
    data = np.loadtxt('data/yearly_KDE_'+str(year)+'_'+str(block_lat)+'_by_'+str(block_lon)+'.txt')
    
    return data.reshape((block_lat, block_lon))


def get_yearly_KDE(year, block_lat_idx, block_lon_idx):
    '''API function for yearly KDE
    '''
    if year==2013:
        return yearly_KDE_2013[block_lat_idx][block_lon_idx]
    elif year==2014:
        return yearly_KDE_2014[block_lat_idx][block_lon_idx]
    elif year==2015:
        return yearly_KDE_2015[block_lat_idx][block_lon_idx]
    else:
        print 'Year out of range: '+year
        return -1


def read_saved_crime_counts(year, hours_interval=6, block_lat=25, block_lon=25):
    """
    Make sure block_lat/block_lon are the same as those for crime_counts
    """
    data = np.loadtxt('data/crime_counts_'+str(year)+'_'+str(block_lat)+'_by_'+str(block_lon)+'.txt')
    total_intervals = 24*365/hours_interval
    
    return data.reshape((total_intervals, block_lat, block_lon))

 
def get_crime_count(dt, block_lat_idx, block_lon_idx, hours_interval=6):
    '''API function for crime count
    '''
    """
    There are lots of zeros
    """
    year = dt.year
    
    year_start = datetime(year,1,1,0,0,0)
    delta = dt-year_start
    
    hours_interval_idx = delta.days*4 + delta.seconds/(3600*hours_interval)
    
    if year==2013:
        return crime_counts_2013[hours_interval_idx][block_lat_idx][block_lon_idx]
    elif year==2014:
        return crime_counts_2014[hours_interval_idx][block_lat_idx][block_lon_idx]
    elif year==2015:
        return crime_counts_2015[hours_interval_idx][block_lat_idx][block_lon_idx]
    else:
        print 'Year out of range: '+year
        return -1


def read_saved_yearly_crime_counts(year, block_lat=100, block_lon=100):
    """
    Make sure block_lat/block_lon are the same as those for crime_counts
    """
    data = np.loadtxt('data/yearly_crime_counts_'+str(year)+'_'+str(block_lat)+'_by_'+str(block_lon)+'.txt')
    
    return data.reshape((block_lat, block_lon))


def get_yearly_crime_count(year, block_lat_idx, block_lon_idx):
    '''API function for crime count
    '''
    if year==2013:
        return yearly_crime_counts_2013[block_lat_idx][block_lon_idx]
    elif year==2014:
        return yearly_crime_counts_2014[block_lat_idx][block_lon_idx]
    elif year==2015:
        return yearly_crime_counts_2015[block_lat_idx][block_lon_idx]
    else:
        print 'Year out of range: '+year
        return -1

        
def get_KDE_block(kernel, nw_corner, se_corner):
    ''' Compute the average of five points
    '''
    (sample_nw, sample_ne, sample_mid, sample_sw, sample_se) = __generate_samples(nw_corner, se_corner)
    
    kde = 0.0
    kde += __read_KDE(kernel, sample_nw[1], sample_nw[0])
    kde += __read_KDE(kernel, sample_ne[1], sample_ne[0])
    kde += __read_KDE(kernel, sample_mid[1], sample_mid[0])
    kde += __read_KDE(kernel, sample_sw[1], sample_sw[0])
    kde += __read_KDE(kernel, sample_se[1], sample_se[0])
    
    return kde/5.0
