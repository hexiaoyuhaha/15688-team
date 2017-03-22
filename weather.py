import pandas as pd
import datetime
import csv

def readDly(data):
    output = []
    for oneline in data:
        id, year, month, element = oneline[:11], oneline[11:15], oneline[15:17], oneline[17:21]
        values = []
        # i+1 = month
        for i in range(31):
            # line length could be 266 and 269
            if 21+8*(i+1) > len(oneline):
                sub = oneline[21+8*i:]
                values.append([sub[:5], ' ', ' ', ' '])
                break;
            sub = oneline[21+8*i: 21+8*(i+1)]
            v = [sub[:5], sub[5], sub[6], sub[7]]
            values.append(v)
        output.append([id, year, month, element, values])
    return output


def get_attribute(attriName):
    outdata = []
    outdata.append(['date', attriName])
    with open('extracted_data/' + fileid + '_' + attriName + '.csv', 'w') as outfile:
        for line in parsed_data:
            if line[3] == attriName:
                numbers = [v[0] for v in line[-1]]
                
                for i, num in enumerate(numbers):
                    if num != '-9999':
                        date = '%s-%s-%02d' % (line[1], line[2], i+1)
                        outdata.append([date, num])
        writer = csv.writer(outfile)
        writer.writerows(outdata)


def load_weather():
    '''
    Load the weather data and return the weather dataframe.

    '''
    # The date is parsed as time series format
    prefix = 'extracted_data/USC00111577_'
    tmax = readTimeSeriesData(prefix, 'TMAX')
    tmin = readTimeSeriesData(prefix, 'TMIN')
    temp = pd.concat([tmax, tmin], join='outer', axis=1)
    tavg = (temp.TMAX + temp.TMIN) / 2.
    
    prcp = readTimeSeriesData(prefix, 'PRCP')
    awdn = readTimeSeriesData(prefix, 'AWND')

    wtfog = readTimeSeriesData(prefix, 'WT01')
    wtrain = readTimeSeriesData(prefix, 'WT16')
    wtsnow = readTimeSeriesData(prefix, 'WT18')
    
    weather = pd.concat([tmax, tmin, tavg, prcp, awdn, wtfog, wtrain, wtsnow], join='outer', axis=1)
    weather = weather.truncate('2013-01-01', '2015-12-31')
    weather[['WT01', 'WT16', 'WT18']] = weather[['WT01', 'WT16', 'WT18']].fillna(value=0)
    return weather 


def readTimeSeriesData(prefix, name):
    '''
    A subfunction called by load_weather
    '''
    data = pd.read_csv(prefix + name + '.csv', index_col='date')
    data.index = pd.to_datetime(data.index)
    return data


if __name__ == "__main__":
    # with open('ghcn-daily/data/chicago/USC00111577.dly') as afile:
    with open('ghcn-daily/chicago/USW00094846.dly') as afile:
        data = [line.strip() for line in afile.readlines()]
        
    # parse the data
    parsed_data = readDly(data)

    # write the parsed data to file
    get_attribute('TMAX')
    get_attribute('TMIN')
    get_attribute('PRCP') # Precipitation (tenths of mm)
    get_attribute('AWND') # Average daily wind speed (tenths of meters per second)
    get_attribute('TSUN') # TSUN = Daily total sunshine (minutes)

    # 01 = Fog, ice fog, or freezing fog (may include heavy fog)
    # 16 = Rain (may include freezing rain, drizzle, and freezing drizzle) 
    # 18 = Snow, snow pellets, snow grains, or ice crystals
    get_attribute('WT01') 
    get_attribute('WT16')
    get_attribute('WT18')


    # some extra data
    get_attribute('TAVG') # TAVG = Average temperature
    get_attribute('TOBS') # TOBS = Temperature at the time of observation
    get_attribute('SNOW') # Snowfall (mm)
    get_attribute('SNWD') # Snow depth (mm)
    get_attribute('WSFM') # WSFM = Fastest mile wind speed (tenths of meters per second)

    # 02 = Heavy fog or heaving freezing fog (not always distinquished from fog)
    # 03 = Thunder
    # 04 = Ice pellets, sleet, snow pellets, or small hail 
    # 05 = Hail (may include small hail)
    # 06 = Glaze or rime 
    # 07 = Dust, volcanic ash, blowing dust, blowing sand, or blowing obstruction
    # 08 = Smoke or haze 
    # 09 = Blowing or drifting snow
    # 11 = High or damaging winds
    get_attribute('WT02') 
    get_attribute('WT03') 
    get_attribute('WT04')
    get_attribute('WT05')
    get_attribute('WT06')
    get_attribute('WT07')
    get_attribute('WT08')
    get_attribute('WT09')
    get_attribute('WT11')
    get_attribute('WT14')

