import numpy as np
from itertools import izip
import pandas as pd
import pickle
import matplotlib.pyplot as plt


def generate_demographic_pickle_file():
    '''
    Processed the demographic data and stored it in a pickle file for later model training
    '''
    boundry = read_county_boundry("demographic_data/county_boundry.csv")
    b_to_c_dict = block_to_community(boundry)

    demographic_data = pd.read_csv('demographic_data/socioeconomic_2008_2012.csv')
    demographic_data.dropna(subset=['Community Area Number'], inplace=True)

    cols = ['PERCENT OF HOUSING CROWDED','PERCENT HOUSEHOLDS BELOW POVERTY','PERCENT AGED 16+ UNEMPLOYED','PERCENT AGED 25+ WITHOUT HIGH SCHOOL DIPLOMA','PERCENT AGED UNDER 18 OR OVER 64','PER CAPITA INCOME ']
    demographic_dict = {k:np.zeros((25,25)) for k in cols}

    for k,array in demographic_dict.iteritems():
        data = demographic_data[k]

        for i in range(25):
            for j in range(25):
                c_idx = b_to_c_dict[i][j]
                if c_idx > 0:
                    array[i][j] = data[c_idx-1]
    
    output = open('demographic.pkl', 'wb')
    pickle.dump(demographic_dict, output)
    output.close()


def read_demographic_data(demographic_pickle_file_path):
    '''Input:  Demographic pickle file path
       Output: A dict containing normalized demographic features.
               key: demographic feature name. ['PERCENT OF HOUSING CROWDED','PERCENT HOUSEHOLDS BELOW POVERTY','PERCENT AGED 16+ UNEMPLOYED','PERCENT AGED 25+ WITHOUT HIGH SCHOOL DIPLOMA','PERCENT AGED UNDER 18 OR OVER 64','PER CAPITA INCOME ']
               value: 25*25 numpy array
    '''
    pkl_file = open(demographic_pickle_file_path, 'rb')
    demographic_dict = pickle.load(pkl_file)
    pkl_file.close()
    
    normalized_demographic_dict = demographic_dict
    for k, v in demographic_dict.iteritems():
        normalized_demographic_dict[k] = v/np.max(v)
    return normalized_demographic_dict

    

def visualize_demographic(data):
    """
    data1 is according latitudes
    data2 is according longitudes
    """
    min1 = 0
    min2 = 0
    max1 = 24
    max2 = 24
    
    X, Y = np.mgrid[min1:max1:1j, min2:max2:1j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    
    fig, ax = plt.subplots()
    ax.imshow(data, cmap=plt.cm.gist_earth_r, extent=[min1, max1, min2, max2])
    plt.show()




def is_in(block_center, community_boundary):
    res = False
    num = len(community_boundary)
    
    for i in range(num)[:-1]:
        p1 = community_boundary[i]
        p2 = community_boundary[i+1]
        
        is_within_edge_y = ((p1[1]-block_center[1]) * (p2[1]-block_center[1])) < 0
        if is_within_edge_y and block_center[0] < ((p2[0]-p1[0]) * (block_center[1]-p1[1]) / (p2[1]-p1[1]) + p1[0]):
            res = not res
            
    return res


def block_to_community(community_boundary_dict, block_lat_num=25, block_lon_num=25):
    res = np.zeros((block_lat_num, block_lon_num), dtype='int')
    
    for lat_idx in range(block_lat_num):
        for lon_idx in range(block_lon_num):
            block_bb= __block_idx_to_bounding_box(lat_idx, lon_idx, block_lat_num, block_lon_num)
            block_center = np.array([(block_bb[0][1]+block_bb[1][1])/2, (block_bb[0][0]+block_bb[1][0])/2]) # switch to [lon, lat]
            
            for key,community_boundary in community_boundary_dict.iteritems():
                if is_in(block_center, community_boundary):
                    res[lat_idx][lon_idx] = int(key)
                    break
            
    return res






def read_county_boundry(boudry_file_path):
    '''Input: the path of the boundry file
       Output: a dict containing all the boundry point coornidate of each county.
               key: int, county number
               value: list of tuples, each tuple is the the coordinate of boundry point
    '''
    areas = pd.read_csv(boudry_file_path)
    the_geom = areas.the_geom.values
    area_num = areas.AREA_NUM_1.values

    geom = [a.split(",") for a in the_geom]
    # coor = [[a.strip().split(" ") for a in b] for b in geom]
    coor = [[(float(a.strip().split(" ")[0]),float(a.strip().split(" ")[1])) for a in b] for b in geom]
    boundry = {k:v for k,v in izip(area_num, coor)}
    return boundry



if __name__ == "__main__":
    generate_demographic_pickle_file()



