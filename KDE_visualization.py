import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from KDE_pre_computation import compute_Gaussian_KDE

def visualize_KDE(kernel, data1, data2):
    '''
    data1 is latitudes array
    data2 is longitudes array
    '''
    chicago_bounding_box = [[42.025339, -87.950502], [41.633514, -87.515073]]
	
    min1 = data1.min()
    min2 = data2.min()
    max1 = data1.max()
    max2 = data2.max()
    
    X, Y = np.mgrid[min1:max1:200j, min2:max2:200j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    
    Z = np.reshape(kernel(positions).T, X.shape)
    
    df = pd.read_csv('chicago_boundary.csv')
    pd.to_numeric(df['latitude'])
    pd.to_numeric(df['longitude'])
    
    fig, ax = plt.subplots(figsize=((chicago_bounding_box[1][1]-chicago_bounding_box[0][1])*50, \
                                    (chicago_bounding_box[0][0]-chicago_bounding_box[1][0])*50))
    ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r, extent=[min1, max1, min2, max2])
    ax.plot(data1, data2, 'k.', markersize=2)
    
    ax.set_xlim([chicago_bounding_box[0][1], chicago_bounding_box[1][1]])
    ax.set_ylim([chicago_bounding_box[1][0], chicago_bounding_box[0][0]])
    
    plt.scatter(df['longitude'].tolist(), df['latitude'].tolist(), c='m', marker='o')
    
    plt.show()
	

if __name__ == "__main__":
    ''' 
    Visualize the output of KDE, this will need the output of KDE_pre_computation.py
    '''
    df_2015 = pd.read_csv('data/chicago_crime_data_2015_sorted.csv')
    df_2015['date'] = pd.to_datetime(df_2015['date'])
    kernel_2015 = compute_Gaussian_KDE(df_2015['longitude'], df_2015['latitude'])

    visualize_KDE(kernel_2015, df_2015['longitude'], df_2015['latitude'])