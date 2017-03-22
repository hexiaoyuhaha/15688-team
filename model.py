from datetime import timedelta
def prepare_data(wdf, train_dt_start, train_dt_end, test_dt_start, test_dt_end, nw_corner, se_corner, hours_interval=6):
    
    # Number: 100*100 * 4*365 = 1000w
    
    X_train, y_train = get_data(wdf, train_dt_start, train_dt_end, nw_corner, se_corner)
    X_test, y_test = get_data(wdf, test_dt_start, test_dt_end, nw_corner, se_corner)
    
    return X_train, X_test, y_train, y_test

def prepare_data_baseline(wdf, train_dt_start, train_dt_end, test_dt_start, test_dt_end, nw_corner, se_corner, hours_interval=6):
    
    X_train, y_train = get_data_baseline(wdf, train_dt_start, train_dt_end, nw_corner, se_corner)
    X_test, y_test = get_data_baseline(wdf, test_dt_start, test_dt_end, nw_corner, se_corner)
    
    return X_train, X_test, y_train, y_test

def get_data(wdf, dt_start, dt_end, nw_corner, se_corner, hours_interval=6):
    X = []
    y = []
    dt = dt_start
    latitude_blocks_num, longitude_blocks_num = 25, 25     # Default
    
    last = time.time()
    
    # n days * 4 time slots
    recent_kernel_num = 3*4 # n=3d
    while dt < dt_end:
        # Avoid the starting point gap
        if (dt - timedelta(hours = recent_kernel_num * 6)).year < 2013:
            dt += timedelta(hours = hours_interval)
            continue
            
        else:
            for i in range(latitude_blocks_num):
                for j in range(longitude_blocks_num):
                    y_unit = 1.0 if get_crime_count(dt, i, j, hours_interval=6) > 0 else 0.0
                    X_unit = []
                    # Len X_unit = 12 + 1 + 8 + 6 + 4 + 2 = 33
                    # KDEs in recent 3 days for 4 time slots * 12
                    recent_kde = []
                    shift = dt.hour / 6
                    for offset in range(recent_kernel_num):
                        recent_kde.append(                                      get_interval_KDE(dt - timedelta(hours = (offset+shift)*6 + 6),                                        i, j, hours_interval))
                    X_unit += recent_kde                 

                    # Long-term kernel * 1
                    k_long = get_yearly_KDE(dt.year-1, i, j)# * get_yearly_crime_count(dt.year, i, j)
                    X_unit.append(k_long)
                    
#                     # Weather Data * 8 
#                     weather = get_weather(wdf, dt.date())
#                     X_unit += list(weather)
                    
                    # Demographic Data * 6
                    for key in demographic_cols:
                        X_unit.append(demographic_dict[key][i][j])
                        
                    # Time Slot * 4 
                    time_slot = [0]*4
                    time_slot[dt.hour/6] = 1
                    X_unit += time_slot
                    
                    # Weekday Slot * 2
                    weekday = [0] * 2
                    weekday[dt.weekday()/5] = 1
                    X_unit += weekday
                    
                    y.append(y_unit)
                    X.append(X_unit)
            
            dt += timedelta(hours = hours_interval)
            if dt.day == 20:
                print "-Get One Month #{} Data for One More Time Slot-".format(len(y))
                end = time.time()
                print "Total:{} seconds,  Unit:{} seconds".format(end-start, end-last)
                last = time.time()
    return X, y


def get_data_baseline(wdf, dt_start, dt_end, nw_corner, se_corner, hours_interval=6):
    X = []
    y = []
    dt = dt_start
    latitude_blocks_num, longitude_blocks_num = 25, 25
    last = time.time()
    
    recent_kernel_num = 3*4 
    while dt < dt_end:
        # Avoid the starting point gap
        if (dt - timedelta(hours = recent_kernel_num * 6)).year < 2013:
            dt += timedelta(hours = hours_interval)
            continue
            
        else:
            for i in range(latitude_blocks_num):
                for j in range(longitude_blocks_num):
                    y_unit = 1.0 if get_crime_count(dt, i, j, hours_interval=6) > 0 else 0.0
                    X_unit = []
                    # Len X_unit = 1 + 4 + 2 = 7
                    # Long-term kernel * 1
                    k_long = get_yearly_KDE(dt.year-1, i, j)# * get_yearly_crime_count(dt.year, i, j)
                    X_unit.append(k_long)
                    
                    # Time Slot * 4 
                    time_slot = [0]*4
                    time_slot[dt.hour/6] = 1
                    X_unit += time_slot
                    
                    # Weekday Slot * 2
                    weekday = [0] * 2
                    weekday[dt.weekday()/5] = 1
                    X_unit += weekday
                    
                    y.append(y_unit)
                    X.append(X_unit)
                    
            dt += timedelta(hours = hours_interval)
            if dt.day == 20:
                print "-Get One Month #{} Data for One More Time Slot-".format(len(y))
                end = time.time()
                print "Total:{} seconds,  Unit:{} seconds".format(end-start, end-last)
                last = time.time()

    return X, y


import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets

def logistic_regression(X_train, X_test, y_train, y_test):
    start = time.time()
    last = time.time()
    logreg = linear_model.LogisticRegression(solver='sag')
    logreg.fit(X_train, y_train)

    print 'Finish Fitting Part'
    end = time.time()
    print "Total:{} seconds,  Unit:{} seconds".format(end-start, end-last)


    P_train = logreg.predict_proba(X_train)
    P_test = logreg.predict_proba(X_test)
    P_train = [p[1] for p in P_train]
    P_test = [p[1] for p in P_test]

    return P_train, P_test



from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import keras.backend as K

def deep_logistic(X_train, y_train, X_test, input_dim, nb_epoch=100, batch_size=625):
    # create model
    p = 0.1
    model = Sequential()
    model.add(Dense(input_dim*2, input_dim=input_dim, init='normal', activation='linear'))
    model.add(Dropout(p))
    model.add(Dense(input_dim/2, init='normal', activation='linear'))
    model.add(Dropout(p))
    model.add(Dense(1, init='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_crossentropy'])
    # sparse_categorical_crossentropy  
    # model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # y_train = np.expand_dims(y_train, -1)
    model.fit(X_train, y_train, nb_epoch=nb_epoch, batch_size=batch_size)
    print model.evaluate(X_test, y_test, batch_size=625)
    P_test = model.predict(X_test)
    P_train = model.predict(X_train)
    return [p[0] for p in P_train], [p[0] for p in P_test]


import numpy as np
import matplotlib.pyplot as plt
def plot_prob(p_crime):
    p_crime.sort()
    x = np.array(p_crime)
    y = np.array([float(i)/len(p_crime) for i in range(len(p_crime))])
    plt.clf()
    plt.plot(x, y)
    plt.xlim([0.0, 0.5])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Probability of Crime Incident')
    plt.ylabel('% Area Surveilled')
    plt.title('Prediction: Probability of Crime Incident')
    plt.legend(loc="lower right")
    plt.show()
    # plt.savefig('1month.png')

    lw = 2
    # plt.plot(x, y, color='darkorange',
    #          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)

    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')


import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp


# In[62]:


def cal_auc_results(p_crime, count, group_size = 25*25):

    data_total_num = len(p_crime)
    group_num = data_total_num / group_size

    roc_auc_group = []
    for k in xrange(group_num):
        start, end = group_size * k, group_size * (k+1)
        data_sorted = [(x,y) for (x,y) in sorted(zip(p_crime[start:end], count[start:end]), reverse=True)]
        area_num = len(data_sorted)
        x = np.array([i/float(area_num) for i in range(area_num)])    
        y = []
        c_sum = sum([data[1] for data in data_sorted])
        per = 0
        for i in range(area_num):
            per += data_sorted[i][1]
            y.append(per/float(c_sum)) 


        # fpr, tpr, thresholds = roc_curve(y, x, pos_label=2)
        roc_auc = auc(x, y)
#         if k%4 == 3:
        roc_auc_group.append(roc_auc)
    return roc_auc_group




def plot_auc_results(x_per, roc_auc_group):

    plt.figure()
    lw = 2
    # plt.plot(x, y, color='darkorange',
    #          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    
    plt.plot_date(x_per, roc_auc_group, lw=lw, fmt='-', color='blue',label='Mean AUC = %0.2f' % np.mean(roc_auc_group))
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    plt.ylim([0.0, 1.05])
    plt.xlabel('Prediction Date')
    plt.ylabel('AUC Value')
    plt.title('Evaluation: AUC performance')
    plt.legend(loc="lower right")
#     plt.axis([min(x_per), max(x_per) , 0.0, 1.0])
#     plt.xticks(np.arange(1,5,4))
#     plt.xticks(np.arange(min(x_per), max(x_per)+1, (-min(x_per)+max(x_per))/4))
    plt.show()
    


def plot_ROC_curve(x, y, roc_auc):
    plt.figure()
    lw = 2
    plt.plot(x, y, color='darkorange',
             lw=lw, label='AUC(area) = %0.2f' % roc_auc)
#     plt.plot(x_base, y_base, color='blue',
#              lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_base)

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('% Area Surveilled')
    plt.ylabel('% Incidents Captured')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

def plot_two_ROC_curve(x, y, x_base, y_base, roc_auc, roc_auc_base):
    plt.figure()
    lw = 2
    plt.plot(x, y, color='darkorange',
             lw=lw, label='KDE+Features: AUC(area) = %0.2f' % roc_auc)
    plt.plot(x_base, y_base, color='blue',
         lw=lw, label='KDE: AUC(area) = %0.2f' % roc_auc_base)
#     plt.plot(x_base, y_base, color='blue',
#              lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_base)

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('% Area Surveilled')
    plt.ylabel('% Incidents Captured')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()    
    
def cal_one_auc_result(p_crime, count):
    data_total_num = len(p_crime)

    data_sorted = [(x,y) for (x,y) in sorted(zip(p_crime, count), reverse=True)]
    area_num = len(data_sorted)
    x = np.array([i/float(area_num) for i in range(area_num)])    
    y = []
    c_sum = sum([data[1] for data in data_sorted])
    per = 0
    for i in range(area_num):
        per += data_sorted[i][1]
        y.append(per/float(c_sum)) 
    roc_auc = auc(x, y)
    # if k%4 == 3:
    return roc_auc, x, y


import time
from datetime import datetime
def main():
    # Example
    start = time.time()

    dt_train_start = datetime(2015,6,1,0,0,0)
    dt_train_end = datetime(2015,9,1,0,0,0)
    dt_test_start = datetime(2015,9,1,0,0,0)
    dt_test_end = datetime(2015,10,1,0,0,0)

    X_train, X_test, y_train, y_test = prepare_data\
            (wdf, dt_train_start, dt_train_end, dt_test_start, dt_test_end,\
             [42.025339, -87.950502], [41.633514, -87.515073], 6)

    dt_train_base_start = datetime(2015,6,1,0,0,0)
    dt_train_base_end = datetime(2015,9,1,0,0,0)
    dt_test_base_start = datetime(2015,9,1,0,0,0)
    dt_test_base_end = datetime(2015,10,1,0,0,0)

    X_train_base, X_test_base, y_train_base, y_test_base = prepare_data_baseline\
            (wdf, dt_train_base_start, dt_train_base_end, dt_test_base_start, dt_test_base_end,\
             [42.025339, -87.950502], [41.633514, -87.515073], 6)


    P_train, P_test = logistic_regression(X_train, X_test, y_train,  y_test)
    P_train_base, P_test_base = logistic_regression(X_train_base, X_test_base, y_train_base, y_test_base)


    # roc_auc, x_trans, y_trans = cal_one_auc_result(P_train, y_train)
    roc_auc, x_trans, y_trans = cal_one_auc_result(P_test, y_test)
    plot_ROC_curve(x_trans, y_trans, roc_auc)


    roc_auc_base, x_trans_base, y_trans_base = cal_one_auc_result(P_train_base, y_train_base)
    plot_ROC_curve(x_trans_base, y_trans_base, roc_auc_base)

    plot_two_ROC_curve(x_trans, y_trans, x_trans_base, y_trans_base, roc_auc, roc_auc_base)



    # get_crime_count(dt, block_lat_idx, block_lon_idx, hours_interval=6)
    year = 2014
    kde = [get_yearly_KDE(year, i, j) for i in xrange(25) for j in xrange(25)]
    count = [get_yearly_crime_count(year, i, j) for i in xrange(25) for j in xrange(25)]

    data_sorted_base = [(x,y) for (x,y) in sorted(zip(kde, count), reverse=True)]
    area_num_base = len(data_sorted_base)

    x_base = np.array([i/float(area_num_base) for i in range(area_num_base)])    
    y_base = []
    c_sum_base = sum([data[1] for data in data_sorted_base])
    per = 0
    for i in range(area_num_base):
        per += data_sorted_base[i][1]
        y_base.append(per/float(c_sum_base))

    roc_auc_base = auc(x_base, y_base)
    plot_ROC_curve(x_base, y_base, roc_auc_base)


    return
 
main()


