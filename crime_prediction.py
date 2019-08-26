# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 10:40:08 2019

@author: Angelo Antonio Manzatto

"""

##################################################################################
# Libraries
##################################################################################  
import os
import math
import pandas as pd

from datetime import datetime
from datetime import timedelta  

import numpy as np

import h5py

import matplotlib.pyplot as plt
import matplotlib.cm 

import seaborn as sns
sns.set()

import keras.backend as K

from keras.models import Model

from keras.layers import Input, Dense, Reshape, Activation, Add
from keras.layers import Conv2D , BatchNormalization 

from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping

from keras.optimizers import Adam

from keras.engine.topology import Layer

############################################################################################
# Load Dataset
############################################################################################
  
dataset_folder = 'dataset'

# Philadelphia Dataset
philadelphia_dataset_folder = os.path.join(dataset_folder, 'philadelphia')
philadelphia_dataset_file = os.path.join(philadelphia_dataset_folder,'crime.csv')

los_angeles_dataset_folder = os.path.join(dataset_folder, 'los_angeles')
los_angeles_dataset_file = os.path.join(los_angeles_dataset_folder,'Crimes_2012-2016.csv')

philadelphia_column_names = ['dc_dist','psa', 'dispatch_date_time','dispatch_date','dispatch_time',
                             'hour','dc_key','location_block','ucr_general','text_general_code','police_districts','month','lon','lat']

philadelphia_df = pd.read_csv(philadelphia_dataset_file, names=philadelphia_column_names, skiprows=1, header=None)

philadelphia_df.head()

############################################################################################
# Pre Process Dataset
############################################################################################

# Convert incident_date column to date format
philadelphia_df['incident_date'] = pd.to_datetime(philadelphia_df['dispatch_date_time'],format='%Y-%m-%d %H:%M:%S')

# Date like features
philadelphia_df['day'] = philadelphia_df['incident_date'].dt.day.astype(np.int16)
philadelphia_df['month'] = philadelphia_df['incident_date'].dt.month.astype(np.int16)
philadelphia_df['year'] = philadelphia_df['incident_date'].dt.year.astype(np.int16)

# Remove all rows withoud lat and lon information
print("Number of NaN Latitude rows: {0}".format(philadelphia_df.lat.isnull().sum()))
print("Number of NaN Longitude rows: {0}".format(philadelphia_df.lon.isnull().sum()))

philadelphia_df = philadelphia_df[(~philadelphia_df.lon.isnull()) & (~philadelphia_df.lat.isnull())]

print("Number of NaN Latitude rows: {0}".format(philadelphia_df.lat.isnull().sum()))
print("Number of NaN Longitude rows: {0}".format(philadelphia_df.lon.isnull().sum()))

sample = philadelphia_df.head()

# Get boundaries of latitude and longitude position
min_lat = philadelphia_df.lat.min()
max_lat = philadelphia_df.lat.max()
min_lon = philadelphia_df.lon.min()
max_lon = philadelphia_df.lon.max()

'''
Min Lat: 39.869991 , Max Lat: 40.137895
Min Lon: -75.277728 , Max Lon: -74.957504

'''

# Check minimum and maximum latitude and longitude
print("Min Lat: {0} , Max Lat: {1}".format(min_lat,max_lat))
print("Min Lon: {0} , Max Lon: {1}".format(min_lon,max_lon))

# Normalize latitude and longitude between [-1,1]
philadelphia_df['lat_n'] = 0.0
philadelphia_df['lon_n'] = 0.0

philadelphia_df['lat_n'] = philadelphia_df['lat'].apply(lambda x: 2. * (x - min_lat)/(max_lat - min_lat)-1.)
philadelphia_df['lon_n'] = philadelphia_df['lon'].apply(lambda x: 2. * (x - min_lon)/(max_lon - min_lon)-1.)

min_norm_lat = philadelphia_df.lat_n.min()
max_norm_lat = philadelphia_df.lat_n.max()
min_norm_lon = philadelphia_df.lon_n.min()
max_norm_lon = philadelphia_df.lon_n.max()

# Check minimum and maximum latitude and longitude
print("Min Norm Lat: {0} , Max Norm Lat: {1}".format(min_norm_lat,max_norm_lat))
print("Min Norm Lon: {0} , Max Norm Lon: {1}".format(min_norm_lon,max_norm_lon))

# Define the grid size of our map division
map_width = 16
map_height = 16

# RAW CONVERSION !!! Should be checked later. This is just for testing purposes
philadelphia_df['square_h'] = np.nan
philadelphia_df['square_w'] = np.nan

# This method receives latitude and longitude normalized between [-1,1] and finds grid position based on map shape (w, h)
def square_location(coord, map_dimension):
    
    location = (coord + 1.) / 2 * (map_dimension-1)

    return  int(round(location))
    

philadelphia_df['square_h'] = philadelphia_df['lat_n'].apply(lambda x: square_location(x, map_height))
philadelphia_df['square_w'] = philadelphia_df['lon_n'].apply(lambda x: square_location(x ,map_width))

min_h = philadelphia_df.square_h.min()
max_h = philadelphia_df.square_h.max()
min_w = philadelphia_df.square_w.min()
max_w = philadelphia_df.square_w.max()

# Check minimum and maximum latitude and longitude
print("Min height: {0} , Max height: {1}".format(min_h,max_h))
print("Min width: {0} , Max width {1}".format(min_w,max_w))

# Aggregate (time, square) to create the incident map sum (crimes)
agg_columns = ['day','month','year','hour','square_h','square_w']

incident_map = philadelphia_df.groupby(agg_columns,as_index=True).size()
incident_map = incident_map.reset_index()

incident_map.rename(columns ={0:'total_crimes'},inplace=True)

# Create a string timestamp
incident_map['ts'] = incident_map[['year','month','day','hour']].apply(lambda x : str(x[0]) + str(x[1]).zfill(2)+str(x[2]).zfill(2)+str(x[3]).zfill(2),axis=1)

# Sort heatmap
incident_map = incident_map.sort_values(by=['ts'])

# Number of crime matrix to generate
ts_count = incident_map['ts'].value_counts()
ts_count = ts_count.reset_index()
ts_count = ts_count.sort_values(by=['index'])
n_ts = len(ts_count)

# Store heatmap matrices on data and timestamps 
data = []
timestamps = []

count = 1
for index, ts in ts_count.iterrows():

    # Get timestamp string
    ts_index = ts['index']
    
    print('Processing timeframe {0}: {1} from {2} %'.format(index,ts_index,100 * np.round(count / len(ts_count),4 )))
    
    # Create heatmap matrix as array since square column is in vector notation
    heatmap = np.zeros((map_height , map_width))
    
    # Select just crimes on the timestamp frame
    incident_map_ts = incident_map[incident_map.ts == ts_index]
    
    # Fill each square of the heatmap matris with the total number of crimes commited
    for _, incident_row in incident_map_ts.iterrows():
        
        heatmap[incident_row['square_h']][incident_row['square_w']] = incident_row['total_crimes']

    data.append(heatmap)
    timestamps.append(incident_row['ts'])
    
    count+=1
    
data = np.asarray(data)
timestamps = np.asarray(timestamps).astype(np.string_) # We have to convert to byte datatype so we can save the .h5 file

# Save dataset preprocessed as .h5 file 
h5f = h5py.File(os.path.join(dataset_folder,'philadelphia_crime_heatmap.h5'), 'w')
h5f.create_dataset('data', data=data)
h5f.create_dataset('timestamps', data=timestamps)
h5f.close()

############################################################################################
# Load H5 Dataset
############################################################################################

dataset_file = os.path.join(dataset_folder,'philadelphia_crime_heatmap.h5')

# Load dataset file
f = h5py.File(dataset_file)
data = f['data'][()]
timestamps = f['timestamps'][()]

# Plot some samples from dataset
n_samples = 5

for i in range(n_samples):
    
    # define the size of images
    f, ax = plt.subplots()
    f.set_figwidth(8)
    f.set_figheight(4)
    
    # randomly select a sample
    idx = np.random.randint(0, len(data))
    heatmap = data[idx]
    
    ax.set_title("Crime Heatmap: {0}".format(timestamps[idx].decode("utf-8")))
    ax.imshow(heatmap)

############################################################################################
# Pre-Process Dataset
############################################################################################
    
# Convert timestamps from ASCII format to string
formated_timestamps = []
for ts in timestamps:
    formated_timestamps.append(ts.decode("utf-8"))
    
# Scale in flow and out flow values on the map matrices to a range between [-1,1]    
min_value = data.min()
max_value = data.max()    

print("Minimum values: {0} , Maximum value: {1}".format(min_value,max_value))

data_scaled = 1. * (data - min_value) / (max_value - min_value)
data_scaled = 2. * data_scaled - 1.

print("Minimum scaled values: {0} , Maximum scaled value: {1}".format(data_scaled.min(),data_scaled.max()))

# Build a dictionary of time stamps. This will ease our work to convert between timestamps to indices to get
# the in/out flow matrices.
ts_dict = {}
ts_list = []
for i, ts in enumerate(formated_timestamps):
    
    converted_time = datetime.strptime(ts, '%Y%m%d%H')
      
    # Add converted time from string to a list for iteration and for a dictionary for search purposes
    ts_list.append(converted_time)
    ts_dict[converted_time] = i
    
# Check missing timestamps between 1 hour difference
missing_timestamps = []

for i in range(len(ts_list)-1):
    
    # Hour offset
    offset = ts_list[i] + timedelta(hours=1)
    
    # If the hour offset key is not in dictionary of timestamps add it to list
    if offset not in ts_dict:
        missing_timestamps.append(offset)
        
print("Missing timestamps count: {0}".format(len(missing_timestamps)))

############################################################################################
# Create Train / Target data
############################################################################################
'''
Minimum granularity will be 1 hour

To create the input for our model we need to aggregate the inflow and outflow matrices according
to three interval of times defined in the article as: closeness, period and trend.

For this project:
 * Closeness is a difference in 1 hour period between two matrices
 * Period is a difference is 24 hours period between two matrices
 * Trend is a difference is 7 days period between two matrices
 
This means that for example, for a data (16 x 16) heatmap matrices collected
at time stamp: 2014 08 07 01:00:00 we will have to do the following transformations:
    
Input closeness = len closeness stack of consecutive matrices distant between closeness interval.
Ex: Len = 3 and interval = 1 hour - stack [2014 08 07 01:00:00, 2014 08 07 02:00:00 , 2014 08 07 03:00:00]  matrices

Input period = len period stack of consecutive matrices distant between period interval.   
Ex: Len = 4 and interval = 24 hours - stack [2014 08 07 01:00:00, 2014 08 08 01:00:00 , 2014 08 09 01:00:00, 2014 08 10 01:00:00] matrices

Input trend = len trend stack of consecutive matrices distant between trend interval.   
Ex: Len = 4 and interval = 168 hours - stack [2014 08 07 01:00:00, 2014 08 14 01:00:00 , 2014 08 21 01:00:00, 2014 08 28 01:00:00] matrices    

This is an important information and the dataset should have little or almost NO disconnected interval between two
inflow / outflow matrices meaning that we should avoid missing hours.
'''

# Convert timestamp to a one hot encoded vector taking into account week way and if it is weekend or not
def one_hot_day_week(timestamp):
    
    converted_time = datetime.strptime(ts, '%Y%m%d%H')
    i = converted_time.weekday()

    one_hot_encoded = np.zeros((8))
    
    # Day week (sunday, monday...) encoder
    one_hot_encoded[i] = 1
    
    # Weekend / Not Weekend encoder
    if i < 5:
        one_hot_encoded[7] = 1
        
    return one_hot_encoded

closeness_interval = 1  # distance between hours
period_interval = 24 * closeness_interval  # number of time intervals in one day
trend_interval = 7 * period_interval

closeness_len = 3 # recent time (closeness) 
period_len = 4 # near history (period) 
trend_len = 4 # distant history (trend) 

closeness_range = [x * closeness_interval for x in range(1,closeness_len+1)]
period_range = [x * period_interval for x in range(1,period_len + 1)]
trend_range = [x * trend_interval  for x in range(1,trend_len+1)]

# Create X, y data    
X_Closeness, X_Period, X_Trend, X_External, Y , Y_timestamp = [],[],[],[],[],[]

# Crete the datasets for closeness, period and trend
# Since we have future predictions as output we need to build the dataset based on the lates trend period as starting point
starting_period = trend_interval * trend_len

# We construct the X, y datasets based on a reversed time interval, from the latest trend to starting closeness
for i in range(starting_period, len(formated_timestamps)):
    print(formated_timestamps[i])
    # Starting period
    date = datetime.strptime(formated_timestamps[i], '%Y%m%d%H') 
    
    check_dates = []
    
    # Get all dates in the closeness interval near the target 
    for c in closeness_range:
        check_dates.append(date - timedelta(hours=c))
        
    for p in period_range:
        check_dates.append(date - timedelta(hours=p))
        
    for t in trend_range:
        check_dates.append(date - timedelta(hours=t))
        
    # Check if all those selected dates exists in our timestamp dictionary and if not go to the next iteration
    break_flag = False
    for check_date in check_dates:
        if check_date not in ts_dict:
            print("Date frame missing!: {0} ".format(formated_timestamps[i]))
            break_flag = True
    
    if break_flag:
        continue
    # Parse again to create de dataset stacking the time range for closeness, period and trend
    
    # X Closeness
    xc = []
    for c in closeness_range:
        xc.append(data_scaled[ts_dict[date - timedelta(hours=c)]])
    xc = np.stack(xc, axis=-1)
    
    # X Period
    xp = []    
    for p in period_range:
        xp.append(data_scaled[ts_dict[date - timedelta(hours=p)]])
    xp = np.stack(xp,axis=-1)
    
    # X Trend
    xt = []    
    for t in trend_range:
        xt.append(data_scaled[ts_dict[date - timedelta(hours=t)]])
    xt = np.stack(xt,axis=-1) 
    
    # Target 
    y = data_scaled[ts_dict[date]]
    
    # Add each created set to the final datasets
    X_Closeness.append(xc)
    X_Period.append(xp)
    X_Trend.append(xt)
    X_External.append(one_hot_day_week(formated_timestamps[i]))
    
    Y.append(y)
    Y_timestamp.append(formated_timestamps[i])
    
X_Closeness = np.asarray(X_Closeness)
X_Period = np.asarray(X_Period)
X_Trend = np.asarray(X_Trend)
X_External = np.asarray(X_External)
Y = np.asarray(Y)

print("X_Closeness shape: ", X_Closeness.shape)
print("X_Period shape: ", X_Period.shape)
print("X_Trend shape: ", X_Trend.shape)
print("X_External shape: ", X_External.shape)
print( "Y shape:", Y.shape)

############################################################################################
# Split dataset into Train / Test
############################################################################################

days_test = 10
n_test = 24 * days_test

# Split dataset into training / test sets
XC_train, XP_train, XT_train,XE_train, Y_train = X_Closeness[:-n_test], X_Period[:-n_test], X_Trend[:-n_test],X_External[:-n_test], Y[:-n_test]
XC_test, XP_test, XT_test, XE_test, Y_test = X_Closeness[-n_test:], X_Period[-n_test:], X_Trend[-n_test:],X_External[-n_test:], Y[-n_test:]
    
# Time stamp split so we can track the period
timestamp_train, timestamp_test = Y_timestamp[:-n_test], Y_timestamp[-n_test:]

# Concatenate closeness , period and trend 
X_train = [XC_train,XP_train,XT_train,XE_train]
X_test = [XC_test,XP_test,XT_test,XE_test]
    
print("X Train size: ", len(X_train))
print("X Test size: ", len(X_test))    

############################################################################################
# Spatial Temporal Residual Network
############################################################################################

############################################################################################
# Fusion Layer
############################################################################################
class FusionLayer(Layer):

    def __init__(self, **kwargs):
        super(FusionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        initial_weight_value = np.random.random(input_shape[1:])
        self.W = K.variable(initial_weight_value)
        self.trainable_weights = [self.W]

    def call(self, x, mask=None):
        return x * self.W
    
    def get_output_shape_for(self, input_shape):
        return input_shape

############################################################################################
# ResNet Identity Block
############################################################################################
def identity_block(inputs, filters, block_id):
    
    x = BatchNormalization(name='block_' + block_id + '_identity_batch_1')(inputs)
    x = Activation('relu', name='block_' + block_id + '_identity_relu_1')(x)
    x = Conv2D(filters, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer='he_normal', name='block_' + block_id + '_identity_conv2d_1')(x)

    x = BatchNormalization(name='block_' + block_id + '_identity_batch_2')(x)
    x = Activation('relu',name='block_' + block_id + '_identity_relu_2')(x)
    x = Conv2D(filters, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer='he_normal', name='block_' + block_id + '_identity_conv2d_2')(x)
    
    x = Add(name='block_' + block_id + '_add')([inputs,x])

    return x

############################################################################################
# Spatial Time Residual Network ST-ResNet
############################################################################################

map_height = 16
map_width = 16
    
c_conf=(map_height, map_width, closeness_len) # closeness
p_conf=(map_height, map_width, period_len) # period
t_conf=(map_height, map_width, trend_len) # trend

# main input
main_inputs = []
outputs = []

for conf, name in zip([c_conf, p_conf, t_conf],['c','p','t']):
    
    map_height, map_width, len_seq = conf
    
    Image = Input(shape=(map_height, map_width, len_seq), name='input_' + name)
    
    main_inputs.append(Image)
    
    x = Conv2D(64, kernel_size=(3,3), padding="same")(Image)
    x = identity_block(x, 64, block_id='0_' + name)
    x = identity_block(x, 64, block_id='1_' + name)
    x = identity_block(x, 64, block_id='2_' + name)
    x = identity_block(x, 64, block_id='3_' + name)
    
    x = Activation('relu')(x)        
    x = Conv2D(1 , kernel_size=(3,3), padding="same")(x)
    
    output = FusionLayer(name="fusion_layer_" + name)(x)
    
    outputs.append(output)
        
# External component
external_dim = 8 # One Hot Encoded Timestamp into [weekday , weekend/notweekend]

# Concatenate external inputs with temporal inputs
external_input = Input(shape=(external_dim,), name='external_input')   
main_inputs.append(external_input)

embedding = Dense(10, name='external_dense_1')(external_input)
embedding = Activation('relu')(embedding)
embedding = Dense(map_height * map_width)(embedding)
embedding = Activation('relu')(embedding)
external_output = Reshape((map_height, map_width , 1),name='external_output')(embedding)

# Fuse all layers
fusion_temporal =  Add(name= 'FusionTemporal')(outputs)

fusion = Add(name='Fusion')([fusion_temporal,external_output])

final_output = Activation('tanh', name='Tanh')(fusion) 

final_output = Reshape((map_height, map_width),name='squeeze_output')(final_output)

model = Model(inputs=main_inputs,outputs=final_output)

############################################################################################
# Training pipeline
############################################################################################

# Metric for our model
def rmse(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true)) ** 0.5

# Hyperparameters
epochs = 100
batch_size = 32
learning_rate = 0.0002
weight_decay = 5e-4
momentum = .9

# callbacks
model_path = 'saved_models'

# File were the best model will be saved during checkpoint     
model_file = os.path.join(model_path,'ph_crime-{val_loss:.4f}.h5')

# Early stop to avoid overfitting our model
early_stopping = EarlyStopping(monitor='val_rmse', patience=5, mode='min')

# Check point for saving the best model
check_pointer = ModelCheckpoint(model_file, monitor='val_rmse', mode='min',verbose=1, save_best_only=True)

# Logger to store loss on a csv file
csv_logger = CSVLogger(filename='ph_crime.csv',separator=',', append=True)

# Create Optimizer
optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

# Compile model for training
model.compile(optimizer, loss='mse' , metrics=[rmse])
model.summary()

# Train the model
history = model.fit(X_train, Y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=0.1,
                    callbacks=[early_stopping, check_pointer, csv_logger],
                    verbose=1)