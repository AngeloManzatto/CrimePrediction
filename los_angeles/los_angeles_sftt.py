# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 16:08:46 2019

@author: Angelo Antonio Manzatto
"""

##################################################################################
# Libraries
##################################################################################  
import os
from datetime import datetime
from datetime import timedelta  
import numpy as np

import pandas as pd

import h5py

import matplotlib.pyplot as plt

from keras.models import Model

from keras.layers import Input, Activation, Average,Reshape, Flatten, Lambda, Dense
from keras.layers import Conv2D ,MaxPooling2D, BatchNormalization 
from keras.layers import LSTM
from keras.layers import TimeDistributed

import keras.backend as K

from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping

from keras.optimizers import Adam

np.random.seed(42) # My nickname Recruta42

############################################################################################
# Classes and Methods
############################################################################################
class CoordToIndex():
    
    def __init__(self, min_coords, max_coords, grid=(16,16)):
        
        '''
        min_coords - minimum latitude, minimum longitude in tuple format (min_lat, min_lon)
        max_coords - maximum latitude, maximum longitude in tuple format (max_lat, max_lon)
        grid - number of cells to divide a grid in tuple format ( height, width)
        '''
        
        self.min_lat, self.min_lon = min_coords
        self.max_lat, self.max_lon = max_coords
        self.grid_h, self.grid_w = grid
        self.map_width_length = abs(self.max_lat - self.min_lat)
        self.map_height_length = abs(self.max_lon - self.min_lon)
         
    def to_idx(self, coords):
         
        lat, lon = coords
        
        '''
        We have to reverse the longitude coordinate since the orientation is from botton to up instead of up to down like
        on images
        '''
        grid_h = abs(int((lon - max_lon) * ( self.grid_h - 1) / self.map_height_length))
        grid_w = int((lat - min_lat) * ( self.grid_w - 1) / self.map_width_length)

        return grid_h, grid_w
    
def find_best_bounding_map(df, grid=(16,16), threshold = 200):
    '''
    df - dataframe with columns lat and log
    grid - number of grid cells in format (height, width)
    threshold - minimum number to search for reducing
    '''
    
    # Make a copy for safety
    df_copy = df.copy()
    
    # Make sure that 
    assert('lat' in df and 'lon' in df)
    
    # Define map grid division for width (latitude) and height (longitude)
    grid_h = grid[0]
    grid_w = grid[1]
    
    # Get boundaries of latitude and longitude position
    min_lat = df_copy.lat.min()
    max_lat = df_copy.lat.max()
    min_lon = df_copy.lon.min()
    max_lon = df_copy.lon.max()
    
    coord_to_index = CoordToIndex((min_lat,min_lon),(max_lat,max_lon),(grid_h,grid_w))     
    
    df_copy[['grid_h','grid_w']] = df_copy[['lat','lon']].apply(lambda x:coord_to_index.to_idx((x[0],x[1])),axis=1, result_type="expand")

  
    # Create a global incident heatmap by summing values with the same grid x and grid y location
    global_incident_heatmap = df_copy.groupby(['grid_w','grid_h'],as_index=True).size()
    global_incident_heatmap = global_incident_heatmap.reset_index()
    global_incident_heatmap.rename(columns ={0:'total'},inplace=True)
    
    global_incident_matrix = np.zeros((grid_h, grid_w))
    
    # Fill each square of the heatmap matris with the total number of crimes commited
    for _, incident_row in global_incident_heatmap.iterrows():
        
        global_incident_matrix[incident_row['grid_h']][incident_row['grid_w']] = incident_row['total']
    
    global_incident_matrix = global_incident_matrix.astype(int)
    
    w_sum = global_incident_matrix.sum(axis=0)
    h_sum = global_incident_matrix.sum(axis=1)
    
    h_min = np.argmin(h_sum < threshold) 
    h_max = grid_h - np.argmin(h_sum[::-1] < threshold) - 1      
    w_min = np.argmin(w_sum < threshold)
    w_max = grid_w - np.argmin(w_sum[::-1] < threshold) - 1 
        
    # Select df based on new boundaries with only relevant occurrences
    selected_df = df_copy[(df_copy.grid_w >= w_min) & 
                          (df_copy.grid_w <= w_max) & 
                          (df_copy.grid_h >= h_min) & 
                          (df_copy.grid_h <= h_max)]
    
    # New heatmap with just the selected dataframe
    selected_global_incident_heatmap = selected_df.groupby(['grid_w','grid_h'],as_index=True).size()
    selected_global_incident_heatmap = selected_global_incident_heatmap.reset_index()
    selected_global_incident_heatmap.rename(columns ={0:'total'},inplace=True)

    # Zero indexing to origin
    selected_global_incident_heatmap['grid_h'] = selected_global_incident_heatmap['grid_h'] - h_min
    selected_global_incident_heatmap['grid_w'] = selected_global_incident_heatmap['grid_w'] - w_min
    
    selected_global_incident_matrix = np.zeros(( h_max - h_min + 1, w_max - w_min + 1))
    
    # Fill each square of the heatmap matris with the total number of crimes commited
    for _, incident_row in selected_global_incident_heatmap.iterrows():
        
        selected_global_incident_matrix[incident_row['grid_h']][incident_row['grid_w']] = incident_row['total']
    
    selected_global_incident_matrix = selected_global_incident_matrix.astype(int)
    
    # Plot Heatmap matrices
    f, (ax1, ax2) = plt.subplots(2, 1)
    f.set_figheight(24)
    f.set_figwidth(24)
    
    ax1.set_title('Original Heatmap')
    ax1.matshow(global_incident_matrix, cmap='winter')
    
    ax2.set_title('Selected Heatmap')
    ax2.matshow(selected_global_incident_matrix, cmap='winter')
    
    for i in range(grid_w-1):
        for j in range(grid_h-1):
            c = global_incident_matrix[j,i]
            ax1.text(i, j, str(c), va='center', ha='center')
            
    for i in range(w_max-w_min+1):
        for j in range(h_max-h_min+1 ):
            c = selected_global_incident_matrix[j,i]
            ax2.text(i, j, str(c), va='center', ha='center')
    
    selected_df = selected_df.drop(['grid_h','grid_w'],1)
    
    return selected_df

############################################################################################
# Load Raw Dataset
############################################################################################
  
dataset_folder = 'dataset'

dataset_file = os.path.join(dataset_folder,'Crimes_2012-2016.csv')

column_names = ['date','division_record','date_crime','time_occurance','area','area_name','rd','crime_code','crime_description',
                'status','status_description','location','cross_street','geo_location']


df = pd.read_csv(dataset_file, names=column_names, skiprows=1, header=None)

############################################################################################
# Pre Process Dataset
############################################################################################

#################################################
# Create timestamp
#################################################

# Create a timestamp column in format YearMonthDayHour based on date_crime + time_occurance
df['ts'] = df[['date_crime','time_occurance']].apply(lambda x :str(x[0]).split('/')[2]+ 
                                                               str(x[0]).split('/')[0]+ 
                                                               str(x[0]).split('/')[1]+
                                                               str(x[1]).zfill(4)[0:2],axis=1)

#################################################
# Create latitude and longitude
#################################################

# Drop columns without latitude and longitude information
df.dropna(subset=['geo_location'],inplace=True)

# Remove "(", ")" from geo location column
df['geo_location'] = df['geo_location'].astype(str)
df['geo_location'] = df['geo_location'].apply(lambda x: x.replace("(","").replace(")",""))

# Create latitude and longitude column based on geo_location column
df[['lon','lat']] = df.geo_location.str.split(",",expand=True)

df['lon'] = df['lon'].astype(float)
df['lat'] = df['lat'].astype(float)

# Drop columns with 0.0 longitude and latitude since they are not feasible
df = df[(df.lon != 0.0) & (df.lat != 0.0)]

#################################################
# Drop duplicates
#################################################

df.drop_duplicates(subset=['lat','lon','ts'], keep='first', inplace=True)

#################################################
# Define best grid for event occurencces
#################################################

# Get boundaries of latitude and longitude position
min_lat = df.lat.min()
max_lat = df.lat.max()
min_lon = df.lon.min()
max_lon = df.lon.max()

'''
http://bboxfinder.com/#33.342700,-118.855100,34.808700,-117.659600

Min Lat: -118.8551 , Max Lat: -117.6596 (x)
Min Lon: 33.3427 , Max Lon: 34.8087 (y)

Image coordinates are different than latitude, longitude coordinates.

(min_x = min_lat, min_y = max_lon)
(max_x = max_lat, max_y = min_lon)
'''
# Check minimum and maximum latitude and longitude
print("Min Lat: {0} , Max Lat: {1}".format(min_lat,max_lat))
print("Min Lon: {0} , Max Lon: {1}".format(min_lon,max_lon))


# Select just the portion of our dataset that has relevant number of occurencces when dividing the map in a grid   
selected_df = find_best_bounding_map(df, grid=(32,32), threshold = 100)

# Create our final dataset 
la_crime_df = selected_df[['ts','lat','lon']]

# Sort values by time stamp
la_crime_df = la_crime_df.sort_values(by='ts')

# Save processed dataset
la_crime_df.to_csv(os.path.join(dataset_folder,'la_crime_2012-2016.csv'), index=False)

############################################################################################
# Load Processed Dataset
############################################################################################
  
dataset_folder = 'dataset'

dataset_file = os.path.join(dataset_folder,'la_crime_2012-2016.csv')

la_crime_df = pd.read_csv(dataset_file)

# Create date colums
la_crime_df['date'] = pd.to_datetime(la_crime_df['ts'],format='%Y%m%d%H')

# Get boundaries of latitude and longitude position
min_lat = la_crime_df.lat.min()
max_lat = la_crime_df.lat.max()
min_lon = la_crime_df.lon.min()
max_lon = la_crime_df.lon.max()

# Check minimum and maximum latitude and longitude
print("Min Lat: {0} , Max Lat: {1}".format(min_lat,max_lat))
print("Min Lon: {0} , Max Lon: {1}".format(min_lon,max_lon))

# Create a mapping between (latitude , longitude) and (grid horizontal , grid vertical)
grid_h = 32
grid_w = 32

coord_to_index = CoordToIndex((min_lat,min_lon),(max_lat,max_lon),(grid_h,grid_w))     
    
la_crime_df[['grid_h','grid_w']] = la_crime_df[['lat','lon']].apply(lambda x:coord_to_index.to_idx((x[0],x[1])),axis=1, result_type="expand")

############################################################################################
# Create Dataset (Heat map, Timestamp)
############################################################################################

# Aggregate (time, square) to create the incident map sum (crimes)
agg_columns = ['ts','grid_h','grid_w']
incident_map = la_crime_df.groupby(agg_columns,as_index=True).size()
incident_map = incident_map.reset_index()    
incident_map.rename(columns ={0:'total'},inplace=True)

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
    heatmap = np.zeros((grid_h , grid_w))
    
    # Select just crimes on the timestamp frame
    incident_map_ts = incident_map[incident_map.ts == ts_index]
    
    # Fill each square of the heatmap matris with the total number of crimes commited
    for _, incident_row in incident_map_ts.iterrows():
        
        heatmap[incident_row['grid_h']][incident_row['grid_w']] = incident_row['total']

    data.append(heatmap)
    timestamps.append(incident_row['ts'])
    
    count+=1

data = np.asarray(data)
timestamps = np.asarray(timestamps).astype(np.string_) # We have to convert to byte datatype so we can save the .h5 file

# Save dataset preprocessed as .h5 file 
h5f = h5py.File(os.path.join(dataset_folder,'la_crime_heatmap_32x32_hourly.h5'), 'w')
h5f.create_dataset('data', data=data)
h5f.create_dataset('timestamps', data=timestamps)
h5f.close()

############################################################################################
# Load H5 Dataset
############################################################################################
dataset_folder = 'dataset'

dataset_file = os.path.join(dataset_folder,'la_crime_heatmap_32x32_hourly.h5')

# Load dataset file
f = h5py.File(dataset_file)
data = f['data'][()]
timestamps = f['timestamps'][()]

# Convert timestamps from ASCII format to string
formated_timestamps = []
for ts in timestamps:
    formated_timestamps.append(ts.decode("utf-8"))
    
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
# Create daily accumulated data
############################################################################################
accumulated_data = []
daily_timestamp = []

current_date = ts_list[0].date()
last_date = ts_list[-1].date()

heatmap_shape = data[0].shape

while(current_date != last_date):
    
    print("Processing: {0}".format(current_date))
    selected_date_keys = [key for key in ts_dict if key.date() == current_date]
    
    # Just process if we have any record on this day
    if len(selected_date_keys) > 0:
        
        daily_timestamp.append( str(current_date.year).zfill(2)  + str(current_date.month).zfill(2) + str(current_date.day))
        
        daily_heatmap = np.zeros(heatmap_shape)
        
        for selected_key in selected_date_keys:
            daily_heatmap += data[ts_dict[selected_key]]
            
        accumulated_data.append(daily_heatmap)
    
    current_date = current_date + timedelta(days=1)

accumulated_data = np.asarray(accumulated_data)

# Plot some samples
n_samples = 5

for i in range(n_samples):

    # randomly select a sample
    idx = np.random.randint(0, len(accumulated_data))
    
    heatmap = accumulated_data[idx]

    date = datetime.strptime(daily_timestamp[idx], '%Y%m%d')
    
    # define the size of images
    f, ax = plt.subplots()
    f.set_figwidth(6)
    f.set_figheight(6)
    
    ax.set_title("Heatmap: {0}".format(date))
    ax.imshow(heatmap, cmap='jet')
    
############################################################################################
# Create Train / Target data
############################################################################################
    
period = 30

X, Y, Y_timestamp = [], [], []

# We construct the X, y datasets based on a reversed time interval, from the latest trend to starting closeness
for i in range(period + 1, len(daily_timestamp)):
  
    X.append(accumulated_data[i- period - 1: i-1])
    Y.append(accumulated_data[i])
    Y_timestamp.append(daily_timestamp[i])
    
X = np.asarray(X)
Y = np.asarray(Y)

 # Correct channels last
X = np.transpose(X,(0,2,3,1))
X = np.expand_dims(X,axis=1)

print("X shape: ", X.shape)
print("Y shape: ", Y.shape)

############################################################################################
# Split dataset into Train / Test
############################################################################################

days_test = 100

# Split dataset into training / test sets
X_train, Y_train, = X[:-days_test], Y[:-days_test]
X_test,  Y_test = X[-days_test:], Y[-days_test:]

X_train = np.expand_dims(X_train,axis=-1)
X_test = np.expand_dims(X_test,axis=-1)
    
# Time stamp split so we can track the period
timestamp_train, timestamp_test = Y_timestamp[:-days_test], Y_timestamp[-days_test:]

print("X Train size: ", len(X_train))
print("X Test size: ", len(X_test))    

############################################################################################
# SFTT Network
############################################################################################

############################################################################################
# ResNet Identity Block
############################################################################################
def residual_block(inputs, filters, block_id):
    
    f1, f2, f3 = filters
    
    pool = MaxPooling2D(pool_size=(2,2),name='block_' + str(block_id) + '_max_pooling')(inputs)

    x = Conv2D(f1, kernel_size=(3,3), padding='same', use_bias=False, kernel_initializer='he_normal', name='block_' + str(block_id) + '_conv_conv2d_1')(pool)
    x = BatchNormalization(name='block_' + str(block_id) + '_conv_batch_1')(x)
    x = Activation('relu', name='block_' + str(block_id) + '_expand_relu')(x)
    
    x = Conv2D(f2, kernel_size=(1,1), padding='same', use_bias=False, kernel_initializer='he_normal', name='block_' + str(block_id) + '_conv_conv2d_2')(x)
    x = BatchNormalization(name='block_' + str(block_id) + '_conv_batch_2')(x)
    x = Activation('relu', name='block_' + str(block_id) + '_depthwise_relu')(x)
    
    x = Conv2D(f3, kernel_size=(3,3), padding='same', use_bias=False, kernel_initializer='he_normal', name='block_' + str(block_id) + '_project_conv2d')(x)
    x = BatchNormalization(name='block_' + str(block_id) + '_project_batch')(x)
    
    shortcut = Conv2D(f3, kernel_size=(3,3), padding='same', strides=(1,1), use_bias=False, kernel_initializer='he_normal', name='block_' + str(block_id) + '_shortcut_conv2d')(pool)
    shortcut = BatchNormalization(name='block_' + str(block_id) + '_shortcut_batch')(shortcut)
    
    average = Average(name='block_' + str(block_id) + '_average')([shortcut,x])
    output = Activation('relu',name='block_' + str(block_id) + '_average_relu')(average)
    
    return output

def SFTT_model(input_shape = (32,32,30)):
    
    map_height, map_width, feature_maps = input_shape

    ######################################################
    # Convolutional
    ######################################################
    
    Image = Input(shape=(map_height, map_width, feature_maps), name='input')
    
    x = Conv2D(32, kernel_size=(1,1),strides=(1,1), padding='same', kernel_initializer='he_normal', name = 'conv1')(Image)
    x = BatchNormalization(name = 'batch_1')(x)
    x = Activation('relu',name='relu_1')(x)
    
    x = residual_block(x,filters = [32, 32, 128], block_id = 0)
    x = residual_block(x,filters = [64, 64, 256], block_id = 1)
    x = residual_block(x,filters = [128, 128, 512], block_id = 2)
    
    sf_output = MaxPooling2D(pool_size=(2,2),name='final_max_pooling')(x)
    
    sf_output = Flatten()(sf_output)
    
    conv_model  = Model(inputs=Image,outputs=sf_output)
    
    ######################################################
    # LSTM 
    ######################################################
    
    input_sequences = Input(shape=(None,map_height,map_width,feature_maps))
    time_distribute = TimeDistributed(Lambda(lambda x: conv_model(x)))(input_sequences) 
    lstm = LSTM(512, return_sequences = True)(time_distribute)
    lstm = LSTM(512, return_sequences = True)(lstm)
    fc1 = Dense(map_height*map_width, activation='relu')(lstm)
    final_output = Reshape((map_height, map_width), name='squeeze_output')(fc1)
    
    sftt_model = Model(inputs=[input_sequences], outputs=[final_output])
    
    return sftt_model

############################################################################################
# Training pipeline
############################################################################################

# Metric for our model
def rmse(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true)) ** 0.5

# Hyperparameters
epochs = 40
batch_size = 32
learning_rate = 0.0005
weight_decay = 5e-4
momentum = .9

# callbacks
model_path = 'saved_models'

# File were the best model will be saved during checkpoint     
model_file = os.path.join(model_path,'la_crime-{val_loss:.4f}.h5')

# Early stop to avoid overfitting our model
early_stopping = EarlyStopping(monitor='val_rmse', patience=5, mode='min')

# Check point for saving the best model
check_pointer = ModelCheckpoint(model_file, monitor='val_rmse', mode='min',verbose=1, save_best_only=True)

# Logger to store loss on a csv file
csv_logger = CSVLogger(filename='la_crime.csv',separator=',', append=True)

# Heatmap parameters
map_height = 32
map_width = 32
period = 30    

# Create ST-ResNet Model
model = SFTT_model(input_shape = (map_height,map_width,period))

# Create Optimizer
optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

# Compile model for training
model.compile(optimizer, loss='mse' , metrics=[rmse])
model.summary()