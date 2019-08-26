# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 09:57:24 2019

@author: Angelo Antonio Manzatto
"""

##################################################################################
# Libraries
##################################################################################  
import os
import math

from datetime import datetime
from datetime import timedelta  

import numpy as np

import pandas as pd

import h5py

import matplotlib.pyplot as plt
import matplotlib.cm 

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

dataset_file = os.path.join(dataset_folder,'Crimes_2012-2016.csv')

column_names = ['date','division_record','date_crime','time_occurance','area','area_name','rd','crime_code','crime_description',
                'status','status_description','location','cross_street','geo_location']


df = pd.read_csv(dataset_file, names=column_names, skiprows=1, header=None)

############################################################################################
# Pre Process Dataset
############################################################################################

# Create a timestamp column in format YearMonthDayHour based on date_crime + time_occurance

df['ts'] = df[['date_crime','time_occurance']].apply(lambda x :str(x[0]).split('/')[2]+ 
                                                               str(x[0]).split('/')[0]+ 
                                                               str(x[0]).split('/')[1]+
                                                               str(x[1]).zfill(4)[0:2],axis=1)
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

# Get boundaries of latitude and longitude position
min_lat = df.lat.min()
max_lat = df.lat.max()
min_lon = df.lon.min()
max_lon = df.lon.max()

'''
Min Lat: -118.8551 , Max Lat: -117.6596 (x)
Min Lon: 33.3427 , Max Lon: 34.8087 (y)

'''
# Check minimum and maximum latitude and longitude
print("Min Lat: {0} , Max Lat: {1}".format(min_lat,max_lat))
print("Min Lon: {0} , Max Lon: {1}".format(min_lon,max_lon))

# Define map grid division for width (latitude) and height (longitude)
map_width_grid = 16
map_height_grid = 16
map_width_length = max_lat - min_lat 
map_height_length = max_lon - min_lon

# Find matrix index based on latitude or longitude coordinate
def coord_to_indice(coord,minimum,n_cells, map_length):
    '''
    coord - a cordinate of latitude OR longitude
    minimum - minimum latitude OR longitude
    n_cells - number of cells in a grid in latitude OR longitude direction
    map_length - length on latitude OR longitude direction
    '''
    return int((coord - minimum) * ( n_cells -1) / map_length)

# Find grid position given a longitude / latitude coordinate based on grid division
df['grid_x'] = df['lat'].apply(lambda x:coord_to_indice(x,min_lat, map_width_grid, map_width_length))
df['grid_y'] = df['lon'].apply(lambda x:coord_to_indice(x,min_lon, map_height_grid, map_height_length))

agg_columns = ['day','month','year','hour','square_h','square_w']
global_incident_heatmap = df.groupby(['grid_x','grid_y'],as_index=True).size()
global_incident_heatmap = global_incident_heatmap.reset_index()
global_incident_heatmap.rename(columns ={0:'total_crimes'},inplace=True)

global_incident_matrix = np.zeros((map_height_grid,map_width_grid))

# Fill each square of the heatmap matris with the total number of crimes commited
for _, incident_row in global_incident_heatmap.iterrows():
    
    global_incident_matrix[incident_row['grid_x']][incident_row['grid_y']] = incident_row['total_crimes']

global_incident_matrix = global_incident_matrix.astype(int)

# Plot global matrix
f, ax = plt.subplots()
f.set_figheight(12)
f.set_figwidth(12)

ax.matshow(global_incident_matrix, cmap=plt.cm.Blues)

ax.add_patch(plt.Rectangle((-0.5, -0.5),16, 16, color='red', fill=True, linewidth=2))
ax.add_patch(plt.Rectangle((1.5, 0.5),10, 9, color='yellow', fill=True, linewidth=2))
ax.add_patch(plt.Rectangle((2.5, 1.5),8, 7, color='green', fill=True, linewidth=2))

for i in range(map_width_grid-1):
    for j in range(map_height_grid-1):
        c = global_incident_matrix[j,i]
        ax.text(i, j, str(c), va='center', ha='center')
        
        
# Plot global incident matrix with selected coordinate boundaries
        
x_min = 3
x_max = 12
y_min = 2
y_max = 10
 
f, ax = plt.subplots()
f.set_figheight(8)
f.set_figwidth(8)
ax.matshow(global_incident_matrix[y_min:y_max,x_min:x_max], cmap=plt.cm.Blues)
for i in range(x_max - x_min - 1):
    for j in range(y_max - y_min -1):
        c = global_incident_matrix[j+ y_min,i + x_min]
        ax.text(i, j, str(c), va='center', ha='center')
        
# Convert boundaries to latitude and lontitude for selection
        
lat_ratio = map_width_length / map_width_grid    
lon_ratio = map_height_length / map_height_grid  

print("Lat ratio: {0} , Lon ratio: {1}".format(lat_ratio,lon_ratio))

new_min_lat = min_lat + x_min * lat_ratio
new_max_lat = max_lat - x_max * lat_ratio
new_min_lon = min_lon + y_min * lon_ratio
new_max_lon = max_lon - y_max * lon_ratio  

# Check minimum and maximum latitude and longitude
print("Min Lat: {0} , New Min Lat: {1}".format(min_lat,new_min_lat))
print("Max Lat: {0} , New Max Lat: {1}".format(max_lat,new_max_lat))
print("Min Lon: {0} , New Min Lon: {1}".format(min_lon,new_min_lon))
print("Max Lon: {0} , New Max Lon: {1}".format(max_lon,new_max_lon))
 
# Normalize latitude and longitude between [-1,1]
df['lat_n'] = 0.0
df['lon_n'] = 0.0

df['lat_n'] = df['lat'].apply(lambda x: 2. * (x - min_lat)/(max_lat - min_lat)-1.)
df['lon_n'] = df['lon'].apply(lambda x: 2. * (x - min_lon)/(max_lon - min_lon)-1.)

min_norm_lat = df.lat_n.min()
max_norm_lat = df.lat_n.max()
min_norm_lon = df.lon_n.min()
max_norm_lon = df.lon_n.max()

# Check minimum and maximum latitude and longitude
print("Min Norm Lat: {0} , Max Norm Lat: {1}".format(min_norm_lat,max_norm_lat))
print("Min Norm Lon: {0} , Max Norm Lon: {1}".format(min_norm_lon,max_norm_lon))

# Select just categories where the value counts are abore 6500

crime_description_count = df.crime_description.value_counts().rename('desc_count')

print("Total crime descriptions: {0}".format(len(crime_description_count)))

df = df.merge(crime_description_count.to_frame(),
                                left_on='crime_description',
                                right_index=True)

crime_threshold = 1000
df = df[df.desc_count > 1000]

# Crime description after applying threshold
crime_description_count = df.crime_description.value_counts().rename('desc_count')
print("Total crime descriptions: {0}".format(len(crime_description_count)))

# Classify the crimes into a smaller list of categories
df['category'] = ''

df.loc[(df.crime_description.str.contains('MOTOR')) & (~df.crime_description.isnull()), 'category'] = 'motor'
df.loc[(df.crime_description.str.contains('BURGLARY')) & (~df.crime_description.isnull()), 'category'] = 'burglary'
df.loc[(df.crime_description.str.contains('HOMICIDE')) & (~df.crime_description.isnull()), 'category'] = 'homicide'

subset = df[(df.crime_description.str.contains('DRUG')) & (~df.crime_description.isnull()) ]
subset_count = subset.crime_description.value_counts()
sample = df[df.category =='motor']

# Create a separated dataset with the only information needed for crime prediction models
df_crime = df[['crime_code','lat_n','lon_n','ts']].copy()

# Create date features
df_crime['year'] = df_crime['ts'].apply(lambda x: int(x[0:4]))
df_crime['month'] = df_crime['ts'].apply(lambda x: int(x[4:6]))
df_crime['day'] = df_crime['ts'].apply(lambda x: int(x[6:8]))
df_crime['hour'] = df_crime['ts'].apply(lambda x: int(x[8:10]))

# Count the number of crimes for each category so we will filter later 
crime_types = df_crime.crime_code.value_counts()

print("Number of categories of crimes: {0}".format(len(crime_types)))

# Plot just the categories that have occurances highter than the threshold
threshold = 10000
mask = crime_types > threshold
crime_types.loc[mask].plot(kind='bar')

# Filter our dataset by the top 10 crime types