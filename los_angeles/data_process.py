# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 09:57:24 2019

@author: Angelo Antonio Manzatto

Article: https://arxiv.org/pdf/1707.03340.pdf

References and credits:
Bao Wang, Duo Zhang, Duanhao Zhang, P. Jeffrey Brantingham and Andrea L. Bertozzi
    
Dataset: https://www.kaggle.com/kingburrito666/los-angeles-crime    
"""

##################################################################################
# Libraries
##################################################################################  
import os

import numpy as np

import pandas as pd

import h5py

import matplotlib.pyplot as plt

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

# Filter just the 2015 crimes
la_crime_df = la_crime_df[(la_crime_df.date.dt.year == 2015) & (la_crime_df.date.dt.month > 6)]

# Get boundaries of latitude and longitude position
min_lat = la_crime_df.lat.min()
max_lat = la_crime_df.lat.max()
min_lon = la_crime_df.lon.min()
max_lon = la_crime_df.lon.max()

# Check minimum and maximum latitude and longitude
print("Min Lat: {0} , Max Lat: {1}".format(min_lat,max_lat))
print("Min Lon: {0} , Max Lon: {1}".format(min_lon,max_lon))

# Create a mapping between (latitude , longitude) and (grid horizontal , grid vertical)
grid_h = 16
grid_w = 16

coord_to_index = CoordToIndex((min_lat,min_lon),(max_lat,max_lon),(grid_h,grid_w))     
    
la_crime_df[['grid_h','grid_w']] = la_crime_df[['lat','lon']].apply(lambda x:coord_to_index.to_idx((x[0],x[1])),axis=1, result_type="expand")

############################################################################################
# Exploratory Data Analysis
############################################################################################

# Plot # crime for last two weeks of 2015
selected_date = pd.date_range(la_crime_df.date.max() - pd.to_timedelta(24*14, unit='h'), la_crime_df.date.max(), freq='H')
crimes_last_two_weeks = la_crime_df[la_crime_df["date"].isin(selected_date)]
crimes_last_two_weeks = crimes_last_two_weeks.groupby(['date'],as_index=True).size()
crimes_last_two_weeks = crimes_last_two_weeks.reset_index()
crimes_last_two_weeks.rename(columns ={0:'total'},inplace=True)

# Get 24 hours accumulated crime rate
crimes_last_two_weeks['accumulated'] = -99
for i in range(len(crimes_last_two_weeks)):
    
    # The first occurance
    if i == 0:
        crimes_last_two_weeks.loc[i,'accumulated'] = crimes_last_two_weeks.loc[i,'total']
    elif i > 0 and crimes_last_two_weeks.loc[i,'date'].hour < crimes_last_two_weeks.loc[i-1,'date'].hour:
        crimes_last_two_weeks.loc[i,'accumulated'] = crimes_last_two_weeks.loc[i,'total']
    else:
        crimes_last_two_weeks.loc[i,'accumulated'] = crimes_last_two_weeks.loc[i - 1,'accumulated'] + crimes_last_two_weeks.loc[i,'total']
 
# Plot total vs accumulates 
f, (ax1, ax2) = plt.subplots(1, 2)
f.set_figwidth(14)    
    
ax1 = crimes_last_two_weeks['total'].plot(linewidth=0.8, ax=ax1)
ax1.set_xlabel("time")
ax1.set_ylabel("# crimes")
ax1.set_title("Hourly crimes for the last two weeks")

ax2 = crimes_last_two_weeks['accumulated'].plot(linewidth=0.8, ax=ax2)
ax2.set_xlabel("time")
ax2.set_ylabel("# accu crimes")
ax2.set_title("Hourly accumulated (24h) crimes for the last two weeks")

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
h5f = h5py.File(os.path.join(dataset_folder,'la_crime_heatmap.h5'), 'w')
h5f.create_dataset('data', data=data)
h5f.create_dataset('timestamps', data=timestamps)
h5f.close()