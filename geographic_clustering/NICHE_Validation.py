# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 10:15:54 2016

@author: alicampion
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.cluster import AffinityPropagation as AP
from collections import Counter
from osgeo import gdal, ogr

import geopandas as gp
from shapely.geometry import Polygon
import pysal as ps
from shapely.geometry import LineString
from matplotlib.colors import ListedColormap   



os.chdir('/Volumes/Lexar/Kimetrica/UNICEF/NICHE_Sampling')

#%%

#Read in data
full_data = pd.read_excel('KITUI_ DATA VERIFICATION.xlsx',0)

#%%
  
#Filter the data to get create AGES columns that describe the age in months for
#each child
for jj in np.array([1,2,3]):
    
    #Create string to reference the column for each child in birth order
    child_string = 'CHILD '+str(jj)+' DOB'
    age_string = 'AGES_'+str(jj)
    
    #Replace the 'n/a' with NaN values and attach to the dataframe
    filler = np.ones(np.where(full_data[child_string]=='n/a')
        [0].shape[0])*np.nan
    full_data.set_value(list(np.where(full_data[child_string]=='n/a')[0]),
                                      child_string,filler)
    
    #Create an empty vector for the age of each child in months
    full_data[age_string] = np.ones(full_data.shape[0])*np.nan
    
    #Loop through each participnt to calculate their age
    for ii in range(0,full_data[child_string].shape[0]):
        
        #If the age of the child exists, then calculate the age
        if type(full_data[child_string][ii]) != float:
            
            #Caclulste the age in days in reference to the start date
            full_data.set_value(ii,age_string,(dt.datetime(2017, 1, 10, 0, 0) 
                    - full_data[child_string][ii]).days)  
                    
    #Convert the age in days to months, round to the nearest month
    full_data.set_value(list(np.arange(0,full_data[age_string].shape[0])),
                                       age_string,np.round(
                                       full_data[age_string]/30))

#Now do the same for pregnant women and their expected delivery date (EDD)
for jj in np.array([1,2,3]):
    if (jj == 1):
        date_string = 'EDD'
    else:
        date_string = 'PW '+str(jj)+' EDD'
    edd_string = 'EDD_'+str(jj)
    
    #Replace 'n/a' Values with NaN values and attach to dataframe
    filler = np.ones(np.where(full_data[date_string]=='n/a')[0].shape[0])*np.nan
    full_data.set_value(list(np.where(full_data[date_string]=='n/a')[0]),
                                      date_string,filler)
                                      
    #Create empty vector to fill with ages                                 
    full_data[edd_string] = np.ones(full_data.shape[0])*np.nan
    
    #Cycle through all EDD's and calculate age in days
    for ii in range(0,full_data[date_string].shape[0]):
        if type(full_data[date_string][ii]) != float:
            full_data.set_value(ii,edd_string,(dt.datetime(2017, 1, 10, 0, 0) -
            full_data[date_string][ii]).days)      
    
    #Convert to months, rounding to nearest month    
    full_data.set_value(list(np.arange(0,full_data[edd_string].shape[0])),
                                       edd_string,np.round(
                                       full_data[edd_string]/30))




#%% Filter Data for correct ages

#Create Empyt X array of geo coordinates only
X = np.ones([full_data['LATITUDE'].shape[0],2])

#Cycle through the X array and fill in with lat and lon of households with 
#children under two or a pregnant woman
for ii in range(0,X.shape[0]):
    if full_data['NO. OF CHILDREN < 2YRS'][ii] > 0 or \
    full_data['HOUSEHOLD HAS PW'][ii] == 'YES':
        X[ii][1] = full_data['LATITUDE'][ii]
        X[ii][0] = full_data['LONGITUDE'][ii]
    else:
        continue

#Go throuhgh and isolate the various households that have any children over 2
a = np.where((full_data['LONGITUDE']>0)==False)[0].reshape(-1,1)
b = np.where(full_data['AGES_1']>=25)[0].reshape(-1,1)
c = np.where(full_data['AGES_2']>=25)[0].reshape(-1,1)
d = np.where(full_data['AGES_3']>=25)[0].reshape(-1,1)
e = np.unique(np.where(X==np.array([1,1]))[0]).reshape(-1,1)

#combine all of the households that have children over 2 into the array empty
empty = np.concatenate((a,b))
empty = np.concatenate((empty,c))
empty = np.concatenate((empty,d))
empty = np.concatenate((empty,e))

#Create an Xclean array that has deleted all of the entries in empy
Xclean = np.delete(X, (empty), axis=0)

#Create a data_clean matrix of all the households contained in Xclean
data_clean = pd.DataFrame(np.ones([Xclean.shape[0],full_data.shape[1]])*np.nan)
data_clean.columns = full_data.columns
for col in data_clean.columns:
    data_clean[col] = np.delete(full_data[col].reshape(-1,1), (empty.reshape(-1,1)), axis=0).reshape(data_clean[col].shape[0])

#Create affinity propagation model
af_model = AP(damping = 0.8)
groupings = af_model.fit(Xclean)

#Isolate cluster centers and number of clusters
cluster_centers_indices = groupings.cluster_centers_indices_
n_clusters_ = len(cluster_centers_indices)

#Create a column in data_clean to describe the cluster/group a 
#household was assigned to

data_clean['CLUSTERS'] = groupings.labels_
Counter(groupings.labels_)

#Export the cleaned, cllustered data to a csv
#data_clean.to_csv('cluster_assignments.csv')           
#%% Map the households

#Create a basemap of the admin zones, only plotting Kitui                 
basemap = gp.read_file('KEN_adm_shp/KEN_adm2.shp')
basemap= basemap[(basemap.NAME_1 == 'Kitui')]
basemap.plot(cmap='gray',figsize = [10,10])
plt.hold(True)

#Define the colormap
cmap = plt.cm.Paired((int(254/n_clusters_)*data_clean['CLUSTERS']))  

#Plot each household, coloring by group assignment
for ii in range(0,data_clean.shape[0]):            
    plt.plot(data_clean['LONGITUDE'][ii],data_clean['LATITUDE'][ii]*-1
            ,'ko',mfc = cmap[ii][0:3], alpha = 0.9, markersize = 10)

#This range wasn't plotting for some reason, so explicitly plot it
for ii in range(520,559):            
    plt.plot(data_clean['LONGITUDE'][ii],data_clean['LATITUDE'][ii]*-1
            ,'ko',mfc = cmap[ii][0:3], alpha = 0.9, markersize = 10)            

#Define cluster colors            
count_clusters = np.arange(0,n_clusters_)
cmap_clusters = plt.cm.Paired((int(254/n_clusters_))*count_clusters)  

#Add a title and axes labels
plt.title('Num. of Clusters: %.0f' % n_clusters_)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.hold(True)

#Plot the cluster centers with the appropriate colors
for ii in range(0,groupings.cluster_centers_.shape[0]):
    plt.plot(groupings.cluster_centers_[ii][0],groupings.cluster_centers_[ii][1]*-1,'*',
             markersize=18, mew = 1.5, mec = [1,1,1], mfc = cmap_clusters[ii][0:3])
plt.grid(True)

#Save the figure if you want 
#plt.savefig('clustered_map.pdf', bbox_inches='tight')
         
            
            #%%
#Get statistics

#35013
#34980.20
#20755.6
#20681.5
#20486.5
#20477.5
#10797.5
#10801.3
#10797.5
#10792.5

def getStats(data_clean, clus):
    inds = np.where(data_clean['CLUSTERS'] == clus)[0]
    num_children = data_clean['NO OF CHILDREN < 2 YEARS'][inds]
    ages_1 = data_clean['AGES_1']
    ages_2 = data_clean['AGES_2']
    return(num_children, ages_1, ages_2)
    
   

#%% Histogram of Pregnatn women
plt.figure(figsize = [8,5])
(counter, bins) = np.histogram(full_data['EDD_1'][~np.isnan(full_data['EDD_1'])],24)
width = 0.85
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, counter, align='center', width=width)
plt.title('Age in Months as of January 10, 2016')
plt.xlabel('Months')
plt.ylabel('Count')
plt.grid(True)
plt.savefig('allmonths.pdf', bbox_inches='tight')
plt.show()

#%% Histogram of all children and pregnant women

#Binned into 3-month intervals
plt.figure(figsize = [8,5])
bintest = np.array([-6,-3,0,3,6,9,12])

#Count each the first, second and third children/women in the household
#separately so you can pull out specific informaiton on each later

test = data_clean['AGES_1'][~np.isnan(data_clean['AGES_1'])]
(counter1, bins) = np.histogram(test[test<25],bintest)
test = data_clean['AGES_2'][~np.isnan(data_clean['AGES_2'])]
(counter2, bins) = np.histogram(test[test<25],bintest)
test = data_clean['AGES_3'][~np.isnan(data_clean['AGES_3'])]
(counter3, bins) = np.histogram(test[test<25],bintest)


test = data_clean['EDD_1'][~np.isnan(data_clean['EDD_1'])]
(counter4, bins) = np.histogram(test[test<25],bintest)
test = data_clean['EDD_2'][~np.isnan(data_clean['EDD_2'])]
(counter5, bins) = np.histogram(test[test<25],bintest)
test = data_clean['EDD_3'][~np.isnan(data_clean['EDD_3'])]
(counter6, bins) = np.histogram(test[test<25],bintest)

width = (bins[1]-bins[0])*.85
center = (bins[:-1] + bins[1:]) / 2

#Plot all participants, or a subset by modifying the counters 
plt.bar(center, counter1+counter2+counter3+counter4+counter5+counter6, align='center', width=width)
plt.title('Age in Months as of March 1, 2017')
plt.xlabel('Months')
plt.ylabel('Count')
plt.grid(True)
#plt.savefig('groupedmonths_allchildren_1221.pdf', bbox_inches='tight')
plt.show()


#%% consistency checks in the clustering

#Exploring the groupings and identifying those that may have been mis-clustered
#(i.e. those that have the same geographic descriptors but have different group 
#assignments, or those that look as if they should change groups )

mis_clustered = ['none']
counter_villages = Counter(data_clean['LOCATION NAME'])
all_villages = np.unique(list(Counter(data_clean['LOCATION NAME']).elements()))

for ii in range(0,len(counter_villages)):
    name = all_villages[ii]
    locs = np.where(data_clean['LOCATION NAME']==name)[0]
    
    cluster_assignments = np.array([]).reshape(-1,1)
    for jj in locs:
        temp = data_clean.loc[jj]
        cluster_assignments = np.concatenate((cluster_assignments,temp['CLUSTERS'].reshape(-1,1)))
    
    cluster_assignments = list(cluster_assignments)
    if cluster_assignments.count(cluster_assignments[0])!=len(cluster_assignments):
        mis_clustered = np.concatenate((mis_clustered,[name]))
        
names_mis_clustered = np.unique(list(Counter(mis_clustered).elements()))

i = 2
inds = np.where(data_clean['LOCATION NAME']==names_mis_clustered[i])[0]
print(inds)
data_clean.loc[inds]['CLUSTERS']


#%% Changes based on data clean - this will change depending on your clustering 
#algorithm and amount of data available
#
#data_clean.set_value(134,'CLUSTERS',0)
#data_clean.set_value(193,'CLUSTERS',13)
#data_clean.set_value(359,'CLUSTERS',8)
#data_clean.set_value(561,'CLUSTERS',12)
#data_clean.set_value(565,'CLUSTERS',12)
#data_clean.set_value(515,'CLUSTERS',8)
#data_clean.set_value(516,'CLUSTERS',8)
#data_clean.set_value(471,'CLUSTERS',7)
#data_clean.set_value(329,'CLUSTERS',5)
#data_clean.set_value(330,'CLUSTERS',5)
#
#data_clean.set_value(514,'CLUSTERS',8)
#data_clean.set_value(517,'CLUSTERS',8)
#data_clean.set_value(518,'CLUSTERS',8)
#data_clean.set_value(556,'CLUSTERS',8)
#data_clean.set_value(468,'CLUSTERS',7)

