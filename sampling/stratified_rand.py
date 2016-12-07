from __future__ import division
from collections import defaultdict
import numpy as np
from base_sample import BaseSample
from sklearn.cluster import AffinityPropagation as AP
import pandas as pd
from collections import Counter

class StratifiedRandom(BaseSample):
    
    def __init__(self, data_frame, number_arms=2):
        super(StratifiedRandom, self).__init__(data_frame, number_arms)

    
    def create_stratum(self, column_names, **kwargs):
        '''
        Use affinity propagation to find number of strata for each column. 
        column_names is a list of the covariates to be split into strata and 
        used for classification. This funciton adds a column to the data frame
        for each column as column_name_strata that gives the strata designation
        for that variable.  The whole data frame is returned.
        '''

        for colname in column_names:
            X = self.data[colname].reshape(-1, 1)
            
            if np.isnan(X).any():
                raise ValueError("There are NaN values in self.data[%s] that the \
                                  clustering algorithm can't handle" % colname)
                                  
            elif np.unique(self.data[colname]).shape[0] <=2:
                string_name = colname+'_strata'
                self.data[string_name] = self.data[colname].astype(int)
        
            else:
                af_model = AP(**kwargs)
                strata_groups = af_model.fit(X)
                
                #cluster_centers_indices = af.cluster_centers_indices_
                #n_clusters_ = len(cluster_centers_indices)
                
                string_name = colname+'_strata'
                self.data[string_name] = strata_groups.labels_
                
        return self.data
        
    
    #In the main function, you need to call create_stratum before create_unique_strata       
    def create_unique_strata(self, column_names):
        '''
        The input should be self.data that has had the strata for each column
        name assigned and had a pre-seeded randomization, meaning each arm
        has at least one randomly assigned participant.
        '''
    
        #Create a column to store concatenated strata strings for each data point
        self.data['strata_string'] = np.ones(len(self.data))*np.nan    
        
        #Initialize variables to be filled in during the loop        
        strata_unique = {}     
        
        #Loop through data points and create their strata strings
        for ind in self.data.index.values:
            similar_val = ''
            for colname in column_names:
                string_name = colname+'_strata'
                similar_val += str(self.data[string_name].loc[ind])
  
            
            #Add the total strata string for that data point
            self.data['strata_string'].set_value(ind,similar_val)
            
            #If the strata string exists, continue. If not, assign it a new value
            if similar_val in list(strata_unique.keys()):
                strata_unique[similar_val].append(ind)
                continue
            else:
                strata_unique[similar_val] = [ind]
        
        return (strata_unique, self.data)
            
       
    def count_arm_assignments(self, strata_unique, key):
        '''
        For each unique strata, count how many are assigned to each arm.
        '''
        #Initialize arm_tally that is the same length as the number of arms
        arm_tally = np.zeros(self.n_arms)              
        
        #Loop through the values in the unique strata and count how many are in each arm
        for value in strata_unique[key]:
            
            #If it is not NaN, add one to the arm_tally for the data point's arm assignment
            if np.isnan(self.data['arm_assignment'][value]) == False:
                arm_tally[int(self.data['arm_assignment'][value]-1)] += 1;
        
        return arm_tally
        
           
           
    def assign_arms(self, column_names, percent_nan = 0.05):
        '''
        Loop through unique strata and assign each data point to an arm.
        '''
        #clear all values with NaNs
        self.data = self.nan_finder(column_names, percent_nan)        
        
        #call create_stratum to create strata for each chosen covariate 
        self.data = self.create_stratum(column_names)
        
        #combine the covariate strata into a unique strata identifier
        (strata_unique, self.data) = self.create_unique_strata(column_names)        
        
        #initiate an empty column in the data frame for arm assignments
        self.data['arm_assignment'] = np.ones(len(self.data))*np.nan   
        
        #Loop through the uniqie strata       
        for key in strata_unique.keys():
            #Loop through the values in the unique stratum
            for value in strata_unique[key]:
                #update the arm_tally based on new assignments
                arm_tally = self.count_arm_assignments(strata_unique, key);
                
                #if more than one arm is empty, choose randomly
                if np.where(arm_tally==0)[0].shape[0] > 1:
                    self.data['arm_assignment'].set_value(
                        value,np.random.choice(list(range(self.n_arms))) + 1
                    )
                    
                #if one is empty, choose that arm assignment
                elif np.where(arm_tally==0)[0].shape[0] == 1:
                    self.data['arm_assignment'].set_value(
                        value, np.where(arm_tally==0)[0][0] + 1
                    )
                    
                #If no arms are empty, choose arm with minimum data points assigned to it
                elif np.where(arm_tally==0)[0].shape[0] < 1:
                    self.data['arm_assignment'].set_value(
                        value, np.where(np.min(arm_tally))[0][0] + 1
                    )
                
                #Something went wrong
                else: 
                    raise ValueError(
                        "arm_tally is empty, meaning there is something \
                        wrong with your unique strata dictionary."
                        )
        
        return self.data
        
