from __future__ import division
from collections import defaultdict
from itertools import combinations
import numpy as np
from base_sample import BaseSample



class SimpleRandomization(BaseSample):
    
    def __init__(self, data_frame, number_arms=2):
        super(SimpleRandomization, self).__init__(data_frame, number_arms)


    def randomize(self):
        '''
        Simple randomization to randomly assign participants to arm of study
        '''
        
        # Add arm assignment column to dataframe or re-initialize to nan
        self.data['arm_assignment'] = np.ones(len(self.data))*np.nan         
        # Grab array of indicies
        df_inds = self.data.index.values        
        
        # Size of population
        n_pop = len(df_inds)
        
        # Determine number of participants per arm  
        n_per_arm = np.floor(n_pop/self.n_arms)
        
        for arm in range(1,self.n_arms):
            random_inds = np.random.choice(df_inds, size=n_per_arm)
            self.data['arm_assignment'].loc[random_inds] = arm
            # Remove these participants from list
            del_inds = df_inds.searchsorted(random_inds)
            df_inds = np.delete(df_inds, del_inds)            
        
        # Assign the remaining participants to last arm
        self.data['arm_assignment'].loc[df_inds] = self.n_arms
        
        return self.data


    def _randomize(self):
        '''
        Simple randomization to randomly assign participants to arm of study
        '''
        
        # Add arm assignment column to dataframe or re-initialize to nan
        self.data['arm_assignment'] = np.ones(len(self.data))*np.nan         
        # Grab array of indicies
        df_inds = self.data.index.values        
        
        n_pop = len(df_inds)
        
        while len(df_inds) >0:
            
            # Loop over arms
            for arm in range(1,self.n_arms+1):
                
                # Randomly choose participant
                random_ind = np.random.choice(df_inds)
                # Assign random participant to arm
                self.data['arm_assignment'].loc[random_ind]=arm
                # Remove participant from list
                del_ind = np.where(df_inds == random_ind)
                df_inds = np.delete(df_inds, del_ind)
                if len(df_inds)==0: 
                    break
        
        return self.data
    
