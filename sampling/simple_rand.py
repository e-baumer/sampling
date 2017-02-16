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
            random_inds = np.random.choice(df_inds, size=int(n_per_arm))
            self.data['arm_assignment'].set_value(tuple(random_inds), arm)
            # Remove these participants from list
            del_inds = df_inds.searchsorted(random_inds)
            df_inds = np.delete(df_inds, del_inds)            
        
        # Assign the remaining participants to last arm
        self.data['arm_assignment'].set_value(tuple(df_inds),self.n_arms)
        
        return self.data


    def randomize_grouped(self, grouped_column):
        '''
        Simple randomization to randomly assign participants to arm of study
        '''
        
        # Add arm assignment column to dataframe or re-initialize to nan
        self.data['arm_assignment'] = np.ones(len(self.data))*np.nan         
        # Grab array of indicies
        df_inds = self.data.index.values   
        grpd_inds = np.unique(self.data[grouped_column].values)
        
        # Size of population
        n_pop = len(grpd_inds)
        
        # Determine number of participants/groups per arm  
        n_per_arm = np.floor(n_pop/self.n_arms)
        
        for arm in range(1,self.n_arms+1):
            random_grps = np.random.choice(grpd_inds, size=int(n_per_arm))
            
            random_inds = []
            for rind in random_grps:
                random_inds.extend(
                    self.data[self.data[grouped_column]==rind].index.tolist()
                )
                
            self.data['arm_assignment'].set_value(tuple(random_inds), arm)
            # Remove these participants from list
            del_inds = grpd_inds.searchsorted(random_grps)
            grpd_inds = np.delete(grpd_inds, del_inds)            
        
        # Assign the remaining participants to random arm
        random_inds = []
        for rind in grpd_inds:
            ind = self.data[self.data[grouped_column]==rind].index.tolist()
            rarm = np.random.choice(range(1,self.n_arms+1), size=1)[0]
            self.data['arm_assignment'].set_value(tuple(ind),rarm)
        
        return self.data
