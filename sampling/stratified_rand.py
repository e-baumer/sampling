from __future__ import division
from collections import defaultdict
import numpy as np
from base_sample import BaseSample


class StratifiedRandom(BaseSample):
    
    def __init__(self, data_frame, number_arms=2):
        super(StratifiedRandom, self).__init__(data_frame, number_arms)

    
    def create_stratum(self, column_name, stratum_values, strat_range=True):
        '''
        
        '''
        
        stratum_id = defaultdict(list)
        
        # Add arm assignment column to dataframe or re-initialize to nan
        self.data['arm_assignment'] = np.ones(len(self.data))*np.nan         

            