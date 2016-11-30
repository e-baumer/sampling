from __future__ import division
from collections import defaultdict
import numpy as np
import scipy.integrate
from statsmodels.tools.tools import ECDF
from .base_sample import BaseSample




class Minimization(BaseSample):
    
    def __init__(self):
        self.integrate = None
        super(Minimization, self).__init__()
    
    
    def set_integration_type(self, int_type='trapz'):
        '''
        Set the integration method for determining area under the Empirical CDF
        '''
        try:
            self.integrate = getattr(scipy.integrate, int_type)
        except AttributeError:
            print "{} is not a valid integration method (trapz, cumtrapz, simps, romb)".format(int_type)
            return False        
        
    
    def calculate_area_continuous(self, vals, int_type='trapz'):
        '''
        Determine Empirical CDF (for continuous covariate) and then determine
        the area under the ECDF curve
        '''

        if self.integrate is None:
            print "You must set the integration method first!"
            return False
        
        ecdf = ECDF(vals, side='left')
    
        area = self.integrate(ecdf.y[1:], ecdf.x[1:])
    
        return area
    
    
    def find_imbalance_continuous(self, covar_arm1, covar_arm2, area1, area2):
        '''
        Find the normalized imbalance metric for a single continuous varible 
        (covariate) as defined by Lin and Su (2012). This is the normalized area 
        difference between the two ECDF defined for two seperate arms of a 
        study. See docs for original paper.
        
        covar_arm1  -- Values of continuous covariate for participants enrolled 
                       in one particular arm
        covar_arm2  -- Values of continuous covariate for participants enrolled 
                       in particular arm to be compared with covar_arm1
        Both covar_arm1 and covar_arm2 correspond to the same covariate
            
        area1  -- Area under the ECDF as given by covar_arm1
        area2  -- Area under the ECDF as given by covar_arm2
        '''

        norm_imbalance = (area1 - area2) /\
            (np.min(np.concatenate(covar_arm1, covar_arm2)) -\
             np.max(np.concatenate(covar_arm1, covar_arm2)))

        return norm_imbalance
        
        
    def find_imbalance_categorical(self, covar_arm1, covar_arm2, n_categories):
        '''
        Find the normalized imbalance metric for a single categorical varible 
        (covariate) as defined by Lin and Su (2012). This is the normalized area 
        difference between the two ECDF defined for two seperate arms of a 
        study. See docs for original paper.
        
        covar_arm1  -- Integer encoded values of categorical covariate for 
                       participants enrolled in one particular arm
        covar_arm2  -- Integer encoded values of categorical covariate for 
                       participants enrolled in particular arm to be compared 
                       with covar_arm1
        n_categories -- Number of categories for categorical covariate
        
        Both covar_arm1 and covar_arm2 correspond to the same covariate
        '''
        
        norm_imbalance = 0.
        n_arm1 = len(covar_arm1)
        n_arm2 = len(covar_arm2)
        
        for i in range(n_categories):
            n_covar_arm1 = len(np.where(covar_arm1==i)[0])
            n_covar_arm2 = len(np.where(covar_arm2==i)[0])
            
            norm_imbalance += abs(n_covar_arm1/n_arm1 - n_covar_arm2/n_arm2) / 2 
            
        return norm_imbalance
    
    
    def minimize(self, covariates_con, covariates_cat, C=0.1, 
                 n_pre=1, n_iter=1, verb=False):
        '''
        
        '''
        
        # Grab array of indicies
        df_inds = self.data_frame.index.values
            
        # Add arm assignment column to dataframe
        self.data_frame['arm_assignment'] = np.ones(len(self.data_frame))*np.nan
                
        # Size of population
        n_pop = len(self.data_frame)
                
        imbalance_coeff = defaultdict(list)
        
        # Iterate over user specified iterations
        for i in range(n_iter):
            
            # Iterate over population
            for j in range(n_pop):
                
                # Randomly assign n_pre participants to each arm
                if j <= (n_pre-1):
                    pre_inds = np.random.choice(df_inds, size=self.arms)
                    self.data_frame['arm_assignment'][pre_inds] = range(
                        1, self.n_arms+1
                    )
                    
                    # Remove these participants from list
                    del_inds = df_inds.searchsorted(pre_inds)
                    df_inds = np.delete(df_inds, del_inds)
                
                # Randomly select participant for assignment
                part_ind = np.random.choice(df_inds)
                
                # Assign participant to each arm and calculate imbalance metrics
                
            
        