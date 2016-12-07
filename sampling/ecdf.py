from __future__ import division
from itertools import combinations
import numpy as np
import scipy.integrate
from statsmodels.tools.tools import ECDF



class EmpCDF(object):
    
    def __init__(self):
        self.integrate = None
    
    def set_integration_type(self, int_type='trapz'):
        '''
        Set the integration method for determining area under the Empirical CDF
        '''
        try:
            self.integrate = getattr(scipy.integrate, int_type)
        except AttributeError:
            print("{} is not a valid integration method (trapz, cumtrapz, simps, romb)".format(int_type))
            return False        
        
    
    def calculate_area_continuous(self, vals, int_type='trapz'):
        '''
        Determine Empirical CDF (for continuous covariate) and then determine
        the area under the ECDF curve
        '''

        if self.integrate is None:
            self.integrate = getattr(scipy.integrate, 'trapz')
            print "No integration type specified for calculating the area"+\
                  " under ECDF. Using trapz"
            #try:
                #raise ValueError('You must set the integration method first!')
            #except ValueError:
                #print("No Integration method set for ECDF")
                #raise
        
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
            (np.min(np.concatenate([covar_arm1, covar_arm2])) -\
             np.max(np.concatenate([covar_arm1, covar_arm2])))

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
    
    
    def calculate_imbalance(self, covariates_con, covariates_cat, min_type):
        '''
        Calculate imbalance coefficient between all arm combinations over all
        covariates, continuous and categorical.
        
        Imbalance coefficients for individual covariates within a comparison of
        two arms are averaged to find a single imbalance coefficient.
        
        '''
    
        imbalance_coeff_arm = []
        arm_list = range(1,self.n_arms+1)
        
        # Loop through all possible combinations of study arms
        for comb in combinations(arm_list,2):
            
            imb_coeff_comb = []
            # Loop through all continuous covariates
            for cont_covar in covariates_con:
                
                # Get values of covariate for comb[0]
                vals_1 = self.extract_values_by_arm(cont_covar, comb[0])
                # Get values of covariate for comb[1]
                vals_2 = self.extract_values_by_arm(cont_covar, comb[1]) 
                
                # Calculate area under ECDF
                area1 = self.calculate_area_continuous(vals_1)
                area2 = self.calculate_area_continuous(vals_2)
                
                # Calculate imbalance coefficient
                imb_coef = self.find_imbalance_continuous(
                    vals_1, vals_2, area1, area2
                )
                imb_coeff_comb.append(imb_coef)
                
            # Loop through all categorical covariates
            for cat_covar in covariates_cat:
                
                # Get values of covariate for comb[0]
                vals_1 = self.extract_values_by_arm(cat_covar, comb[0])
                # Get values of covariate for comb[1]
                vals_2 = self.extract_values_by_arm(cat_covar, comb[1]) 
                          
                # Calculate imbalance coefficient
                imb_coef = self.find_imbalance_categorical(
                    vals_1, vals_2, int(np.nanmax(np.concatenate([vals_1,vals_2]))
                ))
                imb_coeff_comb.append(imb_coef)                

            # Find the mean of all the covariate imbalance coefficients
            # TODO: Implement weighted mean***
            imbalance_coeff_arm.append(np.nanmean(imb_coeff_comb))
    
        # Capture overall imbalance coefficient for all arm combinations
        if min_type.lower() == 'max':
            imbalance_coeff = np.nanmax(imbalance_coeff_arm)
            
        elif min_type.lower() == 'mean':
            imbalance_coeff = np.nanmean(imbalance_coeff_arm)
            
        elif min_type.lower() == 'sum':
            imbalance_coeff = np.nansum(imbalance_coeff_arm)
            
        else:
            print('{} is an unrecognized minimization option type (max, mean, sum)'.format(min_type))
            return False
    
        return imbalance_coeff
    