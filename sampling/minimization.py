from __future__ import division
from collections import defaultdict
from itertools import combinations
import numpy as np
import scipy.integrate
from statsmodels.tools.tools import ECDF
from base_sample import BaseSample




class Minimization(BaseSample):
    
    def __init__(self, data_frame, number_arms=2):
        self.integrate = None
        super(Minimization, self).__init__(data_frame, number_arms)
    
    
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
                vals_2 = self.extract_values_by_arm(cont_covar, comb[0]) 
                
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
                vals_2 = self.extract_values_by_arm(cat_covar, comb[0]) 
                          
                # Calculate imbalance coefficient
                imb_coef = self.find_imbalance_categorical(
                    vals_1, vals_2, np.max(np.concatenate(vals_1,vals_2))
                )
                imb_coeff_comb.append(imb_coef)                

            # Find the mean of all the covariate imbalance coefficients
            # TODO: Implement weighted mean***
            imbalance_coeff_arm.append(np.mean(imb_coeff_comb))
    
        # Capture overall imbalance coefficient for all arm combinations
        if min_type.lower() == 'max':
            imbalance_coeff = np.max(imbalance_coeff_arm)
            
        elif min_type.lower() == 'mean':
            imbalance_coeff = np.mean(imbalance_coeff_arm)
            
        elif min_type.lower() == 'sum':
            imbalance_coeff = np.sum(imbalance_coeff_arm)
            
        else:
            print '{} is an unrecognized minimization option type (max, mean, sum)'.format(min_type)
            return False
    
        return imbalance_coeff
    
    
    def minimize(self, covariates_con, covariates_cat, C=0.1, 
                 n_pre=1, n_iter=1, verb=False, min_type='max'):
        '''
        Use the minimization of area between ECDF technique as described in 
        the Lin and Su (2012) paper. 
        
        covariates_con  -- list of column names in dataframe of continuous 
                           covariates to balance on
        covariates_cat  -- list of column names in dataframe of categorical 
                           covariates to balance on. These must be integer 
                           encoded. Use label_encoder method.
        C               -- Percent below the mean that an arm population is 
                           allowed to deviate before participants are assigned
                           to that arm without balancing. This ensure that similar
                           number of participants are in each arm. The smaller 
                           the number the closer the populations will be.
        n_pre           -- Number of participants to be randomly assigned to each
                           arm before balance takes place. Must be >= 1.
        n_itr           -- Number of iterations to try. Each iteration goes through
                           the entire allocation of participants and calculates
                           a final imbalance score. If more than one iteration
                           is specified, than the allocation for the iteration
                           with the minimum imbalance score is chosen. This is 
                           to hopefully overcome any local minimums
        verb            -- If verb is true than display imbalance coefficients
                           for all iterations and final chosen iteration.
        min_type        -- How to combine the imbalance coefficients for each 
                           arm combiniation. Choices include max, mean, sum. For
                           max the maximum imbalance coefficient is used as the 
                           overall imbalance coefficient. For sum, the sum of the 
                           imbalance coefficients is used. For mean, the mean
                           value of the imbalance coefficient is used.

        '''
        
        # Grab array of indicies
        df_inds = self.data.index.values
            
        # Size of population
        n_pop = len(self.data)
                
        imbalance_coeff = defaultdict(list)
        arm_placements = defaultdict(list)
        
        # Iterate over user specified iterations
        for i in range(n_iter):
            if verb:
                print "Calculating Imbalance Coefficients for iteration: {}".format(i+1)
            
            # Add arm assignment column to dataframe or re-initialize to nan
            self.data['arm_assignment'] = np.ones(len(self.data))*np.nan         
            
            # Iterate over population
            for j in range(n_pop):
                
                # Randomly assign n_pre participants to each arm
                if j <= (n_pre-1):
                    pre_inds = np.random.choice(df_inds, size=self.arms)
                    self.data['arm_assignment'].loc[pre_inds] = range(
                        1, self.n_arms+1
                    )
                    
                    # Remove these participants from list
                    del_inds = df_inds.searchsorted(pre_inds)
                    df_inds = np.delete(df_inds, del_inds)
                
                # Randomly select participant for assignment
                part_ind = np.random.choice(df_inds)
                
                # Test if population imbalance in arms is over user specified
                # limits
                if not (C is None):
                    arm_pop = []
                    
                    # Find population for all arms
                    for arm in range(1,self.n_arms+1):
                        arm_pop.append(
                            len(self.data[
                                data['arm_assignment']
                                ]==arm)
                        )
                    
                    # Find percent difference of arm population from the 
                    # mean population
                    mean_pop = np.mean(arm_pop)
                    arm_pop_ratio = [(mean_pop-x)/mean_pop for x in arm_pop]
                    max_ind = np.max(arm_pop_ratio)
                    
                    # If percent difference is greater than C add participant
                    # to that arm. We ignore cases when arm has more participants
                    # than the mean. We do not want to add additional particpants
                    # to this case.
                    if arm_pop_ratio[max_ind] >= C:
                        arm_assign = max_ind + 1
                        self.data[
                            'arm_assignment'
                            ].loc[part_ind] = arm_assign
                        continue
                
                # Assign participant to each arm and calculate imbalance metrics
                imbalance_arm = []
                for n in range(1, self.n_arms+1):
                    self.data['arm_assignment'].loc[part_ind] = n
                    
                    imbalance_arm.append(
                        self.calculate_imbalance(covariates_con, covariates_cat,
                                                 min_type)
                        )
                    
                # Find the minimum imbalance coefficient and assign to this arm
                min_ind = np.argmin(imbalance_arm)
                self.data['arm_assignment'].loc[part_ind] = min_ind + 1
                    
            # Capture overall imbalance
            imbalance_coeff[i] = self.calculate_imbalance(
                covariates_con, covariates_cat, min_type
            )
            # Capture participant placement
            arm_placements[i] = self.data['arm_assignment']
            
        # Reset arm assignments
        self.data['arm_assignment'] = np.nan
        # Determine overall minimum imbalance coefficient
        min_itr = min(imbalance_coeff, key=imbalance_coeff.get)
        # Set arm assignement
        self.data['arm_assignment'] = arm_placements[min_itr]
        
        if verb:
            for i in range(n_iter):
                print "Imbalance Coefficient for iteration {0:}: {1:}".format(
                    i+1, imbalance_coeff[i]
                )
            
            print "-------------------------------------------------------------------------"
            print "Using Assignments from iteration {0:}, with imbalance coefficient of {1:}".format(
                min_itr+1, imbalance_coeff[min_itr]
            )
        
        return self.data