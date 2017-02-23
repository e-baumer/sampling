from __future__ import division
from collections import defaultdict
import numpy as np
from base_sample import BaseSample

#import time


class Minimization(BaseSample):
    
    def __init__(self, data_frame, number_arms=2):
        self.integrate = None
        super(Minimization, self).__init__(data_frame, number_arms)

    
    def minimize(self, covariates_con, covariates_cat, C=0.1, 
                 grouped_column=None, n_pre=1, n_iter=1, verb=False, 
                 min_type='max', nan_kick=False):
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
        nan_kick        -- If this is true, participants with nans in either 
                           continuous or categorical variables are not assigned
                           to any arm.

        '''

        imbalance_coeff = defaultdict(list)
        arm_placements = defaultdict(list)
        n_non_rand_plc = {i:0 for i in range(1, n_iter+1)}
        
        # Iterate over user specified iterations
        for i in range(n_iter):
            if verb:
                print("Calculating Imbalance Coefficients for iteration: {}".format(i+1))
            
            # Add arm assignment column to dataframe or re-initialize to nan
            self.data['arm_assignment'] = np.ones(len(self.data))*np.nan         
            
            # Grab array of indicies
            df_inds = self.data.index.values       
            
            if not (grouped_column is None):
                df_inds = np.unique(self.data[grouped_column].values)

            # Iterate over population
            j = 0
            while len(df_inds) > 0:
                #start_time = time.time()
                # Randomly assign n_pre participants to each arm
                if j <= (n_pre-1):
                    pre_inds = np.random.choice(df_inds, size=self.n_arms)
                    if not (grouped_column is None):
                        for l, rind in enumerate(pre_inds):
                            inds = self.data[
                                self.data[grouped_column]==rind
                                ].index.tolist()
                            self.data['arm_assignment'].loc[inds] = l+1                            
                    else:
                        self.data['arm_assignment'].set_value(
                            tuple(pre_inds),range(1, self.n_arms+1)
                        )
                              
                    # Remove these participants from list
                    del_inds = df_inds.searchsorted(pre_inds)
                    df_inds = np.delete(df_inds, del_inds)
                    j+=1
                    continue
                
                # Randomly select participant for assignment
                part_ind = np.random.choice(df_inds)
                
                # Do not assign participant if nan_kick is True and they have
                # nan values in covariates
                if nan_kick:
                    nan_test = self.data[
                        covariates_cat.extend(covariates_con)
                    ].loc[part_ind].isnull().values.any()
                    if nan_test:
                        # Remove participant from list
                        del_ind = np.where(df_inds == part_ind)
                        df_inds = np.delete(df_inds, del_ind)
                        j+=1
                        continue
                
                # Test if population imbalance in arms is over user specified
                # limits
                if not (C is None):
                    arm_pop = []
                    
                    # Find population for all arms
                    for arm in range(1,self.n_arms+1):
                        arm_pop.append(
                            len(self.data[self.data['arm_assignment']==arm])
                        )
                    
                    # Find percent difference of arm population from the 
                    # max arm population
                    max_arm = np.max(arm_pop)
                    arm_pop_ratio = [(max_arm-x)/max_arm for x in arm_pop]
                    max_ind = np.argmax(arm_pop_ratio)
                    
                    # If percent difference is greater than C add participant
                    # to that arm
                    if arm_pop_ratio[max_ind] >= C:
                        arm_assign = max_ind + 1
                        
                        if not (grouped_column is None):
                            ind = self.data[
                                self.data[grouped_column]==part_ind
                                ].index.tolist()
                            
                            self.data['arm_assignment'].loc[ind] =arm_assign
                    
                        else:
                            self.data['arm_assignment'].set_value(
                                part_ind, arm_assign
                            )
                        # Remove participant from list
                        del_ind = np.where(df_inds == part_ind)
                        df_inds = np.delete(df_inds, del_ind)                        
                        j+=1
                        n_non_rand_plc[i+1]+=1
                        continue
                
                # Assign participant to each arm and calculate imbalance metrics
                imbalance_arm = []
                for n in range(1, self.n_arms+1):
                    
                    if not (grouped_column is None):
                        ind = self.data[
                            self.data[grouped_column]==part_ind
                            ].index.tolist()    
                        self.data['arm_assignment'].loc[ind] = n
                    else:
                        self.data['arm_assignment'].set_value(part_ind, n)
                    
                    imbalance_arm.append(
                        self.calculate_imbalance(covariates_con, covariates_cat,
                                                 min_type)
                        )
                    
                # Find the minimum imbalance coefficient and assign to this arm
                min_ind = np.argmin(imbalance_arm)
                
                if not (grouped_column is None):
                    ind = self.data[
                        self.data[grouped_column]==part_ind
                        ].index.tolist()    
                    self.data['arm_assignment'].loc[ind] = min_ind + 1
                else:
                    self.data['arm_assignment'].set_value(part_ind, min_ind + 1)
                
                # Remove participant from list
                del_ind = np.where(df_inds == part_ind)
                df_inds = np.delete(df_inds, del_ind)
                
                j+=1
                #end_time = time.time()
                #print "{0:} seconds to complete iteration {1:}".format(end_time-start_time,j-1)
            # Capture overall imbalance
            imbalance_coeff[i] = self.calculate_imbalance(
                covariates_con, covariates_cat, min_type
            )
            # Capture participant placement
            arm_placements[i] = np.copy(self.data['arm_assignment'].values)
            
        # Reset arm assignments
        self.data['arm_assignment'] = np.nan
        # Determine overall minimum imbalance coefficient
        min_itr = min(imbalance_coeff, key=imbalance_coeff.get)
        # Set arm assignement
        self.data['arm_assignment'] = arm_placements[min_itr]
        
        if verb:
            for i in range(n_iter):
                print("Imbalance Coefficient for iteration {0:}: {1:}".format(
                    i+1, imbalance_coeff[i]
                ))
                print("{0:} Participants added to balance arm population for iteration {1:}".format(
                n_non_rand_plc[i+1],i+1))
            
            print("-------------------------------------------------------------------------")
            print("Using Assignments from iteration {0:}, with imbalance coefficient of {1:}".format(
                min_itr+1, imbalance_coeff[min_itr]
            ))
        
        return self.data