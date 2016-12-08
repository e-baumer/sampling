from __future__ import division
from itertools import combinations
import numpy as np
import pandas as pd
import scipy.integrate
from statsmodels.tools.tools import ECDF
from sklearn import preprocessing
import seaborn as sns




class BaseSample(object):
    
    def __init__(self, data_frame, number_arms=2):
        self.integrate = None
        self.data = pd.DataFrame.copy(data_frame)
        self.n_arms = number_arms
        self.label_encoders = {}
        super(BaseSample, self).__init__()


    def extract_values_by_arm(self, col_name, arm):
        '''
        Extract values for participants in a specified arm
        
        col_name  -- Name of dataframe column to extract data from
        arm       -- Data will be extracted for participants in this arm number
        '''
        
        extracted_vals = self.data[
            self.data['arm_assignment']==arm
            ][col_name].values
        
        return extracted_vals
        

    def label_encoder(self, column_name, new_column_name=None, 
                      encoder_name=None):
        '''
        This method integer encodes any categorical data.
        
        column_name     -- Column name from the dataframe to encode as integers
        new_column_name -- If not None, then assign a new column with 
                           new_column_name to dataframe with encoded data
        encoder_name    -- If this is not none the encoder is kept in dictionary
                           label_encoders. It can be accessed by 
                           BaseSample.label_encoder[encoder_name]
        '''
        
        label_encoder = preprocessing.LabelEncoder()
        encoded_data = label_encoder.fit_transform(
            self.data[column_name].values
        )
        
        if not (new_column_name is None):
            self.data[new_column_name] = encoded_data
            
        # Save label encoder if name given
        if not (encoded_data is None):
            self.label_encoders[encoder_name] = label_encoder
        
        return encoded_data, label_encoder
    
    def display_covariate_dist(self, covariate_list, save_file=None):
        '''
        '''
        
        n_covars = len(covariate_list)
        
        for covariate in covariate_list:
            g = sns.FacetGrid(self.data, col="arm_assignment")
            if len(self.data[covariate].unique())>2:
                g.map(sns.distplot, covariate)
            else:
                g.map(sns.distplot, covariate, kde=False)
            if save_file:
                g.savefig(save_file, dpi=450)
                
        if save_file is None:
            sns.plt.show()
                
    def nan_finder(self, column_names, percent_nan = 0.05):
       
        '''
        Looks through all of the data points and finds all values that are NaN
        for any of the covariates to be included (listed in column_names). If
        less than the percent_nan (default 5%) have NaNs, they  will be deleted.
        '''
        
        #Initialize array to store indices of NaN values
        nan_inds = np.array([])
        
        #Cycle through all covariates to be included
        for colname in column_names:
            
            #Find the all NaN values for each column and add to the array
            nan_inds = np.concatenate(
                (nan_inds,np.where(np.isnan(self.data[colname]))[0])
            )
        
        #Extract all unique indices, this includes all of the data points 
        #that have NaN values for any of the covariates        
        all_nans = np.unique(nan_inds)
        
        #If the number of data points with NaN values is less than the specifed
        #total percentage threshold (percent_nan), delete those data points        
        if len(all_nans)/len(self.data) <= percent_nan:
            self.data = self.data.drop(all_nans)
        
        #If there are more data points that have NaN values than the acceptable
        #percentage, print an error message with the percentage.
        else:
            raise ValueError("There are too many data points with NaN values. There \
                            are {:.3f} NaN data points with at least one NaN value for \
                            one of the covariates included. The limit is set to {:.3f}."\
                            .format(len(all_nans)/len(self.data), percent_nan))
            
        return self.data

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

        norm_imbalance = abs(area1 - area2) /\
            (np.max(np.concatenate([covar_arm1, covar_arm2])) -\
             np.min(np.concatenate([covar_arm1, covar_arm2])))

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
    
    
    def calculate_imbalance(self, covariates_con, covariates_cat, 
                            min_type='mean'):
        '''   
        Calculate imbalance coefficient between all arm combinations over all
        covariates, continuous and categorical.
        
        Imbalance coefficients for individual covariates within a comparison of
        two arms are averaged to find a single imbalance coefficient.

        covariates_con  -- list of column names in dataframe of continuous 
                           covariates to balance on
        covariates_cat  -- list of column names in dataframe of categorical 
                           covariates to balance on. These must be integer 
                           encoded. Use label_encoder method.

        min_type        -- How to combine the imbalance coefficients for each 
                           arm combiniation. Choices include max, mean, sum. For
                           max the maximum imbalance coefficient is used as the 
                           overall imbalance coefficient. For sum, the sum of the 
                           imbalance coefficients is used. For mean, the mean
                           value of the imbalance coefficient is used.
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
    
    