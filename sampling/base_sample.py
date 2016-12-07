from ecdf import EmpCDF
from sklearn import preprocessing
import seaborn as sns
import numpy as np



class BaseSample(EmpCDF):
    
    def __init__(self, data_frame, number_arms=2):
        self.data = data_frame
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
        
    
    def evaluate_imbalance(self, covariates_con, covariates_cat, 
                           min_type='mean'):
        '''
        Evaluation of randomization type through imbalance coefficients which
        are defined as the area between the ECDF of the covariates.
        
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
        
        
        imbalance_coeff = self.calculate_imbalance(
            covariates_con, covariates_cat, min_type
        )
        
        return imbalance_coeff