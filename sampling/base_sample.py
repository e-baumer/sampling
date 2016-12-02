from sklearn import preprocessing




class BaseSample(object):
    
    def __init__(self, data_frame, number_arms=2):
        self.data = data_frame
        self.n_arms = number_arms
        self.label_encoders = {}


    def extract_values_by_arm(self, col_name, arm):
        '''
        Extract values for participants in a specified arm
        
        col_name  -- Name of dataframe column to extract data from
        arm       -- Data will be extracted for participants in this arm number
        '''
        
        extracted_vals = self.data[
            self.data['arm_assignment']==arm
            ][col_name].value
        
        return extracted_vals
        

    def label_encoder(self, column_name, new_colmn_name=None, 
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
        
        if not (new_colmn_name is None):
            self.data[new_colmn_name] = encoded_data
            
        # Save label encoder if name given
        if not (encoded_data is None):
            self.label_encoders[encoder_name] = label_encoder
        
        return encoded_data, label_encoder