from sklearn import preprocessing




class BaseSample(object):
    
    def __init__(self, data_frame, number_arms=2):
        self.data_frame = data_frame
        self.n_arms = number_arms
        self.label_encoders = {}


    def label_encoder(self, data, encoder_name=None):
        '''
        
        '''
        
        label_encoder = preprocessing.LabelEncoder()
        encoded_data = label_encoder.fit_transform(data)
        
        # Save label encoder if name given
        if not (encoded_data is None):
            self.label_encoders[encoder_name] = label_encoder
        
        return encoded_data, label_encoder