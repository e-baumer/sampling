#! /usr/bin/python
import os
import imp
import pandas as pd
import sys
base_path = os.path.dirname(os.path.abspath(__file__))
min_path = os.path.join(base_path,'..','sampling')
sys.path.append(min_path)
from minimization import Minimization
from simple_rand import SimpleRandomization
from stratified_rand import StratifiedRandom

pd.options.display.max_columns = 500
pd.options.display.max_rows = 1000
pd.set_option('display.width', 500)


datafile = "sampling_test_data.csv"
data_df = pd.read_csv(datafile)
data_df = data_df[data_df.index <300]


# List of categorical and continuous variables
continuous_list = ['asset_ownership_score','income_percentile', 'dependency_ratio']
categorical_list = ['occupation_encoded', 'has_business', 
                    'have_property']

#----------------
# Testing Methods
#----------------
# Simple Randomization Test
simple_test = SimpleRandomization(data_df, number_arms=3)
simple_test.randomize()

# Minimization Test
min_test = Minimization(data_df, number_arms=3)

min_test.set_integration_type(int_type='trapz')

min_test.label_encoder(
    'occupation', new_column_name='occupation_encoded'
)

min_test.minimize(continuous_list, categorical_list, C=0.20, n_pre=5, n_iter=1,
                  verb=True, min_type='mean')


# Stratification Test
column_names =  ['asset_ownership_score','income_percentile', 'dependency_ratio',
                 'occupation_encoded', 'has_business','have_property']   


strat_test= StratifiedRandom(data_df, number_arms=3)

strat_test.label_encoder(
    'occupation', new_column_name='occupation_encoded'
)

strat_test.assign_arms(column_names, percent_nan = 0.05)

#-------------------
# Evaluating Methods
#-------------------

# Producing Plots to compare covariate distributions across arm assignments
#simple_test.display_covariate_dist(column_names)  
#min_test.display_covariate_dist(column_names)  
#strat_test.display_covariate_dist(column_names)

# Calculate Imbalance Coefficient for each method
simple_test.label_encoder(
    'occupation', new_column_name='occupation_encoded'
)
c_simple = simple_test.calculate_imbalance(continuous_list, categorical_list)
c_strat = strat_test.calculate_imbalance(continuous_list, categorical_list)
c_min = min_test.calculate_imbalance(continuous_list, categorical_list)

print("Imbalance coefficient for Simple Randomization = {}".format(c_simple))
print("Imbalance coefficient for Stratified Randomization = {}".format(c_strat))
print("Imbalance coefficient for Minimization Randomization = {}".format(c_min))

    