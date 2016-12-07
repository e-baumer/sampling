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


datafile = "/home/ebaumer/Data/kimetrica/vulnerability/wfp_kakuma_pmt.dta"
data_df = pd.read_stata(datafile)
data_df = data_df[data_df.index <300]


# List of categorical and continuous variables
continuous_list = ['crowding_index_beds','CEE_NG_1d_pc', 'dependency_ratio']
categorical_list = ['previous_livelihood_encoded', 'has_business', 
                    'have_employment']

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
    'previous_livelihood', new_column_name='previous_livelihood_encoded'
)

min_test.minimize(continuous_list, categorical_list, C=0.15, n_pre=5, n_iter=1,
                  verb=True, min_type='max')


# Stratification Test
column_names =  ['crowding_index_beds','CEE_NG_1d_pc', 'dependency_ratio',
                 'previous_livelihood_encoded', 'has_business','have_employment']

strat_test = StratifiedRandom(data_df, number_arms=3)

strat_test.label_encoder(
    'previous_livelihood', new_column_name='previous_livelihood_encoded'
)

strat_test.assign_arms(column_names, percent_nan = 0.05)

#-------------------
# Evaluating Methods
#-------------------

# Producing Plots to compare covariate distributions across arm assignments
simple_test.display_covariate_dist(column_names)  
min_test.display_covariate_dist(column_names)  
strat_test.display_covariate_dist(column_names)

# Calculate Imbalance Coefficient for each method
c_simple = simple_test.evaluate_imbalance(continuous_list, categorical_list)
c_strat = strat_test.evaluate_imbalance(continuous_list, categorical_list)
c_min = min_test.evaluate_imbalance(continuous_list, categorical_list)

print "Imbalance coefficient for Simple Randomization = {}".format(c_simple)
print "Imbalance coefficient for Stratified Randomization = {}".format(c_strat)
print "Imbalance coefficient for Minimization Randomization = {}".format(c_min)

    