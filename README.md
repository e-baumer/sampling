# Sampling Methods for Control Trials
<!--[![Build Status]()]()-->


## Overview
This is a Python implementation for assigning participants to "randomized" control
trials with the goal of balancing covariates. Currently three types of assignment
are available: Simple Randomization, Stratified Randomization, and Minimization.

###1. Simple Randomization
Simple randomization, randomly assigns the population to a specific arm of the study. 
The user can control the maximum population imbalance between the arms of the study
before participants are assigned to an arm to balance population. This does not
actively balance between covariates; however, with a large enough population, random
sampling should produce balance between covariates. This assumes a normal distribution
of the covariates.

###2. Stratified Randomization
In Stratified randomization, groups are created by categorizing a covariate. From
each strata, a random placement of participants into specific arms occurs. The user 
can control the maximum population imbalance between the arms of the study before 
participants are assigned to an arm to balance population.

###2. Minimization
This particular implementation of minimization uses the difference in the Empirical
Cumulative Distribution Function (ECDF) to balance the covariates as proposed by
Lin and Su (2012) (See docs folder). The normalized area between two ECDFs is used
to quantify the imbalance level in the distributions of a particular covariate.
Imbalance coefficients are calculated for all continuous and categorical covariates
and between all possible arm combinations of the study. Participant placement is 
done to minimize the overall imbalance coefficient. The user can control the maximum 
population imbalance between the arms of the study before participants are assigned 
to an arm to balance population.



## Example Use


## TO DO
1. Lots

## Notes


## Build


## Contacts

Author - Eric Nussbaumer ([ebaumer@gmail.com](mailto:ebaumer@gmail.com))


## License

    Apache License, Version 2.0

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.