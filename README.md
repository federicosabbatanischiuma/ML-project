# Machine Learning for Finance Project

In order to clarify as much as possible the structure of how all different
files are organized, the following README is provided.

Before we introduce the description of how the source code is provided,
it is important to say that for time and power constraint, the models were 
trained from the EPFL cluster, we did not train locally the full models but just a 
"test" version. However, the hyperparamters you will see in this source code, are 
meant to be the optimal ones, which indeed were only used when running 
on the cluster. For more details and specifications on how the code we run on the cluster 
looks like, please refer to our github https://github.com/Kalos14/ML_SIUM . 



The source code is splitted into two directories: "data" and "models".

## Data
This directory contains the .py from which we retrieved our dataset.
This is the only dataset we used to train our models, we downloaded it
via WRDS API, cleaned it and normalized all in the file "download_preprocesses_data.py".

For more insights on the motivations that brought to such cleaning and normalization, pleaser
refer to the report.

## Models

In our study we tried to apply two different models on the same 
dataset to develop trading strategies. For clarity, we thought it was better
to present the two different frameworks in two separate directories

### Ridge and Benchmark

In this directory you can find all the files regarding our implemenation of Ridge regression
and random feature to create optimal portfolio weights.

In each file, at the beginning, there is a more detailed desciption of the purposes
of the code. Here is a very brief summary of each file:

- main_ridge.py : implmentation of ridge regression on a rolling window to predict optimal 
portfolio. The strategy was implemented to measure performance
- in function of increasing number of radom features
- benchmarks.py : again applies the same model as before, but instead of increasing complexity
via random features, focus on few "known" significant factors. This file takes imports a Series of returns,
and compares it with these benchmarks.
- functions_ridge.py : includes the functions that indeed define our model

### TransfmormerFlm

 - main_file.py: implementation of our Transformer-based model. At the beginning
one can change relative variables in order to tune "version" of the model and
subset of the dataset on which apply the same.
 - vol_manager_updated.py : this file imports a Series of return and outputs
an updated version of the same in function of a rolling volatility, similar to what we saw in class.

 - functions_file.py : includes the functions/classes that indeed define our model
 - plotsaver.py : file used to create the plots you will find in the report