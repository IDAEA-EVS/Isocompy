# ***Isocompy***

This repository contains .....

##############################


## ***Package information:***
\
Name: Isocompy

version: 0.1.alpha

Author: Ashkan Hassanzadeh   

Email: ashkan.hassanzadeh@gmail.com

python: 3.*

License: agpl-3.0

##############################

## ***Installation:***
\
Download the isocompy folder and add it to lib folder in python path alongside other libraries.

##############################


## ***Jupyter Notebook:***
\
There is a notebook that explains an example of implementing isocompy on spatial and isotopic data

##############################

## ***Instruction manual:***


### class preprocess:

    The class to preprocess the input variables of each model group and initiate
    the models properties such as cross-validation and  brute-force searching.

#------------------

#### **Parameters:**

(\_\_init__ method of preprocess class)

**tunedpars_rfr** dic default={"min_weight_fraction_leaf":[0,0.02,0.04],"n_estimators":[50,100,150,200,250,300],"criterion": ["mse","mae"],"min_samples_split":[2,5] }

    The dictionary that determines brute-force searching parameters of Random Forest regression. For more details on the regression method inputs, refer to sklearn library.
        
**tunedpars_svr** dic default={"kernel":[ "poly", "rbf", "sigmoid"],"C":np.logspace(-1, 1, 3),"gamma":np.logspace(-3, 1, 3) }

    The dictionary that determines brute-force searching parameters of Support Vector regression. For more details on the regression method inputs, refer to sklearn library.
        
**tunedpars_nusvr** dic default={"kernel":["linear", "poly", "rbf", "sigmoid"] }

    The dictionary that determines brute-force searching parameters of Nu Support Vector Regression. For more details on the regression method inputs, refer to sklearn library.
    
**tunedpars_mlp** dic default={"activation" : [ "logistic", "tanh"],"solver" : ["lbfgs", "sgd", "adam"],"alpha":[0.0001,0.0003],"hidden_layer_sizes":[(50,)*2,(50,)*3,(50,)*4,(100,)*2,(100,)*3,(100,)*4],"max_iter":[1000],"n_iter_no_change":[10]}

    The dictionary that determines brute-force searching parameters of Multilayer Perceptron regression. For more details on the regression method inputs, refer to sklearn library.

**tunedpars_lr** dic default={}

    The dictionary that determines brute-force searching parameters of Linear regression. For more details on the regression method inputs, refer to sklearn library.

**tunedpars_br** dic default={}

    The dictionary that determines brute-force searching parameters of Bayesian Ridge regression. For more details on the regression method inputs, refer to sklearn library.

**tunedpars_ard** dic default={}

    The dictionary that determines brute-force searching parameters of Bayesian ARD regression. For more details on the regression method inputs, refer to sklearn library.

**tunedpars_omp** dic default={}

    The dictionary that determines brute-force searching parameters of Orthogonal Matching Pursuit regression. For more details on the regression method inputs, refer to sklearn library.
    
**tunedpars_elnet** dic default={"l1_ratio":[.1, .5, .7,.9,.99]}

    The dictionary that determines brute-force searching parameters of ElasticNet (Linear regression with combined L1 and L2 priors as regularizer) regression. For more details on the regression method inputs, refer to sklearn library.

**tunedpars_muelnet** dic default={"l1_ratio":[.1, .5, .7,.9,.99]}

    The dictionary that determines brute-force searching parameters of Multi-task ElasticNet (trained with L1/L2 mixed-norm as regularizer) regression. For more details on the regression method inputs, refer to sklearn library.

**which_regs** dic default={"muelnet":True,"rfr":True,"mlp":True,"elnet":True,"omp":True,"br":True,"ard":True,"svr":True,"nusvr":False}

    The dictionary that detemines which regression models have to be included in cross-validation and  brute-force searching process

**apply_on_log** Boolean default=True

    If True, Apart from the main values, fits the models to log(1 + x) (Natural logarithm) of data. If the scores of the regression on logarithm of the data is higher
    than the real values, The chosen method will always calculate the log(1 + x) of the data before fitting the models. Note that if this is the case, to have the real
    outputs, exp(x) - 1 (the inverse of log(1 + x)) will be calculated.

**cv** int or Boolean default="auto"

    If cv="auto", the cross-validation number of folds will be calculated automatically. It is beneficial when there is few data available (max=10, min=2). If cv is an integer values,
    it determines number of folds of the cross-validation.

#------------------
#### **Methods:**

\_\_init__ (self)

fit(self, inp_var, var_name, fields,direc,
    remove_outliers=True, write_outliers_input=True, year_type="all", inc_zeros_inp_var=False, write_integrated_data=True,
    q1=0.05, q3=0.95, IQR_inp_var=True, IQR_rat_inp_var=3, mean_mode_inp_var="arithmetic", elnino=None, lanina=None)

model_pars(self,kwargs)

#------------------
#### **Attributes:**

**model_pars_name_dic** dic

    A dictionary that stores brute-force searching parameters of all models. The keys and values are dictionary names and parameters dictionaries respectively
    
* *EXAMPLE:*

    ```python
    key: "tunedpars_rfr"
    Value: tunedpars_rfr= {"min_weight_fraction_leaf":[0,0.04],"n_estimators":[150,200],"criterion": ["mse"] }

    IMPORTANT NOTE: 'model_pars' method have to be used to change the brute-force searching parameters
    ```

**db_input_args_dics** dic 

        A dictionary that stores input parameters of the fit method
    
**direc** str

    The directory of the data preprocessing class output

**month_grouped_inp_var** list

    A list of monthly grouped data. Each element of the list is a dataframe containing the grouped data of an specific month 


#------------------

