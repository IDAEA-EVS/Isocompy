# ***Isocompy***

This repository contains an open source Python library that focuses on user defined (such as meteorological, spatial, etc.) and isotopic composition variables analysis and generating the regression – statistical estimation models. 

---
---

## ***Package information:***
\
Name: Isocompy

version: 0.1.alpha

Author: Ashkan Hassanzadeh   

Email: ashkan.hassanzadeh@gmail.com

python: 3.*

License: agpl-3.0

---
---

## ***Installation:***
\
Download the isocompy folder and add it to lib folder in python path alongside other libraries.

---
---

## ***Jupyter Notebook:***
\
There is a notebook that explains an example of implementing isocompy on spatial and isotopic data

***Instruction manual:***


# class preprocess( ):

The class to preprocess the input variables of each model group and initiate
the models properties such as cross-validation and  brute-force searching.

---
---
## **Parameters:**

(\_\_init__ method of preprocess class)

**tunedpars_rfr** dic, default= `{"min_weight_fraction_leaf":[0,0.02,0.04],"n_estimators":[50,100,150,2'00,250,300],"criterion": ["mse","mae"],"min_samples_split":[2,5] }`

The dictionary that determines brute-force searching parameters of Random Forest regression. For more details on the regression method inputs, refer to sklearn library.

---        
**tunedpars_svr** dic default=`{"kernel":[ "poly", "rbf", "sigmoid"],"C":np.logspace(-1, 1, 3),"gamma":np.logspace(-3, 1, 3) }`

The dictionary that determines brute-force searching parameters of Support Vector regression. For more details on the regression method inputs, refer to sklearn library.

---        
**tunedpars_nusvr** dic default=`{"kernel":["linear", "poly", "rbf", "sigmoid"] }`

The dictionary that determines brute-force searching parameters of Nu Support Vector Regression. For more details on the regression method inputs, refer to sklearn library.

---    
**tunedpars_mlp** dic default=`{"activation" : [ "logistic", "tanh"],"solver" : ["lbfgs", "sgd", "adam"],"alpha":[0.0001,0.0003],"hidden_layer_sizes":[(50,)*2,(50,)*3,(50,)*4,(100,)*2,(100,)*3,(100,)*4],"max_iter":[1000],"n_iter_no_change":[10]}`

The dictionary that determines brute-force searching parameters of Multilayer Perceptron regression. For more details on the regression method inputs, refer to sklearn library.

---
**tunedpars_lr** dic default=`{}`

The dictionary that determines brute-force searching parameters of Linear regression. For more details on the regression method inputs, refer to sklearn library.

---
**tunedpars_br** dic default=`{}`

The dictionary that determines brute-force searching parameters of Bayesian Ridge regression. For more details on the regression method inputs, refer to sklearn library.

---
**tunedpars_ard** dic default=`{}`

The dictionary that determines brute-force searching parameters of Bayesian ARD regression. For more details on the regression method inputs, refer to sklearn library.

---
**tunedpars_omp** dic default=`{}`

The dictionary that determines brute-force searching parameters of Orthogonal Matching Pursuit regression. For more details on the regression method inputs, refer to sklearn library.

---
**tunedpars_elnet** dic default=`{"l1_ratio":[.1, .5, .7,.9,.99]}`

The dictionary that determines brute-force searching parameters of ElasticNet (Linear regression with combined L1 and L2 priors as regularizer) regression. For more details on the regression method inputs, refer to sklearn library.

---
**tunedpars_muelnet** dic default= `{"l1_ratio":[.1, .5, .7,.9,.99]}`

The dictionary that determines brute-force searching parameters of Multi-task ElasticNet (trained with L1/L2 mixed-norm as regularizer) regression. For more details on the regression method inputs, refer to sklearn library.

---
**which_regs** dic default= `{"muelnet":True, "rfr":True, "mlp":True, "elnet":True, "omp":True, "br":True, "ard":True, "svr":True, "nusvr":False}`

The dictionary that determines which regression models have to be included in cross-validation and  brute-force searching process

---
**apply_on_log** Boolean default=`True`

If `True`, Apart from the main values, fits the models to log(1 + x) (Natural logarithm) of data. If the scores of the regression on logarithm of the data is higher
than the real values, The chosen method will always calculate the log(1 + x) of the data before fitting the models. Note that if this is the case, to have the real
outputs, exp(x) - 1 (the inverse of log(1 + x)) will be calculated.

---
**cv** int or Boolean default=`"auto"`

If cv="auto", the cross-validation number of folds will be calculated automatically. It is beneficial when there is few data available (max=10, min=2). If cv is an integer values,
it determines number of folds of the cross-validation.

---
---

## **Methods:**

* \_\_init__ (self)

* fit()

* model_pars()

---
---
## **Attributes:**

**model_pars_name_dic** dic

A dictionary that stores brute-force searching parameters of all models. The keys and values are dictionary names and parameters dictionaries respectively
    
*EXAMPLE:*

```python

key: "tunedpars_rfr"

Value: tunedpars_rfr= {"min_weight_fraction_leaf":[0,0.04],"n_estimators":[150,200],"criterion": ["mse"] }

```

*IMPORTANT NOTE*: `'model_pars'` method have to be used to change the brute-force searching parameters
    

---
**db_input_args_dics** dic 

A dictionary that stores input parameters of the fit method

---
**direc** str

The directory of the data preprocessing class output

---
**month_grouped_inp_var** list

A list of monthly grouped data. Each element of the list is a dataframe containing the grouped data of an specific month 

---
---
---
# preprocess.fit( )

preprocess.fit(self, inp_var, var_name, fields, direc, remove_outliers=True,write_outliers_input=True, year_type="all", inc_zeros_inp_var=False, write_integrated_data=True, q1=0.05, q3=0.95, IQR_inp_var=True, IQR_rat_inp_var=3, mean_mode_inp_var="arithmetic", elnino=None, lanina=None)


The method to preprocess the input data of the models

---
---
## Parameters:

**inp_var** Pandas dataframe

Input pandas dataframe containing the dependent value, unique IDs
for each sample, and date. (The columns "Value", "ID" and "Date" have to be found in the dataframe)

---
**var_name** str

Name of the dependent variable.

---
**fields** list of strings

List of independent variable names that have to be existed in the inp_var dataframe

---
**direc** str

The directory of the data preprocessing class output

---
**remove_outliers** Boolean default=`True`

To remove outliers based on the introduced variables. If False, the variables related to removing outliers will be ignored.

---
**write_outliers_input** Boolean default=`True`

Effective if remove_outliers=True. To generate .xls file in directory folders of the class after removing outliers

---
**year_type** str default=`"all"`

`"all"`, `"elnino"` or `"lanina"`. In case elnino or lanina years have to be selected from the database

---
**inc_zeros_inp_var** Boolean default=`False`

Effective if remove_outliers=`True`. Removing outliers could be done not considering the zero values in the database.
(The zero values will be seperated, outliers will be removed, then zero values will be added to database)

---
**write_integrated_data** Boolean default=`True`

Generate two .xls outputs consisting of integrated data, and the quantity of data in each month

---
**q1** float default=`0.05`

Effective if remove_outliers=True and IQR_inp_var=False. Lower percentile limit to determine the outliers.

---
**q3** float default=`0.95`

Effective if remove_outliers=True  and IQR_inp_var=False. Upper percentile limit to determine the outliers

---
**IQR_inp_var** Boolean default=`True`

Effective if remove_outliers=True. Determining the upper limit of the outliers using this 
formula: $X*q_{0.75} + IQR_{rat}*abs(X*  q{0.25} - X * q_{0.75})$

Lower limit = q1

---
**IQR_rat_inp_var** float default=`3`

Effective if remove_outliers=True and IQR_inp_var=True. This parameter used in  X*q_0.75 + IQR_rat*abs(X*q_0.25 - X*q_0.75) 
to determine upper boundary limit.

---
**mean_mode_inp_var** str default=`"arithmetic"`

Data averaging method. available options are `"arithmetic"`  or `"geometric"`

---
**elnino** None type or list of integers default=`None`

List of elnino years

---
**lanina** None type or list of integers default=`None`

List of lanina years

---
---
## Attributes:

**db_input_args_dics** dic 

A dictionary that stores input parameters of the fit method

---
**direc** str

The directory of the data preprocessing class output

---
**month_grouped_inp_var** list

A list of monthly grouped data. Each element of the list is a dataframe containing the grouped data of an specific month 

---
---
---
# preprocess.model_pars( )

preprocess.model_pars(self,**kwargs):

To change the brute-force searching parameters that is already determined in a class. 

---
*EXAMPLE:*
```python
#Define the new brute-force searching parameters dictionaries:

#brute-force searching parameters for random forest
brutesearch_ran_for_dic={"min_weight_fraction_leaf":[0,0.04],"n_estimators":[150,200],"criterion": ["mse"] }

#brute-force searching parameters for elastic net
brutesearch_elasticnet_dic={"l1_ratio":[ .5, .9]}

#change the brute-force searching parameters
prep_class.model_pars( "tunedpars_rfr" = brutesearch_ran_for_dic, "tunedpars_elnet" = brutesearch_elasticnet_dic )

```
---
---
---
# class model ( ):

The class to fit the regression models in stage one, predict the stage one results and fit stage two regresison models

---
---
## **Methods:**

* \_\_init__(self)

* st1_fit()

* st1_predict()

* st2_fit()

* choose_estimator_by_meteo_line()

* stage2_output_report()

---
---
## **Attributes:**

(\_\_init__ method of model class)


**direc** str

Directory of the class

---
**st1_model_results_dic** dict

A dictionary consist of st1 model results

---
**st1_varname_list** list

List of the names of independent variables in st1

---
**st1_model_month_list** list

List of desired months to model in st1

---
**used_feature_list** list

List of all used features (strings) in st1

---
**cls_list** list

A list of preprocess class objects that we wish to model in st2

---
**all_pred** Pandas Dataframe

A dataframe of all st1 predictions

---
**predictions_monthly_list** list

Dataframes of predictions of stage one, seperated monthly as list elements 

---
**st2_model_month_list** list

List of desired months to model in st1. Indicated months have to exist in st1_model_month_list

---
**dic_second_stage_names** dict

Helps in generating model_var_dict in st2_fit

---
**st2_model_results_dic** dict

A dictionary consist of st2 model results

---
**dependent_model_selection** boolean

To select the best model based on meteorological line. only useful if there is a linear refrence line (EX:Isotopes)

---
**meteo_coef** float 

If dependent_model_selection=`True`,global_line, coefficient of the line

---
**meteo_intercept** float

If dependent_model_selection=`True`,global_line, intercept of the line

---
**selection_method** str

If dependent_model_selection=`True`, selection_method: `independent`,`local_line`,`global_line`, `point_to_point`

---
**thresh_meteoline_high_scores** None type or float

A threshold to just consider models with scores higher than that value. if none, equal to mean of scores+std of scores/3

---
**model_selection_report** boolean

`True` or `False`, to determine if there is a need to model selection method report

---
---
---
# model.st1_fit ( )

model.st1_fit (`self,var_cls_list,direc,st1_model_month_list="all",args_dic= { "feature_selection" : "auto" , "vif_threshold" : 5, "vif_selection_pairs":[],"correlation_threshold":0.87,"vif_corr":True,"p_val":0.05}`)

The method to fit regression models to identified preprocess class objects in stage one 

---
---
## **Parameters:**

**var_cls_list** list

A list of preprocess class objects to to fit regression models. Regression models will be fitted to each elemnt of the list (a preprocess class object).

---
**direc** str

Directory of the class

---
st1_model_month_list str or list of integers default=`"all"`

List of desired months to model in st1

---
**args_dic** dict default={`"feature_selection":"auto","vif_threshold":5, "vif_selection_pairs":[],"correlation_threshold":0.87,"vif_corr":True,"p_val":0.05}`

A dictionary of parameters that identifies the behaviour of feature selection prior to regressions:    
    
* args_dic[`"feature_selection"`] ="manual": Statistical information will be shown to the user, and the desired features will be
chosen by the user

* args_dic[`"feature_selection"`] ="auto": Feature selection will be done automatically

* args_dic[`"vif_threshold"`] =None: VIF (Variation Inflation Factor) will not be considered as a factor in feature selection

* args_dic[`"vif_threshold"`] = float type: A threshold to identify high VIF values

* args_dic[`"vif_corr"`] = True: If True, use correlation coefficient values to identify multicolinearity in features with high vif value

* args_dic[`"correlation_threshold"`] = 0.87 A threshold to identify high correlation coefficient values

* args_dic[`"vif_selection_pairs"`] = empty list or list of list(s): If empty: feature elimination based on vif will be automatic

if  args_dic[`"vif_selection_pairs"`] =[ [`"a","b"`] ], in case both `"a"` and `"b"` have high vif values and high correlations, the b values will be eliminated

---
---
## **Attributes:**

**direc** str

Directory of the class

---
**st1_model_results_dic** dict

A dictionary consist of st1 model results

---
**st1_varname_list** list

List of the names of independent variables in st1

---
**st1_model_month_list** list

List of desired months to model in st1

---
---
---
# model.st1_predict()
    
model.st1_predict(`self, cls_list, st2_model_month_list=None, trajectories=False, daily_rain_data_for_trajs=None`)

The method to estimate the independent features that modeled in st1 using the new observations that are in a new list (cls_list) which each element is a preprocess class objects 

---
---
## **Parameters:**

**cls_list** list

A list of preprocess class objects that we wish to model in st2

---
**st2_model_month_list** list or None type default=`None`

List of desired months to model in st1. Indicated months have to exist in st1_model_month_list. If None, it will be equal to st1_model_month_list

---
---
## **Attributes:**

**used_feature_list** list

List of all used features (strings) in st1

---
**cls_list** list

A list of preprocess class objects that we wish to model in st2

---
**all_pred Pandas** Dataframe

A dataframe of all st1 predictions in observed samples

---
**predictions_monthly_list** list

Dataframes of predictions of stage one, seperated monthly as list elements 

---
s**t2_model_month_list** list

List of desired months to model in st1. Indicated months have to exist in st1_model_month_list

---
---
---
# model.st2_fit ( )

model.st2_fit (`self,model_var_dict=None, output_report=True, dependent_model_selection=False, dependent_model_selection_list=None, meteo_coef=8, meteo_intercept=10, selection_method="point_to_point", thresh_meteoline_high_scores=None, model_selection_report=True, args_dic={"feature_selection":"auto", "vif_threshold":5, "vif_selection_pairs":[], 
"correlation_threshold":0.87, "vif_corr":True,"p_val":0.05}`):
  
The method to fit regression models to identified preprocess class objects in stage one

---
---

## **Parameters:**

**model_var_dict** None type or dict  default=`None`

A dictionary that determines dependent (key - string) and independent (value) features of the second stage regression models.
Independent features (value) have to be a list of feature names (string).

If `None`, all features (independent st1 features and dependent st1 features) will be
considered as independent features of second stage models.

* *EXAMPLE:*
```pyhton
model_var_dict = {"is1":["CooZ","hmd"],"is2":["prc","hmd"],}
```
---
**output_report** boolean default=`True`

To generate output reports

---
*Parameters used in choose_estimator_by_meteo_line* 

**dependent_model_selection** boolean default=`False`
To select the best model based on a (meteorological) line. only useful if there is a linear refrence line (EX:Isotopes)

---
**dependent_model_selection_list** default=`None`

Used if dependent_model_selection=`True`. List of two features that have to be used in `dependent_model_selection`

---
**meteo_coef** default=`8`

Used if dependent_model_selection=`True` and selection_method=`"global_line"`. Coefficient of the line

---
**meteo_intercept** default=`10`

Used if dependent_model_selection=`True` and selection_method=`"global_line"`. Intercept of the line

---
**selection_method** default="point_to_point"

Used if `dependent_model_selection`=`True`. `selection_method` could be:

* `independent`
* `local_line`: coef and intercept derived from a linear regression of observed data
* `global_line`
* `point_to_point`: find the models pair with shortest average distance between observed and predicted data  

---
**thresh_meteoline_high_scores** None type or float default=`None`

A threshold to just consider models with scores higher than that value. if `None`, equal to mean of scores+std of scores/3

---
**model_selection_report** boolean default =`True`

To determine if there is a need to model selection method report

---
**args_dic** dict default=`{"feature_selection":"auto","vif_threshold":5, "vif_selection_pairs":[], "correlation_threshold": 0.87, "vif_corr": True, "p_val":0.05}`

A dictionary of parameters that identifies the behaviour of feature selection prior to regressions:    
    
* args_dic[`"feature_selection"`] =`"manual"`: Statistical information will be shown to the user, and the desired features will be
chosen by the user

* args_dic[`"feature_selection"`] =`"auto"`: Feature selection will be done automatically

* args_dic[`"vif_threshold"`] =None: VIF (Variation Inflation Factor) will not be considered as a factor in feature selection

* args_dic[`"vif_threshold"`] = float type: A threshold to identify high VIF values

* args_dic[`"vif_corr"`] = `True`: If True, use correlation coefficient values to identify multicolinearity in features with high vif value

* args_dic[`"correlation_threshold"`] = `0.87` A threshold to identify high correlation coefficient values

* args_dic[`"vif_selection_pairs"`] = empty list or list of list(s): If empty: feature elimination based on vif will be automatic

if  args_dic[`"vif_selection_pairs"`] =[ [`"a","b"`] ], in case both `"a"` and `"b"` have high vif values and high correlations, the b values will be eliminated

---
---
## **Attributes:**

**st2_model_results_dic** dict

A dictionary consist of st2 model results

---
*Attributes used in choose_estimator_by_meteo_line* 

**dependent_model_selection** boolean

To select the best model based on meteorological line. only useful if there is a linear refrence line (EX:Isotopes)

---
**meteo_coef** float 

If dependent_model_selection=`True`, global_line, coefficient of the line

---
**meteo_intercept** float

If dependent_model_selection=`True`, global_line, intercept of the line

---
**selection_method** str

If dependent_model_selection=`True`, selection_method: `"independent", "local_line", "global_line", "point_to_point"`

---
**thresh_meteoline_high_scores None type or float

A threshold to just consider models with scores higher than that value. if `None`, equal to mean of scores+std of scores/3

---
**model_selection_report** boolean

To determine if there is a need to model selection method report

---
---
---
# model.choose_estimator_by_meteo_line ( )


model.choose_estimator_by_meteo_line( `self, dependent_model_selection_list, selection_method="point_to_point", model_selection_report=True, thresh_meteoline_high_scores=None, meteo_coef=8, meteo_intercept=10` ):

The method to select the best model based on a (meteorological) line. only useful if there is a linear refrence line (EX:Isotopes).
This method could be called automatically in st2_fit if dependent_model_selection=True. or it can be called after st2_fit execution
to see the changes in best regression models based on different criterias.

*IMPORTANT NOTE:*  Executing this method will update the st2_model_results_dic to match the latest chosen selection_method. st2_model_results_dic stores the second stage results.

---
---

## Parameters:

**dependent_model_selection_list** default=`None`

Used if dependent_model_selection=True. List of two features that have to be used in dependent_model_selection

---
**meteo_coef** default=8

Used if dependent_model_selection=`True` and selection_method=`"global_line"`. Coefficient of the line

---
**meteo_intercept** default=10

Used if dependent_model_selection=`True` and selection_method=`"global_line"`. Intercept of the line

---
**selection_method** default=`"point_to_point"`

Used if dependent_model_selection=`True`. Selection_method could be:

* `"independent"`
* `"local_line"`: coef and intercept derived from a linear regression of observed data
* `"global_line"`
* `"point_to_point"`: find the models pair with shortest average distance between observed and predicted data  

---
**thresh_meteoline_high_scores** None type or float default=None

A threshold to just consider models with scores higher than that value. if none, equal to mean of scores+std of scores/3

---
**model_selection_report** boolean default =`True`

`True` or `False`, to determine if there is a need to model selection method report

---
---
## Attributes:

**st2_model_results_dic** dict

Updated dictionary of st2 model results

---
**dependent_model_selection** boolean

To select the best model based on meteorological line. only useful if there is a linear refrence line (EX:Isotopes)

---
**meteo_coef** float 

If dependent_model_selection=True,global_line, coefficient of the line

---
**meteo_intercept** float

If dependent_model_selection=True,global_line, intercept of the line

---
**selection_method** str

If dependent_model_selection=`True`, selection_method: `"independent","local_line","global_line", "point_to_point"`

---
**thresh_meteoline_high_scores** None type or float

A threshold to just consider models with scores higher than that value. if none, equal to mean of scores+std of scores/3

---
**model_selection_report** boolean

True or False, to determine if there is a need to model selection method report

---
---
---
# model.stage2_output_report ( )

model.stage2_output_report(self,direc=None):

This method is useful to update st2_fit output files results in case they are changed.
(Normally the change can happen if choose_estimator_by_meteo_line method is executed)

---
---
## Parameters:

**direc** str default=`None`

Directory of the output 

---
---
---
# class session ( ):

The class to save and load the objects and sessions

---
---
## **Methods:**

* save()

* load()

* save_session()

* load_session()

---
---
---
# session.save ( )

session.save(self,name="isocompy_saved_object")

The method to save an object

---
---
## **Parameters:**

**name** str default=`"isocompy_saved_object"`

The output name string

---
---
## **Returns:**

**filename** string

Directory of the saved object

---
---
---
# session.load ( )

session.load(direc)

The method to load a pkl object. `direc` is the directory of the object to be loaded.

---
---
## **Returns:**

**obj** object

The loaded object

---
---
---
# session.save_session( )

save_session(direc,name="isocompy_saved_session", *argv):

The method to save a session

---
---
## **Parameters:**

name: str default="isocompy_saved_object"

The output name string

---
**\*argv**

The objects that wanted to be stored in the session

---
---
## **Returns:**

**filename** string 

Directory of the saved session

---
---
---
# session.load_session ( )

session.load_session(dir)

The method to load a session

---
---
## **Parameters:**

**\*argv**

The objects that wanted to be stored in the session

---
---
## **Returns:**

Loads the session

---
---
---
# class evaluation ( )

The class to predict the second stage regression models

---
---

## **Methods:**

* \_\_init__ (self)  

* predict ( )

---
---
## **Attributes:**

**direc** str

directory of the class

---
**monthly_st2_output_list_all_vars** list

list of stage two models outputs, seperated by month

---
**monthly_st2_output_dic_all_vars_df** dict

dictionary of stage two models outputs, seperated by month. key is the month, and value is the output df of that specific month

---
**pred_inputs** Pandas Dataframe

A Dataframe, that have to be contain of features that is used in stage one, that is going to be used to estimate the stage one and two models

---
**st2_predicted_month_list** list

List of the months that have stage two regression models

---
---
---
# evaluation.predict( )

evaluation.predict(`self, cls, pred_inputs, stage2_vars_to_predict=None, direc=None, write_to_file=True` )

The method to predict the second stage regression models

---
---
## **Parameters:**

**cls ** model class

The model class that contains st1 and st2 models

---
**pred_inputs ** Pandas dataframe

A Dataframe, that have to be contain of features that is used in stage one, that is going to be used to estimate the stage one and two models.

It can contain `"month"` field which could be used in evaluating the stage two predictions in observed data.

* *EXAMPLE:*
    ```python

    pred_inputs=model_class.all_preds[["CooX","CooY","CooZ","month","ID"]].reset_index()

    ```

---
**stage2_vars_to_predict ** None type or list of strs default=`None`

List of stage two dependent features to predict the outputs. If `None`, The results will be predicted for all two dependent features

---
**direc ** None type or str default=None

Directory of the class. If `None`, it is the same directory as the model class.

---
**write_to_file ** boolean default=`True`

To write the outputs in .xls files, seperated by the month

---
---
## **Attributes:**

**direc** str

directory of the class

---
**monthly_st2_output_list_all_vars** list

list of stage two models outputs, seperated by month

---
**monthly_st2_output_dic_all_vars_df** dict

dictionary of stage two models outputs, seperated by month. key is the month, and value is the output df of that specific month

---
**pred_inputs** Pandas Dataframe

A Dataframe, that have to be contain of features that is used in stage one, that is going to be used to estimate the stage one and two models

---
**st2_predicted_month_list** list

List of the months that have stage two regression models

---
---
---
# class stats( )   

The class to calculate and generate statistical reports for the second stage models

---
---

## **Methods:**

* annual_stats( )

* mensual_stats( )

---
---
---

# stats.seasonal_stats( )

stats.seasonal_stats(model_cls_obj)

The method to generate statistical reports for the second stage models based on  all specified month in second stage data

---
---
## Parameters:

**model_cls_obj**

Input model class object

---
---
---
# stats.mensual_stats()

stats.mensual_stats(model_cls_obj)

The method to generate statistical reports for the second stage models based on  each specified month in second stage data

---
---
## Parameters:

**model_cls_obj**

Input model class object

---
---
---
# class plots ( )

The method to generate the model class plots

---
---
## Methods:

* best_estimator_plots ( )

* partial_dep_plots ( )

* isotopes_meteoline_plot ( )

---
---
---
# plots.best_estimator_plots()

plots.best_estimator_plots( `cls, st1=True, st2=True` )

The method to plot the model class best estimators 

---
---
## **Parameters:**

**st1** boolean default=`True`

Generate plots for stage one regression models of the model class


**st2** boolean default=`True`

Generate plots for stage one regression models of the model class

---
---
---
# plots.partial_dep_plots( )

plots. partial_dep_plots(`cls,st1=True,st2=True`)

The method to plot the partial dependency of the features of the model class

---
---
## **Parameters:**

**st1** boolean default=`True`

Generate plots for stage one regression models of the model class


**st2** boolean default=`True`

Generate plots for stage two regression models of the model class

---
---
---
# plots.isotopes_meteoline_plot ( )

plots.isotopes_meteoline_plot( `ev_class, iso_class, var_list, iso_18=None, iso_2h=None, a=8, b=10, obs_data=False, residplot=False` )

The method to plot the (meteorological) line between  two features (isotopes) that are determined in var_list

---
---
## **Parameters:**

**ev_class** evaluation class

evaluation class that contains the second stage models predictions

---
**iso_class** model class

model class that contains the second stage models

---
**iso_18** none type or Pandas Dataframe default=`None`

First feature (isotope) observed raw data. Ignored if obs_data=`False`

---
**iso_2h** none type or Pandas Dataframe default=`None`

Second feature (isotope) observed raw data. Ignored if obs_data=`False`

---
**var_list** list of strings

List of strings that identifies the names of two features in the evaluation and model class (in stage two)

---
**a** float default=`8`

Coefficient of the line

---
**b** float default=`10`

Intercept of the line

---
**obs_data** boolean default=`False`

`False` if iso_18 and iso_2h are not observed data.
`True` if the predictions in evaluation class have an specified date, in `"month"` field.

* *EXAMPLE:*

    ```python
    pred_inputs=model_class.all_preds[["CooX","CooY","CooZ","month","ID"]].reset_index()
    ev_class_obs=tools_copy.evaluation()
    ev_class_obs.predict(model_class,pred_inputs,direc=direc)
    tools_copy.plots.isotopes_meteoline_plot(ev_class_obs,model_class,var_list=['is1','is2'],obs_data=True)
    ```
---
**residplot** boolean default=`False`

Ignored if obs_data=`False`. It create residual plots in each month for each ID.

#------------------