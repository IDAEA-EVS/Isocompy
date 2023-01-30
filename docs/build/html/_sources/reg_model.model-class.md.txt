
# class model ( )

The class to fit the regression models in stage one, predict the stage one results and fit stage two regression models

---


## **Methods**

* \_\_init__(self)

* st1_fit()

* st1_predict()

* st2_fit()

* choose_estimator_by_meteo_line()

* stage2_output_report()

---


## **Attributes**

(__init__ method of model class)


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

Dataframes of predictions of stage one, separated monthly as list elements 

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

To select the best model based on meteorological line. only useful if there is a linear reference line (EX:Isotopes)

---
**meteo_coef** float 

If dependent_model_selection=True,global_line, coefficient of the line

---
**meteo_intercept** float

If dependent_model_selection=True,global_line, intercept of the line

---
**selection_method** str

If dependent_model_selection=True, selection_method: independent,local_line,global_line, point_to_point

---
**thresh_meteoline_high_scores** None type or float

A threshold to just consider models with scores higher than that value. if none, equal to mean of scores+std of scores/3

---
**model_selection_report** boolean

True or False, to determine if there is a need to model selection method report

---




## model.st1_fit ( )

model.st1_fit (`self,var_cls_list,direc,st1_model_month_list="all",args_dic= { "feature_selection" : "auto" , "vif_threshold" : 5, "vif_selection_pairs":[],"correlation_threshold":0.87,"vif_corr":True,"p_val":0.05}`)

The method to fit regression models to identified preprocess class objects in stage one 

---


### **Parameters**

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


### **Attributes**

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


## model.st2_fit ( )

model.st2_fit (`self,model_var_dict=None, output_report=True, dependent_model_selection=False, dependent_model_selection_list=None, meteo_coef=8, meteo_intercept=10, selection_method="point_to_point", thresh_meteoline_high_scores=None, model_selection_report=True, args_dic={"feature_selection":"auto", "vif_threshold":5, "vif_selection_pairs":[], 
"correlation_threshold":0.87, "vif_corr":True,"p_val":0.05}`):
  
The method to fit regression models to identified preprocess class objects in stage one

---



### **Parameters**

**model_var_dict** None type or dict  default=`None`

A dictionary that determines dependent (key - string) and independent (value) features of the second stage regression models.
Independent features (value) have to be a list of feature names (string).

If `None`, all features (independent st1 features and dependent st1 features) will be
considered as independent features of second stage models.

* *EXAMPLE:*
```python
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


### **Attributes**

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


## model.choose_estimator_by_meteo_line ( )


model.choose_estimator_by_meteo_line( `self, dependent_model_selection_list, selection_method="point_to_point", model_selection_report=True, thresh_meteoline_high_scores=None, meteo_coef=8, meteo_intercept=10` ):

The method to select the best model based on a (meteorological) line. only useful if there is a linear refrence line (EX:Isotopes).
This method could be called automatically in st2_fit if dependent_model_selection=True. or it can be called after st2_fit execution
to see the changes in best regression models based on different criterias.

*IMPORTANT NOTE:*  Executing this method will update the st2_model_results_dic to match the latest chosen selection_method. st2_model_results_dic stores the second stage results.

---



### **Parameters**

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


### **Attributes**

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


## model.stage2_output_report ( )

model.stage2_output_report(self,direc=None):

This method is useful to update st2_fit output files results in case they are changed.
(Normally the change can happen if choose_estimator_by_meteo_line method is executed)

---


### **Parameters**

**direc** str default=`None`

Directory of the output 