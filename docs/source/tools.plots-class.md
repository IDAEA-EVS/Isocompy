# class plots ( )

The method to generate the model class plots and maps

---


## **Methods**

* best_estimator_plots ( )

* partial_dep_plots ( )

* isotopes_meteoline_plot ( )

* map_generator ( )

---


## plots.best_estimator_plots ( )

plots.best_estimator_plots( `cls, st1=True, st2=True` )

The method to plot the model class best estimators 

---


### **Parameters**

**st1** boolean default=`True`

Generate plots for stage one regression models of the model class


**st2** boolean default=`True`

Generate plots for stage one regression models of the model class

---


## plots.partial_dep_plots ( )

plots. partial_dep_plots(`cls,st1=True,st2=True`)

The method to plot the partial dependency of the features of the model class

---


### **Parameters**

**st1** boolean default=`True`

Generate plots for stage one regression models of the model class


**st2** boolean default=`True`

Generate plots for stage two regression models of the model class

---


## Method plots.isotopes_meteoline_plot ( )

plots.isotopes_meteoline_plot( `ev_class, iso_class, var_list, iso_18=None, iso_2h=None, a=8, b=10, obs_data=False, residplot=False` )

The method to plot the (meteorological) line between  two features (isotopes) that are determined in var_list

---


### **Parameters**

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

---


## Method plots.map_generator ( )

plots.map_generator(`ev_class,feat_list,observed_class_list=None,month_list=None,unit_list=None,opt_title_list=None,x="CooX",y="CooY",shp_file=None,html=`True`,direc=None,minus_to_zero_list=None,max_to_hundred_list=None`)

The method to generate the maps (.png and HTML) of the evaluation class

---


### **Parameters**

**ev_class** evaluation class

Evaluation class that contains the second stage models predictions

---
**feat_list** list

List of strings that identifies the desired features to map  

---
**observed_class_list** none type or list default=`None`

List of the preprocess classes of the observed data. No observed data will be shown in the maps if  observed_class_list=`None`, or an element of the list is none.

---
**month_list** none type or list default=`None`

List of the desired month to generate the maps. If `None`, the maps will be generated for all the months available in evaluation class

---
**unit_list** list of strings default=`None`

List of strings that identifies the units to be shown for every feature in the generated maps

---
**opt_title_list** list of strings default=`None`

List of strings that identifies the titles to be shown for every feature in the generated maps

---
**x** string default="CooX"

Identifies the name of the x (longitude) field in the evaluation class (same as defined in preprocess classess)

---
**y** string default="CooY"

Identifies the name of the y (latitude) field in the evaluation class (same as defined in preprocess classess)

---
**shp_file** none type or string default=`None`

Directory to the shape file to be used in .png maps. If `None`, no shape file will be included in the maps.If shapefile exists, it has to be in the same coordination system as the x & y.

---
**html** boolean default=`True`

If `True`, an HTML version of the maps will be created

---
**direc** none type or string default=`None`

The new directory to store the maps. If `None`, a new folder will be created in the directory that determined in the evaluation class

---
**minus_to_zero_list** none type or list default=`None`

If minus_to_zero_list is a list of booleans, when it is `True`, replace the minus values with zero for that feature. Usage in features such as relative humidity.

---
**max_to_hundred_list** none type or list default=`None`

If max_to_hundred_list is a list of booleans, when it is `True`, replace the values more that 100 with 100 for that feature. Usage in features such as relative humidity.