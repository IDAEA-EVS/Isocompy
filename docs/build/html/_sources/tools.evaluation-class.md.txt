# class evaluation ( )

The class to predict the second stage regression models

---



## **Methods**

* \_\_init__ (self)  

* predict ( )

---


## **Attributes**

**direc** str

directory of the class

---
**monthly_st2_output_list_all_vars** list

list of stage two models outputs, separated by month

---
**monthly_st2_output_dic_all_vars_df** dict

dictionary of stage two models outputs, separated by month. key is the month, and value is the output df of that specific month

---
**pred_inputs** Pandas Dataframe

A Dataframe, that have to be contain of features that is used in stage one, that is going to be used to estimate the stage one and two models

---
**st2_predicted_month_list** list

List of the months that have stage two regression models

---


## evaluation.predict ( )

evaluation.predict(`self, cls, pred_inputs, stage2_vars_to_predict=None, direc=None, write_to_file=True` )

The method to predict the second stage regression models

---


### **Parameters**

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

To write the outputs in .xls files, separated by the month

---


### **Attributes**

**direc** str

directory of the class

---
**monthly_st2_output_list_all_vars** list

list of stage two models outputs, separated by month

---
**monthly_st2_output_dic_all_vars_df** dict

dictionary of stage two models outputs, separated by month. key is the month, and value is the output df of that specific month

---
**pred_inputs** Pandas Dataframe

A Dataframe, that have to be contain of features that is used in stage one, that is going to be used to estimate the stage one and two models

---
**st2_predicted_month_list** list

List of the months that have stage two regression models
