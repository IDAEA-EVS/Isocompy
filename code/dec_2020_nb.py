from os.path import join
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from isocompy import data_preparation_copy,reg_model_copy,tools_copy
import os

#######################
#template help:

"""
    The method to preprocess the input data of the models

    #------------------
    Parameters:

        remove_outliers: Boolean default=True
            To remove outliers...

    #------------------
    Methods:

        __init__(self)  

    #------------------
    Attributes:

        db_input_args_dics dic 
            A dictionary that stores ...

    #------------------
"""
#######################
#importing data
'''temp=pd.read_excel(r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\inputs_sonia_paper\temp_monthly_outliers_removed_12_12_2020.xls",sheet_name="temp_monthly_outliers_removed",header=0,index_col=False,keep_default_na=True)


hum=pd.read_excel(r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\inputs_sonia_paper\hum_monthly_outliers_removed_12_12_2020.xls",sheet_name="hum_monthly_outliers_removed",header=0,index_col=False,keep_default_na=True)



#rain csv Ashkan 2020
#data_file = r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\inputs\pp_daily_all.csv"
#rain = pd.read_csv(data_file,sep=';',header=0,index_col=False,keep_default_na=True)
#date_format='%d/%m/%Y'
#rain['Date'] = pd.to_datetime(rain['Date'],format=date_format)
rain=pd.read_excel(r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\inputs_sonia_paper\rain_monthly_outliers_removed_12_12_2020.xls",sheet_name="rain_monthly_outliers_removed",header=0,index_col=False,keep_default_na=True)

#############################################################    
'''
#isotope files
data_file_iso = r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\inputs_sonia_paper\Isotopes_17_12_2020_limited.xls"
iso_18 = pd.read_excel(data_file_iso,sheet_name="ISOT18O",header=0,index_col=False,keep_default_na=True)
iso_2h=pd.read_excel(data_file_iso,sheet_name="ISOT2H",header=0,index_col=False,keep_default_na=True)
iso_3h=pd.read_excel(data_file_iso,sheet_name="ISOT3",header=0,index_col=False,keep_default_na=True)
#############################################################


#prediction and evaluation
#read points for contour
data_file = r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\inputs_sonia_paper\x_y_z_07_12_2020.xls"
pred_inputs_map=pd.read_excel(data_file,sheet_name="x_y_z_07_12_2020",header=0,index_col=False,keep_default_na=True)
#############################################################
###########################
###########test test test data preparation copy##########################
#preped_rain=data_preparation_copy.preprocess(meteo_input_type_inp_var="monthly",direc=ddir) #your working folder meteo_input_type="monthly" or daily_remove_outliers
ddir=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\test_output_4_may_2021"
'''preped_rain=data_preparation_copy.preprocess()
preped_rain.fit(inp_var=rain,var_name="rain",fields=[],meteo_input_type_inp_var="monthly",direc=ddir)
preped_temp=data_preparation_copy.preprocess()
preped_temp.fit(inp_var=temp,var_name="temp",fields=["CooX","CooY","CooZ"],meteo_input_type_inp_var="monthly",direc=ddir)'''
#iso data prep
prep_iso18=data_preparation_copy.preprocess()
prep_iso18.fit(inp_var=iso_18,var_name="iso_18",fields=["CooX","CooY","CooZ"],meteo_input_type_inp_var="monthly",direc=ddir)
prep_iso2h=data_preparation_copy.preprocess()
prep_iso2h.fit(inp_var=iso_2h,var_name="iso_2h",fields=["CooX","CooY","CooZ"],meteo_input_type_inp_var="monthly",direc=ddir)

#meteo model
#iso_meteo_model=reg_model_copy.model()
#iso_meteo_model.meteo_fit(var_cls_list=[preped_rain,preped_temp],st1_model_month_list=[1],direc=ddir)
#iso_meteo_model_dir=tools_copy.session.save(iso_meteo_model,name='iso_meteo_model') #optional. prints the directory
iso_meteo_model_dir=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\test_output_4_may_2021\iso_meteo_model_03_Jun_2021_18_46_meteoFalse_isoFalse.pkl"
iso_meteo_model=tools_copy.session.load(iso_meteo_model_dir) #optional. prints the directory

#tools_copy.plots.best_estimator_plots(iso_meteo_model,iso_plot=False)
#tools_copy.plots.partial_dep_plots(iso_meteo_model,iso_plot=False)


iso_meteo_model.iso_predic([prep_iso18,prep_iso2h])
iso_meteo_model.all_preds

iso_meteo_model.iso_fit(model_var_dict=None) #ok until here!!


tools_copy.plots.best_estimator_plots(iso_meteo_model,meteo_plot=False)
tools_copy.plots.partial_dep_plots(iso_meteo_model,meteo_plot=False) #ok until here!!

tools_copy.stats.annual_stats(iso_meteo_model)
tools_copy.stats.mensual_stats(iso_meteo_model)#ok until here!!


ev_class_allyear_map=tools_copy.evaluation()
ev_class_allyear_map.predict(iso_meteo_model,pred_inputs=pred_inputs_map,direc=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\test_output_4_may_2021\iso_models\allyear\preds_map")

tools_copy.plots.isotopes_meteoline_plot(ev_class_allyear_map,iso_meteo_model,iso_18,iso_2h,var_list=['iso_18','iso_2h'],month_data=False,obs_data=False,residplot=False)

ev_class_allyear_obs=tools_copy.evaluation()
pred_inputs=iso_meteo_model.all_preds[["CooX","CooY","CooZ","month","ID"]]
ev_class_allyear_obs.predict(iso_meteo_model,pred_inputs,direc=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\test_output_4_may_2021\iso_models\allyear\preds_obs")
tools_copy.plots.isotopes_meteoline_plot(ev_class_allyear_obs,iso_meteo_model,iso_18,iso_2h,var_list=['iso_18','iso_2h'],obs_data=False)
#debugging done
#############################################################
##############################################################
##############################################################
############################
#data preparation
preped_dataset=data_preparation.preprocess(meteo_input_type_rain="monthly",meteo_input_type_temp="monthly",meteo_input_type_hum="monthly",direc=ddir) #your working folder meteo_input_type="monthly" or daily_remove_outliers
preped_dataset.fit(rain,temp,hum,iso_18,iso_2h,iso_3h)
preped_dataset_dir=tools.session.save(preped_dataset,name='preped_dataset') #optional. prints the directory
#############################################################
#meteo model
iso_meteo_model=reg_model.model()
iso_meteo_model.meteo_fit(preped_dataset,temp_fit=True,rain_fit=True,hum_fit=True)
tools.plots.best_estimator_plots(iso_meteo_model,iso_plot=False)
tools.plots.partial_dep_plots(iso_meteo_model,iso_plot=False)
iso_meteo_model_dir=tools.session.save(iso_meteo_model,name='iso_meteo_model') #optional. prints the directory

##########################################################################################################################
iso_meteo_model_dir=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\outputs sonia paper\new_output_18\iso_meteo_model_18_Dec_2020_01_18_meteoTrue_isoFalse.pkl"
preped_dataset_dir=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\test_output_4_may_2021\preped_dataset_04_May_2021_15_14_meteoFalse_isoFalse.pkl"
#All Year
#iso_meteo_model=tools.session.load(iso_meteo_model_dir) #optional. prints the directory
preped_dataset=tools.session.load(preped_dataset_dir) #optional. prints the directory

#change directory
preped_dataset.direc=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\test_output_4_may_2021\iso_models\allyear" 
#iso model all year
iso_meteo_model.iso_predic(preped_dataset,run_iso_whole_year=True,iso_model_month_list=[],trajectories=True)
iso_meteo_model.iso_fit()
tools.plots.best_estimator_plots(iso_meteo_model,meteo_plot=False)
tools.plots.partial_dep_plots(iso_meteo_model,meteo_plot=False)
tools.stats.annual_stats(iso_meteo_model)
tools.stats.mensual_stats(iso_meteo_model)
iso_meteo_model_allyear=tools.session.save(iso_meteo_model,name='iso_meteo_model_allyear') #optional prints the directory
prepdataset_final_allyear=tools.session.save(preped_dataset,name='prepdataset_final_allyear') #optional prints the directory
#predictions map points points
ev_class_allyear_map=tools.evaluation()
ev_class_allyear_map.predict(iso_meteo_model,pred_inputs_map,direc=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\test_output_4_may_2021\iso_models\allyear\preds_map")
tools.plots.isotopes_meteoline_plot(ev_class_allyear_map,iso_meteo_model,iso_18,iso_2h,month_data=False,obs_data=False,id_point=False,residplot=False)
#predictions obs points
ev_class_allyear_obs=tools.evaluation()
pred_inputs=iso_meteo_model.all_preds[["CooX","CooY","CooZ","month",'ID_MeteoPoint']]
ev_class_allyear_obs.predict(iso_meteo_model,pred_inputs,direc=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\new_output_23_dec_old_db\iso_models\allyear\preds_obs")
tools.plots.isotopes_meteoline_plot(ev_class_allyear_obs,iso_meteo_model,iso_18,iso_2h,obs_data=False)
##########################################################################################################################
#1 2 3 12
iso_meteo_model=tools.session.load(iso_meteo_model_dir) #optional. prints the directory
preped_dataset=tools.session.load(preped_dataset_dir) #optional. prints the directory

#change directory
preped_dataset.direc=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\new_output_23_dec_old_db\iso_models\1_2_3_12" 
#iso model 1 2 3 12
iso_meteo_model.iso_predic(preped_dataset,run_iso_whole_year=False,iso_model_month_list=[1,2,3,12])
iso_meteo_model.iso_fit()
tools.plots.best_estimator_plots(iso_meteo_model,meteo_plot=False)
tools.plots.partial_dep_plots(iso_meteo_model,meteo_plot=False)
tools.stats.annual_stats(iso_meteo_model)
tools.stats.mensual_stats(iso_meteo_model)
iso_meteo_model_12312=tools.session.save(iso_meteo_model,name='iso_meteo_model_1_2_3_12') #optional prints the directory
prepdataset_final_12312=tools.session.save(preped_dataset,name='prepdataset_final_1_2_3_12') #optional prints the directory
#predictions map points points
ev_class_12312_map=tools.evaluation()
ev_class_12312_map.predict(iso_meteo_model,pred_inputs_map,direc=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\new_output_23_dec_old_db\iso_models\1_2_3_12\preds_map")
tools.plots.isotopes_meteoline_plot(ev_class_12312_map,iso_meteo_model,iso_18,iso_2h,month_data=False,obs_data=False,id_point=False,residplot=False)
#predictions obs points
ev_class_12312_obs=tools.evaluation()
pred_inputs=iso_meteo_model.all_preds[["CooX","CooY","CooZ","month",'ID_MeteoPoint']]
ev_class_12312_obs.predict(iso_meteo_model,pred_inputs,direc=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\new_output_23_dec_old_db\iso_models\1_2_3_12\preds_obs")
tools.plots.isotopes_meteoline_plot(ev_class_12312_obs,iso_meteo_model,iso_18,iso_2h,obs_data=False)
##########################################################################################################################
#6 7 8
iso_meteo_model=tools.session.load(iso_meteo_model_dir ) #optional. prints the directory
preped_dataset=tools.session.load(preped_dataset_dir ) #optional. prints the directory

#change directory
preped_dataset.direc=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\new_output_23_dec_old_db\iso_models\6_7_8" 
#iso model 6 7 8
iso_meteo_model.iso_predic(preped_dataset,run_iso_whole_year=False,iso_model_month_list=[6,7,8])
iso_meteo_model.iso_fit()
tools.plots.best_estimator_plots(iso_meteo_model,meteo_plot=False)
tools.plots.partial_dep_plots(iso_meteo_model,meteo_plot=False)
tools.stats.annual_stats(iso_meteo_model)
tools.stats.mensual_stats(iso_meteo_model)
iso_meteo_model_678=tools.session.save(iso_meteo_model,name='iso_meteo_model_6_7_8') #optional prints the directory
prepdataset_final_678=tools.session.save(preped_dataset,name='prepdataset_final_6_7_8') #optional prints the directory
#predictions map points points
ev_class_678_map=tools.evaluation()
ev_class_678_map.predict(iso_meteo_model,pred_inputs_map,direc=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\new_output_23_dec_old_db\iso_models\6_7_8\preds_map")
tools.plots.isotopes_meteoline_plot(ev_class_678_map,iso_meteo_model,iso_18,iso_2h,month_data=False,obs_data=False,id_point=False,residplot=False)
#predictions obs points
ev_class_678_obs=tools.evaluation()
pred_inputs=iso_meteo_model.all_preds[["CooX","CooY","CooZ","month",'ID_MeteoPoint']]
ev_class_678_obs.predict(iso_meteo_model,pred_inputs,direc=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\new_output_23_dec_old_db\iso_models\6_7_8\preds_obs")
tools.plots.isotopes_meteoline_plot(ev_class_678_obs,iso_meteo_model,iso_18,iso_2h,obs_data=False)

##########################################################################################################################
#indv month
iso_meteo_modellist=list()
wan_mon=[1,2,3,12,8,6,7]
for m in wan_mon:
    #load meteo
    iso_meteo_model=tools.session.load(iso_meteo_model_dir) #optional. prints the directory
    preped_dataset=tools.session.load(preped_dataset_dir) #optional. prints the directory
    if m==8: iso_meteo_model.cv=2
    preped_dataset.direc=os.path.join(r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\new_output_23_dec_old_db\iso_models\indv_month","m_"+str(m))
    iso_meteo_model.iso_predic(preped_dataset,run_iso_whole_year=False,iso_model_month_list=[m])
    iso_meteo_model.iso_fit()
    tools.stats.annual_stats(iso_meteo_model)
    tools.session.save(iso_meteo_model,name='iso_meteo_model_'+str(m)) #optional prints the directory
    tools.session.save(preped_dataset,name='prepdataset_final_'+str(m)) #optional prints the directory
    iso_meteo_modellist.append(iso_meteo_model)
for (iso_mod,m) in zip(iso_meteo_modellist,wan_mon):
    iso_meteo_model.direc=os.path.join(r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\new_output_23_dec_old_db_ind_test\iso_models\indv_month","m_"+str(m))
    iso_meteo_model.choose_estimator_by_meteo_line(selection_method="point_to_point")
    iso_meteo_model.isotope_output_report()

    tools.plots.best_estimator_plots(iso_mod,meteo_plot=False)
    tools.plots.partial_dep_plots(iso_mod,meteo_plot=False)
    #calculate predictions obs
    ev_class=tools.evaluation()
    pred_inputs=iso_mod.all_preds[["CooX","CooY","CooZ","month",'ID_MeteoPoint']]
    ev_class.predict(iso_mod,pred_inputs,write_to_file_iso_18=False,write_to_file_iso_2h=False,write_to_file_iso_3h=False,direc=os.path.join(r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\new_output_23_dec_old_db_ind_test\iso_models\indv_month","m_"+str(m),"obs_preds_"+str(m)))
    tools.plots.isotopes_meteoline_plot(ev_class,iso_mod,iso_18,iso_2h,month_data=True,obs_data=False,id_point=True,residplot=True)
    
    #calculate predictions map
    ev_class=tools.evaluation()
    ev_class.predict(iso_mod,pred_inputs_map,write_to_file_iso_18=False,write_to_file_iso_2h=False,write_to_file_iso_3h=False,direc=os.path.join(r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\new_output_23_dec_old_db_ind_test\iso_models\indv_month","m_"+str(m),"map_preds_"+str(m)))
    tools.plots.isotopes_meteoline_plot(ev_class,iso_mod,iso_18,iso_2h,month_data=False,obs_data=False,id_point=False,residplot=False)
    
    ##########################################################    
import dill
dill.load_session(r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\new_output_23_dec_old_db\iso_models\whole_dump.pkl")
#############################################################
#############################################################
#############################################################
from os.path import join
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from isocompy import data_preparation,reg_model,tools
import os


#isotope files
data_file_iso = r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\inputs\Isotopes_11_12_2020.xls"
iso_18 = pd.read_excel(data_file_iso,sheet_name="ISOT18O",header=0,index_col=False,keep_default_na=True)
iso_2h=pd.read_excel(data_file_iso,sheet_name="ISOT2H",header=0,index_col=False,keep_default_na=True)
iso_3h=pd.read_excel(data_file_iso,sheet_name="ISOT3",header=0,index_col=False,keep_default_na=True)
#############################################################

#prediction and evaluation
#read points for contour
data_file = r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\inputs\x_y_z_07_12_2020.xls"
pred_inputs_map=pd.read_excel(data_file,sheet_name="x_y_z_07_12_2020",header=0,index_col=False,keep_default_na=True)
#############################################################
iso_meteo_model=tools.session.load(r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\new_output_23_dec_old_db\iso_models\allyear\iso_meteo_model_allyear_22_Dec_2020_21_20_meteoTrue_isoTrue.pkl")
#iso_meteo_model_6_7_8_22_Dec_2020_21_53_meteoTrue_isoTrue
#iso_meteo_model_1_2_3_12_22_Dec_2020_21_41_meteoTrue_isoTrue
#iso_meteo_model_allyear_22_Dec_2020_21_20_meteoTrue_isoTrue
iso_meteo_model.direc=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\new_output_23_dec_old_db_ind_test\iso_models\allyear"
iso_meteo_model.choose_estimator_by_meteo_line(selection_method="independent") #independent point_to_point
iso_meteo_model.isotope_output_report()
tools.plots.best_estimator_plots(iso_meteo_model,meteo_plot=False)
tools.plots.partial_dep_plots(iso_meteo_model,meteo_plot=False)
tools.stats.annual_stats(iso_meteo_model)
tools.stats.mensual_stats(iso_meteo_model)
#predictions map points points
ev_class_678_map=tools.evaluation()
ev_class_678_map.predict(iso_meteo_model,pred_inputs_map,write_to_file_iso_18=False,write_to_file_iso_2h=False,write_to_file_iso_3h=False,direc=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\new_output_23_dec_old_db_ind_test\iso_models\allyear\preds_map")
tools.plots.isotopes_meteoline_plot(ev_class_678_map,iso_meteo_model,iso_18,iso_2h,month_data=False,obs_data=False,id_point=False,residplot=False)
#predictions obs points
ev_class_678_obs=tools.evaluation()
pred_inputs=iso_meteo_model.all_preds[["CooX","CooY","CooZ","month",'ID_MeteoPoint']]
ev_class_678_obs.predict(iso_meteo_model,pred_inputs,write_to_file_iso_18=False,write_to_file_iso_2h=False,write_to_file_iso_3h=False,direc=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\new_output_23_dec_old_db_ind_test\iso_models\allyear\preds_obs")
tools.plots.isotopes_meteoline_plot(ev_class_678_obs,iso_meteo_model,iso_18,iso_2h,obs_data=False)

