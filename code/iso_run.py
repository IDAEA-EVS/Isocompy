from os.path import join
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from isocompy import data_preparation,reg_model,tools
import os


#######################
#importing data
#temp=pd.read_excel(r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\inputs\T_Daily.xlsx",sheet_name="T_Daily",header=0,index_col=False,keep_default_na=True)
temp=pd.read_excel(r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\inputs\temp_monthly_outliers_removed_07_12_2020.xls",sheet_name="temp_monthly_outliers_removed_7",header=0,index_col=False,keep_default_na=True)


#hum=pd.read_excel(r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\inputs\RH_Daily_0_100.xlsx",sheet_name="TODAS",header=0,index_col=False,keep_default_na=True)
hum=pd.read_excel(r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\inputs\hum_monthly_outliers_removed_07_12_2020.xls",sheet_name="hum_monthly_outliers_removed_7",header=0,index_col=False,keep_default_na=True)



#rain csv Ashkan 2020
#data_file = r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\inputs\pp_daily_all.csv"
#rain = pd.read_csv(data_file,sep=';',header=0,index_col=False,keep_default_na=True)
#date_format='%d/%m/%Y'
#rain['Date'] = pd.to_datetime(rain['Date'],format=date_format)
rain=pd.read_excel(r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\inputs\rain_monthly_outliers_removed_08_12_2020.xls",sheet_name="rain_monthly_outliers_removed_2",header=0,index_col=False,keep_default_na=True)

#############################################################    

#isotope files
data_file_iso = r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\inputs\Isotopes_16_11_2020.xls"
iso_18 = pd.read_excel(data_file_iso,sheet_name="ISOT18O",header=0,index_col=False,keep_default_na=True)
iso_2h=pd.read_excel(data_file_iso,sheet_name="ISOT2H",header=0,index_col=False,keep_default_na=True)
iso_3h=pd.read_excel(data_file_iso,sheet_name="ISOT3",header=0,index_col=False,keep_default_na=True)
#############################################################

#prediction and evaluation
#read points for contour
data_file = r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\inputs\Geographic_Data_2000.xls"
pred_inputs=pd.read_excel(data_file,sheet_name=0,header=0,index_col=False,keep_default_na=True)
##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
#to run ISOCOMPY for meteo excluding outliers
#iso_meteo_model_exc_outliers=tools.session.load(r'C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\new_output_06_dec_2020\iso_meteo_model_exc_outliers_07_Dec_2020_12_39_meteoFalse_isoFalse.pkl')
#preped_dataset_exc_outliers=tools.session.load(r'C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\new_output_06_dec_2020\preped_dataset_exc_outliers_07_Dec_2020_12_41_meteoFalse_isoFalse.pkl')
#iso_meteo_model_exc_outliers.which_regs_temp={"rfr":False,"mlp":False,"elnet":False,"omp":False,"br":False,"ard":False,"svr":False,"nusvr":False}
#iso_meteo_model_exc_outliers.which_regs_hum={"rfr":False,"mlp":False,"elnet":False,"omp":False,"br":False,"ard":False,"svr":False,"nusvr":False}

#data preparation
preped_dataset_exc_outliers=data_preparation.preprocess(meteo_input_type="monthly",direc=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\new_output_09_dec_2020_new_iso") #your working folder
preped_dataset_exc_outliers.fit(rain,temp,hum,iso_18,iso_2h,iso_3h)
#meteo model

iso_meteo_model_exc_outliers=reg_model.model()
iso_meteo_model_exc_outliers.meteo_fit(preped_dataset_exc_outliers,temp_fit=True,rain_fit=True,hum_fit=True)
tools.plots.best_estimator_plots(iso_meteo_model_exc_outliers,iso_plot=False)
tools.plots.partial_dep_plots(iso_meteo_model_exc_outliers,iso_plot=False)
ex_meteo_exc_outliers=tools.session.save(iso_meteo_model_exc_outliers,name='iso_meteo_model_exc_outliers') #optional. prints the directory
ex_prep_exc_outliers=tools.session.save(preped_dataset_exc_outliers,name='preped_dataset_exc_outliers') #optional. prints the directory
##########################################################################################################################

##########################################################################################################################
#load meteo
iso_meteo_model=tools.session.load(ex_meteo_exc_outliers)
preped_dataset=tools.session.load(ex_prep_exc_outliers)

#change directory
preped_dataset.direc=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\new_output_09_dec_2020_new_iso\iso_models\allyear_output_exc_outliers" 
#iso model all year
iso_meteo_model.iso_predic(preped_dataset,run_iso_whole_year=True,iso_model_month_list=[])
iso_meteo_model.iso_fit()
tools.plots.best_estimator_plots(iso_meteo_model,meteo_plot=False)
tools.plots.partial_dep_plots(iso_meteo_model,meteo_plot=False)
tools.stats.annual_stats(iso_meteo_model)
tools.stats.mensual_stats(iso_meteo_model)
iso_meteo_model_allyear_exc_outliers=tools.session.save(iso_meteo_model,name='iso_meteo_model_allyear_exc_outliers') #optional prints the directory
prepdataset_final_allyear_exc_outliers=tools.session.save(preped_dataset,name='prepdataset_final_allyear_exc_outliers') #optional prints the directory

##########################################################################################################################
##########################################################################################################################
#load meteo
iso_meteo_model=tools.session.load(ex_meteo_exc_outliers)
preped_dataset=tools.session.load(ex_prep_exc_outliers)

#change directory
preped_dataset.direc=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\new_output_09_dec_2020_new_iso\iso_models\678_output_exc_outliers"


# iso model 678
iso_meteo_model.iso_predic(preped_dataset,run_iso_whole_year=False,iso_model_month_list=[6,7,8])
iso_meteo_model.iso_fit()
tools.stats.annual_stats(iso_meteo_model)
tools.stats.mensual_stats(iso_meteo_model)
tools.plots.best_estimator_plots(iso_meteo_model,meteo_plot=False)
tools.plots.partial_dep_plots(iso_meteo_model,meteo_plot=False)
iso_meteo_model_678_exc_outliers=tools.session.save(iso_meteo_model,name='iso_meteo_model_678_exc_outliers') #optional prints the directory
prepdataset_final_678_exc_outliers=tools.session.save(preped_dataset,name='prepdataset_final_678_exc_outliers') #optional prints the directory
##########################################################################################################################
##########################################################################################################################
#load meteo
iso_meteo_model=tools.session.load(ex_meteo_exc_outliers)
preped_dataset=tools.session.load(ex_prep_exc_outliers)   

#change directory
preped_dataset.direc=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\new_output_09_dec_2020_new_iso\iso_models\12312_output_exc_outliers"

#iso model 12312
iso_meteo_model.iso_predic(preped_dataset,run_iso_whole_year=False,iso_model_month_list=[1,2,3,12])
iso_meteo_model.iso_fit()
tools.stats.annual_stats(iso_meteo_model)
tools.stats.mensual_stats(iso_meteo_model)
tools.plots.best_estimator_plots(iso_meteo_model,meteo_plot=False)
tools.plots.partial_dep_plots(iso_meteo_model,meteo_plot=False)
iso_meteo_model_12312_exc_outliers=tools.session.save(iso_meteo_model,name='iso_meteo_model_12312_exc_outliers') #optional prints the directory
prepdataset_final_12312_exc_outliers=tools.session.save(preped_dataset,name='prepdataset_final_12312_exc_outliers') #optional prints the directory
##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
#predictions
#excluding outliers

#calculate predictions
ev_class_allyear_exc=tools.evaluation()
iso_meteo_model_allyear_exc_outliers=tools.session.load(iso_meteo_model_allyear_exc_outliers)
pred_inputs=iso_meteo_model_allyear_exc_outliers.all_preds[["CooX","CooY","CooZ","month",'ID_MeteoPoint']]
ev_class_allyear_exc.predict(iso_meteo_model_allyear_exc_outliers,pred_inputs,direc=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\new_output_09_dec_2020_new_iso\predictions_excluding_outliers\preds_allyear")
tools.plots.isotopes_meteoline_plot(ev_class_allyear_exc,iso_meteo_model_allyear_exc_outliers,iso_18,iso_2h,month_data=True,obs_data=False)

#################
#calculate predictions
ev_class_678_exc=tools.evaluation()
iso_meteo_model_678_exc_outliers=tools.session.load(iso_meteo_model_678_exc_outliers)
pred_inputs=iso_meteo_model_678_exc_outliers.all_preds[["CooX","CooY","CooZ","month",'ID_MeteoPoint']]
ev_class_678_exc.predict(iso_meteo_model_678_exc_outliers,pred_inputs,direc=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\new_output_09_dec_2020_new_iso\predictions_excluding_outliers\preds_678")
tools.plots.isotopes_meteoline_plot(ev_class_678_exc,iso_meteo_model_678_exc_outliers,iso_18,iso_2h,month_data=True,obs_data=False)

#################
#################
#calculate predictions
ev_class_12312_exc=tools.evaluation()
iso_meteo_model_12312_exc_outliers=tools.session.load(iso_meteo_model_12312_exc_outliers)
pred_inputs=iso_meteo_model_12312_exc_outliers.all_preds[["CooX","CooY","CooZ","month",'ID_MeteoPoint']]
ev_class_12312_exc.predict(iso_meteo_model_12312_exc_outliers,pred_inputs,direc=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\new_output_09_dec_2020_new_iso\predictions_excluding_outliers\preds_12312")
tools.plots.isotopes_meteoline_plot(ev_class_12312_exc,iso_meteo_model_12312_exc_outliers,iso_18,iso_2h,month_data=True,obs_data=False)

import dill
dill.dump_session(r'C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\new_output_09_dec_2020_new_iso\whole_dump.pkl')

##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
#INCLUDING OUTLIERS
#to run ISOCOMPY for meteo including outliers
#data preparation


preped_dataset_inc_outliers=data_preparation.preprocess(meteo_input_type="no_filter" ,direc=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\new_output_4_dec_2020\meteo_models_inc_outliers") #your working folder
preped_dataset_inc_outliers.fit(rain,temp,hum,iso_18,iso_2h,iso_3h)
#meteo model
iso_meteo_model_inc_outliers=reg_model.model()
iso_meteo_model_inc_outliers.meteo_fit(preped_dataset_inc_outliers)
tools.plots.best_estimator_plots(iso_meteo_model_inc_outliers,iso_plot=False)
tools.plots.partial_dep_plots(iso_meteo_model_inc_outliers,iso_plot=False)
meteo_inc_outliers=tools.session.save(iso_meteo_model_inc_outliers,name='iso_meteo_model_inc_outliers') #optional. prints the directory
prep_inc_outliers=tools.session.save(preped_dataset_inc_outliers,name='preped_dataset_inc_outliers') #optional. prints the directory
##########################################################################################################################
##########################################################################################################################
#load meteo
iso_meteo_model=tools.session.load(meteo_inc_outliers)
preped_dataset=tools.session.load(prep_inc_outliers)

#change directory
preped_dataset.direc=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\new_output_27_nov_2020\iso_models\allyear_output_inc_outliers" 

#iso model all year
iso_meteo_model.iso_predic(preped_dataset,run_iso_whole_year=True,iso_model_month_list=[])
iso_meteo_model.iso_fit()
tools.plots.best_estimator_plots(iso_meteo_model,meteo_plot=False)
tools.plots.partial_dep_plots(iso_meteo_model,meteo_plot=False)
tools.stats.annual_stats(iso_meteo_model)
tools.stats.mensual_stats(iso_meteo_model)
iso_meteo_model_allyear_inc_outliers=tools.session.save(iso_meteo_model,name='iso_meteo_model_allyear_inc_outliers') #optional prints the directory
prepdataset_final_allyear_inc_outliers=tools.session.save(preped_dataset,name='prepdataset_final_allyear_inc_outliers') #optional prints the directory

##########################################################################################################################
##########################################################################################################################
#load meteo

iso_meteo_model=tools.session.load(meteo_inc_outliers)
preped_dataset=tools.session.load(prep_inc_outliers)

#change directory
preped_dataset.direc=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\new_output_27_nov_2020\iso_models\678_output_inc_outliers"


# iso model 678
iso_meteo_model.iso_predic(preped_dataset,run_iso_whole_year=False,iso_model_month_list=[6,7,8])
iso_meteo_model.iso_fit()
tools.stats.annual_stats(iso_meteo_model)
tools.stats.mensual_stats(iso_meteo_model)
tools.plots.best_estimator_plots(iso_meteo_model,meteo_plot=False)
tools.plots.partial_dep_plots(iso_meteo_model,meteo_plot=False)
iso_meteo_model_678_inc_outliers=tools.session.save(iso_meteo_model,name='iso_meteo_model_678_inc_outliers') #optional prints the directory
prepdataset_final_678_inc_outliers=tools.session.save(preped_dataset,name='prepdataset_final_678_inc_outliers') #optional prints the directory
##########################################################################################################################
##########################################################################################################################

#load meteo
iso_meteo_model=tools.session.load(meteo_inc_outliers)
preped_dataset=tools.session.load(prep_inc_outliers)   

#change directory
preped_dataset.direc=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\new_output_27_nov_2020\iso_models\12312_output_inc_outliers"
 

#iso model 12312
iso_meteo_model.iso_predic(preped_dataset,run_iso_whole_year=False,iso_model_month_list=[1,2,3,12])
iso_meteo_model.iso_fit()
tools.stats.annual_stats(iso_meteo_model)
tools.stats.mensual_stats(iso_meteo_model)
tools.plots.best_estimator_plots(iso_meteo_model,meteo_plot=False)
tools.plots.partial_dep_plots(iso_meteo_model,meteo_plot=False)
iso_meteo_model_12312_inc_outliers=tools.session.save(iso_meteo_model,name='iso_meteo_model_12312_inc_outliers') #optional prints the directory
prepdataset_final_12312_inc_outliers=tools.session.save(preped_dataset,name='prepdataset_final_12312_inc_outliers') #optional prints the directory
##########################################################################################################################
##########################################################################################################################
##########################################################################################################################

#predictions
#including outliers
#calculate predictions

##########################################################################################################################
ev_class_allyear_inc=tools.evaluation()
iso_meteo_model_allyear_inc_outliers=tools.session.load(iso_meteo_model_allyear_inc_outliers)
pred_inputs=iso_meteo_model_allyear_inc_outliers.all_preds[["CooX","CooY","CooZ","month",'ID_MeteoPoint']]
ev_class_allyear_inc.predict(iso_meteo_model_allyear_inc_outliers,pred_inputs,direc=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\new_output_27_nov_2020\predictions_including_outliers\preds_allyear")
tools.plots.isotopes_meteoline_plot(ev_class_allyear_inc,iso_meteo_model_allyear_inc_outliers,iso_18,iso_2h,month_data=True,obs_data=False)
#################
#################
#calculate predictions
ev_class_678_inc=tools.evaluation()
iso_meteo_model_678_inc_outliers=tools.session.load(iso_meteo_model_678_inc_outliers)
pred_inputs=iso_meteo_model_678_inc_outliers.all_preds[["CooX","CooY","CooZ","month",'ID_MeteoPoint']]
ev_class_678_inc.predict(iso_meteo_model_678_inc_outliers,pred_inputs,direc=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\new_output_27_nov_2020\predictions_including_outliers\preds_678")
tools.plots.isotopes_meteoline_plot(ev_class_678_inc,iso_meteo_model_678_inc_outliers,iso_18,iso_2h,month_data=True,obs_data=False)

#################
#################
#calculate predictions
ev_class_12312_inc=tools.evaluation()
iso_meteo_model_12312_inc_outliers=tools.session.load(iso_meteo_model_12312_inc_outliers)
pred_inputs=iso_meteo_model_12312_inc_outliers.all_preds[["CooX","CooY","CooZ","month",'ID_MeteoPoint']]
ev_class_12312_inc.predict(iso_meteo_model_12312_inc_outliers,pred_inputs,direc=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\new_output_27_nov_2020\predictions_including_outliers\preds_12312")
tools.plots.isotopes_meteoline_plot(ev_class_12312_inc,iso_meteo_model_12312_inc_outliers,iso_18,iso_2h,month_data=True,obs_data=False)
 
import dill
dill.load_session(r'C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\new_output_27_nov_2020\whole_dump.pkl')
##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
##########################################################################################################################


################################################
###############################################
###########################################
#iso model all year
ex_meteo=r'C:\\Users\\Ash kan\\Documents\\meteo_iso_model\\meteo_iso_model_input_code_and_results\\allyear_output_31oct\\exc_iso_meteo_model_allyear31_Oct_2020_15_24_meteoTrue_isoTrue.pkl'
iso_meteo_model=tools.session.load(ex_meteo)
iso_meteo_model.direc=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\test_svr_allyear" 
iso_meteo_model.iso_predic(preped_dataset,run_iso_whole_year=True,iso_model_month_list=[])

#read points for contour
data_file = r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\inputs\Geographic_Data_2000.xls"
pred_inputs=pd.read_excel(data_file,sheet_name=0,header=0,index_col=False,keep_default_na=True)
ev_class_t=tools.evaluation()

ev_class_t.predict(iso_meteo_model,pred_inputs,dir=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\preds_test_svr_allyear")
#prediction plots
tools.session.save(ev_class_t)
dirr=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\preds_test_svr_allyear\isocompy_saved_session_10_Nov_2020_13_31_meteoFalse_isoFalse.pkl" 
ev_class_t=tools.session.load(dirr)
tools.plots.isotopes_meteoline_plot(ev_class_t,iso_meteo_model)

#load data all year
iso_meteo_model_28_all=tools.session.load(r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\all_year_28_oct\exc_iso_meteo_model_allyear19_Oct_2020_20_09_meteoFalse_isoFalse.pkl")

#calculate predictions
pred_28=tools.evaluation()
pred_28.predict(iso_meteo_model_28_all,pred_inputs,dir=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\preds_allyear_28oct")

#################
#load data all year
iso_meteo_model_31_all=tools.session.load(r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\allyear_output_31oct\exc_iso_meteo_model_allyear31_Oct_2020_15_24_meteoTrue_isoTrue.pkl")

#calculate predictions
pred_31=tools.evaluation()
pred_31.predict(iso_meteo_model_28_all,pred_inputs,dir=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\preds_allyear")

#################
#################

a=iso_meteo_model_28_all.iso18_bests_dic[0]["best_estimator"].best_estimator_
a.support_vectors_.shape
iso_meteo_model_28_all.iso18_preds_real_dic[0]
pd.DataFrame( iso_meteo_model_28_all.iso18_bests_dic[0]["best_estimator"].cv_results_).to_excel(r"C:\Users\Ash kan\Desktop\gridres.xls")

iso_meteo_model_28_all.iso18_bests_dic[0]["best_estimator"].best_score_

from statsmodels.stats.power import TTestPower
power_analysis = TTestPower()
power_analysisf=FTestPower()
r2=iso_meteo_model_12312_inc_outliers.iso18_bests_dic[0]["rsquared"]
nobs=iso_meteo_model_12312_inc_outliers.iso18_preds_real_dic[0]['Y_preds'].shape[0]
nof=len(iso_meteo_model_12312_inc_outliers.iso18_bests_dic[0]["used_features"])
power_analysisf.solve_power(effect_size=r2/(1-r2), df_num=nof, df_denom=nobs - nof -1, nobs=nobs, alpha=0.05, power=None,ncc=0)
power_analysis.solve_power(effect_size=r2/(1-r2),  alpha=0.05, power=None,nobs=nobs)

s=power_analysis.solve_power(effect_size=(r2/(1-r2)), df_num=nof, df_denom=None, nobs=None, alpha=0.05, power=0.8,ncc=0)
fig=TTestPower().plot_power(  effect_size=np.array([r2/(1-r2),0.8]),nobs=np.arange(5, 150), alpha=0.05, dep_var='nobs', ax=None, title=None)
fig.show()
#another time with df_denom=None
import dill

dill.load_session(r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\new_output_11_nov_2020\isocompy_session_11_nov_2020.pkl")
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

ex_prep_exc_outliers='C:\\Users\\Ash kan\\Documents\\meteo_iso_model\\meteo_iso_model_input_code_and_results\\new_output_11_nov_2020\\meteo_models_exc_outliers\\preped_dataset_exc_outliers10_Nov_2020_20_28_meteoFalse_isoFalse.pkl'
ex_meteo_exc_outliers='C:\\Users\\Ash kan\\Documents\\meteo_iso_model\\meteo_iso_model_input_code_and_results\\new_output_11_nov_2020\\meteo_models_exc_outliers\\iso_meteo_model_exc_outliers10_Nov_2020_20_28_meteoTrue_isoFalse.pkl'
meteo_inc_outliers='C:\\Users\\Ash kan\\Documents\\meteo_iso_model\\meteo_iso_model_input_code_and_results\\new_output_11_nov_2020\\meteo_models_inc_outliers\\iso_meteo_model_inc_outliers10_Nov_2020_22_45_meteoTrue_isoFalse.pkl'
prep_inc_outliers='C:\\Users\\Ash kan\\Documents\\meteo_iso_model\\meteo_iso_model_input_code_and_results\\new_output_11_nov_2020\\meteo_models_inc_outliers\\preped_dataset_inc_outliers10_Nov_2020_22_45_meteoFalse_isoFalse.pkl'


prep_exc_outliers=tools.session.load(ex_prep_exc_outliers)
for j in prep_exc_outliers.month_grouped_iso18:

    df=j[["CooX"      ,   "CooY" ,   "CooZ" ]]
    X = add_constant(df)
    pd.DataFrame([variance_inflation_factor(X.values, i) 
                for i in range(X.shape[1])], 
                index=X.columns).rename(columns={0:'VIF'})

j=iso_meteo_model_12312_inc_outliers.all_preds
df=j[["CooX"      ,   "CooY" ,   "CooZ", "temp",'rain', 'hum' ]]
X = add_constant(df)
vif_df=pd.DataFrame([variance_inflation_factor(X.values, i) 
            for i in range(X.shape[1])], 
            index=X.columns).rename(columns={0:'VIF'}).drop('const')
fff=vif_df[vif_df["VIF"]>20].index       

'''
    #LOGGED!
    #load meteo
    iso_meteo_model=tools.session.load(ex_meteo)
    preped_dataset=tools.session.load(ex_prep)   

    #change directory
    
    #############################################################
    #iso model 12312
    iso_meteo_model.apply_on_log=False
    iso_meteo_model.iso_predic(preped_dataset,run_iso_whole_year=False,iso_model_month_list=[1,2,3,12])
    iso_meteo_model.iso_fit()
    tools.stats.annual_stats(iso_meteo_model)
    tools.stats.mensual_stats(iso_meteo_model)
    tools.plots.best_estimator_plots(iso_meteo_model,meteo_plot=False)
    tools.plots.partial_dep_plots(iso_meteo_model,meteo_plot=False)
    tools.session.save(iso_meteo_model,name='exc_iso_meteo_model_12312_notlogged') #optional prints the directory
    tools.session.save(preped_dataset,name='exc_prepdataset_final_12312_notlogged') #optional prints the directory
    ##########################################################################################################################
 '''
 ##########################################################################################################################
#load meteo
iso_meteo_model=tools.session.load(meteo_inc_outliers)
preped_dataset=tools.session.load(prep_inc_outliers)   

#change directory
preped_dataset.direc=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\new_output\12312_output_inc_outliers_not_logged"
 

#iso model 12312 not logged
iso_meteo_model.iso_predic(preped_dataset,run_iso_whole_year=False,iso_model_month_list=[1,2,3,12])
iso_meteo_model.apply_on_log=False
iso_meteo_model.iso_fit()
tools.stats.annual_stats(iso_meteo_model)
tools.stats.mensual_stats(iso_meteo_model)
tools.plots.best_estimator_plots(iso_meteo_model,meteo_plot=False)
tools.plots.partial_dep_plots(iso_meteo_model,meteo_plot=False)
iso_meteo_model_12312_inc_outliers_not_logged=tools.session.save(iso_meteo_model,name='iso_meteo_model_12312_inc_outliers_not_logged') #optional prints the directory
prepdataset_final_12312_inc_outliers_not_logged=tools.session.save(preped_dataset,name='prepdataset_final_12312_inc_outliers_not_logged') #optional prints the directory
#################
#################
#calculate predictions
ev_class_12312_inc_not_logged=tools.evaluation()
iso_meteo_model_12312_inc_outliers_not_logged=tools.session.load(iso_meteo_model_12312_inc_outliers_not_logged)
ev_class_12312_inc_not_logged.predict(iso_meteo_model_12312_inc_outliers_not_logged,pred_inputs,direc=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\new_output\predictions_including_outliers\preds_12312_not_logged")
tools.plots.isotopes_meteoline_plot(ev_class_12312_inc_not_logged,iso_meteo_model_12312_inc_outliers_not_logged)
##########################################################################################################################
from sklearn.preprocessing import MinMaxScaler
import itertools
iso_meteo_modellist=list()
wan_mon=[8]
for m in wan_mon:
    #load meteo
    iso_meteo_model=tools.session.load(r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\new_output_09_dec_2020_new_iso\iso_meteo_model_exc_outliers_09_Dec_2020_16_59_meteoTrue_isoFalse.pkl")
    preped_dataset=tools.session.load(r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\new_output_09_dec_2020_new_iso\preped_dataset_exc_outliers_09_Dec_2020_16_59_meteoFalse_isoFalse.pkl")   
    iso_meteo_model.cv=3
    iso_meteo_model.vif_threshold_iso18=None
    iso_meteo_model.vif_threshold_iso2h=None
    iso_meteo_model.vif_threshold_iso3h=None
    #iso_meteo_model.tunedpars_svr_iso18['gamma']=[1.e-08, 1.e-06, 1.e-04]
    #iso_meteo_model.tunedpars_svr_iso2h['gamma']=[1.e-08, 1.e-06, 1.e-04]
    #iso_meteo_model.tunedpars_svr_iso3h['gamma']=[1.e-08, 1.e-06, 1.e-04]
    iso_meteo_model.which_regs_iso18['svr']=False
    iso_meteo_model.which_regs_iso18['elnet']=False
    iso_meteo_model.which_regs_iso2h['svr']=False
    iso_meteo_model.which_regs_iso2h['elnet']=False
    iso_meteo_model.which_regs_iso3h['svr']=False
    iso_meteo_model.which_regs_iso3h['elnet']=False


#change directory
    preped_dataset.direc=os.path.join(r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\new_output_09_dec_2020_indv_month_v2\iso_models",str(m)+"_output_inc_outliers")
 

    #iso model 2
    iso_meteo_model.iso_predic(preped_dataset,run_iso_whole_year=False,iso_model_month_list=[m])
    iso_meteo_model.iso_fit()
    tools.stats.annual_stats(iso_meteo_model)
    tools.plots.best_estimator_plots(iso_meteo_model,meteo_plot=False)
    tools.plots.partial_dep_plots(iso_meteo_model,meteo_plot=False)
    iso_meteo_model_inc_outliers=tools.session.save(iso_meteo_model,name='iso_meteo_model_'+str(m)+'_exc_outliers') #optional prints the directory
    prepdataset_final_inc_outliers=tools.session.save(preped_dataset,name='prepdataset_final_'+str(m)+'_exc_outliers') #optional prints the directory
    iso_meteo_modellist.append(iso_meteo_model)
    #########

for (iso_meteo_model_inc_outliers,m) in zip(iso_meteo_modellist,wan_mon):
    #calculate predictions
    ev_class_1_inc=tools.evaluation()
    #iso_meteo_model_inc_outliers=tools.session.load(iso_meteo_model_inc_outliers)
    pred_inputs=iso_meteo_model_inc_outliers.all_preds[["CooX","CooY","CooZ","month",'ID_MeteoPoint']]
    ev_class_1_inc.predict(iso_meteo_model_inc_outliers,pred_inputs,direc=os.path.join(r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\new_output_09_dec_2020_indv_month_v2","preds_"+str(m)))
    tools.plots.isotopes_meteoline_plot(ev_class_1_inc,iso_meteo_model_inc_outliers,iso_18,iso_2h,month_data=True,obs_data=False,id_point=True,residplot=True)
    ##########################################################
    ######################################################
    iso='iso_18'
    fig, axs =plt.subplots(1,len(iso_meteo_model_inc_outliers.iso18_bests_dic[0]["used_features"]))
    for count,fes in enumerate(iso_meteo_model_inc_outliers.iso18_bests_dic[0]["used_features"]):
        scaler1 = MinMaxScaler()
        scaler2 = MinMaxScaler()
        x=scaler1.fit_transform(iso_meteo_model_inc_outliers.all_preds[[iso]])
        y=scaler2.fit_transform(iso_meteo_model_inc_outliers.all_preds[[fes]])

        axs[count].scatter(x,y)
        axs[count].set_title(str(m)+"_"+iso+"_"+fes)
    fig.savefig(os.path.join(r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\new_output_09_dec_2020_indv_month_v2","preds_"+str(m),str(m)+"_"+iso+'.png'))
    plt.close()
    iso='iso_2h'
    fig, axs =plt.subplots(1,len(iso_meteo_model_inc_outliers.iso2h_bests_dic[0]["used_features"]))

    for count,fes in enumerate(iso_meteo_model_inc_outliers.iso2h_bests_dic[0]["used_features"]):
        scaler1 = MinMaxScaler()
        scaler2 = MinMaxScaler()

        axs[count].scatter(scaler1.fit_transform(iso_meteo_model_inc_outliers.all_preds[[iso]]),scaler2.fit_transform(iso_meteo_model_inc_outliers.all_preds[[fes]]))
        axs[count].set_title(str(m)+"_"+iso+"_"+fes)
    fig.savefig(os.path.join(r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\new_output_09_dec_2020_indv_month_v2","preds_"+str(m),str(m)+"_"+iso+".png"))
    plt.close()


import dill
dill.dump_session(r'C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\new_output_09_dec_2020_indv_month_v2\whole_dump.pkl')


#test sorting problem
db_sortred=tools.session.load(r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\new_output_08_dec_2020_scorexy\iso_models\678_output_exc_outliers\prepdataset_final_678_exc_outliers_08_Dec_2020_20_30_meteoFalse_isoFalse.pkl")
iso_sorted=tools.session.load(r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\new_output_08_dec_2020_scorexy\iso_models\678_output_exc_outliers\iso_meteo_model_678_exc_outliers_08_Dec_2020_20_30_meteoTrue_isoTrue.pkl")

iso_sorted.all_preds[["ID_MeteoPoint","iso_18","iso_2h","month"]]
db_sortred.month_grouped_iso_18[5][["ID_MeteoPoint","iso_18"]]
db_sortred.month_grouped_iso_2h[5][["ID_MeteoPoint","iso_2h"]]

iso_sorted.all_preds[iso_sorted.all_preds["month"]==7][["ID_MeteoPoint","iso_18","iso_2h","month"]]
db_sortred.month_grouped_iso_18[6][["ID_MeteoPoint","iso_18"]]
db_sortred.month_grouped_iso_2h[6][["ID_MeteoPoint","iso_2h"]]

iso_sorted.all_preds[["ID_MeteoPoint","iso_18","iso_2h","month"]]
db_sortred.month_grouped_iso_18[7][["ID_MeteoPoint","iso_18"]]
db_sortred.month_grouped_iso_2h[7][["ID_MeteoPoint","iso_2h"]]


#not sorted db
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from isocompy import data_preparation,reg_model,tools
import os


#######################
#importing data
#temp=pd.read_excel(r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\inputs\T_Daily.xlsx",sheet_name="T_Daily",header=0,index_col=False,keep_default_na=True)
temp=pd.read_excel(r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\inputs\temp_monthly_outliers_removed_07_12_2020.xls",sheet_name="temp_monthly_outliers_removed_7",header=0,index_col=False,keep_default_na=True)


#hum=pd.read_excel(r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\inputs\RH_Daily_0_100.xlsx",sheet_name="TODAS",header=0,index_col=False,keep_default_na=True)
hum=pd.read_excel(r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\inputs\hum_monthly_outliers_removed_07_12_2020.xls",sheet_name="hum_monthly_outliers_removed_7",header=0,index_col=False,keep_default_na=True)



#rain csv Ashkan 2020
#data_file = r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\inputs\pp_daily_all.csv"
#rain = pd.read_csv(data_file,sep=';',header=0,index_col=False,keep_default_na=True)
#date_format='%d/%m/%Y'
#rain['Date'] = pd.to_datetime(rain['Date'],format=date_format)
rain=pd.read_excel(r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\inputs\rain_monthly_outliers_removed_08_12_2020.xls",sheet_name="rain_monthly_outliers_removed_2",header=0,index_col=False,keep_default_na=True)

#############################################################    

#isotope files
data_file_iso = r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\inputs\Isotopes_16_11_2020_not_sorted.xls"
iso_18 = pd.read_excel(data_file_iso,sheet_name="ISOT18O",header=0,index_col=False,keep_default_na=True)
iso_2h=pd.read_excel(data_file_iso,sheet_name="ISOT2H",header=0,index_col=False,keep_default_na=True)
iso_3h=pd.read_excel(data_file_iso,sheet_name="ISOT3",header=0,index_col=False,keep_default_na=True)


db_not_sortred=data_preparation.preprocess(meteo_input_type="monthly",direc=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\test_12_dec_not_sorted") #your working folder
db_not_sortred.fit(rain,temp,hum,iso_18,iso_2h,iso_3h)

iso_not_sorted=tools.session.load(r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\new_output_08_dec_2020_scorexy\iso_meteo_model_exc_outliers_08_Dec_2020_19_13_meteoTrue_isoFalse.pkl")
iso_not_sorted.iso_predic(db_not_sortred,run_iso_whole_year=True,iso_model_month_list=[])
#############################################################
a=db_not_sortred.month_grouped_iso_2h[7].sort_values(by="ID_MeteoPoint",ascending=False)
b=db_not_sortred.month_grouped_iso_18[7]
d=db_not_sortred.month_grouped_iso_3h[7]
c=pd.concat([a["iso_2h"],b["iso_18"],d["iso_3h"]],axis=1,ignore_index=True,join="inner")
c.columns
