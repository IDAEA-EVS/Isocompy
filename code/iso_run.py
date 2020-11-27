import numpy as np
import pandas as pd
from isocompy import data_preparation,reg_model,tools

#######################
#importing data
temp=pd.read_excel(r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\inputs\T_Daily.xlsx",sheet_name="T_Daily",header=0,index_col=False,keep_default_na=True)
hum=pd.read_excel(r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\inputs\RH_Daily_0_100.xlsx",sheet_name="TODAS",header=0,index_col=False,keep_default_na=True)
#rain csv Ashkan 2020
data_file = r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\inputs\pp_daily_all.csv"
rain = pd.read_csv(data_file,sep=';',header=0,index_col=False,keep_default_na=True)
date_format='%d/%m/%Y'
rain['Date'] = pd.to_datetime(rain['Date'],format=date_format)
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
#data preparation
preped_dataset_exc_outliers=data_preparation.preprocess(direc=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\new_output_17_nov_2020\meteo_models_exc_outliers") #your working folder
preped_dataset_exc_outliers.fit(rain,temp,hum,iso_18,iso_2h,iso_3h)
#meteo model
iso_meteo_model_exc_outliers=reg_model.model()
iso_meteo_model_exc_outliers.meteo_fit(preped_dataset_exc_outliers)
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
preped_dataset.direc=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\new_output_17_nov_2020\iso_models\allyear_output_exc_outliers" 
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
preped_dataset.direc=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\new_output_17_nov_2020\iso_models\678_output_exc_outliers"


# iso model 678
iso_meteo_model.iso_predic(preped_dataset,run_iso_whole_year=False,iso_model_month_list=[6,7,8])
iso_meteo_model.iso_fit(p_val=0.07)
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
preped_dataset.direc=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\new_output_17_nov_2020\iso_models\12312_output_exc_outliers"

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
ev_class_allyear_exc.predict(iso_meteo_model_allyear_exc_outliers,pred_inputs,direc=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\new_output_17_nov_2020\predictions_excluding_outliers\preds_allyear")
tools.plots.isotopes_meteoline_plot(ev_class_allyear_exc,iso_meteo_model_allyear_exc_outliers,iso_18,iso_2h)

#################
#################
#calculate predictions
ev_class_678_exc=tools.evaluation()
iso_meteo_model_678_exc_outliers=tools.session.load(iso_meteo_model_678_exc_outliers)
ev_class_678_exc.predict(iso_meteo_model_678_exc_outliers,pred_inputs,direc=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\new_output_17_nov_2020\predictions_excluding_outliers\preds_678")
tools.plots.isotopes_meteoline_plot(ev_class_678_exc,iso_meteo_model_678_exc_outliers,iso_18,iso_2h)

#################
#################
#calculate predictions
ev_class_12312_exc=tools.evaluation()
iso_meteo_model_12312_exc_outliers=tools.session.load(iso_meteo_model_12312_exc_outliers)
ev_class_12312_exc.predict(iso_meteo_model_12312_exc_outliers,pred_inputs,direc=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\new_output_17_nov_2020\predictions_excluding_outliers\preds_12312")
tools.plots.isotopes_meteoline_plot(ev_class_12312_exc,iso_meteo_model_12312_exc_outliers,iso_18,iso_2h)

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


preped_dataset_inc_outliers=data_preparation.preprocess(meteo_input_type="no_filter" ,direc=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\new_output_17_nov_2020\meteo_models_inc_outliers") #your working folder
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
preped_dataset.direc=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\new_output_17_nov_2020\iso_models\allyear_output_inc_outliers" 

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
preped_dataset.direc=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\new_output_17_nov_2020\iso_models\678_output_inc_outliers"


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
preped_dataset.direc=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\new_output_17_nov_2020\iso_models\12312_output_inc_outliers"
 

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
ev_class_allyear_inc.predict(iso_meteo_model_allyear_inc_outliers,pred_inputs,direc=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\new_output_17_nov_2020\predictions_including_outliers\preds_allyear")
tools.plots.isotopes_meteoline_plot(ev_class_allyear_inc,iso_meteo_model_allyear_inc_outliers,iso_18,iso_2h)
#################
#################
#calculate predictions
ev_class_678_inc=tools.evaluation()
iso_meteo_model_678_inc_outliers=tools.session.load(iso_meteo_model_678_inc_outliers)
ev_class_678_inc.predict(iso_meteo_model_678_inc_outliers,pred_inputs,direc=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\new_output_17_nov_2020\predictions_including_outliers\preds_678")
tools.plots.isotopes_meteoline_plot(ev_class_678_inc,iso_meteo_model_678_inc_outliers,iso_18,iso_2h)

#################
#################
#calculate predictions
ev_class_12312_inc=tools.evaluation()
iso_meteo_model_12312_inc_outliers=tools.session.load(iso_meteo_model_12312_inc_outliers)
ev_class_12312_inc.predict(iso_meteo_model_12312_inc_outliers,pred_inputs,direc=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\new_output_17_nov_2020\predictions_including_outliers\preds_12312")
tools.plots.isotopes_meteoline_plot(ev_class_12312_inc,iso_meteo_model_12312_inc_outliers,iso_18,iso_2h)
 
import dill
dill.dump_session(r'C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\new_output_17_nov_2020\whole_dump.pkl')
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