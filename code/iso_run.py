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
data_file_iso = r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\inputs\pysplit_isotope_registro_Pp_19_10_2020.xlsx"
iso_18 = pd.read_excel(data_file_iso,sheet_name="ISOT18O",header=0,index_col=False,keep_default_na=True)
iso_2h=pd.read_excel(data_file_iso,sheet_name="ISOT2H",header=0,index_col=False,keep_default_na=True)
iso_3h=pd.read_excel(data_file_iso,sheet_name="ISOT3",header=0,index_col=False,keep_default_na=True)
#############################################################

#to run ISOCOMPY for meteo
#data preparation
preped_dataset=data_preparation.preprocess(direc=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\meteo_models") #your working folder
preped_dataset.fit(rain,temp,hum,iso_18,iso_2h,iso_3h)
#meteo model
iso_meteo_model=reg_model.model()
iso_meteo_model.meteo_fit(preped_dataset)
tools.plots.best_estimator_plots(iso_meteo_model,iso_plot=False)
tools.plots.partial_dep_plots(iso_meteo_model,iso_plot=False)
ex_meteo=tools.session.save(iso_meteo_model,name='exc_meteo_model') #optional. prints the directory
ex_prep=tools.session.save(preped_dataset,name='exc_prepdataset') #optional. prints the directory
#############################################################
#iso model all year
ex_meteo=r'C:\\Users\\Ash kan\\Documents\\meteo_iso_model\\meteo_iso_model_input_code_and_results\\meteo_models\\exc_meteo_model30_Oct_2020_21_52_meteoTrue_isoFalse.pkl'
ex_prep=r'C:\\Users\\Ash kan\\Documents\\meteo_iso_model\\meteo_iso_model_input_code_and_results\\meteo_models\\exc_prepdataset30_Oct_2020_21_52_meteoFalse_isoFalse.pkl'  
iso_meteo_model=tools.session.load(ex_meteo)
preped_dataset=tools.session.load(ex_prep)
preped_dataset.direc=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\allyear_output" 
iso_meteo_model.iso_predic(preped_dataset,run_iso_whole_year=True,iso_model_month_list=[])
iso_meteo_model.iso_fit()
tools.plots.best_estimator_plots(iso_meteo_model,meteo_plot=False)
tools.plots.partial_dep_plots(iso_meteo_model,meteo_plot=False)
tools.stats.annual_stats(iso_meteo_model)
tools.stats.mensual_stats(iso_meteo_model)
tools.session.save(iso_meteo_model,name='exc_iso_meteo_model_allyear_justsvr') #optional prints the directory
tools.session.save(preped_dataset,name='exc_prepdataset_final_allyear_justsvr') #optional prints the directory

##########################################################################################################################
##########################################################################################################################
#load meteo
iso_meteo_model=tools.session.load(ex_meteo)
preped_dataset=tools.session.load(ex_prep)

#change directory
preped_dataset.direc=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\678_output"

#############################################################
# iso model 678
iso_meteo_model.iso_predic(preped_dataset,run_iso_whole_year=False,iso_model_month_list=[6,7,8])
iso_meteo_model.iso_fit()
tools.stats.annual_stats(iso_meteo_model)
tools.stats.mensual_stats(iso_meteo_model)
tools.plots.best_estimator_plots(iso_meteo_model,meteo_plot=False)
tools.plots.partial_dep_plots(iso_meteo_model,meteo_plot=False)
tools.session.save(iso_meteo_model,name='exc_iso_meteo_model_678') #optional prints the directory
tools.session.save(preped_dataset,name='exc_prepdataset_final_678') #optional prints the directory
##########################################################################################################################
##########################################################################################################################
#load meteo
iso_meteo_model=tools.session.load(ex_meteo)
preped_dataset=tools.session.load(ex_prep)   

#change directory
preped_dataset.direc=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\12312_output"
 
#############################################################
#iso model 12312
iso_meteo_model.iso_predic(preped_dataset,run_iso_whole_year=False,iso_model_month_list=[1,2,3,12])
iso_meteo_model.iso_fit()
tools.stats.annual_stats(iso_meteo_model)
tools.stats.mensual_stats(iso_meteo_model)
tools.plots.best_estimator_plots(iso_meteo_model,meteo_plot=False)
tools.plots.partial_dep_plots(iso_meteo_model,meteo_plot=False)
tools.session.save(iso_meteo_model,name='exc_iso_meteo_model_12312') #optional prints the directory
tools.session.save(preped_dataset,name='exc_prepdataset_final_12312') #optional prints the directory
##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
#load meteo
iso_meteo_model=tools.session.load(ex_meteo)
preped_dataset=tools.session.load(ex_prep)   

#change directory
preped_dataset.direc=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\12312_output_notlogged"
 
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
##########################################################################################################################
#prediction and evaluation
#read points for contour
data_file = r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\inputs\Geographic_Data_2000.xls"
pred_inputs=pd.read_excel(data_file,sheet_name=0,header=0,index_col=False,keep_default_na=True)
###################################################
#load data all year
iso_meteo_model=tools.session.load(r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\allyear_output\exc_iso_meteo_model_allyear31_Oct_2020_15_24_meteoTrue_isoTrue.pkl")

#calculate predictions
ev_class_allyear=tools.evaluation()
ev_class_allyear.predict(iso_meteo_model,pred_inputs,dir=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\preds_allyear")
#################
#################
#load data 678
iso_meteo_model=tools.session.load(r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\678_output\exc_iso_meteo_model_67831_Oct_2020_18_08_meteoTrue_isoTrue.pkl")

#calculate predictions
ev_class_678=tools.evaluation()
ev_class_678.predict(iso_meteo_model,pred_inputs,dir=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\preds_678")
#################
#################
#load data 12312
iso_meteo_model=tools.session.load(r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\12312_output\exc_iso_meteo_model_1231231_Oct_2020_20_12_meteoTrue_isoTrue.pkl")

#calculate predictions
ev_class_12312=tools.evaluation()
ev_class_12312.predict(iso_meteo_model,pred_inputs,dir=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\preds_12312")
#################
#################
#load data 12312 not logged
iso_meteo_model=tools.session.load(r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\12312_output_notlogged\exc_iso_meteo_model_12312_notlogged31_Oct_2020_22_31_meteoTrue_isoTrue.pkl")

#calculate predictions
ev_class_12312=tools.evaluation()
ev_class_12312.predict(iso_meteo_model,pred_inputs,dir=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\preds_12312_notlogged")
###############################################
################################################
###############################################
###########################################
#iso model all year
ex_meteo=r'C:\\Users\\Ash kan\\Documents\\meteo_iso_model\\meteo_iso_model_input_code_and_results\\meteo_models\\exc_meteo_model30_Oct_2020_21_52_meteoTrue_isoFalse.pkl'
ex_prep=r'C:\\Users\\Ash kan\\Documents\\meteo_iso_model\\meteo_iso_model_input_code_and_results\\meteo_models\\exc_prepdataset30_Oct_2020_21_52_meteoFalse_isoFalse.pkl'  
iso_meteo_model=tools.session.load(ex_meteo)
preped_dataset=tools.session.load(ex_prep)
preped_dataset.direc=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\test_svr_allyear" 
iso_meteo_model.iso_predic(preped_dataset,run_iso_whole_year=True,iso_model_month_list=[])
iso_meteo_model.iso_fit()
tools.plots.best_estimator_plots(iso_meteo_model,meteo_plot=False)
tools.plots.partial_dep_plots(iso_meteo_model,meteo_plot=False)
tools.stats.annual_stats(iso_meteo_model)
tools.stats.mensual_stats(iso_meteo_model)
tools.session.save(iso_meteo_model,name='exc_iso_meteo_model_allyear_justsvr') #optional prints the directory
tools.session.save(preped_dataset,name='exc_prepdataset_final_allyear_justsvr') #optional prints the directory
#read points for contour
data_file = r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\inputs\Geographic_Data_2000.xls"
pred_inputs=pd.read_excel(data_file,sheet_name=0,header=0,index_col=False,keep_default_na=True)
ev_class_t=tools.evaluation()
tools.session.load(??)
ev_class_t.predict(iso_meteo_model,pred_inputs,dir=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\preds_test_svr_allyear")
#prediction plots
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