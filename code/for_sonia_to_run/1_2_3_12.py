from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from isocompy import data_preparation,reg_model,tools
import os

#importing data
temp=pd.read_excel(r"C:\Users\usuari\Desktop\10_12_2020\input\temp_monthly_outliers_removed_07_12_2020.xls",sheet_name="temp_monthly_outliers_removed_2",header=0,index_col=False,keep_default_na=True)

hum=pd.read_excel(r"C:\Users\usuari\Desktop\10_12_2020\input\hum_monthly_outliers_removed_07_12_2020.xls",sheet_name="hum_monthly_outliers_removed_07",header=0,index_col=False,keep_default_na=True)

#rain csv Ashkan 2020
#data_file = r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\inputs\pp_daily_all.csv"
#rain = pd.read_csv(data_file,sep=';',header=0,index_col=False,keep_default_na=True)
#date_format='%d/%m/%Y'
#rain['Date'] = pd.to_datetime(rain['Date'],format=date_format)
rain=pd.read_excel(r"C:\Users\usuari\Desktop\10_12_2020\input\rain_monthly_outliers_removed_08_12_2020.xls",sheet_name="rain_monthly_outliers_removed_2",header=0,index_col=False,keep_default_na=True)

#############################################################    
#isotope files
data_file_iso = r"C:\Users\usuari\Desktop\10_12_2020\input\Isotopes_11_12_2020.xls"
iso_18 = pd.read_excel(data_file_iso,sheet_name="ISOT18O",header=0,index_col=False,keep_default_na=True)
iso_2h=pd.read_excel(data_file_iso,sheet_name="ISOT2H",header=0,index_col=False,keep_default_na=True)
iso_3h=pd.read_excel(data_file_iso,sheet_name="ISOT3",header=0,index_col=False,keep_default_na=True)

##########################################################################################################################
#data preparation
preped_dataset=data_preparation.preprocess(meteo_input_type="monthly",direc=r"C:\Users\usuari\Desktop\10_12_2020\meteo_models\12_12_2020") #your working folder
#meteo_input_type="daily_remove_outliers"
preped_dataset.fit(rain,temp,hum,iso_18,iso_2h,iso_3h)
#change directory to put the iso info
preped_dataset.direc=r"C:\Users\usuari\Desktop\10_12_2020\iso_models\12_12_2020\1_2_3_12_output_exc_outliers_Year" 
##########################################################################################################################
#colgas meeo model:
iso_meteo_model=tools.session.load(r"C:\Users\usuari\Desktop\10_12_2020\meteo_models\new_output_10_dec_2020_new_iso\iso_meteo_model_exc_outliers_11_Dec_2020_21_23_meteoTrue_isoFalse.pkl") #r"directoy of .pkl"
##########################################################################################################################
#haces la prediction en puntos de iso
iso_meteo_model.iso_predic(preped_dataset,run_iso_whole_year=False,iso_model_month_list=[1,2,3,12]) #run_iso_whole_year=False,iso_model_month_list=[1,2,3,12]
##########################################################################################################################
#iso model
iso_meteo_model.iso_fit()
tools.plots.best_estimator_plots(iso_meteo_model,meteo_plot=False)
tools.plots.partial_dep_plots(iso_meteo_model,meteo_plot=False)
tools.stats.annual_stats(iso_meteo_model)
tools.stats.mensual_stats(iso_meteo_model)
iso_meteo_model_allyear_exc_outliers=tools.session.save(iso_meteo_model,name='iso_meteo_model_final_exc_outliers') #optional prints the directory
prepdataset_final_allyear_exc_outliers=tools.session.save(preped_dataset,name='prepdataset_final_exc_outliers') #optional prints the directory
##########################################################################################################################
#prediction obs points
pred_inputs=iso_meteo_model.all_preds[["CooX","CooY","CooZ","month",'ID_MeteoPoint']]
ev_class=tools.evaluation()
ev_class.predict(iso_meteo_model,pred_inputs,direc=r"C:\Users\usuari\Desktop\10_12_2020\iso_models\12_12_2020\predictions\preds_1_2_3_12_obs")
#cambia esta linea si quieres correr para hacer el mapa grande de prediction:
tools.plots.isotopes_meteoline_plot(ev_class,iso_meteo_model,iso_18,iso_2h,month_data=True,obs_data=False,id_point=True,residplot=True)
#############################################################
#prediction la mapa
#read points for contour activar para predecir mapas
data_file = r"C:\Users\usuari\Desktop\10_12_2020\input\x_y_z_07_12_2020.xls"
pred_inputs=pd.read_excel(data_file,sheet_name=0,header=0,index_col=False,keep_default_na=True)
ev_class_=tools.evaluation()
ev_class_.predict(iso_meteo_model,pred_inputs,direc=r"C:\Users\usuari\Desktop\10_12_2020\iso_models\12_12_2020\predictions\preds_1_2_3_12_map")
#cambia esta linea si quieres correr para hacer el mapa grande de prediction:
tools.plots.isotopes_meteoline_plot(ev_class_,iso_meteo_model,iso_18,iso_2h,month_data=False,obs_data=False,id_point=False,residplot=False)