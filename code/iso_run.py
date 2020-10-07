import pandas as pd
from Isocompy import data_prep,model,tools

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
############################################################
#isotope files
data_file_iso = r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\inputs\pysplit_isotope_registro_Pp_02_05_2020.xlsx"
iso_18 = pd.read_excel(data_file_iso,sheet_name="ISOT18O",header=0,index_col=False,keep_default_na=True)
iso_2h=pd.read_excel(data_file_iso,sheet_name="ISOT2H",header=0,index_col=False,keep_default_na=True)
iso_3h=pd.read_excel(data_file_iso,sheet_name="ISOT3",header=0,index_col=False,keep_default_na=True)
#############################################################

#to run ISOCOMPY for whole year excluding outliers
preped_dataset=data_prep(direc=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\test_output") #your working folder
preped_dataset.fit(rain,temp,hum,iso_18,iso_2h,iso_3h)
iso_meteo_model=model()
iso_meteo_model.meteo_fit(preped_dataset)
tools.save_session(iso_meteo_model) #optional prints the directory
tools.save_session(preped_dataset) #optional prints the directory

iso_meteo_model.iso_predic(preped_dataset,run_iso_whole_year=True,iso_model_month_list=[])
iso_meteo_model.iso_fit()
tools.save_session(iso_meteo_model) #optional prints the directory
tools.save_session(preped_dataset) #optional prints the directory

#to open the saved session to not run the whole thing again
import dill
with open(r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\test_output\isocompy_saved_session_06-Oct-2020-19-11meteoTrue_isoFalse.pkl", 'rb') as in_strm:
    iso_meteo_model=dill.load(in_strm)
with open(r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\test_output\isocompy_saved_session_06-Oct-2020-19-34meteoFalse_isoFalse.pkl", 'rb') as in_strm:
    preped_dataset=dill.load(in_strm)    

#############################################################