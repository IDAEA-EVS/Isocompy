import os
import dill
from pathlib import Path
from datetime import datetime
from isocompy.isocompy_tool_funcs import predict_points,new_data_prediction_comparison,regional_mensual_plot,f_reg_mutual,best_estimator_and_part_plots
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import numpy as np
from pathlib import Path

import itertools
class session(object):

    @staticmethod
    def save(self,name="isocompy_saved_session_"):
        dateTimeObj = datetime.now().strftime("%d_%b_%Y_%H_%M")
        try:
            self.hum_preds_real_dic

            try:
                self.iso2h_bests
                filename=os.path.join(self.direc,name+dateTimeObj+"_meteoTrue_isoTrue.pkl")

            except:    
                filename=os.path.join(self.direc,name+dateTimeObj+"_meteoTrue_isoFalse.pkl")
        
        except: 

            filename=os.path.join(self.direc,name+dateTimeObj+"_meteoFalse_isoFalse.pkl")
        with open(filename, 'wb') as filee:
            dill.dump(obj=self,file=filee)
        print ("\n\n pkl saved session directory: \n\n",filename)
        return filename
    ##########################################################################################
    @staticmethod
    def load(direc):
        with open(direc, 'rb') as in_strm:
            obj=dill.load(in_strm)
        return obj
        
class evaluation(object):
    
    def predict(self,cls,pred_inputs,dir=None):
        if dir==None: self.dir=cls.direc
        else: self.dir=dir
        #############################################################
        self.pred_inputs=pred_inputs
        #here and iso_prediction, identifying the month should be nicer!
        column_name="predicted_iso18"
        self.monthly_iso18_output=predict_points(self.dir,cls.iso18_bests_dic[0]["used_features"],self.pred_inputs,cls.iso_model_month_list,cls.temp_bests,cls.rain_bests,cls.hum_bests,cls.iso18_bests[0][6],cls.iso18_bests[0][7],cls.iso18_bests_dic[0]["didlog"],cls.iso18_bests_dic[0]["best_estimator"],column_name,trajectory_features_list=cls.col_for_f_reg,run_iso_whole_year=cls.run_iso_whole_year)
        column_name="predicted_iso2h"
        self.monthly_iso2h_output=predict_points(self.dir,cls.iso2h_bests_dic[0]["used_features"],self.pred_inputs,cls.iso_model_month_list,cls.temp_bests,cls.rain_bests,cls.hum_bests,cls.iso2h_bests[0][6],cls.iso2h_bests[0][7],cls.iso2h_bests_dic[0]["didlog"],cls.iso2h_bests_dic[0]["best_estimator"],column_name,trajectory_features_list=cls.col_for_f_reg,run_iso_whole_year=cls.run_iso_whole_year)
        column_name="predicted_iso3h"
        self.monthly_iso3h_output=predict_points(self.dir,cls.iso3h_bests_dic[0]["used_features"],self.pred_inputs,cls.iso_model_month_list,cls.temp_bests,cls.rain_bests,cls.hum_bests,cls.iso3h_bests[0][6],cls.iso3h_bests[0][7],cls.iso3h_bests_dic[0]["didlog"],cls.iso3h_bests_dic[0]["best_estimator"],column_name,trajectory_features_list=cls.col_for_f_reg,run_iso_whole_year=cls.run_iso_whole_year)
    ############################################################
    def regional_mensual_plot():
        pass
        #regional_mensual_plot
        #regional_mensual_plot(x_y_z_,monthly_iso18_output,monthly_iso2h_output)
    ############################################################

    def new_inp_compare():   

        #new_data_prediction_comparison
        data_fl=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\inputs\new_measured.xlsx"
        newd=pd.read_excel(data_fl,sheet_name=0,header=0,index_col=False,keep_default_na=True)
        new_data_prediction=new_data_prediction_comparison(newd,iso_model_month_list,temp_bests,rain_bests,hum_bests,didlog_iso18,didlog_iso2h,x_scaler_iso18,x_scaler_iso2h,used_features_iso18,used_features_iso2h,best_estimator_all_iso18,best_estimator_all_iso2h,y_scaler_iso18,y_scaler_iso2h)
        #writing isotope predictions into a file
        pd.concat([all_preds,Y_preds_iso18,Y_preds_iso2h],axis=1).to_excel(r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\output\isotope_main_data_predictions.xlsx")
        #return Y_preds_iso18,Y_preds_iso2h,rain_preds_real,hum_preds_real,temp_preds_real,temp_bests,rain_bests,hum_bests,monthly_iso2h_output,monthly_iso18_output,x_scaler_iso18,y_scaler_iso18,didlog_iso18,used_features_iso18,rsquared_iso18,best_estimator_all_iso18,best_score_all_iso18,x_scaler_iso2h,y_scaler_iso2h,didlog_iso2h,used_features_iso2h,rsquared_iso2h,best_estimator_all_iso2h,best_score_all_iso2h,predictions_monthly_list, all_preds,month_grouped_list_with_zeros_iso_18,month_grouped_list_with_zeros_iso_2h,month_grouped_list_with_zeros_iso_3h,month_grouped_list_with_zeros_hum,month_grouped_list_with_zeros_rain,month_grouped_list_with_zeros_temp,rain,temp,hum,elnino,lanina,iso_18,iso_2h,iso_3h
        ###########


class  stats(object):   
    @staticmethod
    def annual_stats(model_cls_obj,params="all"): #params=all, meteo, iso
        list_of_dics_meteo=[
            {"inputs":["CooX","CooY","CooZ"],"outputs":["temp"]},
            {"inputs":["CooX","CooY","CooZ"],"outputs":["hum"]},
            {"inputs":["CooX","CooY","CooZ"],"outputs":["rain"]}]
        list_of_dics_iso=[
            {"inputs":["CooX","CooY","CooZ","temp","rain","hum"]+model_cls_obj.col_for_f_reg,"outputs":["iso_2h"]},
            {"inputs":["CooX","CooY","CooZ","temp","rain","hum"]+model_cls_obj.col_for_f_reg,"outputs":["iso_18"]},
            {"inputs":["CooX","CooY","CooZ","temp","rain","hum"]+model_cls_obj.col_for_f_reg,"outputs":["iso_3h"]}]
        
        if params=="all": list_of_dics_for_stats=list_of_dics_meteo+list_of_dics_iso
        elif params=="meteo": list_of_dics_for_stats=list_of_dics_meteo
        elif params=="iso": list_of_dics_for_stats=list_of_dics_iso
        
        f_reg_mutual( os.path.join(model_cls_obj.direc,"annual_statistics_f_test_MI.txt") ,model_cls_obj.all_preds,list_of_dics_for_stats)   
    
    @staticmethod
    def mensual_stats(model_cls_obj,params="all"): #params=all, meteo, iso
        list_of_dics_meteo=[
            {"inputs":["CooX","CooY","CooZ"],"outputs":["temp"]},
            {"inputs":["CooX","CooY","CooZ"],"outputs":["hum"]},
            {"inputs":["CooX","CooY","CooZ"],"outputs":["rain"]}]
        list_of_dics_iso=[
            {"inputs":["CooX","CooY","CooZ","temp","rain","hum"]+model_cls_obj.col_for_f_reg,"outputs":["iso_2h"]},
            {"inputs":["CooX","CooY","CooZ","temp","rain","hum"]+model_cls_obj.col_for_f_reg,"outputs":["iso_18"]},
            {"inputs":["CooX","CooY","CooZ","temp","rain","hum"]+model_cls_obj.col_for_f_reg,"outputs":["iso_3h"]}]
        
        if params=="all": list_of_dics_for_stats=list_of_dics_meteo+list_of_dics_iso
        elif params=="meteo": list_of_dics_for_stats=list_of_dics_meteo
        elif params=="iso": list_of_dics_for_stats=list_of_dics_iso

        all_preds_month=model_cls_obj.all_preds["month"].copy()
        all_preds_month.drop_duplicates(keep = 'last', inplace = True)
        for mn in all_preds_month:
            all_preds_temp=model_cls_obj.all_preds[model_cls_obj.all_preds["month"]==mn]
            Path(os.path.join(model_cls_obj.direc,"mensual_statistics")).mkdir(parents=True, exist_ok=True)
            file_name= os.path.join(model_cls_obj.direc,"mensual_statistics","month_"+str(mn)+"mensual_statistics_f_test_MI.txt")
            f_reg_mutual(file_name,all_preds_temp,list_of_dics_for_stats)

class plots(object):
    @staticmethod
    def best_estimator_plots(cls,meteo_plot=True,iso_plot=True):
        estimator_plot=True
        partial_dep_plot=False
        best_estimator_and_part_plots(cls,meteo_plot,iso_plot,estimator_plot,partial_dep_plot)
    @staticmethod
    def partial_dep_plots(cls,meteo_plot=True,iso_plot=True):
        estimator_plot=False
        partial_dep_plot=True
        best_estimator_and_part_plots(cls,meteo_plot,iso_plot,estimator_plot,partial_dep_plot)
    
    @staticmethod
    def isotopes_meteoline_plot(ev_class,iso_class):
        """
        docstring
        """
        Path(os.path.join(iso_class.direc,"isotopes_meteoline_plots")).mkdir(parents=True,exist_ok=True)
        for num ,(i,j) in enumerate(zip(ev_class.monthly_iso18_output,ev_class.monthly_iso2h_output)):
            plt.scatter(i,j)
            left, right = plt.xlim()
            left1, right1 = plt.ylim()
            a = np.linspace(min(left,left1),max(right,right1),100)
            b=8*a+10
            plt.plot( a , b )
            y_true = 8 * i + 10
            y_pred = j
            
            plt.title("Month: " + str(num+1)+ "__  R2_org: " +str(iso_class.iso18_bests_dic[0]["best_score"])+"__  R2_preds: " +str(r2_score(y_true, y_pred)))
            plt.xlabel("iso_18")
            plt.ylabel("iso_2H")
            plt.savefig(os.path.join(iso_class.direc,"isotopes_meteoline_plots","Month_" + str(num+1)+"_plot.pdf"),dpi=300)
            plt.show()
