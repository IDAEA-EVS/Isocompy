from data_prep import data_preparation
from model_fitting import rfmethod, iso_prediction,f_reg_mutual
import os
import dill
from pathlib import Path
from datetime import datetime

class data_prep(object):

    def __init__(self,direc,meteo_input_type="daily_remove_outliers",write_outliers_input=True,year_type="all",
    inc_zeros=False,write_integrated_data=True,q1=0.05,q3=0.95,IQR_rain=True,IQR_temp=False,IQR_hum=False,IQR_rat_rain=3,IQR_rat_temp=3,IQR_rat_hum=3):

        self.meteo_input_type=meteo_input_type
        self.year_type=year_type
        self.inc_zeros=inc_zeros
        self.IQR_rain=IQR_rain
        self.IQR_temp=IQR_temp
        self.IQR_hum=IQR_hum
        self.write_outliers_input=write_outliers_input
        self.write_integrated_data=write_integrated_data
        self.IQR_rat_rain=IQR_rat_rain
        self.IQR_rat_temp=IQR_rat_temp
        self.IQR_rat_hum=IQR_rat_hum
        self.q1=q1
        self.q3=q3
        self.direc=direc
    
    def fit(self,rain,temp,hum,iso_18,iso_2h,iso_3h,elnino=None,lanina=None):

        if elnino==None: elnino=[]
        if lanina==None: lanina=[]
        self.iso_18=iso_18
        self.iso_2h=iso_2h
        self.iso_3h=iso_3h
        self.month_grouped_iso_18,self.month_grouped_iso_2h,self.month_grouped_iso_3h,self.month_grouped_hum,self.month_grouped_rain,self.month_grouped_temp,self.rain,self.temp,self.hum=data_preparation(rain,temp,hum,iso_18,iso_2h,iso_3h,self.direc,self.meteo_input_type,self.q1,self.q3,self.IQR_rain,self.IQR_temp,self.IQR_hum,self.inc_zeros,self.write_outliers_input,self.write_integrated_data,self.IQR_rat_rain,self.IQR_rat_temp,self.IQR_rat_hum,self.year_type,elnino,lanina)
##########################################################################################
          

#class for isotope and meteorology modeling and prediction
class model(object):
    
    def __init__(self,
    tunedpars_rain={"min_weight_fraction_leaf":[0,0.02,0.04],"n_estimators":[50,100,150,200,250,300],"criterion": ["mse","mae"],"min_samples_split":[2,5] },
    gridsearch_dictionary_rain={"activation" : ["identity", "logistic", "tanh", "relu"],"solver" : ["lbfgs", "sgd", "adam"],"alpha":[0.0001,0.0003,0.0005],"hidden_layer_sizes":[(10,)*2,(25,)*2 ,(50,)*2,(75,)*2,(50,)*3,(50,)*4],"max_iter":[50,100,200],"n_iter_no_change":[5,10,15]},
    tunedpars_temp={"min_weight_fraction_leaf":[0,0.02,0.04],"n_estimators":[50,100,150,200,250,300],"criterion": ["mse","mae"],"min_samples_split":[2,5] },
    gridsearch_dictionary_temp={"activation" : ["identity", "logistic", "tanh", "relu"],"solver" : ["lbfgs", "sgd", "adam"],"alpha":[0.0001,0.0003,0.0005],"hidden_layer_sizes":[(10,)*2,(25,)*2 ,(50,)*2,(75,)*2,(50,)*3,(50,)*4],"max_iter":[50,100,200],"n_iter_no_change":[5,10,15]},
    tunedpars_hum={"min_weight_fraction_leaf":[0,0.02,0.04],"n_estimators":[50,100,150,200,250,300],"criterion": ["mse","mae"],"min_samples_split":[2,5] },
    gridsearch_dictionary_hum={"activation" : ["identity", "logistic", "tanh", "relu"],"solver" : ["lbfgs", "sgd", "adam"],"alpha":[0.0001,0.0003,0.0005],"hidden_layer_sizes":[(10,)*2,(25,)*2 ,(50,)*2,(75,)*2,(50,)*3,(50,)*4],"max_iter":[50,100,200],"n_iter_no_change":[5,10,15]},
    tunedpars_iso18={"min_weight_fraction_leaf":[0,0.01,0.02,0.03,0.04],"n_estimators":[25,50,75,100,150,200,250,300],"criterion": ["mse","mae"],"min_samples_split":[2,5,7] },
    gridsearch_dictionary_iso18={"activation" : ["identity", "logistic", "tanh", "relu"],"solver" : ["lbfgs", "sgd", "adam"],"alpha":[0.0001,0.0003,0.0005,0.001,0.005],"hidden_layer_sizes":[(10,)*2,(25,)*2 ,(50,)*2,(75,)*2,(50,)*3,(50,)*4,(100,)*2,(100,)*3,(100,)*4],"max_iter":[25,50,100,150,200,300],"n_iter_no_change":[5,10,15,20]},
    tunedpars_iso2h={"min_weight_fraction_leaf":[0,0.01,0.02,0.03,0.04],"n_estimators":[25,50,75,100,150,200,250,300],"criterion": ["mse","mae"],"min_samples_split":[2,5,7] },
    gridsearch_dictionary_iso2h={"activation" : ["identity", "logistic", "tanh", "relu"],"solver" : ["lbfgs", "sgd", "adam"],"alpha":[0.0001,0.0003,0.0005,0.001,0.005],"hidden_layer_sizes":[(10,)*2,(25,)*2 ,(50,)*2,(75,)*2,(50,)*3,(50,)*4,(100,)*2,(100,)*3,(100,)*4],"max_iter":[25,50,100,150,200,300],"n_iter_no_change":[5,10,15,20]},
    tunedpars_iso3h={"min_weight_fraction_leaf":[0,0.01,0.02,0.03,0.04],"n_estimators":[25,50,75,100,150,200,250,300],"criterion": ["mse","mae"],"min_samples_split":[2,5,7] },
    gridsearch_dictionary_iso3h={"activation" : ["identity", "logistic", "tanh", "relu"],"solver" : ["lbfgs", "sgd", "adam"],"alpha":[0.0001,0.0003,0.0005,0.001,0.005],"hidden_layer_sizes":[(10,)*2,(25,)*2 ,(50,)*2,(75,)*2,(50,)*3,(50,)*4,(100,)*2,(100,)*3,(100,)*4],"max_iter":[25,50,100,150,200,300],"n_iter_no_change":[5,10,15,20]},
    apply_on_log=True):

        self.tunedpars_rain=tunedpars_rain
        self.gridsearch_dictionary_rain=gridsearch_dictionary_rain

        self.tunedpars_temp=tunedpars_temp
        self.gridsearch_dictionary_temp=gridsearch_dictionary_temp

        self.tunedpars_hum=tunedpars_hum
        self.gridsearch_dictionary_hum=gridsearch_dictionary_hum

        self.tunedpars_iso18=tunedpars_iso18
        self.gridsearch_dictionary_iso18=gridsearch_dictionary_iso18

        self.tunedpars_iso2h=tunedpars_iso2h
        self.gridsearch_dictionary_iso2h=gridsearch_dictionary_iso2h

        self.tunedpars_iso3h=tunedpars_iso3h
        self.gridsearch_dictionary_iso3h=gridsearch_dictionary_iso3h
        
        self.apply_on_log=apply_on_log
    ##########################################################################################

    def meteo_fit(self,prepcls):

        newmatdframe_rain=prepcls.month_grouped_rain
        newmatdframe_temp=prepcls.month_grouped_temp
        newmatdframe_hum=prepcls.month_grouped_hum
        self.direc=prepcls.direc
        meteo_or_iso="meteo"
        inputs=None
        ############################################################
        #RAIN
        temp_rain_hum="Mean_Value_rain"
        model_type="rain"
        self.rain_bests,self.rain_preds_real_dic,self.rain_bests_dic=rfmethod(self.tunedpars_rain,self.gridsearch_dictionary_rain,newmatdframe_rain,temp_rain_hum,model_type,meteo_or_iso,inputs,self.apply_on_log,self.direc)
        #self.rain_bests=best_estimator_all,best_score_all,mutual_info_regression_value,f_regression_value,used_features,rsquared,x_scaler,y_scaler,didlog
        #self.rain_preds_real=Y_preds,Y_temp_fin,X_temp
        ###########################################
        #TEMP
        temp_rain_hum="Mean_Value_temp"
        model_type="temp"
        self.temp_bests,self.temp_preds_real_dic,self.temp_bests_dic=rfmethod(self.tunedpars_temp,self.gridsearch_dictionary_temp,newmatdframe_temp,temp_rain_hum,model_type,meteo_or_iso,inputs,self.apply_on_log,self.direc)
        ##########################################
        #Humidity
        temp_rain_hum="Mean_Value_hum"
        model_type="humid"
        self.hum_bests,self.hum_preds_real_dic,self.hum_bests_dic=rfmethod(self.tunedpars_hum,self.gridsearch_dictionary_hum,newmatdframe_hum,temp_rain_hum,model_type,meteo_or_iso,inputs,self.apply_on_log,self.direc)
        #############################################################     

    ##########################################################################################

    def iso_predic(self,cls,run_iso_whole_year,iso_model_month_list,trajectories=False,daily_rain_data_for_trajs=None,):
        if trajectories==False: self.col_for_f_reg=[]
        self.trajectories=trajectories
        self.iso_model_month_list=iso_model_month_list
        self.run_iso_whole_year=run_iso_whole_year
        self.predictions_monthly_list, self.all_preds,self.all_hysplit_df_list_all_atts,self.col_for_f_reg,self.all_without_averaging=iso_prediction(cls.month_grouped_iso_18,cls.month_grouped_iso_2h,cls.month_grouped_iso_3h,self.temp_bests,self.rain_bests,self.hum_bests,cls.iso_18,daily_rain_data_for_trajs,self.trajectories,self.iso_model_month_list,self.run_iso_whole_year,self.direc)

    ##########################################################################################

    def iso_fit(self,output_report=True):
        newmatdframe_iso18=self.all_preds
        newmatdframe_iso2h=self.all_preds
        newmatdframe_iso3h=self.all_preds
        meteo_or_iso="iso"
        inputs=["CooX","CooY","CooZ","temp","rain","hum"]+self.col_for_f_reg

        ####################################
        temp_rain_hum="iso_18"
        model_type="iso_18"
        self.iso18_bests,self.iso18_preds_real_dic,self.iso18_bests_dic=rfmethod(self.tunedpars_iso18,self.gridsearch_dictionary_iso18,newmatdframe_iso18,temp_rain_hum,model_type,meteo_or_iso,inputs,self.apply_on_log,self.direc)
        ####################################
        temp_rain_hum="iso_2h"
        model_type="iso_2h"
        self.iso2h_bests,self.iso2h_preds_real_dic,self.iso2h_bests_dic=rfmethod(self.tunedpars_iso2h,self.gridsearch_dictionary_iso2h,newmatdframe_iso2h,temp_rain_hum,model_type,meteo_or_iso,inputs,self.apply_on_log,self.direc)
        ####################################
        ####################################
        temp_rain_hum="iso_3h"
        model_type="iso_3h"
        self.iso3h_bests,self.iso3h_preds_real_dic,self.iso3h_bests_dic=rfmethod(self.tunedpars_iso3h,self.gridsearch_dictionary_iso3h,newmatdframe_iso3h,temp_rain_hum,model_type,meteo_or_iso,inputs,self.apply_on_log,self.direc)
        ####################################
        if output_report==True:
            #self.rain_bests=best_estimator_all,best_score_all,mutual_info_regression_value,f_regression_value,used_features,rsquared,x_scaler,y_scaler,didlog
            #writing isotope results to a txt file
            pr_is_18="\n################\n\n best_estimator_all_iso18\n"+str(self.iso18_bests[0][0])+"\n\n################\n\n used_features_iso18 \n"+str(self.iso18_bests[0][4])+"\n\n################\n\n best_score_all_iso18 \n"+str(self.iso18_bests[0][1])+"\n\n################\n\n rsquared_iso18 \n"+str(self.iso18_bests[0][5])+"\n\n################\n\n didlog_iso18 \n"+str(self.iso18_bests[0][-1])+"\n\n#########################\n#########################\n#########################\n"
            pr_is_2h="\n################\n\n best_estimator_all_iso2h\n"+str(self.iso2h_bests[0][0])+"\n\n################\n\n used_features_iso2h \n"+str(self.iso2h_bests[0][4])+"\n\n################\n\n best_score_all_iso2h \n"+str(self.iso2h_bests[0][1])+"\n\n################\n\n rsquared_iso2h \n"+str(self.iso2h_bests[0][5])+"\n\n################\n\n rsquared_iso2h \n"+str(self.iso2h_bests[0][-1])+"\n\n#########################\n#########################\n#########################\n"
            pr_is_3h="\n################\n\n best_estimator_all_iso3h\n"+str(self.iso3h_bests[0][0])+"\n\n################\n\n used_features_iso3h \n"+str(self.iso3h_bests[0][4])+"\n\n################\n\n best_score_all_iso3h \n"+str(self.iso3h_bests[0][1])+"\n\n################\n\n rsquared_iso3h \n"+str(self.iso3h_bests[0][5])+"\n\n################\n\n didlog_iso3h \n"+str(self.iso3h_bests[0][-1])+"\n\n#########################\n#########################\n#########################\n"
            file_name=os.path.join(self.direc,"isotope_modeling_output_report_18_2h_3h.txt")
            m_out_f=open(file_name,'w')
            m_out_f.write(pr_is_18)
            m_out_f.write(pr_is_2h)
            m_out_f.write(pr_is_3h)
            m_out_f.close()
    ###########
    @staticmethod
    def annual_stats(model_cls_obj):
        list_of_dics_for_stats=[
            {"inputs":["CooX","CooY","CooZ"],"outputs":["temp"]},
            {"inputs":["CooX","CooY","CooZ"],"outputs":["hum"]},
            {"inputs":["CooX","CooY","CooZ"],"outputs":["rain"]},
            {"inputs":["CooX","CooY","CooZ","temp","rain","hum"]+model_cls_obj.col_for_f_reg,"outputs":["iso_2h"]},
            {"inputs":["CooX","CooY","CooZ","temp","rain","hum"]+model_cls_obj.col_for_f_reg,"outputs":["iso_18"]},
            {"inputs":["CooX","CooY","CooZ","temp","rain","hum"]+model_cls_obj.col_for_f_reg,"outputs":["iso_3h"]},
        ]

        f_reg_mutual( os.path.join(model_cls_obj.direc,"annual_statistics_f_test_MI.txt") ,model_cls_obj.all_preds,list_of_dics_for_stats)   
    
    @staticmethod
    def mensual_stats(model_cls_obj):
        list_of_dics_for_stats=[
            {"inputs":["CooX","CooY","CooZ"],"outputs":["temp"]},
            {"inputs":["CooX","CooY","CooZ"],"outputs":["hum"]},
            {"inputs":["CooX","CooY","CooZ"],"outputs":["rain"]},
            {"inputs":["CooX","CooY","CooZ","temp","rain","hum"]+model_cls_obj.col_for_f_reg,"outputs":["iso_2h"]},
            {"inputs":["CooX","CooY","CooZ","temp","rain","hum"]+model_cls_obj.col_for_f_reg,"outputs":["iso_18"]},
            {"inputs":["CooX","CooY","CooZ","temp","rain","hum"]+model_cls_obj.col_for_f_reg,"outputs":["iso_3h"]},
        ]
        all_preds_month=model_cls_obj.all_preds["month"].copy()
        all_preds_month.drop_duplicates(keep = 'last', inplace = True)
        for mn in all_preds_month:
            all_preds_temp=model_cls_obj.all_preds[model_cls_obj.all_preds["month"]==mn]
            Path(os.path.join(model_cls_obj.direc,"mensual_statistics")).mkdir(parents=True, exist_ok=True)
            file_name= os.path.join(model_cls_obj.direc,"mensual_statistics","month_"+str(mn)+"mensual_statistics_f_test_MI.txt")
            f_reg_mutual(file_name,all_preds_temp,list_of_dics_for_stats)


class tools(object):

    @staticmethod
    def save_session(model_cls_obj,name="isocompy_saved_session_"):
        dateTimeObj = datetime.now().strftime("%d_%b_%Y_%H_%M")
        try:
            model_cls_obj.hum_preds_real

            try:
                model_cls_obj.iso2h_bests
                filename=os.path.join(model_cls_obj.direc,name+dateTimeObj+"_meteoTrue_isoTrue.pkl")

            except:    
                filename=os.path.join(model_cls_obj.direc,name+dateTimeObj+"_meteoTrue_isoFalse.pkl")
        
        except: 

            filename=os.path.join(model_cls_obj.direc,name+dateTimeObj+"_meteoFalse_isoFalse.pkl")
        with open(filename, 'wb') as filee:
            dill.dump(obj=model_cls_obj,file=filee)
        print ("\n\n pkl saved session directory: \n\n",filename)
    ##########################################################################################
    def predict():
        #############################################################
        #read points for contour
        data_file = r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\inputs\x_y_z.xls"
        x_y_z_=pd.read_excel(data_file,sheet_name=0,header=0,index_col=False,keep_default_na=True)
        #here and iso_prediction, identifying the month should be nicer!
        column_name="predicted_iso18"
        monthly_iso18_output=predict_points(used_features_iso18,x_y_z_,iso_model_month_list,temp_bests,rain_bests,hum_bests,x_scaler_iso18,y_scaler_iso18,didlog_iso18,best_estimator_all_iso18,column_name,trajectory_features_list=col_for_f_reg)
        column_name="predicted_iso2h"
        monthly_iso2h_output=predict_points(used_features_iso2h,x_y_z_,iso_model_month_list,temp_bests,rain_bests,hum_bests,x_scaler_iso2h,y_scaler_iso2h,didlog_iso2h,best_estimator_all_iso2h,column_name,trajectory_features_list=col_for_f_reg)
    ############################################################
    def regional_mensual_plot():
        #regional_mensual_plot
        #regional_mensual_plot(x_y_z_,monthly_iso18_output,monthly_iso2h_output)
        ############################################################
    def new_inp_compare():   
        #new_data_prediction_comparison
        data_fl=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\inputs\new_measured.xlsx"
        newd=pd.read_excel(data_fl,sheet_name=0,header=0,index_col=False,keep_default_na=True)
        new_data_prediction=new_data_prediction_comparison(newd,iso_model_month_list,temp_bests,rain_bests,hum_bests,didlog_iso18,didlog_iso2h,x_scaler_iso18,x_scaler_iso2h,used_features_iso18,used_features_iso2h,best_estimator_all_iso18,best_estimator_all_iso2h,y_scaler_iso18,y_scaler_iso2h)
        #time
        t_total_end=time.time()
        print ("#################################\n######Total run time:\n", t_total_end-t_total_start)
        #writing isotope predictions into a file
        pd.concat([all_preds,Y_preds_iso18,Y_preds_iso2h],axis=1).to_excel(r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\output\isotope_main_data_predictions.xlsx")
        #return Y_preds_iso18,Y_preds_iso2h,rain_preds_real,hum_preds_real,temp_preds_real,temp_bests,rain_bests,hum_bests,monthly_iso2h_output,monthly_iso18_output,x_scaler_iso18,y_scaler_iso18,didlog_iso18,used_features_iso18,rsquared_iso18,best_estimator_all_iso18,best_score_all_iso18,x_scaler_iso2h,y_scaler_iso2h,didlog_iso2h,used_features_iso2h,rsquared_iso2h,best_estimator_all_iso2h,best_score_all_iso2h,predictions_monthly_list, all_preds,month_grouped_list_with_zeros_iso_18,month_grouped_list_with_zeros_iso_2h,month_grouped_list_with_zeros_iso_3h,month_grouped_list_with_zeros_hum,month_grouped_list_with_zeros_rain,month_grouped_list_with_zeros_temp,rain,temp,hum,elnino,lanina,iso_18,iso_2h,iso_3h
