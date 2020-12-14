from pathlib import Path
from isocompy.model_fitting import rfmethod, iso_prediction,print_to_file
import os
import numpy as np

#class for isotope and meteorology modeling and prediction
class model(object):
    
    def __init__(self,
    tunedpars_rfr_rain={"min_weight_fraction_leaf":[0,0.02,0.04],"n_estimators":[50,100,150,200,250,300],"criterion": ["mse","mae"],"min_samples_split":[2,5] },
    tunedpars_svr_rain={"kernel":[ "poly", "rbf", "sigmoid"],"C":np.logspace(-1, 1, 3),"gamma":np.logspace(-3, 1, 3) },
    tunedpars_nusvr_rain={"kernel":["linear", "poly", "rbf", "sigmoid"] },
    tunedpars_mlp_rain={"activation" : [ "logistic", "tanh"],"solver" : ["lbfgs", "sgd", "adam"],"alpha":[0.0001,0.0003],"hidden_layer_sizes":[(50,)*2,(50,)*3,(50,)*4,(100,)*2,(100,)*3,(100,)*4],"max_iter":[1000],"n_iter_no_change":[10]},
    which_regs_rain={"muelnet":True,"rfr":True,"mlp":True,"elnet":True,"omp":True,"br":True,"ard":True,"svr":True,"nusvr":False},
    vif_threshold_rain=None,

    tunedpars_rfr_temp={"min_weight_fraction_leaf":[0,0.02,0.04],"n_estimators":[50,100,150,200,250,300],"criterion": ["mse","mae"],"min_samples_split":[2,5] },
    tunedpars_svr_temp={"kernel":[ "poly", "rbf", "sigmoid"],"C":np.logspace(-1, 1, 3),"gamma":np.logspace(-3, 1, 3) },
    tunedpars_nusvr_temp={"kernel":["linear", "poly", "rbf", "sigmoid"] },
    tunedpars_mlp_temp={"activation" : [ "logistic", "tanh"],"solver" : ["lbfgs", "sgd", "adam"],"alpha":[0.0001,0.0003],"hidden_layer_sizes":[(50,)*2,(50,)*3,(50,)*4,(100,)*2,(100,)*3,(100,)*4],"max_iter":[1000],"n_iter_no_change":[10]},
    which_regs_temp={"muelnet":True,"rfr":True,"mlp":True,"elnet":True,"omp":True,"br":True,"ard":True,"svr":True,"nusvr":False},
    vif_threshold_temp=None,

    tunedpars_rfr_hum={"min_weight_fraction_leaf":[0,0.02,0.04],"n_estimators":[50,100,150,200,250,300],"criterion": ["mse","mae"],"min_samples_split":[2,5] },
    tunedpars_svr_hum={"kernel":[ "poly", "rbf", "sigmoid"],"C":np.logspace(-1, 1, 3),"gamma":np.logspace(-3, 1, 3) },
    tunedpars_nusvr_hum={"kernel":["linear", "poly", "rbf", "sigmoid"] },
    tunedpars_mlp_hum={"activation" : [ "logistic", "tanh"],"solver" : ["lbfgs", "sgd", "adam"],"alpha":[0.0001,0.0003],"hidden_layer_sizes":[(50,)*2,(50,)*3,(50,)*4,(100,)*2,(100,)*3,(100,)*4],"max_iter":[1000],"n_iter_no_change":[10]},
    which_regs_hum={"muelnet":True,"rfr":True,"mlp":True,"elnet":True,"omp":True,"br":True,"ard":True,"svr":True,"nusvr":False},
    vif_threshold_hum=None,

    tunedpars_rfr_iso18={"min_weight_fraction_leaf":[0,0.01,0.02,0.03,0.04],"n_estimators":[25,50,75,100,150,200,250,300],"criterion": ["mse","mae"],"min_samples_split":[2,5,7] },
    tunedpars_svr_iso18={"kernel":[ "poly", "rbf", "sigmoid"],"C":np.logspace(-1, 1, 3),"gamma":np.logspace(-3, 1, 3) },
    tunedpars_nusvr_iso18={"kernel":["linear", "poly", "rbf", "sigmoid"] },
    tunedpars_mlp_iso18={"activation" : [ "logistic", "tanh"],"solver" : ["lbfgs", "sgd", "adam"],"alpha":[0.0001,0.0003],"hidden_layer_sizes":[(50,)*2,(50,)*3,(50,)*4,(100,)*2,(100,)*3,(100,)*4],"max_iter":[1000],"n_iter_no_change":[10]},
    which_regs_iso18={"muelnet":True,"rfr":True,"mlp":True,"elnet":True,"omp":True,"br":True,"ard":True,"svr":True,"nusvr":False},
    vif_threshold_iso18=20,

    tunedpars_rfr_iso2h={"min_weight_fraction_leaf":[0,0.01,0.02,0.03,0.04],"n_estimators":[25,50,75,100,150,200,250,300],"criterion": ["mse","mae"],"min_samples_split":[2,5,7] },
    tunedpars_svr_iso2h={"kernel":[ "poly", "rbf", "sigmoid"],"C":np.logspace(-1, 1, 3),"gamma":np.logspace(-3, 1, 3) },
    tunedpars_nusvr_iso2h={"kernel":["linear", "poly", "rbf", "sigmoid"] },
    tunedpars_mlp_iso2h={"activation" : [ "logistic", "tanh"],"solver" : ["lbfgs", "sgd", "adam"],"alpha":[0.0001,0.0003],"hidden_layer_sizes":[(50,)*2,(50,)*3,(50,)*4,(100,)*2,(100,)*3,(100,)*4],"max_iter":[1000],"n_iter_no_change":[10]},
    which_regs_iso2h={"muelnet":True,"rfr":True,"mlp":True,"elnet":True,"omp":True,"br":True,"ard":True,"svr":True,"nusvr":False},
    vif_threshold_iso2h=20,

    tunedpars_rfr_iso3h={"min_weight_fraction_leaf":[0,0.01,0.02,0.03,0.04],"n_estimators":[25,50,75,100,150,200,250,300],"criterion": ["mse","mae"],"min_samples_split":[2,5,7] },
    tunedpars_svr_iso3h={"kernel":[ "poly", "rbf", "sigmoid"],"C":np.logspace(-1, 1, 3),"gamma":np.logspace(-3, 1, 3) },
    tunedpars_nusvr_iso3h={"kernel":["linear", "poly", "rbf", "sigmoid"] },
    tunedpars_mlp_iso3h={"activation" : [ "logistic", "tanh"],"solver" : ["lbfgs", "sgd", "adam"],"alpha":[0.0001,0.0003],"hidden_layer_sizes":[(50,)*2,(50,)*3,(50,)*4,(100,)*2,(100,)*3,(100,)*4],"max_iter":[1000],"n_iter_no_change":[10]},
    which_regs_iso3h={"muelnet":True,"rfr":True,"mlp":True,"elnet":True,"omp":True,"br":True,"ard":True,"svr":True,"nusvr":False},
    vif_threshold_iso3h=20,

    apply_on_log=True,
    cv="auto"):

        self.tunedpars_rfr_rain=tunedpars_rfr_rain
        self.tunedpars_svr_rain=tunedpars_svr_rain
        self.tunedpars_nusvr_rain=tunedpars_nusvr_rain
        self.tunedpars_mlp_rain=tunedpars_mlp_rain
        self.which_regs_rain=which_regs_rain
        self.vif_threshold_rain=vif_threshold_rain
        
        self.tunedpars_rfr_temp=tunedpars_rfr_temp
        self.tunedpars_svr_temp=tunedpars_svr_temp
        self.tunedpars_nusvr_temp=tunedpars_nusvr_temp
        self.tunedpars_mlp_temp=tunedpars_mlp_temp
        self.which_regs_temp=which_regs_temp
        self.vif_threshold_temp=vif_threshold_temp

        self.tunedpars_rfr_hum=tunedpars_rfr_hum
        self.tunedpars_svr_hum=tunedpars_svr_hum
        self.tunedpars_nusvr_hum=tunedpars_nusvr_hum
        self.tunedpars_mlp_hum=tunedpars_mlp_hum
        self.which_regs_hum=which_regs_hum
        self.vif_threshold_hum=vif_threshold_hum

        self.tunedpars_rfr_iso18=tunedpars_rfr_iso18
        self.tunedpars_svr_iso18=tunedpars_svr_iso18
        self.tunedpars_nusvr_iso18=tunedpars_nusvr_iso18
        self.tunedpars_mlp_iso18=tunedpars_mlp_iso18
        self.which_regs_iso18=which_regs_iso18
        self.vif_threshold_iso18=vif_threshold_iso18

        self.tunedpars_rfr_iso2h=tunedpars_rfr_iso2h
        self.tunedpars_svr_iso2h=tunedpars_svr_iso2h
        self.tunedpars_nusvr_iso2h=tunedpars_nusvr_iso2h
        self.tunedpars_mlp_iso2h=tunedpars_mlp_iso2h
        self.which_regs_iso2h=which_regs_iso2h
        self.vif_threshold_iso2h=vif_threshold_iso2h

        
        self.tunedpars_rfr_iso3h=tunedpars_rfr_iso3h
        self.tunedpars_svr_iso3h=tunedpars_svr_iso3h
        self.tunedpars_nusvr_iso3h=tunedpars_nusvr_iso3h
        self.tunedpars_mlp_iso3h=tunedpars_mlp_iso3h
        self.which_regs_iso3h=which_regs_iso3h
        self.vif_threshold_iso3h=vif_threshold_iso3h

        self.apply_on_log=apply_on_log
        self.cv=cv
    ##########################################################################################

    def meteo_fit(self,prepcls,p_val=0.05,temp_fit=True,rain_fit=True,hum_fit=True):
        newmatdframe_rain=prepcls.month_grouped_rain
        newmatdframe_temp=prepcls.month_grouped_temp
        newmatdframe_hum=prepcls.month_grouped_hum
        self.direc=prepcls.direc
        self.temp_fit=temp_fit
        self.rain_fit=rain_fit
        self.hum_fit=hum_fit
        Path(prepcls.direc).mkdir(parents=True, exist_ok=True)
        meteo_or_iso="meteo"
        inputs=None
        best_dics=dict()
        ############################################################
        #RAIN
        if self.rain_fit==True:
            self.rain_bests,self.rain_preds_real_dic,self.rain_bests_dic=rfmethod(self.tunedpars_rfr_rain,self.tunedpars_svr_rain,self.tunedpars_nusvr_rain,self.tunedpars_mlp_rain,newmatdframe_rain,meteo_or_iso,inputs,self.apply_on_log,self.direc,self.cv,self.which_regs_rain,self.vif_threshold_rain,p_val)
            best_dics["PRECIPITATION"]=self.rain_bests
        ###########################################
        #TEMP
        if self.temp_fit==True:
            self.temp_bests,self.temp_preds_real_dic,self.temp_bests_dic=rfmethod(self.tunedpars_rfr_temp,self.tunedpars_svr_temp,self.tunedpars_nusvr_temp,self.tunedpars_mlp_temp,newmatdframe_temp,meteo_or_iso,inputs,self.apply_on_log,self.direc,self.cv,self.which_regs_temp,self.vif_threshold_temp,p_val)
            best_dics["TEMPERATURE"]=self.temp_bests

        ##########################################
        #Humidity
        if self.hum_fit==True:
            self.hum_bests,self.hum_preds_real_dic,self.hum_bests_dic=rfmethod(self.tunedpars_rfr_hum,self.tunedpars_svr_hum,self.tunedpars_nusvr_hum,self.tunedpars_mlp_hum,newmatdframe_hum,meteo_or_iso,inputs,self.apply_on_log,self.direc,self.cv,self.which_regs_hum,self.vif_threshold_hum,p_val)
            best_dics["HUMIDITY"]=self.hum_bests

        #############################################################     
        print_to_file( os.path.join(self.direc,"meteo_model_temp_rain_hum_12_month_stats.txt"),best_dics )

    ##########################################################################################

    def iso_predic(self,cls,run_iso_whole_year,iso_model_month_list,trajectories=False,daily_rain_data_for_trajs=None):
        if trajectories==False: self.col_for_f_reg=[]
        self.direc=cls.direc
        Path(cls.direc).mkdir(parents=True, exist_ok=True)
        self.trajectories=trajectories
        if run_iso_whole_year==True: iso_model_month_list=[m for m in range(1,13)]
        self.iso_model_month_list=iso_model_month_list
        self.run_iso_whole_year=run_iso_whole_year
        self.predictions_monthly_list, self.all_preds,self.all_hysplit_df_list_all_atts,self.col_for_f_reg,self.all_without_averaging=iso_prediction(cls.month_grouped_iso_18,cls.month_grouped_iso_2h,cls.month_grouped_iso_3h,self.temp_bests,self.rain_bests,self.hum_bests,cls.iso_18,daily_rain_data_for_trajs,self.trajectories,self.iso_model_month_list,self.run_iso_whole_year,self.direc)

    ##########################################################################################

    def iso_fit(self,iso18_fit=True,iso3h_fit=True,iso2h_fit=True,output_report=True,p_val=0.05):
        newmatdframe_iso18=self.all_preds
        newmatdframe_iso2h=self.all_preds
        newmatdframe_iso3h=self.all_preds
        self.iso18_fit=iso18_fit
        self.iso3h_fit=iso3h_fit
        self.iso2h_fit=iso2h_fit
        meteo_or_iso="iso"
        inputs=["CooX","CooY","CooZ","temp","rain","hum"]+self.col_for_f_reg
        Path(self.direc).mkdir(parents=True, exist_ok=True)
        ####################################
        #iso18
        if self.iso18_fit==True:
            newmatdframe_iso18=newmatdframe_iso18.rename(columns={"iso_18": "Value"})
            self.iso18_bests,self.iso18_preds_real_dic,self.iso18_bests_dic=rfmethod(self.tunedpars_rfr_iso18,self.tunedpars_svr_iso18,self.tunedpars_nusvr_iso18,self.tunedpars_mlp_iso18,newmatdframe_iso18,meteo_or_iso,inputs,self.apply_on_log,self.direc,self.cv,self.which_regs_iso18,self.vif_threshold_iso18,p_val)
        ####################################
        #iso2h
        if self.iso2h_fit==True:
            newmatdframe_iso2h=newmatdframe_iso2h.rename(columns={"iso_2h": "Value"})
            self.iso2h_bests,self.iso2h_preds_real_dic,self.iso2h_bests_dic=rfmethod(self.tunedpars_rfr_iso2h,self.tunedpars_svr_iso2h,self.tunedpars_nusvr_iso2h,self.tunedpars_mlp_iso2h,newmatdframe_iso2h,meteo_or_iso,inputs,self.apply_on_log,self.direc,self.cv,self.which_regs_iso2h,self.vif_threshold_iso2h,p_val)
        ####################################
        #iso3h
        if self.iso3h_fit==True:
            newmatdframe_iso3h=newmatdframe_iso3h.rename(columns={"iso_3h": "Value"})
            self.iso3h_bests,self.iso3h_preds_real_dic,self.iso3h_bests_dic=rfmethod(self.tunedpars_rfr_iso3h,self.tunedpars_svr_iso3h,self.tunedpars_nusvr_iso3h,self.tunedpars_mlp_iso3h,newmatdframe_iso3h,meteo_or_iso,inputs,self.apply_on_log,self.direc,self.cv,self.which_regs_iso3h,self.vif_threshold_iso3h,p_val)
        ####################################
        if output_report==True:
            #self.rain_bests=best_estimator_all,best_score_all,mutual_info_regression_value,f_regression_value,used_features,rsquared,x_scaler,y_scaler,didlog
            #writing isotope results to a txt file
            from tabulate import tabulate
            if self.iso18_fit==True: pr_is_18="\n################\n\n best_estimator_all_iso18\n"+str(self.iso18_bests[0][0])+"\n\n################\n\n used_features_iso18 \n"+str(self.iso18_bests[0][4])+"\n\n################\n\n best_score_all_iso18 \n"+str(self.iso18_bests[0][1])+"\n\n################\n\n rsquared_iso18 \n"+str(self.iso18_bests[0][5])+"\n\n################\n\n didlog_iso18 \n"+str(self.iso18_bests[0][-1])+"\n\n################\n\n VIF_iso18 \n"+tabulate(self.iso18_bests_dic[0]["vif"])+"\n\n################\n\n vif_chosen_features \n"+str(self.iso18_bests_dic[0]["vif_chosen_features"])+"\n\n################\n\n f_regression_iso18 \n"+tabulate(self.iso18_bests_dic[0]["f_regression"])+"\n\n################\n\n correlation_iso18 \n"+tabulate(self.iso18_bests_dic[0]["correlation"])+"\n\n#########################\n#########################\n#########################\n"
            if self.iso2h_fit==True:pr_is_2h="\n################\n\n best_estimator_all_iso2h\n"+str(self.iso2h_bests[0][0])+"\n\n################\n\n used_features_iso2h \n"+str(self.iso2h_bests[0][4])+"\n\n################\n\n best_score_all_iso2h \n"+str(self.iso2h_bests[0][1])+"\n\n################\n\n rsquared_iso2h \n"+str(self.iso2h_bests[0][5])+"\n\n################\n\n didlog_iso2h \n"+str(self.iso2h_bests[0][-1])+"\n\n################\n\n VIF_iso2h \n"+tabulate(self.iso2h_bests_dic[0]["vif"])+"\n\n################\n\n f_regression \n"+tabulate(self.iso2h_bests_dic[0]["f_regression"],headers='firstrow')+"\n\n################\n\n correlation_iso2h \n"+tabulate(self.iso2h_bests_dic[0]["correlation"])+"\n\n#########################\n#########################\n#########################\n"
            if self.iso3h_fit==True:pr_is_3h="\n################\n\n best_estimator_all_iso3h\n"+str(self.iso3h_bests[0][0])+"\n\n################\n\n used_features_iso3h \n"+str(self.iso3h_bests[0][4])+"\n\n################\n\n best_score_all_iso3h \n"+str(self.iso3h_bests[0][1])+"\n\n################\n\n rsquared_iso3h \n"+str(self.iso3h_bests[0][5])+"\n\n################\n\n didlog_iso3h \n"+str(self.iso3h_bests[0][-1])+"\n\n################\n\n VIF_iso3h \n"+tabulate(self.iso3h_bests_dic[0]["vif"])+"\n\n################\n\n f_regression \n"+tabulate(self.iso3h_bests_dic[0]["f_regression"],headers='firstrow')+"\n\n################\n\n correlation_iso3h \n"+tabulate(self.iso3h_bests_dic[0]["correlation"])+"\n\n#########################\n#########################\n#########################\n"
            file_name=os.path.join(self.direc,"isotope_modeling_output_report_18_2h_3h.txt")
            m_out_f=open(file_name,'w')
            if self.iso18_fit==True:m_out_f.write(pr_is_18)
            if self.iso2h_fit==True:m_out_f.write(pr_is_2h)
            if self.iso3h_fit==True:m_out_f.write(pr_is_3h)
            m_out_f.close()
