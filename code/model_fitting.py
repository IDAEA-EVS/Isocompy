import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.model_selection import GridSearchCV
from sklearn.exceptions import DataConversionWarning
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import MultiTaskElasticNetCV,LarsCV, OrthogonalMatchingPursuitCV,BayesianRidge,ARDRegression
from sklearn.inspection import plot_partial_dependence
from sklearn.svm import SVR,NuSVR
import os
from pathlib import Path
import dill
from datetime import datetime

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
##########################################################################################
          

#class for isotope and meteorology modeling and prediction
'''class model(object):
    
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
        newmatdframe_temp=prepcls.newmatdframe_temp
        newmatdframe_hum=prepcls.newmatdframe_hum
        self.direc=prepcls.direc
        meteo_or_iso="meteo"
        inputs=None
        ############################################################
        #RAIN
        temp_rain_hum="Mean_Value_rain"
        model_type="rain"
        self.rain_bests,self.rain_preds_real=rfmethod(self.tunedpars_rain,self.gridsearch_dictionary_rain,newmatdframe_rain,temp_rain_hum,model_type,meteo_or_iso,inputs,self.apply_on_log,self.direc)
        #self.rain_bests=best_estimator_all,best_score_all,mutual_info_regression_value,f_regression_value,used_features,rsquared,x_scaler,y_scaler,didlog
        #self.rain_preds_real=Y_preds,Y_temp_fin,X_temp
        ###########################################
        #TEMP
        temp_rain_hum="Mean_Value_temp"
        model_type="temp"
        self.temp_bests,self.temp_preds_real=rfmethod(self.tunedpars_temp,self.gridsearch_dictionary_temp,newmatdframe_temp,temp_rain_hum,model_type,meteo_or_iso,inputs,self.apply_on_log,self.direc)
        ##########################################
        #Humidity
        temp_rain_hum="Mean_Value_hum"
        model_type="humid"
        self.hum_bests,self.hum_preds_real=rfmethod(self.tunedpars_hum,self.gridsearch_dictionary_hum,newmatdframe_hum,temp_rain_hum,model_type,meteo_or_iso,inputs,self.apply_on_log,self.direc)
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
        self.iso18_bests,self.iso18_preds_real=rfmethod(self.tunedpars_iso18,self.gridsearch_dictionary_iso18,newmatdframe_iso18,temp_rain_hum,model_type,meteo_or_iso,inputs,self.apply_on_log,self.direc)
        ####################################
        temp_rain_hum="iso_2h"
        model_type="iso_2h"
        self.iso2h_bests,self.iso2h_preds_real=rfmethod(self.tunedpars_iso2h,self.gridsearch_dictionary_iso2h,newmatdframe_iso2h,temp_rain_hum,model_type,meteo_or_iso,inputs,self.apply_on_log,self.direc)
        ####################################
        ####################################
        temp_rain_hum="iso_3h"
        model_type="iso_3h"
        self.iso3h_bests,self.iso3h_preds_real=rfmethod(self.tunedpars_iso3h,self.gridsearch_dictionary_iso3h,newmatdframe_iso3h,temp_rain_hum,model_type,meteo_or_iso,inputs,self.apply_on_log,self.direc)
        ####################################
        if output_report==True:
            #self.rain_bests=best_estimator_all,best_score_all,mutual_info_regression_value,f_regression_value,used_features,rsquared,x_scaler,y_scaler,didlog
            #writing isotope results to a txt file
            pr_is_18="\n################\n\n best_estimator_all_iso18\n"+str(self.iso18_bests[0])+"\n\n################\n\n used_features_iso18 \n"+str(self.iso18_bests[4])+"\n\n################\n\n best_score_all_iso18 \n"+str(self.iso18_bests[1])+"\n\n################\n\n rsquared_iso18 \n"+str(self.iso18_bests[5])+"\n\n################\n\n didlog_iso18 \n"+str(self.iso18_bests[-1])+"\n\n#########################\n#########################\n#########################\n"
            pr_is_2h="\n################\n\n best_estimator_all_iso2h\n"+str(self.iso2h_bests[0])+"\n\n################\n\n used_features_iso2h \n"+str(self.iso2h_bests[4])+"\n\n################\n\n best_score_all_iso2h \n"+str(self.iso2h_bests[1])+"\n\n################\n\n rsquared_iso2h \n"+str(self.iso2h_bests[5])+"\n\n################\n\n rsquared_iso2h \n"+str(self.iso2h_bests[-1])+"\n\n#########################\n#########################\n#########################\n"
            pr_is_3h="\n################\n\n best_estimator_all_iso3h\n"+str(self.iso3h_bests[0])+"\n\n################\n\n used_features_iso3h \n"+str(self.iso3h_bests[4])+"\n\n################\n\n best_score_all_iso3h \n"+str(self.iso3h_bests[1])+"\n\n################\n\n rsquared_iso3h \n"+str(self.iso3h_bests[5])+"\n\n################\n\n didlog_iso3h \n"+str(self.iso3h_bests[-1])+"\n\n#########################\n#########################\n#########################\n"
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
            file_name= os.path.join(model_cls_obj.direc,"monthly_f_test","month_"+str(mn)+"mensual_statistics_f_test_MI.txt")
            f_reg_mutual(file_name,all_preds_temp,list_of_dics_for_stats)

    @staticmethod
    def save_session(model_cls_obj,name="isocompy_saved_session_"):
        dateTimeObj = datetime.now().strftime("%d-%b-%Y-%H-%M")
        try:
            model_cls_obj.hum_preds_real

            try:
                model_cls_obj.iso2h_bests
                dill.dump(model_cls_obj,os.path.join(model_cls_obj.direc,name+dateTimeObj+"meteoTrue_isoTrue.pkl"))

            except:    
                dill.dump(model_cls_obj,os.path.join(model_cls_obj.direc,name+dateTimeObj+"meteoTrue_isoFalse.pkl"))
        
        except: 

            dill.dump(model_cls_obj,os.path.join(model_cls_obj.direc,name+dateTimeObj+"meteoFalse_isoFalse.pkl"))'''
##########################################################################################
##########################################################################################
#model!
def rfmethod(tunedpars,gridsearch_dictionary,newmatdframe,temp_rain_hum,model_type,meteo_or_iso,inputs,apply_on_log,direc):
    ########################################################################

    #regression functions
    def regressionfunctions(X_temp,Y_temp):
        #OrthogonalMatchingPursuitCV
        reg =OrthogonalMatchingPursuitCV(cv=10,n_jobs=-1).fit(X_temp, Y_temp)
        print ("OMP")
        print ("SCORE:\n",adj_r_sqrd(reg.score(X_temp, Y_temp)) )
        best_score_all=adj_r_sqrd(reg.score(X_temp, Y_temp))
        best_estimator_all=reg
        rsquared=reg.score(X_temp, Y_temp)
        '''########################################################################    
        #RandomForestRegressor
        estrandomfor_temp=GridSearchCV(RandomForestRegressor(random_state =0), tunedpars, cv=10,n_jobs=-1)
        estrandomfor_temp.fit(X_temp, Y_temp.ravel())
        if adj_r_sqrd(estrandomfor_temp.best_score_)>best_score_all:
            best_score_all=adj_r_sqrd(estrandomfor_temp.best_score_)
            best_estimator_all=estrandomfor_temp.best_estimator_
            rsquared=estrandomfor_temp.best_score_
        ########################################################################
        #NEURAL NETWORK
        mlp_temp=GridSearchCV( MLPRegressor(learning_rate="adaptive"), gridsearch_dictionary, cv=5,n_jobs=-1) 
        mlp_temp.fit(X_temp, Y_temp.ravel())
        if adj_r_sqrd(mlp_temp.best_score_)>best_score_all:
            best_score_all=adj_r_sqrd(mlp_temp.best_score_)
            best_estimator_all=mlp_temp.best_estimator_
            rsquared=mlp_temp.best_score_
        ########################################################################
        #(MULTI TASK) ELASTIC NET CV
        #elastic net on standardized data
        reg_n = MultiTaskElasticNetCV(l1_ratio=[.1, .3, .5, .6, .7,.8,.85,.87,.9,.93,.95,.97,0.99],n_jobs =-1,cv=10,normalize=False ).fit(X_temp, Y_temp)
        if adj_r_sqrd(reg_n.score(X_temp, Y_temp))>best_score_all:
            best_score_all=adj_r_sqrd(reg_n.score(X_temp, Y_temp))
            best_estimator_all=reg_n
            rsquared=reg_n.score(X_temp, Y_temp)
        #################################################################################
        #OrthogonalMatchingPursuitCV
        reg =OrthogonalMatchingPursuitCV(cv=10,n_jobs=-1).fit(X_temp, Y_temp)
        print ("OMP")
        print ("SCORE:\n",adj_r_sqrd(reg.score(X_temp, Y_temp)) )
        if adj_r_sqrd(reg.score(X_temp, Y_temp))>best_score_all:
            best_score_all=adj_r_sqrd(reg.score(X_temp, Y_temp))
            best_estimator_all=reg
            rsquared=reg.score(X_temp, Y_temp)
        ####################################################################
        #BayesianRidge
        reg =BayesianRidge(normalize=True).fit(X_temp, Y_temp)
        #cross validation
        scores = cross_val_score(reg, X_temp, Y_temp.ravel(), cv=10,n_jobs =-1)
        print ("BayesianRidge")
        print ("cross - validation cv=10 scores:\n", scores)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        print ("SCORE:\n",adj_r_sqrd(reg.score(X_temp, Y_temp) ))
        if adj_r_sqrd(scores.mean())>best_score_all:
            best_score_all=adj_r_sqrd(reg.score(X_temp, Y_temp))
            best_estimator_all=reg
            rsquared=reg.score(X_temp, Y_temp)
        ####################################################################
        #ARDRegression
        reg =ARDRegression().fit(X_temp, Y_temp)
        #cross validation
        scores = cross_val_score(reg, X_temp, Y_temp.ravel(), cv=10,n_jobs =-1)
        print ("ARDRegression")
        print ("cross - validation cv=10 scores:\n", scores)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        print ("SCORE:\n",adj_r_sqrd(reg.score(X_temp, Y_temp) ))
        if adj_r_sqrd(scores.mean())>best_score_all:
            best_score_all=adj_r_sqrd(reg.score(X_temp, Y_temp))
            best_estimator_all=reg
            rsquared=reg.score(X_temp, Y_temp)
        ####################################################################
        #svr
        tunedpars_svm={"kernel":["linear", "poly", "rbf", "sigmoid"],"C":[1,.95,.9,.8],"epsilon":[.1,.05,.15,.2,.5] }
        reg=GridSearchCV(SVR(), tunedpars_svm, cv=10,n_jobs=-1).fit(X_temp, Y_temp.ravel())
        print ("SVR")
        print ("SCORE:\n",adj_r_sqrd(reg.score(X_temp, Y_temp) ))
        if adj_r_sqrd(reg.score(X_temp, Y_temp))>best_score_all:
            best_score_all=adj_r_sqrd(reg.score(X_temp, Y_temp))
            best_estimator_all=reg
            rsquared=reg.score(X_temp, Y_temp)
        ####################################################################    
        #NuSVR    
        tunedpars_svm={"kernel":["linear", "poly", "rbf", "sigmoid"] }
        reg=GridSearchCV(NuSVR(), tunedpars_svm, cv=10,n_jobs=-1).fit(X_temp, Y_temp.ravel())
        print ("NuSVR")
        print ("SCORE:\n",adj_r_sqrd(reg.score(X_temp, Y_temp) ))
        if adj_r_sqrd(reg.score(X_temp, Y_temp) )>best_score_all:
            best_score_all=adj_r_sqrd(reg.score(X_temp, Y_temp) )
            best_estimator_all=reg
            rsquared=reg.score(X_temp, Y_temp)'''
        #################################################################### 
        return  best_score_all,best_estimator_all,rsquared
    ########################################################################    
    list_bests=list()
    list_preds_real=list()
    if meteo_or_iso=="meteo":
        num_var=3
        colmnss=["CooX","CooY","CooZ"]
        rangee=range(1,13)
    ##################################################
    if meteo_or_iso=="iso":
        
        num_var=len(inputs)
        colmnss=inputs
        rangee=range(1,2)
    for monthnum in rangee:    
        if meteo_or_iso=="meteo":
            newmatdf_temp=newmatdframe[monthnum-1]
            X_temp=newmatdf_temp[["CooX","CooY","CooZ"]].copy().astype(float)
        else:    
            newmatdf_temp=newmatdframe
            X_temp=newmatdframe[inputs].copy().astype(float)
        Y_temp=newmatdf_temp[[temp_rain_hum]].copy()
        X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(X_temp, Y_temp)    
        ##################################################
        #adjusted r squared
        sam_size=newmatdf_temp.shape[0]
        adj_r_sqrd=lambda r2,sam_size=sam_size,num_var=num_var: 1-(1-r2)*(sam_size-1)/(sam_size-num_var-1)
        ##################################################
        #some tests:
        mutual_info_regression_value = mutual_info_regression(X_temp, Y_temp)
        mutual_info_regression_value /= np.max(mutual_info_regression_value)
        print ("mutual_info_regression_value on whole data, standard!")
        print (mutual_info_regression_value)
        f_regression_value=f_regression(X_temp, Y_temp)
        print ("f_regression_value on whole data not standard")
        print (f_regression_value)
        ##################################################
        #just using the significant variables!
        f_less_ind=list(np.where(f_regression_value[1]<=0.05)[0])
        if len(f_less_ind)<2:
            f_regression_value_sor=sorted(f_regression_value[1])[:2]
            f_less_ind1=list(
                (np.where
                (f_regression_value[1]==f_regression_value_sor[0])
                                ) [0]
            )
            f_less_ind2=list((np.where
                (f_regression_value[1]==f_regression_value_sor[1])
                                )[0]   
            )                
            f_less_ind=f_less_ind1+f_less_ind2
        used_features=[colmnss[i] for i in f_less_ind]
        X_temp=X_temp[used_features]
        X_train_temp=X_train_temp[used_features]
        X_test_temp=X_test_temp[used_features]
        ########################################################################
        #scaling
        x_scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()
        x_scaler.fit(X_temp)
        X_train_temp = x_scaler.transform(X_train_temp)
        X_test_temp= x_scaler.transform(X_test_temp)
        X_temp=x_scaler.transform(X_temp)
        y_scaler.fit(Y_temp)
        y_train_temp = y_scaler.transform(y_train_temp)
        y_test_temp= y_scaler.transform(y_test_temp)
        Y_temp=y_scaler.transform(Y_temp)

        #applying regression function on normal data
        best_score_all_normal,best_estimator_all_normal,rsquared_normal=regressionfunctions(X_temp,Y_temp)
        ########################################################################
        #applying regression function on log data
        if apply_on_log==True:
            X_temp1=np.log1p(X_temp)
            Y_temp1=np.log1p(Y_temp)
            best_score_all_log,best_estimator_all_log,rsquared_log=regressionfunctions(X_temp1,Y_temp1)
            ########################################################################
            #comparison between normal and log model
            if best_score_all_normal > best_score_all_log:
                didlog=False
                best_score_all=best_score_all_normal
                best_estimator_all=best_estimator_all_normal
                rsquared=rsquared_normal
            else:
                didlog=True
                best_score_all=best_score_all_log
                best_estimator_all=best_estimator_all_log
                rsquared=rsquared_log
        else:
            didlog=False
        ########################################################################
        # scale transformations
        if didlog==True:
            Y_preds_output = best_estimator_all.predict(X_temp1) #predict created based on logged data, so Y_preds_output is logged!
            #transform log
            Y_preds=np.expm1(Y_preds_output) # inverse transform the log to not logged
        else:
            Y_temp_fin=Y_temp
            Y_preds_output = best_estimator_all.predict(X_temp)
            Y_preds=Y_preds_output.copy()
        #general standard
        Y_temp_fin=Y_temp
        Y_preds=y_scaler.inverse_transform( Y_preds.reshape(-1, 1) )    
        ######################################################################## 
        # Plots       
        lens=len(f_less_ind)
        f_namess=used_features
        if meteo_or_iso=="meteo":
            pltttl="month_"+ str(monthnum) +"_All_data_best_estimator_"+model_type
        else:
            pltttl="Annual_iso_All_data_best_estimator_"+model_type

        pltname=os.path.join(direc,"model_plots",model_type,"best_estimators",pltttl+'.pdf')
        pltnamepardep=os.path.join(direc,"model_plots",model_type,"partial_dependency",'_partial_dependence'+pltttl+'.pdf')
        #folder making and checking:
        Path(os.path.join(direc,"model_plots",model_type,"best_estimators")).mkdir(parents=True,exist_ok=True)
        Path(os.path.join(direc,"model_plots",model_type,"partial_dependency")).mkdir(parents=True,exist_ok=True)

        clnm=model_type+"_"+str(monthnum)
        Y_preds=pd.DataFrame(Y_preds,columns=[clnm])
        plt.scatter(Y_preds,Y_temp_fin)
        plt.title(pltttl)
        plt.xlabel("Prediction")
        plt.ylabel("Real Value")
        left, right = plt.xlim()
        left1, right1 = plt.ylim()
        a = np.linspace(min(left,left1),max(right,right1),100)
        b=a
        plt.plot(a,b)
        plt.savefig(pltname,dpi=300)
        #plt.show()
        plt.close()
        if didlog==False:
            plot_partial_dependence(best_estimator_all, X_temp, features=list(range(0,lens)),feature_names=f_namess)
        else:

            plot_partial_dependence(best_estimator_all, X_temp1, features=list(range(0,lens)),feature_names=f_namess)

        plt.savefig(pltnamepardep,dpi=300)
        #plt.show()
        plt.close()

        list_bests.append([best_estimator_all,best_score_all,mutual_info_regression_value,f_regression_value,used_features,rsquared,x_scaler,y_scaler,didlog])
        list_preds_real.append([Y_preds,Y_temp_fin,X_temp])  
    return list_bests,list_preds_real

        #some old lines
    
##########################################################################################
def iso_prediction(iso_db1,month_grouped_list_with_zeros_iso_2h,month_grouped_list_with_zeros_iso_3h,temp_bests,rain_bests,hum_bests,iso_18,dates_db,trajectories,iso_model_month_list,run_iso_whole_year,direc):
    
    predictions_monthly_list=list()
    col_for_f_reg=list()
    all_without_averaging=list()
    
    if trajectories==True:
        from pysplit_funcs_for_meteo_model import convertCoords,gen_trajs
        altitudes=[3000,5500]
        #generating reports- removing existed files
        if os.path.exists(os.path.join(r'C:\trajectories\RP',"error_in_meteo_file.txt")):
            os.remove(os.path.join(r'C:\trajectories\RP',"error_in_meteo_file.txt"))
        if os.path.exists(os.path.join(r'C:\trajectories\RP',"input_pass_to_bulkrajfun_report.txt")):
            os.remove(os.path.join(r'C:\trajectories\RP',"input_pass_to_bulkrajfun_report.txt"))
        error_in_meteo_file=open(os.path.join(r'C:\trajectories\RP',"error_in_meteo_file.txt"),"a+")
        input_pass_to_bulkrajfun_report=open(os.path.join(r'C:\trajectories\RP',"input_pass_to_bulkrajfun_report.txt"),"a+")
        for alti in altitudes:
            storage_dir =os.path.join(r'C:\trajectories\RP', str(alti))
            if os.path.exists(storage_dir)==False:
                os.mkdir(storage_dir)
            if os.path.exists(os.path.join(storage_dir,"points_all_in_water_report.txt")):
                os.remove(os.path.join(storage_dir,"points_all_in_water_report.txt"))
            if os.path.exists(os.path.join(storage_dir,"points_origin_not_detected_report.txt")):
                os.remove(os.path.join(storage_dir,"points_origin_not_detected_report.txt"))
            if os.path.exists(os.path.join(storage_dir,"traj_shorter_than_runtime_report.txt")):
                os.remove(os.path.join(storage_dir,"traj_shorter_than_runtime_report.txt"))        
    ##########################
    iso_model_month_list_min_one=[n-1 for n in iso_model_month_list]
    for month_num in range(0,len(iso_db1)):
        #if (month_num not in [3,4,5,6,7,8,9,11] and run_iso_whole_year ==True) or run_iso_whole_year==False:
        if (month_num in iso_model_month_list_min_one and run_iso_whole_year ==False) or run_iso_whole_year==True:
            bests_of_all_preds=list()
            col_names=["temp","rain","hum"]
            for bests_of_all, names in zip([temp_bests,rain_bests,hum_bests],col_names):

                #general input transform
                xyz=bests_of_all[month_num][-3].transform(iso_db1[month_num][bests_of_all[month_num][4]])                
                #transform if there is log in input
                if bests_of_all[month_num][-1]==True:
                    xyz=np.log1p(xyz)
                #predict temp
                bests_pred=pd.DataFrame(data=bests_of_all[month_num][0].predict(xyz),columns=[names])
                #inverse transform log output
                if bests_of_all[month_num][-1]==True:
                    bests_pred=np.expm1(bests_pred)
                #inverse transform general
                bests_pred=pd.DataFrame(bests_of_all[month_num][-2].inverse_transform(bests_pred),columns=[names])
            
                bests_of_all_preds.append(bests_pred)
            
            temp_pred=bests_of_all_preds[0]
            rain_pred=bests_of_all_preds[1]
            hum_pred=bests_of_all_preds[2]
            ###################################################################
            ######################
            preds=pd.concat([iso_db1[month_num][["CooX","CooY","CooZ","ID_MeteoPoint","iso_18"]],month_grouped_list_with_zeros_iso_2h[month_num][["iso_2h"]],month_grouped_list_with_zeros_iso_3h[month_num][["iso_3h"]],temp_pred,rain_pred,hum_pred],axis=1,ignore_index =False)   
            ######################
            all_hysplit_df_list_all_atts=list()
            all_hysplit_df_list_all_atts_without_averaging=list()
            #to pysplit trajectories: iterate altitudes
            if trajectories==True:
                 #here, the new coordination system have to be added from excel or calculated
                xy_df_for_hysplit=iso_db1[month_num][["CooX","CooY","ID_MeteoPoint"]]
                xy_df_for_hysplit = iso_db1[month_num].join(xy_df_for_hysplit.apply(convertCoords,axis=1)) 
                for altitude in range(0,len(altitudes)):
                    al=altitudes[altitude]
                    t_d=os.path.join(r'C:\trajectories\RP', str(al))
                    points_all_in_water_report=open(os.path.join(t_d,"points_all_in_water_report.txt"),"a+")    
                    points_origin_not_detected_report=open(os.path.join(t_d,"points_origin_not_detected_report.txt"),"a+")
                    traj_shorter_than_runtime_report=open(os.path.join(t_d,"traj_shorter_than_runtime_report.txt"),"a+")  
                    #the following input is for the times that there is no database for measurment dates, So we use the trajectories of 1 whole month
                    if type(dates_db) is None:
                        all_hysplit_df,nw_ex_all_df=gen_trajs(iso_18,xy_df_for_hysplit,month_num+1,al,points_all_in_water_report,points_origin_not_detected_report,error_in_meteo_file,traj_shorter_than_runtime_report,input_pass_to_bulkrajfun_report,Sampling_date_db=False)
                    else:
                        all_hysplit_df,nw_ex_all_df=gen_trajs(dates_db,xy_df_for_hysplit,month_num+1,al,points_all_in_water_report,points_origin_not_detected_report,error_in_meteo_file,traj_shorter_than_runtime_report,input_pass_to_bulkrajfun_report,Sampling_date_db=True)

                    all_hysplit_df_list_all_atts.append(all_hysplit_df)
                    #all_hysplit_df_list_all_atts_without_averaging.append(nw_ex_all_df)
                    #print ("############all_hysplit_df##########")
                    #print (all_hysplit_df)
                    ################################################### 
                    col_for_hy=["real_distt_alt_n"+"_"+str(al),"real_distt_alt_s"+"_"+str(al),"real_distt_pac_s"+"_"+str(al),"real_distt_pac_n"+"_"+str(al),"percentage_alt_n"+"_"+str(al),"percentage_alt_s"+"_"+str(al),"percentage_pac_s"+"_"+str(al),"percentage_pac_n"+"_"+str(al)]   
                    preds=pd.concat([preds,all_hysplit_df[col_for_hy]],axis=1,ignore_index =False)
                    preds["month"]=month_num+1
                    predictions_monthly_list.append([preds])
                    if altitude==0:
                        col_for_f_reg=col_for_hy
                        all_hysplit_df_list_all_atts_without_averaging=nw_ex_all_df
                    else:
                        col_for_f_reg=col_for_f_reg+col_for_hy   
                        all_hysplit_df_list_all_atts_without_averaging=all_hysplit_df_list_all_atts_without_averaging+nw_ex_all_df
                #print(all_hysplit_df_list_all_atts_without_averaging)        
                #print (list(all_hysplit_df_list_all_atts_without_averaging[0].columns))
                dfdf=pd.DataFrame(all_hysplit_df_list_all_atts_without_averaging,columns=["ID_MeteoPoint","newLat", "newLong","Cumulative_Dist","Dist_from_origin","continentality","altitude","year","month","day"])
                #print (dfdf)
            else:
                preds["month"]=month_num+1
                predictions_monthly_list.append([preds])

            if  run_iso_whole_year==True and month_num==0:
                all_preds=preds
                if trajectories==True:
                    all_without_averaging=dfdf
            elif run_iso_whole_year==False and month_num==min(iso_model_month_list_min_one) :
                all_preds=preds
                if trajectories==True:
                    all_without_averaging=dfdf
            else:
                all_preds=pd.concat([all_preds,preds],ignore_index=True)
                if trajectories==True:
                    all_without_averaging=pd.concat([all_without_averaging,dfdf],ignore_index=True)
    
    if trajectories==True:
        points_all_in_water_report.close()
        points_origin_not_detected_report.close()
        error_in_meteo_file.close() 
        traj_shorter_than_runtime_report.close()
        input_pass_to_bulkrajfun_report.close()
        all_without_averaging.to_excel(os.path.join(direc,"traj_data_no_averaging.xls"))
    #############################################################
    all_preds.to_excel(os.path.join(direc,"predicted_meteo_results_in_iso_stations.xls"))

    return  predictions_monthly_list, all_preds,all_hysplit_df_list_all_atts,col_for_f_reg,all_without_averaging      

###########################################################

#f_reg and mutual
def f_reg_mutual(file_name,all_preds,list_of_dics):
    m_out_f=open(file_name,'w')
    for sets in list_of_dics:
        st_exist=False
        st_exist_m=True
        print ("all_preds[sets[inputs]:", all_preds[sets["inputs"]])
        print ("all_preds[sets[outputs]]:",all_preds[sets["outputs"]])
        try:
            mutual_info = mutual_info_regression(all_preds[sets["inputs"]],all_preds[sets["outputs"]])
        except:
            st_exist_m=False
        try:
            maxx=np.max(mutual_info)
            mutual_info_st=list()
            for jj in range(0,len(mutual_info)):
                mutual_info_st.append(mutual_info[jj]/ maxx)
            st_exist=True
        except:
            print ("MI ALL ZEROSS!!!")   
            
            
        f_reg=f_regression(all_preds[sets["inputs"]],all_preds[sets["outputs"]])
        #print (type(f_reg[0]))
        try:
            mxx=np.max(f_reg[0])
            for jj in range(0,len(f_reg[0])):
                f_reg[0][jj]/= mxx
        except:
            print ("f test all zeross!!")
        m_out_f.write("########################################################")
        m_out_f.write('\n')
        m_out_f.write("########################################################")
        m_out_f.write('\n\n')
        m_out_f.write('inputs:')
        m_out_f.write(str(sets["inputs"]))
        m_out_f.write('\n')
        m_out_f.write('output:')
        m_out_f.write(str(sets["outputs"]))
        m_out_f.write('\n\n\n')
        m_out_f.write("f_regression ")
        m_out_f.write('\n')
        m_out_f.write(str(f_reg[0]))
        m_out_f.write('\n')
        m_out_f.write(str(f_reg[1]))
        m_out_f.write('\n\n')
        m_out_f.write("mutual_info_standard ")
        m_out_f.write('\n')
        if st_exist==True:
            m_out_f.write(str(mutual_info_st))
        else:
            m_out_f.write("Not possible to calculate the STANDARD mutual, possibly division by zero problem!") 
        m_out_f.write('\n\n')
        m_out_f.write("mutual_info ")
        m_out_f.write('\n')     
        if st_exist_m==True:
            m_out_f.write(str(mutual_info))
            m_out_f.write('\n')
        else:
            m_out_f.write("Not possible to calculate mutual. Possibly not enough data")
            m_out_f.write('\n')

            
        
    m_out_f.close()

###########################################################