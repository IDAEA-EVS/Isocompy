from math import gamma
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
from sklearn.linear_model import LarsCV,BayesianRidge,ARDRegression,MultiTaskElasticNet,OrthogonalMatchingPursuit,ElasticNet
from sklearn.inspection import plot_partial_dependence
from sklearn.svm import SVR,NuSVR
import os
from pathlib import Path
import dill
from datetime import datetime
from sklearn.linear_model import LinearRegression
import itertools
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses

def warn(*args, **kwargs):
    pass
warnings.warn = warn
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
##########################################################################################
          
##########################################################################################
#model!
def rfmethod(tunedpars_rfr,tunedpars_svr,tunedpars_nusvr,tunedpars_mlp,newmatdframe,meteo_or_iso,inputs,apply_on_log,direc,cv,which_regs,vif_threshold,p_val):
    ########################################################################
    #regression functions
    def regressionfunctions(X_temp,Y_temp,which_regs):
        tunedpars_lr={}
        tunedpars_br={}
        tunedpars_ard={}
        tunedpars_muelnet={"l1_ratio":[.1, .5, .7,.9,.99]}
        tunedpars_elnet={"l1_ratio":[.1, .5, .7,.9,.99]}

        tunedpars_omp={}
        reg_ref_dic={
            "omp":OrthogonalMatchingPursuit(),
            "muelnet":MultiTaskElasticNet(),
            "elnet":ElasticNet(),
            "rfr":RandomForestRegressor(random_state =0),
            "mlp":MLPRegressor(learning_rate="adaptive"),
            "br":BayesianRidge(),
            "ard":ARDRegression(),
            "svr":SVR(cache_size=8000),
            "nusvr":NuSVR()}

        tunedpars_dic={
            "rfr":tunedpars_rfr,
            "mlp":tunedpars_mlp,
            "br":tunedpars_br,
            "ard":tunedpars_ard,
            "svr":tunedpars_svr,
            "nusvr":tunedpars_nusvr,
            "elnet":tunedpars_elnet,
            "muelnet":tunedpars_muelnet,
            "omp":tunedpars_omp}

        model_dic=dict()
        for key in which_regs.keys():
            if which_regs[key]==True:
                new_v=tunedpars_dic[key]
                new_k=reg_ref_dic[key]
                model_dic[new_k]=new_v



    

        #LinearRegression
        reg =GridSearchCV(LinearRegression(), tunedpars_lr, cv=cv,n_jobs=-1)
        reg.fit(X_temp, Y_temp.ravel())
        #rsquared=reg.best_score_
        rsquared=reg.score(X_temp, Y_temp)
        best_score_all=adj_r_sqrd(reg.score(X_temp, Y_temp)) 
        best_estimator_all=reg

        for ttm in model_dic.items(): 
            tunedpars=ttm[1]
            models=ttm[0]
            reg =GridSearchCV(models, tunedpars, cv=cv,n_jobs=-1)
            try:
                reg.fit(X_temp, Y_temp.ravel())
                if adj_r_sqrd(reg.score(X_temp, Y_temp))>best_score_all:
                    #rsquared=reg.best_score_  
                    rsquared=reg.score(X_temp, Y_temp)
                    best_score_all=adj_r_sqrd(rsquared)
                    best_estimator_all=reg
            except:
                print("####In except: ",ttm)
                pass        
        
        #to transfer score from cv score to normal:
        #rsquared=best_estimator_all.score(X_temp, Y_temp)      
        #best_score_all=adj_r_sqrd(rsquared)        
        #################################################################################
        #(MULTI TASK) ELASTIC NET CV
        #elastic net on standardized data
        '''print (which_regs)
        if which_regs["elnet"]==True: 
            reg = MultiTaskElasticNet(n_jobs =-1,normalize=False,cv=cv ).fit(X_temp, Y_temp)
            if adj_r_sqrd(reg.score(X_temp, Y_temp))>best_score_all:
                best_score_all=adj_r_sqrd(reg.score(X_temp, Y_temp))
                best_estimator_all=reg
                rsquared=reg.score(X_temp, Y_temp)
        #################################################################################
        #OrthogonalMatchingPursuitCV
        if which_regs["omp"]==True:
            reg =OrthogonalMatchingPursuitCV(cv,n_jobs=-1).fit(X_temp, Y_temp)
            if adj_r_sqrd(reg.score(X_temp, Y_temp))>best_score_all:
                best_score_all=adj_r_sqrd(reg.score(X_temp, Y_temp))
                best_estimator_all=reg
                rsquared=reg.score(X_temp, Y_temp)'''        
        #################################################################### 
        return  best_score_all,best_estimator_all,rsquared
    ########################################################################    
    list_bests=list()
    list_preds_real_dic=list()
    list_bests_dic=list()
    if meteo_or_iso=="meteo":
        num_var=3
        colmnss=["CooX","CooY","CooZ"]
        rangee=range(1,13)
    ##################################################
    else:
        
        num_var=len(inputs)
        colmnss=inputs
        rangee=range(1,2)
    for monthnum in rangee:
        if meteo_or_iso=="meteo":
                newmatdf_temp=newmatdframe[monthnum-1]
                X_temp=newmatdf_temp[colmnss].copy().astype(float)
        else:    
            newmatdf_temp=newmatdframe
            X_temp=newmatdframe[colmnss].copy().astype(float)

        if cv=="auto":
            if newmatdf_temp.shape[0]<11: cv=newmatdf_temp.shape[0]-2
            else: cv=10


        if newmatdf_temp.shape[0] > 2 and newmatdf_temp.shape[0]>cv :
            
            Y_temp=newmatdf_temp[["Value"]].copy()
            X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(X_temp, Y_temp)    
            ##################################################
            #adjusted r squared
            sam_size=newmatdf_temp.shape[0]
            adj_r_sqrd=lambda r2,sam_size=sam_size,num_var=num_var: 1-(1-r2)*(sam_size-1)/(sam_size-num_var-1)
            ##################################################
            #correlation coefficient    
            correl_mat=X_temp.corr()
            ##################################################
            #VIF checking
            if vif_threshold !=None:
                from statsmodels.stats.outliers_influence import variance_inflation_factor
                from statsmodels.tools.tools import add_constant
                X = add_constant(X_temp)
                vif_df=pd.DataFrame([variance_inflation_factor(X.values, i) 
                            for i in range(X.shape[1])], 
                            index=X.columns).rename(columns={0:'VIF'}).drop('const')
                print (vif_df)                   
                more_than_thres=vif_df[vif_df["VIF"]>vif_threshold].index

                X=X.drop('const', axis=1)
                vif_df_ommited=vif_df.copy()
                zt_omm=False
                if ("CooZ" and "temp") in more_than_thres:
                    X=X.drop('temp', axis=1)
                    more_than_thres=more_than_thres.drop(['temp',"CooZ"])
                    vif_df_ommited=vif_df_ommited.drop(['temp',"CooZ"])
                    zt_omm=True
                print("more_than_thres after z temp",more_than_thres)    
                if len(more_than_thres)==1:
                    X=X.drop(more_than_thres,axis=1)
                    more_than_thres=more_than_thres.drop(more_than_thres)
                    vif_df_ommited=vif_df_ommited.drop(more_than_thres)
                if len(more_than_thres)>1:
                    num_to_omit=len(more_than_thres)//2
                    if np.isinf(vif_df_ommited["VIF"]).any()==False:
                        more_than_thres=more_than_thres.drop(vif_df_ommited.sort_values("VIF",ascending=False)[0:num_to_omit].index)
                        X=X.drop(more_than_thres, axis=1)
                    else:
                        for its in range(0,num_to_omit):
                            if zt_omm==True:
                                X_corr=X.drop('CooZ',axis=1).corr()
                            else:    
                                X_corr=X.corr()
                            ind_to_omit=abs(X_corr[X_corr!=1]).unstack().sort_values(ascending=False)[0:1].index[0][0]
                            more_than_thres=more_than_thres.drop([ind_to_omit])
                            X=X.drop(ind_to_omit, axis=1)

                print("more_than_thres after >1",more_than_thres)     
                X_temp=X  
                vif_chosen_features=X_temp.columns
            else:
                vif_df="Not Calculated"  
                vif_chosen_features="Not Calculated"  
            #################################################
            #some tests:
            print ("X_temp.columns",X_temp.columns)
            mutual_info_regression_value = mutual_info_regression(X_temp, Y_temp)
            mutual_info_regression_value /= np.max(mutual_info_regression_value)
            f_regression_value=f_regression(X_temp, Y_temp)
            ##################################################
            #just using the significant variables!
            f_less_ind=list(np.where(f_regression_value[1]<=p_val)[0])
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
            used_features=[X_temp.columns[i] for i in f_less_ind]

            X_temp=X_temp[used_features]
            X_train_temp=X_train_temp[used_features]
            X_test_temp=X_test_temp[used_features]
            print ("used_features",used_features)
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
            best_score_all_normal,best_estimator_all_normal,rsquared_normal=regressionfunctions(X_temp,Y_temp,which_regs)
            ########################################################################
            #applying regression function on log data
            if apply_on_log==True:
                X_temp1=np.log1p(X_temp)
                Y_temp1=np.log1p(Y_temp)
                best_score_all_log,best_estimator_all_log,rsquared_log=regressionfunctions(X_temp1,Y_temp1,which_regs)
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
                best_score_all=best_score_all_normal
                best_estimator_all=best_estimator_all_normal
                rsquared=rsquared_normal
            ########################################################################
            # scale transformations
            if didlog==True:
                Y_preds = best_estimator_all.predict(X_temp1) #predict created based on logged data, so Y_preds is logged!
                #transform log
                Y_preds=np.expm1(Y_preds) # inverse transform the log to not logged
            else:
                Y_preds = best_estimator_all.predict(X_temp)
            #general standard
            Y_measured=y_scaler.inverse_transform( Y_temp.reshape(-1, 1) )
            Y_preds=y_scaler.inverse_transform( Y_preds.reshape(-1, 1) )    
            ######################################################################## 

            Y_preds=pd.DataFrame(Y_preds,columns=["Value"])
            list_bests.append([best_estimator_all,best_score_all,mutual_info_regression_value,f_regression_value,used_features,rsquared,x_scaler,y_scaler,didlog])
            list_preds_real_dic.append({"Y_preds":Y_preds,"Y_temp_fin":Y_measured,"X_temp":X_temp})  
            list_bests_dic.append({"best_estimator":best_estimator_all,"best_score":best_score_all,"mutual_info_regression":mutual_info_regression_value,
            "f_regression":f_regression_value,"used_features":used_features,"rsquared":rsquared,"didlog":didlog,"vif":vif_df,"correlation":correl_mat,"vif_chosen_features":vif_chosen_features})
        else:
            Y_preds=pd.DataFrame()
            list_bests.append([None,None,None,None,None,None,None,None,None])
            list_preds_real_dic.append({"Y_preds":pd.DataFrame(),"Y_temp_fin":pd.DataFrame(),"X_temp":pd.DataFrame()})  
            list_bests_dic.append({"best_estimator":None,"best_score":None,"mutual_info_regression":None,
            "f_regression":None,"used_features":[],"rsquared":None,"didlog":None,"vif":None,"correlation":None,"vif_chosen_features":None})
    return list_bests,list_preds_real_dic,list_bests_dic

    
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
            iso_db1[month_num]["ID_MeteoPoint"]=iso_db1[month_num].ID_MeteoPoint.astype(str)
            iso_db1[month_num]=iso_db1[month_num].sort_values(by="ID_MeteoPoint").reset_index()
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
            
            temp_pred=pd.concat([bests_of_all_preds[0],iso_db1[month_num]["ID_MeteoPoint"]],axis=1)
            temp_pred=temp_pred.set_index("ID_MeteoPoint")
            rain_pred=pd.concat([bests_of_all_preds[1],iso_db1[month_num]["ID_MeteoPoint"]],axis=1)
            rain_pred=rain_pred.set_index("ID_MeteoPoint")
            hum_pred=pd.concat([bests_of_all_preds[2],iso_db1[month_num]["ID_MeteoPoint"]],axis=1)
            hum_pred=hum_pred.set_index("ID_MeteoPoint")
            ###################################################################
            ######################
            iso_db1[month_num]=iso_db1[month_num].rename(columns={"Value": "iso_18"}).set_index("ID_MeteoPoint",drop=False)
            month_grouped_list_with_zeros_iso_2h[month_num]=month_grouped_list_with_zeros_iso_2h[month_num].sort_values(by="ID_MeteoPoint").set_index("ID_MeteoPoint").rename(columns={"Value": "iso_2h"})
            month_grouped_list_with_zeros_iso_3h[month_num]=month_grouped_list_with_zeros_iso_3h[month_num].sort_values(by="ID_MeteoPoint").set_index("ID_MeteoPoint").rename(columns={"Value": "iso_3h"})
            preds=pd.concat([iso_db1[month_num][["CooX","CooY","CooZ","ID_MeteoPoint","iso_18"]],month_grouped_list_with_zeros_iso_2h[month_num][["iso_2h"]],month_grouped_list_with_zeros_iso_3h[month_num][["iso_3h"]],temp_pred,rain_pred,hum_pred],axis=1,ignore_index =False,join="inner").reset_index(drop=True)   
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
def print_to_file(file_name,best_dics):

    m_out_f=open(file_name,'w')
    cnt=0
    for featt in best_dics.items():
        for monthnum in range(0,len(featt[1])):
            title=featt[0]+'\nmonth\n'
            m_out_f.write(title)
            m_out_f.write(str(monthnum+1))
            m_out_f.write('\n\n\n Used features \n')
            m_out_f.write(str(featt[1][monthnum][4]))
            m_out_f.write('\n\n BEST SCORE adjusted R squared \n')
            m_out_f.write(str(featt[1][monthnum][1]))
            m_out_f.write('\n\n BEST SCORE R squared \n')
            m_out_f.write(str(featt[1][monthnum][5]))
            m_out_f.write('\n\n F REGRESSION \n')
            try:
                m_out_f.write(str(featt[1][monthnum][3][0]))
                m_out_f.write('\n')
                m_out_f.write(str(featt[1][monthnum][3][1]))
            except: pass

            m_out_f.write('\n\n MUTUAL INFORMATION REGRESSION \n')
            m_out_f.write(str(featt[1][monthnum][2]))
            m_out_f.write('\n\n BEST ESTIMATOR \n')
            m_out_f.write(str(featt[1][monthnum][0]))
            m_out_f.write('\n\nlog(1 + x)? \n')
            m_out_f.write(str(featt[1][monthnum][-1]))
            m_out_f.write('\n')
            m_out_f.write("########################################################\n")
        m_out_f.write('\n"########################################################\n')
        cnt=cnt+1  
    m_out_f.close() 
    return       