import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV,KFold
from sklearn.exceptions import DataConversionWarning
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LarsCV,BayesianRidge,ARDRegression,MultiTaskElasticNet,OrthogonalMatchingPursuit,ElasticNet
from sklearn.svm import SVR,NuSVR
import os
from pathlib import Path
from sklearn.linear_model import LinearRegression
import itertools
import sys
import copy 
import warnings
from isocompy.feature_select import feature_selection
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses

def warn(*args, **kwargs):
    pass
warnings.warn = warn
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
##########################################################################################
#search a dic by keys
def search_keys(dic,needed_value):
    for k,v in dic.items():
        if v == needed_value:
            return k   
##########################################################################################
#model!
def rfmethod(newmatdframe,model_pars_name_dic,stage_1_2,st1_model_month_list,args_dic,fields):
    ########################################################################
    apply_on_log=model_pars_name_dic["apply_on_log"]
    cv=model_pars_name_dic["cv"]
    which_regs=model_pars_name_dic["which_regs"]
    #vif_threshold=model_pars_name_dic["vif_threshold"]
    ########################################################################
    #regression functions
    def regressionfunctions(X_temp,Y_temp,which_regs):
        
        tunedpars_lr=model_pars_name_dic["tunedpars_lr"]

        reg_ref_dic={
            "omp":OrthogonalMatchingPursuit(),
            "muelnet":MultiTaskElasticNet(),
            "elnet":ElasticNet(),
            "rfr":RandomForestRegressor(random_state =0),
            "mlp":MLPRegressor(learning_rate="adaptive",random_state =0),
            "br":BayesianRidge(),
            "ard":ARDRegression(),
            "svr":SVR(),
            "nusvr":NuSVR()}

        tunedpars_dic={
            "rfr":model_pars_name_dic["tunedpars_rfr"],
            "mlp":model_pars_name_dic["tunedpars_mlp"],
            "br":model_pars_name_dic["tunedpars_br"],
            "ard":model_pars_name_dic["tunedpars_ard"],
            "svr":model_pars_name_dic["tunedpars_svr"],
            "nusvr":model_pars_name_dic["tunedpars_nusvr"],
            "elnet":model_pars_name_dic["tunedpars_elnet"],
            "muelnet":model_pars_name_dic["tunedpars_muelnet"],
            "omp":model_pars_name_dic["tunedpars_omp"]}
        models_output_dic=dict()
        model_dic=dict()
        for key in which_regs.keys():
            if which_regs[key]==True:
                model_dic[key]=[tunedpars_dic[key],reg_ref_dic[key]]



        cv1=KFold(n_splits=cv,shuffle=True,random_state=1)

        #LinearRegression
        reg =GridSearchCV(LinearRegression(), tunedpars_lr, cv=cv1,n_jobs=-1,return_train_score=True)
        reg.fit(X_temp, Y_temp.ravel())
        #rsquared=reg.best_score_
        rsquared=reg.score(X_temp, Y_temp)
        best_score_all=adj_r_sqrd(reg.score(X_temp, Y_temp)) 
        best_estimator_all=reg
        models_output_dic["lr"]={"model": reg,"mod_score":rsquared, "predicted_Y": reg.predict(X_temp)}

        for ttm in model_dic.items(): 
            tunedpars=ttm[1][0]
            models=ttm[1][1]
            mod_name=ttm[0]
            reg =GridSearchCV(models, tunedpars, cv=cv1,n_jobs=-1,return_train_score=True)
            try:
                reg.fit(X_temp, Y_temp.ravel())
                mod_score=reg.score(X_temp, Y_temp)
                models_output_dic[mod_name]={"model": reg,"mod_score":mod_score, "predicted_Y": reg.predict(X_temp) }
                if adj_r_sqrd(mod_score)>best_score_all:
                    #rsquared=reg.best_score_  
                    rsquared=mod_score
                    best_score_all=adj_r_sqrd(rsquared)
                    best_estimator_all=reg

            except:
                print("####In except: ",mod_name)
                pass        
        
        #to transfer score from cv score to normal:
        #rsquared=best_estimator_all.score(X_temp, Y_temp)      
        #best_score_all=adj_r_sqrd(rsquared)        
        #################################################################################

        #################################################################### 
        return  best_score_all,best_estimator_all,rsquared,models_output_dic
    ########################################################################    
    list_bests=list()
    list_preds_real_dic=list()
    list_bests_dic=list()
    colmnss=fields #fields

    if stage_1_2==1:
        rangee=range(1,13)
    ##################################################
    else:
        rangee=range(1,2)
    for monthnum in rangee:
        if stage_1_2==1:
            if monthnum in st1_model_month_list:
                newmatdf_temp=newmatdframe[monthnum-1]
                X_temp=newmatdf_temp[colmnss].copy().astype(float)
            else:
                newmatdf_temp=pd.DataFrame()
        else:    
            newmatdf_temp=newmatdframe
            X_temp=newmatdframe[colmnss].copy().astype(float)

        if cv=="auto":
            if newmatdf_temp.shape[0]<11: cv=2
            elif newmatdf_temp.shape[0]/10 >20 : cv=10
            else : cv=5


        if newmatdf_temp.shape[0] > 2 and newmatdf_temp.shape[0]>cv :
            
            Y_temp=newmatdf_temp[["Value"]].copy()
            X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(X_temp, Y_temp)    

            mutual_info_regression_value,f_regression_value,correl_mat,used_features,vif_df,vif_chosen_features,vif_initial=feature_selection(X_temp,Y_temp,args_dic)


            X_temp=X_temp[used_features]
            X_train_temp=X_train_temp[used_features]
            X_test_temp=X_test_temp[used_features]
            print ("used_features",used_features)
            ##################################################
            num_var=len(used_features)
            #adjusted r squared
            sam_size=newmatdf_temp.shape[0]
            adj_r_sqrd=lambda r2,sam_size=sam_size,num_var=num_var: 1-(1-r2)*(sam_size-1)/(sam_size-num_var-1)
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
            best_score_all_normal,best_estimator_all_normal,rsquared_normal,models_output_dic_normal=regressionfunctions(X_temp,Y_temp,which_regs)
            ########################################################################
            #applying regression function on log data
            models_output_dic_log=dict()
            if apply_on_log==True:
                X_temp1=np.log1p(X_temp)
                Y_temp1=np.log1p(Y_temp)
                best_score_all_log,best_estimator_all_log,rsquared_log,models_output_dic_tl=regressionfunctions(X_temp1,Y_temp1,which_regs)
                for k,v in models_output_dic_tl.items():
                    models_output_dic_log["log_"+k]=v
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
            models_output_dic={**models_output_dic_normal,**models_output_dic_log}
            Y_preds=pd.DataFrame(Y_preds,columns=["Value"])
            list_bests.append([best_estimator_all,best_score_all,mutual_info_regression_value,f_regression_value,used_features,rsquared,x_scaler,y_scaler,didlog])
            list_preds_real_dic.append({"Y_preds":Y_preds,"Y_temp_fin":Y_measured,"X_temp":X_temp})  
            list_bests_dic.append({"best_estimator":best_estimator_all,"best_score":best_score_all,"mutual_info_regression":mutual_info_regression_value,
            "f_regression":f_regression_value,"used_features":used_features,"rsquared":rsquared,"didlog":didlog,"vif":vif_df,"vif_initial":vif_initial,"correlation":correl_mat,"vif_chosen_features":vif_chosen_features,"models_output_dic":models_output_dic})
        else:
            Y_preds=pd.DataFrame()
            list_bests.append([None,None,None,None,None,None,None,None,None])
            list_preds_real_dic.append({"Y_preds":pd.DataFrame(),"Y_temp_fin":pd.DataFrame(),"X_temp":pd.DataFrame()})  
            list_bests_dic.append({"best_estimator":None,"best_score":None,"mutual_info_regression":None,
            "f_regression":None,"used_features":[],"rsquared":None,"didlog":None,"vif":None,"correlation":None,"vif_initial":None,"vif_chosen_features":None,"models_output_dic":{}})
    return list_bests,list_preds_real_dic,list_bests_dic

    
##########################################################################################
def iso_prediction(
    cls_list, 
    st1_model_results_dic,
    dates_db,
    trajectories,
    iso_model_month_list,direc):

    predictions_monthly_list=list()
    col_for_f_reg=list()
    all_without_averaging=list()
    
    iso_db1=cls_list[0].month_grouped_inp_var
    if trajectories==True:
        from pysplit_funcs_for_meteo_model import convertCoords,gen_trajs
        altitudes=[3000]
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
        if month_num in iso_model_month_list_min_one:
            iso_db1[month_num]["ID"]=iso_db1[month_num].ID.astype(str)
            iso_db1[month_num]=iso_db1[month_num].sort_values(by="ID").reset_index()
            bests_of_all_preds=list()
            for k,v in st1_model_results_dic.items():
                b_o_l=v["bests"][month_num]
                #general input transform
                xyz=b_o_l[-3].transform(iso_db1[month_num][b_o_l[4]])   #4=usedfeatures             
                #transform if there is log in input
                if b_o_l[-1]==True:
                    xyz=np.log1p(xyz)
                #predict temp
                bests_pred=pd.DataFrame(data=b_o_l[0].predict(xyz),columns=[k])
                #inverse transform log output
                if b_o_l[-1]==True:
                    bests_pred=np.expm1(bests_pred)
                #inverse transform general
                bests_pred=pd.DataFrame(b_o_l[-2].inverse_transform(bests_pred),columns=[k])
                bests_pred=pd.concat([bests_pred,iso_db1[month_num]["ID"]],axis=1)
                bests_pred=bests_pred.set_index("ID")
                bests_of_all_preds.append(bests_pred)
            #print ("bests_of_all_preds",bests_of_all_preds)
            ###################################################################
            ######################
            #to find the common features for second stage
            temp_used_f=list()
            for cls in cls_list:
                temp_used_f.extend(cls.db_input_args_dics["fields"])
            temp_used_f = list(set(temp_used_f)) + ["ID"]
            ######################  
            preds=iso_db1[month_num][temp_used_f].set_index("ID")
            #print (iso_db1, month_num, temp_used_f)
            #print ("preds 375\n",preds)
            for cls in cls_list:
                name=cls.db_input_args_dics["var_name"]
                n=cls.month_grouped_inp_var[month_num].rename(columns={"Value":name }).sort_values(by="ID").set_index("ID",drop=False)
                #print (name,n)
                preds=pd.concat([preds,n[name]],axis=1,ignore_index =False,join="inner")
            for i in bests_of_all_preds:
                preds=pd.concat([preds,i],axis=1,ignore_index =False,join="inner")
            preds=preds.reset_index(drop=False)   
            #print ("preds\n",preds)
            ######################
            all_hysplit_df_list_all_atts=list()
            all_hysplit_df_list_all_atts_without_averaging=list()
            #to pysplit trajectories: iterate altitudes
            if trajectories==True:
                 #here, the new coordination system have to be added from excel or calculated
                xy_df_for_hysplit=iso_db1[month_num][["CooX","CooY","ID"]]
                xy_df_for_hysplit = iso_db1[month_num].join(xy_df_for_hysplit.apply(convertCoords,axis=1)) 
                for altitude in range(0,len(altitudes)):
                    al=altitudes[altitude]
                    t_d=os.path.join(r'C:\trajectories\RP', str(al))
                    points_all_in_water_report=open(os.path.join(t_d,"points_all_in_water_report.txt"),"a+")    
                    points_origin_not_detected_report=open(os.path.join(t_d,"points_origin_not_detected_report.txt"),"a+")
                    traj_shorter_than_runtime_report=open(os.path.join(t_d,"traj_shorter_than_runtime_report.txt"),"a+")  
                    #the following input is for the times that there is no database for measurment dates, So we use the trajectories of 1 whole month
                    if dates_db is None:
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
                dfdf=pd.DataFrame(all_hysplit_df_list_all_atts_without_averaging,columns=["ID","newLat", "newLong","Cumulative_Dist","Dist_from_origin","continentality","altitude","year","month","day"])
            else:
                preds["month"]=month_num+1
                predictions_monthly_list.append([preds])


            if month_num==min(iso_model_month_list_min_one) :
                all_preds=preds
                if trajectories==True:
                    all_without_averaging=dfdf
            else:
                all_preds=pd.concat([all_preds,preds],ignore_index=False)
                if trajectories==True:
                    all_without_averaging=pd.concat([all_without_averaging,dfdf],ignore_index=False)
    
    if trajectories==True:
        points_all_in_water_report.close()
        points_origin_not_detected_report.close()
        error_in_meteo_file.close() 
        traj_shorter_than_runtime_report.close()
        input_pass_to_bulkrajfun_report.close()
        all_without_averaging.to_excel(os.path.join(direc,"traj_data_no_averaging.xls"))
    #############################################################
    all_preds=all_preds.reset_index()
    all_preds.to_csv(os.path.join(direc,"predicted_st1_results_in_st2_observations.csv"))
    
    return  predictions_monthly_list, all_preds,all_hysplit_df_list_all_atts,col_for_f_reg,all_without_averaging      

###########################################################
def st1_print_to_file(direc,best_dics,best_dics_dics_p):
    file_name=os.path.join(direc,"st1_model_12_month_stats.txt")
    m_out_f=open(file_name,'w')
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
                m_out_f.write("F\n")
                m_out_f.write(str(featt[1][monthnum][3].iloc[0]))
                m_out_f.write('\n P values\n')
                m_out_f.write(str(featt[1][monthnum][3].iloc[1]))
            except: pass

            m_out_f.write('\n\n MUTUAL INFORMATION REGRESSION \n')
            m_out_f.write(str(featt[1][monthnum][2]))

            m_out_f.write('\n\n BEST ESTIMATOR & parameters\n')
            try:
                m_out_f.write(str(featt[1][monthnum][0].best_estimator_))
                m_out_f.write('\n\n')
                m_out_f.write(str(featt[1][monthnum][0].best_params_))
            except:
                m_out_f.write(str(featt[1][monthnum][0]))

            m_out_f.write('\n\nlog(1 + x)? \n')
            m_out_f.write(str(featt[1][monthnum][-1]))
            m_out_f.write('\n')
            m_out_f.write("########################################################\n")
        m_out_f.write('\n"########################################################\n')
    #"vif":vif_df,"vif_initial":vif_initial
    for featt in best_dics_dics_p.items():
        for monthnum in range(0,len(featt[1])):
            m_out_f.write("\n\n########\n\n")
            m_out_f.write(featt[0]+'\nmonth'+str(monthnum+1))
            m_out_f.write('\n\nVIF INITIAL \n')
            m_out_f.write(str(featt[1][monthnum]["vif_initial"]))
            m_out_f.write('\n\nVIF FINAL \n')
            m_out_f.write(str(featt[1][monthnum]["vif"]))
            m_out_f.write('\n\nVIF SELECTED FEATURES \n')
            m_out_f.write(str(featt[1][monthnum]["vif_chosen_features"]))
    m_out_f.close()

    ############
    #writing all models
    file_name=os.path.join(direc,"st1_all_models_12_month_stats.txt")
    m_out_f=open(file_name,'w')
    for featt in best_dics_dics_p.items():
        for monthnum in range(0,len(featt[1])):
            m_out_f.write(featt[0]+'\nmonth'+str(monthnum+1))
            dic_al_mods=featt[1][monthnum]["models_output_dic"]
            for k,v in dic_al_mods.items():
                m_out_f.write('\n\nmodel\n'+k+"\n\n"+str(v["model"].best_estimator_)+"\nbest params\n"+str(v["model"].best_params_)+"\n\nR^2:\n"+str(v["mod_score"]))
                m_out_f.write("\n\n#########################\n\n")
        m_out_f.write("\n\n#########################\n#########################\n\n")
    m_out_f.close()



    return       



def isotope_model_selection_by_meteo_line(x,y,thresh,a,b,selection_method,report,direc,all_preds,st2_model_results_dic):

    """
    selection_method:
        independent: old (default)
        local_line
        global_line
        point_to_point
    """
    #adj_r_sqrd=lambda r2,sam_size=sam_size,num_var=num_var: 1-(1-r2)*(sam_size-1)/(sam_size-num_var-1)
    #list_bests.append([best_estimator_all,best_score_all,mutual_info_regression_value,f_regression_value,used_features,rsquared,x_scaler,y_scaler,didlog])
    for k,v in st2_model_results_dic.items():
        if k==x:
            iso18_bests_dic=v["bests_dic"]
            iso18_preds_real_dic=v["preds_real_dic"]
            iso18_bests=v["bests"]

        if k==y:
            iso2h_bests_dic=v["bests_dic"]
            iso2h_preds_real_dic=v["preds_real_dic"]
            iso2h_bests=v["bests"]

    y_scaler_18=iso18_bests[0][7]
    y_scaler_2h=iso2h_bests[0][7]


    #transform allpreds isos:
    all_preds_=copy.deepcopy(all_preds)
    all_preds_.iso_18=y_scaler_18.transform(np.array(all_preds_[[x]]).reshape(-1,1))
    all_preds_.iso_2h=y_scaler_2h.transform(np.array(all_preds_[[y]]).reshape(-1,1))
    all_preds_=round(all_preds_,5)
    iso_18_obs=all_preds_[[x]]
    iso_2h_obs=all_preds_[[y]]
    ################
    #to just choose between good scores!
    iso18_models_=iso18_bests_dic[0]["models_output_dic"]
    iso2h_models_=iso2h_bests_dic[0]["models_output_dic"]

    def choose_high_score(isotopemodels_,thresh,selection_method):
        listt = [(k, v["mod_score"]) for k, v in isotopemodels_.items()]
        from operator import itemgetter
        listt=sorted(listt, key = itemgetter(1))

        listt_vals=np.array([x[1] for x in listt])
        if thresh==None or (thresh != "point_to_point" and thresh>=1):
            thresh=listt_vals.mean()+listt_vals.std()/3
            print ("threshold:",thresh)
 
        if selection_method=="independent":
            key_list=[listt[-1][0]]
        else:
            key_list=list()
            for i in listt:
                if i[1]>=thresh:
                    key_list.append(i[0]) 
        print ("models to choose: ", key_list)     
        isotopemodels=copy.deepcopy(isotopemodels_)
        topop=list(isotopemodels.keys()-key_list)
        for i in topop:
            del isotopemodels[i]
        return isotopemodels    
    
    def set_vals(v_18,log_18_t,predicted_18):
        return v_18["model"],log_18_t,v_18["mod_score"],predicted_18

    iso18_models=choose_high_score(iso18_models_,thresh,selection_method)
    iso2h_models=choose_high_score(iso2h_models_,thresh,selection_method)
    #to make resid report:
    x_transformed_18=iso18_bests[0][6].inverse_transform(iso18_preds_real_dic[0]["X_temp"])
    x_transformed_18=round(pd.DataFrame(x_transformed_18,columns=iso18_bests_dic[0]["used_features"]), 5)
    x_transformed_2h=iso2h_bests[0][6].inverse_transform(iso2h_preds_real_dic[0]["X_temp"])
    x_transformed_2h=round(pd.DataFrame(x_transformed_2h,columns=iso2h_bests_dic[0]["used_features"]), 5)
    used_feats=list(set(iso18_bests_dic[0]["used_features"]).intersection(iso2h_bests_dic[0]["used_features"]))
    x_transformed=x_transformed_18[used_feats]
    ##########################################################################
    if selection_method=="local_line":
        from pylr2 import regress2 
        #change from linear regression to RMA regression
        reg = regress2(np.array(iso_18_obs).reshape(-1, 1), np.array(iso_2h_obs).reshape(-1, 1), _method_type_2="reduced major axis")
        #print ("Method: RMA", "slope: ",reg['slope'],"intercept: ", reg['intercept'],"r: ",reg['r'])
        #reg = LinearRegression().fit(np.array(iso_18_obs).reshape(-1, 1),np.array(iso_2h_obs).reshape(-1, 1))
    elif selection_method=="global_line":
        from pylr2 import regress2 
        reg = regress2(np.array([0,-b/a]).reshape(-1, 1), np.array([b,0]).reshape(-1, 1), _method_type_2="reduced major axis")
        #print ("Method: RMA", "slope: ",reg['slope'],"intercept: ", reg['intercept'],"r: ",reg['r'])
        #reg = LinearRegression().fit(np.array([0,-b/a]).reshape(-1, 1),np.array([b,0]).reshape(-1, 1)) #meteo line
    '''elif selection_method=="point_to_point":
        x_transformed_18=iso18_bests[0][6].inverse_transform(iso18_preds_real_dic[0]["X_temp"])
        x_transformed_18=round(pd.DataFrame(x_transformed_18,columns=iso18_bests_dic[0]["used_features"]), 5)
        x_transformed_2h=iso2h_bests[0][6].inverse_transform(iso2h_preds_real_dic[0]["X_temp"])
        x_transformed_2h=round(pd.DataFrame(x_transformed_2h,columns=iso2h_bests_dic[0]["used_features"]), 5)
        used_feats=list(set(iso18_bests_dic[0]["used_features"]).intersection(iso2h_bests_dic[0]["used_features"]))
        x_transformed=x_transformed_18[used_feats]'''
        
    best_mod_score_to_meteo=None
    sam_size=all_preds.shape[0]
    for k_18,v_18 in iso18_models.items():
        predicted_18=v_18["predicted_Y"]
        log_18_t=False
        if "log" in k_18:
            predicted_18=np.expm1(predicted_18) #it comes already logged! we exp it
            log_18_t=True
        predicted_18=y_scaler_18.inverse_transform(pd.DataFrame(predicted_18))
        for k_2h,v_2h in iso2h_models.items():
            predicted_2h=v_2h["predicted_Y"]
            log_2h_t=False
            if "log" in k_2h :
                predicted_2h=np.expm1(predicted_2h) #it comes already logged! we exp it
                log_2h_t=True
            predicted_2h=y_scaler_2h.inverse_transform(pd.DataFrame(predicted_2h))
            ########################
            #to make resid report:
            df_18=pd.DataFrame(predicted_18,columns=["pred_18"])
            df_2h=pd.DataFrame(predicted_2h,columns=["pred_2h"])
            preds_df=pd.concat([x_transformed,df_18,df_2h],axis=1)
            merged_df=pd.merge(preds_df,all_preds_,on=used_feats)
            merged_df["score_"]=((merged_df.pred_18-merged_df[x])**2 + (merged_df.pred_2h-merged_df[y])**2)**0.5
            score_for_report=merged_df.score_.mean()


            ########################
            if selection_method in ["local_line","global_line"]:

                
                from sklearn.metrics import r2_score

                predicted_2h_on_the_line=reg['slope']*predicted_18.reshape(-1,1)+reg['intercept']

                mod_score_to_meteo=r2_score(predicted_2h.reshape(-1,1),predicted_2h_on_the_line.reshape(-1,1))

                #print ("mod_score: ",mod_score_to_meteo)
                #mod_score_to_meteo=reg.score(predicted_18.reshape(-1,1),predicted_2h.reshape(-1,1))
            elif selection_method=="point_to_point":
                '''df_18=pd.DataFrame(predicted_18,columns=["pred_18"])
                df_2h=pd.DataFrame(predicted_2h,columns=["pred_2h"])
                preds_df=pd.concat([x_transformed,df_18,df_2h],axis=1)
                merged_df=pd.merge(preds_df,all_preds_,on=used_feats)
                merged_df["score_"]=((merged_df.pred_18-merged_df.iso_18)**2 + (merged_df.pred_2h-merged_df.iso_2h)**2)**0.5
                mod_score_to_meteo=merged_df.score_.mean()'''
                mod_score_to_meteo=score_for_report
                #print (mod_score_to_meteo)
            if (best_mod_score_to_meteo==None and selection_method in ["local_line","global_line","point_to_point"]):

                best_mod_score_to_meteo=mod_score_to_meteo
                best_model_18,log_18,mod_score_18,predicted_18_final=set_vals(v_18,log_18_t,predicted_18)
                best_model_2h,log_2h,mod_score_2h,predicted_2h_final=set_vals(v_2h,log_2h_t,predicted_2h)
                best_score_for_report,best_k_18,best_k_2h=score_for_report,k_18,k_2h
            elif (selection_method in ["local_line","global_line"] and mod_score_to_meteo>best_mod_score_to_meteo) or (selection_method=="point_to_point" and mod_score_to_meteo<best_mod_score_to_meteo):

                best_mod_score_to_meteo=mod_score_to_meteo
                best_model_18,log_18,mod_score_18,predicted_18_final=set_vals(v_18,log_18_t,predicted_18)
                best_model_2h,log_2h,mod_score_2h,predicted_2h_final=set_vals(v_2h,log_2h_t,predicted_2h)
                best_score_for_report,best_k_18,best_k_2h=score_for_report,k_18,k_2h
                #print ("changed!",k_18,k_2h)
            ####################################################
            #independent
            if selection_method=="independent":
                #print (" in independent")

                best_model_18,log_18,mod_score_18,predicted_18_final=set_vals(v_18,log_18_t,predicted_18)
                best_model_2h,log_2h,mod_score_2h,predicted_2h_final=set_vals(v_2h,log_2h_t,predicted_2h)
                best_score_for_report,best_k_18,best_k_2h=score_for_report,k_18,k_2h

                
            


    ##############################################################
    #generate report
    if report==True:
        Path(direc).mkdir(parents=True, exist_ok=True)
        file_name=os.path.join(direc,"isotope_modeling_scoring_function_report_18_2h.txt")
        str_to_write="Formula= Argmin( mean( sqrt(Resid " + x +"^2 + Resid "+ y +"^2) ) )\n\n selection_method: " +selection_method+"\n\nModels:\n\n"+x+":\n\n"+ best_k_18+"\n\n adjrsquared  " + x+":\n\n" + str(mod_score_18)+"\n\n"+y +":\n"+best_k_2h+"\n\n adjrsquared  " + y+":\n\n" + str(mod_score_2h)+"\n\nScore: \n (Less is better)\n\n" + str(best_score_for_report)
        m_out_f=open(file_name,'w')
        m_out_f.write(str_to_write)
        m_out_f.close()

    ##############################################################
    #update data: iso18
    new_results_dic=dict()
    #####

    iso18_bests[0][0]=best_model_18
    num_var=len(iso18_bests_dic[0]["used_features"])
    iso18_bests[0][5]=mod_score_18 #rsquared
    iso18_bests[0][-1]=log_18
    iso18_bests[0][1]=1-(1-mod_score_18)*(sam_size-1)/(sam_size-num_var-1) #bestestimator (adjrsquared)
    
    new_results_dic["bests"]=iso18_bests
    #####
    iso18_preds_real_dic[0]["Y_preds"]=pd.DataFrame( predicted_18_final.reshape(-1, 1),columns=["Value"])
    
    new_results_dic["preds_real_dic"]=iso18_preds_real_dic
    #####

    iso18_bests_dic[0]["best_estimator"]=best_model_18
    iso18_bests_dic[0]["best_score"]=1-(1-mod_score_18)*(sam_size-1)/(sam_size-num_var-1) #best_score_all (adjrsquared)
    iso18_bests_dic[0]["rsquared"]=mod_score_18
    iso18_bests_dic[0]["didlog"]=log_18

    new_results_dic["bests_dic"]=iso18_bests_dic

    for k,v in new_results_dic.items():
        st2_model_results_dic[x][k]=v


    new_results_dic=dict()


    #update data: iso2h
    new_results_dic=dict()
    iso2h_bests[0][0]=best_model_2h
    num_var=len(iso2h_bests_dic[0]["used_features"])
    iso2h_bests[0][5]=mod_score_2h #best_score_all
    iso2h_bests[0][-1]=log_2h
    iso2h_bests[0][1]=1-(1-mod_score_2h)*(sam_size-1)/(sam_size-num_var-1) 
    
    new_results_dic["bests"]=iso2h_bests
    #####

    iso2h_preds_real_dic[0]["Y_preds"]=pd.DataFrame( predicted_2h_final.reshape(-1, 1),columns=["Value"])
    
    new_results_dic["preds_real_dic"]=iso2h_preds_real_dic
    #####

    iso2h_bests_dic[0]["best_estimator"]=best_model_2h
    iso2h_bests_dic[0]["best_score"]=1-(1-mod_score_2h)*(sam_size-1)/(sam_size-num_var-1) #best_score_all
    iso2h_bests_dic[0]["rsquared"]=mod_score_2h
    iso2h_bests_dic[0]["didlog"]=log_2h

    new_results_dic["bests_dic"]=iso2h_bests_dic

    for k,v in new_results_dic.items():
        st2_model_results_dic[y][k]=v

    
    return st2_model_results_dic

                
def st2_output_report(st2_model_results_dic,direc):
    #writing isotope results to a txt file
    from tabulate import tabulate

    Path(direc).mkdir(parents=True, exist_ok=True)
    file_name=os.path.join(direc,"stage_2_modeling_output_report.txt")
    m_out_f=open(file_name,'w')
    m_out_f=open(file_name,'w')

    for k,v in st2_model_results_dic.items():
        iso18_bests=v["bests"]
        iso18_bests_dic=v["bests_dic"]
        pr_is_18="\n################\n\n best_estimator_and_params_all"+str(k)+"\n"+str(iso18_bests_dic[0]["best_estimator"].best_estimator_)+"\n\n"+str(iso18_bests_dic[0]["best_estimator"].best_params_)+"\n\n################\n\n used_features_"+str(k)+"\n"+str(iso18_bests[0][4])+"\n\n################\n\n best_score_all_"+str(k)+"\n"+str(iso18_bests[0][1])+"\n\n################\n\n rsquared_"+str(k)+"\n"+str(iso18_bests[0][5])+"\n\n################\n\n didlog_"+str(k)+"\n"+str(iso18_bests[0][-1])+"\n\n################\n\n VIF_INITIAL_"+str(k)+"\n"+tabulate(iso18_bests_dic[0]["vif_initial"])+"\n\n################\n\n VIF_"+str(k)+"\n"+tabulate(iso18_bests_dic[0]["vif"])+"\n\n################\n\n vif_chosen_features \n"+str(iso18_bests_dic[0]["vif_chosen_features"])+"\n\n################\n\n f_regression_"+str(k)+"\n"+str(iso18_bests_dic[0]["f_regression"])+"\n\n################\n\n correlation_"+str(k)+"\n"+tabulate(iso18_bests_dic[0]["correlation"])+"\n\n#########################\n#########################\n#########################\n"

        m_out_f.write(pr_is_18)
    m_out_f.close()


    #################
    #all models report
    file_name=os.path.join(direc,"stage_2_all_models.txt")
    m_out_f=open(file_name,'w')
    for kn,vn in st2_model_results_dic.items():
        iso18_bests_dic=vn["bests_dic"]
        tp=str()
        for k,v in iso18_bests_dic[0]["models_output_dic"].items():
            try:
                tw="\n\n"+k+":\n\nmodel:\n"+str(v['model'].best_estimator_)+"\n\n"+ str(v['model'].best_params_)+"\n\nmod_score:\n"+str(v["mod_score"])+"\n\nmod_cv_score:\n"+str(v["model"].best_score_)+"\n\n########################"
                tp=tp+tw
            except: 
                #print ("except in iso_output_report function!!")
                pass

        prr_is_18="\n\n################\n\n models_output_dic_"+kn +"\n################\n"+tp+"\n\n#########################\n#########################\n#########################\n"
        m_out_f.write(prr_is_18)

    m_out_f.close()



