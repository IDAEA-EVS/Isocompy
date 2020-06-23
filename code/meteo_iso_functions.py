import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.metrics import max_error,classification_report,confusion_matrix,mean_squared_error,mean_absolute_error
from sklearn.feature_selection import GenericUnivariateSelect, f_regression, chi2,mutual_info_regression
from sklearn.model_selection import GridSearchCV
#from sklearn.exceptions import DataConversionWarning
#warnings.filterwarnings(action='ignore', category=DataConversionWarning)
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression,ElasticNetCV,MultiTaskElasticNetCV,LarsCV, OrthogonalMatchingPursuitCV,BayesianRidge,ARDRegression
from sklearn.inspection import plot_partial_dependence
from sklearn.svm import SVR,NuSVR
from sklearn.linear_model import LinearRegression
from pysplit_funcs_for_meteo_model import convertCoords,gen_trajs
import os
def warn(*args, **kwargs):
    pass
import warnings
import pysplit
warnings.warn = warn


###########################################
###################################################################
def iso_prediction(iso_db1,month_grouped_list_with_zeros_iso_2h,month_grouped_list_with_zeros_iso_3h,temp_bests,rain_bests,hum_bests,iso_18,dates_db,justsum=True):
    
    predictions_monthly_list=list()
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
    for month_num in range(0,len(iso_db1)):
        if (month_num not in [3,4,5,6,7,8,9,11] and justsum ==True) or justsum==False:
            bests_of_all_preds=list()
            col_names=["temp","rain","hum"]
            for bests_of_all, names in zip([temp_bests,rain_bests,hum_bests],col_names):

                #general input transform
                xyz=bests_of_all[month_num][-3].transform(iso_db1[month_num][bests_of_all[month_num][4]])                
                #transform if there is log in input
                if bests_of_all[month_num][-1]==True:
                    xyz=np.log1p(xyz)
                else:
                    pass
                #predict temp
                bests_pred=pd.DataFrame(data=bests_of_all[month_num][0].predict(xyz),columns=[names])
                #inverse transform log output
                if bests_of_all[month_num][-1]==True:
                    bests_pred=np.expm1(bests_pred)

                else:
                    pass
                #inverse transform general
                bests_pred=pd.DataFrame(bests_of_all[month_num][-2].inverse_transform(bests_pred),columns=[names])
            
                bests_of_all_preds.append(bests_pred)
            
            temp_pred=bests_of_all_preds[0]
            rain_pred=bests_of_all_preds[1]
            hum_pred=bests_of_all_preds[2]
            ###################################################################
            ######################
            #### 2/19/2020: Ash
            #here, the new coordination system have to be added from excel or calculated
            xy_df_for_hysplit=iso_db1[month_num][["CooX","CooY","ID_MeteoPoint"]]
            xy_df_for_hysplit = iso_db1[month_num].join(xy_df_for_hysplit.apply(convertCoords,axis=1)) 
            ######################
            preds=pd.concat([iso_db1[month_num][["CooX","CooY","CooZ","ID_MeteoPoint","iso_18"]],month_grouped_list_with_zeros_iso_2h[month_num][["iso_2h"]],month_grouped_list_with_zeros_iso_3h[month_num][["iso_3h"]],temp_pred,rain_pred,hum_pred],axis=1,ignore_index =False)   
            ######################
            #to pysplit trajectories: iterate altitudes
            
            all_hysplit_df_list_all_atts=list()
            all_hysplit_df_list_all_atts_without_averaging=list()
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
            if month_num==0:
                all_preds=preds
                all_without_averaging=dfdf
            else:
                all_preds=pd.concat([all_preds,preds],ignore_index=True)
                all_without_averaging=pd.concat([all_without_averaging,dfdf],ignore_index=True)
            '''except:
                print ("IN except!!!!!!!!!!!!!!!!")
                print (month_num)'''
    
    points_all_in_water_report.close()
    points_origin_not_detected_report.close()
    error_in_meteo_file.close() 
    traj_shorter_than_runtime_report.close()
    input_pass_to_bulkrajfun_report.close()
    return  predictions_monthly_list, all_preds,all_hysplit_df_list_all_atts,col_for_f_reg,all_without_averaging      
#grouping data
def grouping_data(rain,which_value,elnino,lanina,month=None,zeross=True,iso=False):
    if zeross==False:
        rain=rain[rain["Value"]!=0]
    if month !=None:
        rain=rain[rain["DateMeas"].dt.month==month] 

    #omiting the top 5 percent of the data
    if iso==False:
        rain_95=rain[rain["Value"]<rain["Value"].quantile(0.95)]
        if rain_95.shape[0]>=5:
            rain=rain_95
    #########
    stations_rain=rain[['ID_MeteoPoint']]
    stations_rain.drop_duplicates(keep = 'last', inplace = True)
    rainmeteoindex=rain.set_index('ID_MeteoPoint')
    newmat_elnino=list()
    newmat_lanina=list()
    newmat_norm=list()
    for index, row in stations_rain.iterrows():
        tempp=rainmeteoindex.loc[row['ID_MeteoPoint']]
        #print ("tempp",len(tempp.shape))
        if len(tempp.shape)!= 1:
            #print (tempp)
            sum_elnino=None
            cnt_elnino=None
            sum_lanina=None
            cnt_lanina=None
            sum_norm=None
            cnt_norm=None
            for index2, row2 in tempp.iterrows():
                if pd.to_datetime(row2["DateMeas"]).year in elnino:
                    if sum_elnino==None:
                        sum_elnino=row2["Value"]
                        cnt_elnino=1
                    else:
                        sum_elnino=sum_elnino+row2["Value"]
                        cnt_elnino=cnt_elnino+1

                elif pd.to_datetime(row2["DateMeas"]).year in lanina:
                    if sum_lanina==None:
                        sum_lanina=row2["Value"]
                        cnt_lanina=1
                    else:
                        sum_lanina=sum_lanina+row2["Value"]
                        cnt_lanina=cnt_lanina+1
                
                else:
                    if sum_norm==None:
                        sum_norm=row2["Value"]
                        cnt_norm=1
                    else:
                        sum_norm=sum_norm+row2["Value"]
                        cnt_norm=cnt_norm+1

            if  sum_elnino !=None:       
                Mean_Value_elnino=sum_elnino/cnt_elnino
                newmat_elnino.append({"ID_MeteoPoint":row['ID_MeteoPoint'], "CooX":tempp["CooX"].iat[0], "CooY":tempp["CooY"].iat[0], "CooZ":tempp["CooZ"].iat[0], "Date":tempp["DateMeas"], which_value:Mean_Value_elnino})
            if sum_lanina !=None:
                Mean_Value_lanina=sum_lanina/cnt_lanina
                newmat_lanina.append({"ID_MeteoPoint":row['ID_MeteoPoint'], "CooX":tempp["CooX"].iat[0], "CooY":tempp["CooY"].iat[0], "CooZ":tempp["CooZ"].iat[0], which_value:Mean_Value_lanina})
            if sum_norm !=None:
                Mean_Value_norm=sum_norm/cnt_norm
                newmat_norm.append({"ID_MeteoPoint":row['ID_MeteoPoint'], "CooX":tempp["CooX"].iat[0], "CooY":tempp["CooY"].iat[0], "CooZ":tempp["CooZ"].iat[0], which_value:Mean_Value_norm})
        else:
            #print ("tempp in len 1:")
            #print (tempp)
            if pd.to_datetime(tempp["DateMeas"]).year in elnino:
                newmat_elnino.append({"ID_MeteoPoint":row['ID_MeteoPoint'], "CooX":tempp["CooX"], "CooY":tempp["CooY"], "CooZ":tempp["CooZ"], which_value:tempp["Value"]})
            elif pd.to_datetime(tempp["DateMeas"]).year in lanina:
                newmat_lanina.append({"ID_MeteoPoint":row['ID_MeteoPoint'], "CooX":tempp["CooX"], "CooY":tempp["CooY"], "CooZ":tempp["CooZ"], which_value:tempp["Value"]})    
            else:
                newmat_norm.append({"ID_MeteoPoint":row['ID_MeteoPoint'], "CooX":tempp["CooX"], "CooY":tempp["CooY"], "CooZ":tempp["CooZ"], which_value:tempp["Value"]})
    
    newmatdf_rain_elnino = pd.DataFrame(newmat_elnino)
    newmatdf_rain_lanina = pd.DataFrame(newmat_lanina)
    newmatdf_rain_norm = pd.DataFrame(newmat_norm)
    return newmatdf_rain_elnino,newmatdf_rain_lanina,newmatdf_rain_norm
###########################################
#model!
def rfmethod(tunedpars,gridsearch_dictionary,newmatdf_temp,temp_rain_hum,monthnum,model_type,meteo_or_iso,inputs=None):
    if meteo_or_iso=="meteo":
        X_temp=newmatdf_temp[["CooX","CooY","CooZ"]].copy().astype(float)
        num_var=3
        colmnss=["CooX","CooY","CooZ"]
    ##################################################
    if meteo_or_iso=="iso":
        X_temp=newmatdf_temp[inputs].copy().astype(float)
        num_var=len(inputs)
        colmnss=inputs
        monthnum=0
    Y_temp=newmatdf_temp[[temp_rain_hum]].copy()
    X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(X_temp, Y_temp)    
    ##################################################
    #adjusted r squared
    sam_size=newmatdf_temp.shape[0]
    adj_r_sqrd=lambda r2,sam_size=sam_size,num_var=num_var: 1-(1-r2)*(sam_size-1)/(sam_size-num_var-1)
    ##################################################
    #some tests:
    mutual_info_regression_value_rain = mutual_info_regression(X_temp, Y_temp)
    mutual_info_regression_value_rain /= np.max(mutual_info_regression_value_rain)
    print ("mutual_info_regression_value on whole data, standard!")
    print (mutual_info_regression_value_rain)
    f_regression_value_rain=f_regression(X_temp, Y_temp)
    print ("f_regression_value on whole data not standard")
    print (f_regression_value_rain)
    ##################################################
    #just using the significant variables!
    f_less_ind=list(np.where(f_regression_value_rain[1]<=0.05)[0])
    if len(f_less_ind)<2:
        f_regression_value_rain_sor=sorted(f_regression_value_rain[1])[:2]
        f_less_ind1=list(
            (np.where
            (f_regression_value_rain[1]==f_regression_value_rain_sor[0])
                            ) [0]
        )
        f_less_ind2=list((np.where
            (f_regression_value_rain[1]==f_regression_value_rain_sor[1])
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
    #print (X_temp)
    #print (Y_temp)
    ########################################################################
    '''best_score_all=-10
    rsquared=-10
    elastic=False
    didlog=False'''
    #RandomForestRegressor
    

    estrandomfor_temp=GridSearchCV(RandomForestRegressor(random_state =0), tunedpars, cv=10,n_jobs=-1)
    estrandomfor_temp.fit(X_temp, Y_temp.ravel())
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
    ########################################################################
    #multivariate linear regression on log data
    X_temp1=np.log1p(X_temp)
    Y_temp1=np.log1p(Y_temp)
    reg = MultiTaskElasticNetCV(l1_ratio=[.1, .3, .5, .6, .7,.8,.85,.87,.9,.93,.95,.97,0.99],n_jobs =-1,cv=10,normalize=False ).fit(X_temp1, Y_temp1)
    elastic=False
    didlog=False
    if adj_r_sqrd(reg.score(X_temp1, Y_temp1))>best_score_all:
        best_score_all=adj_r_sqrd(reg.score(X_temp1, Y_temp1))
        best_estimator_all=reg
        didlog=True
        elastic=True
        rsquared=reg.score(X_temp1, Y_temp1)
        print ("################################\n THE LOG ELASTIC IS THE BEST! \n ################################")
    ####################################################################
    #LarsCV
    reg =LarsCV(cv=10,n_jobs=-1).fit(X_temp, Y_temp)
    print ("LARS")
    print ("SCORE:\n",adj_r_sqrd(reg.score(X_temp, Y_temp)) )
    if adj_r_sqrd(reg.score(X_temp, Y_temp))>best_score_all:
        best_score_all=adj_r_sqrd(reg.score(X_temp, Y_temp))
        best_estimator_all=reg
        rsquared=reg.score(X_temp, Y_temp)
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
    #svm
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
        rsquared=reg.score(X_temp, Y_temp)
    ####################################################################   ''' 
    print ("Best score total adj r square:")
    print (best_score_all)
    print ("sample size,num of variables:\n",sam_size,num_var)
    print ("Best score total r square:")
    print (rsquared)
    print ("Best estimator total:")
    print (best_estimator_all)
    lens=len(f_less_ind)
    f_namess=used_features
    if meteo_or_iso=="meteo":
        pltttl="month_"+ str(monthnum) +"_All_data_best_estimator_"+model_type
    else:
        pltttl="Annual_iso_All_data_best_estimator_"+model_type

    pltname="C:\\Users\\Ash kan\\Documents\\meteo_iso_model\\meteo_iso_model_input_code_and_results\\output\\model_plots\\"+model_type+"\\"+pltttl+'.pdf'
    pltnamepardep="C:\\Users\\Ash kan\\Documents\\meteo_iso_model\\meteo_iso_model_input_code_and_results\\output\\model_plots\\"+model_type+"\\partial_dependency\\"+pltttl+'_partial_dependence'+'.pdf'
    if elastic==True:
        X_temp_fin=X_temp1
        Y_temp_fin=Y_temp1
        Y_preds_output = best_estimator_all.predict(X_temp_fin)
        #transform log
        Y_preds=np.log1p(Y_preds_output)
    else:
        X_temp_fin=X_temp
        Y_temp_fin=Y_temp
        Y_preds_output = best_estimator_all.predict(X_temp_fin)
        Y_preds=Y_preds_output.copy()
    #general standard
    Y_preds=y_scaler.inverse_transform( Y_preds.reshape(-1, 1) )
    clnm=model_type+"_"+str(monthnum)
    Y_preds=pd.DataFrame(Y_preds,columns=[clnm])
    plt.scatter(Y_preds_output,Y_temp_fin)
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
    plot_partial_dependence(best_estimator_all, X_temp_fin, features=list(range(0,lens)),feature_names=f_namess)
    plt.savefig(pltnamepardep,dpi=300)
    #plt.show()
    plt.close()
    #some old lines
    """
        #print ("RF cv results")
        #print (pd.DataFrame(estrandomfor_temp.cv_results_ ))
        #scrf=estrandomfor_temp.score(X_test_temp,y_test_temp)
        '''print ("random forest regressor r2 score:")
        print (temp_rain_hum,estrandomfor_temp.score(X_test_temp,y_test_temp))
        predrf_temp=estrandomfor_temp.predict(X_test_temp)
        print ("mean square error:")
        print (temp_rain_hum,mean_squared_error(y_test_temp, predrf_temp))
        print ("mean abs error:",temp_rain_hum,mean_absolute_error(y_test_temp, predrf_temp))
        print ("max_error",temp_rain_hum,max_error(y_test_temp, predrf_temp))'''
        #estrandomfor_temp.fit(X_temp, Y_temp)
        #cross validation
        #scores = cross_val_score(estrandomfor_temp, X_temp, Y_temp, cv=5,n_jobs =-1)
        #print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        #############################################################
        #plotting random forest results for test dataset against real values
        #plt.scatter(predrf[:,0],y_test["Mean_Value_rain"])
        '''plt.scatter(predrf_temp,y_test_temp[temp_rain_hum])
        plt.title(" RF test data results")
        plt.xlabel("Prediction")
        plt.ylabel("Real Value")
        left, right = plt.xlim()
        left1, right1 = plt.ylim()
        a = np.linspace(0,max(right,right1),100)
        b=a
        plt.plot(a,b)
        plt.show()'''
        #print ("importance of each feature in RF regression:",temp_rain_hum)
        #print ("[X,  Y,  Z]:",estrandomfor_temp.feature_importances_)
        #############################################################
        #############################################################
        '''print ("MLP r2 score for best estimator:")
        print (mlp_temp.score(X_test_temp,y_test_temp))
        predmlp_temp=mlp_temp.predict(X_test_temp)
        print ("mean square error:")
        print (temp_rain_hum,mean_squared_error(y_test_temp, predmlp_temp))
        print ("mean abs error:",temp_rain_hum,mean_absolute_error(y_test_temp, predmlp_temp))
        print ("max_error",temp_rain_hum,max_error(y_test_temp, predmlp_temp))
        #cross validation
        #scores = cross_val_score(mlp_temp, X_temp, Y_temp, cv=5,n_jobs =-1)
        #print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        print("Mean cross-validated of best_score_",mlp_temp.best_score_) 
        print("best_params_",mlp_temp.best_params_ )
        print("best_estimator_",mlp_temp.best_estimator_) 
        #plot mlp
        plt.scatter(predmlp_temp,y_test_temp)
        plt.title(" MLP test data results")
        plt.xlabel("Prediction")
        plt.ylabel("Real Value")
        left, right = plt.xlim()
        left1, right1 = plt.ylim()
        a = np.linspace(0,max(right,right1),100)
        b=a
        plt.plot(a,b)
        plt.show()'''
    """
    #############################################################
    #############################################################
    return Y_preds,X_temp_fin ,newmatdf_temp[[temp_rain_hum]].copy(),X_train_temp, X_test_temp, y_train_temp, y_test_temp,best_estimator_all,best_score_all,mutual_info_regression_value_rain,f_regression_value_rain,x_scaler,y_scaler,didlog,used_features,rsquared 
###########################################################
#pca function
def pcafun(pca_raw_df,kmeans_group_nums=5,filetitlename="no_name"):
    #scaling
    scaler1 = MinMaxScaler()
    scaler1.fit(pca_raw_df)
    pca_raw_df_st = scaler1.transform(pca_raw_df)
    #############################################################
    #Applying PCA
    pca=PCA()
    pcaf=pca.fit(pca_raw_df_st).transform(pca_raw_df_st)
    print ("Variance ratio:", pd.DataFrame(pca.explained_variance_ratio_ ))
    print ("PCA1+PCA2 variance ratio:",pca.explained_variance_ratio_[0]+pca.explained_variance_ratio_[1])
    print ("loadings")
    print (pd.DataFrame(pca.components_ * np.sqrt(pca.explained_variance_)))
    ##############
    #kmeans
    #kmeans_2d = KMeans(n_clusters=kmeans_group_nums, random_state=0).fit(pcaf[:,:2])
    #plot
    pltname="C:\\Users\\Ash kan\\Documents\\meteo_iso_model\\meteo_iso_model_input_code_and_results\\output\\pca_plots"+filetitlename+'.pdf'
    plt.scatter(pcaf[:,0],pcaf[:,1])
    #plt.scatter(kmeans_2d.cluster_centers_[:,0],kmeans_2d.cluster_centers_[:,1])
    plt.title(str(pca_raw_df.columns))
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title(str(pca_raw_df.columns))
    plt.savefig(pltname,dpi=300)
    #plt.show()
    plt.close()
    return pcaf
###########################################################
#function for monthly procedure
def monthly_uniting(which_value,datab,iso=False):
    elnino=list()
    lanina=list()
    datab.rename(columns={"Z":"CooZ"},inplace=True)
    month_grouped_list_with_zeros=list()
    month_grouped_list_without_zeros=list()
    for month in range(1,13):
        rain_cop=datab.copy()
        #with zeros
        newmatdf_rain_elnino,newmatdf_rain_lanina,newmatdf_rain_all2=grouping_data(rain_cop,which_value,elnino,lanina,iso=iso,month=month,zeross=True)
        month_grouped_list_with_zeros.append(newmatdf_rain_all2)
        rain_cop=datab.copy()
        #without zeros
        newmatdf_rain_elnino,newmatdf_rain_lanina,newmatdf_rain_all=grouping_data(rain_cop,which_value,elnino,lanina,iso=iso,month=month,zeross=False)
        month_grouped_list_without_zeros.append(newmatdf_rain_all)
    return    month_grouped_list_with_zeros,month_grouped_list_without_zeros
###########################################################
#importing_preprocess
def importing_preprocess():
    #######################
    #importing data
    data_file = r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\inputs\Meteorological_Data_25_02_UTM.xlsx"
    rain = pd.read_excel(data_file,sheet_name="METEO_CHILE_Rain_All",header=0,index_col=False,keep_default_na=True)
    temper=pd.read_excel(data_file,sheet_name="METEO_CHILE_Temp",header=0,index_col=False,keep_default_na=True)
    hum=pd.read_excel(data_file,sheet_name="METEO_CHILE_HR",header=0,index_col=False,keep_default_na=True)
    ###########################################################
    ###########################################################
    #nino nina processing
    elnino=list()
    lanina=list()
    '''for index, row in NINO_NINA.iterrows():
        if row["WINTER"] == "EN" and row["year"] not in elnino_winter:
            elnino_winter.append(row["year"])
        elif row["WINTER"] == "LN" and row["year"] not in elnino_winter:
            lanina_winter.append(row["year"])
    
        if row["SUMMER"] == "EN" and row["year"] not in elnino_summer:
            elnino_summer.append(row["year"])
        elif row["SUMMER"] == "LN" and row["year"] not in lanina_summer:
            lanina_summer.append(row["year"])  '''          
    #############################################################
    #Group the rain data to average of each station
    #rain
    which_value="Mean_Value_rain"
    datab=rain
    month_grouped_list_with_zeros_rain,month_grouped_list_without_zeros_rain=monthly_uniting(which_value,datab)
    #Group the temperature data to average of each station
    which_value="Mean_Value_temp"
    datab=temper
    month_grouped_list_with_zeros_temp,month_grouped_list_without_zeros_temp=monthly_uniting(which_value,datab)
    #Group the humidity data to average of each station
    which_value="Mean_Value_hum"
    datab=hum
    month_grouped_list_with_zeros_hum,month_grouped_list_without_zeros_hum=monthly_uniting(which_value,datab)
    ############################################################
    ############################################################
    #isotope file preprocess
    data_file_iso = r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\inputs\pysplit_isotope_registro_Pp_02_05_2020.xlsx"
    iso_18 = pd.read_excel(data_file_iso,sheet_name="ISOT18O",header=0,index_col=False,keep_default_na=True)
    iso_2h=pd.read_excel(data_file_iso,sheet_name="ISOT2H",header=0,index_col=False,keep_default_na=True)
    iso_3h=pd.read_excel(data_file_iso,sheet_name="ISOT3",header=0,index_col=False,keep_default_na=True)
    #############################################################
    iso_18.rename(columns={"IdPoint":"ID_MeteoPoint","DateSamp":"DateMeas"},inplace=True)
    iso_2h.rename(columns={"IdPoint":"ID_MeteoPoint","DateSamp":"DateMeas"},inplace=True)
    iso_3h.rename(columns={"IdPoint":"ID_MeteoPoint","DateSamp":"DateMeas"},inplace=True)
    iso_18['CooX_in']=iso_18["CooX"]
    iso_2h['CooX_in']=iso_2h["CooX"]
    iso_3h['CooX_in']=iso_3h["CooX"]
    #############################################################
    which_value="iso_18"
    datab=iso_18
    #newmatdf_iso_18_elnino,newmatdf_iso_18_lanina,newmatdf_iso_18_norm=grouping_data(iso_18,which_value,elnino,lanina)
    month_grouped_list_with_zeros_iso_18,month_grouped_list_without_zeros_iso_18=monthly_uniting(which_value,datab,iso=True)
    #############################################################
    which_value="iso_2h"
    datab=iso_2h
    #newmatdf_iso_2h_elnino,newmatdf_iso_2h_lanina,newmatdf_iso_2h_norm=grouping_data(iso_2h,which_value,elnino,lanina)
    month_grouped_list_with_zeros_iso_2h,month_grouped_list_without_zeros_iso_2h=monthly_uniting(which_value,datab,iso=True)
    #############################################################
    which_value="iso_3h"
    datab=iso_3h
    #newmatdf_iso_3h_elnino,newmatdf_iso_3h_lanina,newmatdf_iso_3h_norm=grouping_data(iso_3h,which_value,elnino,lanina)
    month_grouped_list_with_zeros_iso_3h,month_grouped_list_without_zeros_iso_3h=monthly_uniting(which_value,datab,iso=True)
    #return month_grouped_list_with_zeros_rain,month_grouped_list_without_zeros_rain,month_grouped_list_with_zeros_temp,month_grouped_list_without_zeros_temp,rain,temper,elnino,lanina,newmatdf_rain_elnino,newmatdf_rain_lanina,newmatdf_temp_elnino,newmatdf_temp_lanina,newmatdf_temp_norm,iso_18,iso_2h,iso_3h,newmatdf_iso_18_elnino,newmatdf_iso_18_lanina,newmatdf_iso_18_norm,newmatdf_iso_2h_elnino,newmatdf_iso_2h_lanina,newmatdf_iso_2h_norm,newmatdf_iso_3h_elnino,newmatdf_iso_3h_lanina,newmatdf_iso_3h_norm
    #############################################################
    #2020 datadb
    try:
        data_file_iso = r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\inputs\Date_Rain_02_05_2020.xlsx"
        dates_db = pd.read_excel(data_file_iso,sheet_name="traj_samp_date",header=0,index_col=False,keep_default_na=True)
    except:
        dates_db=None
    return month_grouped_list_with_zeros_iso_18,month_grouped_list_without_zeros_iso_18,month_grouped_list_with_zeros_iso_2h,month_grouped_list_without_zeros_iso_2h,month_grouped_list_with_zeros_iso_3h,month_grouped_list_without_zeros_iso_3h,month_grouped_list_with_zeros_hum,month_grouped_list_without_zeros_hum,month_grouped_list_with_zeros_rain,month_grouped_list_without_zeros_rain,month_grouped_list_with_zeros_temp,month_grouped_list_without_zeros_temp,rain,temper,elnino,lanina,iso_18,iso_2h,iso_3h,dates_db
###################################################################
#print the models results in a file
def print_to_file(file_name,temp_bests,rain_bests,hum_bests):
    m_out_f=open(file_name,'w')
    feat_name=["TEMPERATURE","HUMIDITY","PRECIPITATION"]
    cnt=0
    for featt in [temp_bests,hum_bests,rain_bests]:
        for monthnum in range(0,len(featt)):
            title=feat_name[cnt]+'\nmonth\n'
            m_out_f.write(title)
            m_out_f.write(str(monthnum+1))
            m_out_f.write('\n\n\n Used features \n')
            m_out_f.write(str(featt[monthnum][4]))
            m_out_f.write('\n\n BEST SCORE adjusted R squared \n')
            m_out_f.write(str(featt[monthnum][1]))
            m_out_f.write('\n\n BEST SCORE R squared \n')
            m_out_f.write(str(featt[monthnum][5]))
            m_out_f.write('\n\n F REGRESSION \n')
            m_out_f.write(str(featt[monthnum][3][0]))
            m_out_f.write('\n')
            m_out_f.write(str(featt[monthnum][3][1]))
            m_out_f.write('\n\n MUTUAL INFORMATION REGRESSION \n')
            m_out_f.write(str(featt[monthnum][2]))
            m_out_f.write('\n\n BEST ESTIMATOR \n')
            m_out_f.write(str(featt[monthnum][0]))
            m_out_f.write('\n\nlog(1 + x)? \n')
            m_out_f.write(str(featt[monthnum][-1]))
            m_out_f.write('\n')
            m_out_f.write("########################################################\n")
        m_out_f.write('\n"########################################################\n')
        cnt=cnt+1  
    m_out_f.close() 
    return       
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
#predict points for contouring
def predict_points(used_features_iso18,x_y_z_,no_needed_month,temp_bests,rain_bests,hum_bests,x_scaler_iso18,y_scaler_iso18,didlog_iso18,best_estimator_all_iso18,column_name):

    x_y_z_org=x_y_z_.drop(["FID"],axis=1)
    x_y_z_copy=x_y_z_org.copy()
    monthly_iso_output=list()
    for month in range(0,12):
        if month not in no_needed_month:
            counterr=0
            for meteopredict in [temp_bests,rain_bests,hum_bests]:
                x_y_z=x_y_z_org[meteopredict[month][4]].copy()
                counterr=counterr+1
                #general standard
                x_y_z=meteopredict[month][-3].transform(x_y_z)
                #transform if there is log in input
                if meteopredict[month][-1]==True:
                    x_y_z=np.log1p(x_y_z)
                #predicting    
                meteopredict_res=meteopredict[month][0].predict(x_y_z)
                #inverse transform
                #log
                if meteopredict[month][-1]==True:
                    meteopredict_res=np.expm1(meteopredict_res)
                #general
                if counterr==1:
                    colname="temp"
                elif counterr==2:
                    colname="rain"
                elif counterr==3:
                    colname="hum"
                meteopredict_res=pd.DataFrame(meteopredict[month][-2].inverse_transform(pd.DataFrame(meteopredict_res)),columns=[colname])
                #making the dataframe
                if counterr==1:
                    meteopredict_res_per_month=meteopredict_res
                else:
                    meteopredict_res_per_month=pd.concat([meteopredict_res_per_month,meteopredict_res],axis=1)
            #################################################
            #################################################
            iso_model_input=pd.concat([x_y_z_copy,meteopredict_res_per_month],axis=1)
            #transforming
            iso_model_input=x_scaler_iso18.transform(iso_model_input[used_features_iso18])
            if didlog_iso18==True:
                iso_model_input=np.expm1(iso_model_input)
            #predicting
            each_month_iso_predict=best_estimator_all_iso18.predict(iso_model_input)
            #inverse transform
            #log
            if didlog_iso18==True:
                each_month_iso_predict=np.expm1(each_month_iso_predict)
            #general
            each_month_iso_predict=pd.DataFrame(y_scaler_iso18.inverse_transform(pd.DataFrame(each_month_iso_predict)),columns=[column_name])

            df_to_excel=pd.concat([x_y_z_copy,meteopredict_res_per_month,each_month_iso_predict],axis=1)
            
            addd="C:\\Users\\Ash kan\\Documents\\meteo_iso_model\\meteo_iso_model_input_code_and_results\\output\\iso_excel_out\\"+"out_iso_month_"+str(month+1)+"_"+column_name+".xls"
            df_to_excel.to_excel(addd)
            monthly_iso_output.append(df_to_excel)


    return monthly_iso_output
    
###########################################################
    
def regional_mensual_plot(x_y_z_,monthly_iso18_output,monthly_iso2h_output):
    for feat in ["CooY"]:

        ymax=x_y_z_[feat].max()
        ymin=x_y_z_[feat].min()
        ydis=ymax-ymin
        m=["x","o","*","s","v","^"]
        coll=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        mon_name=["Jan","Feb","Mar","Apr","Nov","Dec"]
        regmean=0
        regcnt=0
        regcept=0
        for mon in range(0,len(monthly_iso18_output)):


            region1_iso18=monthly_iso18_output[mon][monthly_iso18_output[mon][feat]<ymin+(ydis/3)]
            region1_iso2h=monthly_iso2h_output[mon][monthly_iso2h_output[mon][feat]<ymin+(ydis/3)]
            reg1 = LinearRegression().fit(np.array(region1_iso18["predicted_iso18"]).reshape(-1, 1), np.array(region1_iso2h["predicted_iso2h"]).reshape(-1, 1))
            reg1_slope=reg1.coef_
            regmean +=reg1_slope
            regcept +=reg1.intercept_
            regcnt +=1
            print (reg1_slope)
            print (reg1.intercept_)
            reg1.score(np.array(region1_iso18["predicted_iso18"]).reshape(-1, 1), np.array(region1_iso2h["predicted_iso2h"]).reshape(-1, 1))
            plt.scatter(region1_iso18["predicted_iso18"].mean(), region1_iso2h["predicted_iso2h"].mean(),marker=m[mon],c=coll[mon],label=mon_name[mon])
            '''a = np.linspace(-10,0)
            b=7.54*a+8
            plt.plot(a,b)
            plt.show()
            for mon in range(0,len(monthly_iso18_output)):'''
            region2_iso18=monthly_iso18_output[mon][(monthly_iso18_output[mon][feat]>ymin+(ydis/3) ) & (monthly_iso18_output[mon]["CooY"]<ymin+(ydis*2/3))]
            region2_iso2h=monthly_iso2h_output[mon][(monthly_iso2h_output[mon][feat]>ymin+(ydis/3) ) & (monthly_iso2h_output[mon]["CooY"]<ymin+(ydis*2/3))]
            reg2 = LinearRegression().fit(np.array(region2_iso18["predicted_iso18"]).reshape(-1, 1), np.array(region2_iso2h["predicted_iso2h"]).reshape(-1, 1))
            reg2_slope = reg2.coef_
            regmean +=reg2_slope
            regcept +=reg2.intercept_
            regcnt +=1
            print (reg2_slope)
            print (reg2.intercept_)
            reg2.score(np.array(region2_iso18["predicted_iso18"]).reshape(-1, 1), np.array(region2_iso2h["predicted_iso2h"]).reshape(-1, 1))
            plt.scatter(region2_iso18["predicted_iso18"].mean(), region2_iso2h["predicted_iso2h"].mean(),marker=m[mon],c=coll[mon])
            '''a = np.linspace(-10,0)
            b=7.54*a+8
            plt.plot(a,b)
            plt.show()
            plt.show()
            for mon in range(0,len(monthly_iso18_output)):'''
            region3_iso18=monthly_iso18_output[mon][monthly_iso18_output[mon][feat]>ymin+(ydis*2/3)]
            region3_iso2h=monthly_iso2h_output[mon][monthly_iso2h_output[mon][feat]>ymin+(ydis*2/3)]
            reg3 = LinearRegression().fit(np.array(region3_iso18["predicted_iso18"]).reshape(-1, 1), np.array(region3_iso2h["predicted_iso2h"]).reshape(-1, 1))
            reg3_slope=reg3.coef_
            regmean +=reg3_slope
            regcept +=reg3.intercept_
            regcnt +=1
            print (reg3_slope)
            print (reg3.intercept_)
            reg3.score(np.array(region3_iso18["predicted_iso18"]).reshape(-1, 1), np.array(region3_iso2h["predicted_iso2h"]).reshape(-1, 1))
            plt.scatter(region3_iso18["predicted_iso18"].mean(), region3_iso2h["predicted_iso2h"].mean(),marker=m[mon],c=coll[mon])
        a = np.linspace(-10,0)
        print ((regmean/regcnt)[0],(regcept/regcnt)[0])
        
        b2=8*a+10
        b1=(regmean/regcnt)[0]*a+(regcept/regcnt)[0]
        
        plt.plot(a,b1,label=str(round((regmean/regcnt)[0][0],2))+"*a+"+str( round(( regcept/regcnt )[0],2))  )
        plt.plot(a,b2,label='8a+10')
        plt.legend()
        plt.savefig(r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\output\model_plots\Iso_monthly_graph.pdf",dpi=300)
        plt.close()


        #not 3 zone, manual!!
        #read points for contour
        '''data_file = "C:\\Users\\Ash kan\\Documents\\meteo_iso_model\\meteo_iso_model_input_code_and_results\\inputs\\Tarapaca.xlsx"
        x_y_z_=pd.read_excel(data_file,sheet_name=0,header=0,index_col=False,keep_default_na=True)
        no_needed_month=[4,5,6,7,8,9]
        column_name="predicted_iso18"
        monthly_iso18_output_sala_de_ata=predict_points(used_features_iso18,x_y_z_,no_needed_month,temp_bests,rain_bests,hum_bests,x_scaler_iso18,y_scaler_iso18,didlog_iso18,best_estimator_all_iso18,column_name)
        column_name="predicted_iso2h"
        monthly_iso2h_output_sala_de_ata=predict_points(used_features_iso2h,x_y_z_,no_needed_month,temp_bests,rain_bests,hum_bests,x_scaler_iso2h,y_scaler_iso2h,didlog_iso2h,best_estimator_all_iso2h,column_name)
        ############################################################
        m=["x","o","*","s","v","^"]
        coll=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        mon_name=["Jan","Feb","Mar","Apr","Nov","Dec"]
        for mon in range(0,len(monthly_iso18_output_sala_de_ata)):
            region1_iso18=monthly_iso18_output_sala_de_ata[mon]
            region1_iso2h=monthly_iso2h_output_sala_de_ata[mon]
            #plt.scatter(region1_iso18["predicted_iso18"], region1_iso2h["predicted_iso2h"],marker=m[mon],c=coll[mon],label=mon_name[mon])
            plt.scatter(region1_iso18["predicted_iso18"].mean(), region1_iso2h["predicted_iso2h"].mean(),marker=m[mon],c=coll[mon],label=mon_name[mon])
            #reg1 = LinearRegression().fit(np.array(region1_iso18["predicted_iso18"]).reshape(-1, 1), np.array(region1_iso2h["predicted_iso2h"]).reshape(-1, 1))
        a = np.linspace(-6.5,-4)
        b=reg1.coef_[0]*a+reg1.intercept_[0]
        plt.plot(a,b,label=str(round(reg1.coef_[0][0],2))+"*a+"+str( round(reg1.intercept_[0],2))  )
        plt.legend()
        #plt.savefig(("C:\\Users\\Ash kan\\Documents\\meteo_iso_model\\meteo_iso_model_input_code_and_results\\output\\iso_excel_out"+"\\"+mon_name[mon]+".pdf"),dpi=300)
        plt.show()'''
############################################################

# a function to read the new data (input: x y z month/ output:temp, rain, hum, )
def new_data_prediction_comparison(newd,no_needed_month,temp_bests,rain_bests,hum_bests,didlog_iso18,didlog_iso2h,x_scaler_iso18,x_scaler_iso2h,used_features_iso18,used_features_iso2h,best_estimator_all_iso18,best_estimator_all_iso2h,y_scaler_iso18,y_scaler_iso2h):
    x_y_z_month=newd[["CooX","CooY","CooZ","month"]]
    #iso_18_2h=newd[["iso18","iso2h"]]
    x_y_z_month_org=x_y_z_month.copy()
    #making a list of existed month
    existed_month=x_y_z_month[['month']]
    existed_month.drop_duplicates(keep = 'last', inplace = True)
    counter_month=0
    for month in range(0,12):
        if (month not in no_needed_month) and (existed_month.isin([month+1]).any().bool()==True):
            
            counterr=0
            for meteopredict in [temp_bests,rain_bests,hum_bests]:
                #just selecting the data for 1 month
                x_y_z_month_org_2=x_y_z_month_org[x_y_z_month_org["month"]==month+1].copy()
                #selecting coords
                x_y_z=x_y_z_month_org_2[meteopredict[month][4]].copy()
                counterr=counterr+1
                #general standard
                x_y_z=meteopredict[month][-3].transform(x_y_z)
                #transform if there is log in input
                if meteopredict[month][-1]==True:
                    x_y_z=np.log1p(x_y_z)
                #predicting
                meteopredict_res=meteopredict[month][0].predict(x_y_z)
                print ("meteopredict_res",meteopredict_res)
                #inverse transform
                #log
                if meteopredict[month][-1]==True:
                    meteopredict_res=np.expm1(meteopredict_res)
                #general
                if counterr==1:
                    colname="temp"
                elif counterr==2:
                    colname="rain"
                elif counterr==3:
                    colname="hum"
                meteopredict_res=pd.DataFrame(meteopredict[month][-2].inverse_transform(pd.DataFrame(meteopredict_res)),columns=[colname])
                #making the dataframe
                if counterr==1:
                    meteopredict_res_per_month=meteopredict_res
                else:
                    meteopredict_res_per_month=pd.concat([meteopredict_res_per_month,meteopredict_res],axis=1)
                    #meteopredict_res_per_month=temp,rain,hum
            #################################################
            #################################################
            iso_model_input=pd.concat([x_y_z_month_org_2.reset_index(),meteopredict_res_per_month.reset_index()],axis=1)
            
            #here I have to add trajectories:
            #if trajs are needed:
                #gen traj:
                #3 report file
            #    gen_trajs()    
                #add traj to iso_model_input

            #iso18:

            #transforming
            iso18_model_input=x_scaler_iso18.transform(iso_model_input[used_features_iso18])
            if didlog_iso18==True:
                iso18_model_input=np.expm1(iso18_model_input)
            #predicting
            each_month_iso18_predict=best_estimator_all_iso18.predict(iso18_model_input)
            #inverse transform
            #log
            if didlog_iso18==True:
                each_month_iso18_predict=np.expm1(each_month_iso18_predict)
            #general
            each_month_iso18_predict=pd.DataFrame(y_scaler_iso18.inverse_transform(pd.DataFrame(each_month_iso18_predict)),columns=["iso_18"])
            #################################################
            #iso2h:

            #transforming
            iso2h_model_input=x_scaler_iso2h.transform(iso_model_input[used_features_iso2h])
            if didlog_iso2h==True:
                iso2h_model_input=np.expm1(iso2h_model_input)
            #predicting
            each_month_iso2h_predict=best_estimator_all_iso2h.predict(iso2h_model_input)
            #inverse transform
            #log
            if didlog_iso2h==True:
                each_month_iso2h_predict=np.expm1(each_month_iso2h_predict)
            #general
            each_month_iso2h_predict=pd.DataFrame(y_scaler_iso2h.inverse_transform(pd.DataFrame(each_month_iso2h_predict)),columns=["iso_2h"])
            #################################################
            df_to_excel=pd.concat([x_y_z_month_org_2.reset_index(),meteopredict_res_per_month.reset_index(),each_month_iso18_predict.reset_index(),each_month_iso2h_predict.reset_index()],axis=1)
            
            if counter_month==0:
                integr_df=df_to_excel
            else:
                integr_df=pd.concat([integr_df,df_to_excel])
            counter_month=counter_month+1    
            print ("##############\n integr_df last \n",integr_df)    
    integr_df=integr_df[["CooX","CooY","CooZ","temp","rain","hum","iso_18","iso_2h","month"]].reset_index()
    addd="C:\\Users\\Ash kan\\Documents\\meteo_iso_model\\meteo_iso_model_input_code_and_results\\output\\iso_excel_out\\new_data_prediction_comparison.xls"
    integr_df.to_excel(addd)
    return integr_df
############################################################
