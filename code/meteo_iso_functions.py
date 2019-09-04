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

'''
def warn(*args, **kwargs):
    pass
import warnings

warnings.warn = warn'''


###########################################
###################################################################
def iso_prediction(iso_db1,month_grouped_list_with_zeros_iso_2h,month_grouped_list_with_zeros_iso_3h,temp_bests,rain_bests,hum_bests,justsum=True):
    
    predictions_monthly_list=list()
    for month_num in range(0,len(iso_db1)):
        if (month_num not in [4,5,6,7,8,9] and justsum ==True) or justsum==False:
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
            preds=pd.concat([iso_db1[month_num][["CooX","CooY","CooZ","ID_MeteoPoint","iso_18"]],month_grouped_list_with_zeros_iso_2h[month_num][["iso_2h"]],month_grouped_list_with_zeros_iso_3h[month_num][["iso_3h"]],temp_pred,rain_pred,hum_pred],axis=1,ignore_index =False)
            preds["month"]=month_num+1
            predictions_monthly_list.append([preds])
            
            if month_num==0:
                all_preds=preds
            else:
                all_preds=pd.concat([all_preds,preds],ignore_index=True)
            '''except:
                print ("IN except!!!!!!!!!!!!!!!!")
                print (month_num)'''
    return  predictions_monthly_list, all_preds      
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
                newmat_elnino.append({"ID_MeteoPoint":row['ID_MeteoPoint'], "CooX":tempp["CooX"].iat[0], "CooY":tempp["CooY"].iat[0], "CooZ":tempp["CooZ"].iat[0], which_value:Mean_Value_elnino})
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
        Y_temp=newmatdf_temp[[temp_rain_hum]].copy()
        X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(X_temp, Y_temp)
        num_var=3
        colmnss=["CooX","CooY","CooZ"]
    ##################################################
    if meteo_or_iso=="iso":
        X_temp=newmatdf_temp[inputs].copy().astype(float)
        Y_temp=newmatdf_temp[[temp_rain_hum]].copy()
        X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(X_temp, Y_temp)
        num_var=len(inputs)
        colmnss=inputs
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
    #best_score_all=-10
    #rsquared=-10
    #RandomForestRegressor
    estrandomfor_temp=GridSearchCV(RandomForestRegressor(random_state =0), tunedpars, cv=10,n_jobs=-1)
    estrandomfor_temp.fit(X_temp, Y_temp)
    best_score_all=adj_r_sqrd(estrandomfor_temp.best_score_)
    best_estimator_all=estrandomfor_temp.best_estimator_
    rsquared=estrandomfor_temp.best_score_
    ########################################################################
    #NEURAL NETWORK
    mlp_temp=GridSearchCV( MLPRegressor(learning_rate="adaptive"), gridsearch_dictionary, cv=5,n_jobs=-1) 
    mlp_temp.fit(X_temp, Y_temp)
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
    scores = cross_val_score(reg, X_temp, Y_temp, cv=10,n_jobs =-1)
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
    scores = cross_val_score(reg, X_temp, Y_temp, cv=10,n_jobs =-1)
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
    reg=GridSearchCV(SVR(), tunedpars_svm, cv=10,n_jobs=-1).fit(X_temp, Y_temp)
    print ("SVR")
    print ("SCORE:\n",adj_r_sqrd(reg.score(X_temp, Y_temp) ))
    if adj_r_sqrd(reg.score(X_temp, Y_temp))>best_score_all:
        best_score_all=adj_r_sqrd(reg.score(X_temp, Y_temp))
        best_estimator_all=reg
        rsquared=reg.score(X_temp, Y_temp)
    ####################################################################    
    #NuSVR    
    tunedpars_svm={"kernel":["linear", "poly", "rbf", "sigmoid"] }
    reg=GridSearchCV(NuSVR(), tunedpars_svm, cv=10,n_jobs=-1).fit(X_temp, Y_temp)
    print ("NuSVR")
    print ("SCORE:\n",adj_r_sqrd(reg.score(X_temp, Y_temp) ))
    if adj_r_sqrd(reg.score(X_temp, Y_temp) )>best_score_all:
        best_score_all=adj_r_sqrd(reg.score(X_temp, Y_temp) )
        best_estimator_all=reg
        rsquared=reg.score(X_temp, Y_temp)
    ####################################################################    
    print ("Best score total adj r square:")
    print (best_score_all)
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
    else:
        X_temp_fin=X_temp
        Y_temp_fin=Y_temp
    plt.scatter(best_estimator_all.predict(X_temp_fin),Y_temp_fin)
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
    return "estrandomfor_temp",X_temp ,Y_temp,X_train_temp, X_test_temp, y_train_temp, y_test_temp,best_estimator_all,best_score_all,mutual_info_regression_value_rain,f_regression_value_rain,x_scaler,y_scaler,didlog,used_features,rsquared 
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
        newmatdf_rain_elnino,newmatdf_rain_lanina,newmatdf_rain_all2=grouping_data(rain_cop,which_value,elnino,lanina,iso=iso,month=month,zeross=True)
        month_grouped_list_with_zeros.append(newmatdf_rain_all2)
        rain_cop=datab.copy()
        newmatdf_rain_elnino,newmatdf_rain_lanina,newmatdf_rain_all=grouping_data(rain_cop,which_value,elnino,lanina,iso=iso,month=month,zeross=False)
        month_grouped_list_without_zeros.append(newmatdf_rain_all)
    return    month_grouped_list_with_zeros,month_grouped_list_without_zeros
###########################################################
#importing_preprocess
def importing_preprocess():
    #######################
    #importing data
    data_file = r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\inputs\METEO_CHILE_2.xlsx"
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
    data_file_iso = r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\inputs\Rain_Snow_01_08_2019.xlsx"
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
    return month_grouped_list_with_zeros_iso_18,month_grouped_list_without_zeros_iso_18,month_grouped_list_with_zeros_iso_2h,month_grouped_list_without_zeros_iso_2h,month_grouped_list_with_zeros_iso_3h,month_grouped_list_without_zeros_iso_3h,month_grouped_list_with_zeros_hum,month_grouped_list_without_zeros_hum,month_grouped_list_with_zeros_rain,month_grouped_list_without_zeros_rain,month_grouped_list_with_zeros_temp,month_grouped_list_without_zeros_temp,rain,temper,elnino,lanina,iso_18,iso_2h,iso_3h
###################################################################
#print the models results in a file
def print_to_file(file_name,temp_bests,rain_bests,hum_bests):
    m_out_f=open(file_name,'w')
    for monthnum in range(0,len(temp_bests)):
        m_out_f.write("########################################################")
        m_out_f.write('\n TEMPERATURE month')
        m_out_f.write(str(monthnum+1))
        m_out_f.write('\n\n\n Used features \n')
        m_out_f.write(str(temp_bests[monthnum][4]))
        m_out_f.write('\n\n BEST SCORE adjusted R squared \n')
        m_out_f.write(str(temp_bests[monthnum][1]))
        m_out_f.write('\n\n BEST SCORE R squared \n')
        m_out_f.write(str(temp_bests[monthnum][5]))
        m_out_f.write('\n\n F REGRESSION \n')
        m_out_f.write(str(temp_bests[monthnum][3][0]))
        m_out_f.write('\n')
        m_out_f.write(str(temp_bests[monthnum][3][1]))
        m_out_f.write('\n\n MUTUAL INFORMATION REGRESSION \n')
        m_out_f.write(str(temp_bests[monthnum][2]))
        m_out_f.write('\n\n BEST ESTIMATOR \n')
        m_out_f.write(str(temp_bests[monthnum][0]))
        m_out_f.write('\n\nlog(1 + x)? \n')
        m_out_f.write(str(temp_bests[monthnum][-1]))
        m_out_f.write('\n')
    m_out_f.write("########################################################\n########################################################")
    m_out_f.write('\n')
    m_out_f.write("########################################################\n \n")
    for monthnum in range(0,len(temp_bests)):
        m_out_f.write("########################################################\nHUMIDITY month ")
        m_out_f.write(str(monthnum+1))
        m_out_f.write('\n\n BEST SCORE \n')
        m_out_f.write(str(hum_bests[monthnum][1]))
        m_out_f.write('\n\n')  
        m_out_f.write(" F REGRESSION \n")
        m_out_f.write(str(hum_bests[monthnum][3][0]))
        m_out_f.write('\n')
        m_out_f.write(str(hum_bests[monthnum][3][1]))
        m_out_f.write('\n\n MUTUAL INFORMATION REGRESSION\n')
        m_out_f.write(str(hum_bests[monthnum][2]))
        m_out_f.write('\n\n BEST ESTIMATOR \n')
        m_out_f.write(str(hum_bests[monthnum][0]))
        m_out_f.write('\n\n log(1 + x)?\n')
        m_out_f.write(str(hum_bests[monthnum][-1]))
        m_out_f.write('\n')
    m_out_f.write("########################################################\n########################################################")
    m_out_f.write('\n########################################################\n\n')
    for monthnum in range(0,len(temp_bests)):
        m_out_f.write("########################################################\n PRECIPITATION month ")
        m_out_f.write(str(monthnum+1))
        m_out_f.write('\n\nBEST SCORE \n')
        m_out_f.write(str(rain_bests[monthnum][1]))
        m_out_f.write('\n\n F REGRESSION\n')
        m_out_f.write(str(rain_bests[monthnum][3][0]))
        m_out_f.write('\n')
        m_out_f.write(str(rain_bests[monthnum][3][1]))
        m_out_f.write('\n\n  MUTUAL INFORMATION REGRESSION \n')
        m_out_f.write(str(rain_bests[monthnum][2]))
        m_out_f.write('\n\nBEST ESTIMATOR \n')
        m_out_f.write(str(rain_bests[monthnum][0]))
        m_out_f.write('\n\n log(1 + x)?\n')
        m_out_f.write(str(rain_bests[monthnum][-1]))
        m_out_f.write('\n')
    m_out_f.write("######################################################## \n ########################################################")
    m_out_f.write('\n########################################################\n\n')    
    m_out_f.close() 
    return       
###########################################################
#f_reg and mutual
def f_reg_mutual(file_name,all_preds,list_of_dics):
    m_out_f=open(file_name,'w')
    for sets in list_of_dics:
        st_exist=False
        mutual_info = mutual_info_regression(all_preds[sets["inputs"]],all_preds[sets["outputs"]])
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
        if st_exist==True:
            m_out_f.write("mutual_info_standard ")
            m_out_f.write('\n')
            m_out_f.write(str(mutual_info_st))
            m_out_f.write('\n\n')
        m_out_f.write("mutual_info ")
        m_out_f.write('\n')
        m_out_f.write(str(mutual_info))
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