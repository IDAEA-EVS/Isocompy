
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import classification_report,confusion_matrix,mean_squared_error,mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.inspection import partial_dependence,plot_partial_dependence
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.feature_selection import f_classif,SelectPercentile
import time
import dill
from datetime import date
#import function py file
from meteo_iso_functions import regional_mensual_plot,grouping_data,pcafun,rfmethod,importing_preprocess,print_to_file,iso_prediction,f_reg_mutual,predict_points,new_data_prediction_comparison
from sklearn.linear_model import LinearRegression


def warn(*args, **kwargs):
    pass
import warnings

warnings.warn = warn

def model_coup():

        
    ############################################################
    t_total_start=time.time()
    #read files and some pre processing
    month_grouped_list_with_zeros_iso_18,month_grouped_list_without_zeros_iso_18,month_grouped_list_with_zeros_iso_2h,month_grouped_list_without_zeros_iso_2h,month_grouped_list_with_zeros_iso_3h,month_grouped_list_without_zeros_iso_3h,month_grouped_list_with_zeros_hum,month_grouped_list_without_zeros_hum,month_grouped_list_with_zeros_rain,month_grouped_list_without_zeros_rain,month_grouped_list_with_zeros_temp,month_grouped_list_without_zeros_temp,rain,temper,elnino,lanina,iso_18,iso_2h,iso_3h=importing_preprocess()
    ############################################################
    #METEO MODELS!

    #RAIN normal for 12 month
    rain_bests=list()
    rain_preds_real=list()
    for monthnum in range(1,13):
        print ("########################################################")
        print ("##RAIN NORMAL month: ", str(monthnum))
        print ("########################################################")
        temp_rain_hum="Mean_Value_rain"
        #print ("########WITH ZEROS#####")
        tunedpars={"min_weight_fraction_leaf":[0,0.02,0.04],"n_estimators":[50,100,150,200,250,300],"criterion": ["mse","mae"],"min_samples_split":[2,5] }
        gridsearch_dictionary={"activation" : ["identity", "logistic", "tanh", "relu"],"solver" : ["lbfgs", "sgd", "adam"],"alpha":[0.0001,0.0003,0.0005],"hidden_layer_sizes":[(10,)*2,(25,)*2 ,(50,)*2,(75,)*2,(50,)*3,(50,)*4],"max_iter":[50,100,200],"n_iter_no_change":[5,10,15]}
        Y_preds,X_temp_fin ,Y_temp_fin,X_train_rain_normal_with_zeros, X_test_rain_normal_with_zeros, y_train_rain_normal_with_zeros, y_test_rain_normal_with_zeros,best_estimator_all,best_score_all,mutual_info_regression_value,f_regression_value,x_scaler,y_scaler,didlog,used_features,rsquared=rfmethod(tunedpars,gridsearch_dictionary,month_grouped_list_with_zeros_rain[monthnum-1],temp_rain_hum,monthnum,"rain",meteo_or_iso="meteo")
        rain_bests.append([best_estimator_all,best_score_all,mutual_info_regression_value,f_regression_value,used_features,rsquared,x_scaler,y_scaler,didlog])
        rain_preds_real.append([Y_preds,Y_temp_fin])
    ###########################################
    #TEMP
    temp_bests=list()
    temp_preds_real=list()
    for monthnum in range(1,13):

        print ("########################################################")
        print ("##TEMP NORMAL month: ", str(monthnum))
        print ("########################################################")
        temp_rain_hum="Mean_Value_temp"
        tunedpars={"min_weight_fraction_leaf":[0,0.02,0.04],"n_estimators":[50,100,150,200,250,300],"criterion": ["mse","mae"],"min_samples_split":[2,5] }
        gridsearch_dictionary={"activation" : ["identity", "logistic", "tanh", "relu"],"solver" : ["lbfgs", "sgd", "adam"],"alpha":[0.0001,0.0003,0.0005],"hidden_layer_sizes":[(10,)*2,(25,)*2 ,(50,)*2,(75,)*2,(50,)*3,(50,)*4],"max_iter":[50,100,200],"n_iter_no_change":[5,10,15]}
        Y_preds,X_temp_fin ,Y_temp_fin,X_train_temp_normal_with_zeros, X_test_temp_normal_with_zeros, y_train_temp_normal_with_zeros, y_test_temp_normal_with_zeros,best_estimator_all,best_score_all,mutual_info_regression_value,f_regression_value,x_scaler,y_scaler,didlog,used_features,rsquared=rfmethod(tunedpars,gridsearch_dictionary,month_grouped_list_with_zeros_temp[monthnum-1],temp_rain_hum,monthnum,"temp",meteo_or_iso="meteo")
        temp_bests.append([best_estimator_all,best_score_all,mutual_info_regression_value,f_regression_value,used_features,rsquared,x_scaler,y_scaler,didlog])
        temp_preds_real.append([Y_preds,Y_temp_fin])
    ##########################################
    #Humidity
    hum_bests=list()
    hum_preds_real=list()
    for monthnum in range(1,13):
        print ("########################################################")
        print ("##HUMIDITY NORMAL month: ", str(monthnum))
        print ("########################################################")
        temp_rain_hum="Mean_Value_hum"
        tunedpars={"min_weight_fraction_leaf":[0,0.02,0.04],"n_estimators":[50,100,150,200,250,300],"criterion": ["mse","mae"],"min_samples_split":[2,5] }
        gridsearch_dictionary={"activation" : ["identity", "logistic", "tanh", "relu"],"solver" : ["lbfgs", "sgd", "adam"],"alpha":[0.0001,0.0003,0.0005],"hidden_layer_sizes":[(10,)*2,(25,)*2 ,(50,)*2,(75,)*2,(50,)*3,(50,)*4],"max_iter":[50,100,200],"n_iter_no_change":[5,10,15]}
        Y_preds,X_temp_fin ,Y_temp_fin,X_train_hum_normal_with_zeros, X_test_hum_normal_with_zeros, y_train_hum_normal_with_zeros, y_test_hum_normal_with_zeros,best_estimator_all,best_score_all,mutual_info_regression_value,f_regression_value,x_scaler,y_scaler,didlog,used_features,rsquared=rfmethod(tunedpars,gridsearch_dictionary,month_grouped_list_with_zeros_hum[monthnum-1],temp_rain_hum,monthnum,"humid",meteo_or_iso="meteo")
        hum_bests.append([best_estimator_all,best_score_all,mutual_info_regression_value,f_regression_value,used_features,rsquared,x_scaler,y_scaler,didlog])
        hum_preds_real.append([Y_preds,Y_temp_fin])
    #############################################################
    # write outputs to a file 
    file_name=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\output\models_output.txt"
    print_to_file(file_name, temp_bests, rain_bests, hum_bests)
    #############################################################
    #making prediction for the isotope points
    iso_db1=month_grouped_list_with_zeros_iso_18
    predictions_monthly_list, all_preds=iso_prediction(iso_db1,month_grouped_list_with_zeros_iso_2h,month_grouped_list_with_zeros_iso_3h,temp_bests,rain_bests,hum_bests,justsum=True)
    all_preds.to_excel(r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\output\predicted_results.xls")
    #############################################################
    #f_reg and mutual annual
    list_of_dics=[
        {"inputs":["CooX","CooY","CooZ"],"outputs":["temp"]},
        {"inputs":["CooX","CooY","CooZ"],"outputs":["hum"]},
        {"inputs":["CooX","CooY","CooZ"],"outputs":["rain"]},
        {"inputs":["CooX","CooY","CooZ","temp","rain","hum"],"outputs":["iso_2h"]},
        {"inputs":["CooX","CooY","CooZ","temp","rain","hum"],"outputs":["iso_18"]},
        {"inputs":["CooX","CooY","CooZ","temp","rain","hum"],"outputs":["iso_3h"]},
    ]
    file_name=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\output\f_reg_mutual_output_annual.txt"
    f_reg_mutual(file_name,all_preds,list_of_dics)
    #############################################################
    #############################################################
    #f_reg and mutual mensual
    all_preds_month=all_preds["month"].copy()
    all_preds_month.drop_duplicates(keep = 'last', inplace = True)
    for mn in all_preds_month:
        all_preds_temp=all_preds[all_preds["month"]==mn]
        list_of_dics=[
            {"inputs":["CooX","CooY","CooZ"],"outputs":["temp"]},
            {"inputs":["CooX","CooY","CooZ"],"outputs":["hum"]},
            {"inputs":["CooX","CooY","CooZ"],"outputs":["rain"]},
            {"inputs":["CooX","CooY","CooZ","temp","rain","hum"],"outputs":["iso_2h"]},
            {"inputs":["CooX","CooY","CooZ","temp","rain","hum"],"outputs":["iso_18"]},
            {"inputs":["CooX","CooY","CooZ","temp","rain","hum"],"outputs":["iso_3h"]},
        ]
        file_name="C:\\Users\\Ash kan\\Documents\\meteo_iso_model\\meteo_iso_model_input_code_and_results\\output\\monthly_f_test\\"+"month_"+str(mn)+"_f_reg_mutual_output_mensual.txt"
        f_reg_mutual(file_name,all_preds_temp,list_of_dics)
    #############################################################
    '''##################PCA for interpreting the data##############
        print(['CooX', 'CooY', 'CooZ', 'iso_18','temp', 'rain'])
        pca_raw_df=all_preds.drop(columns=["month","ID_MeteoPoint","iso_2h","iso_3h","hum"]).copy()
        pca_fit_Rain_normal=pcafun(pca_raw_df,kmeans_group_nums=5,filetitlename="xyz_18_temp_rain") 
        #############################################################
        print(['CooX', 'CooY', 'CooZ', 'iso_3h','temp', 'rain'])
        pca_raw_df=all_preds.drop(columns=["month","ID_MeteoPoint","iso_2h","iso_18","hum"]).copy()
        pca_fit_Rain_normal=pcafun(pca_raw_df,kmeans_group_nums=5,filetitlename="xyz_3h_temp_rain")
        #############################################################
        print([ 'iso_18','temp', 'rain'])
        pca_raw_df=all_preds.drop(columns=["month","ID_MeteoPoint","iso_2h","iso_3h",'CooX', 'CooY', 'CooZ',"hum"]).copy()
        pca_fit_Rain_normal=pcafun(pca_raw_df,kmeans_group_nums=5,filetitlename="18_temp_rain")
        #############################################################
        print([ 'iso_3h','temp', 'rain'])
        pca_raw_df=all_preds.drop(columns=["month","ID_MeteoPoint","iso_2h","iso_18",'CooX', 'CooY', 'CooZ',"hum"]).copy()
        pca_fit_Rain_normal=pcafun(pca_raw_df,kmeans_group_nums=5,filetitlename="3h_temp_rain")'''
    ############################################################
    #modeling the isotopes:
    tunedpars={"min_weight_fraction_leaf":[0,0.01,0.02,0.03,0.04],"n_estimators":[25,50,75,100,150,200,250,300],"criterion": ["mse","mae"],"min_samples_split":[2,5,7] }
    gridsearch_dictionary={"activation" : ["identity", "logistic", "tanh", "relu"],"solver" : ["lbfgs", "sgd", "adam"],"alpha":[0.0001,0.0003,0.0005,0.001,0.005],"hidden_layer_sizes":[(10,)*2,(25,)*2 ,(50,)*2,(75,)*2,(50,)*3,(50,)*4,(100,)*2,(100,)*3,(100,)*4],"max_iter":[25,50,100,150,200,300],"n_iter_no_change":[5,10,15,20]}
    ####################################
    temp_rain_hum="iso_18"
    Y_preds_iso18,X_temp_fin_iso18 ,Y_temp_fin_iso18,X_train_iso_18_normal_with_zeros, X_test_iso_18_normal_with_zeros, y_train_iso_18_normal_with_zeros, y_test_iso_18_normal_with_zeros,best_estimator_all_iso18,best_score_all_iso18,mutual_info_regression_value_iso18,f_regression_value_iso18,x_scaler_iso18,y_scaler_iso18,didlog_iso18,used_features_iso18,rsquared_iso18=rfmethod(tunedpars,gridsearch_dictionary,all_preds,temp_rain_hum,monthnum,"iso_18", meteo_or_iso="iso",inputs=["CooX","CooY","CooZ","temp","rain","hum"])
    ####################################
    temp_rain_hum="iso_2h"
    Y_preds_iso2h,X_temp_fin_iso2h ,Y_temp_fin_iso2h,X_train_iso_2h_normal_with_zeros, X_test_iso_2h_normal_with_zeros, y_train_iso_2h_normal_with_zeros, y_test_iso_2h_normal_with_zeros,best_estimator_all_iso2h,best_score_all_iso2h,mutual_info_regression_value_iso2h,f_regression_value_iso2h,x_scaler_iso2h,y_scaler_iso2h,didlog_iso2h,used_features_iso2h,rsquared_iso2h=rfmethod(tunedpars,gridsearch_dictionary,all_preds,temp_rain_hum,monthnum,"iso_2h", meteo_or_iso="iso",inputs=["CooX","CooY","CooZ","temp","rain","hum"])
    #############################################################
    #read points for contour
    data_file = r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\inputs\x_y_z.xls"
    x_y_z_=pd.read_excel(data_file,sheet_name=0,header=0,index_col=False,keep_default_na=True)
    no_needed_month=[4,5,6,7,8,9]
    column_name="predicted_iso18"
    monthly_iso18_output=predict_points(used_features_iso18,x_y_z_,no_needed_month,temp_bests,rain_bests,hum_bests,x_scaler_iso18,y_scaler_iso18,didlog_iso18,best_estimator_all_iso18,column_name)
    column_name="predicted_iso2h"
    monthly_iso2h_output=predict_points(used_features_iso2h,x_y_z_,no_needed_month,temp_bests,rain_bests,hum_bests,x_scaler_iso2h,y_scaler_iso2h,didlog_iso2h,best_estimator_all_iso2h,column_name)
    ############################################################
    #regional_mensual_plot
    #regional_mensual_plot(x_y_z_,monthly_iso18_output,monthly_iso2h_output)
    ############################################################
    data_fl=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\inputs\new_measured.xlsx"
    newd=pd.read_excel(data_fl,sheet_name=0,header=0,index_col=False,keep_default_na=True)
    new_data_prediction=new_data_prediction_comparison(newd,no_needed_month,temp_bests,rain_bests,hum_bests,didlog_iso18,didlog_iso2h,x_scaler_iso18,x_scaler_iso2h,used_features_iso18,used_features_iso2h,best_estimator_all_iso18,best_estimator_all_iso2h,y_scaler_iso18,y_scaler_iso2h)
    #time
    t_total_end=time.time()
    print ("#################################\n######Total run time:\n", t_total_end-t_total_start)
    #writing isotope predictions into a file
    pd.concat([all_preds,Y_preds_iso18,Y_preds_iso2h],axis=1).to_excel(r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\output\isotope_main_data_predictions.xlsx")
    return Y_preds_iso18,Y_preds_iso2h,rain_preds_real,hum_preds_real,temp_preds_real,temp_bests,rain_bests,hum_bests,monthly_iso2h_output,monthly_iso18_output,x_scaler_iso18,y_scaler_iso18,didlog_iso18,used_features_iso18,rsquared_iso18,best_estimator_all_iso18,best_score_all_iso18,x_scaler_iso2h,y_scaler_iso2h,didlog_iso2h,used_features_iso2h,rsquared_iso2h,best_estimator_all_iso2h,best_score_all_iso2h,predictions_monthly_list, all_preds,month_grouped_list_with_zeros_iso_18,month_grouped_list_without_zeros_iso_18,month_grouped_list_with_zeros_iso_2h,month_grouped_list_without_zeros_iso_2h,month_grouped_list_with_zeros_iso_3h,month_grouped_list_without_zeros_iso_3h,month_grouped_list_with_zeros_hum,month_grouped_list_without_zeros_hum,month_grouped_list_with_zeros_rain,month_grouped_list_without_zeros_rain,month_grouped_list_with_zeros_temp,month_grouped_list_without_zeros_temp,rain,temper,elnino,lanina,iso_18,iso_2h,iso_3h

    ############################################################
    #some tests
    '''
        print(explained_variance_score(y_test_rain_lanina, predrf_temp))
        rfr=RandomForestRegressor()
        visualizer = ResidualsPlot(rfr)
        visualizer.fit(X_train_rain_lanina, y_train_rain_lanina)  # Fit the training data to the model
        visualizer.score(X_test_rain_lanina.to_numpy(), y_test_rain_lanina.to_numpy())  # Evaluate the model on the test data
        visualizer.poof()                 # Draw/show/poof the data
        #deleting some outliers

        a=Y_rain_lanina[Y_rain_lanina["Mean_Value_rain"]>4].index
        b=X_rain_lanina[X_rain_lanina["CooZ"]>10000].index
        Y_rain_lanina=Y_rain_lanina.drop(b,axis=0)
        X_rain_lanina=X_rain_lanina.drop(b,axis=0)

        #support vector machine regression
        from sklearn.svm import SVR
        clf = SVR( epsilon=0.01)
        clf.fit(X_train_rain_lanina, y_train_rain_lanina)
        print ("r2",clf.score(X_rain_lanina,Y_rain_lanina))
        predrf_temp=clf.predict(X_test_rain_lanina)
        print ("mean square error:",mean_squared_error(y_test_rain_lanina, predrf_temp))
        print ("mean abs error:",mean_absolute_error(y_test_rain_lanina, predrf_temp))
        ############################################################# 
        visualizer = Rank1D(features=["CooX","CooY","CooZ","Mean_Value_rain"], algorithm='shapiro')
        visualizer.fit(X_train_rain_lanina, y_train_rain_lanina)
        visualizer.transform(X_train_rain_lanina)             # Transform the data
        visualizer.poof()  
        mlp = MLPRegressor(activation='logistic',solver='lbfgs',hidden_layer_sizes=(15,)*2,max_iter=100,n_iter_no_change=5)
        mlp.fit(X_train_rain_normal,y_train_rain_normal)
        predrf_rain=mlp.predict(X_test_rain_normal)
        print ("score:", mlp.score(X_test_rain_normal,y_test_rain_normal))
        print ("mean square error:",temp_rain_hum,mean_squared_error(y_test_rain_normal, predrf_rain))
        print ("mean abs error:",temp_rain_hum,mean_absolute_error(y_test_rain_normal, predrf_rain))
        '''
    #############################################################
    # 4 MODELS FOR TEMP, RAIN, HUMID
    '''
        #temperature elnino
        print ("########################################################")
        print ("TEMP ELNINO")
        print ("########################################################")
        temp_rain_hum="Mean_Value_temp"
        min_weight_fraction_leafs=0.1
        n_estimators=300
        estrandomfor_temp_elnino,X_temp_elnino ,Y_temp_elnino,X_train_temp_elnino, X_test_temp_elnino,y_train_temp_elnino, y_test_temp_elnino=rfmethod(min_weight_fraction_leafs,n_estimators,newmatdf_temp_elnino,temp_rain_hum,20)

        #rain elnino
        print ("########################################################")
        print ("RAIN ELNINO")
        print ("########################################################")
        temp_rain_hum="Mean_Value_rain"
        min_weight_fraction_leafs=0.05
        n_estimators=200
        estrandomfor_rain_elnino,X_rain_elnino ,Y_rain_elnino,X_train_rain_elnino, X_test_rain_elnino, y_train_rain_elnino, y_test_rain_elnino=rfmethod(min_weight_fraction_leafs,n_estimators,newmatdf_rain_elnino,temp_rain_hum,5)
        #################
        #temperature lanina
        print ("########################################################")
        print ("TEMP LANINA")
        print ("########################################################")
        temp_rain_hum="Mean_Value_temp"
        min_weight_fraction_leafs=0.1
        n_estimators=400
        estrandomfor_temp_lanina, X_temp_lanina, Y_temp_lanina, X_train_temp_lanina, X_test_temp_lanina, y_train_temp_lanina, y_test_temp_lanina=rfmethod(min_weight_fraction_leafs,n_estimators,newmatdf_temp_lanina,temp_rain_hum,20)

        #rain lanina
        print ("########################################################")
        print ("RAIN LANINA")
        print ("########################################################")
        temp_rain_hum="Mean_Value_rain"
        min_weight_fraction_leafs=0.05
        n_estimators=200
        estrandomfor_rain_lanina,X_rain_lanina ,Y_rain_lanina,X_train_rain_lanina, X_test_rain_lanina, y_train_rain_lanina, y_test_rain_lanina=rfmethod(min_weight_fraction_leafs,n_estimators,newmatdf_rain_lanina,temp_rain_hum,5)
        #################'''
    ############################################################
    #some tests
    '''
        iso_18=iso_18.groupby(pd.Grouper(key='CooX_in')).mean()
        iso_2h=iso_2h.groupby(pd.Grouper(key='CooX_in')).mean()
        iso_3h=iso_3h.groupby(pd.Grouper(key='CooX_in')).mean()
        iso_mergeddf=iso_18.merge(iso_2h,how="inner" )
        iso_mergeddf=iso_mergeddf.merge(iso_3h,how="inner")
        iso_mergeddf=iso_mergeddf.drop(columns="mes")
        iso_mergeddf['CooX_in']=iso_mergeddf["CooX"]
        merged_stat_tem_rain['CooX_in']=merged_stat_tem_rain["CooX"]
        iso_mergeddf=iso_mergeddf.set_index("CooX_in")
        merged_stat_tem_rain=merged_stat_tem_rain.set_index("CooX_in")
        iso_meteo_merged=iso_mergeddf.merge(merged_stat_tem_rain,how="inner")
        iso_meteo_merged.reset_index(drop=True)
        iso_meteo_merged.to_excel('C:\\Users\\Ash kan\\Desktop\\sonia\\iso_meteo_merged.xlsx')
        #####################################################
        #####################################################
        # PCA on all data
        pca_raw_df_isomet=iso_meteo_merged[["CooZ","iso_3h","pred_Rain","pred_Temp"]].copy()
        ###################
        #scaling
        scalerr = MinMaxScaler()
        scalerr.fit(pca_raw_df_isomet)
        pca_raw_df_isomet_st = scalerr.transform(pca_raw_df_isomet)
        #############       
        #calculating pca
        pca_isomet=PCA()
        pcaf_isomet=pca_isomet.fit(pca_raw_df_isomet_st).transform(pca_raw_df_isomet_st)
        print ("Variance ratio:", pd.DataFrame(pca_isomet.explained_variance_ratio_ ))
        plt.scatter(pcaf_isomet[:,0],pcaf_isomet[:,1])
        plt.title('PCA1-PCA2 "CooZ","iso_3h","pred_Rain","pred_Temp"')
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],marker="^")
        #############
        #kmeans
        pcaf_isomet_df=pd.DataFrame(pcaf_isomet[:,:3],columns=["pca1","pca2","pca3"])
        kmeans = KMeans(n_clusters=7, random_state=0).fit(pcaf_isomet_df[["pca1","pca2"]])
        #3d plot
        threedee = plt.figure().gca(projection='3d')
        threedee.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],kmeans.cluster_centers_[:,2],marker="^") 
        threedee.scatter(pcaf_isomet_df["pca1"], pcaf_isomet_df["pca2"], pcaf_isomet_df["pca3"])
        plt.title('PCA1-PCA2-PCA3 "CooY","CooZ","iso_3h","pred_Rain","pred_Temp" ')
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        ####################
        pd.concat([pcaf_isomet_df,pd.DataFrame(kmeans.labels_)],axis=1)
        iso_meteo_pca_kmeans=pd.concat([pca_raw_df_isomet,pd.DataFrame(kmeans.labels_)],axis=1)
        iso_meteo_pca_kmeans.to_excel('C:\\Users\\Ash kan\\Desktop\\sonia\\Winter\\z_3h_temp_rain.xlsx')
        ############################################
        #trying the clustering with DBSCAN
        dbscan=DBSCAN(eps=0.5, min_samples=3)
        dbscan.fit()
        #############################################################
        #mutual_info_regression and f_regression_value
        from sklearn.feature_selection import GenericUnivariateSelect, f_regression, chi2,mutual_info_regression
        mutual_info_regression_value_rain = mutual_info_regression(X, Y["Mean_Value_rain"])
        mutual_info_regression_value_rain /= np.max(mutual_info_regression_value_rain)
        f_regression_value_rain=f_regression(X, Y["Mean_Value_rain"])
        f_regression_value_rain /= np.max(f_regression_value_rain)
        print ("f_regression_value_rain",f_regression_value_rain[0,:])
        print ("mutual_info_regression_value_rain",mutual_info_regression_value_rain)

        mutual_info_regression_value_temp = mutual_info_regression(X, Y["Mean_Value_temp"])
        mutual_info_regression_value_temp /= np.max(mutual_info_regression_value_temp)
        f_regression_value_temp=f_regression(X, Y["Mean_Value_temp"])
        f_regression_value_temp /= np.max(f_regression_value_temp)
        print ("f_regression_value_temp",f_regression_value_temp[0,:])
        print ("mutual_info_regression_value_temp",mutual_info_regression_value_temp)'''
    #############################################################




if __name__ == "__main__":
    Y_preds_iso18,Y_preds_iso2h,rain_preds_real,hum_preds_real,temp_preds_real,temp_bests,rain_bests,hum_bests,monthly_iso2h_output,monthly_iso18_output,x_scaler_iso18,y_scaler_iso18,didlog_iso18,used_features_iso18,rsquared_iso18,best_estimator_all_iso18,best_score_all_iso18,x_scaler_iso2h,y_scaler_iso2h,didlog_iso2h,used_features_iso2h,rsquared_iso2h,best_estimator_all_iso2h,best_score_all_iso2h,predictions_monthly_list, all_preds,month_grouped_list_with_zeros_iso_18,month_grouped_list_without_zeros_iso_18,month_grouped_list_with_zeros_iso_2h,month_grouped_list_without_zeros_iso_2h,month_grouped_list_with_zeros_iso_3h,month_grouped_list_without_zeros_iso_3h,month_grouped_list_with_zeros_hum,month_grouped_list_without_zeros_hum,month_grouped_list_with_zeros_rain,month_grouped_list_without_zeros_rain,month_grouped_list_with_zeros_temp,month_grouped_list_without_zeros_temp,rain,temper,elnino,lanina,iso_18,iso_2h,iso_3h=model_coup() 
    #dill dump session
    today = date.today()
    dump_session_name="dill_dump_"+str(date.today())
    dill.dump_session("C:\\Users\\Ash kan\\Documents\\meteo_iso_model\\meteo_iso_model_input_code_and_results\\code\\"+dump_session_name+".pkl")
    ############################################################   