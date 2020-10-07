import pandas as pd
import time
import dill
from datetime import date
from meteo_iso_functions import rfmethod,importing_preprocess,print_to_file,iso_prediction,f_reg_mutual,predict_points,new_data_prediction_comparison


def warn(*args, **kwargs):
    pass
import warnings

warnings.warn = warn
def model_coup(rerun_meteo=False):

    if rerun_meteo==False:
        #load already ran sessions
        dill.load_session(r"C:\\Users\\Ash kan\\Documents\\meteo_iso_model\\meteo_iso_model_input_code_and_results\\code\\internal_dill_dump_sep_for_papermeteo_after_log_bug.pkl")
    else:        
        ############################################################
        t_total_start=time.time()
        #######################
        #importing data
        #data_file = r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\inputs\Meteorological_Data_25_02_UTM.xlsx"
        #rain = pd.read_excel(data_file,sheet_name="METEO_CHILE_Rain_All",header=0,index_col=False,keep_default_na=True)
        temp=pd.read_excel(r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\inputs\T_Daily.xlsx",sheet_name="T_Daily",header=0,index_col=False,keep_default_na=True)
        hum=pd.read_excel(r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\inputs\RH_Daily_0_100.xlsx",sheet_name="TODAS",header=0,index_col=False,keep_default_na=True)
        #rain csv Ashkan 2020
        data_file = r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\inputs\pp_daily_all.csv"
        
        #rain = pd.read_excel(data_file,sheet_name="T_Daily",header=0,index_col=False,keep_default_na=True)
        rain = pd.read_csv(data_file,sep=';',header=0,index_col=False,keep_default_na=True)
        rain['Date'] = pd.to_datetime(rain['Date'],format=date_format)
        #############################################################    
        ############################################################
        #isotope file preprocess
        data_file_iso = r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\inputs\pysplit_isotope_registro_Pp_02_05_2020.xlsx"
        iso_18 = pd.read_excel(data_file_iso,sheet_name="ISOT18O",header=0,index_col=False,keep_default_na=True)
        iso_2h=pd.read_excel(data_file_iso,sheet_name="ISOT2H",header=0,index_col=False,keep_default_na=True)
        iso_3h=pd.read_excel(data_file_iso,sheet_name="ISOT3",header=0,index_col=False,keep_default_na=True)
        #############################################################
        #2020 trajectory dates datadb for iso predictions
        try:
            data_file_iso = r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\inputs\Date_Rain_02_05_2020.xlsx"
            dates_db = pd.read_excel(data_file_iso,sheet_name="traj_samp_date",header=0,index_col=False,keep_default_na=True)
        except:
            dates_db=None
        ###########################################################
        #nino nina years
        elnino=list()
        lanina=list() 
        ###########################################################
        #read files and some pre processing
        month_grouped_list_with_zeros_iso_18,month_grouped_list_with_zeros_iso_2h,month_grouped_list_with_zeros_iso_3h,month_grouped_list_with_zeros_hum,month_grouped_list_with_zeros_rain,month_grouped_list_with_zeros_temp,rain,temp,hum=importing_preprocess(rain,temp,hum,iso_18,iso_2h,iso_3h,meteo_input_type="monthly",write_outliers_input=True) #meteo_input_type="daily_remove_outliers"
        ############################################################
        #METEO MODELS!

        #RAIN
        rain_bests=list()
        rain_preds_real=list()
        for monthnum in range(1,13):
            print ("########################################################")
            print ("##RAIN NORMAL month: ", str(monthnum))
            print ("########################################################")
            temp_rain_hum="Mean_Value_rain"
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
            #hum_bests.append([best_estimator_all,best_score_all,mutual_info_regression_value,f_regression_value,used_features,rsquared,x_scaler,y_scaler,didlog])
            #hum_preds_real.append([Y_preds,Y_temp_fin])
        #############################################################
        # write outputs to a file 
        file_name=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\output\models_output_temp_rain_hum_12_month_inc_outliers_logforallfunctions.txt"
        print_to_file(file_name, temp_bests, rain_bests, hum_bests)
        #############################################################
        #add a dill dum session:
        dill.dump_session(r"C:\\Users\\Ash kan\\Documents\\meteo_iso_model\\meteo_iso_model_input_code_and_results\\code\\internal_dill_dump_sep_for_papermeteo_after_log_bug_include_outliers_logforallfunctions.pkl")


    #making prediction for the isotope points
    run_iso_whole_year=False
    trajectories=False
    iso_model_month_list=[6,7,8]
    predictions_monthly_list, all_preds,all_hysplit_df_list_all_atts,col_for_f_reg,all_without_averaging=iso_prediction(month_grouped_list_with_zeros_iso_18,month_grouped_list_with_zeros_iso_2h,month_grouped_list_with_zeros_iso_3h,temp_bests,rain_bests,hum_bests,iso_18,dates_db,trajectories,iso_model_month_list,run_iso_whole_year)
    all_preds.to_excel(r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\output\predicted_results_678_include_outliers_logforallfunctions.xls")
    if trajectories==True:
        all_without_averaging.to_excel(r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\output\traj_data_no_averaging.xls")
    #############################################################

    
    #f_reg and mutual annual
    list_of_dics=[
        {"inputs":["CooX","CooY","CooZ"],"outputs":["temp"]},
        {"inputs":["CooX","CooY","CooZ"],"outputs":["hum"]},
        {"inputs":["CooX","CooY","CooZ"],"outputs":["rain"]},
        {"inputs":["CooX","CooY","CooZ","temp","rain","hum"]+col_for_f_reg,"outputs":["iso_2h"]},
        {"inputs":["CooX","CooY","CooZ","temp","rain","hum"]+col_for_f_reg,"outputs":["iso_18"]},
        {"inputs":["CooX","CooY","CooZ","temp","rain","hum"]+col_for_f_reg,"outputs":["iso_3h"]},
    ]
    file_name=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\output\f_reg_mutual_output_annual_678_include_coutliers_logforallfunctions.txt"
    f_reg_mutual(file_name,all_preds,list_of_dics)
    #############################################################
    #############################################################
    #f_reg and mutual mensual
    all_preds_month=all_preds["month"].copy()
    all_preds_month.drop_duplicates(keep = 'last', inplace = True)
    for mn in all_preds_month:
        all_preds_temp=all_preds[all_preds["month"]==mn]
        '''list_of_dics=[
            {"inputs":["CooX","CooY","CooZ"],"outputs":["temp"]},
            {"inputs":["CooX","CooY","CooZ"],"outputs":["hum"]},
            {"inputs":["CooX","CooY","CooZ"],"outputs":["rain"]},
            {"inputs":["CooX","CooY","CooZ","temp","rain","hum","real_distt_alt_n","real_distt_alt_s","real_distt_pac_s","real_distt_pac_n","percentage_alt_n","percentage_alt_s","percentage_pac_s","percentage_pac_n"],"outputs":["iso_2h"]},
            {"inputs":["CooX","CooY","CooZ","temp","rain","hum","real_distt_alt_n","real_distt_alt_s","real_distt_pac_s","real_distt_pac_n","percentage_alt_n","percentage_alt_s","percentage_pac_s","percentage_pac_n"],"outputs":["iso_18"]},
            {"inputs":["CooX","CooY","CooZ","temp","rain","hum","real_distt_alt_n","real_distt_alt_s","real_distt_pac_s","real_distt_pac_n","percentage_alt_n","percentage_alt_s","percentage_pac_s","percentage_pac_n"],"outputs":["iso_3h"]},
        ]'''
        file_name="C:\\Users\\Ash kan\\Documents\\meteo_iso_model\\meteo_iso_model_input_code_and_results\\output\\monthly_f_test\\"+"month_"+str(mn)+"_f_reg_mutual_output_mensual.txt"
        f_reg_mutual(file_name,all_preds_temp,list_of_dics)


    #############################################################
    '''##################PCA for interpreting the data##############
        pca_raw_df=all_preds.drop(columns=["month","ID_MeteoPoint","iso_2h","iso_3h","hum"]).copy()
        pca_fit_Rain_normal=pcafun(pca_raw_df,kmeans_group_nums=5,filetitlename="xyz_18_temp_rain") 
        #############################################################
        pca_raw_df=all_preds.drop(columns=["month","ID_MeteoPoint","iso_2h","iso_18","hum"]).copy()
        pca_fit_Rain_normal=pcafun(pca_raw_df,kmeans_group_nums=5,filetitlename="xyz_3h_temp_rain")
        #############################################################
        pca_raw_df=all_preds.drop(columns=["month","ID_MeteoPoint","iso_2h","iso_3h",'CooX', 'CooY', 'CooZ',"hum"]).copy()
        pca_fit_Rain_normal=pcafun(pca_raw_df,kmeans_group_nums=5,filetitlename="18_temp_rain")
        #############################################################
        pca_raw_df=all_preds.drop(columns=["month","ID_MeteoPoint","iso_2h","iso_18",'CooX', 'CooY', 'CooZ',"hum"]).copy()
        pca_fit_Rain_normal=pcafun(pca_raw_df,kmeans_group_nums=5,filetitlename="3h_temp_rain")'''
    ############################################################
    #modeling the isotopes:
    tunedpars={"min_weight_fraction_leaf":[0,0.01,0.02,0.03,0.04],"n_estimators":[25,50,75,100,150,200,250,300],"criterion": ["mse","mae"],"min_samples_split":[2,5,7] }
    gridsearch_dictionary={"activation" : ["identity", "logistic", "tanh", "relu"],"solver" : ["lbfgs", "sgd", "adam"],"alpha":[0.0001,0.0003,0.0005,0.001,0.005],"hidden_layer_sizes":[(10,)*2,(25,)*2 ,(50,)*2,(75,)*2,(50,)*3,(50,)*4,(100,)*2,(100,)*3,(100,)*4],"max_iter":[25,50,100,150,200,300],"n_iter_no_change":[5,10,15,20]}
    ####################################
    temp_rain_hum="iso_18"
    Y_preds_iso18,X_temp_fin_iso18 ,Y_temp_fin_iso18,X_train_iso_18_normal_with_zeros, X_test_iso_18_normal_with_zeros, y_train_iso_18_normal_with_zeros, y_test_iso_18_normal_with_zeros,best_estimator_all_iso18,best_score_all_iso18,mutual_info_regression_value_iso18,f_regression_value_iso18,x_scaler_iso18,y_scaler_iso18,didlog_iso18,used_features_iso18,rsquared_iso18=rfmethod(tunedpars,gridsearch_dictionary,all_preds,temp_rain_hum,monthnum,"iso_18", meteo_or_iso="iso",inputs=["CooX","CooY","CooZ","temp","rain","hum"]+col_for_f_reg)
    ####################################
    temp_rain_hum="iso_2h"
    Y_preds_iso2h,X_temp_fin_iso2h ,Y_temp_fin_iso2h,X_train_iso_2h_normal_with_zeros, X_test_iso_2h_normal_with_zeros, y_train_iso_2h_normal_with_zeros, y_test_iso_2h_normal_with_zeros,best_estimator_all_iso2h,best_score_all_iso2h,mutual_info_regression_value_iso2h,f_regression_value_iso2h,x_scaler_iso2h,y_scaler_iso2h,didlog_iso2h,used_features_iso2h,rsquared_iso2h=rfmethod(tunedpars,gridsearch_dictionary,all_preds,temp_rain_hum,monthnum,"iso_2h", meteo_or_iso="iso",inputs=["CooX","CooY","CooZ","temp","rain","hum"]+col_for_f_reg)
    ####################################
    ####################################
    temp_rain_hum="iso_3h"
    Y_preds_iso3h,X_temp_fin_iso3h ,Y_temp_fin_iso3h,X_train_iso_3h_normal_with_zeros, X_test_iso_3h_normal_with_zeros, y_train_iso_3h_normal_with_zeros, y_test_iso_3h_normal_with_zeros,best_estimator_all_iso3h,best_score_all_iso3h,mutual_info_regression_value_iso3h,f_regression_value_iso3h,x_scaler_iso3h,y_scaler_iso3h,didlog_iso3h,used_features_iso3h,rsquared_iso3h=rfmethod(tunedpars,gridsearch_dictionary,all_preds,temp_rain_hum,monthnum,"iso_3h", meteo_or_iso="iso",inputs=["CooX","CooY","CooZ","temp","rain","hum"]+col_for_f_reg)
    ####################################
    #writing isotope results to a txt file
    pr_is_18="\n################\n\n best_estimator_all_iso18\n"+str(best_estimator_all_iso18)+"\n\n################\n\n used_features_iso18 \n"+str(used_features_iso18)+"\n\n################\n\n best_score_all_iso18 \n"+str(best_score_all_iso18)+"\n\n################\n\n rsquared_iso18 \n"+str(rsquared_iso18)+"\n\n################\n\n didlog_iso18 \n"+str(didlog_iso18)+"\n\n#########################\n#########################\n#########################\n"
    pr_is_2h="\n################\n\n best_estimator_all_iso2h\n"+str(best_estimator_all_iso2h)+"\n\n################\n\n used_features_iso2h \n"+str(used_features_iso2h)+"\n\n################\n\n best_score_all_iso2h \n"+str(best_score_all_iso2h)+"\n\n################\n\n rsquared_iso2h \n"+str(rsquared_iso2h)+"\n\n################\n\n rsquared_iso2h \n"+str(rsquared_iso2h)+"\n\n#########################\n#########################\n#########################\n"
    pr_is_3h="\n################\n\n best_estimator_all_iso3h\n"+str(best_estimator_all_iso3h)+"\n\n################\n\n used_features_iso3h \n"+str(used_features_iso3h)+"\n\n################\n\n best_score_all_iso3h \n"+str(best_score_all_iso3h)+"\n\n################\n\n rsquared_iso3h \n"+str(rsquared_iso3h)+"\n\n################\n\n didlog_iso3h \n"+str(didlog_iso3h)+"\n\n#########################\n#########################\n#########################\n"
    file_name="C:\\Users\\Ash kan\\Documents\\meteo_iso_model\\meteo_iso_model_input_code_and_results\\output\\isotope_modeling_results_18_2h_3h.txt"
    
    m_out_f=open(file_name,'w')
    m_out_f.write(pr_is_18)
    m_out_f.write(pr_is_2h)
    m_out_f.write(pr_is_3h)
    m_out_f.close()
    ###########
    #dill.dump_session(r"C:\\Users\\Ash kan\\Documents\\meteo_iso_model\\meteo_iso_model_input_code_and_results\\code\\internal_dill_dump_iso_18_2h_model_done_just_summer_alt_3000.pkl")
    if run_iso_whole_year==False:
        dill.dump_session(r"C:\\Users\\Ash kan\\Documents\\meteo_iso_model\\meteo_iso_model_input_code_and_results\\code\\internal_dill_dump_iso_18_2h_model_done_678_iso_after_logbug_including_outliers_logforallfunctions.pkl")

    else:

        dill.dump_session(r"C:\\Users\\Ash kan\\Documents\\meteo_iso_model\\meteo_iso_model_input_code_and_results\\code\\internal_dill_dump_iso_18_2h_model_done_allyear_iso_after_logbug_including_outliers_logforallfunctions.pkl")
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
    #regional_mensual_plot
    #regional_mensual_plot(x_y_z_,monthly_iso18_output,monthly_iso2h_output)
    ############################################################
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



if __name__ == "__main__":
    Y_preds_iso18,Y_preds_iso2h,rain_preds_real,hum_preds_real,temp_preds_real,temp_bests,rain_bests,hum_bests,monthly_iso2h_output,monthly_iso18_output,x_scaler_iso18,y_scaler_iso18,didlog_iso18,used_features_iso18,rsquared_iso18,best_estimator_all_iso18,best_score_all_iso18,x_scaler_iso2h,y_scaler_iso2h,didlog_iso2h,used_features_iso2h,rsquared_iso2h,best_estimator_all_iso2h,best_score_all_iso2h,predictions_monthly_list, all_preds,month_grouped_list_with_zeros_iso_18,month_grouped_list_without_zeros_iso_18,month_grouped_list_with_zeros_iso_2h,month_grouped_list_without_zeros_iso_2h,month_grouped_list_with_zeros_iso_3h,month_grouped_list_without_zeros_iso_3h,month_grouped_list_with_zeros_hum,month_grouped_list_without_zeros_hum,month_grouped_list_with_zeros_rain,month_grouped_list_without_zeros_rain,month_grouped_list_with_zeros_temp,month_grouped_list_without_zeros_temp,rain,temp,hum,elnino,lanina,iso_18,iso_2h,iso_3h=model_coup() 
    #dill dump session
    today = date.today()
    dump_session_name="dill_dump_"+str(date.today())
    dill.dump_session("C:\\Users\\Ash kan\\Documents\\meteo_iso_model\\meteo_iso_model_input_code_and_results\\code\\"+dump_session_name+".pkl")
    ############################################################   

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

