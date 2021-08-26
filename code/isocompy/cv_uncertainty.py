import pandas as pd
import os
import numpy as np
###########################


def model_meanstd(name,input,direct):
    list_all=list()
    for i in range(0,len(input)):
        try:
            cvr_df=pd.DataFrame(input[i]["best_estimator"].cv_results_)
            cvr_df=cvr_df.iloc[input[i]["best_estimator"].best_index_]
            mean_test_score=round(float(cvr_df.mean_test_score),2)
            std_test_score=round(float(cvr_df.std_test_score),2)
            mean_train_score=round(float(cvr_df.mean_train_score),2)
            std_train_score=round(float(cvr_df.std_train_score),2)
            list_all.append([i+1,mean_test_score,std_test_score,mean_train_score,std_train_score])
        except:
            list_all.append([i+1,np.nan,np.nan,np.nan,np.nan])
            
    df = pd.DataFrame (list_all,columns=['Index','mean_test_score','std_test_score','mean_train_score','std_train_score'])
    df.to_excel(os.path.join(direct,name+"_model_cv_prediction_stats.xls"))


'''def init(model_direc,output_direc):
    iso_meteo_model=isocompy.tools.session.load(model_direc)

    try:
        model_meanstd("rain",iso_meteo_model.rain_bests_dic,output_direc)

        model_meanstd("temp",iso_meteo_model.temp_bests_dic,output_direc)

        model_meanstd("hum",iso_meteo_model.hum_bests_dic,output_direc)
    except:pass    

    try:
        model_meanstd("iso18",iso_meteo_model.iso18_bests_dic,output_direc)

        model_meanstd("iso2h",iso_meteo_model.iso2h_bests_dic,output_direc)

        model_meanstd("iso3h",iso_meteo_model.iso3h_bests_dic,output_direc)
    except: print ("NO VALID ISOTOPE MODEL IN THE CLASS!")'''
