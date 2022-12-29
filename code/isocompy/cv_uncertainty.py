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