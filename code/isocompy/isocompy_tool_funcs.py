from sklearn.feature_selection import f_regression, mutual_info_regression
import numpy as np
import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.inspection import plot_partial_dependence
import itertools
from sklearn.linear_model import LinearRegression

###########################################################
#predict points for contouring
def predict_points(dir,write_to_file,x_y_z_org,st2_pred_month_list,st1_model_results_dic,st2_model_results_dic_var,column_name,trajectory_features_list):
    #prepare st1 inputs:
    st1_col_name=[]
    st1_bests_list=[]
    for k,v in st1_model_results_dic.items():
        st1_col_name.append(k)
        st1_bests_list.append(v["bests"])
    #################
    #prepare st2 inputs:
    used_features_st2=st2_model_results_dic_var["bests_dic"][0]["used_features"]
    x_scaler_iso18=st2_model_results_dic_var["bests"][0][6]
    y_scaler_iso18=st2_model_results_dic_var["bests"][0][7]
    didlog_iso18=st2_model_results_dic_var["bests_dic"][0]["didlog"]
    best_estimator_all_iso18=st2_model_results_dic_var["bests_dic"][0]["best_estimator"]
    #################
    monthly_iso_output=list()
    iso_model_month_list_min_one=[n-1 for n in st2_pred_month_list]
    for month in range(0,12):
        if month in iso_model_month_list_min_one:
            counterr=0
            x_y_z_copy=x_y_z_org.copy()
            #################################################
            #meteo prediction
            for meteopredict,colname in zip(st1_bests_list,st1_col_name):
                #just used features
                try:
                    x_y_z=x_y_z_copy[meteopredict[month][4]].copy()
                    counterr=counterr+1
                except:
                    raise Exception("in stage 1 model, month {}, used features are {} which can not found in the dataset. Features that are available in the dataset: {}. Please revise the dataset".format(month+1, str(meteopredict[month][4]), str(x_y_z_copy.columns) ) )    
                #general standard
                x_y_z=meteopredict[month][-3].transform(x_y_z)
                #transform if there is log in input
                if meteopredict[month][-1]==True:
                    x_y_z=np.log1p(x_y_z)
                    x_y_z_copy=x_y_z_copy[~np.isnan(x_y_z).any(axis=1)]
                    x_y_z=x_y_z[~np.isnan(x_y_z).any(axis=1)]
                #predicting    
                meteopredict_res=meteopredict[month][0].predict(x_y_z)
                #inverse transform
                #log
                if meteopredict[month][-1]==True:
                    meteopredict_res=pd.DataFrame(np.expm1(meteopredict_res),index=x_y_z_copy.index)
                    x_y_z_copy=x_y_z_copy[~np.isnan(meteopredict_res).any(axis=1)]
                    meteopredict_res=meteopredict_res[~np.isnan(meteopredict_res).any(axis=1)]
                
                #general
                meteopredict_res=pd.DataFrame(meteopredict_res)
                meteopredict_res=pd.DataFrame(meteopredict[month][-2].inverse_transform(meteopredict_res),columns=[colname],index=x_y_z_copy.index)

                meteopredict_res=pd.concat([meteopredict_res,x_y_z_copy],axis=1)
                meteopredict_res=meteopredict_res.set_index(["ID_preds_st2"],drop=False)
                #making the dataframe
                if counterr==1:
                    meteopredict_res_per_month=meteopredict_res
                else:
                    meteopredict_res_per_month=pd.concat([meteopredict_res_per_month,meteopredict_res[colname]],axis=1)
            #indexed,contain xyz        
            meteopredict_res_per_month=meteopredict_res_per_month[~np.isnan(pd.DataFrame(meteopredict_res_per_month[colname])).any(axis=1)]

            #################################################
            #trajectories prediction
            calc_trajectories=False
            for i in trajectory_features_list:
                if i in used_features_st2:
                    calc_trajectories=True
            if len(trajectory_features_list)==0:
                calc_trajectories=False
            if calc_trajectories==True:
                pass #(?????) 

            #################################################
            #predicting isotopes
            iso_model_input=meteopredict_res_per_month.reset_index(level=["ID_preds_st2"],drop=True)

            #transforming
            if x_scaler_iso18 !=None and used_features_st2!=None:
                try:
                    iso_model_input_us=x_scaler_iso18.transform(iso_model_input[used_features_st2])
                except:
                    raise Exception("in stage 2 model, used features are {} which can not found in the dataset or stage 2 predictions. Features that are available for predicting the second stage are: {}. Please revise the dataset".format(str(used_features_st2), str(iso_model_input.columns) ) )    
                #general standard

                if didlog_iso18==True:
                    iso_model_input_us=np.log1p(iso_model_input_us)
                    iso_model_input=iso_model_input[~np.isnan(iso_model_input_us).any(axis=1)]
                    iso_model_input_us=iso_model_input_us[~np.isnan(iso_model_input_us).any(axis=1)]

                #predicting
                each_month_iso_predict=best_estimator_all_iso18.predict(pd.DataFrame(iso_model_input_us))
                #inverse transform
                #log
                if didlog_iso18==True:
                    each_month_iso_predict=pd.DataFrame(np.expm1(each_month_iso_predict))
                    iso_model_input=iso_model_input.reset_index(drop=True)
                    iso_model_input_us=pd.DataFrame(iso_model_input_us).reset_index(drop=True)
                    iso_model_input=iso_model_input[~np.isnan(each_month_iso_predict).any(axis=1)].reset_index(drop=True)
                    iso_model_input_us=iso_model_input_us[~np.isnan(each_month_iso_predict).any(axis=1)].reset_index(drop=True)
                    each_month_iso_predict=each_month_iso_predict[~np.isnan(each_month_iso_predict).any(axis=1)].reset_index(drop=True)
                #general
                each_month_iso_predict=pd.DataFrame(y_scaler_iso18.inverse_transform(pd.DataFrame(each_month_iso_predict)),columns=[column_name])
                df_to_excel=pd.concat([iso_model_input,each_month_iso_predict],axis=1)
            else:
                df_to_excel=None    
            if write_to_file==True:
                addd=os.path.join(dir,str(month+1)+"_"+column_name+".xls")
                Path(dir).mkdir(parents=True,exist_ok=True)
                if isinstance(df_to_excel, pd.DataFrame):
                    df_to_excel.to_excel(addd)
            monthly_iso_output.append(df_to_excel)
            for ib in range(0,len(monthly_iso_output)):
                if isinstance(monthly_iso_output[ib], pd.DataFrame)==False:
                    try:
                        feats=monthly_iso_output[ib-1].columns
                    except:
                        try:
                            feats=monthly_iso_output[ib+1].columns
                        except:
                            feats=[]    
                    monthly_iso_output[ib]=pd.DataFrame(columns=feats)


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
            reg3.score(np.array(region3_iso18["predicted_iso18"]).reshape(-1, 1), np.array(region3_iso2h["predicted_iso2h"]).reshape(-1, 1))
            plt.scatter(region3_iso18["predicted_iso18"].mean(), region3_iso2h["predicted_iso2h"].mean(),marker=m[mon],c=coll[mon])
        a = np.linspace(-10,0)
        
        b2=8*a+10
        b1=(regmean/regcnt)[0]*a+(regcept/regcnt)[0]
        
        plt.plot(a,b1,label=str(round((regmean/regcnt)[0][0],2))+"*a+"+str( round(( regcept/regcnt )[0],2))  )
        plt.plot(a,b2,label='8a+10')
        plt.legend()
        plt.savefig(r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\output\model_plots\Iso_monthly_graph.pdf",dpi=300)
        plt.close("all")


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
def new_data_prediction_comparison(newd,iso_model_month_list,temp_bests,rain_bests,hum_bests,didlog_iso18,didlog_iso2h,x_scaler_iso18,x_scaler_iso2h,used_features_iso18,used_features_iso2h,best_estimator_all_iso18,best_estimator_all_iso2h,y_scaler_iso18,y_scaler_iso2h):
    x_y_z_month=newd[["CooX","CooY","CooZ","month"]]
    #iso_18_2h=newd[["iso18","iso2h"]]
    x_y_z_month_org=x_y_z_month.copy()
    #making a list of existed month
    existed_month=x_y_z_month[['month']]
    existed_month.drop_duplicates(keep = 'last', inplace = True)
    counter_month=0
    iso_model_month_list_min_one=[n-1 for n in iso_model_month_list]
    for month in range(0,12):
        if (month in iso_model_month_list_min_one) and (existed_month.isin([month+1]).any().bool()==True):
            
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
                iso18_model_input=np.log1p(iso18_model_input)
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
                iso2h_model_input=np.log1p(iso2h_model_input)
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
    integr_df=integr_df[["CooX","CooY","CooZ","temp","rain","hum","iso_18","iso_2h","month"]].reset_index()
    addd="C:\\Users\\Ash kan\\Documents\\meteo_iso_model\\meteo_iso_model_input_code_and_results\\output\\iso_excel_out\\new_data_prediction_comparison.xls"
    integr_df.to_excel(addd)
    return integr_df
############################################################

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
    plt.close("all")
    return pcaf
###########################################################


#f_reg and mutual
def f_reg_mutual(file_name,all_preds,list_of_dics):
    m_out_f=open(file_name,'a')
    for sets in list_of_dics:
        st_exist=False
        st_exist_m=True
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
        try:
            mxx=np.max(f_reg[0])
            for jj in range(0,len(f_reg[0])):
                f_reg[0][jj]/= mxx
        except:
            print ("f test all zeross!!")
        m_out_f.write("########################################################\n\n\n########################################################\n\n\ninputs:\n")
        m_out_f.write(str(sets["inputs"]))
        m_out_f.write('\n\noutput:')
        m_out_f.write(str(sets["outputs"]))
        m_out_f.write('\n\n\n\n f_regression \n\n')
        m_out_f.write(str(f_reg[0]))
        m_out_f.write('\n')
        m_out_f.write(str(f_reg[1]))
        m_out_f.write('\n\n\n mutual_info_standard\n\n')
        if st_exist==True:
            m_out_f.write(str(mutual_info_st))
        else:
            m_out_f.write("Not possible to calculate the STANDARD mutual, possibly division by zero problem!") 
        m_out_f.write('\n\n\n mutual_info \n \n')     
        if st_exist_m==True:
            m_out_f.write(str(mutual_info))
            m_out_f.write('\n')
        else:
            m_out_f.write("Not possible to calculate mutual. Possibly not enough data \n\n")

    m_out_f.close()

###########################################################
def best_estimator_and_partial_dep(    
        used_features,
        st1_or_st2,
        monthnum,
        model_type,
        direc,
        Y_preds,
        Y_measured,
        didlog,
        X_temp,
        best_estimator_all,
        estimator_plot,
        partial_dep_plot):
            # Plots       
            lens=len(used_features)
            if st1_or_st2=="st1":
                pltttl="month_"+ str(monthnum) +"_All_data_best_estimator_"+model_type
            else:
                pltttl="Annual_st2_All_data_best_estimator_"+model_type
            
            #########################################################################################################
            #########################################################################################################
            #Estimator plots
            if estimator_plot==True:
                pltname=os.path.join(direc,"model_plots",model_type,"best_estimators",pltttl+'.png')
                #folder making and checking:
                Path(os.path.join(direc,"model_plots",model_type,"best_estimators")).mkdir(parents=True,exist_ok=True)
                clnm=model_type+"_"+str(monthnum)
                #Y_preds=pd.DataFrame(Y_preds,columns=[clnm])
                fig, ax = plt.subplots()
                ax.scatter(Y_preds,Y_measured)

                plt.xlabel("Prediction")
                plt.ylabel("Real Value")
                left, right = plt.xlim()
                left1, right1 = plt.ylim()
                a = np.linspace(min(left,left1),max(right,right1),100)
                b=a
                reg = LinearRegression().fit(a.reshape(-1,1),b.reshape(-1,1))
                #residual mean and stds
                resd=Y_preds-Y_measured
                mean_obs=round(Y_measured.mean() ,2)
                mean_pred=round(Y_preds.mean(),2).Value
                mean_resids=round(resd.mean(),2).Value

                std_obs=round(Y_measured.std(),2)
                std_pred=round(Y_preds.std(),2).Value
                std_resids=round(resd.std(),2).Value
                tit="mo."+str(monthnum)+"|R2:"+str( round( reg.score(Y_preds,Y_measured) , 2 ))+"|mn_rsd:{} std_rsd:{}|mn_obs:{} std_obs:{}|mn_prd:{} std_prd:{}".format(mean_resids, std_resids, mean_obs, std_obs, mean_pred, std_pred)
                plt.title(tit,fontsize=8)
                plt.plot(a,b,label="y="+str(round(reg.coef_[0][0],2))+"x+"+str(round(reg.intercept_[0],2)))
                plt.legend()
                fig = plt.gcf()
                fig.savefig(pltname,dpi=300)
                plt.close("all")
            #########################################################################################################
            #########################################################################################################
            #partial dependency plots
            if partial_dep_plot==True:
                pltnamepardep=os.path.join(direc,"model_plots",model_type,"partial_dependency",'_partial_dependence'+pltttl+'.png')
                #folder making and checking:
                Path(os.path.join(direc,"model_plots",model_type,"partial_dependency")).mkdir(parents=True,exist_ok=True)
                if didlog==False:
                    figg, axx = plt.subplots()
                    disp=plot_partial_dependence(best_estimator_all, X_temp, features=list(range(0,lens)),feature_names=used_features,ax=axx)
                else:
                    figg, axx = plt.subplots()
                    disp=plot_partial_dependence(best_estimator_all, np.log1p(X_temp), features=list(range(0,lens)),feature_names=used_features,ax=axx)
                figg.savefig(pltnamepardep,dpi=300)
                #plt.show()
                plt.close(figg)

def best_estimator_and_part_plots(cls,st1,st2,estimator_plot,partial_dep_plot):

    #check if meteo models exists
    if st1==True:
        #try:
        #cls.rain_bests
        #cls.temp_bests
        #cls.hum_bests
        st1_or_st2="st1"
        res_dics=cls.st1_model_results_dic
        st1_model_month_list_min_one=[n-1 for n in cls.st1_model_month_list]
        for k,v in res_dics.items():
            for monthnum in st1_model_month_list_min_one:
                mon_dic=v["bests_dic"][monthnum]
                mon_dic_rea=v["preds_real_dic"][monthnum]
                if None not in [mon_dic["used_features"],mon_dic["didlog"],mon_dic["best_estimator"]]:
                    best_estimator_and_partial_dep(    
                            mon_dic["used_features"],
                            st1_or_st2,
                            monthnum+1,
                            direc=cls.direc,
                            Y_preds=mon_dic_rea["Y_preds"],
                            Y_measured=mon_dic_rea["Y_temp_fin"],
                            didlog=mon_dic["didlog"],
                            X_temp=mon_dic_rea["X_temp"],
                            best_estimator_all=mon_dic["best_estimator"],
                            estimator_plot=estimator_plot,
                            partial_dep_plot=partial_dep_plot,
                            model_type=k)


        '''
            rangee=range(0,12)
            for monthnum in rangee:
            zip_of_lists=zip(
                [cls.rain_bests_dic[monthnum]["used_features"],cls.temp_bests_dic[monthnum]["used_features"],cls.hum_bests_dic[monthnum]["used_features"]],
                ["rain","temp","humid"],
                [cls.rain_preds_real_dic[monthnum]["Y_preds"],cls.temp_preds_real_dic[monthnum]["Y_preds"],cls.hum_preds_real_dic[monthnum]["Y_preds"]],
                [cls.rain_preds_real_dic[monthnum]["Y_temp_fin"],cls.temp_preds_real_dic[monthnum]["Y_temp_fin"],cls.hum_preds_real_dic[monthnum]["Y_temp_fin"]],
                [cls.rain_preds_real_dic[monthnum]["didlog"],cls.temp_bests[monthnum][-1],cls.hum_bests[monthnum][-1]], #didlog
                [cls.rain_preds_real_dic[monthnum]["X_temp"],cls.temp_preds_real_dic[monthnum]["X_temp"],cls.hum_preds_real_dic[monthnum]["X_temp"]],
                [cls.rain_bests_dic[monthnum]["best_estimator"],cls.temp_bests_dic[monthnum]["best_estimator"],cls.hum_bests_dic[monthnum]["best_estimator"]])

            for (used_features,model_type,Y_preds,Y_measured,didlog,X_temp,best_estimator_all) in zip_of_lists:
                if None not in [used_features,didlog,best_estimator_all]:
                    best_estimator_and_partial_dep(    
                        used_features,
                        meteo_or_iso,
                        monthnum+1,
                        model_type,
                        cls.direc,
                        Y_preds,
                        Y_measured,
                        didlog,
                        X_temp,
                        best_estimator_all,
                        estimator_plot,
                        partial_dep_plot)'''
        #except:
        #    pass
    if st2==True:

        #try:
        #cls.iso18_bests
        #cls.iso2h_bests
        #cls.iso3h_bests
        st1_or_st2="st2"
        res_dics=cls.st2_model_results_dic
        st2_model_month_list_min_one=[n-1 for n in cls.st2_model_month_list]
        for k,v in res_dics.items():
            for monthnum in st2_model_month_list_min_one:
                mon_dic=v["bests_dic"][0]
                mon_dic_rea=v["preds_real_dic"][0]
                if None not in [mon_dic["used_features"],mon_dic["didlog"],mon_dic["best_estimator"]]:
                    best_estimator_and_partial_dep(    
                            mon_dic["used_features"],
                            st1_or_st2,
                            monthnum+1,
                            direc=cls.direc,
                            Y_preds=mon_dic_rea["Y_preds"],
                            Y_measured=mon_dic_rea["Y_temp_fin"],
                            didlog=mon_dic["didlog"],
                            X_temp=mon_dic_rea["X_temp"],
                            best_estimator_all=mon_dic["best_estimator"],
                            estimator_plot=estimator_plot,
                            partial_dep_plot=partial_dep_plot,
                            model_type=k)
