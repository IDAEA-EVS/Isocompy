import copy
import os
import dill
from pathlib import Path
from datetime import datetime
from matplotlib import cm
from isocompy.isocompy_tool_funcs import f_reg_mutual,best_estimator_and_part_plots,predict_points
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import numpy as np
import itertools #do not remove it. ZIP used!
import matplotlib as mpl
from sklearn.linear_model import LinearRegression
from pylr2 import regress2 
from sklearn.metrics import r2_score
import isocompy.create_maps

class session(object):
    """
        The class to save and load the objects and sessions

        #------------------
        Methods:

            save(self,name="isocompy_saved_object")

            load(direc)

            save_session(direc,name="isocompy_saved_session", *argv)

            load_session(dir)

        #------------------
    """

    @staticmethod
    def save(self,name="isocompy_saved_object"):
        """
            The method to save an object

            #------------------
            Parameters:

                name: str default="isocompy_saved_object"
                    The output name string

            #------------------
            Returns:

                filename string 
                    Directory of the saved object

            #------------------
        """
        dateTimeObj = datetime.now().strftime("_%d_%b_%Y_%H_%M")
        
        try:
            self.st1_model_results_dic

            try:
                self.st2_model_results_dic
                filename=os.path.join(self.direc,name+dateTimeObj+"_st1True_st2True.pkl")

            except:    
                filename=os.path.join(self.direc,name+dateTimeObj+"_st1True_st2False.pkl")
        
        except: 

            filename=os.path.join(self.direc,name+dateTimeObj+"_st1False_st2False.pkl")
        with open(filename, 'wb') as filee:
            dill.dump(obj=self,file=filee)
        print ("\n\n pkl saved object directory: \n\n",filename)
        return filename
    ##########################################################################################
    @staticmethod
    def load(direc):
        """
            The method to load a pkl object. `direc` is the directory of the object to be loaded.

            #------------------
            Returns:

                obj pkl object
                    The loaded object

            #------------------
        """
        with open(direc, 'rb') as in_strm:
            obj=dill.load(in_strm)
        return obj
        
    @staticmethod
    def save_session(direc,name="isocompy_saved_session", *argv):
        """
            The method to save a session

            #------------------
            Parameters:

                name: str default="isocompy_saved_object"
                    The output name string

                *argv: the objects that wanted to be stored in the session

            #------------------
            Returns:

                filename string 
                    Directory of the saved session

            #------------------
        """
        dateTimeObj = datetime.now().strftime("_%d_%b_%Y_%H_%M")
        filename=os.path.join(direc,name+dateTimeObj+".pkl")
        for arg in argv:
            arg
        dill.dump_session(filename)   
        print ("\n\n pkl saved session directory: \n\n",filename)

    @staticmethod
    def load_session(dir):
        """
            The method to load a session

            #------------------
            Parameters:

                *argv: the objects that wanted to be stored in the session

            #------------------
            Returns:

                Loads the session

            #------------------
        """
        return dill.load_session(dir)

class evaluation(object):
    """
        The class to predict the second stage regression models

        #------------------
        Methods:

            __init__(self)  

            predict(self,cls,pred_inputs,stage2_vars_to_predict=None,direc=None,write_to_file=True)

        #------------------
        Attributes:

            direc: str
                directory of the class

            monthly_st2_output_list_all_vars: list
                list of stage two models outputs, seperated by month


            monthly_st2_output_dic_all_vars_df: dict
                dictionary of stage two models outputs, seperated by month. key is the month, and value is the output df of that specific month


            pred_inputs: Pandas Dataframe
                A Dataframe, that have to be contain of features that is used in stage one, that is going to be used to estimate the stage one and two models

            st2_predicted_month_list: list
                List of the months that have stage two regression models

        #------------------
    """    

    def __init__(self):
        self.direc=r""
        self.monthly_st2_output_list_all_vars=[]
        self.monthly_st2_output_dic_all_vars_df={}
        self.pred_inputs=pd.DataFrame()
        self.monthly_st2_output_list_all_vars=[]
        self.st2_predicted_month_list=[]


    def predict(self,cls,pred_inputs,stage2_vars_to_predict=None,direc=None,write_to_file=True):
        """
            The method to predict the second stage regression models

            #------------------
            Parameters:

                cls: model class
                    The model class that contains st1 and st2 models
                
                
                pred_inputs: Pandas dataframe
                    A Dataframe, that have to be contain of features that is used in stage one, that is going to be used to estimate the stage one and two models.
                    It can contain "month" field which could be used in evaluating the stage two predictions in observed data.
                    EXAMPLE:

                        pred_inputs=model_class.all_preds[["CooX","CooY","CooZ","month","ID"]].reset_index()

                
                stage2_vars_to_predict: None type or list of strs default=None
                    List of stage two dependent features to predict the outputs. If None, The results will be predicted for all two dependent features

                direc: None type or str default=None
                    Directory of the class. If None, it is the same directory as the model class.
                
                write_to_file: boolean default=True
                    To write the outputs in .xls files, seperated by the month

            #------------------
            Attributes:

                direc: str
                    directory of the class

                monthly_st2_output_list_all_vars: list
                    list of stage two models outputs, seperated by month


                monthly_st2_output_dic_all_vars_df: dict
                    dictionary of stage two models outputs, seperated by month. key is the month, and value is the output df of that specific month


                pred_inputs: Pandas Dataframe
                    A Dataframe, that have to be contain of features that is used in stage one, that is going to be used to estimate the stage one and two models

                st2_predicted_month_list: list
                    List of the months that have stage two regression models

            #------------------
        """

        if direc==None: self.direc=cls.direc
        else: self.direc=direc
        Path(self.direc).mkdir(parents=True, exist_ok=True)
        #############################################################
        self.pred_inputs=pred_inputs.rename(columns = {'ID': 'ID_preds_st2'}, inplace = False)
        if stage2_vars_to_predict==None: stage2_vars_to_predict=list(cls.st2_model_results_dic.keys())
        #here and iso_prediction, identifying the month should be nicer!
        temp_month_list=None
        for st2_var in stage2_vars_to_predict:
            if st2_var in list(cls.st2_model_results_dic.keys()):
                column_name="predicted_"+str(st2_var)
                monthly_st2_output_list_var=predict_points(
                    self.direc,
                    write_to_file,
                    x_y_z_org=self.pred_inputs, #xyz
                    st2_pred_month_list=cls.st2_model_month_list,
                    st1_model_results_dic=cls.st1_model_results_dic,
                    st2_model_results_dic_var=cls.st2_model_results_dic[st2_var],
                    column_name=column_name,
                    trajectory_features_list=cls.col_for_f_reg)

                self.monthly_st2_output_list_all_vars.append(monthly_st2_output_list_var)
                if temp_month_list==None:
                    temp_month_list=cls.st2_model_month_list
                elif temp_month_list!=cls.st2_model_month_list:   temp_month_list=False
                else: pass

            else: raise Exception ("{} can not be found in stage 2 models!".format(st2_var))
        #############################################################
        #to write all predictions in one file
        if type(temp_month_list)==list:
            lnn=len(self.monthly_st2_output_list_all_vars[0])
            for mon in range(0,lnn): #month
                n=0
                for i in self.monthly_st2_output_list_all_vars: #st2 vars
                    if n==0:
                        df=i[mon]
                        df=df.set_index("ID_preds_st2",drop=True)
                    else:
                        ndf=i[mon]
                        ndf=ndf.set_index("ID_preds_st2",drop=True)
                        cols_to_use = ndf.columns.difference(df.columns)
                        df = pd.merge(df, ndf[cols_to_use], left_index=True, right_index=True, how='inner')
                    n=n+1 
                addd=os.path.join(self.direc,str(temp_month_list[mon])+"_all_vars_st2_prediction.csv")
                Path(self.direc).mkdir(parents=True,exist_ok=True)
                df.to_csv(addd)
                self.monthly_st2_output_dic_all_vars_df[temp_month_list[mon]]=df
            self.st2_predicted_month_list=temp_month_list

            
                       
    ############################################################
    #def regional_mensual_plot():
        #pass
        #regional_mensual_plot
        #regional_mensual_plot(x_y_z_,monthly_iso18_output,monthly_iso2h_output)
    ############################################################

    #def new_inp_compare():   
        #pass
        #new_data_prediction_comparison
        #data_fl=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\inputs\new_measured.xlsx"
        #newd=pd.read_excel(data_fl,sheet_name=0,header=0,index_col=False,keep_default_na=True)
        #new_data_prediction=new_data_prediction_comparison(newd,iso_model_month_list,temp_bests,rain_bests,hum_bests,didlog_iso18,didlog_iso2h,x_scaler_iso18,x_scaler_iso2h,used_features_iso18,used_features_iso2h,best_estimator_all_iso18,best_estimator_all_iso2h,y_scaler_iso18,y_scaler_iso2h)
        #writing isotope predictions into a file
        #pd.concat([all_preds,Y_preds_iso18,Y_preds_iso2h],axis=1).to_excel(r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\output\isotope_main_data_predictions.xlsx")
        #return Y_preds_iso18,Y_preds_iso2h,rain_preds_real,hum_preds_real,temp_preds_real,temp_bests,rain_bests,hum_bests,monthly_iso2h_output,monthly_iso18_output,x_scaler_iso18,y_scaler_iso18,didlog_iso18,used_features_iso18,rsquared_iso18,best_estimator_all_iso18,best_score_all_iso18,x_scaler_iso2h,y_scaler_iso2h,didlog_iso2h,used_features_iso2h,rsquared_iso2h,best_estimator_all_iso2h,best_score_all_iso2h,predictions_monthly_list, all_preds,month_grouped_list_with_zeros_iso_18,month_grouped_list_with_zeros_iso_2h,month_grouped_list_with_zeros_iso_3h,month_grouped_list_with_zeros_hum,month_grouped_list_with_zeros_rain,month_grouped_list_with_zeros_temp,rain,temp,hum,elnino,lanina,iso_18,iso_2h,iso_3h
        ###########


class  stats(object):   
    """
        The class to calculate and generate statistical reports for the second stage models
        #------------------
        Methods:

            annual_stats(model_cls_obj)
            mensual_stats(model_cls_obj)
        #------------------
    """
    @staticmethod
    def seasonal_stats(model_cls_obj): #params=all, meteo, iso
        """
            The method to generate statistical reports for the second stage models based on  all specified month in second stage data

            #------------------
            Parameters:

                model_cls_obj
                    Input model class object

            #------------------

        """
        #iso_meteo_model.st1_model_results_dic["rain"]["bests_dic"][0]["used_features"] : ['CooX', 'CooY', 'CooZ']
        #iso_meteo_model.st1_varname_list: ['rain', 'temp']
        list_of_dics_for_stats=[]
        for k,v in model_cls_obj.st2_model_results_dic.items():
            ls_temp=list(v["bests_dic"][0]['correlation'].index)+model_cls_obj.col_for_f_reg
            list_of_dics_for_stats.append( {"inputs":ls_temp,"outputs":[k]})

        
        f_reg_mutual( os.path.join(model_cls_obj.direc,"annual_statistics_f_test_MI.txt") ,model_cls_obj.all_preds,list_of_dics_for_stats)   
    
    @staticmethod
    def monthly_stats(model_cls_obj): #params=all, meteo, iso
        """
            The method to generate statistical reports for the second stage models based on  each specified month in second stage data

            #------------------
            Parameters:

                model_cls_obj
                    Input model class object

            #------------------

        """
        list_of_dics_for_stats=[]
        for k,v in model_cls_obj.st2_model_results_dic.items():
            ls_temp=list(v["bests_dic"][0]['correlation'].index)+model_cls_obj.col_for_f_reg
            list_of_dics_for_stats.append( {"inputs":ls_temp,"outputs":[k]})

        all_preds_month=model_cls_obj.all_preds["month"].copy()
        all_preds_month.drop_duplicates(keep = 'last', inplace = True)
        for mn in all_preds_month:
            all_preds_temp=model_cls_obj.all_preds[model_cls_obj.all_preds["month"]==mn]
            Path(os.path.join(model_cls_obj.direc,"mensual_statistics")).mkdir(parents=True, exist_ok=True)
            file_name= os.path.join(model_cls_obj.direc,"mensual_statistics","month_"+str(mn)+"mensual_statistics_f_test_MI.txt")
            m_out_f=open(file_name,'w')
            m_out_f.write('\n\n month: ')
            m_out_f.write(str(mn))
            m_out_f.write("\n\n")
            m_out_f.close()
            f_reg_mutual(file_name,all_preds_temp,list_of_dics_for_stats)

class plots(object):
    """
        The method to generate the model class plots
        #------------------
        Methods:

            best_estimator_plots(cls,meteo_plot=True,iso_plot=True)

            partial_dep_plots(cls,meteo_plot=True,iso_plot=True)

            isotopes_meteoline_plot(ev_class,iso_class,iso_18,iso_2h,var_list,a=8,b=10,obs_data=False,residplot=False)

            map_generator(ev_class,feat_list,observed_class_list=None,month_list=None,unit_list=None,opt_title_list=None,x="CooX",y="CooY",shp_file=None,html=True,direc=None,minus_to_zero=False,max_to_hundred=False)
        #------------------
    """
    @staticmethod
    def best_estimator_plots(cls,st1=True,st2=True):
        """
            The method to plot the model class best estimators 

            #------------------
            Parameters:

                st1: boolean default=True
                    Generate plots for stage one regression models of the model class
                
                
                st2: boolean default=True
                    Generate plots for stage one regression models of the model class

            #------------------
        """
        estimator_plot=True
        partial_dep_plot=False
        best_estimator_and_part_plots(cls,st1,st2,estimator_plot,partial_dep_plot)
    @staticmethod
    def partial_dep_plots(cls,st1=True,st2=True):
        """
            The method to plot the partial dependency of the features of the model class

            #------------------
            Parameters:

                st1: boolean default=True
                    Generate plots for stage one regression models of the model class
                
                
                st2: boolean default=True
                    Generate plots for stage one regression models of the model class

            #------------------
        """
        estimator_plot=False
        partial_dep_plot=True
        best_estimator_and_part_plots(cls,st1,st2,estimator_plot,partial_dep_plot)
    
    @staticmethod
    def isotopes_meteoline_plot(ev_class,iso_class,var_list,iso_18=None,iso_2h=None,a=8,b=10,obs_data=False,residplot=False):
        """
            The method to plot the (meteorological) line between  two features (isotopes)
            that are determined in var_list

            #------------------
            Parameters:

                ev_class: evaluation class
                    evaluation class that contains the second stage models predictions
                
                
                iso_class: model class
                    model class that contains the second stage models
                
                iso_18: none type or Pandas Dataframe default=None
                    First feature (isotope) observed raw data. Ignored if obs_data=False
                
                iso_2h: none type or Pandas Dataframe default=None
                    Second feature (isotope) observed raw data. Ignored if obs_data=False
                
                var_list: list of strings
                    List of strings that identifies the names of two features in the evaluation and model class (in stage two)
                
                a: float default=8
                    Coefficient of the line
                

                b: float default=10
                    Intercept of the line
                
                obs_data: boolean default=False

                    False if iso_18 and iso_2h are not observed data.
                    True if the predictions in evaluation class have an specified date, in "month" field.

                    EXAMPLE:
                        pred_inputs=model_class.all_preds[["CooX","CooY","CooZ","month","ID"]].reset_index()
                        ev_class_obs=tools_copy.evaluation()
                        ev_class_obs.predict(model_class,pred_inputs,direc=direc)
                        tools_copy.plots.isotopes_meteoline_plot(ev_class_obs,model_class,var_list=['is1','is2'],obs_data=True)
                
                residplot: boolean default=False
                    Ignored if month_data=False. It create residual plots in each month for each ID.

            #------------------
        """
        if var_list[0]=="iso_18": iso18_st=f'\N{GREEK SMALL LETTER DELTA}\N{SUPERSCRIPT ONE}\N{SUPERSCRIPT EIGHT}O'
        else:  iso18_st= var_list[0]
        if var_list[1] =="iso_2h":  iso2h_st=f'\N{GREEK SMALL LETTER DELTA}\N{SUPERSCRIPT TWO}H'
        else: iso2h_st=var_list[1]

        vsmow_st='(\u2030 VSMOW)'

        tot=list()
        Path(os.path.join(ev_class.direc,"isotopes_meteoline_plots")).mkdir(parents=True,exist_ok=True)
        all_preds=iso_class.all_preds
        st2_predicted_month_list=ev_class.st2_predicted_month_list
        for k in st2_predicted_month_list:
            ############
            plt.close("all")
            mpl.style.use("seaborn-whitegrid")
            ###################
            i=ev_class.monthly_st2_output_dic_all_vars_df[k]

            i["month"]=k
            tot.append(i)
            if i.shape[0]==0:
                continue
            ###################
            v0="predicted_"+str(var_list[0])
            v1="predicted_"+str(var_list[1])
            #i.rename(columns={v0:"predicted_iso18",v1:"predicted_iso2h"},inplace=True)
            if obs_data==True:
                iso_18["month"]=pd.to_datetime(iso_18['Date']).dt.month
                iso_2h["month"]=pd.to_datetime(iso_2h['Date']).dt.month

            fig, ax = plt.subplots()
            
            #reg = LinearRegression().fit(i[v0].to_frame().values,i[v1].to_frame().values)
            
            
            reg = regress2(i[v0].to_frame().values, i[v1].to_frame().values, _method_type_2="reduced major axis")
        
            if obs_data==True:

                ax.scatter(iso_18[iso_18["month"]==k]["Value"],iso_2h[iso_2h["month"]==k]["Value"],marker="x",c="g",label="Original",s=18)
                s = pd.concat([df.columns.to_series() for df in (i, all_preds)])
                
                # keep all duplicates only, then extract unique names
                res = s[s.duplicated(keep=False)].unique()
                res=list(res)
                try:
                    res.remove("index")
                except:
                    pass    

                merged_all=pd.merge(i,all_preds[all_preds["month"]==k],on=res)
                ####################
                evenly_spaced_interval = np.linspace(0, 1, merged_all.shape[0] )
                colors = [cm.rainbow(x) for x in evenly_spaced_interval]
                for st, color in enumerate(colors):
                    ax.scatter(merged_all[v0][st],merged_all[v1][st],marker=".",c=np.array([color]),s=90)
                    ax.scatter(merged_all[var_list[0]][st],merged_all[var_list[1]][st],marker="^",c=np.array([color]),s=20)

            else:
                ax.scatter(i[v0],i[v1],marker=".",c="b",label="Estimated",s=70)
                ax.scatter(all_preds[all_preds["month"]==k][var_list[0]],all_preds[all_preds["month"]==k][var_list[1]],marker="^",c="c",label="Observed - Monthly",s=18)
            #################
            #Estimated LMWL & GMWL
            left, right = ax.get_xlim()
            left1, right1 = ax.get_ylim()
            met_x = np.linspace(min(left,left1),max(right,right1),100)
            met_y= a*met_x+b
            met_y_preds=reg['slope']*met_x+reg['intercept']
            ax.plot(met_x,met_y,color="red",label="GMWL: "+iso2h_st+"="+str(a)+iso18_st+"+"+str(b)) #GMWL

            ax.plot(met_x,met_y_preds.reshape(-1,1),linestyle='dashed',c="brown",label="Estimated LMWL:"+iso2h_st+"="+str(round(reg['slope'],1))+iso18_st+"+"+str(round(reg['intercept'],1)))
            #################
            #Observed LMWL
            
            #reg2=LinearRegression().fit(all_preds[all_preds["month"]==k][var_list[0]].to_frame(),all_preds[all_preds["month"]==k][var_list[1]].to_frame())
            reg2 = regress2(all_preds[all_preds["month"]==k][var_list[0]],all_preds[all_preds["month"]==k][var_list[1]], _method_type_2="reduced major axis")
        
            
            met_y_preds_2=reg2['slope']*met_x+reg2['intercept']
            
            ax.plot(met_x,met_y_preds_2.reshape(-1,1),linestyle='dashdot',c="yellow",label="Observed LMWL:"+iso2h_st+"="+str(round(reg2['slope'],1))+iso18_st+"+"+str(round(reg2['intercept'],1)))

            ax.set_xlim(left,right)
            ax.set_ylim(left1,right1)
            #score to r2_score
            #ax.set_title("M: " + str(k)+ " |R2 org: " +str(round(iso_class.iso18_bests_dic[0]["best_score"],2))+" |R2 Pred. to Met.: " +str(round(r2_score(y_true, y_pred),2))+ " |R2 Mean monthly obs.: " +str(round(reg.score(i[v0].to_frame().values,j[v1].to_frame().values),2))+ " |R2 Pred. to New Met. Line: " +str(round(reg2.score(all_preds[all_preds["month"]==k]["iso_18"].to_frame(),all_preds[all_preds["month"]==k][var_list[1]].to_frame()),2)) )
            ax.set_xlabel(iso18_st+vsmow_st)
            ax.set_ylabel(iso2h_st+vsmow_st)
            ax.legend()
            fig.savefig(os.path.join(ev_class.direc,"isotopes_meteoline_plots","Month_" + str(k)+"_plot.png"),dpi=300)
            plt.close(fig)
            #residuals plot
            if residplot==True and obs_data==True:
                merged_all.drop_duplicates(inplace=True,ignore_index=True)
                resds_df=merged_all
                resid_var0=str("residual_")+var_list[0]
                resid_var1=str("residual_")+var_list[1]
                resds_df[resid_var0]=resds_df[var_list[0] ] - resds_df[v0] 
                resds_df=resds_df.sort_values(by=resid_var0)
                plt.scatter(np.arange(len(resds_df['ID'])),  resds_df[resid_var0] )
                ax = plt.gca()
                ax.set_xticks(np.arange(len(resds_df['ID'])))
                ax.xaxis.set_ticklabels(resds_df['ID'], rotation=90)
                
                mean_obs=round(resds_df[var_list[0] ].mean() ,2)
                mean_pred=round(resds_df[v0].mean(),2)
                mean_resids=round(resds_df[resid_var0].mean(),2)

                std_obs=round(resds_df[var_list[0] ].std(),2)
                std_pred=round(resds_df[v0].std(),2)
                std_resids=round(resds_df[resid_var0].std(),2)
                tit=iso18_st+"|mn_resid: {} std_resid: {} mn_obs: {} std_obs: {} mn_pred: {} std_pred:{}".format(mean_resids, std_resids, mean_obs, std_obs, mean_pred, std_pred)
                #plt.title(tit)
                fig = plt.gcf()
                fig.savefig(os.path.join(ev_class.direc,"isotopes_meteoline_plots",resid_var0+"_Month_" + str(k)+"_plot.png"), bbox_inches = "tight")
                plt.close("all")

                resds_df[resid_var1]=resds_df[var_list[1] ] - resds_df[v1]
                resds_df=resds_df.sort_values(by=resid_var1)
                plt.scatter(np.arange(len(resds_df['ID'])),  resds_df[resid_var1] )
                ax = plt.gca()
                ax.set_xticks(np.arange(len(resds_df['ID'])))
                ax.xaxis.set_ticklabels(resds_df['ID'], rotation=90)
                mean_obs=round(resds_df[var_list[1] ].mean() ,2)
                mean_pred=round(resds_df[v1].mean(),2)
                mean_resids=round(resds_df[resid_var1].mean(),2)

                std_obs=round(resds_df[var_list[1] ].std(),2)
                std_pred=round(resds_df[v1].std(),2)
                std_resids=round(resds_df[resid_var1].std(),2)

                tit=iso2h_st+"|mn_resid: {} std_resid: {} mn_obs: {} std_obs: {} mn_pred: {} std_pred:{}".format(mean_resids, std_resids, mean_obs, std_obs, mean_pred, std_pred)
                #plt.title(tit)
                fig = plt.gcf()
                fig.savefig(os.path.join(ev_class.direc,"isotopes_meteoline_plots",resid_var1+"_Month_" + str(k)+"_plot.png"), bbox_inches = "tight")
                plt.close("all")





        if len(tot)!=0:
            

            vv=pd.concat(tot)
            mpl.style.use("seaborn-whitegrid")

            plt.scatter(vv[v0],vv[v1],marker=".",c="b",label="Estimated",s=70)
            #reg = LinearRegression().fit(vv[v0].to_frame().values,vv[v1].to_frame().values)
            
            reg = regress2(vv[v0].to_frame().values,vv[v1].to_frame().values, _method_type_2="reduced major axis")
            #print ("Method: RMA", "slope: ",reg['slope'],"intercept: ", reg['intercept'],"r: ",reg['r'])
            
            if obs_data==True:
                plt.scatter(iso_18[iso_18["month"].isin(st2_predicted_month_list)]["Value"],iso_2h[iso_2h["month"].isin(st2_predicted_month_list)]["Value"],marker="x",c="g",label="Original",s=18)
            plt.scatter(all_preds[all_preds["month"].isin(st2_predicted_month_list)][var_list[0]],all_preds[all_preds["month"].isin(st2_predicted_month_list)][var_list[1]],marker="^",c="c",label="Observed - Monthly",s=18)
            left, right = plt.xlim()
            left1, right1 = plt.ylim()
            met_x = np.linspace(min(left,left1),max(right,right1),100)
            met_y= a*met_x+b
            met_y_preds=reg['slope']*met_x+reg['intercept']
            plt.plot(met_x,met_y,color="red",label="GMWL: "+iso2h_st+"="+str(a)+iso18_st+"+"+str(b))
            plt.plot(met_x,met_y_preds.reshape(-1,1),linestyle='dashed',c="brown",label="Estimated LMWL: "+iso2h_st+"="+str(round(reg['slope'],1))+iso18_st+"+"+str(round(reg['intercept'],1)))
            #reg2 = LinearRegression().fit(all_preds[all_preds["month"].isin(st2_predicted_month_list)][var_list[0]].to_frame(),all_preds[all_preds["month"].isin(st2_predicted_month_list)][var_list[1]].to_frame())
            
            reg2 = regress2(all_preds[all_preds["month"].isin(st2_predicted_month_list)][var_list[0]],all_preds[all_preds["month"].isin(st2_predicted_month_list)][var_list[1]], _method_type_2="reduced major axis")
            #print ("Method: RMA", "slope: ",reg2['slope'],"intercept: ", reg2['intercept'],"r: ",reg2['r'])
        
            met_y_preds_2=reg2['slope']*met_x+reg2['intercept']
            plt.plot(met_x,met_y_preds_2.reshape(-1,1),linestyle='dashdot',c="yellow",label="Observed LMWL: "+iso2h_st+"="+str(round(reg2['slope'],1))+iso18_st+"+"+str(round(reg2['intercept'],1)))
            plt.xlim(left,right)
            plt.ylim(left1,right1)
            plt.xlabel(iso18_st+vsmow_st)
            plt.ylabel(iso2h_st+vsmow_st)
            R2_Pred_Met=round(r2_score(8*vv[v0] + 10, vv[v1]),2)

            #score to r2score
            v1_on_the_line=reg['slope']*vv[v0]+reg['intercept']
            _score_=r2_score(vv[v1],v1_on_the_line)

            R2_Pred_NewMet=round(_score_,2)

            #score to r2score
            reg2_on_the_line=reg2['slope']*all_preds[all_preds["month"].isin(st2_predicted_month_list)][var_list[0]]+reg2['intercept']
            _score_=r2_score(all_preds[all_preds["month"].isin(st2_predicted_month_list)][var_list[1]],reg2_on_the_line)


            R2_Pred_obs=round(_score_,2)
            
            
            tit=("All| R2 Pred_Met: {}"
                "|R2 Pred_NewMet: {}" 
                "|R2 Pred_obs: {}|"  
                ).format(R2_Pred_Met,R2_Pred_NewMet,R2_Pred_obs)


            if residplot==True and obs_data==True:
                s = pd.concat([df.columns.to_series() for df in (vv,all_preds)])
                # keep all duplicates only, then extract unique names
                res = s[s.duplicated(keep=False)].unique()
                res=list(res)
                resds_df=pd.merge(vv,all_preds,on=res)
                resds_df.drop_duplicates(inplace=True,ignore_index=True)
                resds_df[resid_var0]=resds_df[var_list[0] ] - resds_df[v0] 
                resds_df[resid_var1]=resds_df[var_list[1] ] - resds_df[v1]

                mean_obs=round(resds_df[var_list[0] ].mean() ,2)
                mean_pred=round(resds_df[v0].mean(),2)
                mean_resids=round(resds_df[resid_var0].mean(),2)

                std_obs=round(resds_df[var_list[0] ].std(),2)
                std_pred=round(resds_df[v0].std(),2)
                std_resids=round(resds_df[resid_var0].std(),2)
                tit_18=var_list[0] +"mn_resid: {} std_resid: {} mn_obs: {} std_obs: {} mn_pred: {} std_pred:{}|".format(mean_resids, std_resids, mean_obs, std_obs, mean_pred, std_pred)
                
                mean_obs=round(resds_df[var_list[1] ].mean() ,2)
                mean_pred=round(resds_df[v1].mean(),2)
                mean_resids=round(resds_df[resid_var1].mean(),2)

                std_obs=round(resds_df[var_list[1] ].std(),2)
                std_pred=round(resds_df[v1].std(),2)
                std_resids=round(resds_df[resid_var1].std(),2)

                tit_2h=var_list[1]+"mn_resid: {} std_resid: {} mn_obs: {} std_obs: {} mn_pred: {} std_pred:{}|".format(mean_resids, std_resids, mean_obs, std_obs, mean_pred, std_pred)
                
                from sklearn.metrics import mean_absolute_error, mean_squared_error
                mae18=mean_absolute_error(resds_df[var_list[0] ] , resds_df[v0] )
                mse18= mean_squared_error(resds_df[var_list[0] ] , resds_df[v0] )

                mae2h=mean_absolute_error(resds_df[var_list[1] ] , resds_df[v1] )
                mse2h=mean_squared_error(resds_df[var_list[1] ] , resds_df[v1] )
                tit_error=var_list[0]+"mae: {:.2f} mse: {:.2f} |".format(mae18,mse18)+var_list[1]+ "mae: {:.2f} mse: {:.2f}|".format(mae2h,mse2h)
                m_out_f=open(os.path.join(ev_class.direc,"isotopes_meteoline_plots","all_range_plot_residuals_&_errors.txt"),'w')
                m_out_f.write(tit+str("\n#######\n")+tit_error+str("\n#######\n")+tit_18+str("\n#######\n")+tit_2h)
                m_out_f.close()



                tit=tit+tit_error+tit_18+tit_2h

            #plt.title(tit, fontsize=10,wrap=True)
            plt.legend()
            plt.savefig(os.path.join(ev_class.direc,"isotopes_meteoline_plots","all_range_plot.png"),dpi=300, bbox_inches = "tight")
            plt.close("all")

    @staticmethod
    def map_generator(ev_class,feat_list,observed_class_list=None,month_list=None,unit_list=None,opt_title_list=None,x="CooX",y="CooY",shp_file=None,html=True,direc=None,minus_to_zero_list=None,max_to_hundred_list=None):
        """
            The method to generate the maps (.png and HTML) of the evaluation class

            #------------------
            Parameters:

                ev_class: evaluation class
                    Evaluation class that contains the second stage models predictions
                
                
                feat_list: list
                    List of strings that identifies the desired features to map  
                
                observed_class_list: none type or list default=None
                    List of the preprocess classes of the observed data. No observed data will be shown in the maps if  observed_class_list=None, or an element of the list is none.
                
                month_list: none type or list default=None
                    List of the desired month to generate the maps. If None, the maps will be generated for all the months available in evaluation class
                
                unit_list: list of strings default=None
                    List of strings that identifies the units to be shown for every feature in the generated maps
                
                opt_title_list: list of strings default=None
                    List of strings that identifies the titles to be shown for every feature in the generated maps
                

                x: string default="CooX"
                    Identifies the name of the x (longitude) field in the evaluation class (same as defined in preprocess classess)
                
                y: string default="CooY"
                    Identifies the name of the y (latitude) field in the evaluation class (same as defined in preprocess classess)
                
                shp_file: none type or string default=None
                    Directory to the shape file to be used in .png maps. If None, no shape file will be included in the maps.If shapefile exists, it has to be in the same coordination system as the x & y.
                
                html: boolean default=True
                    If True, an HTML version of the maps will be created
                
                direc: none type or string default=None
                    The new directory to store the maps. If None, a new folder will be created in the directory that determined in the evaluation class
                
                minus_to_zero_list: none type or list default=None
                    If minus_to_zero_list is a list of booleans, when it is True, replace the minus values with zero for that feature. Usage in features such as relative humidity.

                max_to_hundred_list: none type or list default=None
                    If max_to_hundred_list is a list of booleans, when it is True, replace the values more that 100 with 100 for that feature. Usage in features such as relative humidity.
            #------------------
        """

        if month_list==None: month_list=list(ev_class.monthly_st2_output_dic_all_vars_df.keys())
        for month in month_list:
            if month in  list(ev_class.monthly_st2_output_dic_all_vars_df.keys()):
                for n,feat in enumerate(feat_list):
                    if unit_list==None or unit_list[n]==None:unit=""
                    else: unit=str(unit_list[n])

                    if opt_title_list==None or opt_title_list[n]==None:opt_title=None
                    else: opt_title=str(opt_title_list[n])

                    if observed_class_list==None or observed_class_list[n]==None:
                        observed_data=None
                    else: observed_data=observed_class_list[n].month_grouped_inp_var[month-1]

                    ls1=list(ev_class.monthly_st2_output_dic_all_vars_df[month].columns)
                    #because sometimes we add the "predicted_" prefix to the features
                    pred_feat=None
                    if feat in ls1:
                        pred_feat=False
                    if "predicted_"+feat in ls1:
                        pred_feat=True
                        feat="predicted_"+feat
                    if pred_feat!=None:

                        print ('data for month{} and feature {} exists. creating maps'.format(month,feat))
                        
                        df=copy.deepcopy(ev_class.monthly_st2_output_dic_all_vars_df[month])
                        if minus_to_zero_list!=None:
                            if minus_to_zero_list[n]==True:
                                df.loc[df[feat]<=0, feat] = 0
                        if max_to_hundred_list!=None:
                            if max_to_hundred_list[n]==True:
                                df.loc[df[feat]>100, feat] = 100

                        if html ==True:

                            if direc==None:
                                Path(os.path.join(ev_class.direc,str(feat)+"_maps")).mkdir(parents=True,exist_ok=True)
                                dir_bokeh=os.path.join(ev_class.direc,str(feat)+"_maps",str(feat)+"_Month_" + str(month)+"_interactive.html")
                            else:
                                dir_bokeh=direc  
                            print (df.columns)  
                            isocompy.create_maps.create_maps_bokeh(df,feat=feat,CooX=x,CooY=y,dir=dir_bokeh,unit=unit,opt_title=opt_title,observed_data=observed_data)

                        #.png
                        if direc==None:
                            Path(os.path.join(ev_class.direc,str(feat)+"_maps")).mkdir(parents=True,exist_ok=True)
                            dir_gpd=os.path.join(ev_class.direc,str(feat)+"_maps",str(feat)+"_Month_" + str(month)+"_fig.png")
                        else:
                            dir_gpd=direc

                        isocompy.create_maps.make_maps_gpd(df,shp_dir=shp_file,feat=feat,CooX=x,CooY=y,dir=dir_gpd,unit=unit,opt_title=opt_title,observed_data=observed_data)




