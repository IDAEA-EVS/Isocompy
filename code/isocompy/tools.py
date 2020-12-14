import os
import dill
from pathlib import Path
from datetime import datetime
from isocompy import isocompy_tool_funcs
from isocompy.isocompy_tool_funcs import new_data_prediction_comparison,regional_mensual_plot,f_reg_mutual,best_estimator_and_part_plots
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import numpy as np
import itertools #do not remove it. ZIP used!
import matplotlib as mpl
from sklearn.linear_model import LinearRegression

class session(object):

    @staticmethod
    def save(self,name="isocompy_saved_object"):
        dateTimeObj = datetime.now().strftime("_%d_%b_%Y_%H_%M")
        try:
            self.hum_preds_real_dic

            try:
                self.iso2h_bests
                filename=os.path.join(self.direc,name+dateTimeObj+"_meteoTrue_isoTrue.pkl")

            except:    
                filename=os.path.join(self.direc,name+dateTimeObj+"_meteoTrue_isoFalse.pkl")
        
        except: 

            filename=os.path.join(self.direc,name+dateTimeObj+"_meteoFalse_isoFalse.pkl")
        with open(filename, 'wb') as filee:
            dill.dump(obj=self,file=filee)
        print ("\n\n pkl saved object directory: \n\n",filename)
        return filename
    ##########################################################################################
    @staticmethod
    def load(direc):
        with open(direc, 'rb') as in_strm:
            obj=dill.load(in_strm)
        return obj
        
    @staticmethod
    def save_session(direc,name="isocompy_saved_session", *argv):
        dateTimeObj = datetime.now().strftime("_%d_%b_%Y_%H_%M")
        filename=os.path.join(direc,name+dateTimeObj+".pkl")
        for arg in argv:
            arg
        dill.dump_session(filename)   
        print ("\n\n pkl saved session directory: \n\n",filename)

    @staticmethod
    def load_session(dir):
        return dill.load_session(dir)

class evaluation(object):
    
    def predict(self,cls,pred_inputs,direc=None):
        if direc==None: self.direc=cls.direc
        else: self.direc=direc
        Path(self.direc).mkdir(parents=True, exist_ok=True)
        #############################################################
        self.pred_inputs=pred_inputs
        #here and iso_prediction, identifying the month should be nicer!
        column_name="predicted_iso18"
        self.monthly_iso18_output=isocompy_tool_funcs.predict_points(self.direc,cls.iso18_bests_dic[0]["used_features"],self.pred_inputs,cls.iso_model_month_list,cls.temp_bests,cls.rain_bests,cls.hum_bests,cls.iso18_bests[0][6],cls.iso18_bests[0][7],cls.iso18_bests_dic[0]["didlog"],cls.iso18_bests_dic[0]["best_estimator"],column_name,trajectory_features_list=cls.col_for_f_reg,run_iso_whole_year=cls.run_iso_whole_year)
        column_name="predicted_iso2h"
        self.monthly_iso2h_output=isocompy_tool_funcs.predict_points(self.direc,cls.iso2h_bests_dic[0]["used_features"],self.pred_inputs,cls.iso_model_month_list,cls.temp_bests,cls.rain_bests,cls.hum_bests,cls.iso2h_bests[0][6],cls.iso2h_bests[0][7],cls.iso2h_bests_dic[0]["didlog"],cls.iso2h_bests_dic[0]["best_estimator"],column_name,trajectory_features_list=cls.col_for_f_reg,run_iso_whole_year=cls.run_iso_whole_year)
        column_name="predicted_iso3h"
        self.monthly_iso3h_output=isocompy_tool_funcs.predict_points(self.direc,cls.iso3h_bests_dic[0]["used_features"],self.pred_inputs,cls.iso_model_month_list,cls.temp_bests,cls.rain_bests,cls.hum_bests,cls.iso3h_bests[0][6],cls.iso3h_bests[0][7],cls.iso3h_bests_dic[0]["didlog"],cls.iso3h_bests_dic[0]["best_estimator"],column_name,trajectory_features_list=cls.col_for_f_reg,run_iso_whole_year=cls.run_iso_whole_year)
    ############################################################
    def regional_mensual_plot():
        pass
        #regional_mensual_plot
        #regional_mensual_plot(x_y_z_,monthly_iso18_output,monthly_iso2h_output)
    ############################################################

    def new_inp_compare():   

        #new_data_prediction_comparison
        data_fl=r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\inputs\new_measured.xlsx"
        newd=pd.read_excel(data_fl,sheet_name=0,header=0,index_col=False,keep_default_na=True)
        new_data_prediction=new_data_prediction_comparison(newd,iso_model_month_list,temp_bests,rain_bests,hum_bests,didlog_iso18,didlog_iso2h,x_scaler_iso18,x_scaler_iso2h,used_features_iso18,used_features_iso2h,best_estimator_all_iso18,best_estimator_all_iso2h,y_scaler_iso18,y_scaler_iso2h)
        #writing isotope predictions into a file
        pd.concat([all_preds,Y_preds_iso18,Y_preds_iso2h],axis=1).to_excel(r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\output\isotope_main_data_predictions.xlsx")
        #return Y_preds_iso18,Y_preds_iso2h,rain_preds_real,hum_preds_real,temp_preds_real,temp_bests,rain_bests,hum_bests,monthly_iso2h_output,monthly_iso18_output,x_scaler_iso18,y_scaler_iso18,didlog_iso18,used_features_iso18,rsquared_iso18,best_estimator_all_iso18,best_score_all_iso18,x_scaler_iso2h,y_scaler_iso2h,didlog_iso2h,used_features_iso2h,rsquared_iso2h,best_estimator_all_iso2h,best_score_all_iso2h,predictions_monthly_list, all_preds,month_grouped_list_with_zeros_iso_18,month_grouped_list_with_zeros_iso_2h,month_grouped_list_with_zeros_iso_3h,month_grouped_list_with_zeros_hum,month_grouped_list_with_zeros_rain,month_grouped_list_with_zeros_temp,rain,temp,hum,elnino,lanina,iso_18,iso_2h,iso_3h
        ###########


class  stats(object):   
    @staticmethod
    def annual_stats(model_cls_obj,params="all"): #params=all, meteo, iso
        list_of_dics_meteo=[
            {"inputs":["CooX","CooY","CooZ"],"outputs":["temp"]},
            {"inputs":["CooX","CooY","CooZ"],"outputs":["hum"]},
            {"inputs":["CooX","CooY","CooZ"],"outputs":["rain"]}]
        list_of_dics_iso=[
            {"inputs":["CooX","CooY","CooZ","temp","rain","hum"]+model_cls_obj.col_for_f_reg,"outputs":["iso_2h"]},
            {"inputs":["CooX","CooY","CooZ","temp","rain","hum"]+model_cls_obj.col_for_f_reg,"outputs":["iso_18"]},
            {"inputs":["CooX","CooY","CooZ","temp","rain","hum"]+model_cls_obj.col_for_f_reg,"outputs":["iso_3h"]}]
        
        if params=="all": list_of_dics_for_stats=list_of_dics_meteo+list_of_dics_iso
        elif params=="meteo": list_of_dics_for_stats=list_of_dics_meteo
        else: list_of_dics_for_stats=list_of_dics_iso #params=iso
        
        f_reg_mutual( os.path.join(model_cls_obj.direc,"annual_statistics_f_test_MI.txt") ,model_cls_obj.all_preds,list_of_dics_for_stats)   
    
    @staticmethod
    def mensual_stats(model_cls_obj,params="all"): #params=all, meteo, iso
        list_of_dics_meteo=[
            {"inputs":["CooX","CooY","CooZ"],"outputs":["temp"]},
            {"inputs":["CooX","CooY","CooZ"],"outputs":["hum"]},
            {"inputs":["CooX","CooY","CooZ"],"outputs":["rain"]}]
        list_of_dics_iso=[
            {"inputs":["CooX","CooY","CooZ","temp","rain","hum"]+model_cls_obj.col_for_f_reg,"outputs":["iso_2h"]},
            {"inputs":["CooX","CooY","CooZ","temp","rain","hum"]+model_cls_obj.col_for_f_reg,"outputs":["iso_18"]},
            {"inputs":["CooX","CooY","CooZ","temp","rain","hum"]+model_cls_obj.col_for_f_reg,"outputs":["iso_3h"]}]
        
        if params=="all": list_of_dics_for_stats=list_of_dics_meteo+list_of_dics_iso
        elif params=="meteo": list_of_dics_for_stats=list_of_dics_meteo
        else : list_of_dics_for_stats=list_of_dics_iso #params=="iso"

        all_preds_month=model_cls_obj.all_preds["month"].copy()
        all_preds_month.drop_duplicates(keep = 'last', inplace = True)
        for mn in all_preds_month:
            all_preds_temp=model_cls_obj.all_preds[model_cls_obj.all_preds["month"]==mn]
            Path(os.path.join(model_cls_obj.direc,"mensual_statistics")).mkdir(parents=True, exist_ok=True)
            file_name= os.path.join(model_cls_obj.direc,"mensual_statistics","month_"+str(mn)+"mensual_statistics_f_test_MI.txt")
            f_reg_mutual(file_name,all_preds_temp,list_of_dics_for_stats)

class plots(object):
    @staticmethod
    def best_estimator_plots(cls,meteo_plot=True,iso_plot=True):
        estimator_plot=True
        partial_dep_plot=False
        best_estimator_and_part_plots(cls,meteo_plot,iso_plot,estimator_plot,partial_dep_plot)
    @staticmethod
    def partial_dep_plots(cls,meteo_plot=True,iso_plot=True):
        estimator_plot=False
        partial_dep_plot=True
        best_estimator_and_part_plots(cls,meteo_plot,iso_plot,estimator_plot,partial_dep_plot)
    
    @staticmethod
    def isotopes_meteoline_plot(ev_class,iso_class,iso_18,iso_2h,a=8,b=10,month_data=False,obs_data=True,id_point=True,residplot=True):
        """
        docstring
        """
        tot=list()
        Path(os.path.join(ev_class.direc,"isotopes_meteoline_plots")).mkdir(parents=True,exist_ok=True)
        for (i,j,k) in zip(ev_class.monthly_iso18_output,ev_class.monthly_iso2h_output,iso_class.iso_model_month_list):
            mpl.style.use("seaborn")
            if id_point==True and month_data==True:
                merged_i1_2h=pd.merge(i,j,on=["CooX","CooY",'CooZ', 'month','ID_MeteoPoint'])
            elif month_data==True:
                merged_i1_2h=pd.merge(i,j,on=["CooX","CooY",'CooZ', 'month'])
            else:
                merged_i1_2h=pd.merge(i,j,on=["CooX","CooY",'CooZ'])
            #print (i['ID_MeteoPoint'].shape)
            #print (j['ID_MeteoPoint'].shape)
            #print (merged_i1_2h['ID_MeteoPoint'].shape)
            i=merged_i1_2h
            j=merged_i1_2h
            if month_data==True:
                i=i[i["month"]==k]
                j=j[j["month"]==k]
                tot.append(i)
            else:
                tot.append(merged_i1_2h)

            plt.scatter(i.predicted_iso18,j.predicted_iso2h,marker=".",c="b",label="Predicted",s=70)
            reg = LinearRegression().fit(i.predicted_iso18.to_frame().values,j.predicted_iso2h.to_frame().values)
            if obs_data==True:
                plt.scatter(iso_18[iso_18["month"]==k]["Value"],iso_2h[iso_2h["month"]==k]["Value"],marker="x",c="g",label="Original",s=18)
            plt.scatter(iso_class.all_preds[iso_class.all_preds["month"]==k]["iso_18"],iso_class.all_preds[iso_class.all_preds["month"]==k]["iso_2h"],marker="^",c="c",label="Monthly",s=18)
            left, right = plt.xlim()
            left1, right1 = plt.ylim()
            met_x = np.linspace(min(left,left1),max(right,right1),100)
            met_y= a*met_x+b
            met_y_preds=reg.coef_*met_x+reg.intercept_
            plt.plot(met_x,met_y,color="red",label="Met.Line="+str(a)+"x+"+str(b))
            plt.plot(met_x,met_y_preds.reshape(-1,1),linestyle='dashed',c="brown",label="Pred. Met. Line="+str(round(reg.coef_[0][0],1))+"x+"+str(round(reg.intercept_[0],1)))
            reg2=LinearRegression().fit(iso_class.all_preds[iso_class.all_preds["month"]==k]["iso_18"].to_frame(),iso_class.all_preds[iso_class.all_preds["month"]==k]["iso_2h"].to_frame())
            met_y_preds_2=reg2.coef_*met_x+reg2.intercept_
            plt.plot(met_x,met_y_preds_2.reshape(-1,1),linestyle='dashdot',c="yellow",label="Mean monthly obs. Line="+str(round(reg2.coef_[0][0],1))+"x+"+str(round(reg2.intercept_[0],1)))

            y_true = a * i.predicted_iso18 + b
            y_pred = j.predicted_iso2h
            #plt.plot(a,y_true,color="red")
            plt.xlim(left,right)
            plt.ylim(left1,right1)
            plt.title("M: " + str(k)+ " |R2 org: " +str(round(iso_class.iso18_bests_dic[0]["best_score"],2))+" |R2 Pred. to Met.: " +str(round(r2_score(y_true, y_pred),2))+ " |R2 Mean monthly obs.: " +str(round(reg.score(i.predicted_iso18.to_frame().values,j.predicted_iso2h.to_frame().values),2))+ " |R2 Pred. to New Met. Line: " +str(round(reg2.score(iso_class.all_preds[iso_class.all_preds["month"]==k]["iso_18"].to_frame(),iso_class.all_preds[iso_class.all_preds["month"]==k]["iso_2h"].to_frame()),2)) )
            plt.xlabel("iso_18")
            plt.ylabel("iso_2H")
            plt.legend()
            plt.savefig(os.path.join(ev_class.direc,"isotopes_meteoline_plots","Month_" + str(k)+"_plot.png"),dpi=300)
            plt.close()
            #residuals plot
            if id_point==True and residplot==True:
                resds_df=pd.merge(i,iso_class.all_preds[iso_class.all_preds["month"]==k],on=['ID_MeteoPoint'])
                resds_df["residual_18"]=resds_df['iso_18' ] - resds_df["predicted_iso18"] 
                print (resds_df['ID_MeteoPoint'])
                resds_df=resds_df.sort_values(by="residual_18")
                plt.scatter(np.arange(len(resds_df['ID_MeteoPoint'])),  resds_df["residual_18"] )
                ax = plt.gca()
                ax.set_xticks(np.arange(len(resds_df['ID_MeteoPoint'])))
                ax.xaxis.set_ticklabels(resds_df['ID_MeteoPoint'], rotation=90)
                plt.savefig(os.path.join(ev_class.direc,"isotopes_meteoline_plots","Residuals_iso18_Month_" + str(k)+"_plot.png"))
                plt.title("Residuals_iso18_Month_" + str(k))
                plt.close()


                resds_df["residual_2h"]=resds_df['iso_2h' ] - resds_df["predicted_iso2h"]
                plt.scatter(np.arange(len(resds_df['ID_MeteoPoint'])),  resds_df["residual_2h"] )
                ax = plt.gca()
                ax.set_xticks(np.arange(len(resds_df['ID_MeteoPoint'])))
                ax.xaxis.set_ticklabels(resds_df['ID_MeteoPoint'], rotation=90)
                plt.savefig(os.path.join(ev_class.direc,"isotopes_meteoline_plots","Residuals_iso2h_Month_" + str(k)+"_plot.png"))
                plt.title("Residuals_iso2h_Month_" + str(k))
                plt.close()







        vv=pd.concat(tot)
        vv2=pd.concat(tot)
        mpl.style.use("seaborn")
        plt.scatter(vv.predicted_iso18,vv2.predicted_iso2h,marker=".",c="b",label="Predicted",s=70)
        reg = LinearRegression().fit(vv.predicted_iso18.to_frame().values,vv2.predicted_iso2h.to_frame().values)
        if obs_data==True:
            plt.scatter(iso_18[iso_18["month"].isin(iso_class.iso_model_month_list)]["Value"],iso_2h[iso_2h["month"].isin(iso_class.iso_model_month_list)]["Value"],marker="x",c="g",label="Original",s=18)
        plt.scatter(iso_class.all_preds[iso_class.all_preds["month"].isin(iso_class.iso_model_month_list)]["iso_18"],iso_class.all_preds[iso_class.all_preds["month"].isin(iso_class.iso_model_month_list)]["iso_2h"],marker="^",c="c",label="Monthly",s=18)
        left, right = plt.xlim()
        left1, right1 = plt.ylim()
        met_x = np.linspace(min(left,left1),max(right,right1),100)
        met_y= a*met_x+b
        met_y_preds=reg.coef_*met_x+reg.intercept_
        plt.plot(met_x,met_y,color="red",label="Met.Line="+str(a)+"x+"+str(b))
        plt.plot(met_x,met_y_preds.reshape(-1,1),linestyle='dashed',c="brown",label="Pred. Met. Line="+str(round(reg.coef_[0][0],1))+"x+"+str(round(reg.intercept_[0],1)))
        reg2 = LinearRegression().fit(iso_class.all_preds[iso_class.all_preds["month"].isin(iso_class.iso_model_month_list)]["iso_18"].to_frame(),iso_class.all_preds[iso_class.all_preds["month"].isin(iso_class.iso_model_month_list)]["iso_2h"].to_frame())
        met_y_preds_2=reg2.coef_*met_x+reg2.intercept_
        plt.plot(met_x,met_y_preds_2.reshape(-1,1),linestyle='dashdot',c="yellow",label="Mean monthly obs. Line="+str(round(reg2.coef_[0][0],1))+"x+"+str(round(reg2.intercept_[0],1)))
        plt.xlim(left,right)
        plt.ylim(left1,right1)
        plt.xlabel("iso_18")
        plt.ylabel("iso_2H")
        plt.title("All Range | R2 Pred. to Met. Line: " +str(round(r2_score(8*vv.predicted_iso18 + 10, vv2.predicted_iso2h),2))+ " | R2 Pred. to New Met. Line: " +str(round(reg.score(vv.predicted_iso18.to_frame().values,vv2.predicted_iso2h.to_frame().values),2))+ " |  R2 Pred. to New Met. Line: " +str(round(reg2.score(iso_class.all_preds[iso_class.all_preds["month"].isin(iso_class.iso_model_month_list)]["iso_18"].to_frame(),iso_class.all_preds[iso_class.all_preds["month"].isin(iso_class.iso_model_month_list)]["iso_2h"].to_frame()),2) ))
        plt.legend()
        plt.savefig(os.path.join(ev_class.direc,"isotopes_meteoline_plots","all_range_plot.png"),dpi=300)
        plt.close()