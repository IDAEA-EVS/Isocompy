import pandas as pd
#from pathlib import Path
import os
#class for data preparation
'''class prep(object):

    def __init__(self,direc,meteo_input_type="daily_remove_outliers",write_outliers_input=True,year_type="all",
    inc_zeros=False,write_integrated_data=True,q1=0.05,q3=0.95,IQR_rain=True,IQR_temp=False,IQR_hum=False,IQR_rat_rain=3,IQR_rat_temp=3,IQR_rat_hum=3):

        self.meteo_input_type=meteo_input_type
        self.year_type=year_type
        self.inc_zeros=inc_zeros
        self.IQR_rain=IQR_rain
        self.IQR_temp=IQR_temp
        self.IQR_hum=IQR_hum
        self.write_outliers_input=write_outliers_input
        self.write_integrated_data=write_integrated_data
        self.IQR_rat_rain=IQR_rat_rain
        self.IQR_rat_temp=IQR_rat_temp
        self.IQR_rat_hum=IQR_rat_hum
        self.q1=q1
        self.q3=q3
        self.direc=direc
    
    def fit(self,rain,temp,hum,iso_18,iso_2h,iso_3h,elnino=None,lanina=None):

        if elnino==None: elnino=[]
        if lanina==None: lanina=[]
        self.iso_18=iso_18
        self.iso_2h=iso_2h
        self.iso_3h=iso_3h
        self.month_grouped_iso_18,self.month_grouped_iso_2h,self.month_grouped_iso_3h,self.month_grouped_hum,self.month_grouped_rain,self.month_grouped_temp,self.rain,self.temp,self.hum=data_preparation(rain,temp,hum,iso_18,iso_2h,iso_3h,self.direc,self.meteo_input_type,self.q1,self.q3,self.IQR_rain,self.IQR_temp,self.IQR_hum,self.inc_zeros,self.write_outliers_input,self.write_integrated_data,self.IQR_rat_rain,self.IQR_rat_temp,self.IQR_rat_hum,self.year_type,elnino,lanina)
'''
#test
#preped_d=prep().fit(rain,temp,hum,iso_18,iso_2h,iso_3h)
                                      
##########################################################################################
##########################################################################################


#grouping data
def grouping_data(rain,which_value,elnino,lanina,filter_avraged,month=None,zeross=True):
    if zeross==False:
        rain=rain[rain["Value"]!=0]
    if month !=None:
        rain=rain[rain["Date"].dt.month==month] 

    #omiting the top 5 percent of the data
    if filter_avraged==True:
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
        if len(tempp.shape)!= 1:
            #print (tempp)
            sum_elnino=None
            cnt_elnino=None
            sum_lanina=None
            cnt_lanina=None
            sum_norm=None
            cnt_norm=None
            for index2, row2 in tempp.iterrows():
                if pd.to_datetime(row2["Date"]).year in elnino:
                    if sum_elnino==None:
                        sum_elnino=row2["Value"]
                        cnt_elnino=1
                    else:
                        sum_elnino=sum_elnino+row2["Value"]
                        cnt_elnino=cnt_elnino+1

                elif pd.to_datetime(row2["Date"]).year in lanina:
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
                newmat_elnino.append({"ID_MeteoPoint":row['ID_MeteoPoint'], "CooX":tempp["CooX"].iat[0], "CooY":tempp["CooY"].iat[0], "CooZ":tempp["CooZ"].iat[0], "Date":tempp["Date"], which_value:Mean_Value_elnino})
            if sum_lanina !=None:
                Mean_Value_lanina=sum_lanina/cnt_lanina
                newmat_lanina.append({"ID_MeteoPoint":row['ID_MeteoPoint'], "CooX":tempp["CooX"].iat[0], "CooY":tempp["CooY"].iat[0], "CooZ":tempp["CooZ"].iat[0], which_value:Mean_Value_lanina})
            if sum_norm !=None:
                Mean_Value_norm=sum_norm/cnt_norm
                newmat_norm.append({"ID_MeteoPoint":row['ID_MeteoPoint'], "CooX":tempp["CooX"].iat[0], "CooY":tempp["CooY"].iat[0], "CooZ":tempp["CooZ"].iat[0], which_value:Mean_Value_norm})
        else:
            #print ("tempp in len 1:")
            #print (tempp)
            if pd.to_datetime(tempp["Date"]).year in elnino:
                newmat_elnino.append({"ID_MeteoPoint":row['ID_MeteoPoint'], "CooX":tempp["CooX"], "CooY":tempp["CooY"], "CooZ":tempp["CooZ"], which_value:tempp["Value"]})
            elif pd.to_datetime(tempp["Date"]).year in lanina:
                newmat_lanina.append({"ID_MeteoPoint":row['ID_MeteoPoint'], "CooX":tempp["CooX"], "CooY":tempp["CooY"], "CooZ":tempp["CooZ"], which_value:tempp["Value"]})    
            else:
                newmat_norm.append({"ID_MeteoPoint":row['ID_MeteoPoint'], "CooX":tempp["CooX"], "CooY":tempp["CooY"], "CooZ":tempp["CooZ"], which_value:tempp["Value"]})
    
    newmatdf_rain_elnino = pd.DataFrame(newmat_elnino)
    newmatdf_rain_lanina = pd.DataFrame(newmat_lanina)
    newmatdf_rain_norm = pd.DataFrame(newmat_norm)
    return newmatdf_rain_elnino,newmatdf_rain_lanina,newmatdf_rain_norm
###########################################
#function for monthly procedure
def monthly_uniting(which_value,datab,elnino,lanina,filter_avraged):

    month_grouped_list_with_zeros=list()
    month_grouped_list_with_zeros_elnino=list()
    month_grouped_list_with_zeros_lanina=list()
    for month in range(1,13):
        rain_cop=datab.copy()
        newmatdf_rain_elnino,newmatdf_rain_lanina,newmatdf_rain_all=grouping_data(rain_cop,which_value,elnino,lanina,filter_avraged,month=month,zeross=True)
        month_grouped_list_with_zeros.append(newmatdf_rain_all)
        month_grouped_list_with_zeros_elnino.append(newmatdf_rain_elnino)
        month_grouped_list_with_zeros_lanina.append(newmatdf_rain_lanina)

    return    month_grouped_list_with_zeros,month_grouped_list_with_zeros_elnino,month_grouped_list_with_zeros_lanina
###########################################################
#to remove outliers from daily meteorological data
def remove_outliers(rain,q1,q3,IQR,inc_zeros,IQR_rat): #inc_zeros_remove zeros to find outliers, but add them in the end to include them in the main db
    rain_main_list_df=list()
    list_total=list()
    list_true=list()
    list_stations=list()
    list_uplimit=list()
    list_max=list()
    list_ave=list()
    if inc_zeros==True:
        rain_=rain
    else:    
        rain_=rain[rain["Value"]!=0] #to find points without zeros to calculate outliers. but in model, we enter the zero points also!
        rain_zeros=rain[rain["Value"]==0]
        rain_zeros.insert(2,"outlier",True,True)
    stations=rain_[["CooY","CooX"]].drop_duplicates()
    for index,row in stations.iterrows():
        temp_rain_=rain_[rain_["CooY"]==row["CooY"]]
        temp_rain=temp_rain_[temp_rain_["CooX"]==row["CooX"]]
        if IQR==True:
            uplimit=temp_rain["Value"].quantile(0.75)+IQR_rat*abs(temp_rain["Value"].quantile(0.25)-temp_rain["Value"].quantile(.75))
            rain_main_bool=temp_rain["Value"].between(0,uplimit)
        else:
            uplimit=temp_rain["Value"].quantile(q3)
            rain_main_bool=temp_rain["Value"].between(temp_rain["Value"].quantile(q1), uplimit)

        list_true.append(rain_main_bool.value_counts()[1]) #true
        list_total.append(rain_main_bool.size) #total
        list_stations.append(temp_rain.iloc[0]['ID_MeteoPoint'])
        list_uplimit.append(uplimit)
        list_max.append(temp_rain["Value"].max())
        list_ave.append(temp_rain["Value"].mean())
        temp_rain.insert(2,"outlier",rain_main_bool,True)
        rain_main_list_df.append(temp_rain)
    rain_main=pd.concat(rain_main_list_df)
    if inc_zeros==False:
        rain_main=pd.concat([rain_main,rain_zeros])
    rain_df_station_outliers = pd.DataFrame(data={'ID_MeteoPoint':list_stations,'True': list_true, 'Total': list_total, 'Uplimit':list_uplimit,'Max':list_max, 'Mean':list_ave})

    return rain_main,rain_df_station_outliers
###########################################################
#importing_preprocess
def data_preparation(rain,temp,hum,iso_18,iso_2h,iso_3h,direc,meteo_input_type,q1,q3,IQR_rain,IQR_temp,IQR_hum,inc_zeros,write_outliers_input,write_integrated_data,IQR_rat_rain,IQR_rat_temp,IQR_rat_hum,year_type,elnino,lanina):
    #main is q1 &q3. less than 0.25 more than .75 are outliers.
    if meteo_input_type=="daily_remove_outliers":
        #to remove the outliers
        rain,rain_df_station_outliers=remove_outliers(rain,q1,q3,IQR_rain,inc_zeros,IQR_rat_rain)
        rain=rain[rain['outlier']==True]

        temp,temp_df_station_outliers=remove_outliers(temp,q1,q3,IQR_temp,inc_zeros,IQR_rat_temp)
        temp=temp[temp['outlier']==True]

        hum,hum_df_station_outliers=remove_outliers(hum,q1,q3,IQR_hum,inc_zeros,IQR_rat_hum)
        hum=hum[hum['outlier']==True]

    ###########################################################
    rain['Date'] = pd.to_datetime(rain['Date'])#,format=date_format)
    rain = rain.groupby(['ID_MeteoPoint','CooX','CooY','Month', pd.Grouper(key='Date', freq='m')]).agg({"Longitude":'mean',"Latitude":'mean',"CooZ":'mean', 'Value':'sum'})
    rain=rain.reset_index().sort_values(['Date','ID_MeteoPoint'])

    temp['Date'] = pd.to_datetime(temp['Date'])#,format=date_format)
    temp = temp.groupby(['ID_MeteoPoint','CooX','CooY','Month', pd.Grouper(key='Date', freq='m')]).agg({"Longitude":'mean',"Latitude":'mean',"CooZ":'mean', 'Value':'mean'})
    temp=temp.reset_index().sort_values(['Date','ID_MeteoPoint'])

    hum['Date'] = pd.to_datetime(hum['Date'])#,format=date_format)
    hum = hum.groupby(['ID_MeteoPoint','CooX','CooY','Month', pd.Grouper(key='Date', freq='m')]).agg({"Longitude":'mean',"Latitude":'mean',"CooZ":'mean', 'Value':'mean'})
    hum=hum.reset_index().sort_values(['Date','ID_MeteoPoint'])

    ############################################################
    #write inputs to file:
    if write_outliers_input==True:
        rain.to_csv(os.path.join(direc,"rain_main.csv"))
        temp.to_csv(os.path.join(direc,"temp_main.csv"))
        hum.to_csv(os.path.join(direc,"hum_main.csv"))
        if meteo_input_type=="daily_remove_outliers":
            rain_df_station_outliers.to_excel(os.path.join(direc,"rain_df_station_outliers.xlsx"))
            temp_df_station_outliers.to_excel(os.path.join(direc,"temp_df_station_outliers.xlsx"))
            hum_df_station_outliers.to_excel(os.path.join(direc,"hum_df_station_outliers.xlsx"))
    ###########################################################
    #Group the rain data to average of each station
    #rain
    which_value="Mean_Value_rain"
    datab=rain
    month_grouped_list_with_zeros_rain_all,month_grouped_list_with_zeros_rain_elnino,month_grouped_list_with_zeros_rain_lanina=monthly_uniting(which_value,datab,elnino,lanina,filter_avraged=False)
    #Group the temperature data to average of each station
    which_value="Mean_Value_temp"
    datab=temp
    month_grouped_list_with_zeros_temp_all,month_grouped_list_with_zeros_temp_elnino,month_grouped_list_with_zeros_temp_lanina=monthly_uniting(which_value,datab,elnino,lanina,filter_avraged=False)
    #Group the humidity data to average of each station
    which_value="Mean_Value_hum"
    datab=hum
    month_grouped_list_with_zeros_hum_all,month_grouped_list_with_zeros_hum_elnino,month_grouped_list_with_zeros_hum_lanina=monthly_uniting(which_value,datab,elnino,lanina,filter_avraged=False)
    if year_type=="all":
        month_grouped_list_with_zeros_rain=month_grouped_list_with_zeros_rain_all
        month_grouped_list_with_zeros_temp=month_grouped_list_with_zeros_temp_all
        month_grouped_list_with_zeros_hum=month_grouped_list_with_zeros_hum_all
    elif year_type=="elnino":
        month_grouped_list_with_zeros_rain=month_grouped_list_with_zeros_rain_elnino
        month_grouped_list_with_zeros_temp=month_grouped_list_with_zeros_temp_elnino
        month_grouped_list_with_zeros_hum=month_grouped_list_with_zeros_hum_elnino
    elif year_type=="lanina":        
        month_grouped_list_with_zeros_rain=month_grouped_list_with_zeros_rain_lanina
        month_grouped_list_with_zeros_temp=month_grouped_list_with_zeros_temp_lanina
        month_grouped_list_with_zeros_hum=month_grouped_list_with_zeros_hum_lanina
    ############################################################
    #write integrated (averaged) inputs to a file
    if write_integrated_data==True:
        for i in range(0,len(month_grouped_list_with_zeros_rain)):
            month_grouped_list_with_zeros_rain[i]["month"]=i+1
            month_grouped_list_with_zeros_temp[i]["month"]=i+1
            month_grouped_list_with_zeros_hum[i]["month"]=i+1
        rain_int=pd.concat(month_grouped_list_with_zeros_rain)
        temp_int=pd.concat(month_grouped_list_with_zeros_temp)
        hum_int=pd.concat(month_grouped_list_with_zeros_hum)
        rain_int.to_csv(os.path.join(direc,"rain_int.csv"))
        temp_int.to_csv(os.path.join(direc,"temp_int.csv"))
        hum_int.to_csv(os.path.join(direc,"hum_int.csv"))

    #############################################################
    iso_18['CooX_in']=iso_18["CooX"]
    iso_2h['CooX_in']=iso_2h["CooX"]
    iso_3h['CooX_in']=iso_3h["CooX"]
    #############################################################
    which_value="iso_18"
    datab=iso_18
    #newmatdf_iso_18_elnino,newmatdf_iso_18_lanina,newmatdf_iso_18_norm=grouping_data(iso_18,which_value,elnino,lanina)
    month_grouped_list_with_zeros_iso_18_allyear,month_grouped_list_with_zeros_iso_18_elnino,month_grouped_list_with_zeros_iso_18_lanina=monthly_uniting(which_value,datab,elnino,lanina,filter_avraged=False)
    
    which_value="iso_2h"
    datab=iso_2h
    #newmatdf_iso_2h_elnino,newmatdf_iso_2h_lanina,newmatdf_iso_2h_norm=grouping_data(iso_2h,which_value,elnino,lanina)
    month_grouped_list_with_zeros_iso_2h_allyear,month_grouped_list_with_zeros_iso_2h_elnino,month_grouped_list_with_zeros_iso_2h_lanina=monthly_uniting(which_value,datab,elnino,lanina,filter_avraged=False)
    
    which_value="iso_3h"
    datab=iso_3h
    #newmatdf_iso_3h_elnino,newmatdf_iso_3h_lanina,newmatdf_iso_3h_norm=grouping_data(iso_3h,which_value,elnino,lanina)
    month_grouped_list_with_zeros_iso_3h_allyear,month_grouped_list_with_zeros_iso_3h_elnino,month_grouped_list_with_zeros_iso_3h_lanina=monthly_uniting(which_value,datab,elnino,lanina,filter_avraged=False)
    #return month_grouped_list_with_zeros_rain,month_grouped_list_without_zeros_rain,month_grouped_list_with_zeros_temp,month_grouped_list_without_zeros_temp,rain,temper,elnino,lanina,newmatdf_rain_elnino,newmatdf_rain_lanina,newmatdf_temp_elnino,newmatdf_temp_lanina,newmatdf_temp_norm,iso_18,iso_2h,iso_3h,newmatdf_iso_18_elnino,newmatdf_iso_18_lanina,newmatdf_iso_18_norm,newmatdf_iso_2h_elnino,newmatdf_iso_2h_lanina,newmatdf_iso_2h_norm,newmatdf_iso_3h_elnino,newmatdf_iso_3h_lanina,newmatdf_iso_3h_norm
    
    if year_type=="all":
        month_grouped_list_with_zeros_iso_18=month_grouped_list_with_zeros_iso_18_allyear
        month_grouped_list_with_zeros_iso_2h=month_grouped_list_with_zeros_iso_2h_allyear
        month_grouped_list_with_zeros_iso_3h=month_grouped_list_with_zeros_iso_3h_allyear
    elif year_type=="elnino":
        month_grouped_list_with_zeros_iso_18=month_grouped_list_with_zeros_iso_18_elnino
        month_grouped_list_with_zeros_iso_2h=month_grouped_list_with_zeros_iso_2h_elnino
        month_grouped_list_with_zeros_iso_3h=month_grouped_list_with_zeros_iso_3h_elnino
    elif year_type=="lanina":
        month_grouped_list_with_zeros_iso_18=month_grouped_list_with_zeros_iso_18_lanina
        month_grouped_list_with_zeros_iso_2h=month_grouped_list_with_zeros_iso_2h_lanina
        month_grouped_list_with_zeros_iso_3h=month_grouped_list_with_zeros_iso_3h_lanina
    #############################################################
    #count the available data in each month:
    namess=["rain_count","temp_count","hum_count","iso18_count","iso2h_count"]  
    for (mon_list,group_name) in zip([month_grouped_list_with_zeros_rain,month_grouped_list_with_zeros_temp,month_grouped_list_with_zeros_hum,month_grouped_list_with_zeros_iso_18,month_grouped_list_with_zeros_iso_2h],namess):
        num_rows_list=list()
        month_list=list()
        for each_month in range(0,len(mon_list)):
            num_rows_list.append(mon_list[each_month].shape[0])
            month_list.append(each_month+1)
        foo=pd.DataFrame(data={'month':month_list,'data_count': num_rows_list})
        foo.to_excel(os.path.join(direc,group_name+'.xls'))
    #############################################################    
    return month_grouped_list_with_zeros_iso_18,month_grouped_list_with_zeros_iso_2h,month_grouped_list_with_zeros_iso_3h,month_grouped_list_with_zeros_hum,month_grouped_list_with_zeros_rain,month_grouped_list_with_zeros_temp,rain,temp,hum
###################################################################