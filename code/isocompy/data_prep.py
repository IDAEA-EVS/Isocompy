import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import copy
                                      
##########################################################################################
##########################################################################################

def geo_mean(iterable):
    scaler = MinMaxScaler()
    a=scaler.fit_transform(np.array(iterable).reshape(-1, 1))
    a=a.prod()**(1.0/len(a))
    return float(scaler.inverse_transform(a.reshape(-1, 1)))
#grouping data
def grouping_data(inp_var,fields,year_type,elnino,lanina,filter_avraged,mean_mode,month,zeross=True):
    if zeross==False:
        inp_var=inp_var[inp_var["Value"]!=0]
    if month !=None:
        inp_var=inp_var[inp_var["Date"].dt.month==month] 

    #omiting the top 5 percent of the data
    if filter_avraged==True:
        inp_var_95=inp_var[inp_var["Value"]<inp_var["Value"].quantile(0.95)]
        if inp_var_95.shape[0]>=5:
            inp_var=inp_var_95
    #########
    stations_inp_var=inp_var[['ID']]
    stations_inp_var.drop_duplicates(keep = 'last', inplace = True)
    inp_varmeteoindex=inp_var.set_index('ID')

    newmat_norm=list()
    for index, row in stations_inp_var.iterrows():
        tempp=inp_varmeteoindex.loc[row['ID']]
        if len(tempp.shape)!= 1:
            #print (tempp)
            sum_elnino=list()
            sum_lanina=list()
            sum_norm=list()
            for index2, row2 in tempp.iterrows():
                if pd.to_datetime(row2["Date"]).year in elnino:
                    sum_elnino.append(row2["Value"])

                elif pd.to_datetime(row2["Date"]).year in lanina:
                    sum_lanina.append(row2["Value"])
                else:
                    sum_norm.append(row2["Value"])
            if year_type=="all": sum_fin=sum_norm
            elif year_type=="elnino" and len(elnino)!=0: sum_fin=sum_elnino
            elif year_type=="lanina" and len(lanina)!=0: sum_fin=sum_lanina
            else: print ("Revise year_type and list of elnino lanina years")

            if len(sum_fin) !=0:
                if mean_mode=="geometric": 
                    Mean_Value_norm=geo_mean(sum_fin)
                else:        
                    Mean_Value_norm=sum(sum_fin)/len(sum_fin)
                t_dic={"ID":row['ID'], "Value":Mean_Value_norm}  
                for f in fields:
                    t_dic[f]=tempp[f].iat[0]
                newmat_norm.append(t_dic)
        else:
            #print ("tempp in len 1:")
            #print (tempp)
            if (pd.to_datetime(tempp["Date"]).year in elnino and year_type=="elnino") or (pd.to_datetime(tempp["Date"]).year in lanina and year_type=="lanina") or (year_type=="all"):
                t_dic={"ID":row['ID'],"Value":tempp["Value"]}
                for f in fields:
                    t_dic[f]=tempp[f]
                newmat_norm.append(t_dic)
                
    newmatdf_inp_var_all = pd.DataFrame(newmat_norm)
    return newmatdf_inp_var_all
###########################################
#function for monthly procedure
def monthly_uniting(datab,fields,year_type,elnino,lanina,filter_avraged,mean_mode):

    month_grouped_list=list()

    for month in range(1,13):
        inp_var_cop=datab.copy()
        newmatdf_inp_var_all=grouping_data(inp_var_cop,fields,year_type,elnino,lanina,filter_avraged,mean_mode=mean_mode,month=month,zeross=True)
        month_grouped_list.append(newmatdf_inp_var_all)

    return    month_grouped_list
###########################################################
#to remove outliers from daily meteorological data
def remove_outliers_func(inp_var,fields,q1,q3,IQR,inc_zeros,IQR_rat): #inc_zeros_remove zeros to find outliers, but add them in the end to include them in the main db
    inp_var_main_list_df=list()
    list_total=list()
    list_true=list()
    list_stations=list()
    list_uplimit=list()
    list_max=list()
    list_ave=list()
    if inc_zeros==True:
        inp_var_=inp_var
    else:
        inp_var_=inp_var[inp_var["Value"]!=0] #to find points without zeros to calculate outliers. but the zero values will be added to the inputs at the end of this stage!
        inp_var_zeros=inp_var[inp_var["Value"]==0]
        inp_var_zeros.insert(2,"outlier",True,True)

    stations=inp_var_[fields].drop_duplicates()

    for index,row in stations.iterrows():

        inp_var_t=copy.deepcopy(inp_var_) #to reset the inp_var_ every time!

        for f in fields:
            inp_var_t=inp_var_t[inp_var_t[f]==row[f]]

        if IQR==True:
            uplimit=inp_var_t["Value"].quantile(0.75)+IQR_rat*abs(inp_var_t["Value"].quantile(0.25)-inp_var_t["Value"].quantile(.75))
            inp_var_tmain_bool=inp_var_t["Value"].between(q1,uplimit)
        else:
            uplimit=inp_var_t["Value"].quantile(q3)
            inp_var_tmain_bool=inp_var_t["Value"].between(inp_var_t["Value"].quantile(q1), uplimit)
        try:
            list_true.append(inp_var_tmain_bool.value_counts()[True]) #true
        except:
            list_true.append(0)    
            
        list_total.append(inp_var_tmain_bool.size) #total
        list_stations.append(inp_var_t.iloc[0]['ID'])
        list_uplimit.append(uplimit)
        list_max.append(inp_var_t["Value"].max())
        list_ave.append(inp_var_t["Value"].mean())
        inp_var_t.insert(2,"outlier",inp_var_tmain_bool,True)
        inp_var_main_list_df.append(inp_var_t)

    inp_var_main=pd.concat(inp_var_main_list_df)
    if inc_zeros==False:
        inp_var_main=pd.concat([inp_var_main,inp_var_zeros])
    inp_var_df_station_outliers = pd.DataFrame(data={'ID':list_stations,'True': list_true, 'Total': list_total, 'Uplimit':list_uplimit,'Max':list_max, 'Mean':list_ave})

    return inp_var_main,inp_var_df_station_outliers
###########################################################
#importing_preprocess
def data_preparation_func(inp_var,var_name,fields,direc,remove_outliers,q1,q3,IQR_inp_var,inc_zeros_inp_var,write_outliers_input,write_integrated_data,IQR_rat_inp_var,year_type,elnino,lanina,mean_mode_inp_var,per_year_integration_method):
    #main is q1 &q3. less than 0.25 more than .75 are outliers.
    if remove_outliers==True:
        #to remove the outliers
        inp_var,inp_var_df_station_outliers=remove_outliers_func(inp_var,fields,q1,q3,IQR_inp_var,inc_zeros_inp_var,IQR_rat_inp_var)
        inp_var=inp_var[inp_var['outlier']==True]
        inp_var.to_csv(os.path.join(direc,var_name+"_outliers_removed_1.csv"))

    ###########################################################
    inp_var['Date'] = pd.to_datetime(inp_var['Date'])#,format=date_format)
    gr_list=['ID']+fields+ [pd.Grouper(key='Date', freq='m')]
    inp_var = inp_var.groupby(gr_list).agg({'Value':per_year_integration_method})
    inp_var=inp_var.reset_index().sort_values(['Date','ID'])
    #write inputs to file:
    if write_outliers_input==True:
        inp_var.to_csv(os.path.join(direc,var_name+"_monthly_2.csv"))
        if remove_outliers==True:
            inp_var_df_station_outliers.to_csv(os.path.join(direc,var_name+"_df_outliers.csv"))
    ###########################################################
    #Group the inp_var data to average of each station
    #inp_var
    datab=inp_var
    month_grouped_list=monthly_uniting(datab,fields,year_type,elnino,lanina,mean_mode=mean_mode_inp_var, filter_avraged=False)

    #############################################################
    #write integrated (averaged) inputs to a file
    if write_integrated_data==True:
        for i in range(0,len(month_grouped_list)):
            month_grouped_list[i]["month"]=i+1
        inp_var_int=pd.concat(month_grouped_list)
        inp_var_int.to_csv(os.path.join(direc,var_name+"_model_input_3.csv"))
    #############################################################
    #count the available data in each month:
    if write_integrated_data==True:
        num_rows_list=list()
        month_list=list()
        for each_month in range(0,len(month_grouped_list)):
            num_rows_list.append(month_grouped_list[each_month].shape[0])
            month_list.append(each_month+1)
        foo=pd.DataFrame(data={'month':month_list,'data_count': num_rows_list})
        foo.to_excel(os.path.join(direc,var_name+'.xls'))
    #############################################################    
    return month_grouped_list,inp_var
###################################################################