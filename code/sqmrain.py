import pandas as pd
data_file_P = r'C:\Users\Ash kan\Desktop\sonia\Rain_SQM.xlsx'
rain2 = pd.read_excel(data_file_P,sheetname=0,header=0,index_col=False,keep_default_na=True)
rainmeteoindex2=rain2.set_index('CooX')
stations_rain2=rain2[['CooX']]
stations_rain2.drop_duplicates(keep = 'last', inplace = True)
cnt=0
for index, row in stations_rain2.iterrows():
    tempp=rainmeteoindex2.loc[row['CooX']]
    #print (tempp)
    #Mean_Value=tempgroup["Value"].mean()
    if cnt==0:
        cnt=cnt+1
        tempgroup=tempp.groupby(pd.Grouper(key='FechaMedicion',freq='M')).sum()
        tempgroup['CooX']=row['CooX']
        #print (tempp["NombreEstacion"].iat[0])
        tempgroup['NombreEstacion']=tempp["NombreEstacion"].iat[0]
        #print (tempgroup)
    else:
        tempgroup2=tempp.groupby(pd.Grouper(key='FechaMedicion',freq='M')).sum()
        tempgroup2['CooX']=row['CooX']
        tempgroup2['NombreEstacion']=tempp["NombreEstacion"].iat[0]
        #print (tempgroup2)
        tempgroup=pd.concat([tempgroup,tempgroup2])
tempgroup.to_excel(r'C:\Users\Ash kan\Desktop\sonia\output_rain_sqm.xlsx')
np.isnan(tempgroup["HR prom"]).count()