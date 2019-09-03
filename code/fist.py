import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
from sklearn.inspection import partial_dependence
from sklearn.inspection import plot_partial_dependence
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.feature_selection import f_classif,SelectPercentile
from sklearn.metrics import mean_squared_error,mean_absolute_error

###########################################################
#importing data
data_file = r'C:\Users\Ash kan\Desktop\sonia\METEO_CHILE.xlsx'
rain = pd.read_excel(data_file,sheet_name="Rain_Winter",header=0,index_col=False,keep_default_na=True)
temper=pd.read_excel(data_file,sheet_name="Temp_Winter",header=0,index_col=False,keep_default_na=True)
tempermeteoindex=temper.set_index('ID_MeteoPoint')
rainmeteoindex=rain.set_index('ID_MeteoPoint')
###########################################################
#making a dataframe of the stations:
stations_temp=temper[['ID_MeteoPoint']]
stations_temp.drop_duplicates(keep = 'last', inplace = True)
stations_rain=rain[['ID_MeteoPoint']]
stations_rain.drop_duplicates(keep = 'last', inplace = True)
###########################################################
#Group the temperature data to average of each station 
newmat1=list()
for index, row in stations_temp.iterrows():
    tempp=tempermeteoindex.loc[row['ID_MeteoPoint']]
    Mean_Value=tempp["Value"].mean()
    if len(tempp.shape)!= 1:
        newmat1.append({"ID_MeteoPoint":row['ID_MeteoPoint'], "CooX":tempp.iat[1,2], "CooY":tempp.iat[0,3], "Z":tempp.iat[0,4], "Mean_Value_temp":Mean_Value})
newmatdf_temp = pd.DataFrame(newmat1)
#newmatdf_temp.to_excel(r'C:\Users\Ash kan\Desktop\sonia\output_temp.xlsx')  
#############################################################
#Group the temperature data to average of each station
newmat2=list()
for index, row in stations_rain.iterrows():
    tempp=rainmeteoindex.loc[row['ID_MeteoPoint']]
    #tempgroup=tempp.groupby(pd.Grouper(key='DateMeas',freq='M')).max()
    Mean_Value=tempp["Value"].mean()
    if len(tempp.shape)!= 1:
        newmat2.append({"ID_MeteoPoint":row['ID_MeteoPoint'], "CooX":tempp.iat[1,2], "CooY":tempp.iat[0,3], "Z":tempp.iat[0,4], "Mean_Value_rain":Mean_Value})
newmatdf_rain = pd.DataFrame(newmat2)
#newmatdf_rain.to_excel(r'C:\Users\Ash kan\Desktop\sonia\output_rain.xlsx')
#############################################################
#merge temp and rain dataframes:
mergeddf=newmatdf_rain.merge(newmatdf_temp,how="inner")      
mergeddf.to_excel(r'C:\Users\Ash kan\Desktop\sonia\output_merged_temper_preci.xlsx')
#############################################################
# pre-processing data for modeling
X=mergeddf[["CooX","CooY","Z"]].copy().astype(float)
Y=mergeddf[["Mean_Value_rain","Mean_Value_temp"]].copy()
X_train, X_test, y_train, y_test = train_test_split(X, Y)
'''scaler1 = StandardScaler()
scaler1.fit(X_train)
scaler2 = StandardScaler()
scaler2.fit(y_train)
X_train = scaler1.transform(X_train)
y_train= scaler2.transform(y_train)
X_test = scaler1.transform(X_test)
y_test= scaler2.transform(y_test)'''

#GradientBoostingRegressor
#est = GradientBoostingRegressor(n_estimators=100, max_depth=4,learning_rate=0.1,random_state=1)
#est.fit(X_train, y_train[:,0])
#############################################################
#RandomForestRegressor
estrandomfor=RandomForestRegressor(n_estimators=200,criterion="mse",n_jobs=-1,min_weight_fraction_leaf=0.05)
estrandomfor.fit(X_train, y_train)
from sklearn.metrics import r2_score,explained_variance_score
predrf=estrandomfor.predict(X_test)
scrf=r2_score(y_test,predrf,multioutput='raw_values')
print (scrf)
print ("random forest regressor r2 score:",estrandomfor.score(X_test,y_test))
print ("explained_variance_score:",explained_variance_score(y_test,predrf))


print ("mean square error:",mean_squared_error(y_test[["Mean_Value_rain"]], predrf[:,0]))
print ("mean abs error:",mean_absolute_error(y_test[["Mean_Value_rain"]], predrf[:,0]))
#############################################################
#plotting random forest results for test dataset against real values
plt.scatter(predrf[:,1],y_test["Mean_Value_temp"])
#plt.scatter(predrf[:,0],y_test["Mean_Value_rain"])
plt.title(" RF test data results- Mean rain and Temp values")
plt.xlabel("Prediction")
plt.ylabel("Real Value")
a = np.linspace(0,18,100)
b=a
plt.plot(a,b)
plt.show()

print ("importance of each feature in RF regression:")
print ("[X,  Y,  Z]:",estrandomfor.feature_importances_)
#############################################################
'''#while in random forest
import warnings
warnings.simplefilter(action='ignore', category=Warning)
estrandomfor=RandomForestRegressor(n_estimators=350,criterion="mse",n_jobs=-1,min_weight_fraction_leaf=0.01)
iterr=0
sc=0
while iterr<=500 and sc<=0.90:
        iterr=iterr+1
        estrandomfor.fit(X_train,y_train)
        sc=estrandomfor.score(X_test,y_test)
        print ("iter:",iterr)
        print ("score:", sc)
#############################################
#mlp = MLPRegressor(activation='relu',solver='adam',hidden_layer_sizes=(15,)*2,max_iter=1000,n_iter_no_change=10)
#mlp.fit(X_train,y_train)
#print ("score:", mlp.score(X_test,y_test))
import warnings
warnings.simplefilter(action='ignore', category=Warning)
iterr=0
sc=0
mlp = MLPRegressor(activation='relu',solver='adam',hidden_layer_sizes=(15,)*2,max_iter=1000,n_iter_no_change=10)
while iterr<=10000 and sc<=0.75:
    iterr=iterr+1
    mlp.fit(X_train,y_train)
    sc=mlp.score(X_test,y_test)
    print ("iter:",iterr)
    print ("score:", sc)
    predictions = mlp.predict(X_test)
    plt.scatter(predictions[:,0],y_test[:,0])
    plt.scatter(predictions[:,1],y_test[:,1])
    plt.plot(a,b)
    sc=mlp.score(X_test,y_test)
    print ("score:", sc)
print ("iter:",iterr)'''
#############################################################
#isotopes pre processing:
data_files = r'C:\Users\Ash kan\Desktop\sonia\Isotope_divided.xlsx'
isotopes = pd.read_excel(data_files,sheet_name="Isotope_PRE_and_Snow",header=0,index_col=False,keep_default_na=True)
#making a dataframe of the stations:
stations_iso=isotopes[["CooX","CooY","CooZ"]]
stations_iso.drop_duplicates(subset ="CooX",keep = 'last', inplace = True)
stations_iso.rename(columns={"CooZ":"Z"},inplace=True)
stations_iso=stations_iso.reset_index(drop=True)
#############################################################
#making prediction for the isotope points
predrf=estrandomfor.predict(stations_iso)
predrf_df=pd.DataFrame(data=predrf,columns=['pred_Rain', 'pred_Temp'])
rescoords=pd.concat([stations_iso,predrf_df],axis=1,ignore_index =False)
#############################################################
#writing predicted results to an excel file
stations_IdChemSamp=isotopes[["CooX","IdChemSamp"]]
stations_IdChemSamp.drop_duplicates(subset ="CooX",keep = 'last', inplace = True)
stations_IdChemSamp=stations_IdChemSamp.reset_index(drop=True)
merged_stat_tem_rain=stations_IdChemSamp.merge(rescoords,how="inner")
merged_stat_tem_rain.to_excel(r'C:\Users\Ash kan\Desktop\sonia\output_predicted_winter.xlsx')
#############################################################
#############################################################
##################PCA for interpreting the data##############
#preparing the needed data
pca_raw_df=merged_stat_tem_rain[["CooX","CooY","Z","pred_Rain","pred_Temp"]].copy()
#scaling
scaler1 = MinMaxScaler()
scaler1.fit(pca_raw_df)
pca_raw_df_st = scaler1.transform(pca_raw_df)
#############################################################
#Applying PCA
pca=PCA()
pcaf=pca.fit(pca_raw_df_st).transform(pca_raw_df_st)
print ("Variance ratio:", pd.DataFrame(pca.explained_variance_ratio_ ))
##############
num_of_significant_pcas=2
pcaf_df=pd.DataFrame(pcaf[:,:num_of_significant_pcas])
plt.scatter(pcaf[:,0],pcaf[:,1])
plt.title("PCA1-PCA2 (rain/temp/xyz) ")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
#kmeans groups
kmeans_group_nums=6
kmeans_2d = KMeans(n_clusters=kmeans_group_nums, random_state=0).fit(pcaf_df)
pcagroups=pd.concat([merged_stat_tem_rain["IdChemSamp"],pca_raw_df,pd.DataFrame(kmeans_2d.labels_)],axis=1)
'''#manual groups
pcaf_df=pd.DataFrame(pcaf[:,:2],columns=["pca1","pca2"])
groups=list()
for index,row in pcaf_df.iterrows():
        if row["pca1"]>=0 and row["pca1"]<=0.5 and row["pca2"]<-0.1 :
                groups.append({"Group":1})
        elif row["pca1"]>=0.20  and row["pca2"]>=-0.1 : 
                groups.append({"Group":2}) 
        elif row["pca1"]>=-0.1 and row["pca1"]<=0.1 and row["pca2"]>0.5 :              
                groups.append({"Group":3})
        elif row["pca1"]<0 and row["pca1"]>=-0.60 and row["pca2"]>-0.2  and row["pca2"]<0.4  : 
                groups.append({"Group":4}) 
        elif row["pca1"]<-0.70: 
                groups.append({"Group":5})             
        else:
                groups.append({"Group":6})
                print  (row["pca1"],row["pca2"])       
groups = pd.DataFrame(groups)
pcagroups=pd.concat([pca_raw_df,merged_stat_tem_rain["IdChemSamp"],groups],axis=1)'''
pcagroups.to_excel(r'C:\Users\Ash kan\Desktop\sonia\output_pca_kmeans_temp_rain_xyz.xlsx')  
#############################################################
#isotope file preprocess
data_file_iso = r'C:\Users\Ash kan\Desktop\sonia\Isotope_divided.xlsx'
iso_18 = pd.read_excel(data_file_iso,sheet_name="18O_Winter",header=0,index_col=False,keep_default_na=True)
iso_2h=pd.read_excel(data_file_iso,sheet_name="2H_Winter",header=0,index_col=False,keep_default_na=True)
iso_3h=pd.read_excel(data_file_iso,sheet_name="3H_Winter",header=0,index_col=False,keep_default_na=True)
iso_18.rename(columns={"Value":"iso_18"},inplace=True)
iso_2h.rename(columns={"Value":"iso_2h"},inplace=True)
iso_3h.rename(columns={"Value":"iso_3h"},inplace=True)
iso_18['CooX_in']=iso_18["CooX"]
iso_2h['CooX_in']=iso_2h["CooX"]
iso_3h['CooX_in']=iso_3h["CooX"]
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
merged_stat_tem_rain.rename(columns={"CooZ":"Z"},inplace=True)
iso_meteo_merged=iso_mergeddf.merge(merged_stat_tem_rain,how="inner")
iso_meteo_merged.reset_index(drop=True)
iso_meteo_merged.to_excel(r'C:\Users\Ash kan\Desktop\sonia\iso_meteo_merged.xlsx')
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
#######################
# #PCA loading matrix
loading_matrix=pca.components_.T * np.sqrt(pca.explained_variance_)
#######################
#######################
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
iso_meteo_pca_kmeans.to_excel(r'C:\Users\Ash kan\Desktop\sonia\Winter\z_3h_temp_rain.xlsx')
############################################
#trying the clustering with DBSCAN
dbscan=DBSCAN(eps=0.5, min_samples=3)
dbscan.fit()
############################################
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
print ("mutual_info_regression_value_temp",mutual_info_regression_value_temp)
##############################################
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
diabetes = datasets.load_diabetes()
X = diabetes.data[:150]
y = diabetes.target[:150]
from sklearn.metrics import sum_squared_error

def p_vals_per_coef(pred, true, coefs, X):
        sse = sum_squared_error(pred,true)/ float(X.shape[0] - X.shape[1])
        standard_error = np.array([np.sqrt(np.diagonal(sse * np.linalg.inv(np.dot(X.T, X))))])
        t_stats = coefs / standard_error