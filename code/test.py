rain2=rain[rain["Value"]!=0]

rainmonth6=pd.DataFrame(data=rain2.loc[rain2["DateMeas"].dt.month.isin([6,7,8])] ,columns=['DateMeas',"Value","ID_MeteoPoint"])
rainmonth6=pd.DataFrame(data=rainmonth6.loc[rain2["ID_MeteoPoint"].isin(["Alcerreca"])],columns=["ID_MeteoPoint",'DateMeas',"Value"])
#rainmonth6=rainmonth6[rainmonth6["Value"]<np.percentile(rainmonth6["Value"],90)]
plt.scatter(rainmonth6["DateMeas"],rainmonth6["Value"])
################
rainmonth7=pd.DataFrame(data=rain.loc[rain["DateMeas"].dt.month==7],columns=['DateMeas',"Value"])
plt.scatter(rainmonth7["DateMeas"],rainmonth7["Value"])
###############
rainmonth8=pd.DataFrame(data=rain.loc[rain["DateMeas"].dt.month==8],columns=['DateMeas',"Value"])
plt.scatter(rainmonth8["DateMeas"],rainmonth8["Value"])
###############
rainmonth9=pd.DataFrame(data=rain.loc[rain["DateMeas"].dt.month==9],columns=['DateMeas',"Value"])
plt.scatter(rainmonth9["DateMeas"],rainmonth9["Value"])


plt.scatter(rainmonth6["DateMeas"],rainmonth6["Value"])
plt.title(" main ")
plt.xlabel("Prediction")
plt.ylabel("Real Value")
left, right = plt.xlim()
left1, right1 = plt.ylim()
a = np.linspace(0,max(right,right1),100)
b=a
plt.plot(a,b)
plt.show()

from sklearn.linear_model import Lasso
alpha = 0.005
lasso = Lasso(alpha=alpha)
y_pred_lasso = lasso.fit(X_train_rain_normal, y_train_rain_normal).predict(X_test_rain_normal)
r2_score_lasso = r2_score(y_test_rain_normal, y_pred_lasso)
r2_score_lasso
from sklearn.linear_model import ElasticNet
alpha = 0.1
enet = ElasticNet(alpha=alpha, l1_ratio=0.7)
y_pred_enet  = enet.fit(X_train_rain_normal, y_train_rain_normal).predict(X_test_rain_normal)
r2_score_enet = r2_score(y_test_rain_normal, y_pred_enet )
r2_score_enet


def rfmethod(gridsearch_dictionary,newmatdf_temp,temp_rain_hum):
    X_temp=newmatdf_temp[["CooX","CooY","CooZ"]].copy().astype(float)
    Y_temp=newmatdf_tempS[[temp_rain_hum]].copy()
    X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(X_temp, Y_temp)
    
    estmlp_temp=GridSearchCV( MLPRegressor(), tunedpars, cv=10,n_jobs=-1)
    estmlp_temp.fit(X_train_temp, y_train_temp)
    print ("random forest regressor r2 score:")
    print (temp_rain_hum,estrandomfor_temp.score(X_test_temp,y_test_temp))
    predrf_temp=estrandomfor_temp.predict(X_test_temp)
    print ("mean square error:")
    print (temp_rain_hum,mean_squared_error(y_test_temp, predrf_temp))
    print ("mean abs error:",temp_rain_hum,mean_absolute_error(y_test_temp, predrf_temp))
    print ("max_error",temp_rain_hum,max_error(y_test_temp, predrf_temp))
    #estrandomfor_temp.fit(X_temp, Y_temp)
    #cross validation
    scores = cross_val_score(estrandomfor_temp, X_temp, Y_temp, cv=5,n_jobs =-1)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    #############################################################
    #plotting random forest results for test dataset against real values
    #plt.scatter(predrf[:,0],y_test["Mean_Value_rain"])
    plt.scatter(predrf_temp,y_test_temp[temp_rain_hum])
    plt.title(" RF test data results")
    plt.xlabel("Prediction")
    plt.ylabel("Real Value")
    left, right = plt.xlim()
    left1, right1 = plt.ylim()
    a = np.linspace(0,max(right,right1),100)
    b=a
    plt.plot(a,b)
    plt.show()
    #print ("importance of each feature in RF regression:",temp_rain_hum)
    #print ("[X,  Y,  Z]:",estrandomfor_temp.feature_importances_)
    return estrandomfor_temp,X_temp ,Y_temp,X_train_temp, X_test_temp, y_train_temp, y_test_temp  
