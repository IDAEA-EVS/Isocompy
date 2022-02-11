import pandas as pd
import numpy as np
import copy
from sklearn.feature_selection import f_regression, mutual_info_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
def vif_calc(X_temp):
    X = add_constant(X_temp)
    vif_df=pd.DataFrame([variance_inflation_factor(X.values, i) 
                for i in range(X.shape[1])], 
                index=X.columns).rename(columns={0:'VIF'}).drop('const')
    print ("VIF\n",vif_df)     
    X=X.drop('const', axis=1)
    return X,vif_df 

#args_dic={"feature_selection":"auto","vif_threshold":5, "vif_selection_pairs":[],
#"correlation_threshold":0.87,"vif_corr":True,"p_val":0.05}


def feature_selection(X_temp,Y_temp,args_dic):
    ###############
    #print some stats:

    #correlation coefficient    
    correl_mat=X_temp.corr()
    correl_mat = pd.DataFrame(correl_mat,columns=X_temp.columns,index=X_temp.columns)
    print("Correlation matrix:\n",correl_mat)
    ##################################################
    X,vif_df= vif_calc(X_temp)   
    vif_initial=copy.deepcopy(vif_df)
    print("VIF_initial\n",vif_df)
    #some tests:
    try:
        mutual_info_regression_value = mutual_info_regression(X_temp, Y_temp)
        mutual_info_regression_value /= np.max(mutual_info_regression_value)
        print ("mutual_info__value:\n",mutual_info_regression_value)
    except:
        mutual_info_regression_value=None

    f_reg=f_regression(X_temp, Y_temp)
    f_regression_value=pd.DataFrame(f_reg,columns=X_temp.columns)
    print ("f regression - p_values:\n", f_regression_value)
    ###############
    #manual
    if args_dic["feature_selection"]=="manual":
        used_features=input("Enter the list of the selected features:")
        vif_chosen_features="Not Calculated" 
        vif_df="Not Calculated" 

    else:
        vif_threshold=args_dic["vif_threshold"]
        cor_th=args_dic["correlation_threshold"]
        #VIF checking
        if vif_threshold !=None:
            #correlation:
            correl=correl_mat[(correl_mat<-cor_th) | (correl_mat>cor_th)]
            #print ("correl",correl)
            more_than_thres=vif_df[vif_df["VIF"]>vif_threshold].index

            #omit introduced pairs:

            if len(args_dic["vif_selection_pairs"])>0:
                for feat_pair in args_dic["vif_selection_pairs"]:
                    if (feat_pair[0] in more_than_thres) and ( feat_pair[1] in more_than_thres) and ( (args_dic["vif_corr"]==True and np.isnan(correl[feat_pair[0]][feat_pair[1]])==False) or args_dic["vif_corr"]==False ) :
                        X=X.drop(feat_pair[0], axis=1)
                        #calculate vif again
                        X,vif_df= vif_calc(X)
                        more_than_thres=vif_df[vif_df["VIF"]>vif_threshold].index
            if len(more_than_thres)==1:
                X=X.drop(more_than_thres,axis=1)
                #calculate vif again
                X,vif_df= vif_calc(X)
                more_than_thres=vif_df[vif_df["VIF"]>vif_threshold].index
            if len(more_than_thres)>1:
                for i in range (0,len(more_than_thres)):
                    if len(more_than_thres)>0:
                        to_drop=vif_df[vif_df.VIF==vif_df.VIF.max()].index[0]
                        X=X.drop(to_drop, axis=1)
                        X,vif_df= vif_calc(X)
                        more_than_thres=vif_df[vif_df["VIF"]>vif_threshold].index
            vif_chosen_features=list(X.columns)
        else:
            vif_df="Not Calculated"  
            vif_chosen_features="Not Calculated"  
        #################################################
        #f test  just using the significant variables!
        p_val=args_dic["p_val"]

        f_reg=f_regression_value[vif_chosen_features].drop([0])
        f_regt=f_reg.T
        f_less_col=list( f_regt[f_regt[1]<=p_val].index )

        #f_less_ind=list(np.where(f_reg[1]<=p_val)[0])
        if len(f_less_col)<2:
            f_regt_sort=f_regt.sort_values(by=1)
            f_less_col=[ f_regt_sort.index[0],f_regt_sort.index[1] ]
            """
                f_less_ind1=list(
                    (np.where
                    (f_reg[1]==f_regression_value_sor[0])
                                    ) [0]
                )
                f_less_ind2=list((np.where
                    (f_reg[1]==f_regression_value_sor[1])
                                    )[0]   
                ) 
                f_less_ind=f_less_ind1+f_less_ind2
            """
        used_features=f_less_col 


    return mutual_info_regression_value,f_regression_value,correl_mat,used_features,vif_df,vif_chosen_features,vif_initial