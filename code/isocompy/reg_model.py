from pathlib import Path
from isocompy.model_fitting import rfmethod, iso_prediction,st1_print_to_file,isotope_model_selection_by_meteo_line,st2_output_report
from isocompy import cv_uncertainty as cvun
import os
import pandas as pd
#class for stage 1 and 2 modeling and prediction
class model(object):

    """
        The class to fit the regression models in stage one, predict the stage one results and fit stage two regresison models

        #------------------
        Methods:

            __init__(self)

            st1_fit(self,var_cls_list,direc,st1_model_month_list="all",args_dic={"feature_selection":"auto","vif_threshold":5, "vif_selection_pairs":[], "correlation_threshold":0.87,"vif_corr":True,"p_val":0.05})

            st1_predict(self,cls_list,st2_model_month_list=None,trajectories=False,daily_rain_data_for_trajs=None)

            st2_fit(self,model_var_dict,output_report=True,dependent_model_selection=False,dependent_model_selection_list=None,meteo_coef=8,meteo_intercept=10,selection_method="point_to_point",thresh=None,model_selection_report=True,args_dic={"feature_selection":"auto","vif_threshold":5, "vif_selection_pairs":[],"correlation_threshold":0.87,"vif_corr":True,"p_val":0.05})
            
            choose_estimator_by_meteo_line(self,dependent_model_selection_list,selection_method="point_to_point",model_selection_report=True,thresh=None,meteo_coef=8,meteo_intercept=10)

            stage2_output_report(self,direc=None)


        #------------------
        Attributes:
        (__init__ method of model class)


            #st1

            direc str
                Directory of the class

            st1_model_results_dic dict
                A dictionary consist of st1 model results

            st1_varname_list list
                List of the names of independent variables in st1

            st1_model_month_list list
                List of desired months to model in st1
            

            #st1 prediction

            used_feature_list list
                List of all used features (strings) in st1

            cls_list list
                A list of preprocess class objects that we wish to model in st2

            all_pred Pandas Dataframe
                A dataframe of all st1 predictions

            predictions_monthly_list list
                Dataframes of predictions of stage one, seperated monthly as list elements 
            
            st2_model_month_list list
                List of desired months to model in st1. Indicated months have to exist in st1_model_month_list

            #st2

            dic_second_stage_names dict
                Helps in generating model_var_dict in st2_fit

            st2_model_results_dic dict
                A dictionary consist of st2 model results


           
            
            

            #  model selection in st2 based on a line

            dependent_model_selection boolean
                To select the best model based on meteorological line. only useful if there is a linear refrence line (EX:Isotopes)

            meteo_coef float 
                If dependent_model_selection=True,global_line, coefficient of the line

            meteo_intercept float
                If dependent_model_selection=True,global_line, intercept of the line

            selection_method str
                If dependent_model_selection=True, selection_method: independent,local_line,global_line, point_to_point

            thresh_meteoline_high_scores None type or float
                A threshold to just consider models with scores higher than that value. if none, equal to mean of scores+std of scores/3
            
            model_selection_report boolean
                True or Flase, to determine if there is a need to model selection method report

        #------------------

    """
    
    def __init__(self):
        
        #st1
        self.direc=r"" #Directory of the class
        self.st1_model_results_dic=dict()
        self.st1_varname_list=list() #list of the names of independent variables in st1
        self.st1_model_month_list=[] #list of desired months to model in st1
        self.used_feature_list =[] #list of all used features (strings) in st1
        
        #st1 prediction
        self.cls_list=[] #A list of preprocess class objects that we wish to model in st2.
        self.all_preds=pd.DataFrame() #a dataframe of all st1 predictions
        self.predictions_monthly_list=[] #dataframes of predictions of stage one, seperated monthly as list elements 

        #st2
        self.dic_second_stage_names={} #helps in generating model_var_dict in st2_fit
        self.st2_model_month_list=[] #list of desired months to model in st1. Indicated months have to exist in st1_model_month_list
        self.st2_model_results_dic=dict()   
        

        #model selection method parameters
        self.dependent_model_selection=False #to select the best model based on meteorological line. only useful if there is a linear refrence line (EX:isotopes)
        self.meteo_coef=8 #if dependent_model_selection=True,global_line, coefficient of the line
        self.meteo_intercept=10 #if dependent_model_selection=True,global_line, intercept of the line
        self.selection_method="point_to_point" #if dependent_model_selection=True, selection_method: independent,local_line,global_line, point_to_point
        self.thresh_meteoline_high_scores=None #a threshold to just consider models with scores higher than that value. if none, equal to mean of scores+std of scores/3
        self.model_selection_report=True
        
        #trajs
        self.col_for_f_reg=[] #trajs
        self.trajectories=False #trajs
        self.all_hysplit_df_list_all_atts=[] #trajs
        self.col_for_f_reg=[] #trajs        
        self.all_without_averaging=pd.DataFrame() #trajs

    ##########################################################################################

    def st1_fit(self,var_cls_list,direc,st1_model_month_list="all",args_dic={"feature_selection":"auto","vif_threshold":5, "vif_selection_pairs":[],"correlation_threshold":0.87,"vif_corr":True,"p_val":0.05}):
    
        """
            The method to fit regression models to identified preprocess class objects in stage one 

            #------------------
            Parameters:

                var_cls_list list
                    A list of preprocess class objects to to fit regression models. Regression models will be fitted to each elemnt
                    of the list (a preprocess class object).

                direc str
                    Directory of the class

                st1_model_month_list str or list of integers default="all"
                    List of desired months to model in st1
                
                args_dic dict default={"feature_selection":"auto","vif_threshold":5, "vif_selection_pairs":[],"correlation_threshold":0.87,"vif_corr":True,"p_val":0.05}
                    A dictionary of parameters that identifies the behaviour of feature selection prior to regressions:    
                        
                        args_dic["feature_selection"] ="manual": Statistical information will be shown to the user, and the desired features will be
                        chosen by the user

                        args_dic["feature_selection"] ="auto": Feature selection will be done automatically

                        args_dic["vif_threshold"] =None: VIF (Variation Inflation Factor) will not be considered as a factor in feature selection

                        args_dic["vif_threshold"] = float type: A threshold to identify high VIF values

                        args_dic["vif_corr"] = True: If True, use correlation coefficient values to identify multicolinearity in features with high vif value
                        
                        args_dic["correlation_threshold"] = 0.87 A threshold to identify high correlation coefficient values
                        
                        args_dic["vif_selection_pairs"] = empty list or list of list(s): If empty: feature elimination based on vif will be automatic
                        if  args_dic["vif_selection_pairs"] =[ ["a","b"] ], in case both "a" and "b" have high vif values and high correlations, the b values will be eliminated

            #------------------
            Attributes:

                direc str
                    Directory of the class

                st1_model_results_dic dict
                    A dictionary consist of st1 model results

                st1_varname_list list
                    List of the names of independent variables in st1

                st1_model_month_list list
                    List of desired months to model in st1

            #------------------
        """

        if st1_model_month_list=="all":
            self.st1_model_month_list=[n for n in range(1,13) ]
        else:     self.st1_model_month_list=st1_model_month_list
        self.direc=direc
        Path(self.direc).mkdir(parents=True, exist_ok=True)
        stage_1_2=1
        best_dics_p=dict()
        best_dics_dics_p=dict()
        ############################################################
        
        for prepcls in  var_cls_list:
            self.st1_varname_list.append(prepcls.db_input_args_dics["var_name"])
            fields=prepcls.db_input_args_dics["fields"]

            bests,preds_real_dic,bests_dic=rfmethod(prepcls.month_grouped_inp_var,prepcls.model_pars_name_dic,stage_1_2,self.st1_model_month_list,args_dic,fields)
            results_dic={"bests":bests,"preds_real_dic":preds_real_dic,"bests_dic":bests_dic,"db_input_args_dics":prepcls.db_input_args_dics}
            self.st1_model_results_dic[prepcls.db_input_args_dics["var_name"]]=results_dic
            best_dics_p[prepcls.db_input_args_dics["var_name"]]=bests
            best_dics_dics_p[prepcls.db_input_args_dics["var_name"]]=bests_dic
        ###########################################

        #############################################################     
        st1_print_to_file(self.direc ,best_dics_p,best_dics_dics_p )
        #model prediction stats prediction_uncertainty_stats
        Path(os.path.join(self.direc,"st1_prediction_uncertainty_stats")).mkdir(parents=True, exist_ok=True)
        for k,v in best_dics_dics_p.items():
            cvun.model_meanstd(k,v,os.path.join(self.direc,"st1_prediction_uncertainty_stats"))
    ##########################################################################################

    def st1_predict(self,cls_list,st2_model_month_list=None,trajectories=False,daily_rain_data_for_trajs=None):
        """
            The method to estimate the independent features that modeled in st1 using the new observations that are in 
            a new list (cls_list) which each element is a preprocess class objects 

            #------------------
            Parameters:

                cls_list list
                    A list of preprocess class objects that we wish to model in st2
                
                
                st2_model_month_list list or None type default=None
                    List of desired months to model in st1. Indicated months have to exist in st1_model_month_list. If None,
                    if will be equal to st1_model_month_list

            #------------------
            Attributes:

                used_feature_list list
                    List of all used features (strings) in st1

                cls_list list
                    A list of preprocess class objects that we wish to model in st2

                all_pred Pandas Dataframe
                    A dataframe of all st1 predictions in observed samples

                predictions_monthly_list list
                    Dataframes of predictions of stage one, seperated monthly as list elements 
                
                st2_model_month_list list
                    List of desired months to model in st1. Indicated months have to exist in st1_model_month_list

            #------------------
        """
        #self=iso_meteo_model
        #cls=prep_dataset
        ########################################
        #check the fields in iso are available in fields in cls_list
        used_feature_list=list()
        for k,v in  self.st1_model_results_dic.items():
            for n in v["bests_dic"]:
                used_feature_list.extend(n["used_features"])
        self.used_feature_list = list(set(used_feature_list)) #to remove duplicates  
        self.cls_list=cls_list
        for cls in cls_list:
            self.dic_second_stage_names[cls.db_input_args_dics["var_name"]]=None
            
            for f in cls.db_input_args_dics["fields"]:
                if f not in self.used_feature_list:
                    raise Exception ("used features in the first stage can not be find in fields in second stage! so the prediction can not be done! correct the database!")
        #######################################    

        if trajectories==False: self.col_for_f_reg=[]
        Path(self.direc).mkdir(parents=True, exist_ok=True)
        self.trajectories=trajectories
        #solving months
        if st2_model_month_list==None: self.st2_model_month_list=self.st1_model_month_list
        if st2_model_month_list=="all": self.st2_model_month_list=[m for m in range(1,13)]
        if type(st2_model_month_list) is type(list()):
            t_l=list()
            for i in st2_model_month_list:
                if i in self.st1_model_month_list:
                    t_l.append(i)
                else: raise Exception ("month {} in st2_model_month_list can not be found in the modeled month in stage one".format(i))    
            self.st2_model_month_list=t_l  
        ###################################
        if type(cls_list) != type(list()) or len(cls_list)==0:
            raise Exception ("Revise cls_list!")
        ###################################
        
        self.predictions_monthly_list, self.all_preds,self.all_hysplit_df_list_all_atts,self.col_for_f_reg,self.all_without_averaging=iso_prediction(self.cls_list,
            self.st1_model_results_dic,
            dates_db=daily_rain_data_for_trajs,
            trajectories=self.trajectories,
            iso_model_month_list=self.st2_model_month_list,
            direc=self.direc)

    ##########################################################################################

    def st2_fit(self,model_var_dict=None,output_report=True,dependent_model_selection=False,dependent_model_selection_list=None,meteo_coef=8,meteo_intercept=10,selection_method="point_to_point",thresh_meteoline_high_scores=None,model_selection_report=True,args_dic={"feature_selection":"auto","vif_threshold":5, "vif_selection_pairs":[],
            "correlation_threshold":0.87,"vif_corr":True,"p_val":0.05}):
        
        """
            The method to fit regression models to identified preprocess class objects in stage one

            #------------------
            Parameters:

                model_var_dict None type or dict  default=None
                    A dictionary that determines dependent (key - string) and independent (value) features of the second stage regression models.
                    Independent features (value) have to be a list of feature names (string).
    
                    If None, all features (independent st1 features and dependent st1 features) will be
                    considered as independent features of second stage models.
                    
                    EXAMPLE: model_var_dict = {"is1":["CooZ","hmd"],"is2":["prc","hmd"],}
                
                output_report boolean default=True
                    To generate output reports
                
                #Parameters used in choose_estimator_by_meteo_line 
                
                dependent_model_selection boolean default=False
                    To select the best model based on a (meteorological) line. only useful if there is a linear refrence line (EX:Isotopes)
                
                dependent_model_selection_list default=None
                    Used if dependent_model_selection=True. List of two features that have to be used in dependent_model_selection
                
                meteo_coef default=8
                    Used if dependent_model_selection=True and selection_method="global_line". Coefficient of the line
                
                meteo_intercept default=10
                    Used if dependent_model_selection=True and selection_method="global_line". Intercept of the line
                
                selection_method default="point_to_point"
                    Used if dependent_model_selection=True. Selection_method could be:
                    independent
                    local_line: coef and intercept derived from a linear regression of observed data
                    global_line
                    point_to_point: find the models pair with shortest average distance between observed and predicted data  
                
                thresh_meteoline_high_scores None type or float default=None
                    A threshold to just consider models with scores higher than that value. if none, equal to mean of scores+std of scores/3
                
                model_selection_report boolean default =True
                    True or False, to determine if there is a need to model selection method report
                
                args_dic dict default={"feature_selection":"auto","vif_threshold":5, "vif_selection_pairs":[],"correlation_threshold":0.87,"vif_corr":True,"p_val":0.05}
                    A dictionary of parameters that identifies the behaviour of feature selection prior to regressions:    
                        
                        args_dic["feature_selection"] ="manual": Statistical information will be shown to the user, and the desired features will be
                        chosen by the user

                        args_dic["feature_selection"] ="auto": Feature selection will be done automatically

                        args_dic["vif_threshold"] =None: VIF (Variation Inflation Factor) will not be considered as a factor in feature selection

                        args_dic["vif_threshold"] = float type: A threshold to identify high VIF values

                        args_dic["vif_corr"] = True: If True, use correlation coefficient values to identify multicolinearity in features with high vif value
                        
                        args_dic["correlation_threshold"] = 0.87 A threshold to identify high correlation coefficient values
                        
                        args_dic["vif_selection_pairs"] = empty list or list of list(s): If empty: feature elimination based on vif will be automatic
                        if  args_dic["vif_selection_pairs"] =[ ["a","b"] ], in case both "a" and "b" have high vif values and high correlations, the b values will be eliminated

            #------------------
            Attributes:

                st2_model_results_dic dict
                    A dictionary consist of st2 model results


                #Attributes used in choose_estimator_by_meteo_line 
                dependent_model_selection boolean
                    To select the best model based on meteorological line. only useful if there is a linear refrence line (EX:Isotopes)

                meteo_coef float 
                    If dependent_model_selection=True,global_line, coefficient of the line

                meteo_intercept float
                    If dependent_model_selection=True,global_line, intercept of the line

                selection_method str
                    If dependent_model_selection=True, selection_method: independent,local_line,global_line, point_to_point

                thresh_meteoline_high_scores None type or float
                    A threshold to just consider models with scores higher than that value. if none, equal to mean of scores+std of scores/3
                
                model_selection_report boolean
                    True or False, to determine if there is a need to model selection method report

            #------------------
        """
        #self=iso_meteo_model
        #cls=prep_dataset
        stage_1_2=2
        self.dependent_model_selection=dependent_model_selection
        self.meteo_coef=meteo_coef
        self.meteo_intercept=meteo_intercept
        self.selection_method=selection_method
        self.thresh_meteoline_high_scores=thresh_meteoline_high_scores
        self.model_selection_report=model_selection_report
        Path(self.direc).mkdir(parents=True, exist_ok=True)
        ####################################################
        if model_var_dict==None: model_var_dict=self.dic_second_stage_names
         
        for k,input_v_list in model_var_dict.items():
            if input_v_list==None: input_v_list=self.used_feature_list+self.st1_varname_list #(x,y,z)  + (temp,hum)
            else:
                for i in input_v_list:
                    if i not in list(self.all_preds.columns):
                        raise Exception ("Input variable(s)  not existed in stage one model predictions. input_variables_st2_list elements have to be found in {} ".format(self.all_preds.columns))

            
            input_v_list=input_v_list+self.col_for_f_reg
            ####################################
            # stage 2
            
            for st2_predic_cls in self.cls_list:
                if st2_predic_cls.db_input_args_dics["var_name"]==k:
                    newmatdframe=self.all_preds.rename(columns={k: "Value"})

                    bests,preds_real_dic,bests_dic=rfmethod(
                        newmatdframe,
                        st2_predic_cls.model_pars_name_dic,
                        stage_1_2,
                        args_dic=args_dic,
                        fields=input_v_list,
                        st1_model_month_list=None)
                    results_dic={"bests":bests,"preds_real_dic":preds_real_dic,"bests_dic":bests_dic,"db_input_args_dics":st2_predic_cls.db_input_args_dics}
                    self.st2_model_results_dic[k]=results_dic

                ####################################
        if self.dependent_model_selection==True and dependent_model_selection_list!=None and len(dependent_model_selection_list)==2:
            self.st2_model_results_dic=isotope_model_selection_by_meteo_line(
                dependent_model_selection_list[0],
                dependent_model_selection_list[1],
                self.thresh_meteoline_high_scores,
                self.meteo_coef,self.meteo_intercept,self.selection_method,self.model_selection_report,self.direc,
                self.all_preds,
                self.st2_model_results_dic)
        ####################################
        if output_report==True:

            st2_output_report(self.st2_model_results_dic,self.direc)
            
            #model prediction stats prediction_uncertainty_stats
            Path(os.path.join(self.direc,"st2_prediction_uncertainty_stats")).mkdir(parents=True, exist_ok=True)
            for k,v in self.st2_model_results_dic.items():
                cvun.model_meanstd(k,v["bests_dic"],os.path.join(self.direc,"st2_prediction_uncertainty_stats"))


    def choose_estimator_by_meteo_line(self,dependent_model_selection_list,selection_method="point_to_point",model_selection_report=True,thresh_meteoline_high_scores=None,meteo_coef=8,meteo_intercept=10):
        """
            The method to select the best model based on a (meteorological) line. only useful if there is a linear refrence line (EX:Isotopes).
            This method could be called automatically in st2_fit if dependent_model_selection=True. or it can be called after st2_fit execution
            to see the changes in best regression models based on different criterias.
            
            IMPORTANT NOTE:  Executing this method will update the st2_model_results_dic to match the latest chosen selection_method. st2_model_results_dic stores the second stage results.

            #------------------
            Parameters:
                
                dependent_model_selection_list default=None
                    Used if dependent_model_selection=True. List of two features that have to be used in dependent_model_selection
                
                meteo_coef default=8
                    Used if dependent_model_selection=True and selection_method="global_line". Coefficient of the line
                
                meteo_intercept default=10
                    Used if dependent_model_selection=True and selection_method="global_line". Intercept of the line
                
                selection_method default="point_to_point"
                    Used if dependent_model_selection=True. Selection_method could be:
                    independent
                    local_line: coef and intercept derived from a linear regression of observed data
                    global_line
                    point_to_point: find the models pair with shortest average distance between observed and predicted data  
                
                thresh_meteoline_high_scores None type or float default=None
                    A threshold to just consider models with scores higher than that value. if none, equal to mean of scores+std of scores/3
                
                model_selection_report boolean default =True
                    True or False, to determine if there is a need to model selection method report
            #------------------
            Attributes:

                st2_model_results_dic dict
                    Updated dictionary of st2 model results

                dependent_model_selection boolean
                    To select the best model based on meteorological line. only useful if there is a linear refrence line (EX:Isotopes)

                meteo_coef float 
                    If dependent_model_selection=True,global_line, coefficient of the line

                meteo_intercept float
                    If dependent_model_selection=True,global_line, intercept of the line

                selection_method str
                    If dependent_model_selection=True, selection_method: independent,local_line,global_line, point_to_point

                thresh_meteoline_high_scores None type or float
                    A threshold to just consider models with scores higher than that value. if none, equal to mean of scores+std of scores/3
                
                model_selection_report boolean
                    True or False, to determine if there is a need to model selection method report

            #------------------
        """
        self.meteo_coef=meteo_coef
        self.meteo_intercept=meteo_intercept
        self.thresh_meteoline_high_scores=thresh_meteoline_high_scores
        self.selection_method=selection_method
        self.model_selection_report=model_selection_report
        
        self.st2_model_results_dic=isotope_model_selection_by_meteo_line(
                        dependent_model_selection_list[0],
                        dependent_model_selection_list[1],
                        self.thresh_meteoline_high_scores,
                        self.meteo_coef,self.meteo_intercept,self.selection_method,self.model_selection_report,self.direc,
                        self.all_preds,
                        self.st2_model_results_dic)
    

    def stage2_output_report(self,direc=None):
        """
            This method is useful to update st2_fit output files results in case they are changed.
            (Normally the change can happen if choose_estimator_by_meteo_line method is executed)
            #------------------
            Parameters:
                
                direc str default=None
                    Directory of the output 
            #------------------
        """
        if direc==None: direc=self.direc
        st2_output_report(self.st2_model_results_dic,self.direc)
