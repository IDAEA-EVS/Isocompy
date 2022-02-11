from isocompy.data_prep import data_preparation_func
from pathlib import Path
import numpy as np

class preprocess(object):
    """
        The class to preprocess the input variables of each model group and initiate
        the models properties such as cross-validation and  brute-force searching.

        #------------------
        Parameters:

            (__init__ method of preprocess class)

            tunedpars_rfr: dic default={"min_weight_fraction_leaf":[0,0.02,0.04],"n_estimators":[50,100,150,200,250,300],"criterion": ["mse","mae"],"min_samples_split":[2,5] }
                The dictionary that determines brute-force searching parameters of Random Forest regression. For more details on the regression method inputs, refer to sklearn library.
            
            tunedpars_svr: dic default={"kernel":[ "poly", "rbf", "sigmoid"],"C":np.logspace(-1, 1, 3),"gamma":np.logspace(-3, 1, 3) }
                The dictionary that determines brute-force searching parameters of Support Vector regression. For more details on the regression method inputs, refer to sklearn library.
            
            tunedpars_nusvr: dic default={"kernel":["linear", "poly", "rbf", "sigmoid"] }
                The dictionary that determines brute-force searching parameters of Nu Support Vector Regression. For more details on the regression method inputs, refer to sklearn library.
            
            tunedpars_mlp: dic default={"activation" : [ "logistic", "tanh"],"solver" : ["lbfgs", "sgd", "adam"],"alpha":[0.0001,0.0003],"hidden_layer_sizes":[(50,)*2,(50,)*3,(50,)*4,(100,)*2,(100,)*3,(100,)*4],"max_iter":[1000],"n_iter_no_change":[10]}
                The dictionary that determines brute-force searching parameters of Multilayer Perceptron regression. For more details on the regression method inputs, refer to sklearn library.

            tunedpars_lr: dic default={}
                The dictionary that determines brute-force searching parameters of Linear regression. For more details on the regression method inputs, refer to sklearn library.

            tunedpars_br: dic default={}
                The dictionary that determines brute-force searching parameters of Bayesian Ridge regression. For more details on the regression method inputs, refer to sklearn library.

            tunedpars_ard: dic default={}
                The dictionary that determines brute-force searching parameters of Bayesian ARD regression. For more details on the regression method inputs, refer to sklearn library.

            tunedpars_omp: dic default={}
                The dictionary that determines brute-force searching parameters of Orthogonal Matching Pursuit regression. For more details on the regression method inputs, refer to sklearn library.
                
            tunedpars_elnet: dic default={"l1_ratio":[.1, .5, .7,.9,.99]}
                The dictionary that determines brute-force searching parameters of ElasticNet (Linear regression with combined L1 and L2 priors as regularizer) regression. For more details on the regression method inputs, refer to sklearn library.

            tunedpars_muelnet: dic default={"l1_ratio":[.1, .5, .7,.9,.99]}
                The dictionary that determines brute-force searching parameters of Multi-task ElasticNet (trained with L1/L2 mixed-norm as regularizer) regression. For more details on the regression method inputs, refer to sklearn library.
            
            which_regs: dic default={"muelnet":True,"rfr":True,"mlp":True,"elnet":True,"omp":True,"br":True,"ard":True,"svr":True,"nusvr":False}
                The dictionary that detemines which regression models have to be included in cross-validation and  brute-force searching process

            apply_on_log: Boolean default=True
                If True, Apart from the main values, fits the models to log(1 + x) (Natural logarithm) of data. If the scores of the regression on logarithm of the data is higher
                than the real values, The chosen method will always calculate the log(1 + x) of the data before fitting the models. Note that if this is the case, to have the real
                outputs, exp(x) - 1 (the inverse of log(1 + x)) will be calculated.
            cv: int or Boolean default="auto"
                If cv="auto", the cross-validation number of folds will be calculated automatically. It is beneficial when there is few data available (max=10, min=2). If cv is an integer values,
                it determines number of folds of the cross-validation.

        #------------------
        Methods:

            __init__(self)

            fit(self, inp_var, var_name, fields,direc,
            remove_outliers=True, write_outliers_input=True, year_type="all", inc_zeros_inp_var=False, write_integrated_data=True,
            q1=0.05, q3=0.95, IQR_inp_var=True, IQR_rat_inp_var=3, mean_mode_inp_var="arithmetic", elnino=None, lanina=None)

            model_pars(self,**kwargs)
        
        #------------------
        Attributes:

            model_pars_name_dic dic
                A dictionary that stores brute-force searching parameters of all models. The keys and values are dictionary names and parameters dictionaries respectively
                
                EXAMPLE:
                
                    key: "tunedpars_rfr"
                    Value: tunedpars_rfr= {"min_weight_fraction_leaf":[0,0.04],"n_estimators":[150,200],"criterion": ["mse"] }

                IMPORTANT NOTE: 'model_pars' method have to be used to change the brute-force searching parameters

            db_input_args_dics dic 
                    A dictionary that stores input parameters of the fit method
                
            direc str
                The directory of the data preprocessing class output

            month_grouped_inp_var list
                A list of monthly grouped data. Each element of the list is a dataframe containing the grouped data of an specific month 


        #------------------
    """

    

    def __init__(self):
        tunedpars_rfr={"min_weight_fraction_leaf":[0,0.02,0.04],"n_estimators":[50,100,150,200,250,300],"criterion": ["mse","mae"],"min_samples_split":[2,5] }
        tunedpars_svr={"cache_size":[10000],"kernel":[ "poly", "rbf", "sigmoid"],"C":np.logspace(-1, 1, 3),"gamma":np.logspace(-3, 1, 3) }
        tunedpars_nusvr={"kernel":["linear", "poly", "rbf", "sigmoid"] }
        tunedpars_mlp={"activation" : [ "logistic", "tanh"],"solver" : ["lbfgs", "sgd", "adam"],"alpha":[0.0001,0.0003],"hidden_layer_sizes":[(50,)*2,(50,)*3,(50,)*4,(100,)*2,(100,)*3,(100,)*4],"max_iter":[1000],"n_iter_no_change":[10]}
        which_regs={"muelnet":True,"rfr":True,"mlp":True,"elnet":True,"omp":True,"br":True,"ard":True,"svr":True,"nusvr":False}
        tunedpars_lr={}
        tunedpars_br={}
        tunedpars_ard={}
        tunedpars_omp={}
        tunedpars_muelnet={"l1_ratio":[.1, .5, .7,.9,.99]}
        tunedpars_elnet={"l1_ratio":[.1, .5, .7,.9,.99]}
        apply_on_log=True
        cv="auto"

        self.month_grouped_inp_var=[]
        self.direc=r""
        self.db_input_args_dics={
        "var_name":"",
        "fields":[],
        "direc":self.direc,
        "remove_outliers":True,
        "write_outliers_input":True,
        "year_type":"all",
        "inc_zeros_inp_var":False,
        "write_integrated_data":True,
        "q1":0.05,
        "q3":0.95,
        "IQR_inp_var":False,
        "IQR_rat_inp_var":3,
        "mean_mode_inp_var":"arithmetic",
        "per_year_integration_method":"mean",
        "elnino":[],
        "lanina":[]}



        
        self.model_pars_name_dic={"tunedpars_rfr":tunedpars_rfr,
        "tunedpars_svr":tunedpars_svr,
        "tunedpars_nusvr":tunedpars_nusvr,
        "tunedpars_mlp":tunedpars_mlp,
        "tunedpars_lr":tunedpars_lr,
        "tunedpars_br":tunedpars_br,
        "tunedpars_ard":tunedpars_ard,
        "tunedpars_muelnet":tunedpars_muelnet,
        "tunedpars_elnet":tunedpars_elnet,
        "tunedpars_omp":tunedpars_omp,
        "which_regs":which_regs,
        "apply_on_log":apply_on_log,
        "cv":cv}



    def fit(self,
        inp_var,
        var_name,
        fields,
        direc,
        remove_outliers=True,
        write_outliers_input=True,
        year_type="all",
        inc_zeros_inp_var=True,
        write_integrated_data=True,
        q1=0.05,
        q3=0.95,
        IQR_inp_var=True,
        IQR_rat_inp_var=3,
        mean_mode_inp_var="arithmetic", #geometric
        per_year_integration_method="mean", #"sum"
        elnino=None,
        lanina=None):

        """
            The method to preprocess the input data of the models

            #------------------
            Parameters:
            
                inp_var: Pandas dataframe
                    Input pandas dataframe containing the dependent value, unique IDs
                    for each sample, and date. (The columns "Value", "ID" and "Date" have to be found in the dataframe)
                
                var_name: str
                    Name of the dependent variable.
                
                fields: list of strings
                    List of independent variable names that have to be existed in the inp_var dataframe
                
                direc: str
                    The directory of the data preprocessing class output
                
                remove_outliers: Boolean default=True
                    To remove outliers based on the introduced variables. If False, the variables related
                    to removing outliers will be ignored.
                
                write_outliers_input: Boolean default=True
                    Effective if remove_outliers=True. To generate .xls file in directory folders of the class after removing outliers
                
                year_type: str default="all"
                    "all", "elnino" or "lanina". In case elnino or lanina years have to be selected from the database

                inc_zeros_inp_var: Boolean default=False
                    Effective if remove_outliers=True. Removing outliers could be done not considering the zero values in the database.
                    (The zero values will be seperated, outliers will be removed, then zero values will be added to database)
                
                write_integrated_data: Boolean default=True
                    Generate two .xls outputs consisting of integrated data, and the quantity of data in each month
                
                q1: float default=0.05
                    Effective if remove_outliers=True and IQR_inp_var=False. Lower percentile limit to determine the outliers.
                
                q3: float default=0.95
                    Effective if remove_outliers=True  and IQR_inp_var=False. Upper percentile limit to determine the outliers
                
                IQR_inp_var: Boolean default=True
                    Effective if remove_outliers=True. Determining the upper limit of the outliers using this 
                    formula: X*q_0.75 + IQR_rat*abs(X*q_0.25 - X*q_0.75)
                    Lower limit = q1
                
                IQR_rat_inp_var: float default=3
                    Effective if remove_outliers=True and IQR_inp_var=True. This parameter used in  X*q_0.75 + IQR_rat*abs(X*q_0.25 - X*q_0.75) 
                    to determine upper boundary limit.
                
                mean_mode_inp_var: str default="arithmetic"
                    Data averaging method. available options are arithmetic  or geometric

                per_year_integration_method: str default="mean"
                    Data integration method in year month of each year. available options are "mean" and "sum"
                
                elnino: None type or list of integers default=None
                    List of elnino years
                
                lanina: None type or list of integers default=None
                    List of lanina years

            #------------------
            Attributes:

                db_input_args_dics dic 
                    A dictionary that stores input parameters of the fit method
                
                direc str
                    The directory of the data preprocessing class output

                month_grouped_inp_var list
                    A list of monthly grouped data. Each element of the list is a dataframe containing the grouped data of an specific month 

            #------------------
        """


        #make a dictionary of the input args
        self.db_input_args_dics={
        "var_name":var_name,
        "fields":fields,
        "direc":direc,
        "remove_outliers":remove_outliers,
        "write_outliers_input":write_outliers_input,
        "year_type":year_type,
        "inc_zeros_inp_var":inc_zeros_inp_var,
        "write_integrated_data":write_integrated_data,
        "q1":q1,
        "q3":q3,
        "IQR_inp_var":IQR_inp_var,
        "IQR_rat_inp_var":IQR_rat_inp_var,
        "mean_mode_inp_var":mean_mode_inp_var, #geometric
        "per_year_integration_method":per_year_integration_method, #sum or mean
        "elnino":elnino,
        "lanina":lanina}
        self.direc=direc
        Path(self.direc).mkdir(parents=True, exist_ok=True)
        if elnino==None: elnino=[]
        if lanina==None: lanina=[]
        
    
        self.month_grouped_inp_var,self.inp_var=data_preparation_func(
            inp_var,
            var_name,
            fields,
            direc,
            remove_outliers,
            q1,
            q3,
            IQR_inp_var,
            inc_zeros_inp_var,
            write_outliers_input,
            write_integrated_data,
            IQR_rat_inp_var,
            year_type,
            elnino,
            lanina,
            mean_mode_inp_var=mean_mode_inp_var,
            per_year_integration_method=per_year_integration_method)      


    def model_pars(self,**kwargs):
        """
            To change the brute-force searching parameters that is already determined in a class. 

            #------------------
            EXAMPLE:

            #Define the new brute-force searching parameters dictionaries:

            brutesearch_ran_for_dic={"min_weight_fraction_leaf":[0,0.04],"n_estimators":[150,200],"criterion": ["mse"] }

            brutesearch_elasticnet_dic={"l1_ratio":[ .5, .9]}

            prep_class.model_pars( "tunedpars_rfr" = brutesearch_ran_for_dic, "tunedpars_elnet" = brutesearch_elasticnet_dic )
            
            #------------------
        """
        for k,v in kwargs.items():
            if str(k) in list(self.model_pars_name_dic.keys()):
                self.model_pars_name_dic[str(k)]=v
            else: print ("The variable name {} is incorrect! check and try again".format(k))




