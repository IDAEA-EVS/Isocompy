



#####################################################
import pysplit
import pandas as pd
from mpl_toolkits.basemap import Basemap
import os
import shutil
def gen_trajs(iso_18,xy_df_for_hysplit,month_real,altitude,points_all_in_water_report,points_origin_not_detected_report,error_in_meteo_file,traj_shorter_than_runtime_report,input_pass_to_bulkrajfun_report):
    working_dir = r'C:/hysplit4/working'
    meteo_dir = r'C:/Users/Ash kan/Documents/meteo_iso_model/meteo_iso_model_input_code_and_results/inputs/meteo_for_trajs_pysplit/meteo_for_traj_new_name'
    basename = 'RP'
    hours = [12]
    altitudes = [altitude]
    runtime = -650
    all_hysplit=list()
    cols=[
        "ID_MeteoPoint",
        "newLat", 
        "newLong",
        "real_distt_alt_n"+"_"+str(altitude),
        "real_distt_alt_s"+"_"+str(altitude),
        "real_distt_pac_s"+"_"+str(altitude),
        "real_distt_pac_n"+"_"+str(altitude),
        "straight_distt_alt_n"+"_"+str(altitude),
        "straight_distt_alt_s"+"_"+str(altitude),
        "straight_distt_pac_s"+"_"+str(altitude),
        "straight_distt_pac_n"+"_"+str(altitude),
        "percentage_alt_n"+"_"+str(altitude),
        "percentage_alt_s"+"_"+str(altitude),
        "percentage_pac_s"+"_"+str(altitude),
        "percentage_pac_n"+"_"+str(altitude),
        "num_of_all_trajs"+"_"+str(altitude),
        "num_of_unknown_origins"+"_"+str(altitude)
        ]
    #making alt folders
    storage_dir1 =os.path.join(r'C:\trajectories', basename + "\\" +str(altitude))
    if not os.path.isdir(storage_dir1):
        os.mkdir(storage_dir1)
    ######################            

    ###########################
    for index, row in xy_df_for_hysplit.iterrows():
        storage_dir =os.path.join(storage_dir1,str(row["ID_MeteoPoint"]) )
        if not os.path.isdir(storage_dir):
            os.mkdir(storage_dir)
        storage_dir =os.path.join(storage_dir, str(month_real))
        location = (row["newLat"], row["newLong"])  
        tem=iso_18.loc[iso_18["CooX"]==row["CooX"]]
        temp=tem.loc[tem["CooY"]==row["CooY"]]
        dates_1=pd.DatetimeIndex(temp["DateMeas"])
        dates=dates_1[dates_1.month.isin([month_real])]
        years=dates.year.to_list()
        years = list(dict.fromkeys(years))
        #months=dates.month.to_list()
        #months = list(dict.fromkeys(months))
        months=[month_real]
        ###############################
        print_for_input_pass_to_bulkrajfun_report="\n\n##################\n"+str(altitudes)+"\n"+"\n"+str(months)+"\n"+str(years)
        print_for_input_pass_to_bulkrajfun_report="\n\n##################\n"+str(altitudes)+"\n"+str(row["ID_MeteoPoint"])+"\n"+str(months)+"\n"+str(years)
        input_pass_to_bulkrajfun_report.write(print_for_input_pass_to_bulkrajfun_report)
        print (print_for_input_pass_to_bulkrajfun_report)
        ###############################
        #print ("years, month:\n",years, months)
        pysplit.generate_bulktraj(basename, working_dir, storage_dir, meteo_dir,
                                years, months, hours, altitudes, location, runtime,
                                error_in_meteo_file=error_in_meteo_file,
                                meteo_bookends=([4,5], [1]),
                                monthslice=slice(0, 32, 1),
                                meteoyr_2digits=False,
                                get_reverse=False,
                                get_clipped=False)
        if len(os.listdir(storage_dir) )==0:
            print ("remove empty folder!")
            shutil.rmtree( storage_dir) 
            continue
        storage_dir =os.path.join(storage_dir,'*')                       
        umn_tg = pysplit.make_trajectorygroup(storage_dir)
        bm = Basemap()   # default: projection='cyl'
        real_traj_dist_list_all_years_all_hours=list()
        straight_distt_alt_n=0
        straight_distt_alt_s=0
        straight_distt_pac_s=0
        straight_distt_pac_n=0
        cnt_alt_n=0
        cnt_alt_s=0
        cnt_pac_s=0
        cnt_pac_n=0
        real_distt_alt_n=0
        real_distt_alt_s=0
        real_distt_pac_s=0
        real_distt_pac_n=0
        cnt_unk=0
        for traj in umn_tg:
            timestep_is_border=-1*len(traj.data.geometry)
            #sometimes hysplit trajectories is not complete because z in coordination reaches the ground!
            if len(traj.data.geometry)<101:
                #write it to a report file
                traj_shorter_than_runtime_report.write("\n#####Trajectory data#####\n")
                traj_shorter_than_runtime_report.write(str(traj.data.loc[0]))
                traj_shorter_than_runtime_report.write("\n")
                print("traj_shorter_than_runtime")
                continue
            for i in range(len(traj.data.geometry)-1,-1,-1):
                timestep_is_border=timestep_is_border+1
                lng=traj.data.geometry[-i].x
                lat=traj.data.geometry[-i].y
                if (lng >-100 and lng <-32) and (lat>-60 and lat<14) and bm.is_land(lng, lat)==True:  #True or False
                    #print ("timestep_is_border:",timestep_is_border)
                    break
            if timestep_is_border==0:
                print("points_all_in_water")
                points_all_in_water_report.write("\n\n##########\n##########\n")
                points_all_in_water_report.write(str(traj.data.loc[timestep_is_border]))
                points_all_in_water_report.write("\n")
                continue
                #print ("ATTENTION, Still in land, more timesteps!!!!")    
            #else:
            #    timestep_is_border=timestep_is_border+1
            #print ("timestep_is_water",timestep_is_border)
            '''traj.generate_reversetraj(working_dir, meteo_dir,run=timestep_is_border,
                                reverse_dir='default',
                                meteo_interval='monthly',
                                hysplit="C:\\hysplit4\\exec\\hyts_std")'''
            #traj.load_reversetraj()
            #traj.calculate_distance(reverse=True)
            traj.calculate_distance()
            #print("len",len(traj.data.geometry))
            long_coast=traj.data.loc[timestep_is_border].geometry.x
            lat_coast=traj.data.loc[timestep_is_border].geometry.y
            if long_coast <= -32 and long_coast >=-69 and lat_coast>=-15 and lat_coast<=14: #ok
                continentality="atl_n"
                cnt_alt_n=cnt_alt_n+1
                straight_distt_alt_n=straight_distt_alt_n+traj.data.loc[timestep_is_border]["Dist_from_origin"]
                real_distt_alt_n=real_distt_alt_n+traj.data.loc[timestep_is_border]["Cumulative_Dist"]
            elif long_coast <= -32 and long_coast >=-69 and lat_coast>=-60 and lat_coast<=-15: #ok
                continentality="atl_s"
                cnt_alt_s=cnt_alt_s+1
                straight_distt_alt_s=straight_distt_alt_s+traj.data.loc[timestep_is_border]["Dist_from_origin"]
                real_distt_alt_s=real_distt_alt_s+traj.data.loc[timestep_is_border]["Cumulative_Dist"]
            elif long_coast <= -69 and long_coast >=-100 and lat_coast>=-60 and lat_coast<=-15:    
                continentality="pac_s"
                cnt_pac_s=cnt_pac_s+1
                straight_distt_pac_s=straight_distt_pac_s+traj.data.loc[timestep_is_border]["Dist_from_origin"]
                real_distt_pac_s=real_distt_pac_s+traj.data.loc[timestep_is_border]["Cumulative_Dist"]
            elif long_coast <= -69 and long_coast >=-100 and lat_coast>=-15 and lat_coast<=14:    
                continentality="pac_n"
                cnt_pac_n=cnt_pac_n+1
                straight_distt_pac_n=straight_distt_pac_n+traj.data.loc[timestep_is_border]["Dist_from_origin"]
                real_distt_pac_n=real_distt_pac_n+traj.data.loc[timestep_is_border]["Cumulative_Dist"]

            else:
                continentality="UNKNOWN"
                cnt_unk=cnt_unk+1
                points_origin_not_detected_report.write("\n\n##### First point #####\n")
                points_origin_not_detected_report.write(str(traj.data.loc[0].geometry))
                points_origin_not_detected_report.write("\n")
                points_origin_not_detected_report.write(str(traj.data.loc[0].DateTime))
                points_origin_not_detected_report.write("\n#####last point #####\n")
                points_origin_not_detected_report.write(str(timestep_is_border))
                points_origin_not_detected_report.write("\n")
                points_origin_not_detected_report.write(str(traj.data.loc[timestep_is_border]))
                print ("Attention: couldn't find the origin, skipping!",timestep_is_border)

        average_real_dist=list()
        average_straight_distt=list()
        porc_list=list()
        cnt_sum=(cnt_alt_n+cnt_alt_s+cnt_pac_n+cnt_pac_s)        
        real_dist_list=[real_distt_alt_n,real_distt_alt_s,real_distt_pac_s,real_distt_pac_n]  
        str_dist_list=[straight_distt_alt_n,straight_distt_alt_s,straight_distt_pac_s,straight_distt_pac_n]
        cnt_list=[cnt_alt_n,cnt_alt_s,cnt_pac_s,cnt_pac_n]    
        for real_dist,str_dist,cnt in zip(real_dist_list,str_dist_list,cnt_list): 
            porc_list.append(cnt/cnt_sum)
            try:
                average_real_dist.append(real_dist/cnt)
                average_straight_distt.append(str_dist/cnt)
            except:
                average_real_dist.append(0)
                average_straight_distt.append(0)
        te=[row["ID_MeteoPoint"],row["newLat"], row["newLong"]]+ average_real_dist+ average_straight_distt+porc_list+[cnt_sum]+[cnt_unk]    
        all_hysplit.append(te)
    all_hysplit_df = pd.DataFrame(all_hysplit, columns =cols)
    #all_hysplit_df.to_excel(r"C:\Users\Ash kan\Documents\meteo_iso_model\meteo_iso_model_input_code_and_results\output\all_hysplit.xls")
    return all_hysplit_df     
        ###
        # output have to be something with the same dimension as xyhysplit so it can be added to all_preds after this function!        


#############################################
#############################################

#Ashkan 25/2/2020 change coordination for pysplit
def convertCoords(row):
    from pyproj import Proj, transform
    inProj = Proj(init='epsg:32719')
    outProj = Proj(init='epsg:4326')
    x2,y2 = transform(inProj,outProj,row['CooX'],row['CooY'])
    return pd.Series({'newLong':x2,'newLat':y2})