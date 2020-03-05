



#####################################################
import pysplit
import pandas as pd
from mpl_toolkits.basemap import Basemap
import os
def gen_trajs(iso_18,xy_df_for_hysplit,month_real):
    working_dir = r'C:/hysplit4/working'
    meteo_dir = r'C:/Users/Ash kan/Documents/meteo_iso_model/meteo_iso_model_input_code_and_results/inputs/meteo_for_trajs_pysplit/meteo_for_traj_new_name'
    basename = 'RP'
    hours = [12]
    altitudes = [500]
    runtime = -450
    cntttrr=0
    all_hysplit=list()
    cols=[
        "ID_MeteoPoint",
        "newLat", 
        "newLong",
        "real_distt_alt_n",
        "real_distt_alt_s",
        "real_distt_pac_s",
        "real_distt_pac_n",
        "straight_distt_alt_n",
        "straight_distt_alt_s",
        "straight_distt_pac_s",
        "straight_distt_pac_n",
        "percentage_alt_n",
        "percentage_alt_s",
        "percentage_pac_s",
        "percentage_pac_n",
        "num_of_all_trajs","num_of_unknown_origins"
        ]
    for index, row in xy_df_for_hysplit.iterrows():
        cntttrr=cntttrr+1
        storage_dir =os.path.join(r'C:\trajectories', basename + "\\" + str(row["ID_MeteoPoint"]) )
        if not os.path.isdir(storage_dir):
            os.mkdir(storage_dir)
        storage_dir =os.path.join(storage_dir, str(month_real))
        location = (row["newLat"], row["newLong"])  
        tem=iso_18.loc[iso_18["CooX"]==row["CooX"]]
        temp=tem.loc[tem["CooY"]==row["CooY"]]
        dates=pd.DatetimeIndex(temp["DateMeas"])
        years=dates.year.to_list()
        years = list(dict.fromkeys(years))
        #months=dates.month.to_list()
        #months = list(dict.fromkeys(months))
        months=[month_real]
        #print ("years, month:\n",years, months)
        pysplit.generate_bulktraj(basename, working_dir, storage_dir, meteo_dir,
                                years, months, hours, altitudes, location, runtime,meteo_bookends=([4,5], [1]), 
                                monthslice=slice(0, 32, 1), get_reverse=False,
                                get_clipped=False)
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
            timestep_is_border=1
            for i in traj.data.geometry:
                timestep_is_border=timestep_is_border-1
                if bm.is_land(i.x, i.y)==False:  #True or False
                    break
            if timestep_is_border==traj.data.Timestep.min():
                print ("ATTENTION, Still in land, more timesteps!!!!")    
            #else:
            #    timestep_is_border=timestep_is_border+1
            print ("timestep_is_water",timestep_is_border)
            '''traj.generate_reversetraj(working_dir, meteo_dir,run=timestep_is_border,
                                reverse_dir='default',
                                meteo_interval='monthly',
                                hysplit="C:\\hysplit4\\exec\\hyts_std")'''
            #traj.load_reversetraj()
            #traj.calculate_distance(reverse=True)
            traj.calculate_distance()
            long_coast=traj.data.loc[timestep_is_border].geometry.x
            lat_coast=traj.data.loc[timestep_is_border].geometry.y
            if long_coast <= -35.1 and long_coast >=-64.7 and lat_coast>=-15 and lat_coast<=10:
                continentality="atl_n"
                cnt_alt_n=cnt_alt_n+1
                straight_distt_alt_n=straight_distt_alt_n+traj.data.loc[timestep_is_border]["Dist_from_origin"]
                real_distt_alt_n=real_distt_alt_n+traj.data.loc[timestep_is_border]["Cumulative_Dist"]
            elif long_coast <= -35.1 and long_coast >=-64.7 and lat_coast>=-54.7 and lat_coast<=-15:
                continentality="atl_s"
                cnt_alt_s=cnt_alt_s+1
                straight_distt_alt_s=straight_distt_alt_s+traj.data.loc[timestep_is_border]["Dist_from_origin"]
                real_distt_alt_s=real_distt_alt_s+traj.data.loc[timestep_is_border]["Cumulative_Dist"]
            elif long_coast <= -64.7 and long_coast >=-75.5 and lat_coast>=-54.7 and lat_coast<=-15:    
                continentality="pac_s"
                cnt_pac_s=cnt_pac_s+1
                straight_distt_pac_s=straight_distt_pac_s+traj.data.loc[timestep_is_border]["Dist_from_origin"]
                real_distt_pac_s=real_distt_pac_s+traj.data.loc[timestep_is_border]["Cumulative_Dist"]
            elif long_coast <= -64.7 and long_coast >=-75.5 and lat_coast>=-15 and lat_coast<=10:    
                continentality="pac_n"
                cnt_pac_n=cnt_pac_n+1
                straight_distt_pac_n=straight_distt_pac_n+traj.data.loc[timestep_is_border]["Dist_from_origin"]
                real_distt_pac_n=real_distt_pac_n+traj.data.loc[timestep_is_border]["Cumulative_Dist"]

            else:
                continentality="UNKNOWN"
                cnt_unk=cnt_unk+1
                print (traj.data.loc[timestep_is_border])
                print ("Attention: couldn't find the origin, skipping!")

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
    '''#join trajs to trajs groups
    for each trajs:
        check points in polygon. 
        if not, pop put (from the end)
        trajs_dist.append(trajs.dist)
        clustering? 4 or 3
        trajs coordinate check. (4groups)
    add info to original data frame'''
    return 

#Ashkan 25/2/2020 change coordination for pysplit
def convertCoords(row):
    from pyproj import Proj, transform
    inProj = Proj(init='epsg:32719')
    outProj = Proj(init='epsg:4326')
    x2,y2 = transform(inProj,outProj,row['CooX'],row['CooY'])
    return pd.Series({'newLong':x2,'newLat':y2})