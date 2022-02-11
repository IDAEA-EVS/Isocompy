from bokeh.plotting import figure
from bokeh.io import save
from bokeh.transform import linear_cmap
from bokeh.palettes import Plasma256 as palette
from bokeh.models import ColorBar,ColumnDataSource
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Point
from mpl_toolkits.axes_grid1 import make_axes_locatable
#to create maps without a shape file: utm inputs Proj has to be defined to make the conversation to lat-long
def create_maps_bokeh(df,feat,dir,CooX,CooY,unit,opt_title,observed_data):

    p = figure(tools='pan, wheel_zoom', match_aspect=True)
    ##########
    ##########
    source=ColumnDataSource(df)
    print (df.columns)

    # we use the mapper for the color of the circles
    mapper = linear_cmap(feat, palette, df[feat].min(), df[feat].max()) 

    if opt_title==None:
        p=figure(title= feat + "  ( "+unit+ ")", match_aspect=True)
    else:    
        p=figure(title=opt_title, match_aspect=True)

    #draw the points
    p.square(CooX,CooY, size=6, alpha=0.1,color=mapper,source=source)

    if isinstance(observed_data,pd.DataFrame)==True and observed_data.empty==False:
        observed_data["legend"]="Measured data"
        source_obs=ColumnDataSource(observed_data)
        p.circle(CooX,CooY, size=10, alpha=0.8,color='black',source=source_obs,legend="legend")

    # and we add a color scale to see which values the colors 
    # correspond to 
    color_bar = ColorBar(color_mapper=mapper['transform'], 
                            location=(0,0))
    p.add_layout(color_bar, 'right')

    save(p,dir)
##############
#Example:
#myProj = "+proj=utm +zone=19S, +south +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
#data_file = r"C:\Users\Ash kan\Documents\meteo_iso_model_input_code_and_results\case_studies\salar_de_atacama\1_all_vars_st2_prediction.xls"
#df=pd.read_excel(data_file,sheet_name="1_all_vars_st2_prediction",header=0,index_col=False,keep_default_na=True)
#create_maps_bokeh(df,feat="prc",myProj=myProj)
#create_maps_bokeh(df,feat="predicted_iso_18",myProj=myProj)
#########################
#########################

#to create maps with an specified shapefile
def make_maps_gpd(df,shp_dir,feat,dir,CooX,CooY,unit,opt_title,observed_data):

    #prepare the geodataframe
    df["Coordinates"] = list(zip(df[CooX], df[CooY]))
    df["Coordinates"] = df["Coordinates"].apply(Point)
    gdf = gpd.GeoDataFrame(df, geometry="Coordinates")
    df=df.drop(columns=['Coordinates'])
    ########
    #plot

    fig, ax = plt.subplots()    

    #observed_data
    if isinstance(observed_data,pd.DataFrame)==True and observed_data.empty==False:
        #prepare the geodataframe
        ax.scatter(observed_data[CooX],observed_data[CooY],marker='x',s=8,alpha=1,color='black',label="Measured data")
        legend1=ax.legend(prop={'size': 6},loc='lower right',facecolor='None',edgecolor='black')
        fig.gca().add_artist(legend1)    


    #set the legend
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    #plt the feature    
    
    
    gdf_p=gdf.plot( marker='s', markersize=8,alpha=1,column=feat,legend=True,cax=cax,ax=ax, cmap="viridis")
    fig = gdf_p.figure
    cb_ax = fig.axes[1]
    cb_ax.tick_params(labelsize=5)
    #plt.colormaps()



    #plot the base shape    
    #read the shape file
    if shp_dir!=None:
        shp_gdf=gpd.read_file(shp_dir)
        shp_gdf.plot(ax=gdf_p, edgecolor='black',color='None')

    if isinstance(observed_data,pd.DataFrame)==True and observed_data.empty==False:
        #prepare the geodataframe
        ax.scatter(observed_data[CooX],observed_data[CooY],marker='x',s=8,alpha=1,color='black',label="Measured data")
    #set plot limits
    ax.set_xlim([gdf.CooX.min(), gdf.CooX.max()])
    ax.set_ylim([gdf.CooY.min(), gdf.CooY.max()])
    
    #set labels
    #gdf_p.set_xlabel(str(CooX))
    #gdf_p.set_ylabel(str(CooY))
    if opt_title==None:
        gdf_p.set_title(feat + "  ( "+unit+ ")",fontSize=10)
    else:    
        gdf_p.set_title(opt_title,fontSize=10)

    plt.setp(ax.get_yticklabels(), rotation=90,fontsize=5)
    plt.setp(ax.get_xticklabels(), rotation=0,fontsize=5)
    #gdf_p.spines['top'].set_visible(False)
    #gdf_p.spines['right'].set_visible(False)
    plt.savefig(dir,dpi=300)
    plt.close("all")
##############
#Example:
#data_file = r"C:\Users\Ash kan\Documents\meteo_iso_model_input_code_and_results\case_studies\salar_de_atacama\manual_filtered_data\outputs\preds_map_all_points\1_all_vars_st2_prediction.csv"
#df=pd.read_csv(data_file,header=0,index_col=False,keep_default_na=True)
#shp_dir=r"C:\Users\Ash kan\Desktop\shpestaciones\Cuenca.shp"
#make_maps_gpd(df,shp_dir,feat="CooZ")