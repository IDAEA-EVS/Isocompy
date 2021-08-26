import pysplit
import pandas as pd
from mpl_toolkits.basemap import Basemap
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import pysplit_funcs_for_meteo_model



all_hysplit_df,nw_ex_all_df=pysplit_funcs_for_meteo_model.gen_trajs(
    iso_18,
    xy_df_for_hysplit,
    month_real,altitude,
    points_all_in_water_report,
    points_origin_not_detected_report,
    error_in_meteo_file,
    traj_shorter_than_runtime_report,
    input_pass_to_bulkrajfun_report,
    Sampling_date_db=False)
