"""
Isocompy repository contains an open source Python library that focuses on
 user defined (such as meteorological, spatial, etc.) and isotopic composition
  variables analysis and generating the regression - statistical estimation 
  models. 

Classes: preprocess, model, session, evaluation, stats, plots
"""
__version__ = "1.0.0"

from .data_preparation import preprocess
from .reg_model import model
from .tools import session,evaluation,stats,plots

