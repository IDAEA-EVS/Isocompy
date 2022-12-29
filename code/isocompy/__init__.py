"""
An open source library for environmental isotopic modelling 
"""
__version__ = "1.0.0.1"

from .data_preparation import preprocess
from .reg_model import model
from .tools import session,evaluation,stats,plots

