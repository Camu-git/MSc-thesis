import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import netCDF4 as nc

from matplotlib.widgets import Slider
import IPython
import ipympl

import dfm_tools as dfmt
import matplotlib
matplotlib.use('TkAgg')

from interactive import *

# %matplotlib widget

trim2 = "Test_Q200_Islope_3/Ashld_0.2/trim-001.nc"

# PlotCrossXDIR(trim2, [40,50,60])

PlotVerticalCrossSection(trim2)