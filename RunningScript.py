# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 18:55:55 2021

@author: KIDDO
"""
# =============================================================================
# Importing libraries

import os
import matplotlib.image as img
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd
import glob
from collections import defaultdict
import PIL
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None
from pathlib import Path

import rasterio
from rasterio.plot import show
from rasterio.plot import show_hist
from rasterio.merge import merge
from rasterio.mask import mask
from shapely.geometry import box, Point, mapping
import geopandas as gpd
from fiona.crs import from_epsg
from osgeo import gdal
import pycrs
import pyproj
import json
from functools import partial
from shapely.ops import transform

# =============================================================================
# =============================================================================
# 

# loading data
stdata=pd.read_excel(r'C:\Users\KIDDO\Downloads\SU Study\Traineeship\UsedStations.xlsx')

# defining head directory
headdir=r'C:\Users\KIDDO\Downloads\SU Study\Traineeship\Urban Heat Island\Data_22T_23P'

# getting a list of all subdirectories within the head directory
subdir=[x[0] for x in os.walk(headdir)]

# iterating the analysis through the directories
for i in range (0, len(subdir)):
    for j in range(0,len(stdata)):        
        if subdir[i].find(stdata.iloc[j,2]) == -1:
            print("No analysis here!")
            search_crit = "6*.tif"
            q = os.path.join(subdir[i], search_crit)
            tiff_test = glob.glob(q)
        elif len(tiff_test) <1:
                print('I said, No analysis here!')
        else:
            lat, lon = stdata.iloc[j,3], stdata.iloc[j,4],# defining station's coordinates
            workdir = subdir[i]
            exec(open(r'C:\Users\KIDDO\Downloads\SU Study\Traineeship\Urban Heat Island\python\UHIEMainScript.py').read()) # execute main model

# =============================================================================
