# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 15:24:33 2021

@author: KIDDO
"""

# =============================================================================
# 

import os
os.chdir(r'C:\Users\KIDDO\Downloads\SU Study\Traineeship\Urban Heat Island\Data_22T_23P\1.Tullinge A\Historiska_ortofoton_1960_PAN_tif__d9c09fff-9a63-40a6-96fe-0f5046249428_')
import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL

import rasterio
from rasterio.plot import show
from rasterio.plot import show_hist
from rasterio.mask import mask
from shapely.geometry import box
import geopandas as gpd
from fiona.crs import from_epsg
from osgeo import gdal
import pycrs

# import georaster uninstalled
# conda install -c conda-forge georaster
# =============================================================================
# PIL.Image.MAX_IMAGE_PIXELS = 100000000

# # RasterIO Method
# fp = r'656_66_05.tif'
# image = rasterio.open(fp)
# show(image)

# # GDAL & Matplotlib Method
# dataset = gdal.Open(fp, gdal.GA_ReadOnly) 

# # Note GetRasterBand() takes band no. starting from 1 not 0
# band = dataset.GetRasterBand(1)
# arr = band.ReadAsArray()
# plt.imshow(arr)

# =============================================================================
# Clipping

# Filepaths
fp = r'656_66_05.tif'
out_tif = r'ClipExp.tif'

# opening the raster
data = rasterio.open(fp)

# plotting the data
show(data, cmap='gray')

# Tullinge A Coordinates:
# Decimal (lat,lon): (59.1785, 17.9093) 
# Northing easting (E,N): (666246.163183566, 6563554.18604587) 

# creating a boundary box with shapely around the Area of interest
# WGS84 coordinates

# x=0.01
# minx, miny = 17.9093-x, 59.1785-x
# maxx, maxy = 17.9092+x, 59.1785+x

x=1000
minx, miny = 666246.163183566-x, 6563554.18604587-x
maxx, maxy = 666246.163183566+x, 6563554.18604587+x

bbox = box(minx, miny, maxx, maxy)

# Inserting the box into a geodataframe
# geo = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=from_epsg(4326))
geo = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=from_epsg(3006))

# Re-project into the same coordinate system as the raster data (sweref99)
# geo = geo.to_crs(crs=data.crs.data)

# Getting the geometry coordinates in RasterIO-compatible format (E,N)

# Creating the coordinate function
def getFeatures(gdf):
    import json
    return [json.loads(gdf.to_json())['features'][0]['geometry']]

# Getting the coordinates
coords = getFeatures(geo)

# Clipping the raster with the polygon
out_img, out_transform = mask(dataset=data, shapes=coords, crop=True)

# Copying the metadata
out_meta = data.meta.copy()
print(out_meta)

# Parsing EPSG code from the (coordinate reference system) CRS (creating a Proj4 string to ensure 
# that projection information is saved properly)
epsg_code = int(data.crs.data['init'][5:])
print(epsg_code)

# updating the metadata with new dimensions, transform (affine) 
# and CRS (as Proj4 text). NOTE: the crs code here manually reprojects it with
# sweref info but not the sweref is not identified by ArcMap.
out_meta.update({"driver": "GTiff", "height": out_img.shape[1],
                 "width": out_img.shape[2],"transform": out_transform,
                 "crs": '+proj=utm +zone=33 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs'})
                 # "crs": pycrs.parse.from_epsg_code(epsg_code).to_proj4()})

# saving the clipped raster to disk
with rasterio.open(out_tif, "w", **out_meta) as dest:
    dest.write(out_img)

# plotting our new clipped raster

clipped = rasterio.open(out_tif)
show(clipped, cmap='gray')

# =============================================================================

