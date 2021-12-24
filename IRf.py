# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 18:26:40 2021

@author: KIDDO
"""

import earthpy
import os
import matplotlib.image as img
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd
import glob
from glob import glob
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


from matplotlib.colors import ListedColormap
import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep

swer='+proj=utm +zone=33 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs'

# =============================================================================
# Preparation

# File and folder paths
# Changing the work directory
workdir=r'C:\Users\KIDDO\Downloads\SU Study\Traineeship\Urban Heat Island\Data_22T_23P\2.Södertälje\Ortofoto_IRF_0_5_m_latest_tif__a367a994-1eb0-4a92-95a5-889537aa1c56_'

os.chdir(workdir)

# Script directories
script_dir = dirpath = os.getcwd()
out_fp = r'Mosaic.tif'

# extracting station's name and image type from parent folders
# station's name
zapath=Path(dirpath)
parentpath=zapath.parent.absolute() # gets parent path of directory
splitparpath=os.path.split(parentpath) # splits parent path
stname = splitparpath[1] # gets station name from split path

# Coordinates:
lat, lon = 59.2141, 17.6291 
northing, easting = 650084.04312309, 6566851.5500514 


# image type
splitpath = os.path.split(dirpath) # splits current path
iminfo = splitpath[1] # full string of image info

# if statement to extract image info from full string
if iminfo.find('1960') != -1:
    print('Historical')
    imtype= 'Historical'
elif iminfo.find('IRF_0_5_m_latest') != -1:
    print('Latest')
    imtype='latest'
else:
    print('Nah!')

# =============================================================================
# Raster Mosaics

# Make a search criteria to select the orthophotos
search_criteria = "6*.tif"
q = os.path.join(dirpath, search_criteria)

# Searching for the tiff files using the glob function
tiff_fps = glob.glob(q)

# creating an empty list for mosaic components
src_files_to_mosaic = []

# opening the mosaic components (tif) in read mode with RasterIO
for fp in tiff_fps:
    src = rasterio.open(fp)
    src_files_to_mosaic.append(src)

# Merging 
mosaic, out_trans = merge(src_files_to_mosaic)

# displaying the results
show(mosaic, cmap='gray')

# Copying the metadata
out_meta = src.meta.copy()

# Updating the metadata
out_meta.update({"driver": "GTiff", "height": mosaic.shape[1],
                  "width": mosaic.shape[2], "transform": out_trans,
                  "crs": swer})

# Writing the mosaic RasterIO to disk
with rasterio.open(out_fp, "w", **out_meta) as dest:
    dest.write(mosaic)
# =============================================================================
# 
# Clipping

# Filepaths
fp = r'Mosaic.tif'
out_tif = r'Clip.tif'

# opening the raster
data = rasterio.open(fp)

# plotting the data
show(data, cmap='gray')

# creating a central point with shapely # lon, lat
point = Point(lon, lat)

# creating a local projection for that point
local_azimuthal_projection = f"+proj=aeqd +R=6371000 +units=m +lat_0={point.y} +lon_0={point.x}"

# transform (wgs84-lap)
wgs84_to_aeqd = partial(pyproj.transform,
                        pyproj.Proj('+proj=longlat +datum=WGS84 +no_defs'),
                        pyproj.Proj(local_azimuthal_projection),)

# transform (lap-wgs84)
aeqd_to_wgs84 = partial(pyproj.transform,
                        pyproj.Proj(local_azimuthal_projection),
                        pyproj.Proj('+proj=longlat +datum=WGS84 +no_defs'),)

# transform (lap-sweref99)
aeqd_to_swer = partial(pyproj.transform,
                       pyproj.Proj(local_azimuthal_projection),
                       pyproj.Proj(swer),)

# first point transformation
point_transformed = transform(wgs84_to_aeqd, point)

# creating a 100m radius buffer (product is 200m) 
loc_buffer = point_transformed.buffer(100)

# final transformation for the shapefile
buffer_wgs84 = transform(aeqd_to_swer, loc_buffer)

# Inserting the buffer into a geodataframe
geo = gpd.GeoDataFrame({'geometry': buffer_wgs84}, index=[0], crs=from_epsg(3006))

# Getting the geometry coordinates in RasterIO-compatible format (E,N)
# Creating the coordinate extraction function
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

# updating the metadata with new dimensions, transform (affine) 
# and CRS (as Proj4 text). 
# NOTE: the crs code here manually reprojects it with
# sweref info but not the sweref is not identified by ArcMap.
out_meta.update({"driver": "GTiff", "height": out_img.shape[1],
                 "width": out_img.shape[2],"transform": out_transform,
                 "crs": swer})

# saving the clipped raster to disk
with rasterio.open(out_tif, "w", **out_meta) as dest:
    dest.write(out_img)
    
# plotting the new clipped raster
clipped = rasterio.open(out_tif)
show(clipped, cmap='gray')

# =============================================================================
# =============================================================================
# Classification

landsat_path = glob("Mosaic.tif")

landsat_path.sort()

for idx, f in enumerate(landsat_path):
    print(f"{idx}: {f}")

# GDAL file and band definitions
driver = gdal.GetDriverByName('GTiff')
file = gdal.Open(r'Clip.tif')

bandir = file.GetRasterBand(2)
b1 = listair = bandir.ReadAsArray()

bandred = file.GetRasterBand(3)
b2 = listared = bandred.ReadAsArray()

bandgreen = file.GetRasterBand(1)
b3 = listagreen = bandgreen.ReadAsArray()

plt.imshow(listair)
plt.imshow(listared)
plt.imshow(listagreen)

ndvi_l8 = es.normalized_diff(b1, b2)
ep.plot_bands(ndvi_l8, cmap="RdYlGn", cols=1, vmin=-1, vmax=1)

# =============================================================================
# =============================================================================
# 
# ndvi_class_bins = [-np.inf, 0, 0.33, 0.36, np.inf]

ndvi_class_bins = [-np.inf, 0.14, 0.27, 0.36, np.inf]

ndvi_landsat_class = np.digitize(ndvi_l8, ndvi_class_bins)

# Apply the nodata mask to the newly classified NDVI data
ndvi_landsat_class = np.ma.masked_where(np.ma.getmask(ndvi_l8), ndvi_landsat_class)

# and check the number of unique values in our now classified dataset
np.unique(ndvi_landsat_class)

# Define color map
nbr_colors = ["lightgray", "honeydew", "forestgreen", "darkgreen"]

# nbr_colors = ['lightblue',"lightgray", 'tan',"honeydew", "forestgreen", "darkgreen"]

nbr_cmap = ListedColormap(nbr_colors)

# Define class names
ndvi_cat_names = ["Urban Areas","light to no vegetation","intermediate Vegetation","Heavy Vegetation"]


# Get list of classes
classes_l8 = np.unique(ndvi_landsat_class)

classes_l8 = classes_l8.tolist()

# The mask returns a value of none in the classes. remove that
classes_l8 = classes_l8[0:4]

# Plot your data
fig, ax = plt.subplots(figsize=(12, 12))

im = ax.imshow(ndvi_landsat_class, cmap=nbr_cmap)

ep.draw_legend(im_ax=im, classes=classes_l8, titles=ndvi_cat_names)

ax.set_title("Landsat 8 - Normalized Difference Vegetation Index (NDVI) Classes", fontsize=14)
# Text(0.5, 1.0, 'Landsat 8 - Normalized Difference Vegetation Index (NDVI) Classes')

ax.set_axis_off()

plt.tight_layout()
# =============================================================================
