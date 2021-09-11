# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 15:24:33 2021

@author: MahmoudSuliman
"""

# =============================================================================
# 

import os
os.chdir(r'C:\Users\KIDDO\Downloads\SU Study\Traineeship\Urban Heat Island\Data_22T_23P\2.Södertälje\Historiska_ortofoton_1960_PAN_tif__493271c1-5839-4fc1-b9cb-081f4f83da6d_')
import matplotlib.image as img
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd
import glob
import PIL
from PIL import Image, ImageDraw

import rasterio
from rasterio.plot import show
from rasterio.plot import show_hist
from rasterio.merge import merge
from rasterio.mask import mask
from shapely.geometry import box
import geopandas as gpd
from fiona.crs import from_epsg
from osgeo import gdal
import pycrs

# import georaster
# conda install -c conda-forge georaster

swer='+proj=utm +zone=33 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs'
# =============================================================================
# Raster Mosaic and preparation

############# add an if statement to this so that it does not automatically
############# mosaics

# File and folder paths
dirpath = r'C:\Users\KIDDO\Downloads\SU Study\Traineeship\Urban Heat Island\Data_22T_23P\2.Södertälje\Historiska_ortofoton_1960_PAN_tif__493271c1-5839-4fc1-b9cb-081f4f83da6d_'
out_fp = r'Mosaic.tif'

# Make a search criteria to select the orthophotos
search_criteria = "656*.tif"
q = os.path.join(dirpath, search_criteria)

# print(q)

# Searching for our files using the glob function
dem_fps = glob.glob(q)
# dem_fps

# creating an empty list for mosaic components
src_files_to_mosaic = []

# opening the mosaic components (tif) in read mode with RasterIO
for fp in dem_fps:
    src = rasterio.open(fp)
    src_files_to_mosaic.append(src)

# src_files_to_mosaic

# Merging 
mosaic, out_trans = merge(src_files_to_mosaic)

# displaying the results
show(mosaic, cmap='gray')

# Copy the metadata
out_meta = src.meta.copy()

# Update the metadata
out_meta.update({"driver": "GTiff", "height": mosaic.shape[1],
                 "width": mosaic.shape[2], "transform": out_trans,
                 "crs": swer})

# Write the mosaic RasterIO to disk
with rasterio.open(out_fp, "w", **out_meta) as dest:
    dest.write(mosaic)


# =============================================================================
# Clipping

# Filepaths
fp = r'Mosaic.tif'
out_tif = r'Clip.tif'

# opening the raster
data = rasterio.open(fp)

# plotting the data
show(data, cmap='gray')

# Södertälje Coordinates:
# Decimal (lat,lon): (59.2141, 17.6291) 
# Northing easting (E,N): (650084.04312309, 6566851.5500514) 

# creating a boundary box with shapely around the Area of interest
# WGS84 coordinates

# x=0.01
# minx, miny = 17.9093-x, 59.1785-x
# maxx, maxy = 17.9092+x, 59.1785+x

x=300 #600*600m
minx, miny = 650084.04312309-x, 6566851.5500514-x
maxx, maxy = 650084.04312309+x, 6566851.5500514+x

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
# epsg_code = int(data.crs.data['init'][5:])
# print(epsg_code)

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
# =============================================================================
# Classification

driver = gdal.GetDriverByName('GTiff')
file = gdal.Open(r'Clip.tif')
band = file.GetRasterBand(1)
lista = band.ReadAsArray()

# reclassification
lista[np.where( lista <= 0 )] = 1
lista[np.where((0 < lista) & (lista <= 70)) ] = 2
lista[np.where((70 < lista) & (lista <= 105)) ] = 3
lista[np.where((105 < lista) & (lista <= 200)) ] = 4
lista[np.where( lista > 200 )] = 5

# create new file
file2 = driver.Create( 'Classified.tif', file.RasterXSize , file.RasterYSize , 1)
file2.GetRasterBand(1).WriteArray(lista)

# spatial ref system
proj = file.GetProjection()
georef = file.GetGeoTransform()
file2.SetProjection(proj)
file2.SetGeoTransform(georef)
file2.FlushCache()

# Filepaths
fp = r'Classified.tif'

# opening the raster
data = rasterio.open(fp)

# creating a colormap

colors = ['darkgreen', 'forestgreen', 'honeydew', 'lightgray']  # R -> G -> B
n_bins = [1, 2, 3, 4]  # Discretizes the interpolation into bins
cmap_name = 'my_list'
fig, axs = plt.subplots(2, 2, figsize=(6, 9))
for n_bin, ax in zip(n_bins, axs.ravel()):
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bin)

# plotting the data
fig, ax = plt.subplots(figsize=(10,8))
image_hidden = ax.imshow(data.read()[0], cmap=cm)
cbar= fig.colorbar(image_hidden, ax=ax, ticks=[2.375,3.15,3.875,4.65])
cbar.ax.yaxis.set_tick_params(width=0)
plt.gcf().axes[0].yaxis.get_major_formatter().set_scientific(False)
plt.gcf().axes[0].xaxis.get_major_formatter().set_scientific(False)
cbar.ax.set_yticklabels(['Heavy vegetation \nand water bodies', 
                         'Intermediate Vegetation\nand shadows', 'light to no Vegetation',
                         'Urban Areas'], fontsize=16, weight='bold')  # vertically oriented colorbar
show(data, ax=ax, cmap=cm)
plt.savefig('ClassifiedColored.jpg', dpi=300, bbox_inches='tight')


# =============================================================================
# =============================================================================
# Circular cutting 

# removing ticks and converting the image for pillow
fig, ax = plt.subplots(figsize=(10,10))
image_hidden = ax.imshow(data.read()[0], cmap=cm)
show(data, ax=ax, cmap=cm)
plt.xticks([])
plt.yticks([])
plt.savefig('Pillimge.jpg', dpi=300, bbox_inches='tight')


img=Image.open('Pillimge.jpg')
height,width = img.size
lum_img = Image.new('L', [height,width] , 0)  
draw = ImageDraw.Draw(lum_img)
draw.pieslice([(50,30), (height-30,width-50)], 0, 360, 
              fill = 255, outline = "white")
img_arr =np.array(img)
lum_img_arr =np.array(lum_img)
display(Image.fromarray(lum_img_arr))
final_img_arr = np.dstack((img_arr,lum_img_arr))
display(Image.fromarray(final_img_arr))
Image.fromarray(final_img_arr).save('CircleUrkle.png')
# =============================================================================
# =============================================================================
# Evaluation graph

#  Mosaiced.jpg, clipped.jpg, Pillimage.jpg, CircleUrkle.png

# saving the Mosaic and clipped images ####takes a long time, skip in analysis####

# before mosaics
fp = r'656_64_55.tif'
data = rasterio.open(fp)
fig, ax = plt.subplots(figsize=(10,10))
image_hidden = ax.imshow(data.read()[0], cmap='gray')
show(data, ax=ax, cmap='gray')
plt.gcf().axes[0].yaxis.get_major_formatter().set_scientific(False)
plt.gcf().axes[0].xaxis.get_major_formatter().set_scientific(False)
plt.savefig('Raw1.jpg', dpi=300, bbox_inches='tight')

fp = r'656_65_50.tif'
data = rasterio.open(fp)
fig, ax = plt.subplots(figsize=(10,10))
image_hidden = ax.imshow(data.read()[0], cmap='gray')
show(data, ax=ax, cmap='gray')
plt.gcf().axes[0].yaxis.get_major_formatter().set_scientific(False)
plt.gcf().axes[0].xaxis.get_major_formatter().set_scientific(False)
plt.savefig('Raw2.jpg', dpi=300, bbox_inches='tight')


# Mosaic
fp = r'Mosaic.tif'
data = rasterio.open(fp)
fig, ax = plt.subplots(figsize=(10,10))
image_hidden = ax.imshow(data.read()[0], cmap='gray')
show(data, ax=ax, cmap='gray')
plt.gcf().axes[0].yaxis.get_major_formatter().set_scientific(False)
plt.gcf().axes[0].xaxis.get_major_formatter().set_scientific(False)
plt.savefig('Mosaiced.jpg', dpi=300, bbox_inches='tight')

# Clip
fp = r'Clip.tif'
data = rasterio.open(fp)
fig, ax = plt.subplots(figsize=(10,10))
image_hidden = ax.imshow(data.read()[0], cmap='gray')
show(data, ax=ax, cmap='gray')
plt.gcf().axes[0].yaxis.get_major_formatter().set_scientific(False)
plt.gcf().axes[0].xaxis.get_major_formatter().set_scientific(False)
plt.savefig('Clipped.jpg', dpi=300, bbox_inches='tight')

# =============================================================================
