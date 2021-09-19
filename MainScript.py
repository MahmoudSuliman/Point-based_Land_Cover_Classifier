# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 15:24:33 2021

@author: MahmoudSuliman
"""

# =============================================================================
# 

import os
os.chdir(r'C:\Users\KIDDO\Downloads\SU Study\Traineeship\Urban Heat Island\Data_22T_23P\2.Södertälje\Historiska_ortofoton_1960_PAN_tif__493271c1-5839-4fc1-b9cb-081f4f83da6d_')
script_dir=r'C:\Users\KIDDO\Downloads\SU Study\Traineeship\Urban Heat Island\Data_22T_23P\2.Södertälje\Historiska_ortofoton_1960_PAN_tif__493271c1-5839-4fc1-b9cb-081f4f83da6d_'

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

# import georaster
# conda install -c conda-forge georaster

swer='+proj=utm +zone=33 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs'
# =============================================================================
# Raster Mosaic and preparation

############# add an if statement to this so that it does not automatically
############# mosaics (the name must be mosaic tho for the data visualization)

# File and folder paths
dirpath = r'C:\Users\KIDDO\Downloads\SU Study\Traineeship\Urban Heat Island\Data_22T_23P\2.Södertälje\Historiska_ortofoton_1960_PAN_tif__493271c1-5839-4fc1-b9cb-081f4f83da6d_'
out_fp = r'Mosaic.tif'

# extracting station's name and image type from parent folders

# station's name
zapath=Path(dirpath)
parentpath=zapath.parent.absolute() # gets parent path of directory
splitparpath=os.path.split(parentpath) # splits parent path
stname = splitparpath[1] # gets station name from split path

# image type
splitpath = os.path.split(dirpath) # splits current path
iminfo = splitpath[1] # full string of image info

# if statement to extract image info from full string
if iminfo.find('1960') != -1:
    print('Historical (1960)')
    imtype= 'Historical (1960)'
elif iminfo.find('PAN_0_5_m_latest') != -1:
    print('Latest')
    imtype='latest'
else:
    print('Nah!')

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

# creating a central point with shapely # lon, lat
point = Point(17.6291, 59.2141)

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

# creating a 200m radius buffer (product is 400m) 
loc_buffer = point_transformed.buffer(50)

# final transformation for the shapefile
buffer_wgs84 = transform(aeqd_to_swer, loc_buffer)


# Inserting the box into a geodataframe
geo = gpd.GeoDataFrame({'geometry': buffer_wgs84}, index=[0], crs=from_epsg(3006))

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
# and CRS (as Proj4 text). 

# NOTE: the crs code here manually reprojects it with
# sweref info but not the sweref is not identified by ArcMap.
out_meta.update({"driver": "GTiff", "height": out_img.shape[1],
                 "width": out_img.shape[2],"transform": out_transform,
                 "crs": swer})

# saving the clipped raster to disk
with rasterio.open(out_tif, "w", **out_meta) as dest:
    dest.write(out_img)

# # masking no data (background) values
# with rasterio.open(out_tif, "r+") as dataset:
#     dataset.nodata = 0
    
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

# cropping the color bar
cbim = Image.open('ClassifiedColored.jpg')
cbim_crop = cbim.crop((2100, 0, 3145, 1947))

fig, ax = plt.subplots(figsize=(10,10.5))
plt.imshow(cbim_crop)
plt.axis('off') # removes ticks and border (spines)
plt.savefig('CBar.jpg', dpi=300, bbox_inches='tight')

# Circular cutting 

# clipped
data = clipped
# removing ticks and converting the image for pillow
fig, ax = plt.subplots(figsize=(10,10))
image_hidden = ax.imshow(data.read()[0], cmap='gray')
show(data, ax=ax, cmap='gray')
plt.xticks([])
plt.yticks([])
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)
plt.savefig('GrayPill.jpg', dpi=300, bbox_inches='tight')


cimg=Image.open('GrayPill.jpg')
height,width = cimg.size
lum_img = Image.new('L', [height,width] , 0)  
draw = ImageDraw.Draw(lum_img)
draw.pieslice([(50,30), (height-30,width-50)], 0, 360, 
              fill = 255, outline = "white")
img_arr =np.array(cimg)
lum_img_arr =np.array(lum_img)
display(Image.fromarray(lum_img_arr))
final_img_arr = np.dstack((img_arr,lum_img_arr))
display(Image.fromarray(final_img_arr))
Image.fromarray(final_img_arr).save('CircleGray.png')

# convert image to remove background data (no data)
image = Image.open('CircleGray.png').convert('RGBA')
new_image = Image.new("RGBA", image.size, "WHITE") # Create a white rgba background
new_image.paste(image, (0, 0), image)              # Paste the image on the background. Go to the links given below for details.
new_image.convert('RGB').save('CircleGray.jpg')  # Save as JPEG

# classified
# removing ticks and converting the image for pillow
fig, ax = plt.subplots(figsize=(10,10))
image_hidden = ax.imshow(data.read()[0], cmap=cm)
show(data, ax=ax, cmap=cm)
plt.xticks([])
plt.yticks([])
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)
plt.savefig('Pillimge.jpg', dpi=300, bbox_inches='tight')


cimg=Image.open('Pillimge.jpg')
height,width = cimg.size
lum_img = Image.new('L', [height,width] , 0)  
draw = ImageDraw.Draw(lum_img)
draw.pieslice([(50,30), (height-30,width-50)], 0, 360, 
              fill = 255, outline = "white")
img_arr =np.array(cimg)
lum_img_arr =np.array(lum_img)
display(Image.fromarray(lum_img_arr))
final_img_arr = np.dstack((img_arr,lum_img_arr))
display(Image.fromarray(final_img_arr))
Image.fromarray(final_img_arr).save('CircleUrkle.png')

# convert image to remove background data (no data)
image = Image.open('CircleUrkle.png').convert('RGBA')
new_image = Image.new("RGBA", image.size, "WHITE") # Create a white rgba background
new_image.paste(image, (0, 0), image)              # Paste the image on the background. Go to the links given below for details.
new_image.convert('RGB').save('CircleUrkle.jpg')  # Save as JPEG

# plotting full cbar image

im1 = Image.open('CircleUrkle.jpg')
im2 = Image.open('CBar.jpg')

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

get_concat_h(im1, im2).save('WorkingCBar.jpg')

plt.imshow(Image.open('WorkingCBar.jpg'))
plt.axis('off') # removes ticks and border (spines)
plt.savefig('fullCircleCBar.jpg', dpi=300, bbox_inches='tight')

# =============================================================================
# =============================================================================
# Evaluation graph

# create evaluation directory
eva_dir = os.path.join(script_dir, 'Evaluate/')

if not os.path.isdir(eva_dir):
    os.makedirs(eva_dir)

#  Mosaiced.jpg, clipped.jpg, Pillimage.jpg, CircleUrkle.png

# saving the Mosaic and clipped images ####takes a long time, skip in analysis####

# before mosaics #### make it in a loop based on name ######
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
plt.savefig('Clipped.png', dpi=300, bbox_inches='tight')

# Evaluation figure plotting
fig, axs = plt.subplots(1,2, figsize=(10,8))
fig.suptitle('Insert the name and number of the station', y=0.85, fontsize=20)
axs[0].imshow(img.imread('CircleGray.jpg'))
axs[0].axis('off') # removes ticks and border (spines)
axs[0].set_title('Clipped')
axs[1].imshow(img.imread('CircleUrkle.jpg'))
axs[1].axis('off') # removes ticks and border (spines)
axs[1].set_title('Classified')
fig.tight_layout(pad=4.0)
plt.savefig(eva_dir+'Evaluate_'+'add stations name'+'.jpg', dpi=300, bbox_inches='tight')

# second clipping
# Filepaths
fp = r'Mosaic.tif'
out_tif = r'Clip400.tif'

# opening the raster
data = rasterio.open(fp)

# Södertälje Coordinates:
# Decimal (lat,lon): (59.2141, 17.6291) 
# Northing easting (E,N): (650084.04312309, 6566851.5500514) 

# creating a central point with shapely # lon, lat
point = Point(17.6291, 59.2141)

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

# creating a 200m radius buffer (product is 400m) 
loc_buffer = point_transformed.buffer(200)

# final transformation for the shapefile
buffer_wgs84 = transform(aeqd_to_swer, loc_buffer)


# Inserting the box into a geodataframe
geo = gpd.GeoDataFrame({'geometry': buffer_wgs84}, index=[0], crs=from_epsg(3006))

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
# and CRS (as Proj4 text). 

# NOTE: the crs code here manually reprojects it with
# sweref info but not the sweref is not identified by ArcMap.
out_meta.update({"driver": "GTiff", "height": out_img.shape[1],
                 "width": out_img.shape[2],"transform": out_transform,
                 "crs": swer})

# saving the clipped raster to disk
with rasterio.open(out_tif, "w", **out_meta) as dest:
    dest.write(out_img)

# # masking no data (background) values
# with rasterio.open(out_tif, "r+") as dataset:
#     dataset.nodata = 0
    
# plotting our new clipped raster
clipped = rasterio.open(out_tif)
show(clipped, cmap='gray')

# Clip
fp = r'Clip400.tif'
data = rasterio.open(fp)
fig, ax = plt.subplots(figsize=(10,10))
image_hidden = ax.imshow(data.read()[0], cmap='gray')
show(data, ax=ax, cmap='gray')
plt.gcf().axes[0].yaxis.get_major_formatter().set_scientific(False)
plt.gcf().axes[0].xaxis.get_major_formatter().set_scientific(False)
plt.savefig('Clipped400.png', dpi=300, bbox_inches='tight')

# Evaluation figure plotting
fig, axs = plt.subplots(1,2, figsize=(10,8))
fig.suptitle('Insert the name and number of the station', y=0.85, fontsize=20)
axs[0].imshow(img.imread('Clipped400.png'))
axs[0].axis('off') # removes ticks and border (spines)
axs[0].set_title('Clipped 400m')
axs[1].imshow(img.imread('Clipped.png'))
axs[1].axis('off') # removes ticks and border (spines)
axs[1].set_title('Clipped 100m')
fig.tight_layout(pad=4.0)
plt.savefig(eva_dir+'CEvaluate_'+'add stations name'+'.jpg', dpi=300, bbox_inches='tight')

# =============================================================================
# =============================================================================
# Extracting values


im = Image.open('Classified.tif') #.convert('RGB')
by_color = defaultdict(int)
for pixel in im.getdata():
    by_color[pixel] += 1 # number of pixels with Tex (0,0,0) RGB values


# =============================================================================

