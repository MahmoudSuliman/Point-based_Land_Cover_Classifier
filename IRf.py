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

# GDAL file and band definitions
driver = gdal.GetDriverByName('GTiff')
file = gdal.Open(r'Clip.tif')

# loading the different bands

# IR
bandir = file.GetRasterBand(2)
b1 = listair = bandir.ReadAsArray()

# red
bandred = file.GetRasterBand(3)
b2 = listared = bandred.ReadAsArray()

# green
bandgreen = file.GetRasterBand(1)
b3 = listagreen = bandgreen.ReadAsArray()

# plotting the bands
plt.imshow(listair)
plt.imshow(listared)
plt.imshow(listagreen)

# calculating ndvi
ndvi = es.normalized_diff(b1, b2)

# test plot
ep.plot_bands(ndvi, cmap="RdYlGn", cols=1, vmin=-1, vmax=1)

# binning
ndvi_class_bins = [-np.inf, 0.14, 0.27, 0.36, np.inf] # defining bins
ndvi_landsat_class = np.digitize(ndvi, ndvi_class_bins) # binning the array

# Apply the nodata mask to the newly classified NDVI data
ndvi_landsat_class = np.ma.masked_where(np.ma.getmask(ndvi), ndvi_landsat_class)

# and check the number of unique values in our now classified dataset
np.unique(ndvi_landsat_class)

# ep plotting -----------------------------------------------------------------
 
# Define color map
nbr_colors = ["lightgray", "honeydew", "forestgreen", "darkgreen"]
nbr_cmap = ListedColormap(nbr_colors)

# Define class names
ndvi_cat_names = ["Urban Areas","light to no vegetation","intermediate Vegetation","Heavy Vegetation"]

# Get list of classes
classes = np.unique(ndvi_landsat_class)
classes = classes.tolist()

# The mask returns a value of none in the classes. remove that
classes = classes[0:4]

# Plotting
fig, ax = plt.subplots(figsize=(12, 12))
im = ax.imshow(ndvi_landsat_class, cmap=nbr_cmap)
ep.draw_legend(im_ax=im, classes=classes, titles=ndvi_cat_names)
ax.set_title("Landsat 8 - Normalized Difference Vegetation Index (NDVI) Classes", fontsize=14)
ax.set_axis_off()
plt.tight_layout()

# old classification
# band calculation based on ndvi eq
lista=(b1-b2)/(b1+b2)

ndvi_class_bins = [-np.inf, 0.14, 0.27, 0.36, np.inf] # defining bins
lista = np.digitize(ndvi, ndvi_class_bins) # binning the array

plt.imshow(lista)

lista[np.where( lista <= 0.14 )] = 4 # Nan
lista[np.where((0.14 < lista) & (lista <= 0.27)) ] = 3 # I.veg
lista[np.where((0.27 < lista) & (lista <= 0.36)) ] = 2 # L.veg
lista[np.where( lista > 0.36 )] = 1 # urban'''

plt.imshow(ndvi_landsat_class)

# creating new file
file2 = driver.Create( 'Classified4.tif', file.RasterXSize , file.RasterYSize , 1)
file2.GetRasterBand(1).WriteArray(lista)

# Spatial referencing system obs
proj = file.GetProjection()
georef = file.GetGeoTransform()
file2.SetProjection(proj)
file2.SetGeoTransform(georef)
file2.FlushCache()

# Adding  and opening output's filepath
fp = r'Classified4.tif'
data = rasterio.open(fp)

# creating a custom colormap
colors = ['lightgray','honeydew','forestgreen','darkgreen']  # R -> G -> B
n_bins = [1, 2, 3, 4]  # Discretizes the interpolation into bins
cmap_name = 'my_list'
fig, axs = plt.subplots(2, 2, figsize=(6, 9))
for n_bin, ax in zip(n_bins, axs.ravel()):
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bin)

# plotting the data
fig, ax = plt.subplots(figsize=(10,8))
image_hidden = ax.imshow(data.read()[0], cmap=cm)
cbar= fig.colorbar(image_hidden, ax=ax, ticks=[1.475,2.45,3.6,4.55])
cbar.ax.yaxis.set_tick_params(width=0)
plt.gcf().axes[0].yaxis.get_major_formatter().set_scientific(False)
plt.gcf().axes[0].xaxis.get_major_formatter().set_scientific(False)
cbar.ax.set_yticklabels(['Urban Areas','light to no Vegetation','Intermediate Vegetation\nand shadows',
                         'Heavy vegetation \nand water bodies'], fontsize=16, weight='bold')  # vertically oriented colorbar
show(data, ax=ax, cmap=cm)
plt.savefig('ClassifiedColored.jpg', dpi=300, bbox_inches='tight')

# cropping the color bar
cbim = Image.open('ClassifiedColored.jpg')
cbim_crop = cbim.crop((2100, 0, 3145, 1947))

# Plotting and saving the cropped colorbar
fig, ax = plt.subplots(figsize=(10,10.5))
plt.imshow(cbim_crop)
plt.axis('off') # removes ticks and border (spines)
plt.savefig('CBar.jpg', dpi=300, bbox_inches='tight')
# =============================================================================
# =============================================================================
# Circular cutting 
# classified
fp = r'Classified4.tif'
data = rasterio.open(fp)

# removing ticks and converting the image for pillow
fig, ax = plt.subplots(figsize=(10,10))
image_hidden = ax.imshow(data.read()[0], cmap=cm)
show(data, ax=ax, cmap=cm)
plt.xticks([])
plt.yticks([])
plt.savefig('Pillimge.jpg', dpi=300, bbox_inches='tight')

# Cropping using pillow
cimg=Image.open('Pillimge.jpg')
height,width = cimg.size
lum_img = Image.new('L', [height,width] , 0)  
draw = ImageDraw.Draw(lum_img)
draw.pieslice([(50,30), (height-30,width-50)], 0, 360, 
              fill = 255, outline = "white")
img_arr =np.array(cimg)
lum_img_arr =np.array(lum_img)
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
