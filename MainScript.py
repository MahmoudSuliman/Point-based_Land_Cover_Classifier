# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 15:24:33 2021

@author: MahmoudSuliman
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

swer='+proj=utm +zone=33 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs'

# =============================================================================
# Preparation

# File and folder paths
# Changing the work directory
# workdir=r'C:\Users\KIDDO\Downloads\SU Study\Traineeship\Urban Heat Island\Data_22T_23P\17.Höljes\Ortofoto_PAN_0_5_m_latest_tif__4285278c-bc76-4124-8214-168a49070a62_'

os.chdir(workdir)

# Script directories
script_dir = dirpath = os.getcwd()
out_fp = r'Mosaic.tif'

# creating results, classification and evaluation directory
res_dir = os.path.join(script_dir, 'Results/')
if not os.path.isdir(res_dir):
    os.makedirs(res_dir)

for i in ['00','01','02','03','04','05','06','07','08','09','10']:
          # ,'a1','a2','a3','a4','a5','a6','a7','a8']:
    cla_dir = os.path.join(script_dir, 'Results/'+i)
    if not os.path.isdir(cla_dir):
        os.makedirs(cla_dir)

for i in ['00','01','02','03','04','05','06','07','08','09','10']:
          # ,'a1','a2','a3','a4','a5','a6','a7','a8']:
    eva_dir = os.path.join(script_dir, 'Results/'+i+'/Evaluate')
    if not os.path.isdir(eva_dir):
        os.makedirs(eva_dir)

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
    print('Historical')
    imtype= 'Historical'
elif iminfo.find('PAN_0_5_m_latest') != -1:
    print('Latest')
    imtype='latest'
else:
    print('Nah!')

# =============================================================================
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
# =============================================================================
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

# clipped
data = clipped

# removing ticks and converting the image for pillow
fig, ax = plt.subplots(figsize=(10,10))
image_hidden = ax.imshow(data.read()[0], cmap='gray')
show(data, ax=ax, cmap='gray')
plt.xticks([])
plt.yticks([])
plt.savefig(r'GrayPill.jpg', dpi=300, bbox_inches='tight')

# Cropping using pillow
cimg=Image.open(r'GrayPill.jpg')
height,width = cimg.size
lum_img = Image.new('L', [height,width] , 0)  
draw = ImageDraw.Draw(lum_img)
draw.pieslice([(50,30), (height-30,width-50)], 0, 360, 
              fill = 255, outline = "white")
img_arr =np.array(cimg)
lum_img_arr =np.array(lum_img)
final_img_arr = np.dstack((img_arr,lum_img_arr))
display(Image.fromarray(final_img_arr))
Image.fromarray(final_img_arr).save('CircleGray.png')

# Convert image to remove background data (no data)
image = Image.open('CircleGray.png').convert('RGBA')
new_image = Image.new("RGBA", image.size, "WHITE") # Create a white rgba background
new_image.paste(image, (0, 0), image)              # Paste the image on the background. Go to the links given below for details.
new_image.convert('RGB').save('CircleGray.jpg')  # Save as JPEG



# =============================================================================
# =============================================================================
# Classification

# GDAL file and band definitions
driver = gdal.GetDriverByName('GTiff')
file = gdal.Open(r'Clip.tif')
band = file.GetRasterBand(1)
lista = band.ReadAsArray()

# -----------------------------------------------------------------------------
# Defining classification scenarios
clcases= {}

# site specific cases
clcases['''01.södertälje old 
        (bright streets and houses, grey intermediate veg)'''
        ] = '''lista[np.where( lista <= 0 )] = 1 # Nan
lista[np.where((0 < lista) & (lista <= 70)) ] = 2 # H.veg
lista[np.where((70 < lista) & (lista <= 105)) ] = 3 # I.veg
lista[np.where((105 < lista) & (lista <= 220)) ] = 4 # L.veg
lista[np.where( lista > 220 )] = 5 # urban'''

clcases['''02.södertälje new 
        (dark rooftops, not so dark vegetation and greyish streets)'''
        ] = '''lista[np.where( lista <= 0 )] = 1 # Nan
lista[np.where((0 < lista) & (lista <= 70)) ] = 2 # H.veg
lista[np.where((90 < lista) & (lista <= 130)) ] = 3 # I.veg
lista[np.where((130 < lista) & (lista <= 175)) ] = 4 # L.veg
lista[np.where((lista > 175))] = 5
lista[np.where((70 < lista) & (lista <= 90)) ] = 5'''

clcases['03.tullinge old (bright light veg)'
        ]= '''lista[np.where( lista <= 0 )] = 4 # Nan
lista[np.where((205 < lista) & (lista <= 206)) ] = 2 # H.veg
lista[np.where((206 < lista) & (lista <= 207)) ] = 2 # I.veg
lista[np.where((207 < lista) & (lista <= 209)) ] = 3 # I.veg
lista[np.where((3 < lista) & (lista <= 4)) ] = 4 # I.veg'
lista[np.where((4 < lista) & (lista <= 205)) ] = 4 # L.veg
lista[np.where( lista > 209 )] = 5 # urban'''

clcases['''04.Tulinge new 
        (semi-normal i.veg kinda bright urban)'''
        ]= '''lista[np.where( lista <= 0 )] = 1 # Nan
lista[np.where((0 < lista) & (lista <= 70)) ] = 2 # H.veg
lista[np.where((70 < lista) & (lista <= 90)) ] = 3 # I.veg
lista[np.where((90 < lista) & (lista <= 170)) ] = 4 # L.veg
lista[np.where( lista > 170 )] = 5 # urban'''

clcases['''05.stockholm old 
        (dark rooftops, semi-bright streets that mix with lots of lveg)'''
        ] = '''lista[np.where( lista <= 0 )] = 1 # Nan
lista[np.where((0 < lista) & (lista <= 30)) ] = 2 # H.veg
lista[np.where((90 < lista) & (lista <= 100)) ] = 3 # I.veg
lista[np.where((100 < lista) & (lista <= 190)) ] = 4 # L.veg
lista[np.where((lista > 190))] = 5
lista[np.where((30 < lista) & (lista <= 90)) ] = 5'''

clcases['''06.stockholm new 
         (lots of trees, greyish rooftops)'''
        ] = '''lista[np.where( lista <= 0 )] = 1 # Nan
lista[np.where((0 < lista) & (lista <= 70)) ] = 2 # H.veg
lista[np.where((70 < lista) & (lista <= 105)) ] = 3 # I.veg
lista[np.where((105 < lista) & (lista <= 190)) ] = 4 # L.veg
lista[np.where( lista > 190 )] = 5 # urban'''

clcases['''07.skövde old 
        (dark grey rooftops, slightly lighter streets)'''
        ] = '''lista[np.where( lista <= 0 )] = 1 # Nan
lista[np.where((0 < lista) & (lista <= 70)) ] = 2 # H.veg
lista[np.where((70 < lista) & (lista <= 105)) ] = 3 # I.veg
lista[np.where((105 < lista) & (lista <= 110)) ] = 4 # L.veg
lista[np.where( lista > 110 )] = 5 # urban'''

clcases['''08.skövde new 
        (dark grey rooftops, lighter streets)'''
        ] = '''lista[np.where( lista <= 0 )] = 1 # Nan
lista[np.where((0 < lista) & (lista <= 70)) ] = 2 # H.veg
lista[np.where((70 < lista) & (lista <= 120)) ] = 3 # I.veg
lista[np.where((120 < lista) & (lista <= 130)) ] = 4 # L.veg
lista[np.where( lista > 130 )] = 5 # urban'''

clcases['''09.oskarshamn new 
        (dark urban area, greyish streets)'''
        ] = '''lista[np.where( lista <= 0 )] = 1 # Nan
lista[np.where((0 < lista) & (lista <= 30)) ] = 5 # H.veg
lista[np.where((30 < lista) & (lista <= 70)) ] = 2 # H.veg
lista[np.where((70 < lista) & (lista <= 90)) ] = 3 # I.veg
lista[np.where((90 < lista) & (lista <= 160)) ] = 4 # L.veg
lista[np.where( lista > 160 )] = 5 # urban'''

clcases['''10.holjes new 
        (semi dark i.veg, all veg throughout)'''
        ] = '''lista[np.where( lista <= 0 )] = 1 # Nan 
lista[np.where((0 < lista) & (lista <= 80)) ] = 2 # H.veg
lista[np.where((80 < lista) & (lista <= 160)) ] = 3 # I.veg
lista[np.where((160 < lista) & (lista <= 170)) ] = 4 # L.veg
lista[np.where( lista > 170 )] = 5 # urban'''

# clcombs=np.array([0,0,0]).reshape((1,3))
# for i in[70]:
#     for j in [90,130]:
#         for k in [140,180,200,240]:
#             clcombs2=np.array([i,j,k]).reshape((1,3))
#             clcombs=np.concatenate((clcombs,clcombs2), axis=0)

# clcombs=np.delete(clcombs,(0,0),0)

# clextrakeys = ['a1','a2','a3','a4','a5','a6','a7','a8']

# for i in range(0,len(clcombs)):
#     clcases[clextrakeys[i]] = '''lista[np.where( lista <= 0 )] = 1 # Nan 
# lista[np.where((0 < lista) & (lista <= '''+str(clcombs[i,0])+''')) ] = 2 # H.veg
# lista[np.where(('''+str(clcombs[i,0])+''' < lista) & (lista <= '''+str(clcombs[i,1])+''')) ] = 3 # I.veg
# lista[np.where(('''+str(clcombs[i,1])+''' < lista) & (lista <= '''+str(clcombs[i,2])+''')) ] = 4 # L.veg
# lista[np.where( lista > '''+str(clcombs[i,2])+''' )] = 5 # urban'''


clnum=[]
for keys, values in clcases.items():
    clnum.append(keys[:2])

for key, value in clcases.items():
    for i in range(0,len(clnum)):
        if key[:2]==clnum[i]:
            lista = band.ReadAsArray()
            exec(value)
            
# -----------------------------------------------------------------------------
# creating new file
            file2 = driver.Create( r'Results/'+clnum[i]+'/Classified.tif', file.RasterXSize , file.RasterYSize , 1)
            file2.GetRasterBand(1).WriteArray(lista)
            
# Spatial referencing system obs
            proj = file.GetProjection()
            georef = file.GetGeoTransform()
            file2.SetProjection(proj)
            file2.SetGeoTransform(georef)
            file2.FlushCache()
            
# Adding  and opening output's filepath
            fp = r'Results/'+clnum[i]+'/Classified.tif'
            data = rasterio.open(fp)
            
# creating a custom colormap
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
            plt.savefig(r'Results/'+clnum[i]+'/ClassifiedColored.jpg', dpi=300, bbox_inches='tight')
            
            # cropping the color bar
            cbim = Image.open(r'Results/'+clnum[i]+'/ClassifiedColored.jpg')
            cbim_crop = cbim.crop((2100, 0, 3145, 1947))
            
            # Plotting and saving the cropped colorbar
            fig, ax = plt.subplots(figsize=(10,10.5))
            plt.imshow(cbim_crop)
            plt.axis('off') # removes ticks and border (spines)
            plt.savefig(r'Results/'+clnum[i]+'/CBar.jpg', dpi=300, bbox_inches='tight')
            
            # =============================================================================
            # =============================================================================
            # Circular cutting 
            
            # classified
            fp = r'Results/'+clnum[i]+'/Classified.tif'
            data = rasterio.open(fp)
            
            # removing ticks and converting the image for pillow
            fig, ax = plt.subplots(figsize=(10,10))
            image_hidden = ax.imshow(data.read()[0], cmap=cm)
            show(data, ax=ax, cmap=cm)
            plt.xticks([])
            plt.yticks([])
            plt.savefig(r'Results/'+clnum[i]+'/Pillimge.jpg', dpi=300, bbox_inches='tight')
            
            # Cropping using pillow
            cimg=Image.open(r'Results/'+clnum[i]+'/Pillimge.jpg')
            height,width = cimg.size
            lum_img = Image.new('L', [height,width] , 0)  
            draw = ImageDraw.Draw(lum_img)
            draw.pieslice([(50,30), (height-30,width-50)], 0, 360, 
                          fill = 255, outline = "white")
            img_arr =np.array(cimg)
            lum_img_arr =np.array(lum_img)
            final_img_arr = np.dstack((img_arr,lum_img_arr))
            display(Image.fromarray(final_img_arr))
            Image.fromarray(final_img_arr).save(r'Results/'+clnum[i]+'/CircleUrkle.png')
            
            # convert image to remove background data (no data)
            image = Image.open(r'Results/'+clnum[i]+'/CircleUrkle.png').convert('RGBA')
            new_image = Image.new("RGBA", image.size, "WHITE") # Create a white rgba background
            new_image.paste(image, (0, 0), image)              # Paste the image on the background. Go to the links given below for details.
            new_image.convert('RGB').save(r'Results/'+clnum[i]+'/CircleUrkle.jpg')  # Save as JPEG
            
            # plotting full cbar image
            
            im1 = Image.open(r'Results/'+clnum[i]+'/CircleUrkle.jpg')
            im2 = Image.open(r'Results/'+clnum[i]+'/CBar.jpg')
            
            def get_concat_h(im1, im2):
                dst = Image.new('RGB', (im1.width + im2.width, im1.height))
                dst.paste(im1, (0, 0))
                dst.paste(im2, (im1.width, 0))
                return dst
            
            get_concat_h(im1, im2).save(r'Results/'+clnum[i]+'/WorkingCBar.jpg')
            
            plt.imshow(Image.open(r'Results/'+clnum[i]+'/WorkingCBar.jpg'))
            plt.axis('off') # removes ticks and border (spines)
            plt.savefig(r'Results/'+clnum[i]+'/fullCircleCBar.jpg', dpi=300, bbox_inches='tight')
            
            # =============================================================================
            # =============================================================================
            # Evaluation graphs
            
            
            # -----------------------------------------------------------------------------
            # Classification Evaluation
            
            # Raw products
            for h in range(0,len(tiff_fps)):
                fp = tiff_fps[h]
                data = rasterio.open(fp)
                fig, ax = plt.subplots(figsize=(10,10))
                image_hidden = ax.imshow(data.read()[0], cmap='gray')
                show(data, ax=ax, cmap='gray')
                plt.gcf().axes[0].yaxis.get_major_formatter().set_scientific(False)
                plt.gcf().axes[0].xaxis.get_major_formatter().set_scientific(False)
                plt.savefig('Raw'+str(h+1)+'.jpg', dpi=300, bbox_inches='tight')
            
            # Mosaic
            fp = r'Mosaic.tif'
            data = rasterio.open(fp)
            fig, ax = plt.subplots(figsize=(10,10))
            image_hidden = ax.imshow(data.read()[0], cmap='gray')
            show(data, ax=ax, cmap='gray')
            plt.gcf().axes[0].yaxis.get_major_formatter().set_scientific(False)
            plt.gcf().axes[0].xaxis.get_major_formatter().set_scientific(False)
            plt.savefig(r'Results/'+clnum[i]+'/Mosaiced.jpg', dpi=300, bbox_inches='tight')
            
            # Clip
            fp = r'Clip.tif'
            data = rasterio.open(fp)
            fig, ax = plt.subplots(figsize=(10,10))
            image_hidden = ax.imshow(data.read()[0], cmap='gray')
            show(data, ax=ax, cmap='gray')
            plt.gcf().axes[0].yaxis.get_major_formatter().set_scientific(False)
            plt.gcf().axes[0].xaxis.get_major_formatter().set_scientific(False)
            plt.savefig(r'Results/'+clnum[i]+'/Clipped.png', dpi=300, bbox_inches='tight')
            
            # Evaluation figure plotting
            fig, axs = plt.subplots(1,2, figsize=(10,8))
            fig.suptitle(stname+' ('+imtype+')', y=0.85, fontsize=20)
            axs[0].imshow(img.imread('CircleGray.jpg'))
            axs[0].axis('off') # removes ticks and border (spines)
            axs[0].set_title('Clipped')
            axs[1].imshow(img.imread(r'Results/'+clnum[i]+'/CircleUrkle.jpg'))
            axs[1].axis('off') # removes ticks and border (spines)
            axs[1].set_title('Classified')
            fig.tight_layout(pad=4.0)
            plt.savefig(r'Results/'+clnum[i]+'/Evaluate/Eva_'+stname+' ('+imtype+')'+'.jpg', dpi=300, bbox_inches='tight')
            
            plt.close('all')
            # -----------------------------------------------------------------------------
            
            # =============================================================================
            # =============================================================================
            # Extracting values
            
            im = Image.open(r'Results/'+clnum[i]+'/Classified.tif') #.convert('RGB')
            by_color = defaultdict(int)
            for pixel in im.getdata():
                by_color[pixel] += 1 # number of pixels with 2(h.veg), 3(l.veg), 4(no.veg), 5(urban)
            
            pixres = pd.DataFrame(data=(by_color.items())).transpose()
            pixres.to_csv(r'Results/'+clnum[i]+'/Evaluate/'+stname+'_'+imtype+'_'+clnum[i]+'_pixres.csv')

# Clipping Evaluation

fp = r'Mosaic.tif'; out_tif = r'Clip400.tif'
data = rasterio.open(fp); point = Point(lon, lat)
local_azimuthal_projection = f"+proj=aeqd +R=6371000 +units=m +lat_0={point.y} +lon_0={point.x}"
wgs84_to_aeqd = partial(pyproj.transform,
                        pyproj.Proj('+proj=longlat +datum=WGS84 +no_defs'),
                        pyproj.Proj(local_azimuthal_projection),)
aeqd_to_wgs84 = partial(pyproj.transform,
                        pyproj.Proj(local_azimuthal_projection),
                        pyproj.Proj('+proj=longlat +datum=WGS84 +no_defs'),)
aeqd_to_swer = partial(pyproj.transform,
                       pyproj.Proj(local_azimuthal_projection),
                       pyproj.Proj(swer),)
point_transformed = transform(wgs84_to_aeqd, point)

loc_buffer = point_transformed.buffer(400)
buffer_wgs84 = transform(aeqd_to_swer, loc_buffer)
geo = gpd.GeoDataFrame({'geometry': buffer_wgs84}, index=[0], crs=from_epsg(3006))
coords = getFeatures(geo)
out_img, out_transform = mask(dataset=data, shapes=coords, crop=True)
out_meta = data.meta.copy()
print(out_meta)
out_meta.update({"driver": "GTiff", "height": out_img.shape[1],
                 "width": out_img.shape[2],"transform": out_transform,
                 "crs": swer})
with rasterio.open(out_tif, "w", **out_meta) as dest:
    dest.write(out_img)
clipped = rasterio.open(out_tif)
show(clipped, cmap='gray')

# classified
fp = r'Clip400.tif'
data = rasterio.open(fp)

# removing ticks and converting the image for pillow
fig, ax = plt.subplots(figsize=(10,10))
image_hidden = ax.imshow(data.read()[0], cmap='gray')
show(data, ax=ax, cmap='gray')
plt.xticks([])
plt.yticks([])
plt.savefig(r'Clip400.jpg', dpi=300, bbox_inches='tight')

# Cropping using pillow
cimg=Image.open(r'Clip400.jpg')
height,width = cimg.size
lum_img = Image.new('L', [height,width] , 0)  
draw = ImageDraw.Draw(lum_img)
draw.pieslice([(50,30), (height-30,width-50)], 0, 360, 
              fill = 255, outline = "white")
img_arr =np.array(cimg)
lum_img_arr =np.array(lum_img)
final_img_arr = np.dstack((img_arr,lum_img_arr))
display(Image.fromarray(final_img_arr))
Image.fromarray(final_img_arr).save(r'CircleClip400.png')

# convert image to remove background data (no data)
image = Image.open(r'CircleClip400.png').convert('RGBA')
new_image = Image.new("RGBA", image.size, "WHITE") # Create a white rgba background
new_image.paste(image, (0, 0), image)              # Paste the image on the background. Go to the links given below for details.
new_image.convert('RGB').save(r'CircleClip400.jpg')  # Save as JPEG

# Evaluation figure plotting
fig, axs = plt.subplots(1,2, figsize=(10,8))
fig.suptitle(stname+' ('+imtype+')', y=0.85, fontsize=20)
axs[0].imshow(img.imread(r'CircleClip400.jpg'))
axs[0].axis('off') # removes ticks and border (spines)
axs[0].set_title('Clipped 400m radius')
axs[1].imshow(img.imread('CircleGray.jpg'))
axs[1].axis('off') # removes ticks and border (spines)
axs[1].set_title('Clipped 100m radius')
fig.tight_layout(pad=4.0)
plt.savefig(r'Results/'+clnum[i]+'/Evaluate/CEva_'+stname+' ('+imtype+')'+'.jpg', dpi=300, bbox_inches='tight')

# =============================================================================

