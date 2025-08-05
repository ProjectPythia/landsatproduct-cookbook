# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Landsat SST utility functions
# Author: Tasha Snow

# Note: After making changes to the `.ipynb` version of SSTutils, `File > Save notebook as`, 
# change extension to `.py`, make executable in terminal with `chmod +x SSTutils.py`, and rerun `imports` cell.

# +
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from skimage import exposure
from skimage.io import imsave, imread
from osgeo import ogr
import pystac_client
from pyproj import Transformer
from datetime import date, timedelta, datetime
from dateutil.relativedelta import relativedelta
import geopandas as gpd
import pandas as pd
import geoviews as gv
import hvplot.pandas
import intake
import xarray as xr
import numpy as np
from numpy.random import default_rng
import intake
from pyproj import Proj, transform
from osgeo import gdal
from sklearn.neighbors import BallTree
import earthaccess
import gzip

# for progress bar
from ipywidgets import IntProgress
from IPython.display import display
from ipywidgets import interact, Dropdown
import time
from tqdm.notebook import trange, tqdm

import boto3
import rasterio as rio
from rasterio.features import rasterize
from rasterio.session import AWSSession
import dask
import os
import rioxarray
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.warp import Resampling as resample
import cartopy.crs as ccrs
import cartopy
from pykrige.ok import OrdinaryKriging
from sklearn.linear_model import LinearRegression, RANSACRegressor
from scipy.odr import Model, RealData, ODR
import scipy.odr as odr
import scipy
import statsmodels.formula.api as smf
from shapely.geometry.polygon import Polygon, Point
import pygmt
import gc
import pytz
import pyproj
import math
from pathlib import Path
from matplotlib.patches import Polygon as Pgon
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler

# + editable=true slideshow={"slide_type": ""}
# Functions to search and open Lansat scenes
'''
Functions to search, open, and analyze Landsat scenes.
Search_stac finds the Landsat scene based on user parameters, 
plot_search plots the locations of the landsat scenes from the search,
landsat_to_xarray takes one of those scenes and puts all bands into an xarray,
and create_masks produces cloud/ice/water masks for the scene. Subset_img 
subsets a landsat scene with coordinates that have been reprojected from lat/lon
and may be flipped in which is larger in the pair. Lsat_reproj can be used to reproject
while ensuring x and y pairs don't get flipped (common converting between espg 3031 and wgs84.
'''

def landsat_to_xarray(sceneid, catalog, bandNames=None):
    """
    Loads selected Landsat bands (and QA layers for later cloud masking) from an 
    AWS S3 bucket (via the STAC item's alternate href) into an xarray Dataset.

    Parameters
    ----------
    sceneid : intake.STAC item
        A single STAC item pointing to Landsat assets.
    catalog : ?????
    bandNames : list of str, optional
        Names of bands to load (e.g., ['red', 'swir16']). If None, all non-thermal
        bands are included by default.

    Returns
    -------
    xr.DataArray
        A 3D xarray DataArray (dimensions: band, y, x) with a scalar coordinate
        for the observation time. The bands will include 'qa_pixel', 'qa_radsat',
        and 'VZA' in addition to any requested reflectance/thermal bands.
    
    Notes
    -------
    If multiple scenes are merged later on, xarray will fill non-overlapping areas with NaNs.
    
    """

    # Retrieve the STAC item from the catalog by its ID
    item = catalog[sceneid.id]

    bands = []
    band_names = []

    if bandNames is None:
        # Get band names
        for k in item.keys():
            M = getattr(item, k).metadata
            if 'eo:bands' in M:
                resol = M['eo:bands'][0]['gsd']
                if resol >= 30: # thermal bands are up sampled from 100 to 30
                    band_names.append(k)
    else:
        band_names = bandNames

    # Add QA bands for creating cloud mask later
    if 'qa_pixel' not in band_names:
        band_names.append('qa_pixel')
    
    band_names.append('VZA')
    band_names.append('qa_radsat')

    # Construct xarray for scene by concatenating all desired bands (including QA)
    for band_name in band_names:
        asset = sceneid.assets[band_name]
        href = asset.extra_fields['alternate']['s3']['href']
        band = xr.open_dataset(href, engine='rasterio', chunks=dict(band=1, x=512, y=512))
        band['band'] = [band_name]
        bands.append(band)
    ls_scene = xr.concat(bands, dim='band')
    ls_scene.coords['id'] = sceneid.id
    ls_scene.coords['time'] = item.metadata['datetime'].strftime('%Y-%m-%dT%H:%M:%S')
    ls_scene = ls_scene['band_data']

    return ls_scene

##########################

def create_masks(ls_scene, cloud_mask=True, ice_mask=False, ocean_mask=False):
    """
    Creates cloud, ice, and ocean masks from a Landsat scene QA band. By default, 
    clouds are labeled as 1, ice as 2, ocean as 3, and all other pixels are NaN.

    Parameters
    ----------
    ls_scene : xarray.DataArray
        A Landsat scene loaded with a 'qa_pixel' band (as created by `landsat_to_xarray`).
    cloud_mask : bool, optional
        Whether to generate the cloud mask. Default is True.
    ice_mask : bool, optional
        Whether to generate the ice mask. Default is False.
    ocean_mask : bool, optional
        Whether to generate the ocean mask. Default is False.

    Returns
    -------
    xarray.DataArray
        The same input xarray object, but with an added `"mask"` coordinate. 
        In that mask, cloud pixels are assigned 1, ice pixels 2, ocean pixels 3, 
        and everything else is set to NaN.
    """
    
    cloud = []
    ocean = []
    ice = []

    qa = ls_scene.sel(band='qa_pixel').astype('uint16')

    n,c = np.unique(qa, return_counts=True)

    for j in range(len(n)):
        longform = f'{n[j]:016b}'
        if (longform[-7]=='0')|(longform[-3]=='1'): #bit 2 and 6 are for cirrus and clear sky
            cloud.append(n[j])
        if longform[-8:]=='11000000': #bit 6 and 7 give clear sky and water, lower bits need to be 0 
            ocean.append(n[j])
        if longform[-7:]=='1100000': #bit 5 and 6 give ice and clear sky 
            ice.append(n[j])

    if 0 in cloud:
        cloud.remove(0)
    if 1 in cloud:
        cloud.remove(1)

    # mask cloud, ice, and ocean
    if cloud_mask==True:
        # cloud is 2
        mask_c = xr.where(qa.isin(cloud), 1, np.nan)

    if ice_mask==True:
        mask_c = xr.where(qa.isin(ice), 2, mask_c)

    if ocean_mask==True:
        mask_c = xr.where(qa.isin(ocean), 3, mask_c)

    ls_scene.coords['mask'] = (('y', 'x'), mask_c.data)
        
    return ls_scene

##########################

def normalize(array):
    '''
    normalize a dask array so all value are between 0 and 1
    '''
    array_min = array.min(skipna=True)
    array_max = array.max(skipna=True)
    return (array - array_min) / (array_max - array_min)

##########################

def search_stac(url, collection, gjson_outfile=None, bbox=None, timeRange=None, filename=None):
    """
    Search a STAC API for Landsat images based on either:
    - Bounding box and time range, or
    - Specific filename (STAC 'id').

    Parameters:
    -----------
    url : str
        URL to the STAC API.
    collection : str
        Collection name (e.g., "landsat-c2-l2").
    gjson_outfile : str or None
        Output file to save the search result as GeoJSON (optional).
    bbox : list or None
        Bounding box [west, south, east, north] (optional).
    timeRange : str or None
        Time range in ISO format, e.g., '2021-09-01/2023-03-31' (optional).
    filename : str or None
        Exact filename (product ID) to search for (optional).

    Returns:
    --------
    item_collection : pystac.ItemCollection
        Collection of matching STAC items.
    """
    
    api = pystac_client.Client.open(url)

    if filename:
        # Search by filename (ID)
        search = api.search(
            collections=[collection],
            ids=[filename],
        )
        # print(f"Searching for filename: {filename}")
    
    elif bbox and timeRange:
        # Search by bbox and timeRange
        search = api.search(
            bbox=bbox,
            datetime=timeRange,
            collections=[collection],
        )
        # print(f"Searching for items in bbox {bbox} and timeRange {timeRange}")
    
    else:
        raise ValueError("Must provide either a filename, or both bbox and timeRange.")

    items = search.item_collection()

    # print(f"Found {len(items)} item(s)")

    if gjson_outfile:
        items.save_object(gjson_outfile)
    
    return items

###############

def get_lst_mask(lstfile):
    """
    Generates an open ocean mask from a Landsat scene based on the QA band information.

    This function searches for a Landsat scene using a provided filename, loads the 
    'qa_pixel' band, applies cloud, ice, and ocean masking, and then extracts only 
    the open ocean pixels. The output is a mask where open ocean pixels are 1, and 
    all other pixels are NaN.

    Parameters
    ----------
    lstfile : str
        Path or name of the Landsat file used to derive the corresponding STAC search ID.

    Returns
    -------
    numpy.ndarray
        A 2D mask array where open ocean pixels are 1, and all other pixels are NaN.
    """
    filename = lstfile[:-11]
    items = search_stac(url,collection,filename=filename)
    
    # Open stac catalog for some needed info
    catalog = intake.open_stac_item_collection(items)
    sceneid = items[0]
    print(sceneid.id)
    
    scene = catalog[sceneid.id]
    
    # Open all desired bands for one scene
    ls_scene0 = landsat_to_xarray(sceneid,catalog,bandNames=['qa_pixel'])
    ls_scene0 = ls_scene0.rio.write_crs("epsg:3031", inplace=True)
    
    # Create a classification mask, applying cloud, ice, and ocean masks
    ls_scene0 = create_masks(ls_scene0, cloud_mask=True, ice_mask=True, ocean_mask=True)
    
    # Initialize a mask array and set all pixels not classified as open ocean (mask != 3) to NaN
    mask = np.ones(ls_scene0.shape[1:])
    mask[ls_scene0.mask!=3] = np.nan

    try:
        del ls_scene0
    except:
        pass
    
    gc.collect()

    return mask

##########################

def plot_search(gf,satellite,colnm):
    # Plot search AOI and frames on a map using Holoviz Libraries (more on these later)
    cols = gf.loc[:,('id',colnm[0],colnm[1],'geometry')]
    alpha = 1/gf.shape[0]**0.5 # transparency scales w number of images

    footprints = cols.hvplot(geo=True, line_color='k', hover_cols=[colnm[0],colnm[1]], alpha=alpha, title=satellite,tiles='ESRI')
    tiles = gv.tile_sources.CartoEco.options(width=700, height=500) 
    labels = gv.tile_sources.StamenLabels.options(level='annotation')
    tiles * footprints * labels
    
    return footprints

##########################

def subset_img(da,polarx,polary):
    '''
    ***Only works for square grid cropping along the orientation of the grid (not when cropping along lat/lon in a 3031 grid
    
    Subset image in xarray to desired coordinates. Because Landsat polar stereo projection can be oriented
    in many different directions, when coordinates to subset an image are reprojected from lat/lon they may get 
    flipped for which is larger in the pair. This function checks to make sure we are getting a proper subset and 
    avoids 0 pixels on the x or y axis. 
    
    Note: Input shape dimensions and dataarray v. dataset changes things so input needs to be a dataarray w 
          2 dimensions (x,y)
    
    Input:
    da = xarray DataArray to be subset
    polarx = x coordinates to subset by in polar stereographic projection
    polary = y coordinates to subset by in polar stereographic projection
    
    Output:
    ls_sub = subset xarray DataArray
    
    '''
    # ***Landsat shape dimensions are one fewer than they are for LandsatCalibration [0,1] not [1,2], no .to_array() or Band
    ls_sub = da.sel(y=slice(polary[1],polary[0]),x=slice(polarx[0],polarx[1]))

    # Check for right dimensions because y order changes sometimes
    if (ls_sub.x.shape[0]==0) & (ls_sub.y.shape[0]==0):
        # print ('L8 x and y shapes are 0')
        ls_sub = da.sel(y=slice(polary[0],polary[1]),x=slice(polarx[1],polarx[0]))
    elif ls_sub.y.shape[0]==0:
        # print ('L8 y shape is 0')
        ls_sub = da.sel(y=slice(polary[0],polary[1]),x=slice(polarx[0],polarx[1]))
    elif ls_sub.x.shape[0]==0:
        # print ('L8 x shape is 0')
        ls_sub = da.sel(y=slice(polary[1],polary[0]),x=slice(polarx[1],polarx[0]))
    # print(ls_sub.shape)
    
    return ls_sub

##########################

def lsat_reproj(old_cs,new_cs,lbox):
    '''
    Reprojects a bounding box from an old coordinate system to a new one, and checks
    for round-trip transformation errors. The resulting bounding box coordinate order
    may be flipped if the input coordinates indicate an inverted orientation. 
    Diagnostic information is printed, and the transformed bounding box is returned.

    Parameters
    ----------
    old_cs : str
        The Proj4 or EPSG string for the original (source) coordinate system.
        For example: 'epsg:4326' or '+proj=longlat +datum=WGS84 +no_defs'.
    new_cs : str
        The Proj4 or EPSG string for the target coordinate system.
    lbox : list or tuple of float
        A bounding box specified as [ULX, LRY, LRX, ULY] in the old coordinate system.
        - ULX: Upper Left X
        - LRY: Lower Right Y
        - LRX: Lower Right X
        - ULY: Upper Left Y

    Returns
    -------
    bbox : list of tuples
        The transformed bounding box in the new coordinate system. The point order
        depends on whether the original bounding box was flipped or not:
        - Flipped orientation: [(lULX, lLLY), (lLLX, lULY), (lLRX, lURY), (lURX, lLRY)]
        - Normal orientation:  [(lULX, lULY), (lLLX, lLLY), (lLRX, lLRY), (lURX, lURY)]
    checkbox : numpy.ndarray
        An array of the round-trip check coordinates in the old coordinate system
        after transforming back from the new coordinate system. Used to verify
        the accuracy of the transformation.

    Notes
    -----
    - A threshold of 0.5 (`test_threshold`) is used to check whether the
      round-trip transformation error is too high. If the Euclidean distance
      between the original coordinates and the transformed-back coordinates
      exceeds this threshold, a warning is printed.
    - The function prints diagnostic messages, including orientation checks
      and the final bounding box. If the original bounding box was inverted
      (LRY > ULY), a 'flipped orientation' message is displayed, and the points
      are reordered accordingly.
    - The function has not been extensively tested with grids that are rotated
      or otherwise do not follow the normal bounding-box assumptions.

    Examples
    --------
    >>> old_cs = 'epsg:4326'
    >>> new_cs = 'epsg:3031'
    >>> lbox = [-60, -85, 30, -70]  # [ULX, LRY, LRX, ULY]
    >>> bbox, checkbox = lsat_reproj(old_cs, new_cs, lbox)
    >>> bbox
    [(-6671686.551, 241102.289), ... ]  # Example coordinates
    >>> checkbox
    array([-60.3, -70.1,  30.2, -84.9]) # Round-trip result

    bbox comes out with the points out of order for making a polygon though pairs are correct. Order is 0,3,1,2 when done in normal projection. 
    Haven't tested for flipped grid.
    '''
    
    test_threshold = 0.5
    
    # Create a transform object to convert between coordinate systems
    inProj = Proj(init=old_cs)
    outProj = Proj(init=new_cs)
    
    ULX,LRY,LRX,ULY = lbox

    [lULX,lLRX], [lULY,lLRY] =  transform(inProj,outProj,[ULX,LRX], [ULY,LRY], always_xy=True)
    [cULX,cLRX], [cULY,cLRY] =  transform(outProj,inProj,[lULX,lLRX], [lULY,lLRY], always_xy=True)
    [lLLX,lURX], [lLLY,lURY] =  transform(inProj,outProj,[ULX,LRX], [LRY,ULY], always_xy=True)
    [cLLX,cURX], [cLLY,cURY] =  transform(outProj,inProj,[lLLX,lURX], [lLLY,lURY], always_xy=True)

    if LRY>ULY:
        bbox = [(lULX,lLLY),(lLLX,lULY),(lLRX,lURY),(lURX,lLRY)]
        # print('lsat_reproj flipped orientation')
    else:
        bbox = [(lULX,lULY),(lLLX,lLLY),(lLRX,lLRY),(lURX,lURY)]
        # print('lsat_reproj normal orientation')

    checkbox = np.array([cULX,cULY,cLRX,cLRY])
    if np.linalg.norm(checkbox - np.array([ULX,ULY,LRX,LRY])) > test_threshold:
        print(f"Round-trip transformation error 1 of {np.linalg.norm(checkbox - np.array([ULX,ULY,LRX,LRY]))}")
    checkbox = np.array([cLLX,cLLY,cURX,cURY])
    if np.linalg.norm(checkbox - np.array([ULX,LRY,LRX,ULY])) > test_threshold:
        print(f"Round-trip transformation error 2 of {np.linalg.norm(checkbox - np.array([ULX,LRY,LRX,ULY]))}")
    # print (f'bbox={bbox}')
    # print (f'lbox={lbox}')
    # print (f'checkbox={checkbox}')
    
    return bbox,checkbox

##########################

def crop_xarray_dataarray_with_polygon(dataarray, polygon):
    """
    Crop an xarray.DataArray using a polygon.
    
    Parameters:
    - dataarray: xarray.DataArray with x and y coordinates.
    - polygon: Shapely Polygon object defining the crop area.
    
    Returns:
    - Cropped xarray.DataArray.
    """
    # Generate a 2D array of shapely Point objects for each grid point
    lon, lat = np.meshgrid(dataarray.x.values, dataarray.y.values)
    points = np.vectorize(Point)(lon, lat)
    
    # Create a mask where points within the polygon are True
    mask_func = np.vectorize(polygon.contains)
    mask = mask_func(points)
    
    # Convert the mask to an xarray.DataArray
    mask_da = xr.DataArray(mask, dims=["y", "x"], coords={"y": dataarray.y, "x": dataarray.x})
    
    # Apply the mask to the dataarray, cropping to the polygon
    # Use where method with drop=True to drop values outside the polygon
    cropped_dataarray = dataarray.where(mask_da, drop=True)
    
    return cropped_dataarray

##########################

def km_to_decimal_degrees(km, latitude, direction='latitude'):
    """
    Convert a distance in kilometers to decimal degrees of latitude or longitude,
    given a specific latitude.

    Parameters
    ----------
    km : float
        The distance in kilometers to be converted.
    latitude : float
        The latitude (in decimal degrees, from -90 to +90) where the conversion
        is being applied. Used only if direction='longitude'.
    direction : str, optional
        Either 'latitude' or 'longitude'. Determines whether to convert
        km to decimal degrees of latitude or longitude. Default is 'latitude'.

    Returns
    -------
    float
        The approximate decimal degrees that correspond to the given distance in km
        at the specified latitude (for longitude) or globally (for latitude).

    Notes
    -----
    1° latitude ~ 111.32 km everywhere on Earth.
    1° longitude ~ 111.32 km * cos(latitude), which is why the
        conversion depends on the specified latitude for 'longitude'.
    This function uses a spherical Earth approximation and is not exact
    at very high latitudes or for large distances.

    Examples
    --------
    >>> # Convert 10 km to decimal degrees of latitude (anywhere)
    >>> km_to_decimal_degrees(10, latitude=0, direction='latitude')
    0.0898...

    >>> # Convert 10 km to decimal degrees of longitude at latitude 69°S
    >>> km_to_decimal_degrees(10, latitude=-69, direction='longitude')
    0.2515...
    """
    if direction.lower() == 'latitude':
        # 1 degree of latitude ≈ 111.32 km (on average)
        deg = km / 111.32
    elif direction.lower() == 'longitude':
        # 1 degree of longitude ≈ 111.32 km * cos(lat)
        deg = km / (111.32 * math.cos(math.radians(latitude)))
    else:
        raise ValueError("direction must be 'latitude' or 'longitude'")
    return deg

##########################

def crosses_idl(coords):
    '''
    Determine if the set of coordinates crosses the International Dateline in a way that will mess up the creation of a polygon
    
    Variables:
    coords = list of lon, lat tuples
    
    Output:
    True or False
    '''
    
    for (lon1, lat1), (lon2, lat2) in zip(coords, coords[1:]):
        if abs(lon1 - lon2) >= 180:
            return True
    return False

##########################

def plot_geotiff(filepath):
    # Open the geotiff file
    with rio.open(filepath) as src:
        # Reproject the dataset to lat/lon
        transform, width, height = rio.warp.calculate_default_transform(
            src.crs, 'EPSG:4326', src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': 'EPSG:4326',
            'transform': transform,
            'width': width,
            'height': height
        })

        # Read the data and reproject
        with rio.MemoryFile() as memfile:
            with memfile.open(**kwargs) as dst:
                rio.warp.reproject(
                    source=rio.band(src, 1),
                    destination=rio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs='EPSG:4326',
                    resampling=rio.enums.Resampling.nearest
                )
                data = dst.read(1)

    # Plot the data
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE)
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    ax.imshow(data, transform=ccrs.PlateCarree(), origin='upper', extent=dst.bounds, cmap='viridis')
    ax.set_title(os.path.basename(filepath))
    plt.show()


##########################

def create_geotiff_dropdown(directory):
    # Create a dropdown widget with all GeoTIFF files in the directory
    tif_files = [f for f in os.listdir(directory) if f.endswith('.tif')]
    dropdown = Dropdown(options=tif_files, description='Select a file:')
    
    # Update function to plot based on the selected file
    def update_plot(selected_file):
        plot_geotiff(os.path.join(directory, selected_file))
    
    interact(update_plot, selected_file=dropdown)

##########################

# Preprocess to add time dimension and the file name to open_mfdataset for landsat using the filename
def add_time_dim(ds):
    lstr = ds.encoding["source"].split("LC0",1)[1]
    times = pd.to_datetime(lstr[14:22]+lstr[38:44], format='%Y%m%d%H%M%S')
    idee = ds.encoding["source"].split("/")[8][:-4] # The first number depends on how many subdirectories the file is in
    return ds.assign_coords(time=times,ID=idee)


# +
# Atmospheric correction and production of SST
'''
Functions to find the matching MODIS water vapor image for atmospheric correction and production of SST.
Open_MODIS finds and downloads the closest MODIS water vapor image to a specific landsat image. Get_wv
aligns and subsets the modis image grid to landsat using MODISlookup and subsamples and extracts the data 
onto the Landsat grid using uniqueMODIS
'''

# def open_MODIS(ls_scene,scene,modout_path):
#     '''
#     Search MOD/MDY07 atmospheric data and open water vapor for data collected closest in time to 
#     Landsat scene.
    
#     Input:
#     ls_scene = xarray dataset with Landsat scene
#     modout_path = directory path for MODIS data
#     scene = STAC catalog item
    
#     Output:
#     mod07 = xarray dataset with MODIS (MOD/MDY07) water vapor 
#     modfilenm = MODIS filename for image used in atm correction
#     '''

#     # Get spatial extent of Landsat scene in lat/lon
#     mbbox = (scene.metadata['bbox'][0], scene.metadata['bbox'][1], scene.metadata['bbox'][2], scene.metadata['bbox'][3]) #(west, south, east, north) 
#     lsatpoly = Polygon([(mbbox[0],mbbox[1]),(mbbox[0],mbbox[3]),(mbbox[2],mbbox[3]),(mbbox[2],mbbox[1]),(mbbox[0],mbbox[1])]) # ensure full lineup between landsat and modis

#     ls_time = pd.to_datetime(ls_scene.time.values)
#     calc_dt = datetime.strptime(ls_time.strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
#     start_dt = (calc_dt + timedelta(days=-0.5)).strftime('%Y-%m-%d %H:%M:%S')
#     end_dt = (calc_dt + timedelta(days=0.5)).strftime('%Y-%m-%d %H:%M:%S')

#     # Gather all files from search location from Terra and Aqua for the same day as the Landsat image
#     results = earthaccess.search_data(
#         short_name='MOD07_L2',
#         bounding_box=mbbox,
#         # Day of a landsat scene to day after - searches day of only
#         temporal=(start_dt,end_dt)
#     )
#     results2 = earthaccess.search_data(
#         short_name='MYD07_L2',
#         bounding_box=mbbox,
#         # Day of a landsat scene to day after - searches day of only
#         temporal=(start_dt,end_dt)
#     )
#     results = results + results2
#     print (f'{len(results)} TOTAL granules')

#     # Accept only granules that overlap at least 100% with Landsat (percent_dif<0.1 is the other option)
#     best_grans = []
#     for granule in results:
#         try:
#             granule['umm']['SpatialExtent']['HorizontalSpatialDomain']['Geometry']['GPolygons']
#         except Exception as error:
#             print(error)
#             continue
#         for num in range(len(granule['umm']['SpatialExtent']['HorizontalSpatialDomain']['Geometry']['GPolygons'])):
#             try:
#                 map_points = [(xi['Longitude'],xi['Latitude']) for xi in granule['umm']['SpatialExtent']['HorizontalSpatialDomain']['Geometry']['GPolygons'][num]['Boundary']['Points']]
#                 pgon = Polygon(map_points)
#                 percent_dif = lsatpoly.difference(pgon).area/lsatpoly.area
#                 if percent_dif < 0.1:
#                     if crosses_idl(map_points):
#                         print (f'A granule has messed up polygon that likely crosses the International DateLine')
#                     else:
#                         best_grans.append(granule)
#                         continue
#             except Exception as error:
#                 print(error)
#                 # Would love to raise an exception for a valueerror except for GEOSError but not sure how 
#     print(f'{len(best_grans)} TOTAL granules w overlap')

#     # Find MODIS image closest in time to the Landsat image
#     Mdates = [pd.to_datetime(granule['umm']['TemporalExtent']['RangeDateTime']['BeginningDateTime']) for granule in best_grans]
#     ind = Mdates.index(min( Mdates, key=lambda x: abs(x - pytz.utc.localize(pd.to_datetime(ls_time)))))
#     print(f'Time difference between MODIS and Landsat: {abs(Mdates[ind] - pytz.utc.localize(pd.to_datetime(ls_time)))}')

#     # Download MODIS data if needed

#     # # This doesn't work because xarray can't open legacy HDF EOS data formats
#     # mod07 = xr.open_mfdataset(earthaccess.open(results))

#     # Use these access pathways while S3 streaming is not working
#     data_links = [granule.data_links(access="external") for granule in best_grans[ind:ind+1]]
#     netcdf_list = [g._filter_related_links("USE SERVICE API")[0].replace(".html", ".nc4") for g in best_grans[ind:ind+1]]
#     # This is going to be slow as we are asking Opendap to format HDF into NetCDF4 so we only processing 3 granules
#     # and Opendap is very prone to failures due concurrent connections, not ideal.
#     file_handlers = earthaccess.download(netcdf_list,modout_path,provider='NSIDC')

#     # Open MODIS data
#     mod_list = os.listdir(modout_path)
#     mod_list = [file for file in mod_list if file[-3:]=='nc4']
#     print(mod_list)
#     modfilenm = mod_list[0]
    
#     os.rename(f'{modout_path}/{modfilenm}', f'{modout_path}/{modfilenm}.gz')
#     with gzip.open(f'{modout_path}/{modfilenm}.gz', 'rb') as f_in:
#         with open(f'{modout_path}/{modfilenm}', 'wb') as f_out:
#             f_out.write(f_in.read())

#     mod07 = xr.open_dataset(f'{modout_path}/{modfilenm}')
#     mod07 = mod07.rio.write_crs('epsg:4326')

#     # Delete MODIS file
#     os.remove(f'{modout_path}/{modfilenm}')
#     os.remove(f'{modout_path}/{modfilenm}.gz')
    
#     return mod07,modfilenm

# ##########################

# # Notes for changes - MODISlookup2 doesn't need to output lat/lon, but if want to do the check, 
# # can take the lat/lon check out of aligne and do it in 
# def get_wv(ls_scene,mod07,spacing,param,scene,interp=0):
#     '''
#     Aligns and resamples MODIS water vapor data to match the spatial resolution and 
#     alignment of a given Landsat scene. The function optionally applies interpolation 
#     to improve the data quality.

#     Parameters:
#     ls_scene (xarray.Dataset): The Landsat scene dataset containing spatial coordinates.
#     mod07 (xarray.Dataset): The MODIS dataset containing water vapor data and coordinates (MOD/MDY07).
#     spacing (list): Desired spatial resolution (y, x) for alignment with MODIS data in meters.
#     param (str): Parameter name for the desired dataset within the MODIS file.
#     scene: 
#     interp (int): Controls interpolation mode - 0 for none, 1 for bicubic kriging interpolation.

#     Returns:
#     WV_xr (xarray.DataArray): The processed xarray data array containing the Landsat-aligned 
#                               and resampled water vapor data from MODIS.

#     Note:
#     This function is also used in LandsatCalibration; any changes here should consider potential impacts there 
#     - may need to be copied/generalized.

#     The function performs several key operations:
#     1. Defines the bounding box for the Landsat scene based on its spatial coordinates.
#     2. Extracts the relevant water vapor data from the MODIS dataset using the specified parameter key.
#     3. Validates the geographic coordinate ranges (latitude and longitude) of the MODIS data.
#     4. Applies PyGMT interpolation if requested to generate a smoother water vapor data surface.
#     5. Utilizes a lookup function to align MODIS data indices with the Landsat grid based on the specified spatial resolution.
#     6. Aligns and resamples the MODIS data to match the Landsat scene's grid and spatial resolution.
#     7. Adjusts the coordinate system of the output to ensure compatibility with further processing or analysis.

#     Difference: no bicubic spline interpolation in LsatCalib during the upsampling, don't set new indexes at the end
#     '''
#     # Read in desired variables
#     ULX = ls_scene.x[0] 
#     ULY = ls_scene.y[0]  
#     LRX = ls_scene.x[-1] 
#     LRY = ls_scene.y[-1] 
#     box = [ULX,LRX,ULY,LRY]
    
#     #Extract desired datasets from MODIS file from lookup key
#     data = mod07[param].values
#     lat, lon = mod07.Latitude, mod07.Longitude
#     #data.attributes()

#     # Test lat is in correct range
#     if ~((lat <= 90) & (lat >= -90)).all():
#         print('MODIS latitude not between -90 and 90')
#     # Test lon is in correct range
#     if ~((lon <= 180) & (lon >= -180)).all():
#         print('MODIS longitude not between -180 and 180')

#     # ***Need to use climatology to retrieve quantile data from this area
    
#     # # Get rid of low outliers from over ice, cutoff for 98.5%
#     # outlier = np.quantile(data[np.isfinite(data)],0.015) #0.015
#     # mask2 = np.ones(data.shape)
#     # mask2[data<outlier] = np.nan
#     # data = np.around(mask2*data,decimals=5)

#     # Interpolate using PyGMT
#     if interp==1:  
#         grid = interpMOD(data,lat,lon)
        
#         # Produce indicies for aligning MODIS pixel subset to match Landsat image at 4000m (or 300)resolution
#         indiciesMOD,lines,samples,lat,lon = MODISlookup(mod07,ls_scene,box,spacing,scene,interpgrid=grid)
#         data = grid.values
        
#     else:
#         # Produce indicies for aligning MODIS pixel subset to match Landsat image at 4000m (or 300)resolution
#         indiciesMOD,lines,samples,lat,lon = MODISlookup(mod07,ls_scene,box,spacing,scene)

#     # Align and resample MODIS WV to Landsat at indicated spacing with correct axes
#     dataOutWV_xr = alignMODIS(data,lat,lon,param,indiciesMOD,lines,samples,mod07,ls_scene,spacing)
    
#     # # Resample WV to Landsat resolution and interpolate with B-spline
#     # # Need to use 0.1k (this samples at .1 of the grid)
#     # # Output of shape fits and need to adjust x and y coords cuz are wrong
#     ups_factor = 30/spacing[0]
#     WV_upsample = pygmt.grdsample(grid=dataOutWV_xr, spacing=f'{ups_factor}k', interpolation='c')
#     # WV_upsample = xr.open_dataarray(lsatpath+'WV_upsample_B-spline_'+str(ls_scene.id.values))
#     # # Resample WV to Landsat resolution manual - no interpolation
#     # WV_resamp = MODresample(ls_scene,dataOutWV,y1,x1,spacing)

#     # Put into Xarray
#     # Sometimes spacing works properly with -1 and sometimes not
#     latnew = np.arange(dataOutWV_xr.latitude[0],dataOutWV_xr.latitude[-1]+1,(dataOutWV_xr.latitude[-1]-dataOutWV_xr.latitude[0])/(WV_upsample.shape[0]-1))
#     if (WV_upsample.shape[0]!=latnew.shape[0]):
#         latnew = np.arange(dataOutWV_xr.latitude[0],dataOutWV_xr.latitude[-1]+1,(dataOutWV_xr.latitude[-1]-dataOutWV_xr.latitude[0])/(WV_upsample.shape[0]))

#     # Put into Xarray
#     latnew = ls_scene.y[:WV_upsample.shape[0]].values
#     lonnew = ls_scene.x[:WV_upsample.shape[1]].values
#     if dataOutWV_xr.latitude[0]!=latnew[0]:
#         print('Aligned y dim needs to start with the same coordinate as ls_scene')
#     if dataOutWV_xr.longitude[0]!=lonnew[0]:
#         print('Aligned x dim needs to start with the same coordinate as ls_scene')
    
#     WV_xr = xr.DataArray(WV_upsample,name='SST',dims=["y","x"], coords={"latitude": (["y"],latnew), "longitude": (["x"],lonnew)})

#     WV_xr = WV_xr.rio.write_crs("epsg:3031", inplace=True)
#     WV_xr = WV_xr.rename({'longitude':'x','latitude':'y'})
    
#     return WV_xr

# ##########################
            
# def interpMOD(data,lat,lon):
#     """
#     Interpolate spatial water vapor data using PyGMT.

#     This function takes arrays of water vapor data along with corresponding latitude and longitude values,
#     performs interpolation to fill in gaps in the data, and produces a continuous surface representation of water vapor.

#     Args:
#         data (numpy.ndarray): 2D array of water vapor measurements.
#         lat (numpy.ndarray): 2D array of latitude values corresponding to `data`.
#         lon (numpy.ndarray): 2D array of longitude values corresponding to `data`.

#     Returns:
#         grid (xarray.DataArray): A PyGMT grid object representing the interpolated surface of water vapor data.
#     """
    
#     # Interpolate using PyGMT
#     # Extract necessary data into Pandas DataFrame (required for PyGMT)
#     df = pd.DataFrame({
#         'longitude': lon.values.flatten(), # Flatten to convert from 2D to 1D array
#         'latitude': lat.values.flatten(),
#         'water_vapor': data.flatten() # Actual data values to be interpolated
#     })

#     # Remove missing or NaN values from DataFrame, as `surface` cannot handle them
#     df = df.dropna(subset=['water_vapor'])

#     # Determine the geographical extent for the interpolation based on the provided data points.
#     # This is necessary to define the spatial domain over which PyGMT will perform interpolation.
#     # [xmin, xmax, ymin, ymax] - made this the full image
#     region = [df.longitude.min(), df.longitude.max(), df.latitude.min(), df.latitude.max()]
#     # Alternatively, if the region is predefined (e.g., from metadata), it can be set directly.

#     # Use PyGMT to interpolate the data. PyGMT requires a session context to manage memory and configuration
#     # settings efficiently during its operations.
#     with pygmt.clib.Session() as session:
#         # Perform grid-based surface interpolation.
#         # The `data` parameter takes longitude, latitude, and water vapor as a NumPy array.
#         # `region` specifies the geographical extent.
#         # `spacing` sets the resolution of the output grid, here 0.3 km for high resolution.
#         # `tension` controls the stiffness of the interpolating surface. A value of 0.25 gives a balance between
#         # fitting the data closely and producing a smooth surface.
#         grid = pygmt.surface(
#             data=df[['longitude', 'latitude', 'water_vapor']].to_numpy(),  # Input data as NumPy array
#             region=region,  
#             spacing=['0.15k','0.05k'],  # f'0.3k'
#             tension=0.95,  
#         )   
    
#     return grid

# ##########################

# def MODISlookup(mod07,lsat_filt_msk,box,spacing,scene,interpgrid=None):
#     '''
#     Look up indices for aligning MODIS product to the Landsat grid
#     # Modified from http://stackoverflow.com/questions/2922532/obtain-latitude-and-longitude-from-a-geotiff-file 
#     # and Shane Grigsby

#     Variables:    
#     mod07 = xarray with MODIS data with crs 4326 assigned
#     lsat_filt_msk =  Landsat xarray DataArray
#     box = list with [left easting,right easting,top northing,bottom northing]
#     spacing = desired pixel size for extraction, list of [east/west, north/south] 
#           (recommend choosing a number that allows for fast calculations and even division by 30)
#     scene = 
#     interpgrid = xarray of mod07 data that has been through interpolation in PyGMT (optional)

#     Output:
#     indiciesMOD = indicies used to project MODIS pixels to match Landsat pixels
#     lines = number of lines in Landsat file/MODIS output shape
#     samples = number of samples in Landsat file/MODIS output shape
#     lon,lat = 2D lon and lat coordinates for grid
#     '''
#     test_threshold = 5
    
#     if interpgrid is None:
#         lat, lon = mod07.Latitude.values, mod07.Longitude.values
#     else:
#         lat, lon = interpgrid.lat, interpgrid.lon
#         lon, lat = np.meshgrid(lon,lat)

#     # Test lat is in correct range
#     if ~((lat <= 90) & (lat >= -90)).all():
#         print('MODIS latitude not between -90 and 90')
#     # Test lon is in correct range
#     if ~((lon <= 180) & (lon >= -180)).all():
#         print('MODIS longitude not between -180 and 180')

#     # Get the existing coordinate system
#     old_cs = ls_scene.rio.crs # 'epsg:3031'
#     new_cs = mod07.rio.crs # 'epsg:4326'

#     # Create a transform object to convert between coordinate systems
#     inProj = Proj(init=old_cs)
#     outProj = Proj(init=new_cs)

#     # Parse coordinates and spacing to different variables
#     west,east,north,south = box
#     ewspace,nsspace = spacing

#     # Setting up grid, x coord from here to here at this spacing, mesh grid makes 2D
#     samples = len(np.r_[west:east+1:ewspace])
#     lines = len(np.r_[north:south-1:nsspace])#ns space is -300, could also do 30 instead of 300, but would just have duplicate pixels
#     if lines==0:
#         lines = len(np.r_[south:north-1:nsspace])

#     # x1, y1 = np.meshgrid(np.r_[west:east:ewspace],np.r_[north:south:nsspace]) # offset by 1 meter to preserve shape
#     ewdnsamp = int(spacing[0]/30)
#     nsdnsamp = int(spacing[1]/30)
    
#     # Set up coarser sampling and check to make sure is in the same orientation as the original Landsat grid
#     xresamp = ls_scene.x.isel(x=slice(None, None, ewdnsamp)).values
#     if xresamp[0]!=ls_scene.x.values[0]:
#         xresamp = ls_scene.x.isel(x=slice(None, None, -ewdnsamp)).values
#         print('x resample reversed')
#     yresamp = ls_scene.y.isel(y=slice(None, None, nsdnsamp)).values
#     if yresamp[0]!=ls_scene.y.values[0]:
#         yresamp = ls_scene.y.isel(y=slice(None, None, -nsdnsamp)).values
#         print('y resample reversed')
#     x1, y1 = np.meshgrid(xresamp,yresamp)
#     LScoords = np.vstack([x1.ravel(),y1.ravel()]).T
#     if (LScoords[0,0]!=ls_scene.x.values[0]) |  (LScoords[0,1]!=ls_scene.y.values[0]):
#         raise Exception('Landsat coordinates do not match expected during MODIS lookup')

#     # Ravel so ND can lookup easily
#     # Convert from LS map coords to lat lon --> x = lon, y = lat (usually?)

#     ###Make into test
#     # Test that reprojection is working correctly on first and last grid point using round-trip transformation
#     xs1, ys1 =  transform(inProj,outProj,LScoords[0,0], LScoords[0,1], radians=True, always_xy=True)
#     xsl1, ysl1 =  transform(outProj,inProj,xs1, ys1, radians=True, always_xy=True)
#     if np.linalg.norm(np.array([xsl1, ysl1]) - LScoords[0,:]) > test_threshold:
#         print(f"Round-trip transformation error for point {LScoords[0,:]}, {np.linalg.norm(np.array([xsl1, ysl1]) - LScoords[0,:])}")
#     else:
#         # If passes, run on entire grid
#         xs, ys =  transform(inProj,outProj,LScoords[:,0], LScoords[:,1], radians=True, always_xy=True)
#     ###
    
#     # Produce landsat reprojected to lat/lon and ensure lat is in 0 column
#     # Test: landsat data is in correct orientation as long as lat is in col 0 and lon in col 1
#     grid_coords = test_gridcoords(xs,ys,scene)

#     # Test that lines and samples match grid_coords
#     if len(grid_coords) != lines*samples:
#         raise Exception(f'Size of grid coordinates do not match low resolution Landsat dims: {len(grid_coords)} vs. {lines*samples}. Check that spacing is negative for y')
#     MODIS_coords = np.vstack([lat.ravel(),lon.ravel()]).T
#     MODIS_coords *= np.pi / 180. # to radians

#     # Build lookup, haversine = calc dist between lat,lon pairs so can do nearest neighbor on sphere - if did utm it would be planar
#     MOD_Ball = BallTree(MODIS_coords,metric='haversine') #sklearn library
#     distanceMOD, indiciesMOD= MOD_Ball.query(grid_coords, dualtree=True, breadth_first=True)
        
#     return indiciesMOD,lines,samples,lat,lon

# ##########################

# def test_gridcoords(xs,ys,scene):
#     '''
#     Test to ensure grid lat and lon are not swapped during reprojection and output grid coordinates
#     that have been raveled and stacked for input into BallTree
    
#     Variables:
#     xs = 1D radians representing longitude 
#     ys = 1D radians representing latitude
#     scene = catalog item for landsat image
    
#     Output:
#     grid_coords = two columns of x/y radian pairs representing lon/lat
#     '''
    
#     # Convert radians to lat/lon
#     x_check = xs * 180. / np.pi
#     y_check = ys * 180. / np.pi
    
#     # We know lat is ys and lon is xs if this is true so goes in 0 column position to match MODIS
#     if ((-90 <= y_check) & (y_check <= -60)).all() & ~((-90 <= x_check) & (x_check <= -60)).all():
#         grid_coords = np.vstack([ys.ravel(),xs.ravel()]).T # note y / x switch (i.e., lat long convention)
#         print('Latitude in proper position')

#     # A small subset of data have lat and lon that falls between -60 and -90 so test if the landsat metadata confirms that
#     elif ((-90 <= y_check) & (y_check <= -60)).all():
#         llons = np.array((float(scene.metadata['bbox'][0]), float(scene.metadata['bbox'][2])))
#         # ys is latitude if true here
#         if ((-90 <= llons) & (llons <= -60)).all():
#             grid_coords = np.vstack([ys.ravel(),xs.ravel()]).T # note y / x switch (i.e., lat long convention)
#             print('Latitude in proper position')
#         # xs is latitude if not and goes in 0 column position
#         else:
#             grid_coords = np.vstack([xs.ravel(),ys.ravel()]).T 
#             print('Latitude in wrong position')

#     # Otherwise xs is latitude and goes in 0 column position
#     else:
#         grid_coords = np.vstack([xs.ravel(),ys.ravel()]).T
#         print('Latitude in wrong position')
    
#     return grid_coords

# ##########################

# def alignMODIS(data,lat,lon,param,indiciesMOD,lines,samples,mod07,ls_scene,spacing):
#     test_threshold = 5
    
#     # Check to ensure lat/lon and data have compatible shapes
#     if (np.shape(lat)== np.shape(lon)== np.shape(data))==False:
#         raise Exception("Error in creating indicies, lat/lon and data shapes do not match")
        
#     # Extract MODIS data into Landsat grid and gather unique data values
#     dataOut,uniqWV = uniqueMODIS(data,param,indiciesMOD,lines,samples)
    
#     # Check grid directionality and create matching x/y for new grid
#     # Define the source and target coordinate reference systems (CRS)
#     src_crs = mod07.rio.crs #'epsg:4326'  MODIS
#     target_crs = ls_scene.rio.crs #crs[6:] # 'epsg:3031' Landsat

#     # Create a PyProj transformer
#     transformer = pyproj.Transformer.from_crs(src_crs, target_crs, always_xy=True)
#     transformer_test = pyproj.Transformer.from_crs(target_crs, src_crs, always_xy=True)

#     # Test that reprojection is working correctly on first and last modis grid point
#     xm1,xm2 = lon[0,0],lon[-1,-1]
#     ym1,ym2 = lat[0,0],lat[-1,-1]
#     xx,yy = [xm1,xm2], [ym1,ym2]
#     xs1, ys1 =  transformer.transform(xx,yy)
#     xsl1, ysl1 = transformer_test.transform(xs1, ys1)
#     for i,n in enumerate(xsl1):
#         if np.linalg.norm(np.array([xsl1[i], ysl1[i]]) - [xx[i],yy[i]]) > test_threshold:
#             print(f"Round-trip transformation error for {sceneid}, {np.linalg.norm(np.array([xsl1[i], ysl1[i]]) - xx[i],yy[i])}")
    
#     # Spacing to create x and y parameters at the correct spacing
#     redy = int(abs(spacing[0]/30))
#     redx = int(abs(spacing[1]/30))

#     # Set up coarser sampling and check to make sure is in the same orientation as the original Landsat grid
#     xgrid = ls_scene.x.isel(x=slice(None, None, redx)).values
#     if xgrid[0]!=ls_scene.x.values[0]:
#         xgrid = ls_scene.x.isel(x=slice(None, None, -redx)).values
#     ygrid = ls_scene.y.isel(y=slice(None, None, redy)).values
#     if ygrid[0]!=ls_scene.y.values[0]:
#         ygrid = ls_scene.y.isel(y=slice(None, None, -redy)).values
#     if (xgrid[0]!=ls_scene.x.values[0]) |  (ygrid[0]!=ls_scene.y.values[0]):
#         raise Exception('Landsat coordinates do not match expected during MODIS lookup')
    
#     # Create xarray from numpy array
#     dataOut_xr = xr.DataArray(dataOut,name='SST',dims=["y","x"], coords={"latitude": (["y"],ygrid), "longitude": (["x"],xgrid)})
    
#     return dataOut_xr

# ##########################

# def uniqueMODIS(data,param,indiciesMOD,lines,samples):
#     '''
#     Extracts data values and unique values from desired MODIS dataset that corresponds to Landsat file
#     # Modified from http://stackoverflow.com/questions/2922532/obtain-latitude-and-longitude-from-a-geotiff-file
    
#     Variables: 
#     data = array with MOD07 data in crs 4326 assigned 
#     param =  string for desired dataset from MODIS file
#     indiciesMOD = indicies output for neighest neighbor query from MODIS to Landsat coordinates
#     lines = number of lines in Landsat file/MODIS output shape
#     samples = number of samples in Landsat file/MODIS output shape
    
#     Output:
#     dataOut = MODIS atm image subset and aligned to Landsat image pixels
#     uniq = uniq MODIS atm values within area of Landsat image
#     #counts = count for each unique value in subset
#     '''
#     # Convert from K to C
#     KtoC = -273.15
    
#     # Scaling coefficients for MODIS data
#     wv_scale = 0.0010000000474974513
#     ozone_scale = 0.10000000149011612

#     # Reproject data from MODIS into corresponding postions for Landsat pixels for water vapor and ozone
#     if param == 'sst':
#         dataOut = np.reshape(np.array(data.ravel())[indiciesMOD],(lines,samples))#* # to scale?
#         dataOut[dataOut < -3] = np.nan
#         MODimg = np.array(data)#* # to scale?
#         MODimg[MODimg < 0] = np.nan
#     elif param == 'Water_Vapor':
#         dataOut = np.reshape(np.array(data.ravel())[indiciesMOD] * wv_scale,(lines,samples))
#         dataOut[dataOut < 0] = np.nan
#         MODimg = np.array(data*wv_scale)
#         MODimg[MODimg < 0] = np.nan
#     elif param == 'Total_Ozone':
#         dataOut = np.reshape(np.array(data.ravel())[indiciesMOD] * ozone_scale,(lines,samples))
#         dataOut[dataOut < 225] = np.nan
#         dataOut[dataOut > 430] = np.nan
#         MODimg = np.array(data*ozone_scale)
#         MODimg[MODimg < 0] = np.nan

#     # Get unique values for datasets within Landsat extent
#     #uniq, inverse, counts= np.unique(dataOut, return_inverse=True, return_counts=True)
#     uniq = set(dataOut[np.isfinite(dataOut)])
    
#     return dataOut,uniq # Can also output MODimg and inverse and counts if desired


# Atmospheric correction and production of SST
'''
Functions to find the matching MODIS water vapor image for atmospheric correction and production of SST.
Open_MODIS finds and downloads the closest MODIS water vapor image to a specific landsat image. Get_wv
aligns and subsets the modis image grid to landsat using MODISlookup and subsamples and extracts the data 
onto the Landsat grid using uniqueMODIS
'''

def open_MODIS(ls_scene,scene,modout_path):
    '''
    Search MOD/MDY07 atmospheric data and open water vapor for data collected closest in time to 
    Landsat scene.
    
    Input:
    ls_scene = xarray dataset with Landsat scene
    modout_path = directory path for MODIS data
    scene = STAC catalog item
    
    Output:
    mod07 = xarray dataset with MODIS (MOD/MDY07) water vapor 
    modfilenm = MODIS filename for image used in atm correction
    '''

    # Get spatial extent of Landsat scene in lat/lon
    mbbox = (scene.metadata['bbox'][0], scene.metadata['bbox'][1], scene.metadata['bbox'][2], scene.metadata['bbox'][3]) #(west, south, east, north) 
    lsatpoly = Polygon([(mbbox[0],mbbox[1]),(mbbox[0],mbbox[3]),(mbbox[2],mbbox[3]),(mbbox[2],mbbox[1]),(mbbox[0],mbbox[1])]) # ensure full lineup between landsat and modis

    ls_time = pd.to_datetime(ls_scene.time.values)
    calc_dt = datetime.strptime(ls_time.strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
    start_dt = (calc_dt + timedelta(days=-0.5)).strftime('%Y-%m-%d %H:%M:%S')
    end_dt = (calc_dt + timedelta(days=0.5)).strftime('%Y-%m-%d %H:%M:%S')

    # Gather all files from search location from Terra and Aqua for the same day as the Landsat image
    results = earthaccess.search_data(
        short_name='MOD07_L2',
        bounding_box=mbbox,
        # Day of a landsat scene to day after - searches day of only
        temporal=(start_dt,end_dt)
    )
    results2 = earthaccess.search_data(
        short_name='MYD07_L2',
        bounding_box=mbbox,
        # Day of a landsat scene to day after - searches day of only
        temporal=(start_dt,end_dt)
    )
    results = results + results2
    print (f'{len(results)} TOTAL granules')

    # Accept only granules that overlap at least 100% with Landsat (percent_dif<0.1 is the other option)
    best_grans = []
    for granule in results:
        try:
            granule['umm']['SpatialExtent']['HorizontalSpatialDomain']['Geometry']['GPolygons']
        except Exception as error:
            print(error)
            continue
        for num in range(len(granule['umm']['SpatialExtent']['HorizontalSpatialDomain']['Geometry']['GPolygons'])):
            try:
                map_points = [(xi['Longitude'],xi['Latitude']) for xi in granule['umm']['SpatialExtent']['HorizontalSpatialDomain']['Geometry']['GPolygons'][num]['Boundary']['Points']]
                pgon = Polygon(map_points)
                percent_dif = lsatpoly.difference(pgon).area/lsatpoly.area
                if percent_dif == 0.0:
                    if crosses_idl(map_points):
                        print (f'A granule has a problematic polygon that likely crosses the International DateLine')
                    else:
                        best_grans.append(granule)
                        continue
            except Exception as error:
                print(error)
                # Would love to raise an exception for a valueerror except for GEOSError but not sure how 
    print(f'{len(best_grans)} TOTAL granules w overlap')

    # Find MODIS image closest in time to the Landsat image
    Mdates = [pd.to_datetime(granule['umm']['TemporalExtent']['RangeDateTime']['BeginningDateTime']) for granule in best_grans]
    ind = Mdates.index(min( Mdates, key=lambda x: abs(x - pytz.utc.localize(pd.to_datetime(ls_time)))))
    print(f'Time difference between MODIS and Landsat: {abs(Mdates[ind] - pytz.utc.localize(pd.to_datetime(ls_time)))}')

    # Download MODIS data if needed

    # # This doesn't work because xarray can't open legacy HDF EOS data formats
    # mod07 = xr.open_mfdataset(earthaccess.open(results))

    # Use these access pathways while S3 streaming is not working
    data_links = [granule.data_links(access="external") for granule in best_grans[ind:ind+1]]
    netcdf_list = [g._filter_related_links("USE SERVICE API")[0].replace(".html", ".nc4") for g in best_grans[ind:ind+1]]
    # This is going to be slow as we are asking Opendap to format HDF into NetCDF4 so we only processing 3 granules
    # and Opendap is very prone to failures due concurrent connections, not ideal.
    file_handlers = earthaccess.download(netcdf_list,modout_path,provider='NSIDC')

    # Open MODIS data
    mod_list = os.listdir(modout_path)
    mod_list = [file for file in mod_list if file[-3:]=='nc4']
    # print(mod_list)
    modfilenm = mod_list[0]
    
    os.rename(f'{modout_path}/{modfilenm}', f'{modout_path}/{modfilenm}.gz')
    with gzip.open(f'{modout_path}/{modfilenm}.gz', 'rb') as f_in:
        with open(f'{modout_path}/{modfilenm}', 'wb') as f_out:
            f_out.write(f_in.read())

    mod07 = xr.open_dataset(f'{modout_path}/{modfilenm}')
    mod07 = mod07.rio.write_crs('epsg:4326')

    # Delete MODIS file
    os.remove(f'{modout_path}/{modfilenm}')
    os.remove(f'{modout_path}/{modfilenm}.gz')
    
    return mod07,modfilenm

##########################
    
def find_MODIS(lonboundsC,latboundsC,ls_scene):
    '''
    Finds the MODIS SST scene most closely coincident to a Landsat scene
    Uses full Landsat scene extent, not cropped
    
    Variables: 
    ls_scene = xarray for one Landsat scene
    
    Outputs:
    mod_scene = xarray of MODIS SST image coincident in time with the Landsat scene
    granules[ind]['umm']['GranuleUR'] = modis file name
    min_time = the time difference between the Landsat image acquisition and chosen MODIS image
    
    **not done, Differences from NLSST: 0.0 used as percent_dif requiring 100% overlap between MODIS and Landsat here since the subset area is so small
    '''
    
    mbox = (lonboundsC[0],latboundsC[0],lonboundsC[1],latboundsC[1]) #east, south,west,north

    # Construct a polygon to select a best fit MODIS image based on overlap
    # Using the entire Landsat image
    ls_scene_reproj = ls_scene.rio.reproject("EPSG:4326")
    xmin,xmax,ymin,ymax = ls_scene_reproj.x.values[0],ls_scene_reproj.x.values[-1],ls_scene_reproj.y.values[0],ls_scene_reproj.y.values[-1]
    lsatpoly = Polygon([(xmin,ymin),(xmin,ymax),(xmax,ymax),(xmax,ymin),(xmin,ymin)])
    
    # Get date/time for Landsat image and search for corresponding MODIS imagery  
    ls_time = pd.to_datetime(ls_scene.time.values)
    calc_dt = datetime.strptime(ls_time.strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
    start_dt = (calc_dt + timedelta(days=-0.5)).strftime('%Y-%m-%d %H:%M:%S')
    end_dt = (calc_dt + timedelta(days=0.5)).strftime('%Y-%m-%d %H:%M:%S')
    
    # Gather all files from search location from Terra and Aqua for the same day as the Landsat image
    granules = earthaccess.search_data(
        short_name='MODIS_T-JPL-L2P-v2019.0',
        bounding_box=mbox,
        # Day of a landsat scene to day after - searches day of only
        temporal=(start_dt,end_dt)
    )
    granules2 = earthaccess.search_data(
        short_name='MODIS_A-JPL-L2P-v2019.0', #MODIS_AQUA_L3_SST_THERMAL_DAILY_4KM_NIGHTTIME_V2019.0
        bounding_box=mbox,
        # Day of a landsat scene to day after - searches day of only
        temporal=(start_dt,end_dt)
    )
    granules = granules + granules2
    print (f'{len(granules)} TOTAL MODIS granules')

    # Accept only MODIS granules that overlap at least a perscribed amount with Landsat, in this case 100% => percent_dif=0.0
    best_grans = []
    for granule in granules:
        try:
            granule['umm']['SpatialExtent']['HorizontalSpatialDomain']['Geometry']['GPolygons']
        except Exception as error:
            print(error)
            continue
            # Would love to raise an exception for a valueerror except for GEOSError
        for num in range(len(granule['umm']['SpatialExtent']['HorizontalSpatialDomain']['Geometry']['GPolygons'])):
            try:
                # Extract points, make into a polygon
                map_points = [(xi['Longitude'],xi['Latitude']) for xi in granule['umm']['SpatialExtent']['HorizontalSpatialDomain']['Geometry']['GPolygons'][num]['Boundary']['Points']]
                pgon = Polygon(map_points)
                percent_dif = lsatpoly.difference(pgon).area/lsatpoly.area
                # If the polygon covers the landsat area, check to make sure it doesn't cross the international date line with a messed up polygon (these are searched wrong in earthaccess so probably need adjustment there)
                if percent_dif == 0.0:
                    if crosses_idl(map_points):
                        print (f'A granule has messed up polygon that likely crosses the International DateLine')
                    else:
                        best_grans.append(granule)
                        continue
            except Exception as error:
                print(error)
                # Would love to raise an exception for a valueerror except for GEOSError
    print(f'{len(best_grans)} remaining MODIS granules')

    # Find MODIS image closest in time to each Landsat image
    # Make Landsat datetime timezone aware (UTC)
    Mdates = [pd.to_datetime(granule['umm']['TemporalExtent']['RangeDateTime']['BeginningDateTime']) for granule in best_grans]
    ind = Mdates.index(min( Mdates, key=lambda x: abs(x - pytz.utc.localize(ls_time))))
    time_dif = abs(Mdates[ind] - pytz.utc.localize(pd.to_datetime(ls_time)))
    print(f'Time difference between MODIS and Landsat: {time_dif}')

    mod_scene = xr.open_dataset(earthaccess.open(best_grans[ind:ind+1])[0])
    mod_scene = mod_scene.rio.write_crs("epsg:4326", inplace=True) 
    
    return mod_scene, granules[ind]['umm']['GranuleUR'],time_dif

##########################  

# Notes for changes - MODISlookup2 doesn't need to output lat/lon, but if want to do the check, 
# can take the lat/lon check out of aligne and do it in 
def get_wv(ls_scene,mod07,spacing,param,scene,interp=0):
    '''
    Aligns and resamples MODIS water vapor data to match the spatial resolution and 
    alignment of a given Landsat scene. The function optionally applies interpolation 
    to improve the data quality.

    Parameters:
    ls_scene (xarray.Dataset): The Landsat scene dataset containing spatial coordinates.
    mod07 (xarray.Dataset): The MODIS dataset containing water vapor data and coordinates (MOD/MDY07).
    spacing (list): Desired spatial resolution (y, x) for alignment with MODIS data in meters.
    param (str): Parameter name for the desired dataset within the MODIS file.
    scene: 
    interp (int): Controls interpolation mode - 0 for none, 1 for bicubic kriging interpolation.

    Returns:
    WV_xr (xarray.DataArray): The processed xarray data array containing the Landsat-aligned 
                              and resampled water vapor data from MODIS.

    Note:
    This function is also very similar to get_sst used in LandsatCalibration; any changes here should consider potential 
    impacts there - may need to be copied/generalized.

    The function performs several key operations:
    1. Defines the bounding box for the Landsat scene based on its spatial coordinates.
    2. Extracts the relevant water vapor data from the MODIS dataset using the specified parameter key.
    3. Validates the geographic coordinate ranges (latitude and longitude) of the MODIS data.
    4. Applies PyGMT interpolation if requested to generate a smoother water vapor data surface.
    5. Utilizes a lookup function to align MODIS data indices with the Landsat grid based on the specified spatial resolution.
    6. Aligns and resamples the MODIS data to match the Landsat scene's grid and spatial resolution.
    7. Adjusts the coordinate system of the output to ensure compatibility with further processing or analysis.

    Difference: no bicubic spline interpolation in LsatCalib during the upsampling, don't set new indexes at the end
    '''
    # Read in desired variables
    ULX = ls_scene.x[0] 
    ULY = ls_scene.y[0]  
    LRX = ls_scene.x[-1] 
    LRY = ls_scene.y[-1] 
    box = [ULX,LRX,ULY,LRY]
    
    #Extract desired datasets from MODIS file from lookup key (automatically scaled by xarray so no need to do it here)
    data = mod07[param].values
    lat, lon = mod07.Latitude, mod07.Longitude
    #data.attributes()

    # Test lat is in correct range
    if ~((lat <= 90) & (lat >= -90)).all():
        print('MODIS latitude not between -90 and 90')
    # Test lon is in correct range
    if ~((lon <= 180) & (lon >= -180)).all():
        print('MODIS longitude not between -180 and 180')

    # ***Need to use climatology to retrieve quantile data from this area
    
    # # Get rid of low outliers from over ice, cutoff for 98.5%
    # outlier = np.quantile(data[np.isfinite(data)],0.015) #0.015
    # mask2 = np.ones(data.shape)
    # mask2[data<outlier] = np.nan
    # data = np.around(mask2*data,decimals=5)

    # Interpolate using PyGMT
    if interp==1:  
        grid = interpMOD(data,lat,lon)
        
        # Produce indicies for aligning MODIS pixel subset to match Landsat image at 4000m (or 300)resolution
        indiciesMOD,lines,samples,lat,lon = MODISlookup(mod07,ls_scene,box,spacing,scene,interpgrid=grid)
        data = grid.values
        
    else:
        # Produce indicies for aligning MODIS pixel subset to match Landsat image at 4000m (or 300)resolution
        indiciesMOD,lines,samples,lat,lon = MODISlookup(mod07,ls_scene,box,spacing,scene)

    # Align and resample MODIS WV to Landsat at indicated spacing with correct axes
    dataOutWV_xr = alignMODIS(data,lat,lon,param,indiciesMOD,lines,samples,mod07,ls_scene,spacing)
    
    # # Resample WV to Landsat resolution and interpolate with B-spline
    # # Need to use 0.1k (this samples at .1 of the grid)
    # # Output of shape fits and need to adjust x and y coords cuz are wrong
    ups_factor = 30/spacing[0]
    WV_upsample = pygmt.grdsample(grid=dataOutWV_xr, spacing=f'{ups_factor}k', interpolation='c')

    # Put into Xarray
    # Sometimes spacing works properly with -1 and sometimes not
    latnew = np.arange(dataOutWV_xr.latitude[0],dataOutWV_xr.latitude[-1]+1,(dataOutWV_xr.latitude[-1]-dataOutWV_xr.latitude[0])/(WV_upsample.shape[0]-1))
    if (WV_upsample.shape[0]!=latnew.shape[0]):
        latnew = np.arange(dataOutWV_xr.latitude[0],dataOutWV_xr.latitude[-1]+1,(dataOutWV_xr.latitude[-1]-dataOutWV_xr.latitude[0])/(WV_upsample.shape[0]))

    # Put into Xarray
    latnew = ls_scene.y[:WV_upsample.shape[0]].values
    lonnew = ls_scene.x[:WV_upsample.shape[1]].values
    if dataOutWV_xr.latitude[0]!=latnew[0]:
        print('Aligned y dim needs to start with the same coordinate as ls_scene')
    if dataOutWV_xr.longitude[0]!=lonnew[0]:
        print('Aligned x dim needs to start with the same coordinate as ls_scene')
    
    WV_xr = xr.DataArray(WV_upsample,name='SST',dims=["y","x"], coords={"latitude": (["y"],latnew), "longitude": (["x"],lonnew)})

    WV_xr = WV_xr.rio.write_crs("epsg:3031", inplace=True)
    WV_xr = WV_xr.rename({'longitude':'x','latitude':'y'})
    
    return WV_xr

##########################  

def get_sst(ls_scene,mod07,spacing,param):
    '''
    ***This is copied in LandsatCalibration, modifications have been made but some may tranfer
    
    Create MODIS files aligned and subsampled to Landsat
    
    Variables:
    ls_scene = xarray dataset of a Landsat scene
    mod07 = xarray datarray with MODIS L2 SST data
    spacing = list of desired spatial resolution of output data from the alignment of MODIS to Landsat in y and x (e.g.,[300,-300])
    param = string for desired dataset from MODIS file
    
    Output:
    WV_xr = xarray dataarray of Landsat aligned and upsampled modis data from desired dataset
    
    Differences from NLSST: scene is not a parameter (used for test_gridcoords), SST gets extracted differently into data/lat/lon
    
    '''
    # Read in desired variables and paths
    
    uniqWV = []

    ULX = ls_scene.x[0] 
    ULY = ls_scene.y[0]
    LRX = ls_scene.x[-1]
    LRY = ls_scene.y[-1] 
    box = [ULX,LRX,ULY,LRY]
    
    #Extract desired datasets from MODIS file
    if param == 'sea_surface_temperature': 
        data = mod07[0,:,:]
        lat, lon = mod07.lat, mod07.lon
    else: 
        data = mod07[param].values
        lat, lon = mod07.Latitude, mod07.Longitude    

    # Produce indicies for aligning MODIS pixel subset to match Landsat image at 4000m (or 300)resolution
    indiciesMOD,lines,samples = MODISslookup(mod07,ls_scene,box,spacing)

    # Align MODIS SST to Landsat on slightly upsampled grid # have the option to output `uniqImgWV` if want to know range of data
    dataOutWV_xr = alignMODIS(data,lat,lon,param,indiciesMOD,lines,samples,mod07,ls_scene,spacing)

    # Resample MODIS to Landsat resolution and interpolate with B-spline
    # Output of shape fits and need to adjust x and y coords cuz are wrong
    ups_factor = 30/spacing[0]
    WV_upsample = pygmt.grdsample(grid=dataOutWV_xr, spacing=f'{ups_factor}k') # ,interpolation='b' if prefer to interpolate with bspline but don't think it is useful here

    # Put into Xarray
    latnew = ls_scene.y[:WV_upsample.shape[0]].values
    lonnew = ls_scene.x[:WV_upsample.shape[1]].values
    if dataOutWV_xr.latitude[0]!=latnew[0]:
        print('Aligned y dim needs to start with the same coordinate as ls_scene')
    if dataOutWV_xr.longitude[0]!=lonnew[0]:
        print('Aligned x dim needs to start with the same coordinate as ls_scene')
    
    WV_xr = xr.DataArray(WV_upsample,name='SST',dims=["y","x"], coords={"latitude": (["y"],latnew), "longitude": (["x"],lonnew)})
    WV_xr = WV_xr.rio.write_crs("epsg:3031", inplace=True)
    WV_xr = WV_xr.rename({'longitude':'x','latitude':'y'})
    WV_xr = WV_xr.set_index(x='x')
    WV_xr = WV_xr.set_index(y='y')
    
    return WV_xr
           

##########################
            
def interpMOD(data,lat,lon):
    """
    Interpolate spatial water vapor data using PyGMT.

    This function takes arrays of water vapor data along with corresponding latitude and longitude values,
    performs interpolation to fill in gaps in the data, and produces a continuous surface representation of water vapor.

    Args:
        data (numpy.ndarray): 2D array of water vapor measurements.
        lat (numpy.ndarray): 2D array of latitude values corresponding to `data`.
        lon (numpy.ndarray): 2D array of longitude values corresponding to `data`.

    Returns:
        grid (xarray.DataArray): A PyGMT grid object representing the interpolated surface of water vapor data.
    """
    
    # Interpolate using PyGMT
    # Extract necessary data into Pandas DataFrame (required for PyGMT)
    df = pd.DataFrame({
        'longitude': lon.values.flatten(), # Flatten to convert from 2D to 1D array
        'latitude': lat.values.flatten(),
        'water_vapor': data.flatten() # Actual data values to be interpolated
    })

    # Remove missing or NaN values from DataFrame, as `surface` cannot handle them
    df = df.dropna(subset=['water_vapor'])

    # Determine the geographical extent for the interpolation based on the provided data points.
    # This is necessary to define the spatial domain over which PyGMT will perform interpolation.
    # [xmin, xmax, ymin, ymax] - made this the full image
    region = [df.longitude.min(), df.longitude.max(), df.latitude.min(), df.latitude.max()]
    # Alternatively, if the region is predefined (e.g., from metadata), it can be set directly.

    # Use PyGMT to interpolate the data. PyGMT requires a session context to manage memory and configuration
    # settings efficiently during its operations.
    with pygmt.clib.Session() as session:
        # Perform grid-based surface interpolation.
        # The `data` parameter takes longitude, latitude, and water vapor as a NumPy array.
        # `region` specifies the geographical extent.
        # `spacing` sets the resolution of the output grid, here 0.3 km for high resolution.
        # `tension` controls the stiffness of the interpolating surface. A value of 0.25 gives a balance between
        # fitting the data closely and producing a smooth surface.
        grid = pygmt.surface(
            data=df[['longitude', 'latitude', 'water_vapor']].to_numpy(),  # Input data as NumPy array
            region=region,  
            spacing=['0.15k','0.05k'],  # f'0.3k'
            tension=0.95,  
        )   
    
    return grid

##########################

def MODISlookup(mod07,ls_scene,box,spacing,scene,interpgrid=None):
    '''
    Look up indices for aligning MODIS product to the Landsat grid
    # Modified from http://stackoverflow.com/questions/2922532/obtain-latitude-and-longitude-from-a-geotiff-file 
    # and Shane Grigsby

    Variables:    
    mod07 = xarray with MODIS data with crs 4326 assigned
    ls_scene =  Landsat xarray DataArray
    box = list with [left easting,right easting,top northing,bottom northing]
    spacing = desired pixel size for extraction, list of [east/west, north/south] 
          (recommend choosing a number that allows for fast calculations and even division by 30)
    scene = 
    interpgrid = xarray of mod07 data that has been through interpolation in PyGMT (optional)

    Output:
    indiciesMOD = indicies used to project MODIS pixels to match Landsat pixels
    lines = number of lines in Landsat file/MODIS output shape
    samples = number of samples in Landsat file/MODIS output shape
    lon,lat = 2D lon and lat coordinates for grid
    '''
    test_threshold = 5
    
    if interpgrid is None:
        lat, lon = mod07.Latitude.values, mod07.Longitude.values
    else:
        lat, lon = interpgrid.lat, interpgrid.lon
        lon, lat = np.meshgrid(lon,lat)

    # Test lat is in correct range
    if ~((lat <= 90) & (lat >= -90)).all():
        print('MODIS latitude not between -90 and 90')
    # Test lon is in correct range
    if ~((lon <= 180) & (lon >= -180)).all():
        print('MODIS longitude not between -180 and 180')

    # Get the existing coordinate system
    old_cs = ls_scene.rio.crs # 'epsg:3031'
    new_cs = mod07.rio.crs # 'epsg:4326'

    # Create a transform object to convert between coordinate systems
    inProj = Proj(init=old_cs)
    outProj = Proj(init=new_cs)

    # Parse coordinates and spacing to different variables
    west,east,north,south = box
    ewspace,nsspace = spacing

    # Setting up grid, x coord from here to here at this spacing, mesh grid makes 2D
    samples = len(np.r_[west:east+1:ewspace])
    lines = len(np.r_[north:south-1:nsspace])#ns space is -300, could also do 30 instead of 300, but would just have duplicate pixels
    if lines==0:
        lines = len(np.r_[south:north-1:nsspace])

    # x1, y1 = np.meshgrid(np.r_[west:east:ewspace],np.r_[north:south:nsspace]) # offset by 1 meter to preserve shape
    ewdnsamp = int(spacing[0]/30)
    nsdnsamp = int(spacing[1]/30)
    
    # Set up coarser sampling and check to make sure is in the same orientation as the original Landsat grid
    xresamp = ls_scene.x.isel(x=slice(None, None, ewdnsamp)).values
    if xresamp[0]!=ls_scene.x.values[0]:
        xresamp = ls_scene.x.isel(x=slice(None, None, -ewdnsamp)).values
        # print('x resample reversed')
    yresamp = ls_scene.y.isel(y=slice(None, None, nsdnsamp)).values
    if yresamp[0]!=ls_scene.y.values[0]:
        yresamp = ls_scene.y.isel(y=slice(None, None, -nsdnsamp)).values
        # print('y resample reversed')
    x1, y1 = np.meshgrid(xresamp,yresamp)
    LScoords = np.vstack([x1.ravel(),y1.ravel()]).T
    if (LScoords[0,0]!=ls_scene.x.values[0]) |  (LScoords[0,1]!=ls_scene.y.values[0]):
        raise Exception('Landsat coordinates do not match expected during MODIS lookup')

    # Ravel so ND can lookup easily
    # Convert from LS map coords to lat lon --> x = lon, y = lat (usually?)

    ###Make into test
    # Test that reprojection is working correctly on first and last grid point using round-trip transformation
    xs1, ys1 =  transform(inProj,outProj,LScoords[0,0], LScoords[0,1], radians=True, always_xy=True)
    xsl1, ysl1 =  transform(outProj,inProj,xs1, ys1, radians=True, always_xy=True)
    if np.linalg.norm(np.array([xsl1, ysl1]) - LScoords[0,:]) > test_threshold:
        print(f"Round-trip transformation error for point {LScoords[0,:]}, {np.linalg.norm(np.array([xsl1, ysl1]) - LScoords[0,:])}")
    else:
        # If passes, run on entire grid
        xs, ys =  transform(inProj,outProj,LScoords[:,0], LScoords[:,1], radians=True, always_xy=True)
    ###
    
    # Produce landsat reprojected to lat/lon and ensure lat is in 0 column
    # Test: landsat data is in correct orientation as long as lat is in col 0 and lon in col 1
    grid_coords = test_gridcoords(xs,ys,scene)

    # Test that lines and samples match grid_coords
    if len(grid_coords) != lines*samples:
        raise Exception(f'Size of grid coordinates do not match low resolution Landsat dims: {len(grid_coords)} vs. {lines*samples}. Check that spacing is negative for y')
    MODIS_coords = np.vstack([lat.ravel(),lon.ravel()]).T
    MODIS_coords *= np.pi / 180. # to radians

    # Build lookup, haversine = calc dist between lat,lon pairs so can do nearest neighbor on sphere - if did utm it would be planar
    MOD_Ball = BallTree(MODIS_coords,metric='haversine') #sklearn library
    distanceMOD, indiciesMOD= MOD_Ball.query(grid_coords, dualtree=True, breadth_first=True)
        
    return indiciesMOD,lines,samples,lat,lon

##########################           

def MODISsstlookup (mod07,ls_scene,box,spacing):
    '''
    Look up atmospheric consituents from MODIS product for each Landsat pixel
    # Modified from http://stackoverflow.com/questions/2922532/obtain-latitude-and-longitude-from-a-geotiff-file 
    # and Shane Grigsby

    Variables:    
    mod07 = xarray with MODIS data with crs 4326 assigned
    ls_scene =  Landsat xarray DataArray
    box = list with [left easting,right easting,top northing,bottom northing]
    spacing = desired pixel size for extraction, list of [east/west, north/south] 
          (recommend choosing a number that allows for fast calculations and even division by 30)

    Output:
    indiciesMOD = indicies used to project MODIS pixels to match Landsat pixels
    lines = number of lines in Landsat file/MODIS output shape
    samples = number of samples in Landsat file/MODIS output shape
    x1,y1 = x and y coordinates for grid
    
    Differences from NLSST: lat/lon variables named differently in SST vs WV files, no interpolation,
    test_gridcoords does not use `scene`, don't need to output lat/lon because do not interpolate and make 
    new ones
    
    '''
    test_threshold = 5 
    
    lat, lon = mod07.lat, mod07.lon # Different for SST vs WV
    
    # Test lat is in correct range
    if ~((lat <= 90) & (lat >= -90)).all():
        print('MODIS latitude not between -90 and 90')
    # Test lon is in correct range
    if ~((lon <= 180) & (lon >= -180)).all():
        print('MODIS longitude not between -180 and 180')

    # Get the existing coordinate system
    old_cs = ls_scene.rio.crs # 'epsg:3031'
    new_cs = mod07.rio.crs # 'epsg:4326'

    # Create a transform object to convert between coordinate systems
    inProj = Proj(init=old_cs)
    outProj = Proj(init=new_cs)

    # Parse coordinates and spacing to different variables
    west,east,north,south = box
    ewspace,nsspace = spacing

    # Setting up grid, x coord from here to here at this spacing, mesh grid makes 2D
    samples = len(np.r_[west:east+1:ewspace])
    lines = len(np.r_[north:south-1:nsspace])#ns space is -300, could also do 30 instead of 300, but would just have duplicate pixels
    if lines==0:
        lines = len(np.r_[south:north-1:nsspace])
        
    # x1, y1 = np.meshgrid(np.r_[west:east:ewspace],np.r_[north:south:nsspace]) # offset by 1 meter to preserve shape
    ewdnsamp = int(spacing[0]/30)
    nsdnsamp = int(spacing[1]/30)

    # Set up coarser sampling and check to make sure is in the same orientation as the original Landsat grid
    xresamp = ls_scene.x.isel(x=slice(None, None, ewdnsamp)).values
    if xresamp[0]!=ls_scene.x.values[0]:
        xresamp = ls_scene.x.isel(x=slice(None, None, -ewdnsamp)).values
        
    yresamp = ls_scene.y.isel(y=slice(None, None, nsdnsamp)).values
    if yresamp[0]!=ls_scene.y.values[0]:
        yresamp = ls_scene.y.isel(y=slice(None, None, -nsdnsamp)).values

    x1, y1 = np.meshgrid(xresamp,yresamp)
    LScoords = np.vstack([x1.ravel(),y1.ravel()]).T
    if (LScoords[0,0]!=ls_scene.x.values[0]) |  (LScoords[0,1]!=ls_scene.y.values[0]):
        raise Exception('Landsat coordinates do not match expected during MODIS lookup')

    # Ravel so ND can lookup easily
    # Convert from LS map coords to lat lon --> x = lon, y = lat (usually?)
    # Test that reprojection is working correctly on first and last grid point using round-trip transformation
    xs1, ys1 =  transform(inProj,outProj,LScoords[0,0], LScoords[0,1], radians=True, always_xy=True)
    xsl1, ysl1 =  transform(outProj,inProj,xs1, ys1, radians=True, always_xy=True)
    if np.linalg.norm(np.array([xsl1, ysl1]) - LScoords[0,:]) > test_threshold:
        print(f"Round-trip transformation error for point {LScoords[0,:]}, {np.linalg.norm(np.array([xsl1, ysl1]) - LScoords[0,:])}")
    else:
        # If passes, run on entire grid
        xs, ys =  transform(inProj,outProj,LScoords[:,0], LScoords[:,1], radians=True, always_xy=True)

    # Produce landsat reprojected to lat/lon and ensure lat is in 0 column
    grid_coords = np.vstack([ys.ravel(),xs.ravel()]).T
    # Test that lines and samples match grid_coords
    if len(grid_coords) != lines*samples:
        raise Exception(f'Size of grid coordinates do not match low resolution Landsat dims: {len(grid_coords)} vs. {lines*samples}. Check that spacing is negative for y')
    MODIS_coords = np.vstack([lat.values.ravel(),lon.values.ravel()]).T
    MODIS_coords *= np.pi / 180. # to radians
    
    # Build lookup, haversine = calc dist between lat,lon pairs so can do nearest neighbor on sphere - if did utm it would be planar
    MOD_Ball = BallTree(MODIS_coords,metric='haversine') #sklearn library
    distanceMOD, indiciesMOD= MOD_Ball.query(grid_coords, dualtree=True, breadth_first=True)
        
    return indiciesMOD,lines,samples

##########################

def test_gridcoords(xs,ys,scene):
    '''
    Test to ensure grid lat and lon are not swapped during reprojection and output grid coordinates
    that have been raveled and stacked for input into BallTree
    
    Variables:
    xs = 1D radians representing longitude 
    ys = 1D radians representing latitude
    scene = catalog item for landsat image
    
    Output:
    grid_coords = two columns of x/y radian pairs representing lon/lat
    '''
    
    # Convert radians to lat/lon
    x_check = xs * 180. / np.pi
    y_check = ys * 180. / np.pi
    
    # We know lat is ys and lon is xs if this is true so goes in 0 column position to match MODIS
    if ((-90 <= y_check) & (y_check <= -60)).all() & ~((-90 <= x_check) & (x_check <= -60)).all():
        grid_coords = np.vstack([ys.ravel(),xs.ravel()]).T # note y / x switch (i.e., lat long convention)
        # print('Latitude in proper position')

    # A small subset of data have lat and lon that falls between -60 and -90 so test if the landsat metadata confirms that
    elif ((-90 <= y_check) & (y_check <= -60)).all():
        llons = np.array((float(scene.metadata['bbox'][0]), float(scene.metadata['bbox'][2])))
        # ys is latitude if true here
        if ((-90 <= llons) & (llons <= -60)).all():
            grid_coords = np.vstack([ys.ravel(),xs.ravel()]).T # note y / x switch (i.e., lat long convention)
            # print('Latitude in proper position')
        # xs is latitude if not and goes in 0 column position
        else:
            grid_coords = np.vstack([xs.ravel(),ys.ravel()]).T 
            print('Latitude in wrong position')

    # Otherwise xs is latitude and goes in 0 column position
    else:
        grid_coords = np.vstack([xs.ravel(),ys.ravel()]).T
        print('Latitude in wrong position')
    
    return grid_coords

##########################

def test_gridcoords_calib(xs,ys):
    '''
    Test to ensure grid lat and lon are not swapped during reprojection and output grid coordinates
    that have been raveled and stacked for input into BallTree. There is some uncertainty only when the image is
    taken between -60 and -90 longitude because lat and lon can have the same values.
    
    Variables:
    xs = 1D radians representing longitude 
    ys = 1D radians representing latitude
    
    Output:
    grid_coords = two columns of x/y radian pairs representing lon/lat
    
    Differences from NLSST: elif is different than NLSST pipeline
    '''
    
    # Convert radians to lat/lon
    x_check = xs * 180. / np.pi
    y_check = ys * 180. / np.pi
    
    # We know lat is ys and lon is xs if this is true so goes in 0 column position to match MODIS
    if ((-90 <= y_check) & (y_check <= -60)).all() & ~((-90 <= x_check) & (x_check <= -60)).all():
        grid_coords = np.vstack([ys.ravel(),xs.ravel()]).T # note y / x switch (i.e., lat long convention)
        print('Latitude in proper position')

    # A small subset of data have lat and lon that falls between -60 and -90 so test if the landsat metadata confirms that
    elif ((-90 <= y_check) & (y_check <= -60)).all():
        # xs is latitude if not and goes in 0 column position
        grid_coords = np.vstack([ys.ravel(),xs.ravel()]).T 
        print('Latitude in uncertain position, may be incorrect')

    # Otherwise xs is latitude and goes in 0 column position
    else:
        grid_coords = np.vstack([xs.ravel(),ys.ravel()]).T
        print('Latitude in wrong position')
    
    return grid_coords

##########################

def alignMODIS(data,lat,lon,param,indiciesMOD,lines,samples,mod07,ls_scene,spacing):
    '''
    Align MODIS image to Landsat and resample at indicated spacing
    
    Variables:
    data =
    lat = 
    lon = 
    param =
    indiciesMOD =
    lines = 
    samples =
    mod07 = 
    ls_scene =
    spacing =
    
    Output:
    dataOut_xr = 
    
    Not currently set, but can also output: 
    uniqImg = uniq MODIS atm values within area of Landsat image
    '''
    test_threshold = 5
    
    # Check to ensure lat/lon and data have compatible shapes
    if (np.shape(lat)== np.shape(lon)== np.shape(data))==False:
        raise Exception("Error in creating indicies, lat/lon and data shapes do not match")
        
    # Extract MODIS data into Landsat grid and gather unique data values
    dataOut,uniqWV = uniqueMODIS(data,param,indiciesMOD,lines,samples)
    
    # Check grid directionality and create matching x/y for new grid
    # Define the source and target coordinate reference systems (CRS)
    src_crs = mod07.rio.crs #'epsg:4326'  MODIS
    target_crs = ls_scene.rio.crs #crs[6:] # 'epsg:3031' Landsat

    # Create a PyProj transformer
    transformer = pyproj.Transformer.from_crs(src_crs, target_crs, always_xy=True)
    transformer_test = pyproj.Transformer.from_crs(target_crs, src_crs, always_xy=True)

    # Test that reprojection is working correctly on first and last modis grid point
    xm1,xm2 = lon[0,0],lon[-1,-1]
    ym1,ym2 = lat[0,0],lat[-1,-1]
    xx,yy = [xm1,xm2], [ym1,ym2]
    xs1, ys1 =  transformer.transform(xx,yy)
    xsl1, ysl1 = transformer_test.transform(xs1, ys1)
    for i,n in enumerate(xsl1):
        if np.linalg.norm(np.array([xsl1[i], ysl1[i]]) - [xx[i],yy[i]]) > test_threshold:
            print(f"Round-trip transformation error for {sceneid}, {np.linalg.norm(np.array([xsl1[i], ysl1[i]]) - xx[i],yy[i])}")
    
    # Spacing to create x and y parameters at the correct spacing
    redy = int(abs(spacing[0]/30))
    redx = int(abs(spacing[1]/30))

    # From SST# Set up coarser sampling and check to make sure is in the same orientation as the original Landsat grid
    # xgrid = ls_scene.x.isel(x=slice(None, None, redx)).values
    # if xgrid[0]!=ls_scene.x.values[0]:
    #     xgrid = ls_scene.x.isel(x=slice(None, None, -redx)).values
    # ygrid = ls_scene.y.isel(y=slice(None, None, redy)).values
    # if ygrid[0]!=ls_scene.y.values[0]:
    #     ygrid = ls_scene.y.isel(y=slice(None, None, -redy)).values
    # if (xgrid[0]!=ls_scene.x.values[0]) |  (ygrid[0]!=ls_scene.y.values[0]):
    #     raise Exception('Landsat coordinates do not match expected during MODIS lookup')

    #From LandsatCalib
    # Set up coarser sampling grid to match spacing and check to make sure is in the same orientation as the original Landsat grid
    xgrid = ls_scene.x.values[0::red_x]
    if len(xgrid)==1:
        xgrid = ls_scene.x.values[0::-red_x]
    if xgrid[0]!=ls_scene.x.values[0]:
        xgrid = np.flip(xgrid)
        print ('Align x flip')
    ygrid = ls_scene.y.values[0::red_y]
    if len(ygrid)==1:
        ygrid = ls_scene.y.values[0::-red_y]
    if ygrid[0]!=ls_scene.y.values[0]:
        ygrid = np.flip(ygrid)
        print ('Align y flip')
    if (xgrid[0]!=ls_scene.x.values[0]) |  (ygrid[0]!=ls_scene.y.values[0]):
        raise Exception(f'Landsat coordinates do not match expected during MODIS align')
    
    # Create xarray from numpy array
    dataOut_xr = xr.DataArray(dataOut,name='SST',dims=["y","x"], coords={"latitude": (["y"],ygrid), "longitude": (["x"],xgrid)})
    
    return dataOut_xr

##########################

def uniqueMODIS(data,param,indiciesMOD,lines,samples):
    '''
    Extracts data values and unique values from desired MODIS dataset that corresponds to Landsat file
    No scaling needed - xarray automatically scales for you
    # Modified from http://stackoverflow.com/questions/2922532/obtain-latitude-and-longitude-from-a-geotiff-file
    
    Variables: 
    data = array with MOD07 data in crs 4326 assigned 
    param =  string for desired dataset from MODIS file
    indiciesMOD = indicies output for neighest neighbor query from MODIS to Landsat coordinates
    lines = number of lines in Landsat file/MODIS output shape
    samples = number of samples in Landsat file/MODIS output shape
    
    Output:
    dataOut = MODIS atm image subset and aligned to Landsat image pixels
    uniq = uniq MODIS atm values within area of Landsat image
    #counts = count for each unique value in subset
    '''
    # Convert from K to C
    KtoC = -273.15
    
    # Reproject data from MODIS into corresponding postions for Landsat pixels for the desired dataset
    # Remove unrealistic data/outliers
    # Scaling has already been automatically done by xarray
    if param == 'sea_surface_temperature':  
        #Extract desired datasets from MODIS file from lookup key
        # Move to adjusted grid and rescale data
        dataOut = np.reshape(np.array(data.values.ravel())[indiciesMOD],(lines,samples)) + KtoC #* # to scale?
        dataOut[dataOut < -3.5] = np.nan
    elif param == 'Water_Vapor':
        dataOut = np.reshape(np.array(data.ravel())[indiciesMOD],(lines,samples))
        dataOut[dataOut < 0] = np.nan
        MODimg = np.array(data)
        MODimg[MODimg < 0] = np.nan
    elif param == 'Total_Ozone':
        dataOut = np.reshape(np.array(data.ravel())[indiciesMOD],(lines,samples))
        dataOut[dataOut < 225] = np.nan
        dataOut[dataOut > 430] = np.nan
        MODimg = np.array(data)
        MODimg[MODimg < 0] = np.nan

    # Get unique values for datasets within Landsat extent
    #uniq, inverse, counts= np.unique(dataOut, return_inverse=True, return_counts=True)
    uniq = set(dataOut[np.isfinite(dataOut)])
    
    return dataOut,uniq # Can also output MODimg and inverse and counts if desired


# +
# Functions for deriving SST retrieval coefficients
'''
These functions help to derive the SST monthly correction coefficients
prep_retrieval prepares the inputs for running the multiple regression that determines the coefficients, including
converting ERA-5 specific humidity data to total column water vapor in spec_hu_to_tcwv. Derive retrieval then takes
the inputs and runs an OLS multiple regression to derive the coefficients.

Functions to search, open, and analyze Landsat scenes.
Search_stac finds the Landsat scene based on user parameters, 
plot_search plots the locations of the landsat scenes from the search,
landsat_to_xarray takes one of those scenes and puts all bands into an xarray,
and create_masks produces cloud/ice/water masks for the scene. Subset_img 
subsets a landsat scene with coordinates that have been reprojected from lat/lon
and may be flipped in which is larger in the pair. Lsat_reproj can be used to reproject
while ensuring x and y pairs don't get flipped (common converting between espg 3031 and wgs84.
'''
def prep_retrieval(atmpath,prefix,spec_hu_file):
    '''
    Create the inputs for the SST algorithm using the atmospheric column inputs and outputs from 
    the MODTRAN model runs for Landsat. Uses specific humidity to calculate total column water vapor 
    for the retrieval multiple regression.

    Units for ERA5 specific humidity listed here under main variables: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=overview
    
    Variable:
    atmpath = directory path where MODTRAN outputs are stored (str)
    prefix = beginning of file path name for MODTRAN outputs - files created by Andy Harris (str) 
    spec_hu_file = file path for era5 input file for MODTRAN that includes specific humidity (str)
    
    Intermediates:
    modtran_lut = pandas dataframe of MODTRAN outputs
    modtran_atm = pandas dataframe of era5 atmopheric columns for input into MODTRAN
    
    Outputs:
    modtran_lut = pandas dataframe of MODTRAN outputs with total column water vapor [cm] added as 
                  a column
    '''
    
    # Open and concatenate MODTRAN outputs for SST algorithm
    # Get file paths
    modtr_list = os.listdir(atmpath)
    modtr_list = [file for file in modtr_list if file.startswith(prefix)]
    modtr_list.sort()

    # Open into pandas and concatenate
    df_list = []
    lut_cols = ['wind spd [m/s]','Surface T[K]','TOA T[K]','transmittance','jakobian']
    
    if len(modtr_list)>1:
        for mfile in modtr_list:
            df_list.append(pd.read_csv(f'{atmpath}/{mfile}', sep=' ',header=None,names=lut_cols))

        modtran_lut = pd.concat(df_list, ignore_index=True)
    else:
        mfile = modtr_list[0]
        modtran_lut = pd.read_csv(f'{atmpath}/{mfile}', sep=' ',header=None,names=lut_cols)
    
    modtran_lut['TCWV [cm]'] = np.nan
    
    # Open atm profiles for input of water vapor specific humidity
    atm_cols = ['Altitude [km]', 'pressure [hPa]', 'temp [K]', 'spec humidity [kg/kg]']
    modtran_atm = pd.read_csv(f'{atmpath}/{spec_hu_file}', sep='\t',header=None,names=atm_cols)
    
    # Run integral to get tcwv
    modtran_lut = spec_hu_to_tcwv(modtran_lut,modtran_atm)

    tcwvmin = modtran_lut['TCWV [cm]'].min()
    tcwvmax = modtran_lut['TCWV [cm]'].max()
    print (f'TCWV min: {tcwvmin}, max: {tcwvmax}')
    
    return modtran_lut

##########################

def concat_modtran_months(months,atmpath):
    # Create a list to store the DataFrame for each month in the window.
    modtran_list = []

    n = 0
    
    for mo in months:
        TCWV_input_file = atmpath / f"TCWV_{mo}.csv"
        if os.path.isfile(TCWV_input_file):
            # print(f"  Month {mo}: retrieval input exists")
            modtran = pd.read_csv(TCWV_input_file)
        else:
            spec_hu_file = f"modtran_atmprofiles_{mo}.txt"
            modtran_output_file = f"modtran_atmprofiles_{mo}.bts+tau+dbtdsst.txt"
            modtran = prep_retrieval(atmpath, modtran_output_file, spec_hu_file)
            modtran.to_csv(TCWV_input_file, index=False)
        
        # Remove rows with Surface T values of 271.46 and 271.461.
        modtran = modtran[~modtran['Surface T[K]'].isin([271.46, 271.461])]
        print(f'{mo}: {modtran.shape[0]}')
        
        modtran_list.append(modtran)

        n = n + modtran.shape[0]
    
    # Concatenate the DataFrames for the three months in the window.
    modtran_lut = pd.concat(modtran_list, ignore_index=True)

    return modtran_lut,n

##########################

def derive_coeffs(atmpath,simTOA_transformer,simWV_transformer,simT_transformer):
    # Derive retrieval coefficiencts from MODTRAN files - 3 month rolling window
    months = ['01','02','03','04','05','06','07','08','09','10','11','12']
    atmcor = {}  # To store the regression results for each middle month.
    
    # Loop over months by index so we can get the previous and next month via modulo arithmetic.
    for i, middle_month in enumerate(months):
        # Determine the rolling window months: previous, current, and next (with wrap-around)
        prev_month = months[(i - 1) % 12]
        next_month = months[(i + 1) % 12]
        window_months = [prev_month, middle_month, next_month]
        
        print(f"Processing rolling window for middle month {middle_month}")
        
        modtran_lut,_ = concat_modtran_months(window_months,atmpath)
    
        modtran_lut_norm = modtran_lut
    
        modtran_lut_norm['Surface T[K]'] = simT_transformer.transform(modtran_lut[['Surface T[K]']])
        modtran_lut_norm['TOA T[K]'] = simTOA_transformer.transform(modtran_lut[['TOA T[K]']])
        modtran_lut_norm['TCWV [cm]'] = simWV_transformer.transform(modtran_lut[['TCWV [cm]']])
        
        # Run the regression using your derive_retrieval function on the concatenated DataFrame.
        retrieval_results = derive_retrieval(modtran_lut_norm)
        a1 = np.around(retrieval_results.params.toa, 2)
        a2 = np.around(retrieval_results.params.tcwv_toa, 2)
        a3 = np.around(retrieval_results.params.Intercept, 2)
        r2 = np.around(retrieval_results.rsquared, 2)
        
        pa1 = np.around(retrieval_results.pvalues[1], 3)
        pa2 = np.around(retrieval_results.pvalues[2], 3)
        pa3 = np.around(retrieval_results.pvalues[0], 3)
        
        # Store the regression coefficients and R2 for the middle month.
        atmcor[middle_month] = {"a1": a1, "a2": a2, "a3": a3}
        print(f"Rolling for month {middle_month}: toa = {a1}, tcwv_toa = {a2}, Intercept = {a3}, R2 = {r2}")
        print(f"p-values for month {middle_month}: toa = {pa1}, tcwv_toa = {pa2}, Intercept = {pa3}")
    
    print(retrieval_results.summary())
    
    return atmcor,retrieval_results,modtran_lut_norm

##########################

def derive_retrieval(modtran_lut):
    '''
    Derive the retrieval coefficients from the atmospheric column inputs and outputs to the MODTRAN
    model runs for Landsat using multiple regression. 
    
    Variables:
    modtran_lut = pandas dataframe that includes columns for surface temperature [K], top of 
                  atmosphere brightness temperature [K], total column water vapor [cm]
    
    Outputs:
    results = multiple regression summary and derived coefficients (ak) for retrieval atmospheric correction
    '''
    # Run OLS multiple regression to derive atmospheric correction coefficients
    df_newnames = modtran_lut.rename(columns={'Surface T[K]': 'surface', 'TOA T[K]': 'toa', 'TCWV [cm]': 'tcwv'})
    df_newnames['tcwv_toa'] = df_newnames['tcwv']*df_newnames['toa']
    results = smf.ols('surface ~ toa + tcwv_toa', data=df_newnames).fit()
 
    return results

##########################

def derive_retrieval_ransac(modtran_lut, residual_threshold=1.0, max_trials=100, random_state=42):
    """
    Derive the retrieval coefficients from the atmospheric column inputs and outputs to the MODTRAN
    model runs for Landsat using RANSAC regression.
    
    Variables:
    -----------
    modtran_lut : pandas DataFrame
        A dataframe that includes columns for surface temperature [K], top of atmosphere 
        brightness temperature [K], and total column water vapor [cm]. Expected column names are:
            'Surface T[K]', 'TOA T[K]', and 'TCWV [cm]'
    
    residual_threshold : float, default 1.0
        Maximum residual for a data point to be classified as an inlier.
        
    max_trials : int, default 100
        Maximum number of iterations for the RANSAC algorithm.
        
    random_state : int, default 42
        Random seed for reproducibility.
    
    Returns:
    --------
    results : dict
        Dictionary containing the estimated coefficients with keys:
            'Intercept', 'toa', 'tcwv_toa'
    """
    # Rename columns to standard names
    df_newnames = modtran_lut.rename(columns={
        'Surface T[K]': 'surface',
        'TOA T[K]': 'toa',
        'TCWV [cm]': 'tcwv'
    })

    # Remove rows with any NaN values
    df_newnames = df_newnames.dropna()
    
    # Create the interaction term
    df_newnames['tcwv_toa'] = df_newnames['tcwv'] * df_newnames['toa']
    
    # Prepare the predictor matrix and response vector
    X = df_newnames[['toa', 'tcwv_toa']]
    y = df_newnames['surface']
    
    # Set up the RANSAC regression with a base LinearRegression estimator
    base_estimator = LinearRegression()
    ransac = RANSACRegressor(estimator=base_estimator,
                             max_trials=max_trials,
                             residual_threshold=residual_threshold,
                             random_state=random_state)
    
    # Fit the model
    ransac.fit(X, y)
    
    # Extract the fitted parameters
    # The intercept and coefficients are stored in the underlying estimator.
    params = {
        'Intercept': ransac.estimator_.intercept_,
        'toa': ransac.estimator_.coef_[0],
        'tcwv_toa': ransac.estimator_.coef_[1]
    }
    
    return params


##########################

def derive_retrieval_odr(modtran_lut):
    """
    Derive the retrieval coefficients from the atmospheric column inputs and outputs 
    to the MODTRAN model runs for Landsat using Orthogonal Distance Regression.
    
    Parameters:
    -----------
    modtran_lut : pandas DataFrame
        A dataframe that includes columns for surface temperature [K],
        top-of-atmosphere brightness temperature [K], and total column water vapor [cm].
        Expected column names:
            'Surface T[K]', 'TOA T[K]', and 'TCWV [cm]'
    
    Returns:
    --------
    results : dict
        Dictionary containing the estimated coefficients:
            'Intercept', 'toa', and 'tcwv_toa'
    """
    # Rename columns to standard names
    df_newnames = modtran_lut.rename(columns={
        'Surface T[K]': 'surface',
        'TOA T[K]': 'toa',
        'TCWV [cm]': 'tcwv'
    })
    
    # Remove any rows that contain NaNs
    df_newnames = df_newnames.dropna()
    
    # Create the interaction term
    df_newnames['tcwv_toa'] = df_newnames['tcwv'] * df_newnames['toa']
    
    # Prepare the independent variables and the dependent variable.
    # For ODR, X must be an array of shape (n_predictors, n_points)
    X = df_newnames[['toa', 'tcwv_toa']].values.T  # shape: (2, n)
    y = df_newnames['surface'].values              # shape: (n,)
    
    # Define the linear model function for ODR.
    # beta[0] is the intercept, beta[1] is the coefficient for 'toa', and beta[2] for 'tcwv_toa'
    def linear_model(beta, x):
        return beta[0] + beta[1] * x[0] + beta[2] * x[1]
    
    # Create an ODR Model object
    model = odr.Model(linear_model)
    
    # Prepare the data for ODR. (If you have measurement errors, you can pass them via sx and sy)
    data = odr.RealData(X, y)
    
    # Create an ODR instance with an initial guess for the parameters.
    # Here we use an initial guess: [0.0, 1.0, 1.0]
    odr_instance = odr.ODR(data, model, beta0=[0.0, 1.0, 1.0])
    
    # Run the ODR regression.
    out = odr_instance.run()
    
    # Return the parameters in a dictionary similar to the statsmodels output.
    results = {
        'Intercept': out.beta[0],
        'toa': out.beta[1],
        'tcwv_toa': out.beta[2],
        # Optionally, you can also return diagnostics:
        'sum_square': out.sum_square,
        'res_var': out.res_var
    }
    
    return results

##########################

def spec_hu_to_tcwv(modtran_lut, modtran_atm, atm_levels=37):
    '''
    Calculate total column water vapor by integrating across all atmospheric pressure levels
    using hydrostatic approximation.

    Output:
    modtran_lut = original dataframe with added TCWV column in [cm]
    '''
    g = 9.80665  # gravity [m/s^2]

    m = 0
    for y in tqdm(range(modtran_lut.shape[0]-1)):
        r = m + atm_levels
        df = modtran_atm.iloc[m:r]

        tcwv_pa = 0
        for i in range(1, len(df)):
            p0 = df['pressure [hPa]'].iloc[i-1] * 100  # convert to Pa
            p1 = df['pressure [hPa]'].iloc[i] * 100
            q0 = df['spec humidity [kg/kg]'].iloc[i-1]
            q1 = df['spec humidity [kg/kg]'].iloc[i]
            dq = (q0 + q1) / 2  # trapezoidal average
            dp = p1 - p0        # pressure difference
            tcwv_pa += dq * dp  # integral sum: q * dp

        # Final TCWV in kg/m^2 → mm (same numerically) → cm
        tcwv_kg_m2 = tcwv_pa / g
        tcwv_cm = tcwv_kg_m2 / 10

        modtran_lut.loc[y, 'TCWV [cm]'] = tcwv_cm
        m = r

    return modtran_lut


# +
# Functions to produce SST with atmospheric correction
'''
Functions to produce SST with atmospheric correction
apply_retrieval preps the masks and thermal data then runs the entire retrieval correction and calibration pipeline in lsatAtmCorr,
lsatAtmCorr calculates top of atmosphere brightness temperatures from thermal digital numbers data in TOA_BT and applies 
the atmospheric correction to get absolute temperatures [C] using retrieval.
'''

##########################

def TOA_BT(ls_thermal,scene):
    '''
    Calculate TOA radiance and brightness temperature using MTL json
    
    ls_thermal = xarray dataset of Landsat thermal data computed
    scene = catalog item for landsat scene
    
    Using equations from https://www.usgs.gov/landsat-missions/using-usgs-landsat-level-1-data-product
    '''
    
    # Calculate radiances using MTL data
    s3 = boto3.client("s3")

    # Extract bucket and key for json MTL file
    # Example: bucket = "usgs-landsat" ; key = "collection02/level-1/standard/oli-tirs/2019/002/113/LC08_L1GT_002113_20190206_20201016_02_T2/LC08_L1GT_002113_20190206_20201016_02_T2_MTL.json"
    s3_url = scene['MTL.json'].metadata['alternate']['s3']['href']
    bucket = s3_url.split('/')[2].strip()
    key = s3_url.split(bucket)[1].strip()[1:]

    # Get MLT data
    res = s3.get_object(Bucket=bucket, Key=key, RequestPayer="requester")
    MTL = res["Body"].read().decode("utf-8")

    # Get important constants from MTL
    ind = MTL.find('K1_CONSTANT_BAND_10')
    K1_10 = float(MTL[ind+23:ind+31])
    ind = MTL.find('K2_CONSTANT_BAND_10')
    K2_10 = float(MTL[ind+23:ind+32])
    ind = MTL.find('RADIANCE_MULT_BAND_10')
    ML10 = float(MTL[ind+25:ind+35])
    ind = MTL.find('RADIANCE_ADD_BAND_10')
    AL10 = float(MTL[ind+24:ind+31])

    # Mask no data 
    DN_masked = ls_thermal.where(ls_thermal != 0)

    # Top of Atmosphere radiance for Band 10
    Llambda = ML10 * DN_masked + AL10

    # Top of Atmosphere brightness temperature for Band 10
    T10 = K2_10 / np.log((K1_10 / Llambda) + 1)
    return T10

##########################

def retrieval(toa, wv, a1, a2, a3, simT_transformer, simTOA_transformer, simWV_transformer):
    '''
    Calculates the surface temperature.

    Variables:
    toa = calculated TOA [K] (xarray.DataArray)
    wv = total column water vapor (xarray.DataArray)
    ak = derived retrieval coefficients

    Output:
    SST = sea surface temperature [C] (xarray.DataArray)
    '''
    # Convert toa DataArray to numpy, preserving shape.
    toa_arr = toa.values
    original_toa_shape = toa_arr.shape
    # Reshape to a column vector for the transformer
    toa_norm = simTOA_transformer.transform(toa_arr.reshape(-1, 1))
    toa_norm = toa_norm.reshape(original_toa_shape)
    
    # Convert wv DataArray to numpy array and transform similarly.
    original_wv_shape = wv.shape
    wv_norm = simWV_transformer.transform(wv.reshape(-1, 1))
    wv_norm = wv_norm.reshape(original_wv_shape)
    
    # Calculate normalized SST using the retrieval coefficients and water vapor
    SST_norm = a3 + a1 * toa_norm + a2 * wv_norm * toa_norm
    
    # Inverse transform SST: again reshape as needed.
    original_sst_shape = SST_norm.shape
    SST_norm_flat = SST_norm.reshape(-1, 1)
    SST_flat = simT_transformer.inverse_transform(SST_norm_flat)
    SST = SST_flat.reshape(original_sst_shape) - 273.15
    
    # Optionally, convert the result back into an xarray.DataArray,
    # preserving the original toa coordinates and dimensions:
    SST = xr.DataArray(SST, coords=toa.coords, dims=toa.dims)
    
    return SST

##########################

def lsatAtmCorr(ls_thermal,scene,mask,modwv,a1,a2,a3,simT_transformer,simTOA_transformer,simWV_transformer):
    '''
    Applies atmospheric correction to top of atmosphere (TOA) brightness temperatures and converts to 
    absolute temperature for Landsat thermal images. Uses a derived coefficients for a non-linear
    sea surface temperature algorithm (retrieval) and water vapor from MODIS to produce the atmospheric 
    correction.

    Variables:
    ls_thermal = xarray dataset of Landsat thermal band computed 
    scene = STAC catalog item for the Landsat scene
    modwv = Water Vapor array (or other varrying parameter) from MODIS that has been 
            processed to the same dimensionality and pixel size as the Landsat image
    ak = derived retrieval coefficients 

    Previously defined:
    ak = derived retrieval coefficients

    Output: SST 2D array and GTiff of atmospheric corrected absolute temperatures [C]

    '''
    T10 = TOA_BT(ls_thermal,scene)
    T10 = mask * T10
    SST = retrieval(T10,modwv,a1,a2,a3,simT_transformer,simTOA_transformer,simWV_transformer)
    SST = SST.compute()
        
    return SST

##########################

def apply_retrieval(ls_thermal,scene,mask,WV_xr,atmcor,simT_transformer,simTOA_transformer,simWV_transformer):
    '''
    Use MODIS water vapor and landsat DN in retrieval algorithm to derive sea surface temperature.
    
    Variables:
    ls_scene = xarray dataset of Landsat scene  
    scene = STAC catalog item for the Landsat scene
    WV_xr = xarray dataarray of MODIS water vapor values matching timing of the landsat scene  
    atmcor = dictionary of derived retrieval coefficients for all months
    
    Outputs:
    SST = multiple regression summary and derived coefficients (ak) for retrieval atmospheric correction
    Also saves a cloud-optimized geotiff of SST
    '''
    try:
        wv2 = mask*WV_xr.values
        wv3 = mask*np.around(wv2,decimals=5) # usually w2 but skippping outliers for now

        means = np.nanmean(wv3)
        print (f'Mean water vapor value is: {means}, min: {np.nanmin(wv3)}, max: {np.nanmax(wv3)}')
        
        # Select appropriate atmospheric correction coefficients
        month = scene.metadata['datetime'].month
        a_mo = f'{month}'.zfill(2)
        a1 = atmcor[a_mo]['a1']
        a2 = atmcor[a_mo]['a2']
        a3 = atmcor[a_mo]['a3']
        
        # Apply atmospheric correction to get absolute temps in C
        SST = lsatAtmCorr(ls_thermal,scene,mask,wv3,a1,a2,a3,simT_transformer,simTOA_transformer,simWV_transformer)

        return SST
        
    except Exception as e: 
        print(e)
        print (f'atm correction of {ls_scene.id.values} failed')
