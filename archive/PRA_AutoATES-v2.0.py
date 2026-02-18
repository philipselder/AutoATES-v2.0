#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue October 11 09:54:00 2022
    Copyright (C) <2022>  <Håvard Boutera Toft>
    htla@nve.no

    This python script reimplements the Potential Release Area proposed 
    by Veitinger et al. (2016) and Sharp et al., (2018). The script has
    been modified to suit AutoATES v2.0 and is rewritten using Python
    libraries.

    References:
    https://github.com/jocha81/Avalanche-release
        Veitinger, J., Purves, R. S., & Sovilla, B. (2016). Potential 
    slab avalanche release area identification from estimated winter 
    terrain: a multi-scale, fuzzy logic approach. Natural Hazards and 
    Earth System Sciences, 16(10), 2211-2225.
        Sharp, A. E. A. (2018). Evaluating the Exposure of Heliskiing 
    Ski Guides to Avalanche Terrain Using a Fuzzy Logic Avalanche 
    Susceptibility Model. University of Leeds: Leeds, UK.

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

    Description of inputs and defaults.
        forest_type:    'stems', 'bav', 'pcc', 'sen2cc', and 'no_forest'
        DEM:            A raster using the GeoTiff format (int16, nodata=-9999)
        FOREST:         A raster using the GeoTiff format (int16, nodata=0)
        radius:         The radius of the windshelter function. A general recommendation is to use 60 m, so if the cell size is 10 m, the radius should be 6.
        prob:           Default is 0.5, (see Veitinger et al. 2016 for more information).
        winddir:        The prevailing wind direction (0-360). Default for AutoATES v2.0 is 0
        windtol:        The number of degrees to each side of the prevailing wind (0-180). Default for AutoATES v2.0 is 180.
        pra_thd:        The cut off value for the binary PRA output. Default for AutoATES is 0.15
        sf:             The SieveFilter removes small clusters of cells smaller than the dessignated value in the binary PRA output. I.e., sf=3 means that release areas with less than 3 cells will be made no release cell.
"""

# import standard libraries
import numpy as np
import rasterio, rasterio.mask
from osgeo import gdal
import os
from numpy.lib.stride_tricks import as_strided
from collections import deque
import sys
from datetime import datetime
import cupy as cp
from numpy.lib.stride_tricks import as_strided as np_as_strided
from cupy.lib.stride_tricks import as_strided as cp_as_strided

# --- Example
# stems (10m raster):       python PRA/PRA_AutoATES-v2.0.py stems PRA/DEM.tif PRA/FOREST.tif 6 0.5 0 180 0.15 3
# no_forest (10m raster):   python PRA/PRA_AutoATES-v2.0.py no_forest PRA/DEM.tif 6 0.5 0 180 0.15 3
# PSE - testing: .venv/Scripts/python.exe PRA/PRA_AutoATES-v2.0.py pcc PRA/AP_DEM_Clip.tif PRA/AP_Canopy.tif 60 0.5 0 180 0.15 3
def PRA(forest_type, DEM, FOREST, radius, prob, winddir, windtol, pra_thd, sf):
    
    ##########################
    # --- Check input files
    ##########################

    path = os.path.join(os.getcwd(), "PRA")
    os.makedirs(path, exist_ok=True)

    f= open("PRA/log.txt","w+")
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    f.write("Start time = {}\n".format(current_time))

    # Check if path exits
    if os.path.exists(DEM) is False:
        print("The DEM path {} does not exist".format(DEM))

    if forest_type in ['pcc', 'stems', 'bav', 'sen2cc']:
        # Check if path exits
        if os.path.exists(FOREST) is False:
            print("The forest path {} does not exist\n".format(FOREST))

        print(forest_type, DEM, FOREST, radius, prob, winddir, windtol, pra_thd, sf)
        f.write("forest_type: {}, DEM: {}, FOREST: {}, radius: {}, prob: {}, winddir: {}, windtol {}, pra_thd: {}, sf: {}\n".format(forest_type, DEM, FOREST, radius, prob, winddir, windtol, pra_thd, sf))

    if forest_type in ['no_forest']:
        print(forest_type, DEM, radius, prob, winddir, windtol, pra_thd, sf)
        f.write("forest_type: {}, DEM: {}, radius: {}, prob: {}, winddir: {}, windtol {}, pra_thd: {}, sf: {}\n".format(forest_type, DEM, radius, prob, winddir, windtol, pra_thd, sf))

    #########################
    # --- Define functions
    #########################

    def sliding_window_view(arr, window_shape, steps):
        """
        Create a sliding window view of an array.
        The window moves by `steps` in each dimension.
        """
        # Determine whether to use numpy or cupy
        xp = cp.get_array_module(arr)
        as_strided = cp_as_strided if xp == cp else np_as_strided

        in_shape = arr.shape
        nbytes = arr.dtype.itemsize

        # number of per-byte steps to take to fill window
        window_strides_arr = xp.cumprod(xp.array(arr.shape[:0:-1]))[::-1]
        window_strides = tuple(window_strides_arr.tolist()) + (1,)

        # number of per-byte steps to take to place window
        step_strides = tuple(xp.array(window_strides[-len(steps):]) * xp.array(steps))
        # number of bytes to step to populate sliding window view
        strides = tuple(int(i) * nbytes for i in step_strides + window_strides)

        outshape_arr = (xp.array(in_shape) - xp.array(window_shape)) // xp.array(steps) + 1
        # outshape: ([X, (...), Z], ..., [Wx, (...), Wz])
        outshape = tuple(outshape_arr.tolist())
        outshape = outshape + arr.shape[:-len(steps)] + tuple(window_shape)
        if xp == cp:
            return as_strided(arr, shape=outshape, strides=strides)
        else:
            return as_strided(arr, shape=outshape, strides=strides, writeable=False)

    def sector_mask(shape,centre,radius,angle_range): # used in windshelter_prep
        """
        Return a boolean mask for a circular sector. The start/stop angles in  
        `angle_range` should be given in clockwise order.
        """

        x,y = np.ogrid[:shape[0],:shape[1]]
        cx,cy = centre
        tmin,tmax = np.deg2rad(angle_range)

        # ensure stop angle > start angle
        if tmax < tmin:
                tmax += 2*np.pi

        # convert cartesian --> polar coordinates
        r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy)
        theta = np.arctan2(x-cx,y-cy) - tmin

        # wrap angles between 0 and 2*pi
        theta %= (2*np.pi)

        # circular mask
        circmask = r2 <= radius*radius

        # angular mask
        anglemask = theta <= (tmax-tmin)

        a = circmask*anglemask

        return a

    def windshelter_prep(radius, direction, tolerance, cellsize):
        x_size = y_size = 2*radius+1
        x_arr, y_arr = np.mgrid[0:x_size, 0:y_size]
        cell_center = (radius, radius)
        dist = (np.sqrt((x_arr - cell_center[0])**2 + (y_arr - cell_center[1])**2))*cellsize
        # dist = np.round(dist, 5)

        mask = sector_mask(dist.shape, (radius, radius), radius, (direction, tolerance))
        mask[radius, radius] = True # bug fix

        return dist, mask

    def windshelter(x, prob, dist, mask, radius): # applying the windshelter function
        data = x*mask
        data[data==profile['nodata']]=np.nan
        data[data==0]=np.nan
        center = data[radius, radius]
        data[radius, radius]=np.nan
        data = np.arctan((data-center)/dist)
        data = np.nanquantile(data, prob)
        return data

    def windshelter_window(radius, prob):

        dist, mask = windshelter_prep(radius, winddir - windtol + 270, winddir + windtol + 270, cell_size)
        
        # --- Move dist and mask to GPU
        dist_cp = cp.asarray(dist)
        mask_cp = cp.asarray(mask)

        # --- Create window view on the GPU array directly
        window_shape = ((radius*2)+1, (radius*2)+1)
        window = sliding_window_view(cp_array_windshelter[-1], window_shape, (1, 1))

        nc, nr = window.shape[0], window.shape[1]
        print("Bounds of windshelter window:")
        print(str(nc))
        print(str(nr))
        print("Processing windows on GPU...")

        # --- Vectorized windshelter calculation on GPU ---
        # Reshape mask for broadcasting
        mask_cp = mask_cp.reshape(1, 1, mask_cp.shape[0], mask_cp.shape[1])
        
        # --- Batch processing to avoid out-of-memory errors ---
        batch_size_i = 128  # Adjust this based on your GPU memory
        batch_size_j = 128 # Adjust this based on your GPU memory
        ws = cp.empty((nc, nr), dtype=cp.float32)

        for i in range(0, nc, batch_size_i):
            end_i = min(i + batch_size_i, nc)
            print(f"processing batch starting at {str(i)}")
            for j in range(0, nr, batch_size_j):
                end_j = min(j + batch_size_j, nr)
                window_batch = window[i:end_i, j:end_j] # Batch of windows

                # Apply mask to the batch
                data = window_batch * mask_cp
                
                # Set nodata and 0 values to NaN
                data[data == profile['nodata']] = cp.nan
                data[data == 0] = cp.nan
                
                # Extract center pixel value for each window in the batch
                center = data[:, :, radius, radius].copy()
                
                # Set center pixel in each window to NaN
                data[:, :, radius, radius] = cp.nan
                
                # Reshape center to allow broadcasting for subtraction
                center_reshaped = center[:, :, cp.newaxis, cp.newaxis]
                
                # Perform arctan calculation for the batch
                data = cp.arctan((data - center_reshaped) / dist_cp)
                
                # Calculate quantile for the batch and store it
                ws[i:end_i, j:end_j] = cp.quantile(data, prob, axis=(-2, -1))

                # Clean up GPU memory
                del window_batch, data, center, center_reshaped
                cp.get_default_memory_pool().free_all_blocks()

        # --- End of vectorized calculation ---

        ws_cpu = ws.get()
        del ws
        cp.get_default_memory_pool().free_all_blocks()

        data = np.pad(ws_cpu, pad_width=radius, mode='constant', constant_values=-9999)
        data = data.reshape(1, data.shape[0], data.shape[1])
        data = data.astype('float32')
        
        return data


    #######################
    # Calculate slope and windshelter
    #######################
    
    print("Calculating slope angle")

    with rasterio.open(DEM) as src:
        array = src.read(1)
        # PSE - converted dem array to int for larger datasets
        array = array.astype('int')
        profile = src.profile
        array[np.where(array < -100)] = 0
        # cp_array_slope = cp.asarray(array)
        
    cell_size = profile['transform'][0]

    # Evaluate gradient in two dimensions
    
    px, py = np.gradient(array, cell_size)
    slope = np.sqrt(px ** 2 + py ** 2)

    # If needed in degrees, convert using
    slope_deg = np.degrees(np.arctan(slope))
    slope_deg = slope_deg.astype(np.float32)

    print("Calculating windshelter")

    # Calculate windshelter
    with rasterio.open(DEM) as src:
        array = src.read()
        array = array.astype('float')
        cp_array_windshelter = cp.asarray(array)
        profile = src.profile
        cell_size = profile['transform'][0]
        print("Sending array to windshelter_window function")

    # start of commented out code for unit testing
    data = windshelter_window(radius, prob)

    print("Saving raster to PRA/windshelter.tif")
    with rasterio.open(DEM) as src:
        profile = src.profile
    profile.update({"dtype": "float32", "nodata": -9999})

    f.write(f'data before saving to PRA/windshelter.tif: {data}')

    windshelter = np.nan_to_num(data, nan=-9999)

    f.write(f'data after saving to PRA/windshelter.tif: \n')
    f.write(str(np.unique(windshelter[~np.isnan(windshelter)])))

    # Save raster to path using meta data from dem.tif (i.e. projection)
    with rasterio.open('PRA/windshelter.tif', "w", **profile) as dest:
        dest.write(windshelter)
    # end of commented out code for unit testing

    print("Defining Cauchy functions")

    #######################
    # --- Cauchy functions
    #######################

    # --- Define bell curve parameters for slope
    a = 11
    b = 4
    c = 43

    f.write("Cauchy slope function: a={}, b={}, c={}\n".format(a, b, c))

    slopeC = 1/(1+((slope_deg-c)/a)**(2*b))

    # --- Define bell curve parameters for windshelter
    a = 3
    b = 10
    c = 3
    f.write("Cauchy windshelter function: a={}, b={}, c={}\n".format(a, b, c))

    with rasterio.open("PRA/windshelter.tif") as src:
        windshelter = src.read()
        windshelter = windshelter.astype('float16')

    windshelterC = 1/(1+((windshelter-c)/a)**(2*b))
    f.write('After conversion: \n')
    f.write(str(np.unique(windshelterC[~np.isnan(windshelterC)])))

    # --- Define bell curve parameters for forest stem density
    if forest_type in ['stems']:
        a = 350
        b = 2
        c = -120
        f.write("Cauchy forest function (stems): a={}, b={}, c={}\n".format(a, b, c))

    if forest_type in ['bav']:
        a = 20
        b = 3.5
        c = -10
        f.write("Cauchy forest function (bav): a={}, b={}, c={}\n".format(a, b, c))

    if forest_type in ['sen2cc']:
        a = 50 # still finalizing defualts for Sen2cc, likeily will be region dependent based on local forest structure
        b = 1.5
        c = 0
        f.write("Cauchy forest function (sen2cc): a={}, b={}, c={}\n".format(a, b, c)) 
    
    # --- Define bell curve parameters for percent canopy cover
    if forest_type in ['pcc', 'no_forest']:
        a = 40
        b = 3.5
        c = -15

        if forest_type in ['pcc']:
            f.write("Cauchy forest function (pcc): a={}, b={}, c={}\n".format(a, b, c))
        if forest_type in ['no_forest']:
            f.write("No forest input given\n")

    if forest_type in ['pcc', 'stems', 'bav']:
        with rasterio.open(DEM) as dem_src:
            dem_profile = dem_src.profile
        with rasterio.open(FOREST) as src:
            with rasterio.vrt.WarpedVRT(src, crs=dem_profile['crs'], transform=dem_profile['transform'], width=dem_profile['width'], height=dem_profile['height']) as vrt:
                forest = vrt.read()

    
    if forest_type in ['no_forest']:
        with rasterio.open(DEM) as src:
            forest = src.read()
            # forest = np.where(forest > -100, 0, forest)
    forest = forest.astype(np.int16)
    forestC = 1/(1+((forest-c)/a)**(2*b)).astype(np.float32)
    # --- Ares with no forest and assigned -9999 will get a really small value which suggest dense forest. This function fixes this, but might have to be adjusted depending on the input dataset.
    forestC[np.where(forestC <= 0.00001)] = 1

    slopeC = np.round(slopeC, 5).astype(np.float32)
    # windshelterC = np.round(windshelterC, 5).astype(np.float32)
    forestC = np.round(forestC, 5).astype(np.float32)
    f.write('ForestC unique values: \n')
    f.write(str(np.unique(forestC[np.nonzero(forestC)])))

    #######################
    # --- Fuzzy logic operator
    #######################

    print("Starting the Fuzzy Logic Operator")

    f.write('slopeC unique: \n')
    f.write(str(np.unique(slopeC[np.nonzero(slopeC)])))

    f.write('windshelterC unique: \n')
    f.write(str(np.unique(windshelterC[np.nonzero(windshelterC)])))

    f.write('forestC unique: \n')
    f.write(str(np.unique(forestC[np.nonzero(forestC)])))

    minvar = np.minimum(slopeC, windshelterC)
    minvar = np.minimum(minvar, forestC)
    f.write('Before minvar rounding: \n')
    f.write(str(np.unique(minvar[np.nonzero(minvar)])))

    PRA = (1-minvar)*minvar+minvar*(slopeC+windshelterC+forestC)/3
    f.write('Before PRA rounding: \n')
    f.write(str(np.unique(PRA[~np.isnan(PRA)])))
    PRA = np.round(PRA, 5)
    PRA = PRA * 100
    f.write('After PRA rounding: \n')
    f.write(str(np.unique(PRA[~np.isnan(PRA)])))

    # --- Update metadata
    profile.update({'dtype': 'int16', 'nodata': -9999})

    # --- Save raster to path using meta data from dem.tif (i.e. projection)
    with rasterio.open('PRA/PRA_continous.tif', "w", **profile) as dest:
        dest.write(PRA)

    # --- Reclassify PRA to be used as input for FlowPy
    profile.update({'nodata': -9999})
    pra_thd = pra_thd * 100
    PRA[(0 <= PRA) & (PRA < pra_thd)] = 0
    PRA[(pra_thd <= PRA) & (PRA <= 100)] = 1

    with rasterio.open('PRA/PRA_binary.tif', "w", **profile) as dest:
        dest.write(PRA)

    # --- Remove islands smaller than 3 pixels
    sievefilter = sf + 1
    Image = gdal.Open('PRA/PRA_binary.tif', 1)  # open image in read-write mode
    Band = Image.GetRasterBand(1)
    gdal.SieveFilter(srcBand=Band, maskBand=None, dstBand=Band, threshold=sievefilter, connectedness=8, callback=gdal.TermProgress_nocb)
    del Image, Band  # close the datasets.
    
    print('PRA complete')
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    f.write("Stop time = {}\n".format(current_time))
    f.close()

if __name__ == "__main__":
    forest_type = str(sys.argv[1])
    if forest_type in ['pcc', 'stems', 'bav']:
        DEM = sys.argv[2]
        FOREST = sys.argv[3]
        radius = int(sys.argv[4])
        prob = float(sys.argv[5])
        winddir = int(sys.argv[6])
        windtol = int(sys.argv[7])
        pra_thd = float(sys.argv[8])
        sf = int(sys.argv[9])
        PRA(forest_type, DEM, FOREST, radius, prob, winddir, windtol, pra_thd, sf)
    if forest_type in ['no_forest']:
        DEM = sys.argv[2]
        radius = int(sys.argv[3])
        prob = float(sys.argv[4])
        winddir = int(sys.argv[5])
        windtol = int(sys.argv[6])
        pra_thd = float(sys.argv[7])
        sf = int(sys.argv[8])
        PRA(forest_type, DEM, DEM, radius, prob, winddir, windtol, pra_thd, sf)
