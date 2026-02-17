from osgeo import gdal
import numpy as np
import os

def reclassify_raster_by_threshold_gdal(input_path, output_path, threshold):
    """
    Reclassifies a raster file based on a threshold value using GDAL.
    
    Parameters:
    -----------
    input_path : str
        Path to the input .tif raster file
    output_path : str
        Path to save the reclassified raster
    threshold : float
        Threshold value for reclassification (below = 0, above = 1)
    """
    # Open input dataset
    src = gdal.Open(input_path)
    band = src.GetRasterBand(1)
    data = band.ReadAsArray()
    
    # Debug: inspect data
    print(f"Data type: {data.dtype}")
    print(f"Min value: {np.nanmin(data)}")
    print(f"Max value: {np.nanmax(data)}")
    print(f"Unique values: {np.unique(data)}")
    print(f"Threshold: {threshold}")
    
    # Reclassify
    reclassified = np.where(data < threshold, 0, 1).astype(np.uint8)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create output dataset
    driver = gdal.GetDriverByName('GTiff')
    dst = driver.Create(output_path, src.RasterXSize, src.RasterYSize, 1, gdal.GDT_Byte)
    if dst is None:
        print(f"Failed to create output file {output_path}")
        print(f"Error: {gdal.GetLastErrorMsg()}")
        src = None
        return
    dst.SetGeoTransform(src.GetGeoTransform())
    dst.SetProjection(src.GetProjection())
    
    # Write data
    out_band = dst.GetRasterBand(1)
    out_band.WriteArray(reclassified)
    out_band.FlushCache()
    
    dst = None
    src = None
    
    print(f"Reclassified raster saved to {output_path}")

reclassify_raster_by_threshold_gdal(r'D:\Phil\PhD\DEM\South_Craigieburn\CB_Canopy_5m.tif', r'D:\Phil\PhD\AutoATES-v2.0\data\Inputs\CB_Canopy_Bool_5m.tif', threshold=10)