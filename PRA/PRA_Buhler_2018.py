from PRA_Buhler_OBIA import OBIAPRADelineation
import os

input_path = r'D:\Phil\PhD\AutoATES-v2.0\data\Inputs'

obia = OBIAPRADelineation(os.path.join(input_path, 'AP_DEM_Clip_5m.tif'), os.path.join(input_path, 'AP_Canopy_Bool_5m.tif'), output_dir=r'D:\Phil\PhD\AutoATES-v2.0\data\PRA')
results = obia.run_full_analysis()