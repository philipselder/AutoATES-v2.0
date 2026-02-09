from PRA_Buhler_OBIA import OBIAPRADelineation

obia = OBIAPRADelineation(r'D:\Phil\PhD\AutoATES-v2.0\PRA\Inputs\AP_DEM_Clip_5m.tif', r'D:\Phil\PhD\AutoATES-v2.0\PRA\Inputs\AP_Canopy_Bool_5m.tif')
results = obia.run_full_analysis()