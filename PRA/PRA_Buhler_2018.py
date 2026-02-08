from PRA_Buhler_OBIA import OBIAPRADelineation

obia = OBIAPRADelineation(r'Inputs\AP_DEM_Clip_5m.tif', r'Inputs\AP_Canopy_5m.tif')
results = obia.run_full_analysis()