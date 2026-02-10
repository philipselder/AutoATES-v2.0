from PRA_Buhler_OBIA import OBIAPRADelineation

obia = OBIAPRADelineation(r'path/to/dem.tif', r'path/to/canopy_bool.tif')
results = obia.run_full_analysis()