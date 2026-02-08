# OBIA-Based PRA Delineation Algorithm

## Overview

This algorithm implements Object-Based Image Analysis (OBIA) for delineating Potential Release Areas (PRAs) for avalanches based on the methodology described by Bühler et al. (2013). It processes a Digital Elevation Model (DEM) at 5-meter resolution and derives key terrain parameters to identify avalanche-prone slopes.

## References

Bühler, Y., Purves, R. S., Hilbich, C., Weise, K., Buser, O., & Salzmann, N. (2013). Avalanche hazard mapping for practical applications: A comparison of approaches and results. Journal of Glaciology, 59(213), 324-334.

## Key Features

### Terrain Derivatives Calculated

1. **Slope Angle** (degrees)
   - First derivative of elevation
   - Filtered with 5×5 distance-weighted mean filter
   - Critical parameter for avalanche release
   - Key values: 30-60° (frequent), 28-60° (extreme)

2. **Aspect** (degrees and 8-sector classification)
   - Downslope direction of maximum gradient
   - 8-sector classification: N, NE, E, SE, S, SW, W, NW
   - Key for delineating between different PRAs
   - Weighted 3x more important in segmentation

3. **Plan Curvature** (rad/100 hm⁻¹)
   - Curvature perpendicular to slope direction
   - Indicates convex/concave terrain
   - Threshold: > 6 rad/100 hm⁻¹ excluded (gullies, ridges)

4. **Profile Curvature** (rad/100 hm⁻¹)
   - Curvature along slope direction
   - Change in slope angle

5. **Ruggedness** (normalized 0-1)
   - Terrain roughness independent of slope angle
   - Window size: 9 pixels (45m at 5m resolution)
   - Thresholds:
     - Rough terrain: > 0.03
     - Very rough: > 0.08
     - Sparse in natural terrain: > 0.1
   - Frequent scenario: exclude > 0.06
   - Extreme scenario: exclude > 0.08

6. **Fold** (unitless)
   - Change in adjacent normal vectors
   - Indicates ridges, gullies, abrupt terrain changes
   - Important for PRA delineation

## Processing Workflow

### Frequent Scenario (30-60° slopes)

```
1. Select slopes 30-60°
   ↓
2. Remove high ruggedness (> 0.06)
   ↓
3. Remove high plan curvature (> 6 rad/100hm)
   ↓
4. Remove small areas (< 500 m²)
   ↓
5. Segment into objects
   - Multiresolution segmentation
   - Weight aspect variations 3x more than slope/fold
   ↓
6. Classify forest coverage
   ↓
Output: Binary PRA map
```

### Extreme Scenario (28-60° slopes)

```
1. Select slopes 28-60°
   ↓
2. Remove high ruggedness (> 0.08)
   ↓
3. Remove high plan curvature (> 6 rad/100hm)
   ↓
4. Segment into objects
   ↓
5. Region growing algorithm
   - Merge adjacent objects with similar:
     * Aspect sector
     * Slope angle
     * Fold values
   ↓
6. Classify forest coverage
   ↓
Output: Binary PRA map (larger, more connected)
```

## Usage

### Installation Requirements

```bash
pip install numpy rasterio scipy
```

### Basic Usage

```python
from PRA_Buhler_OBIA import OBIAPRADelineation

# Initialize with DEM (forest layer is optional)
obia = OBIAPRADelineation(
    dem_path='data/dem.tif',
    forest_path='data/forest.tif',  # optional
    output_dir='./PRA_outputs'
)

# Run full analysis
results = obia.run_full_analysis()
```

### Command Line Usage

```bash
# With forest layer
python PRA_Buhler_OBIA.py data/dem.tif data/forest.tif outputs/

# Without forest layer
python PRA_Buhler_OBIA.py data/dem.tif

# Custom output directory
python PRA_Buhler_OBIA.py data/dem.tif data/forest.tif custom_outputs/
```

### Step-by-Step Usage

```python
from PRA_Buhler_OBIA import OBIAPRADelineation

# Initialize
obia = OBIAPRADelineation('data/dem.tif', 'data/forest.tif')

# Calculate individual derivatives
slope = obia.calculate_slope()
aspect, aspect_sectors = obia.calculate_aspect()
plan_curv, profile_curv = obia.calculate_curvature()
ruggedness = obia.calculate_ruggedness(window_size=9)
fold = obia.calculate_fold()

# Delineate PRAs
pra_frequent = obia.delineate_pra_frequent()
pra_extreme = obia.delineate_pra_extreme()
```

## Input Requirements

### Digital Elevation Model (DEM)
- **Format**: GeoTIFF (.tif)
- **Resolution**: 5 meters (resampled from original 2m)
- **Data Type**: Float32 or Int16
- **NoData Value**: Should be defined in raster metadata
- **Coordinate System**: Must be consistent across all input layers

### Forest Layer (Optional)
- **Format**: GeoTIFF (.tif)
- **Resolution**: Same as DEM (5m)
- **Data Type**: Binary (0=no forest, 1=forest)
- **Same extent and coordinate system as DEM**

## Output Files

All outputs are saved as GeoTIFF files in the specified output directory:

### Terrain Derivatives
- `slope_angle.tif` - Slope in degrees
- `aspect_degrees.tif` - Aspect in degrees (0-360)
- `aspect_sectors.tif` - 8-sector classification (0-7)
- `plan_curvature.tif` - Plan curvature (rad/100hm)
- `profile_curvature.tif` - Profile curvature (rad/100hm)
- `ruggedness.tif` - Normalized ruggedness (0-1)
- `fold.tif` - Fold values

### PRA Outputs
- `PRA_frequent_scenario.tif` - Binary PRA for frequent scenario
- `PRA_extreme_scenario.tif` - Binary PRA for extreme scenario

### Logging
- `OBIA_processing_log.txt` - Detailed processing log with statistics

## Algorithm Parameters

### Fixed Parameters (from Bühler et al. 2013)

| Parameter | Frequent | Extreme | Description |
|-----------|----------|---------|-------------|
| Slope angle | 30-60° | 28-60° | Release slope angle range |
| Ruggedness threshold | ≤ 0.06 | ≤ 0.08 | Maximum roughness allowed |
| Plan curvature threshold | ≤ 6 | ≤ 6 | rad/100hm, excludes gullies/ridges |
| Minimum object size | 500 m² | N/A | Removes small isolated areas |
| Ruggedness window | 9×9 pixels | 9×9 pixels | 45m at 5m resolution |
| Slope filter | 5×5 | N/A | Distance-weighted mean filter |
| Aspect weight | 3x | 3x | Relative to slope and fold |

### Modifiable Parameters

You can adjust these in the code:

```python
# Slope filtering
slope = obia.calculate_slope(filter_size=5)  # Kernel size

# Ruggedness window
ruggedness = obia.calculate_ruggedness(window_size=9)  # 9x9 = 45m

# Minimum object size
pra = obia.remove_small_objects(mask, min_size_m2=500)

# Slope ranges (in delineation methods)
# Modify: slope_mask = (self.slope >= 30) & (self.slope <= 60)
```

## Methodological Notes

### Slope Calculation
- Uses gradient-based method with cells at 5m resolution
- Distance-weighted mean filter reduces noise from isolated steep/flat pixels
- Creates more homogeneous objects for segmentation

### Aspect Classification
- Divides compass into 8 equal sectors (45° each)
- Starting from North (0-45° = N, 45-90° = NE, etc.)
- Changes in aspect are primary PRA delineation boundaries

### Curvature
- Plan curvature (perpendicular to flow): Eliminates convex ridges and concave gullies
- Profile curvature (along flow): Measures slope change
- High values indicate bedrock exposures or narrow gullies

### Ruggedness
- Based on normal vector divergence within window
- Independent of slope angle
- Effective for identifying rough terrain unsuitable for avalanche propagation

### Fold
- Calculated as divergence of normal vectors
- Effective ridge/gully indicator
- Complementary to ruggedness

## Known Limitations

1. **Simple Segmentation**: Current implementation uses basic seed-growing. For production use, consider:
   - Proprietary algorithms (e.g., Trimble eCognition Developer)
   - Advanced Python implementations (e.g., scikit-image, GDAL)

2. **Region Growing**: Extreme scenario region growing is simplified. Full implementation should:
   - Track segment adjacency
   - Calculate similarity metrics
   - Iteratively merge adjacent objects

3. **Forest Classification**: Currently binary mask application. Enhanced version could:
   - Distinguish forest density
   - Apply probabilistic forest masking
   - Consider forest distribution patterns

4. **Edge Effects**: Derivatives near raster edges may be less reliable

5. **Resolution Dependency**: Algorithm tuned for 5m resolution. Adjust parameters for different resolutions.

## Performance Notes

- Processing time depends on DEM size
- Ruggedness calculation (9×9 window) is computationally intensive
- For large areas, consider tiling and processing in sections
- All calculations are in float32 to balance precision and memory

## Quality Control

The algorithm outputs a processing log containing:
- Number of cells in each filtering step
- Mean values of terrain parameters
- Counts of objects before/after segmentation
- Output file locations and statistics

Review the log file to verify expected results:
```
Slope cells 30-60°: 150,000
After ruggedness filter: 145,000
After curvature filter: 140,000
After size filter: 130,000
Final PRA cells (frequent): 130,000
```

## Validation

Recommended validation approaches:
1. Visual comparison with aerial imagery
2. Expert avalanche path identification
3. Comparison with historical avalanche locations
4. Sensitivity analysis of threshold parameters
5. Cross-validation with other PRA delineation methods

## Future Enhancements

1. **Advanced Segmentation**: Integration with scikit-image or OpenCV
2. **GPU Acceleration**: CUDA/CuPy support for large datasets
3. **Probabilistic Output**: Fuzzy logic for uncertainty quantification
4. **Multi-scale Analysis**: Hierarchy of object scales
5. **Parameter Optimization**: Automated threshold calibration
6. **Web Interface**: REST API for cloud-based processing

## License

This implementation is provided as-is for research and operational purposes.
Cite the original methodology:

Bühler, Y., Purves, R. S., Hilbich, C., Weise, K., Buser, O., & Salzmann, N. (2013)

## Contact & Support

For questions about the algorithm methodology, refer to the original publication.
For implementation-specific issues, review the processing log and intermediate outputs.
