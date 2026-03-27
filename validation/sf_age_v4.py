import os

os.environ[
    'PROJ_DATA'] = r"C:/Users/franzisf/PycharmProjects/panalesis_atlas/venv/Lib/site-packages/pyproj/proj_dir/share/proj"

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling
import geopandas as gpd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
import pandas as pd

from pathlib import Path
import json

project_root = Path(__file__).resolve().parent.parent
config_path = project_root / "config.json"
with open(config_path, "r") as f:
    config = json.load(f)

cob_path          = config["validation"]["cob_path"]
_section          = config["validation"]["seafloor_age"]
ref_path          = _section["ref_path"]
model_path        = _section["model_path"]
output_csv        = _section["output_csv"]
output_fig        = _section["output_fig"]
output_diagnostic = _section["output_diagnostic"]

# Analysis options - only oceanic for seafloor age
ANALYZE_OCEANIC_ONLY = True

# Color scheme
COLOR_REF = '#408c80'
COLOR_MODEL = '#408c80'

# =============================================================================
# LOAD AND PREPARE DATA
# =============================================================================
print("=" * 80)
print("LOADING DATA")
print("=" * 80)

# Load COB polygon
cob = gpd.read_file(cob_path)
print(f"COB CRS: {cob.crs}")

# Load reference raster
with rasterio.open(ref_path) as src_ref:
    ref_data = src_ref.read(1)
    ref_transform = src_ref.transform
    ref_crs = src_ref.crs
    ref_nodata = src_ref.nodata
    ref_profile = src_ref.profile

print(f"Reference raster CRS: {ref_crs}")
print(f"Reference shape: {ref_data.shape}")
print(f"Reference NoData value: {ref_nodata}")

# Reproject COB if needed
if cob.crs != ref_crs:
    cob = cob.to_crs(ref_crs)

# Load model raster and resample to reference grid
with rasterio.open(model_path) as src_model:
    model_raw = src_model.read(1)
    model_nodata = src_model.nodata
    model_transform = src_model.transform
    model_crs = src_model.crs

print(f"Model NoData value: {model_nodata}")

print("\nResampling model raster to match reference grid...")
model_data = np.empty_like(ref_data, dtype=np.float32)

reproject(
    source=model_raw,
    destination=model_data,
    src_transform=model_transform,
    src_crs=model_crs,
    dst_transform=ref_transform,
    dst_crs=ref_crs,
    resampling=Resampling.bilinear
)

# =============================================================================
# CREATE MASKS
# =============================================================================
print("\nCreating masks...")
cont_mask = rasterize(
    [(geom, 1) for geom in cob.geometry],
    out_shape=ref_data.shape,
    transform=ref_transform,
    fill=0,
    dtype='uint8'
)

ocean_mask = (cont_mask == 0)
continent_mask = (cont_mask == 1)

# =============================================================================
# COMPREHENSIVE VALID DATA FILTERING
# =============================================================================
print("\nApplying NoData filters...")

# Reference valid mask
if ref_nodata is not None:
    valid_ref = (ref_data != ref_nodata)
else:
    valid_ref = np.ones_like(ref_data, dtype=bool)

# Model valid mask - handle multiple possible NoData representations
if model_nodata is not None:
    valid_model = (model_data != model_nodata)
else:
    valid_model = np.ones_like(model_data, dtype=bool)

# Explicitly filter common NoData values
valid_model = valid_model & (model_data != -9999) & (model_data != -9999.0)

# Filter extreme/unrealistic values for seafloor age (0-300 Myr)
valid_model = valid_model & (model_data >= 0) & (model_data <= 300)
valid_ref = valid_ref & (ref_data >= 0) & (ref_data <= 300)

# Check for NaN or Inf values
valid_data = valid_ref & valid_model & np.isfinite(ref_data) & np.isfinite(model_data)

print(f"Total valid pixels: {np.sum(valid_data)}")
print(f"Continental pixels: {np.sum(continent_mask & valid_data)}")
print(f"Oceanic pixels: {np.sum(ocean_mask & valid_data)}")

# =============================================================================
# DIAGNOSTIC PLOT
# =============================================================================
print("\nCreating diagnostic plot...")
fig_diag, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Model data coverage
ax1 = axes[0, 0]
coverage_map = np.ones_like(model_data) * np.nan
coverage_map[valid_model] = 1  # Valid data
coverage_map[~valid_model & ocean_mask] = 0  # NoData in ocean
coverage_map[~valid_model & continent_mask] = -1  # NoData on continent

im1 = ax1.imshow(coverage_map, cmap='RdYlGn', vmin=-1, vmax=1, interpolation='nearest')
ax1.set_title('Model Data Coverage\n(Green=Valid, Yellow=Ocean NoData, Red=Continent NoData)')
ax1.axis('off')
plt.colorbar(im1, ax=ax1, label='Coverage', ticks=[-1, 0, 1])

# 2. Model seafloor age (where valid)
ax2 = axes[0, 1]
plot_data = model_data.copy()
plot_data[~valid_model] = np.nan
im2 = ax2.imshow(plot_data, cmap='viridis', vmin=0, vmax=280, interpolation='nearest')
ax2.set_title('Model Seafloor Age (Valid Pixels Only)')
ax2.axis('off')
plt.colorbar(im2, ax=ax2, label='Age (Myr)')

# 3. Reference seafloor age
ax3 = axes[1, 0]
plot_ref = ref_data.copy()
plot_ref[~valid_ref] = np.nan
im3 = ax3.imshow(plot_ref, cmap='viridis', vmin=0, vmax=280, interpolation='nearest')
ax3.set_title('Reference Seafloor Age (NOAA)')
ax3.axis('off')
plt.colorbar(im3, ax=ax3, label='Age (Myr)')

# 4. Difference map (where both valid)
ax4 = axes[1, 1]
diff_map = np.ones_like(model_data) * np.nan
valid_comparison = valid_data & ocean_mask
diff_map[valid_comparison] = model_data[valid_comparison] - ref_data[valid_comparison]
im4 = ax4.imshow(diff_map, cmap='RdBu_r', vmin=-50, vmax=50, interpolation='nearest')
ax4.set_title('Difference (Model - Reference)\nOcean Only, Where Both Valid')
ax4.axis('off')
plt.colorbar(im4, ax=ax4, label='Difference (Myr)')

plt.tight_layout()
plt.savefig(output_diagnostic, dpi=150, bbox_inches='tight')
print(f"Diagnostic plot saved to: {output_diagnostic}")
plt.close()


# =============================================================================
# STATISTICAL ANALYSIS FUNCTION
# =============================================================================
def calculate_statistics(ref, model, mask, region_name):
    """Calculate comprehensive statistics for validation."""
    ref_vals = ref[mask]
    model_vals = model[mask]

    if len(ref_vals) == 0:
        print(f"Warning: No valid data for {region_name}")
        return None

    # Double-check for any remaining invalid values
    valid_idx = np.isfinite(ref_vals) & np.isfinite(model_vals)
    valid_idx = valid_idx & (ref_vals >= 0) & (model_vals >= 0)

    ref_vals = ref_vals[valid_idx]
    model_vals = model_vals[valid_idx]

    if len(ref_vals) == 0:
        print(f"Warning: No valid data after secondary filtering for {region_name}")
        return None

    errors = model_vals - ref_vals
    abs_errors = np.abs(errors)

    # Basic statistics
    pearson_r, pearson_p = stats.pearsonr(ref_vals, model_vals)
    spearman_rho, spearman_p = stats.spearmanr(ref_vals, model_vals)

    # Distribution comparison tests
    ks_stat, ks_pvalue = stats.ks_2samp(ref_vals, model_vals)
    wasserstein_dist = stats.wasserstein_distance(ref_vals, model_vals)

    # Additional correlation metrics
    kendall_tau, kendall_p = stats.kendalltau(ref_vals, model_vals)

    # Bias metrics
    bias = np.mean(errors)
    bias_percent = (bias / np.mean(np.abs(ref_vals))) * 100 if np.mean(np.abs(ref_vals)) != 0 else np.nan

    # Skill scores
    # Nash-Sutcliffe Efficiency
    nse = 1 - (np.sum(errors ** 2) / np.sum((ref_vals - np.mean(ref_vals)) ** 2))

    # Index of Agreement (Willmott, 1981)
    ioa = 1 - (np.sum(errors ** 2) /
               np.sum((np.abs(model_vals - np.mean(ref_vals)) +
                       np.abs(ref_vals - np.mean(ref_vals))) ** 2))

    # Normalized RMSE
    nrmse = np.sqrt(np.mean(errors ** 2)) / (np.max(ref_vals) - np.min(ref_vals))

    # Model efficiency based on standard deviation
    mef = 1 - (np.std(errors, ddof=1) / np.std(ref_vals, ddof=1)) ** 2

    stats_dict = {
        'Region': region_name,
        'N_pixels': len(ref_vals),

        # Reference statistics
        'Ref_mean': np.mean(ref_vals),
        'Ref_stderr': np.std(ref_vals, ddof=1) / np.sqrt(len(ref_vals)),
        'Ref_std': np.std(ref_vals, ddof=1),
        'Ref_min': np.min(ref_vals),
        'Ref_max': np.max(ref_vals),
        'Ref_range': np.max(ref_vals) - np.min(ref_vals),
        'Ref_median': np.median(ref_vals),
        'Ref_skewness': stats.skew(ref_vals),
        'Ref_kurtosis': stats.kurtosis(ref_vals),

        # Model statistics
        'Model_mean': np.mean(model_vals),
        'Model_stderr': np.std(model_vals, ddof=1) / np.sqrt(len(model_vals)),
        'Model_std': np.std(model_vals, ddof=1),
        'Model_min': np.min(model_vals),
        'Model_max': np.max(model_vals),
        'Model_range': np.max(model_vals) - np.min(model_vals),
        'Model_median': np.median(model_vals),
        'Model_skewness': stats.skew(model_vals),
        'Model_kurtosis': stats.kurtosis(model_vals),

        # Error statistics
        'Mean_Bias': bias,
        'Bias_Percent': bias_percent,
        'RMS_Diff': np.sqrt(np.mean(errors ** 2)),
        'MAE': np.mean(abs_errors),
        'Std_Diff': np.std(errors, ddof=1),
        'Max_AbsDiff': np.max(abs_errors),
        'Min_Error': np.min(errors),
        'Max_Error': np.max(errors),

        # Correlation metrics
        'Pearson_r': pearson_r,
        'Pearson_p': pearson_p,
        'Spearman_rho': spearman_rho,
        'Spearman_p': spearman_p,
        'Kendall_tau': kendall_tau,
        'Kendall_p': kendall_p,
        'R2': r2_score(ref_vals, model_vals),

        # Skill scores
        'Nash_Sutcliffe': nse,
        'Index_Agreement': ioa,
        'NRMSE': nrmse,
        'Model_Efficiency': mef,

        # Distribution comparison
        'KS_statistic': ks_stat,
        'KS_pvalue': ks_pvalue,
        'Wasserstein_distance': wasserstein_dist,

        # Error percentiles
        'P5_AbsDiff': np.percentile(abs_errors, 5),
        'P10_AbsDiff': np.percentile(abs_errors, 10),
        'P25_AbsDiff': np.percentile(abs_errors, 25),
        'P50_AbsDiff': np.percentile(abs_errors, 50),
        'P75_AbsDiff': np.percentile(abs_errors, 75),
        'P90_AbsDiff': np.percentile(abs_errors, 90),
        'P95_AbsDiff': np.percentile(abs_errors, 95),
        'P99_AbsDiff': np.percentile(abs_errors, 99),
    }

    return stats_dict


# =============================================================================
# CALCULATE STATISTICS
# =============================================================================
print("\n" + "=" * 80)
print("CALCULATING STATISTICS")
print("=" * 80)

results = []

if ANALYZE_OCEANIC_ONLY:
    oceanic_stats = calculate_statistics(ref_data, model_data, ocean_mask & valid_data, "Oceanic")
    if oceanic_stats is not None:
        results.append(oceanic_stats)

if len(results) == 0:
    print("ERROR: No valid statistics could be calculated. Check your data!")
    exit(1)

# Create DataFrame
stats_df = pd.DataFrame(results)

# =============================================================================
# PRINT RESULTS
# =============================================================================
print("\n" + "=" * 80)
print("SEAFLOOR AGE VALIDATION RESULTS")
print("=" * 80)

for idx, row in stats_df.iterrows():
    region = row['Region']
    print(f"\n{'=' * 80}")
    print(f"{region.upper()} REGION")
    print(f"{'=' * 80}")
    print(f"  Number of pixels: {int(row['N_pixels']):,}")

    print(f"\n  REFERENCE (Seton 2020):")
    print(f"    Mean: {row['Ref_mean']:.1f} ± {row['Ref_stderr']:.1f} Myr")
    print(f"    Median: {row['Ref_median']:.1f} Myr")
    print(f"    Std deviation: {row['Ref_std']:.1f} Myr")
    print(f"    Range: [{row['Ref_min']:.1f}, {row['Ref_max']:.1f}] Myr (span: {row['Ref_range']:.1f} Myr)")
    print(f"    Skewness: {row['Ref_skewness']:.3f}")
    print(f"    Kurtosis: {row['Ref_kurtosis']:.3f}")

    print(f"\n  PANALESIS MODEL:")
    print(f"    Mean: {row['Model_mean']:.1f} ± {row['Model_stderr']:.1f} Myr")
    print(f"    Median: {row['Model_median']:.1f} Myr")
    print(f"    Std deviation: {row['Model_std']:.1f} Myr")
    print(f"    Range: [{row['Model_min']:.1f}, {row['Model_max']:.1f}] Myr (span: {row['Model_range']:.1f} Myr)")
    print(f"    Skewness: {row['Model_skewness']:.3f}")
    print(f"    Kurtosis: {row['Model_kurtosis']:.3f}")

    print(f"\n  ERROR METRICS (Model - Reference):")
    print(f"    Mean bias: {row['Mean_Bias']:.1f} Myr ({row['Bias_Percent']:.2f}%)")
    print(f"    RMS difference: {row['RMS_Diff']:.1f} Myr  ← PRIMARY METRIC")
    print(f"    Mean absolute error: {row['MAE']:.1f} Myr")
    print(f"    Std of errors: {row['Std_Diff']:.1f} Myr")
    print(f"    Error range: [{row['Min_Error']:.1f}, {row['Max_Error']:.1f}] Myr")
    print(f"    Max absolute error: {row['Max_AbsDiff']:.1f} Myr")

    print(f"\n  ERROR DISTRIBUTION (Absolute):")
    print(f"    5th percentile: {row['P5_AbsDiff']:.1f} Myr")
    print(f"    10th percentile: {row['P10_AbsDiff']:.1f} Myr")
    print(f"    25th percentile: {row['P25_AbsDiff']:.1f} Myr")
    print(f"    50th percentile (median): {row['P50_AbsDiff']:.1f} Myr")
    print(f"    75th percentile: {row['P75_AbsDiff']:.1f} Myr")
    print(f"    90th percentile: {row['P90_AbsDiff']:.1f} Myr")
    print(f"    95th percentile: {row['P95_AbsDiff']:.1f} Myr")
    print(f"    99th percentile: {row['P99_AbsDiff']:.1f} Myr")

    print(f"\n  CORRELATION METRICS:")
    print(f"    Pearson r: {row['Pearson_r']:.4f} (p={row['Pearson_p']:.2e})")
    print(f"    Spearman ρ: {row['Spearman_rho']:.4f} (p={row['Spearman_p']:.2e})")
    print(f"    Kendall τ: {row['Kendall_tau']:.4f} (p={row['Kendall_p']:.2e})")
    print(f"    R²: {row['R2']:.4f}")

    # Interpretation
    rho_diff = row['Spearman_rho'] - row['Pearson_r']
    if abs(rho_diff) > 0.05:
        if rho_diff > 0:
            print(f"    → Spearman > Pearson: Model captures seafloor age patterns")
            print(f"      but may have non-linear scaling (Δ={rho_diff:+.4f})")
        else:
            print(f"    → Pearson > Spearman: Possible non-monotonic errors (Δ={rho_diff:+.4f})")
    else:
        print(f"    → Similar correlation values: Relationship is approximately linear")

    print(f"\n  SKILL SCORES:")
    print(f"    Nash-Sutcliffe Efficiency: {row['Nash_Sutcliffe']:.4f} (1.0=perfect, 0.0=mean baseline)")
    print(f"    Index of Agreement: {row['Index_Agreement']:.4f} (1.0=perfect, 0.0=worst)")
    print(f"    Normalized RMSE: {row['NRMSE']:.4f} (0.0=perfect)")
    print(f"    Model Efficiency Factor: {row['Model_Efficiency']:.4f}")

    print(f"\n  DISTRIBUTION COMPARISON:")
    print(f"    Kolmogorov-Smirnov statistic: {row['KS_statistic']:.4f} (p={row['KS_pvalue']:.2e})")
    print(f"    Wasserstein distance: {row['Wasserstein_distance']:.1f} Myr")

    # Interpretation of distribution tests
    if row['KS_pvalue'] < 0.01:
        print(f"    → Distributions are significantly different (p < 0.01)")
    elif row['KS_pvalue'] < 0.05:
        print(f"    → Distributions are significantly different (p < 0.05)")
    else:
        print(f"    → Distributions are not significantly different (p ≥ 0.05)")

# =============================================================================
# SAVE STATISTICS
# =============================================================================
stats_df.to_csv(output_csv, index=False)
print(f"\n{'=' * 80}")
print(f"Detailed statistics saved to: {output_csv}")
print(f"{'=' * 80}")

# =============================================================================
# VISUALIZATION
# =============================================================================
print("\nGenerating validation plots...")

sns.set(style="whitegrid")
fig = plt.figure(figsize=(16, 10))

# Create a 2x3 grid
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# Get data for plotting
ref_vals_ocean = ref_data[ocean_mask & valid_data].flatten()
model_vals_ocean = model_data[ocean_mask & valid_data].flatten()

# Plot 1: KDE - Oceanic
ax1 = fig.add_subplot(gs[0, 0])
sns.kdeplot(ref_vals_ocean, ax=ax1, color=COLOR_REF, linewidth=2, label='NOAA (Müller et al., 2016)')
sns.kdeplot(model_vals_ocean, ax=ax1, color=COLOR_MODEL, linewidth=2, linestyle='--', label='PANALESIS (This Study)')
ax1.set_xlabel('Seafloor Age [Myr]', fontsize=11)
ax1.set_ylabel('Density', fontsize=11)
ax1.set_title('a) Oceanic Distribution', fontsize=12, fontweight='bold', loc='left')
ax1.legend(fontsize=9, loc='upper right')
ax1.grid(False)

# Add statistics annotation
ocean_row = stats_df[stats_df['Region'] == 'Oceanic'].iloc[0]
stats_text = f"KS: {ocean_row['KS_statistic']:.4f}\nWD: {ocean_row['Wasserstein_distance']:.1f} Myr"
ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 2: Scatter plot
ax2 = fig.add_subplot(gs[0, 1])
# Subsample for plotting if too many points
n_sample = min(50000, len(ref_vals_ocean))
if n_sample < len(ref_vals_ocean):
    idx_sample = np.random.choice(len(ref_vals_ocean), n_sample, replace=False)
else:
    idx_sample = np.arange(len(ref_vals_ocean))
ax2.scatter(ref_vals_ocean[idx_sample], model_vals_ocean[idx_sample],
            alpha=0.1, s=1, color=COLOR_REF)
# 1:1 line
min_val = min(ref_vals_ocean.min(), model_vals_ocean.min())
max_val = max(ref_vals_ocean.max(), model_vals_ocean.max())
ax2.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1, label='1:1 line')
ax2.set_xlabel('NOAA Seafloor Age [Myr]', fontsize=11)
ax2.set_ylabel('PANALESIS Seafloor Age [Myr]', fontsize=11)
ax2.set_title('b) Oceanic Scatter', fontsize=12, fontweight='bold', loc='left')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Add R² annotation
r2_text = f"R² = {ocean_row['R2']:.4f}\nρ = {ocean_row['Spearman_rho']:.4f}"
ax2.text(0.98, 0.02, r2_text, transform=ax2.transAxes, fontsize=9,
         verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 3: Error histogram
ax3 = fig.add_subplot(gs[0, 2])
errors_ocean = model_vals_ocean - ref_vals_ocean
ax3.hist(errors_ocean, bins=100, color=COLOR_REF, alpha=0.7, edgecolor='black', linewidth=0.5)
ax3.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
ax3.axvline(np.mean(errors_ocean), color='blue', linestyle='--', linewidth=2,
            label=f'Mean bias: {np.mean(errors_ocean):.1f} Myr')
ax3.set_xlabel('Error [Myr] (Model - Reference)', fontsize=11)
ax3.set_ylabel('Frequency', fontsize=11)
ax3.set_title('c) Error Distribution', fontsize=12, fontweight='bold', loc='left')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Cumulative error distribution
ax4 = fig.add_subplot(gs[1, 0])
sorted_abs_errors = np.sort(np.abs(errors_ocean))
cumulative = np.arange(1, len(sorted_abs_errors) + 1) / len(sorted_abs_errors) * 100
ax4.plot(sorted_abs_errors, cumulative, color=COLOR_REF, linewidth=2)
ax4.axhline(50, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax4.axhline(90, color='orange', linestyle='--', linewidth=1, alpha=0.5)
ax4.axvline(ocean_row['P50_AbsDiff'], color='red', linestyle='--', linewidth=1, alpha=0.5)
ax4.axvline(ocean_row['P90_AbsDiff'], color='orange', linestyle='--', linewidth=1, alpha=0.5)
ax4.set_xlabel('Absolute Error [Myr]', fontsize=11)
ax4.set_ylabel('Cumulative Percentage [%]', fontsize=11)
ax4.set_title('d) Cumulative Error Distribution', fontsize=12, fontweight='bold', loc='left')
ax4.grid(True, alpha=0.3)
ax4.text(0.98, 0.50, f'50th: {ocean_row["P50_AbsDiff"]:.1f} Myr', transform=ax4.transAxes,
         fontsize=9, verticalalignment='center', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
ax4.text(0.98, 0.90, f'90th: {ocean_row["P90_AbsDiff"]:.1f} Myr', transform=ax4.transAxes,
         fontsize=9, verticalalignment='center', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3))

# Plot 5: Error vs Reference Age
ax5 = fig.add_subplot(gs[1, 1])
# Bin the data for clearer visualization
age_bins = np.linspace(0, 280, 30)
bin_centers = (age_bins[:-1] + age_bins[1:]) / 2
binned_errors = []
binned_std = []
for i in range(len(age_bins) - 1):
    mask = (ref_vals_ocean >= age_bins[i]) & (ref_vals_ocean < age_bins[i + 1])
    if np.sum(mask) > 10:
        binned_errors.append(np.mean(errors_ocean[mask]))
        binned_std.append(np.std(errors_ocean[mask]))
    else:
        binned_errors.append(np.nan)
        binned_std.append(np.nan)

binned_errors = np.array(binned_errors)
binned_std = np.array(binned_std)
valid_bins = ~np.isnan(binned_errors)

ax5.plot(bin_centers[valid_bins], binned_errors[valid_bins], 'o-', color=COLOR_REF, linewidth=2, markersize=4)
ax5.fill_between(bin_centers[valid_bins],
                  binned_errors[valid_bins] - binned_std[valid_bins],
                  binned_errors[valid_bins] + binned_std[valid_bins],
                  alpha=0.3, color=COLOR_REF)
ax5.axhline(0, color='k', linestyle='--', linewidth=1)
ax5.set_xlabel('Reference Seafloor Age [Myr]', fontsize=11)
ax5.set_ylabel('Mean Error [Myr]', fontsize=11)
ax5.set_title('e) Error vs. Seafloor Age', fontsize=12, fontweight='bold', loc='left')
ax5.grid(True, alpha=0.3)

# Plot 6: Summary statistics
ax6 = fig.add_subplot(gs[1, 2])
metrics = ['RMSE', 'MAE', 'Median\nAbs Error']
values = [ocean_row['RMS_Diff'], ocean_row['MAE'], ocean_row['P50_AbsDiff']]
colors = ['#bc272d', '#F1960F', '#408c80']

bars = ax6.bar(metrics, values, color=colors, alpha=0.8)
ax6.set_ylabel('Error [Myr]', fontsize=11)
ax6.set_title('f) Error Metrics Summary', fontsize=12, fontweight='bold', loc='left')
ax6.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width() / 2., height,
             f'{val:.1f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(output_fig, dpi=300, bbox_inches='tight')
print(f"Validation plots saved to: {output_fig}")
plt.show()

print("\n" + "=" * 80)
print("VALIDATION COMPLETE")
print("=" * 80)