# -*- coding: utf-8 -*-
"""
Command-line script for 7T fMRI Group-Level Analysis using Nilearn.

This script performs a second-level GLM analysis and non-parametric
permutation testing on fMRI contrast maps.

Memory Optimization Notes:
- Using an explicit mask (--mask) significantly reduces data loaded.
- Computes an intersection mask to exclude non-finite AND zero-variance voxels.
- Starting with n_jobs=1 for permutation minimizes parallel memory load.
- Explicit deletion (del) of large variables and garbage collection (gc.collect).
- Closing matplotlib figures (plt.close) after saving.
- Using memory mapping (mmap=True) when loading masks/bg images via nibabel.
- Nilearn caching (--cache_dir/--memory_level) applies ONLY to SecondLevelModel fitting.
- NOTE: non_parametric_inference itself does NOT use memory/memory_level args.
"""

import os
import glob
import argparse
import pandas as pd
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import gc
import pickle
import warnings

from nilearn import plotting, __version__ as nilearn_version
from nilearn.glm.second_level import SecondLevelModel, non_parametric_inference
from scipy.stats import norm
from nilearn.masking import compute_multi_epi_mask, apply_mask
from nilearn.image import new_img_like

print(f"Using Nilearn version: {nilearn_version}")

# --- Helper Functions ---

def setup_directories(args):
    """Creates output directories if they don't exist."""
    os.makedirs(args.output_dir, exist_ok=True)
    plot_dir = os.path.join(args.output_dir, 'plots')
    map_dir = os.path.join(args.output_dir, 'maps')
    if args.save_plots:
        os.makedirs(plot_dir, exist_ok=True)
    if args.save_maps:
        os.makedirs(map_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    return plot_dir, map_dir


def load_input_files(input_dir, file_pattern):
    """Loads the list of second-level input NIfTI files."""
    search_path = os.path.join(input_dir, file_pattern)
    input_files = sorted(glob.glob(search_path))
    if not input_files:
        raise FileNotFoundError(f"No files found matching pattern: {search_path}")
    print(f"Found {len(input_files)} input files.")
    for f in input_files:
        if not os.path.isfile(f):
            print(f"Warning: Input file not found or is not a file: {f}")
    return input_files


def create_design_matrix(input_files):
    """Creates a simple intercept-only design matrix."""
    design_matrix = pd.DataFrame([1] * len(input_files), columns=['intercept'])
    print("Design Matrix:")
    print(design_matrix.head())
    return design_matrix


def load_nifti_with_mmap(filepath, file_description="file"):
    """Loads a NIfTI image using memory mapping if possible."""
    if not filepath or not os.path.exists(filepath):
        print(f"Warning: {file_description.capitalize()} file path not provided or not found: {filepath}")
        return None
    try:
        img = nib.load(filepath, mmap=True)
        print(f"Loaded {file_description} with memory mapping: {filepath}")
        print(f"  Shape: {img.shape}, Dtype: {img.get_data_dtype()}, Affine:\n{img.affine}")
        return img
    except Exception as e:
        print(f"Warning: Could not load {file_description} '{filepath}'. Error: {e}")
        return None


def compute_intersection_mask(user_mask_img, input_files, map_dir, save_maps_flag, std_tol=1e-6):
    """
    Computes an intersection mask including only voxels within the user mask
    that have finite values AND non-zero standard deviation across all input files.
    """
    if user_mask_img is None:
        print("Warning: No user mask provided. Cannot compute intersection mask.")
        return None

    print("\nComputing intersection mask (User Mask AND Finite AND Non-Zero Variance)...")
    try:
        user_mask_data = user_mask_img.get_fdata().astype(bool)
        ref_affine = user_mask_img.affine
        ref_shape = user_mask_img.shape
        n_voxels_user = np.sum(user_mask_data)
        print(f"Initial user mask contains {n_voxels_user} voxels.")

        # 1. Find voxels finite across all files AND within user mask
        finite_mask_all_files = np.copy(user_mask_data) # Start with user mask
        for i, f_path in enumerate(input_files):
            print(f"  Checking finiteness in file {i+1}/{len(input_files)}: {os.path.basename(f_path)}", end='\r')
            try:
                img = nib.load(f_path, mmap=True)
                if not np.allclose(img.affine, ref_affine) or img.shape != ref_shape:
                    print(f"\nWarning: Shape/Affine mismatch for {f_path}. Skipping file for finiteness check.")
                    continue
                data = img.get_fdata()
                finite_mask_this_file = np.isfinite(data)
                finite_mask_all_files &= finite_mask_this_file
            except Exception as e:
                print(f"\nError loading/processing file {f_path} for finiteness check: {e}")
        print("\nFinished finiteness check.")
        n_voxels_finite_in_user = np.sum(finite_mask_all_files)
        print(f"Found {n_voxels_finite_in_user} voxels within user mask with finite values across all files.")

        if n_voxels_finite_in_user == 0:
            print("ERROR: No voxels found with finite values across all files within the user mask.")
            return None

        # 2. Extract data for these voxels and check standard deviation
        print("Extracting data for variance check (this might take memory)...")
        preliminary_mask_img = new_img_like(user_mask_img, finite_mask_all_files.astype(np.int8))
        try:
             # This step requires loading masked data - potential memory bottleneck
             masked_data_all_subjects = apply_mask(input_files, preliminary_mask_img)
        except Exception as e:
             print(f"\nERROR during apply_mask for variance check: {e}")
             if "memory" in str(e).lower():
                  print("This may indicate insufficient memory to load masked data for variance check.")
             return None # Cannot proceed if data extraction fails

        print(f"Calculating standard deviation across {masked_data_all_subjects.shape[0]} subjects for {masked_data_all_subjects.shape[1]} voxels...")
        with warnings.catch_warnings():
             warnings.simplefilter("ignore", category=RuntimeWarning)
             std_dev_per_voxel = np.std(masked_data_all_subjects, axis=0)

        non_zero_std_mask_1d = std_dev_per_voxel > std_tol
        n_voxels_nonzero_std = np.sum(non_zero_std_mask_1d)
        print(f"Found {n_voxels_nonzero_std} voxels with non-zero standard deviation (tolerance > {std_tol}).")

        if n_voxels_nonzero_std == 0:
             print("ERROR: All voxels within the finite user mask have zero standard deviation across subjects!")
             return None

        # 3. Create the final mask
        final_mask_data = np.zeros(ref_shape, dtype=bool)
        final_mask_data[finite_mask_all_files] = non_zero_std_mask_1d
        final_voxel_count = np.sum(final_mask_data)
        print(f"Final intersection mask contains {final_voxel_count} voxels.")

        intersection_mask_img = new_img_like(user_mask_img, final_mask_data.astype(np.int8))

        if save_maps_flag:
            mask_path_out = os.path.join(map_dir, 'computed_intersection_mask_finite_nonzero_std.nii.gz')
            intersection_mask_img.to_filename(mask_path_out)
            print(f"Saved final intersection mask to: {mask_path_out}")

        del masked_data_all_subjects, std_dev_per_voxel, preliminary_mask_img
        gc.collect()

        return intersection_mask_img

    except Exception as e:
        print(f"ERROR during intersection mask computation: {e}")
        return None


# --- GLM Function (Keeps memory/memory_level arguments for SecondLevelModel) ---
def run_second_level_glm(input_files, design_matrix, smoothing_fwhm, final_mask_img, memory_level, memory_path, n_jobs_glm):
    """Fits the second-level GLM using the final intersection mask."""
    print(f"\nFitting Second Level GLM (FWHM={smoothing_fwhm}mm)...")
    print(f"  Using final intersection mask: {'YES' if final_mask_img else 'NO (Potential Issue!)'}")
    if final_mask_img:
        print(f"  Mask contains {np.sum(final_mask_img.get_fdata() > 0)} voxels.")
    else:
         print("ERROR: Cannot run GLM without a valid mask.")
         return None
    print(f"  n_jobs for GLM fitting: {n_jobs_glm}")
    print(f"  Cache dir for GLM: {memory_path}, Level: {memory_level}") # Show caching info
    second_level_model = SecondLevelModel(
        smoothing_fwhm=smoothing_fwhm,
        mask_img=final_mask_img, # Use the final, cleaned mask
        memory=memory_path,     # Pass cache path to SecondLevelModel
        memory_level=memory_level,# Pass cache level to SecondLevelModel
        n_jobs=n_jobs_glm,
        verbose=1
    )
    try:
        second_level_model.fit(input_files, design_matrix=design_matrix)
        print("GLM fitting completed.")
    except Exception as e:
        print(f"ERROR during SecondLevelModel fitting: {e}")
        raise
    return second_level_model


# --- Permutation Function (memory/memory_level args REMOVED from signature) ---
def run_non_parametric_inference(input_files, design_matrix, smoothing_fwhm, n_perm,
                                 threshold, tfce, two_sided, n_jobs_perm, final_mask_img):
    """Runs the non-parametric permutation testing using the final intersection mask."""
    print(f"\nStarting Non-Parametric Inference...")
    print(f"  Using final intersection mask: {'YES' if final_mask_img else 'NO (Potential Issue!)'}")
    if final_mask_img:
         print(f"  Mask contains {np.sum(final_mask_img.get_fdata() > 0)} voxels.")
    else:
         print("ERROR: Cannot run non-parametric inference without a valid mask.")
         return None
    print(f"  Smoothing FWHM: {smoothing_fwhm}mm")
    print(f"  Permutations: {n_perm}")
    print(f"  Cluster/TFCE threshold: {threshold}")
    print(f"  TFCE enabled: {tfce}")
    print(f"  Two-sided test: {two_sided}")
    print(f"  Jobs for permutations: {n_jobs_perm} (Using n_jobs > 1 significantly increases memory!)")

    if n_jobs_perm > 1:
        print("Warning: Using n_jobs > 1 for permutations can consume large amounts of memory.")

    try:
        # memory and memory_level args are NOT passed to nilearn's non_parametric_inference
        out_dict = non_parametric_inference(
            second_level_input=input_files,
            design_matrix=design_matrix,
            model_intercept=True,
            mask=final_mask_img, # Use the final, cleaned mask
            n_perm=n_perm,
            two_sided_test=two_sided,
            smoothing_fwhm=smoothing_fwhm,
            n_jobs=10,
            threshold=threshold,
            tfce=tfce,
            verbose=2
        )
        print("Non-Parametric Inference finished.")
    except Exception as e:
        print(f"ERROR during non_parametric_inference: {e}")
        raise
    return out_dict

# --- Other functions (plotting, etc.) ---
# ... (compute_glm_contrast, plot_and_save_glm, plot_and_save_permutation_results etc. as they were) ...
def compute_glm_contrast(model, contrast_def, output_type='z_score'):
    """Computes the contrast map from the fitted GLM."""
    print(f"Computing GLM contrast '{contrast_def}' ({output_type})...")
    try:
        stat_map = model.compute_contrast(
            second_level_contrast=contrast_def,
            output_type=output_type,
        )
        print("Contrast computation completed.")
    except Exception as e:
        print(f"ERROR computing contrast: {e}")
        raise
    return stat_map


def plot_and_save_glm(z_map, p_threshold, plot_dir, map_dir, save_plots_flag, save_maps_flag, prefix='glm'):
    """Plots and saves the GLM results."""
    if not (save_plots_flag or save_maps_flag):
        print("Skipping GLM plot/save.")
        return

    threshold_val = norm.isf(p_threshold)
    print(f"Processing GLM results (unc. p < {p_threshold}, z > {threshold_val:.3f})...")

    if save_plots_flag:
        try:
            plot_path = os.path.join(plot_dir, f'{prefix}_zmap_glass_uncp{p_threshold}.png')
            display = plotting.plot_glass_brain(
                z_map,
                threshold=threshold_val,
                colorbar=True,
                display_mode='z',
                plot_abs=False,
                title=f'Second Level GLM (unc p<{p_threshold})',
                output_file=plot_path # Saves directly, potentially less memory
            )
            print(f"Saved GLM plot to: {plot_path}")
            plt.close('all') # Close all open figures
            del display # Explicitly delete the display object
        except Exception as e:
            print(f"ERROR saving GLM plot: {e}")
            plt.close('all') # Ensure figures are closed on error
        finally:
            gc.collect()

    if save_maps_flag:
        try:
            map_path = os.path.join(map_dir, f'{prefix}_zmap_uncp{p_threshold}.nii.gz')
            z_map.to_filename(map_path)
            print(f"Saved GLM z-map to: {map_path}")
        except Exception as e:
            print(f"ERROR saving GLM map: {e}")

def plot_and_save_permutation_results(out_dict, logp_threshold, plot_dir, map_dir, bg_img, save_plots_flag, save_maps_flag):
    """Plots and saves the permutation test results."""
    if out_dict is None: # Check if results dict is None
         print("Skipping permutation plotting/saving because results dictionary is None.")
         return
    if not (save_plots_flag or save_maps_flag):
        print("Skipping permutation plot/save.")
        return

    print(f"\nProcessing Permutation results (-log10(p) > {logp_threshold})...")

    IMAGES = {}
    TITLES = {}

    # Check which maps are available in the output dictionary
    if 't' in out_dict and out_dict['t'] is not None:
        pass # print("Raw T-map available in output.")
    if 'logp_max_t' in out_dict and out_dict['logp_max_t'] is not None:
        IMAGES['voxel_corr'] = out_dict['logp_max_t']
        TITLES['voxel_corr'] = 'Permutation Test\n(Voxel-Level FWE corrected -log10p)'
    if 'logp_max_size' in out_dict and out_dict['logp_max_size'] is not None:
        IMAGES['cluster_size_corr'] = out_dict['logp_max_size']
        TITLES['cluster_size_corr'] = 'Permutation Test\n(Cluster-Size FWE corrected -log10p)'
    if 'logp_max_mass' in out_dict and out_dict['logp_max_mass'] is not None:
        IMAGES['cluster_mass_corr'] = out_dict['logp_max_mass']
        TITLES['cluster_mass_corr'] = 'Permutation Test\n(Cluster-Mass FWE corrected -log10p)'
    if 'logp_tfce' in out_dict and out_dict['logp_tfce'] is not None: # Nilearn >= 0.9 name change
        IMAGES['tfce_corr'] = out_dict['logp_tfce']
        TITLES['tfce_corr'] = 'Permutation Test\n(TFCE FWE corrected -log10p)'
    elif 'logp_max_tfce' in out_dict and out_dict['logp_max_tfce'] is not None: # Older Nilearn name
        IMAGES['tfce_corr'] = out_dict['logp_max_tfce']
        TITLES['tfce_corr'] = 'Permutation Test\n(TFCE FWE corrected -log10p)'


    # --- Plotting Glass Brains ---
    n_images = len(IMAGES)
    if n_images > 0 and save_plots_flag:
        n_cols = min(n_images, 2)
        n_rows = (n_images + n_cols - 1) // n_cols
        fig, axes = plt.subplots(figsize=(6 * n_cols, 5 * n_rows), nrows=n_rows, ncols=n_cols, squeeze=False)
        axes = axes.flatten() # Ensure axes is always flat

        print("Plotting permutation results (glass brain)...")
        img_counter = 0
        for key, img in IMAGES.items():
            if img is None: continue # Skip if map is None
            ax = axes[img_counter]
            try:
                plotting.plot_glass_brain(
                    img,
                    colorbar=True,
                    vmax=None, # Auto-scale vmax based on data
                    display_mode='z',
                    plot_abs=False,
                    axes=ax,
                    threshold=logp_threshold, # Threshold on -log10(p)
                    title=TITLES[key]
                )
            except Exception as e:
                 print(f"ERROR plotting glass brain for {key}: {e}")
            img_counter += 1

        # Hide unused axes
        for i in range(img_counter, len(axes)):
            axes[i].axis('off')

        fig.suptitle(f'Permutation Test Results (-log10 p-values > {logp_threshold})')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

        try:
            plot_path = os.path.join(plot_dir, f'permutation_summary_logp{logp_threshold}.png')
            fig.savefig(plot_path)
            print(f"Saved permutation summary plot to: {plot_path}")
        except Exception as e:
            print(f"ERROR saving permutation summary plot: {e}")
        finally:
             plt.close(fig) # Close figure to free memory
             gc.collect()

    # --- Save Maps ---
    if save_maps_flag:
        print("Saving permutation maps...")
        for key, img in IMAGES.items():
             if img is None: continue # Skip if map is None
             try:
                map_path = os.path.join(map_dir, f'perm_{key}_logp_thr{logp_threshold}.nii.gz')
                img.to_filename(map_path)
                print(f"Saved {key} permutation map to: {map_path}")
             except Exception as e:
                  print(f"ERROR saving map {key}: {e}")

    # --- Plotting Stat Map on BG (e.g., TFCE if available) ---
    tfce_key = 'tfce_corr' # Default key
    if tfce_key in IMAGES and IMAGES[tfce_key] is not None and bg_img and save_plots_flag:
        print(f"Plotting {tfce_key} map on background image...")
        try:
            stat_map_img = IMAGES[tfce_key]
            plot_path_stat = os.path.join(plot_dir, f'perm_{tfce_key}_statmap_logp{logp_threshold}.png')
            display_stat = plotting.plot_stat_map(
                stat_map_img,
                display_mode='ortho', # Use 'x', 'y', 'z', or 'ortho'
                threshold=logp_threshold,
                bg_img=bg_img,
                title=f'{TITLES[tfce_key]}\n(-log10p > {logp_threshold})',
                cut_coords=None, # Auto cut coords can be better than fixed ones
                output_file=plot_path_stat # Save directly
            )
            print(f"Saved {tfce_key} stat map plot to: {plot_path_stat}")
            plt.close('all') # Close all figures
            del display_stat # Explicitly delete

        except Exception as e:
            print(f"Warning: Could not plot/save {tfce_key} stat map. Error: {e}")
            plt.close('all') # Ensure figures are closed on error
        finally:
            gc.collect()
    elif tfce_key in IMAGES and IMAGES[tfce_key] is not None and not bg_img and save_plots_flag:
         print(f"Skipping stat map plot for {tfce_key} because no background image (--bg_img) was provided.")

    # --- Clean up ---
    del IMAGES, TITLES # Delete intermediate plot helpers
    gc.collect()


# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(
        description="Run Group-Level fMRI Analysis with GLM and Permutation Testing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show defaults in help
    )

    # --- Input/Output Arguments ---
    parser.add_argument('--input_dir', type=str, required=True,
                        help="Directory containing the first-level contrast maps (e.g., z-maps).")
    parser.add_argument('--input_pattern', type=str, default='*zmap*.nii.gz',
                        help="Glob pattern to find input files within input_dir. Use quotes if pattern contains '*'.")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Directory to save the results (maps and plots). Will be created if it doesn't exist.")
    parser.add_argument('--mask', type=str, required=True,
                        help="Path to a NIfTI mask file (e.g., GM mask). REQUIRED for robust NaN/variance handling via intersection.")
    parser.add_argument('--bg_img', type=str, default=None,
                        help="Path to a NIfTI background image for plotting stat maps (e.g., MNI template).")
    parser.add_argument('--cache_dir', type=str, default=None, # Kept for SecondLevelModel
                        help="Path to a directory for Nilearn's caching mechanism (used by GLM fitting). Recommended for speeding up re-runs.")

    # --- GLM Arguments ---
    glm_group = parser.add_argument_group('GLM Options')
    glm_group.add_argument('--glm_contrast', type=str, default='intercept',
                           help="Contrast definition for the GLM.")
    glm_group.add_argument('--glm_p_threshold', type=float, default=0.001,
                           help="Uncorrected p-value threshold for plotting GLM results.")
    glm_group.add_argument('--n_jobs_glm', type=int, default=1,
                           help="Number of CPU cores for GLM fitting (usually 1 is sufficient).")


    # --- Permutation Arguments ---
    perm_group = parser.add_argument_group('Permutation Test Options')
    perm_group.add_argument('--run_permutation', action='store_true',
                            help="Flag to run non-parametric permutation testing (can be memory/time intensive).")
    perm_group.add_argument('--n_permutations', type=int, default=1000,
                            help="Number of permutations. Increase for robust results (e.g., 5000+), requires more time/memory.")
    perm_group.add_argument('--perm_cluster_threshold', type=float, default=0.001,
                            help="Threshold (p-value) used for TFCE/cluster formation during permutations. Does NOT set final significance.")
    perm_group.add_argument('--perm_logp_threshold', type=float, default=1.301,
                            help="Plotting/saving threshold for corrected permutation results (-log10(p)). Default: 1.301 (~ p < 0.05).")
    perm_group.add_argument('--tfce', action=argparse.BooleanOptionalAction, default=True,
                            help="Use Threshold-Free Cluster Enhancement (TFCE). Disable with --no-tfce.")
    perm_group.add_argument('--two_sided', action=argparse.BooleanOptionalAction, default=True,
                            help="Perform a two-sided test. Disable with --no-two-sided for one-sided.")
    perm_group.add_argument('--n_jobs_perm', type=int, default=1,
                           help="Number of CPU cores for permutation testing. IMPORTANT: >1 increases memory load significantly. Start with 1.")

    # --- Performance/Memory Arguments ---
    perf_group = parser.add_argument_group('Performance and Memory Options')
    perf_group.add_argument('--smoothing_fwhm', type=float, default=3.0,
                           help="Smoothing FWHM (mm) applied within analysis models. Set to 0 if input data is already smoothed.")
    perf_group.add_argument('--std_tol', type=float, default=1e-6,
                           help="Tolerance for standard deviation check when creating intersection mask.")
    perf_group.add_argument('--memory_level', type=int, default=1, choices=[0, 1, 2], # Kept for SecondLevelModel
                           help="Nilearn caching level (0=None, 1=CacheIntermediate, 2=CacheAll) for GLM fitting. Requires --cache_dir.")

    # --- Saving Arguments ---
    save_group = parser.add_argument_group('Saving Options')
    save_group.add_argument('--save_plots', action=argparse.BooleanOptionalAction, default=True,
                           help="Save plots of the results.")
    save_group.add_argument('--save_maps', action=argparse.BooleanOptionalAction, default=True,
                           help="Save resulting NIfTI maps.")
    save_group.add_argument('--save_perm_dict', action='store_true',
                           help="Save the full output dictionary from permutation testing as a pickle file.")


    # --- Parse Arguments ---
    args = parser.parse_args()

    # --- Validate Arguments ---
    if not args.mask:
        parser.error("--mask argument is required.")
    # Check cache dir only if memory level > 0 (useful for GLM)
    if args.memory_level > 0 and not args.cache_dir:
        parser.error("--cache_dir is required when --memory_level > 0 (for GLM caching).")
    if args.n_jobs_perm < 1 and args.n_jobs_perm != -1:
         parser.error("--n_jobs_perm must be >= 1 or -1.")
    if args.n_jobs_glm < 1 and args.n_jobs_glm != -1:
         parser.error("--n_jobs_glm must be >= 1 or -1.")
    if args.n_jobs_perm == -1:
         import multiprocessing
         num_cores = multiprocessing.cpu_count()
         print(f"Warning: --n_jobs_perm set to -1. Using all available cores ({num_cores}). This may consume significant memory.")
         args.n_jobs_perm = num_cores
    if args.n_jobs_glm == -1:
         args.n_jobs_glm = 1
         print("Info: --n_jobs_glm set to -1, defaulting to 1 core for GLM fitting.")


    # --- Setup ---
    plot_dir, map_dir = setup_directories(args)
    second_level_input_files = load_input_files(args.input_dir, args.input_pattern)
    design_matrix = create_design_matrix(second_level_input_files)

    # --- Load User Mask ---
    user_mask_img = load_nifti_with_mmap(args.mask, "user mask")
    if user_mask_img is None:
         print(f"ERROR: Could not load the required user mask file: {args.mask}")
         exit(1)

    # --- Compute Intersection Mask (Finite + Non-Zero Std Dev) ---
    final_analysis_mask_img = compute_intersection_mask(
        user_mask_img,
        second_level_input_files,
        map_dir,
        args.save_maps,
        args.std_tol # Pass tolerance
    )
    if final_analysis_mask_img is None:
         print("ERROR: Failed to create a valid intersection mask. Exiting.")
         exit(1)

    # --- Background Image for Plotting ---
    bg_img_loaded = None
    if args.bg_img:
        bg_img_loaded = load_nifti_with_mmap(args.bg_img, "background image")

    # --- Run GLM Analysis ---
    # Pass the final_analysis_mask_img and caching arguments
    second_level_model = run_second_level_glm(
        second_level_input_files,
        design_matrix,
        args.smoothing_fwhm,
        final_analysis_mask_img, # Use the intersection mask
        args.memory_level,      # Pass GLM cache level
        args.cache_dir,         # Pass GLM cache path
        args.n_jobs_glm
    )
    if second_level_model is None:
         print("ERROR: GLM fitting failed. Exiting.")
         exit(1)

    z_map = compute_glm_contrast(second_level_model, args.glm_contrast)

    plot_and_save_glm(
        z_map,
        args.glm_p_threshold,
        plot_dir,
        map_dir,
        args.save_plots,
        args.save_maps,
        prefix=f'glm_{args.glm_contrast.replace(" ", "_")}'
    )

    # --- Clean up GLM variables ---
    print("Cleaning up GLM variables...")
    del z_map, second_level_model
    gc.collect()

    # --- Run Permutation Testing (Optional) ---
    perm_results = None
    if args.run_permutation:
        # Pass the final_analysis_mask_img
        # DO NOT pass memory_level or memory_path here
        perm_results = run_non_parametric_inference(
            input_files=second_level_input_files,
            design_matrix=design_matrix,
            smoothing_fwhm=args.smoothing_fwhm,
            n_perm=args.n_permutations,
            threshold=args.perm_cluster_threshold,
            tfce=args.tfce,
            two_sided=args.two_sided,
            n_jobs_perm=args.n_jobs_perm,
            final_mask_img=final_analysis_mask_img # Use the intersection mask
        )

        # --- Save Permutation Dictionary (Optional) ---
        if args.save_perm_dict and perm_results is not None:
            pickle_path = os.path.join(args.output_dir, 'permutation_results_dict.pkl')
            print(f"\nSaving permutation results dictionary to: {pickle_path}")
            try:
                with open(pickle_path, 'wb') as f_pkl:
                    pickle.dump(perm_results, f_pkl, protocol=pickle.HIGHEST_PROTOCOL)
                print("Successfully saved permutation dictionary.")
            except Exception as e:
                print(f"ERROR saving permutation dictionary: {e}")
                if os.path.exists(pickle_path):
                    try: os.remove(pickle_path)
                    except OSError: pass

        # --- Plot and Save Permutation Maps ---
        if perm_results is not None:
            plot_and_save_permutation_results(
                perm_results,
                args.perm_logp_threshold,
                plot_dir,
                map_dir,
                bg_img=bg_img_loaded,
                save_plots_flag=args.save_plots,
                save_maps_flag=args.save_maps
            )

        # --- Clean up permutation variables ---
        print("Cleaning up permutation variables...")
        if perm_results is not None:
            del perm_results
        gc.collect()

    # --- Final Cleanup ---
    print("Final cleanup...")
    del second_level_input_files, design_matrix, user_mask_img, final_analysis_mask_img, bg_img_loaded
    gc.collect()
    print("\nAnalysis finished.")


if __name__ == "__main__":
    # Check if required libraries are importable
    try:
        import nilearn
        import nibabel
        import pandas
        import scipy
        import matplotlib
        import numpy
        import pickle
        import warnings
    except ImportError as e:
        print(f"Error: Missing required library: {e}")
        print("Please install the necessary libraries (e.g., pip install nilearn nibabel pandas scipy matplotlib numpy)")
        exit(1)

    main()
