#!/usr/bin/env python
from nilearn import image as niimg
from nilearn.image import resample_to_img, binarize_img
import nibabel as nib
import numpy as np
import os
import sys
import gc
import traceback
from brainiak.searchlight.searchlight import Searchlight
from mpi4py import MPI

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def mask_usplus(A, msk, myrad, mask_params):
    """
    Calculate correlation for US+ trials across two runs
    using subject-specific trial counts from mask_params
    """
    try:
        if not mask_params:
            return np.nan
            
        # Extract mask parameters
        usplusrun1 = mask_params['usplusrun1']
        usminusrun1 = mask_params['usminusrun1']
        usplusrun2 = mask_params['usplusrun2']
        usminusrun2 = mask_params['usminusrun2']
        total_run1 = mask_params['total_run1'] 
        total_run2 = mask_params['total_run2']
        
        A = A[0]
        A = A.reshape((-1, A.shape[-1]))
        corrmat = np.corrcoef(A.T)
        corrmat = np.arctanh(corrmat)  # Fisher z-transform
        
        corrmat = np.tril(corrmat, k=-1)
        run1 = corrmat[:32, :32]
        run2 = corrmat[32:, 32:]
        
        # Create dynamic mask based on actual trial counts
        usplus1 = np.tril(np.zeros((32, 32)))
        usplus1[:usplusrun1, :usplusrun1] = 1
        usplus1 = np.tril(usplus1, k=-1)
        
        usplus2 = np.tril(np.zeros((32, 32)))
        usplus2[:usplusrun2, :usplusrun2] = 1
        usplus2 = np.tril(usplus2, k=-1)
        
        # Calculate mean correlation for US+ trials in run 1
        usplus_run1 = np.multiply(usplus1, run1)
        usplus_run1_mask = np.where(usplus1 != 0)
        if len(usplus_run1_mask[0]) > 0:
            usplus_run1 = usplus_run1[usplus_run1_mask].mean()
        else:
            usplus_run1 = np.nan
        
        # Calculate mean correlation for US+ trials in run 2
        usplus_run2 = np.multiply(usplus2, run2)
        usplus_run2_mask = np.where(usplus2 != 0)
        if len(usplus_run2_mask[0]) > 0:
            usplus_run2 = usplus_run2[usplus_run2_mask].mean()
        else:
            usplus_run2 = np.nan
        
        # Average across runs
        usplusavg = ((usplus_run1 + usplus_run2) / 2)
        
        # Cleanup
        del A, corrmat, run1, run2, usplus1, usplus2
        gc.collect()
        
        return usplusavg
    except Exception as e:
        if rank == 0:
            print(f"Error in mask_usplus: {str(e)}")
            traceback.print_exc()
        return np.nan

def mask_usminus(A, msk, myrad, mask_params):
    """
    Calculate correlation for US- trials across two runs
    using subject-specific trial counts from mask_params
    """
    try:
        if not mask_params:
            return np.nan
            
        # Extract mask parameters
        usplusrun1 = mask_params['usplusrun1']
        usminusrun1 = mask_params['usminusrun1']
        usplusrun2 = mask_params['usplusrun2']
        usminusrun2 = mask_params['usminusrun2']
        total_run1 = mask_params['total_run1']
        total_run2 = mask_params['total_run2']
        
        A = A[0]
        A = A.reshape((-1, A.shape[-1]))
        corrmat = np.corrcoef(A.T)
        corrmat = np.arctanh(corrmat)  # Fisher z-transform
        
        corrmat = np.tril(corrmat, k=-1)
        run1 = corrmat[:32, :32]
        run2 = corrmat[32:, 32:]
        
        # Create dynamic mask based on actual trial counts
        usminus1 = np.tril(np.zeros((32, 32)))
        usminus1[usplusrun1:usminusrun1, usplusrun1:usminusrun1] = 1
        usminus1 = np.tril(usminus1, k=-1)
        
        usminus2 = np.tril(np.zeros((32, 32)))
        usminus2[usplusrun2:usminusrun2, usplusrun2:usminusrun2] = 1
        usminus2 = np.tril(usminus2, k=-1)
        
        # Calculate mean correlation for US- trials in run 1
        usminus_run1 = np.multiply(usminus1, run1)
        usminus_run1_mask = np.where(usminus1 != 0)
        if len(usminus_run1_mask[0]) > 0:
            usminus_run1 = usminus_run1[usminus_run1_mask].mean()
        else:
            usminus_run1 = np.nan
        
        # Calculate mean correlation for US- trials in run 2
        usminus_run2 = np.multiply(usminus2, run2)
        usminus_run2_mask = np.where(usminus2 != 0)
        if len(usminus_run2_mask[0]) > 0:
            usminus_run2 = usminus_run2[usminus_run2_mask].mean()
        else:
            usminus_run2 = np.nan
        
        # Average across runs
        usminusavg = ((usminus_run1 + usminus_run2) / 2)
        
        # Cleanup
        del A, corrmat, run1, run2, usminus1, usminus2
        gc.collect()
        
        return usminusavg
    except Exception as e:
        if rank == 0:
            print(f"Error in mask_usminus: {str(e)}")
            traceback.print_exc()
        return np.nan

def get_trial_counts_from_csv(subject_id, csv_path):
    """
    Get trial counts from a CSV file that stores mask sizes.
    Calculate the correct position of US+ and US- trials based on CS+ and CS- sizes.
    
    CSV format should have columns: subject, run, condition, size
    Each run has a total of 32 trials, and trials are ordered as: CS+, CS-, US+, US-
    
    Parameters:
    -----------
    subject_id : str
        Subject ID to get counts for
    csv_path : str
        Path to CSV file with mask sizes
        
    Returns:
    --------
    dict
        Dictionary with mask parameters for the searchlight analysis
    """
    try:
        if rank == 0:
            print(f"Getting trial counts for {subject_id} from CSV: {csv_path}")
        
        # Load CSV file - only on rank 0
        mask_params = None
        if rank == 0:
            if not os.path.exists(csv_path):
                print(f"ERROR: Mask size CSV file not found: {csv_path}")
                return None
                
            try:
                import pandas as pd
                mask_df = pd.read_csv(csv_path)
                
                # Filter to this subject only
                subject_df = mask_df[mask_df['subject'] == subject_id]
                
                if len(subject_df) == 0:
                    print(f"ERROR: No mask data for {subject_id} in {csv_path}")
                    return None
                
                # Initialize parameters
                mask_params = {}
                
                # Get run 1 parameters
                run1_df = subject_df[subject_df['run'] == 1]
                
                # Get CS+ size for run 1
                csplus_run1 = run1_df[run1_df['condition'] == 'CS+']['size'].values
                if len(csplus_run1) > 0:
                    csplus_run1 = int(csplus_run1[0])
                else:
                    print(f"WARNING: No CS+ data for {subject_id} run 1, using default of 8")
                    csplus_run1 = 8
                
                # Get CS- size for run 1
                csminus_run1 = run1_df[run1_df['condition'] == 'CS-']['size'].values
                if len(csminus_run1) > 0:
                    csminus_run1 = int(csminus_run1[0])
                else:
                    print(f"WARNING: No CS- data for {subject_id} run 1, using default of 8")
                    csminus_run1 = 8
                
                # Calculate cumulative index for CS- (CS+ end)
                csminus_run1_start = csplus_run1
                # Calculate cumulative index for CS- end
                csminus_run1_end = csminus_run1_start + csminus_run1
                
                # Get US+ size for run 1
                usplus_run1 = run1_df[run1_df['condition'] == 'US+']['size'].values
                if len(usplus_run1) > 0:
                    usplus_run1 = int(usplus_run1[0])
                else:
                    print(f"WARNING: No US+ data for {subject_id} run 1, using default of 6")
                    usplus_run1 = 6
                
                # Calculate cumulative index for US+ end
                usplus_run1_end = csminus_run1_end + usplus_run1
                
                # Get US- size for run 1
                usminus_run1 = run1_df[run1_df['condition'] == 'US-']['size'].values
                if len(usminus_run1) > 0:
                    usminus_run1 = int(usminus_run1[0])
                else:
                    print(f"WARNING: No US- data for {subject_id} run 1, using default of 8")
                    usminus_run1 = 8
                
                # Calculate cumulative index for US- end
                usminus_run1_end = usplus_run1_end + usminus_run1
                
                # Store parameters for run 1
                mask_params['csminusrun1'] = csminus_run1_end  # End index of CS- (start of US+)
                mask_params['usplusrun1'] = usplus_run1_end    # End index of US+ (start of US-)
                mask_params['usminusrun1'] = usminus_run1_end  # End index of US-
                mask_params['total_run1'] = 32  # Fixed size of 32
                
                # Get run 2 parameters
                run2_df = subject_df[subject_df['run'] == 2]
                
                # Get CS+ size for run 2
                csplus_run2 = run2_df[run2_df['condition'] == 'CS+']['size'].values
                if len(csplus_run2) > 0:
                    csplus_run2 = int(csplus_run2[0])
                else:
                    print(f"WARNING: No CS+ data for {subject_id} run 2, using default of 8")
                    csplus_run2 = 8
                
                # Get CS- size for run 2
                csminus_run2 = run2_df[run2_df['condition'] == 'CS-']['size'].values
                if len(csminus_run2) > 0:
                    csminus_run2 = int(csminus_run2[0])
                else:
                    print(f"WARNING: No CS- data for {subject_id} run 2, using default of 8")
                    csminus_run2 = 8
                
                # Calculate cumulative index for CS- (CS+ end)
                csminus_run2_start = csplus_run2
                # Calculate cumulative index for CS- end
                csminus_run2_end = csminus_run2_start + csminus_run2
                
                # Get US+ size for run 2
                usplus_run2 = run2_df[run2_df['condition'] == 'US+']['size'].values
                if len(usplus_run2) > 0:
                    usplus_run2 = int(usplus_run2[0])
                else:
                    print(f"WARNING: No US+ data for {subject_id} run 2, using default of 6")
                    usplus_run2 = 6
                
                # Calculate cumulative index for US+ end
                usplus_run2_end = csminus_run2_end + usplus_run2
                
                # Get US- size for run 2
                usminus_run2 = run2_df[run2_df['condition'] == 'US-']['size'].values
                if len(usminus_run2) > 0:
                    usminus_run2 = int(usminus_run2[0])
                else:
                    print(f"WARNING: No US- data for {subject_id} run 2, using default of 8")
                    usminus_run2 = 8
                
                # Calculate cumulative index for US- end
                usminus_run2_end = usplus_run2_end + usminus_run2
                
                # Store parameters for run 2
                mask_params['csminusrun2'] = csminus_run2_end  # End index of CS- (start of US+)
                mask_params['usplusrun2'] = usplus_run2_end    # End index of US+ (start of US-)
                mask_params['usminusrun2'] = usminus_run2_end  # End index of US-
                mask_params['total_run2'] = 32  # Fixed size of 32
                
                print(f"Trial positions for {subject_id}:")
                print(f"  Run 1: CS+ = 0-{csplus_run1}, CS- = {csplus_run1}-{csminus_run1_end}, " +
                      f"US+ = {csminus_run1_end}-{usplus_run1_end}, US- = {usplus_run1_end}-{usminus_run1_end}")
                print(f"  Run 2: CS+ = 0-{csplus_run2}, CS- = {csplus_run2}-{csminus_run2_end}, " +
                      f"US+ = {csminus_run2_end}-{usplus_run2_end}, US- = {usplus_run2_end}-{usminus_run2_end}")
                
            except Exception as e:
                print(f"Error reading mask CSV: {str(e)}")
                traceback.print_exc()
                return None
                
        # Broadcast parameters to all ranks
        mask_params = comm.bcast(mask_params, root=0)
        return mask_params
        
    except Exception as e:
        if rank == 0:
            print(f"Error getting trial counts from CSV for {subject_id}: {str(e)}")
            traceback.print_exc()
        return None

def prepare_mask(subject_id, lss_file, anat_root, func_root, output_root):
    """
    Prepare and resample the gray matter mask for searchlight analysis
    """
    if rank == 0:
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_root, exist_ok=True)
            
            # Define paths
            mask_path = os.path.join(anat_root, subject_id, 'gm_cropped.nii.gz')

            
            
            # Resample mask to functional space
            mask_resampled = resample_to_img(mask_path, lss_file, interpolation='nearest')
            mask_resampled_bin = binarize_img(mask_resampled)
            
            # Get mask data
            mask_data = mask_resampled_bin.get_fdata()
            affine_mat = mask_resampled.affine
            
            print(f"Mask prepared for {subject_id}: {np.sum(mask_data)} voxels in mask")
            return mask_data, mask_resampled_bin, affine_mat
            
        except Exception as e:
            print(f"Error preparing mask for {subject_id}: {str(e)}")
            traceback.print_exc()
            comm.Abort(1)
    else:
        mask_data, mask_resampled_bin, affine_mat = None, None, None
    
    # Broadcast mask data to all ranks
    mask_data = comm.bcast(mask_data, root=0)
    mask_resampled_bin = comm.bcast(mask_resampled_bin, root=0)
    affine_mat = comm.bcast(affine_mat, root=0)
    
    return mask_data, mask_resampled_bin, affine_mat

def run_searchlight_analysis(subject_id, lss_file, mask_data, mask_resampled_bin, affine_mat, 
                            analysis_type, output_path, mask_params, sl_radius=2, pool_size=4):
    """
    Run the searchlight analysis for a specific subject and condition
    """
    try:
        # Load the 4D LSS data
        if rank == 0:
            print(f"Loading 4D data for {subject_id} from {lss_file}")
        
        lss_data = nib.load(lss_file).get_fdata()
        
        # Set up the searchlight
        sl = Searchlight(sl_rad=sl_radius, max_blk_edge=5, min_active_voxels_proportion=0.5)
        
        if rank == 0:
            print(f"Setting up searchlight for {subject_id}")
            print(f"Input data shape: {lss_data.shape}")
            print(f"Input mask shape: {mask_data.shape}")
        
        # Distribute data
        sl.distribute([lss_data], mask_data)
        
        # Broadcast mask parameters to all ranks
        sl.broadcast(mask_params)
        
        # Run appropriate searchlight analysis
        if rank == 0:
            print(f"Running {analysis_type} searchlight analysis for {subject_id}")
        
        if analysis_type == "usplus":
            sl_result = sl.run_searchlight(mask_usplus, pool_size=pool_size)
        elif analysis_type == "usminus":
            sl_result = sl.run_searchlight(mask_usminus, pool_size=pool_size)
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")
        
        if rank == 0:
            print(f"Completed {analysis_type} searchlight analysis for {subject_id}")
            
            # Process and save results
            sl_result = sl_result.astype('double')
            non_nan_values = sl_result[~np.isnan(sl_result)]
            if len(non_nan_values) > 0:
                print(f"Result range: {np.min(non_nan_values)} to {np.max(non_nan_values)}")
            else:
                print("Warning: All results are NaN")
            
            # Replace NaNs with zeros
            sl_result[np.isnan(sl_result)] = 0
            
            # Create and save Nifti image
            sl_nii = nib.Nifti1Image(sl_result, affine=affine_mat)
            nib.save(sl_nii, output_path)
            print(f"Saved result to {output_path}")
            
            # Cleanup
            del sl_nii
        
        # Clean up memory
        del lss_data, sl_result, sl
        gc.collect()
        
    except Exception as e:
        if rank == 0:
            print(f"Error in searchlight analysis for {subject_id} ({analysis_type}): {str(e)}")
            traceback.print_exc()
        comm.Abort(1)

def process_subject(subject_id, rsa_dir, anat_root, func_root, output_root, mask_csv_path):
    """
    Process a single subject through all analysis steps
    """
    try:
        if rank == 0:
            print(f"\n==== Processing subject {subject_id} ====")
        
        # Define file paths
        lss_dir = os.path.join(rsa_dir, 'fear_acq', 'LSS_maps/concatenated/')
        fourd_map_path = os.path.join(lss_dir, f"{subject_id}_lssmap.nii.gz")
        
        if not os.path.exists(fourd_map_path):
            if rank == 0:
                print(f"ERROR: 4D map not found for {subject_id}: {fourd_map_path}")
            return
        
        # Get trial counts from CSV
        mask_params = get_trial_counts_from_csv(subject_id, mask_csv_path)
        
        
        # Prepare mask
        mask_data, mask_resampled_bin, affine_mat = prepare_mask(
            subject_id, fourd_map_path, anat_root, func_root, output_root)
        
        # Run US+ searchlight
        usplus_output = os.path.join(output_root, f"{subject_id}_USplus_fearacq_betweenitem_radius2_lss.nii.gz")
        run_searchlight_analysis(
            subject_id, fourd_map_path, mask_data, mask_resampled_bin, affine_mat, 
            "usplus", usplus_output, mask_params, sl_radius=2, pool_size=4)
        
        # Wait for all ranks to finish US+ analysis
        comm.Barrier()
        
        # Run US- searchlight
        usminus_output = os.path.join(output_root, f"{subject_id}_USminus_fearacq_betweenitem_radius2_lss.nii.gz")
        run_searchlight_analysis(
            subject_id, fourd_map_path, mask_data, mask_resampled_bin, affine_mat, 
            "usminus", usminus_output, mask_params, sl_radius=2, pool_size=4)
        
        # Wait for all ranks to finish US- analysis
        comm.Barrier()
        
        if rank == 0:
            print(f"Completed all analyses for {subject_id}")
    
    except Exception as e:
        if rank == 0:
            print(f"Error processing subject {subject_id}: {str(e)}")
            traceback.print_exc()
        comm.Abort(1)

if __name__ == "__main__":
    if len(sys.argv) != 7:
        if rank == 0:
            print("Usage: python script.py <subject_id> <rsa_dir> <anat_root> <func_root> <output_root> <mask_csv_path>")
        comm.Abort(1)
    
    subject_id = sys.argv[1]
    rsa_dir = sys.argv[2]
    anat_root = sys.argv[3]
    func_root = sys.argv[4]
    output_root = sys.argv[5]
    mask_csv_path = sys.argv[6]
    
    if rank == 0:
        print(f"Starting searchlight RSA for subject {subject_id}")
        print(f"RSA dir: {rsa_dir}")
        print(f"Anat root: {anat_root}")
        print(f"Func root: {func_root}")
        print(f"Output root: {output_root}")
        print(f"Mask CSV path: {mask_csv_path}")
        print(f"MPI size: {size} ranks\n")
    
    try:
        # Create output directory if it doesn't exist
        if rank == 0:
            os.makedirs(output_root, exist_ok=True)
        
        # Process the subject
        process_subject(subject_id, rsa_dir, anat_root, func_root, output_root, mask_csv_path)
        
        if rank == 0:
            print(f"Successfully completed all analyses for subject {subject_id}")
        
    except Exception as e:
        if rank == 0:
            print(f"Error in main function: {str(e)}")
            traceback.print_exc()
        comm.Abort(1)
