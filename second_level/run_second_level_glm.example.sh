
#!/usr/bin/env bash
python run_second_level_glm.py   --input_dir /path/to/combined_maps   --output_dir /path/to/group_level/average_3mm   --input_pattern '*average_effect*'   --mask /path/to/mni_gm_mask_06.nii.gz   --bg_img /path/to/mni_075mm.nii.gz   --smoothing_fwhm 3   --glm_p_threshold 0.001   --run_permutation   --n_permutations 1000   --perm_cluster_threshold 0.001   --perm_logp_threshold 1.301   --tfce   --two_sided   --n_jobs_perm 10   --cache_dir /path/to/nilearn_cache   --save_plots   --save_maps   --save_perm_dict   --memory_level 0
