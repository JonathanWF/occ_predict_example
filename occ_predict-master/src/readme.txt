This is the read me for how to use these scripts to process WSIs

1. Using tile_batches_from_histomics.py, extract all of the image tiles from HistomicsTK while resizing them to the desired resolution.  In the case of the project all tiles were resized to 2 mpp.

2. Rounding errors can occur such that tiles are 1 pixel over or under in a dimension.  Use only_1990_by_1990.m to rectify this.

3. Upload all of the tiles to the CCR in the inference folder.  Upload all ROIs and Ground Truths for training, validation, and testing to the respective folders under data/processsed. Alternatively, with a larger set of data, upload to the data/raw folder and run build_dataset.py

4. Open PuTTY, navigate to the working directory and run sbatch slurm-unet-train.

5. Run sbatch src/slurm-unet-infer-cpu to run inference on all of the tiles in the infer folder.  Saved in the experiments/base_model/output_imgs folder will be the original input tile, the raw label map before smoothing, and the probability images for all of the classes.

6. Download all of the tiles from the CCR.  Run smooth_prob_create_map.m to create a smoothed probability map.  

7. Move the smoothed tiles to separate folders in the same directory for each WSI. Run Stitch_LabelsandTiles_for_WSIs.m on this directory and you'll have the stitched original downsampled WSI and the corresponding label map.

8. Finally, run Extract_Tumor_Nontumor_Area.m to generate the tissue and the label map for the tumor/non-tumor ROI.
