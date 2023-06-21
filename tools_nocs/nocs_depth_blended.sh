python nocs_depth_blended.py \
    --img_path ../data/NOCS/train/00000/0000_color_pad_4_crop_512.png \
    --depth_path ../data/NOCS/train/00000/0000_composed_pad_4_crop_512.png \
    --mask_path ../data/NOCS/train/00000/0000_mask_pad_4_crop_512.png \
    --output_dir ./controlnet_output/nocs_depth_blended_dila1 \
    --dilation_iterations 1 \
