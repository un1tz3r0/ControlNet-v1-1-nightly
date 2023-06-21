python nocs_depth_blended.py \
    --img_path ../data/NOCS/train/00000/0000_color_pad_4_crop_512.png \
    --depth_path ../data/NOCS/train/00000/0000_composed_pad_4_crop_512.png \
    --mask_path ../data/NOCS/train/00000/0000_mask_pad_4_crop_512.png \
    --output_dir ./controlnet_output_nocs_depth_given_blended_pixel_blend_02 \
    --percentage_of_pixel_blending 0.2 \
