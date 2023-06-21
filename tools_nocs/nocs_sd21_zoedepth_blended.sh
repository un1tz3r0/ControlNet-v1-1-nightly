python nocs_depth_blended.py \
    --prompt "a bottle on table with water in it and label on it" \
    --img_path ../data/NOCS/train/00000/0000_color_pad_4_crop_512.png \
    --depth_path ../data/NOCS/train/00000/0000_composed_pad_4_crop_512.png \
    --mask_path ../data/NOCS/train/00000/0000_mask_pad_4_crop_512.png \
    --output_dir ./controlnet_output/nocs_sd21_zoedepth_blended_dila1_train_00000 \
    --dilation_radius 1 \
    --sd "21" \
    --zoe \

python nocs_depth_blended.py \
    --prompt "a bottle on table" \
    --img_path "../data/NOCS/train/00017/0005_color_pad_1_crop_512.png" \
    --depth_path "../data/NOCS/train/00017/0005_composed_pad_1_crop_512.png" \
    --mask_path "../data/NOCS/train/00017/0005_mask_pad_1_crop_512.png" \
    --output_dir "./controlnet_output/nocs_sd21_zoedepth_blended_dila1_train_00017" \
    --dilation_radius 1 \
    --sd "21" \
    --zoe \

python nocs_depth_blended.py \
    --prompt "a glass bowl" \
    --img_path "../data/NOCS/train/00017/0005_color_pad_3_crop_512.png" \
    --depth_path "../data/NOCS/train/00017/0005_composed_pad_3_crop_512.png" \
    --mask_path "../data/NOCS/train/00017/0005_mask_pad_3_crop_512.png" \
    --output_dir "./controlnet_output/nocs_sd21_zoedepth_blended_dila1_train_00017" \
    --dilation_radius 1 \
    --sd "21" \
    --zoe \

python nocs_depth_blended.py \
    --prompt "a glass bowl" \
    --img_path "../data/NOCS/train/00122/0000_color_pad_3_crop_512.png" \
    --depth_path "../data/NOCS/train/00122/0000_composed_pad_3_crop_512.png" \
    --mask_path "../data/NOCS/train/00122/0000_mask_pad_3_crop_512.png" \
    --output_dir "./controlnet_output/nocs_sd21_zoedepth_blended_dila1_train_00122" \
    --dilation_radius 1 \
    --sd "21" \
    --zoe \

python nocs_depth_blended.py \
    --prompt "a glass bowl on table" \
    --img_path "../data/NOCS/real_train/scene_1/0000_color_pad_5_crop_512.png" \
    --depth_path "../data/NOCS/real_train/scene_1/0000_depth_pad_5_crop_512.png" \
    --mask_path "../data/NOCS/real_train/scene_1/0000_mask_pad_5_crop_512.png" \
    --output_dir "./controlnet_output/nocs_sd21_zoedepth_blended_dila1_real_train_scene_1" \
    --dilation_radius 1 \
    --sd "21" \
    --zoe \

python nocs_depth_blended.py \
    --prompt "a glass pull-tab can on table" \
    --img_path "../data/NOCS/real_train/scene_1/0000_color_pad_2_crop_512.png" \
    --mask_path "../data/NOCS/real_train/scene_1/0000_mask_pad_2_crop_512.png" \
    --output_dir "./controlnet_output/nocs_sd21_zoedepth_blended_dila1_real_train_scene_1" \
    --dilation_radius 1 \
    --sd "21" \
    --zoe \

python nocs_depth_blended.py \
    --prompt "a transparent pull-tab can on table" \
    --img_path "../data/NOCS/real_train/scene_1/0000_color_pad_2_crop_512.png" \
    --mask_path "../data/NOCS/real_train/scene_1/0000_mask_pad_2_crop_512.png" \
    --output_dir "./controlnet_output/nocs_sd21_zoedepth_blended_dila1_real_train_scene_1" \
    --dilation_radius 1 \
    --sd "21" \
    --zoe \

python nocs_depth_blended.py \
    --prompt "a glass bowl on table" \
    --img_path "../data/NOCS/real_train/scene_1/0000_color_pad_3_crop_512.png" \
    --mask_path "../data/NOCS/real_train/scene_1/0000_mask_pad_3_crop_512.png" \
    --output_dir "./controlnet_output/nocs_sd21_zoedepth_blended_dila1_real_train_scene_1" \
    --dilation_radius 1 \
    --sd "21" \
    --zoe \

python nocs_depth_blended.py \
    --prompt "a glass cup on table" \
    --img_path "../data/NOCS/real_train/scene_1/0000_color_pad_4_crop_512.png" \
    --mask_path "../data/NOCS/real_train/scene_1/0000_mask_pad_4_crop_512.png" \
    --output_dir "./controlnet_output/nocs_sd21_zoedepth_blended_dila1_real_train_scene_1" \
    --dilation_radius 1 \
    --sd "21" \
    --zoe \

python nocs_depth_blended.py \
    --prompt "a glass can on table" \
    --img_path "../data/NOCS/real_train/scene_6/0000_color_pad_3_crop_512.png" \
    --mask_path "../data/NOCS/real_train/scene_6/0000_mask_pad_3_crop_512.png" \
    --output_dir "./controlnet_output/nocs_sd21_zoedepth_blended_dila1_real_train_scene_6" \
    --dilation_radius 1 \
    --sd "21" \
    --zoe \

python nocs_depth_blended.py \
    --prompt "a plastic water bottle on table, some part is transparent" \
    --img_path "../data/NOCS/real_train/scene_5/0000_color_pad_2_crop_512.png" \
    --mask_path "../data/NOCS/real_train/scene_5/0000_mask_pad_2_crop_512.png" \
    --output_dir "./controlnet_output/nocs_sd21_zoedepth_blended_dila1_real_train_scene_5" \
    --dilation_radius 1 \
    --sd "21" \
    --zoe \

python nocs_depth_blended.py \
    --prompt "a glass mug on table, top-down perspective" \
    --img_path "../data/NOCS/real_train/scene_4/0000_color_pad_4_crop_512.png" \
    --mask_path "../data/NOCS/real_train/scene_4/0000_mask_pad_4_crop_512.png" \
    --output_dir "./controlnet_output/nocs_sd21_zoedepth_blended_dila1_real_train_scene_4" \
    --dilation_radius 1 \
    --sd "21" \
    --zoe \

python nocs_depth_blended.py \
    --prompt "a glass mug on table" \
    --img_path "../data/NOCS/real_test/scene_1/0000_color_pad_4_crop_512.png" \
    --depth_path "../data/NOCS/real_test/scene_1/0000_depth_pad_4_crop_512.png" \
    --mask_path "../data/NOCS/real_test/scene_1/0000_mask_pad_4_crop_512.png" \
    --output_dir ./controlnet_output/nocs_sd21_zoedepth_blended_dila1_real_test_scene_1 \
    --dilation_radius 1 \
    --sd "21" \
    --zoe \

python nocs_depth_blended.py \
    --prompt "an open laptop on table" \
    --img_path "../data/NOCS/real_test/scene_2/0000_color_pad_1_crop_512.png" \
    --depth_path "../data/NOCS/real_test/scene_2/0000_depth_pad_1_crop_512.png" \
    --mask_path "../data/NOCS/real_test/scene_2/0000_mask_pad_1_crop_512.png" \
    --output_dir ./controlnet_output/nocs_sd21_zoedepth_blended_dila1_real_test_scene_2 \
    --dilation_radius 1 \
    --sd "21" \
    --zoe \

python nocs_depth_blended.py \
    --prompt "a water bottle on table with water in it and label on it" \
    --img_path "../data/NOCS/real_test/scene_2/0000_color_pad_2_crop_512.png" \
    --depth_path "../data/NOCS/real_test/scene_2/0000_depth_pad_2_crop_512.png" \
    --mask_path "../data/NOCS/real_test/scene_2/0000_mask_pad_2_crop_512.png" \
    --output_dir ./controlnet_output/nocs_sd21_zoedepth_blended_dila1_real_test_scene_2 \
    --dilation_radius 1 \
    --sd "21" \
    --zoe \

python nocs_depth_blended.py \
    --prompt "a water bottle on table with water in it and label on it" \
    --img_path "../data/NOCS/real_test/scene_5/0001_color_pad_2_crop_512.png" \
    --depth_path "../data/NOCS/real_test/scene_5/0001_depth_pad_2_crop_512.png" \
    --mask_path "../data/NOCS/real_test/scene_5/0001_mask_pad_2_crop_512.png" \
    --output_dir ./controlnet_output/nocs_sd21_zoedepth_blended_dila1_real_test_scene_5 \
    --dilation_radius 1 \
    --sd "21" \
    --zoe \
