#!/#!bin/bash

#python match_features_demo_MICCAI_UZ.py sift 1 2 --m /media/discoGordo/dataset_leon/UZ/HCULB_256_mask_eq.png
#python match_features_demo_MICCAI_UZ.py sift 1 40 --m /media/discoGordo/dataset_leon/UZ/HCULB_256_mask_eq.png
#python match_features_demo_MICCAI_UZ.py orb 1 2 --m /media/discoGordo/dataset_leon/UZ/HCULB_256_mask_eq.png
#python match_features_demo_MICCAI_UZ.py orb 1 40 --m /media/discoGordo/dataset_leon/UZ/HCULB_256_mask_eq.png
#python match_features_demo_MICCAI_UZ.py pretrained/superpoint_v1.pth 1 2 --m /media/discoGordo/dataset_leon/UZ/HCULB_256_mask_eq.png
#python match_features_demo_MICCAI_UZ.py pretrained/superpoint_v1.pth 1 40 --m /media/discoGordo/dataset_leon/UZ/HCULB_256_mask_eq.png
#python match_features_demo_MICCAI_UZ.py superpoint_ucluzlabel100/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 2 --m /media/discoGordo/dataset_leon/UZ/HCULB_256_mask_eq.png
#python match_features_demo_MICCAI_UZ.py superpoint_ucluzlabel100/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 40 --m /media/discoGordo/dataset_leon/UZ/HCULB_256_mask_eq.png
#python match_features_demo_MICCAI_UZ.py superpoint_ucluzlabel100_spec100pixels/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 2 --m /media/discoGordo/dataset_leon/UZ/HCULB_256_mask_eq.png
#python match_features_demo_MICCAI_UZ.py superpoint_ucluzlabel100_spec100pixels/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 40 --m /media/discoGordo/dataset_leon/UZ/HCULB_256_mask_eq.png

#python match_features_demo_MICCAI_UZ_grid.py sift 1 2 --m /media/discoGordo/dataset_leon/UZ/HCULB_512_mask_eq.png
#python match_features_demo_MICCAI_UZ_grid.py sift 1 40 --m /media/discoGordo/dataset_leon/UZ/HCULB_512_mask_eq.png
#python match_features_demo_MICCAI_UZ_grid.py orb 1 2 --m /media/discoGordo/dataset_leon/UZ/HCULB_512_mask_eq.png
#python match_features_demo_MICCAI_UZ_grid.py orb 1 40 --m /media/discoGordo/dataset_leon/UZ/HCULB_512_mask_eq.png
#python match_features_demo_MICCAI_UZ_grid.py pretrained/superpoint_v1.pth 1 2 --m /media/discoGordo/dataset_leon/UZ/HCULB_512_mask_eq.png
#python match_features_demo_MICCAI_UZ_grid.py pretrained/superpoint_v1.pth 1 40 --m /media/discoGordo/dataset_leon/UZ/HCULB_512_mask_eq.png
#python match_features_demo_MICCAI_UZ_grid.py superpoint_ucluzlabel100/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 2 --m /media/discoGordo/dataset_leon/UZ/HCULB_512_mask_eq.png
#python match_features_demo_MICCAI_UZ_grid.py superpoint_ucluzlabel100/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 40 --m /media/discoGordo/dataset_leon/UZ/HCULB_512_mask_eq.png
#python match_features_demo_MICCAI_UZ_grid.py superpoint_ucluzlabel100_spec100pixels/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 2 --m /media/discoGordo/dataset_leon/UZ/HCULB_512_mask_eq.png
#python match_features_demo_MICCAI_UZ_grid.py superpoint_ucluzlabel100_spec100pixels/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 40 --m /media/discoGordo/dataset_leon/UZ/HCULB_512_mask_eq.png

##spec
#python match_features_demo_MICCAI_UZ_512.py sift 1 2 --m /media/discoGordo/dataset_leon/UZ/HCULB_512_mask_eq.png --spec
#python match_features_demo_MICCAI_UZ_512.py sift 1 40 --m /media/discoGordo/dataset_leon/UZ/HCULB_512_mask_eq.png --spec
#python match_features_demo_MICCAI_UZ_512.py pretrained/superpoint_v1.pth 1 2 --m /media/discoGordo/dataset_leon/UZ/HCULB_512_mask_eq.png --spec
#python match_features_demo_MICCAI_UZ_512.py pretrained/superpoint_v1.pth 1 40 --m /media/discoGordo/dataset_leon/UZ/HCULB_512_mask_eq.png --spec
#python match_features_demo_MICCAI_UZ_512.py superpoint_ucluzlabel100_spec100pixels/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 2 --m /media/discoGordo/dataset_leon/UZ/HCULB_512_mask_eq.png --spec
#python match_features_demo_MICCAI_UZ_512.py superpoint_ucluzlabel100_spec100pixels/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 40 --m /media/discoGordo/dataset_leon/UZ/HCULB_512_mask_eq.png --spec

##green
#python match_features_demo_MICCAI_UZ_512.py sift 1 2 --m /media/discoGordo/dataset_leon/UZ/HCULB_512_mask_eq.png --green
#python match_features_demo_MICCAI_UZ_512.py sift 1 40 --m /media/discoGordo/dataset_leon/UZ/HCULB_512_mask_eq.png --green
#python match_features_demo_MICCAI_UZ_512.py pretrained/superpoint_v1.pth 1 2 --m /media/discoGordo/dataset_leon/UZ/HCULB_512_mask_eq.png --green
#python match_features_demo_MICCAI_UZ_512.py pretrained/superpoint_v1.pth 1 40 --m /media/discoGordo/dataset_leon/UZ/HCULB_512_mask_eq.png --green
#python match_features_demo_MICCAI_UZ_512.py superpoint_ucluzlabel100/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 2 --m /media/discoGordo/dataset_leon/UZ/HCULB_512_mask_eq.png --green
#python match_features_demo_MICCAI_UZ_512.py superpoint_ucluzlabel100/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 40 --m /media/discoGordo/dataset_leon/UZ/HCULB_512_mask_eq.png --green
#python match_features_demo_MICCAI_UZ_512.py superpoint_ucluzlabel100_green/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 2 --m /media/discoGordo/dataset_leon/UZ/HCULB_512_mask_eq.png --green
#python match_features_demo_MICCAI_UZ_512.py superpoint_ucluzlabel100_green/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 40 --m /media/discoGordo/dataset_leon/UZ/HCULB_512_mask_eq.png --green
#python match_features_demo_MICCAI_UZ_grid.py superpoint_ucluzlabel100_green/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 2 --m /media/discoGordo/dataset_leon/UZ/HCULB_512_mask_eq.png --green
#python match_features_demo_MICCAI_UZ_grid.py superpoint_ucluzlabel100_green/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 40 --m /media/discoGordo/dataset_leon/UZ/HCULB_512_mask_eq.png --green

##1080
#python match_features_demo_MICCAI_UZ_1080_inliers_MIDL2022_H.py sift 1 2 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png
#python match_features_demo_MICCAI_UZ_1080_inliers_MIDL2022.py sift 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png
#python match_features_demo_MICCAI_UZ_1080_inliers_MIDL2022_H.py orb 1 2 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png
#python match_features_demo_MICCAI_UZ_1080_inliers_MIDL2022.py orb 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png
#python match_features_demo_MICCAI_UZ_1080_inliers_MIDL2022_H.py pretrained/superpoint_v1.pth 1 2 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png
#python match_features_demo_MICCAI_UZ_1080_inliers_MIDL2022.py pretrained/superpoint_v1.pth 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png
#python match_features_demo_MICCAI_UZ_1080_inliers_MIDL2022_H.py superpoint_ucluzlabel100_Bn/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 2 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png
#python match_features_demo_MICCAI_UZ_1080_inliers_MIDL2022.py superpoint_ucluzlabel100_Bn/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png
#python match_features_demo_MICCAI_UZ_1080_inliers_MIDL2022_H.py superpoint_ucluzlabel100_spec100pixels_Bn/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 2 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png
#python match_features_demo_MICCAI_UZ_1080_inliers_MIDL2022.py superpoint_ucluzlabel100_spec100pixels_Bn/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png

##spec
#python match_features_demo_MICCAI_UZ_1080_inliers_MIDL2022_H.py sift 1 2 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png --spec
#python match_features_demo_MICCAI_UZ_1080_inliers_MIDL2022.py sift 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png --spec
#python match_features_demo_MICCAI_UZ_1080_inliers_MIDL2022_H.py pretrained/superpoint_v1.pth 1 2 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png --spec
#python match_features_demo_MICCAI_UZ_1080_inliers_MIDL2022.py pretrained/superpoint_v1.pth 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png --spec
#python match_features_demo_MICCAI_UZ_1080_inliers_MIDL2022_H.py superpoint_ucluzlabel100/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 2 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png --spec
#python match_features_demo_MICCAI_UZ_1080_inliers_MIDL2022.py superpoint_ucluzlabel100/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png --spec
#python match_features_demo_MICCAI_UZ_1080_inliers_MIDL2022_H.py superpoint_ucluzlabel100_spec100pixels/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 2 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png --spec
#python match_features_demo_MICCAI_UZ_1080_inliers_MIDL2022.py superpoint_ucluzlabel100_spec100pixels/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png --spec

#python train4.py train_joint configs/superpoint_uz_train_heatmap.yaml superpoint_ucluzlabel100_B16_fast --eval --debug
#python train4.py train_joint configs/superpoint_uz_train_heatmap_spec.yaml superpoint_ucluzlabel100_spec100pixels --eval --debug
#python train4.py train_joint configs/superpoint_uz_train_heatmap_spec_ga.yaml superpoint_ucluzlabel100_specga_9-4_d7 --eval --debug

##gaussians
#python match_features_demo_MICCAI_UZ_1080_inliers_MIDL2022_H.py pretrained/superpoint_v1.pth 1 2 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png
#python match_features_demo_MICCAI_UZ_1080_inliers_MIDL2022.py pretrained/superpoint_v1.pth 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png
#python match_features_demo_MICCAI_UZ_1080_inliers_MIDL2022_H.py superpoint_ucluzlabel100/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 2 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png
#python match_features_demo_MICCAI_UZ_1080_inliers_MIDL2022.py superpoint_ucluzlabel100/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png
#python match_features_demo_MICCAI_UZ_1080_inliers_MIDL2022_H.py superpoint_ucluzlabel100_spec100pixels/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 2 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png
#python match_features_demo_MICCAI_UZ_1080_inliers_MIDL2022.py superpoint_ucluzlabel100_spec100pixels/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png

#python match_features_demo_MICCAI_UZ_1080_inliers_MIDL2022_H.py superpoint_ucluzlabel100_specga_5-2/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png
#python match_features_demo_MICCAI_UZ_1080_inliers_MIDL2022.py superpoint_ucluzlabel100_specga_5-2/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png
#python match_features_demo_MICCAI_UZ_1080_inliers_MIDL2022_H.py superpoint_ucluzlabel100_specga_9-4/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png
#python match_features_demo_MICCAI_UZ_1080_inliers_MIDL2022.py superpoint_ucluzlabel100_specga_9-4/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png
#python match_features_demo_MICCAI_UZ_1080_inliers_MIDL2022_H.py superpoint_ucluzlabel100_specga_13-6/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png
#python match_features_demo_MICCAI_UZ_1080_inliers_MIDL2022.py superpoint_ucluzlabel100_specga_13-6/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png
#python match_features_demo_MICCAI_UZ_1080_inliers_MIDL2022_H.py superpoint_ucluzlabel100_specga_5-2_d3/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png
#python match_features_demo_MICCAI_UZ_1080_inliers_MIDL2022.py superpoint_ucluzlabel100_specga_5-2_d3/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png
#python match_features_demo_MICCAI_UZ_1080_inliers_MIDL2022_H.py superpoint_ucluzlabel100_specga_9-4_d3/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png
#python match_features_demo_MICCAI_UZ_1080_inliers_MIDL2022.py superpoint_ucluzlabel100_specga_9-4_d3/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png
#python match_features_demo_MICCAI_UZ_1080_inliers_MIDL2022_H.py superpoint_ucluzlabel100_specga_13-6_d3/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png
#python match_features_demo_MICCAI_UZ_1080_inliers_MIDL2022.py superpoint_ucluzlabel100_specga_13-6_d3/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png

#python match_features_demo_MIDL2022_E.py sift 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png
#python match_features_demo_MIDL2022_E.py sift 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png --o 2
#python match_features_demo_MIDL2022_E.py sift 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png --o 5
#python match_features_demo_MIDL2022_E.py sift 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png --c 4
#python match_features_demo_MIDL2022_E.py sift 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png --c 10
#python match_features_demo_MIDL2022_E.py sift 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png --e 20
#python match_features_demo_MIDL2022_E.py sift 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png --e 40
#python match_features_demo_MIDL2022_E.py sift 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png --s 12
#python match_features_demo_MIDL2022_E.py sift 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png --s 8
#python match_features_demo_MIDL2022_E.py sift 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png --c 10 --ratio 50
#python match_features_demo_MIDL2022_E.py sift 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png --c 10 --ratio 75
python match_features_demo_MIDL2022_E.py superpoint_ucluzlabel100_specga_9-4_d3/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png #--ratio 80
python match_features_demo_MIDL2022_E.py sift 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png --c 10 --ratio 80
#python match_features_demo_MIDL2022_E.py sift 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png --c 10 --ratio 85
#python match_features_demo_MIDL2022_E.py sift 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png --c 10 --ratio 95
python match_features_demo_MIDL2022_E.py orb 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png
#python match_features_demo_MIDL2022_E.py orb 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png --sf 11
#python match_features_demo_MIDL2022_E.py orb 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png --sf 15
#python match_features_demo_MIDL2022_E.py orb 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png --nl 6
#python match_features_demo_MIDL2022_E.py orb 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png --nl 10
#python match_features_demo_MIDL2022_E.py orb 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png --ratio 50
#python match_features_demo_MIDL2022_E.py orb 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png --ratio 75
#python match_features_demo_MIDL2022_E.py orb 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png --ratio 85
#python match_features_demo_MIDL2022_E.py orb 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png --ratio 95
python match_features_demo_MIDL2022_E.py pretrained/superpoint_v1.pth 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png
#python match_features_demo_MIDL2022_E.py superpoint_tracking_v0/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png --nms 4 #--thsim 80
#python match_features_demo_MIDL2022_E.py superpoint_tracking_v1/checkpoints/superPointNet_400000_checkpoint.pth.tar 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png --nms 4 #--thsim 80
#python match_features_demo_MIDL2022_E.py superpoint_homography_v0/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png --nms 4 #--thsim 80
#python match_features_demo_MIDL2022_E.py superpoint_homography_v1/checkpoints/superPointNet_400000_checkpoint.pth.tar 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png --nms 4 #--thsim 80
#python match_features_demo_MIDL2022_E.py superpoint_em_v0/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png --nms 4 #--thsim 80
#python match_features_demo_MIDL2022_E.py superpoint_em_v1/checkpoints/superPointNet_500000_checkpoint.pth.tar 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png --nms 4 #--thsim 80
#python match_features_demo_MIDL2022_E.py superpoint_ucluzlabel100/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png
#python match_features_demo_MIDL2022_E.py superpoint_ucluzlabel100_specga_9-4_d5/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png
#python match_features_demo_MIDL2022_E.py superpoint_ucluzlabel100_specga_9-4_d7/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png
#python match_features_demo_MIDL2022_E.py superpoint_ucluzlabel100_specga_9-4_d3/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png --ratio 50
#python match_features_demo_MIDL2022_E.py superpoint_ucluzlabel100_specga_9-4_d3/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png --ratio 75
#python match_features_demo_MIDL2022_E.py superpoint_ucluzlabel100_specga_9-4_d3/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png --ratio 85
#python match_features_demo_MIDL2022_E.py superpoint_ucluzlabel100_specga_9-4_d3/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png --ratio 95
#python match_features_demo_MIDL2022_E.py superpoint_ucluzlabel100_specga_13-6/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png
#python match_features_demo_MIDL2022_E.py superpoint_ucluzlabel100_Bn/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png
#python match_features_demo_MIDL2022_E.py sift 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png --c 10 --ratio 80 --spec
#python match_features_demo_MIDL2022_E.py orb 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png --spec
#python match_features_demo_MIDL2022_E.py pretrained/superpoint_v1.pth 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png --spec
#python match_features_demo_MIDL2022_E.py superpoint_ucluzlabel100_specga_9-4_d3/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png --spec
#python match_features_demo_MIDL2022_E.py superpoint_ucluzlabel100_Bn/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png --spec

##hyper kvasir
#python match_features_demo_hyper-kvasir_576_inliers_MICCAI2022.py sift 1 2 --m /media/discoGordo/dataset_leon/hyper-kvasir/labeled-videos/lower-gi-tract/quality-of-mucosal-view/hyper-kvasir_mask_512_eq.png
#python match_features_demo_hyper-kvasir_576_inliers_MICCAI2022.py sift 1 25 --m /media/discoGordo/dataset_leon/hyper-kvasir/labeled-videos/lower-gi-tract/quality-of-mucosal-view/hyper-kvasir_mask_512_eq.png
#python match_features_demo_hyper-kvasir_576_inliers_MICCAI2022.py orb 1 2 --m /media/discoGordo/dataset_leon/hyper-kvasir/labeled-videos/lower-gi-tract/quality-of-mucosal-view/hyper-kvasir_mask_512_eq.png
#python match_features_demo_hyper-kvasir_576_inliers_MICCAI2022.py orb 1 25 --m /media/discoGordo/dataset_leon/hyper-kvasir/labeled-videos/lower-gi-tract/quality-of-mucosal-view/hyper-kvasir_mask_512_eq.png
#python match_features_demo_hyper-kvasir_576_inliers_MICCAI2022.py pretrained/superpoint_v1.pth 1 2 --m /media/discoGordo/dataset_leon/hyper-kvasir/labeled-videos/lower-gi-tract/quality-of-mucosal-view/hyper-kvasir_mask_512_eq.png
#python match_features_demo_hyper-kvasir_576_inliers_MICCAI2022.py pretrained/superpoint_v1.pth 1 25 --m /media/discoGordo/dataset_leon/hyper-kvasir/labeled-videos/lower-gi-tract/quality-of-mucosal-view/hyper-kvasir_mask_512_eq.png
#python match_features_demo_hyper-kvasir_576_inliers_MICCAI2022.py superpoint_ucluzlabel100_Bn/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 2 --m /media/discoGordo/dataset_leon/hyper-kvasir/labeled-videos/lower-gi-tract/quality-of-mucosal-view/hyper-kvasir_mask_512_eq.png
#python match_features_demo_hyper-kvasir_576_inliers_MICCAI2022.py superpoint_ucluzlabel100_Bn/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 25 --m //media/discoGordo/dataset_leon/hyper-kvasir/labeled-videos/lower-gi-tract/quality-of-mucosal-view/hyper-kvasir_mask_512_eq.png
#python match_features_demo_hyper-kvasir_576_inliers_MICCAI2022.py superpoint_ucluzlabel100_spec100pixels_Bn/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 2 --m /media/discoGordo/dataset_leon/hyper-kvasir/labeled-videos/lower-gi-tract/quality-of-mucosal-view/hyper-kvasir_mask_512_eq.png
#python match_features_demo_hyper-kvasir_576_inliers_MICCAI2022.py superpoint_ucluzlabel100_spec100pixels_Bn/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 25 --m /media/discoGordo/dataset_leon/hyper-kvasir/labeled-videos/lower-gi-tract/quality-of-mucosal-view/hyper-kvasir_mask_512_eq.png

#python match_features_demo_hyper-kvasir_512_video_MICCAI2022.py sift 1 40 --m /media/discoGordo/dataset_leon/hyper-kvasir/labeled-videos/lower-gi-tract/quality-of-mucosal-view/hyper-kvasir_mask_512_eq.png --c 10 --ratio 80
#python match_features_demo_hyper-kvasir_512_video_MICCAI2022.py orb 1 40 --m /media/discoGordo/dataset_leon/hyper-kvasir/labeled-videos/lower-gi-tract/quality-of-mucosal-view/hyper-kvasir_mask_512_eq.png
#python match_features_demo_hyper-kvasir_512_video_MICCAI2022.py pretrained/superpoint_v1.pth 1 40 --m /media/discoGordo/dataset_leon/hyper-kvasir/labeled-videos/lower-gi-tract/quality-of-mucosal-view/hyper-kvasir_mask_512_eq.png
#python match_features_demo_hyper-kvasir_512_video_MICCAI2022.py superpoint_ucluzlabel100_Bn/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 40 --m /media/discoGordo/dataset_leon/hyper-kvasir/labeled-videos/lower-gi-tract/quality-of-mucosal-view/hyper-kvasir_mask_512_eq.png
#python match_features_demo_hyper-kvasir_512_video_MICCAI2022.py superpoint_ucluzlabel100_spec100pixels_Bn/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 40 --m /media/discoGordo/dataset_leon/hyper-kvasir/labeled-videos/lower-gi-tract/quality-of-mucosal-view/hyper-kvasir_mask_512_eq.png

#python match_features_demo_hyper-kvasir_512_inliers_MICCAI2022.py sift 1 25 --m /media/discoGordo/dataset_leon/hyper-kvasir/labeled-videos/lower-gi-tract/quality-of-mucosal-view/hyper-kvasir_mask_512_eq.png --c 10 --ratio 80
#python match_features_demo_hyper-kvasir_512_inliers_MICCAI2022.py orb 1 25 --m /media/discoGordo/dataset_leon/hyper-kvasir/labeled-videos/lower-gi-tract/quality-of-mucosal-view/hyper-kvasir_mask_512_eq.png
#python match_features_demo_hyper-kvasir_512_inliers_MICCAI2022.py pretrained/superpoint_v1.pth 1 25 --m /media/discoGordo/dataset_leon/hyper-kvasir/labeled-videos/lower-gi-tract/quality-of-mucosal-view/hyper-kvasir_mask_512_eq.png
#python match_features_demo_hyper-kvasir_512_inliers_MICCAI2022.py superpoint_ucluzlabel100_specga_9-4_d3/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 25 --m /media/discoGordo/dataset_leon/hyper-kvasir/labeled-videos/lower-gi-tract/quality-of-mucosal-view/hyper-kvasir_mask_512_eq.png
#python match_features_demo_hyper-kvasir_512_inliers_MICCAI2022.py superpoint_ucluzlabel100_Bn/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 25 --m /media/discoGordo/dataset_leon/hyper-kvasir/labeled-videos/lower-gi-tract/quality-of-mucosal-view/hyper-kvasir_mask_512_eq.png

#python match_features_demo_Nerthus_576_inliers_MIDL2022.py sift 1 2 --m /media/discoGordo/dataset_leon/nerthus-dataset-frames/Nerthus_mask_512_eq.png
#python match_features_demo_Nerthus_576_inliers_MIDL2022.py sift 1 25 --m /media/discoGordo/dataset_leon/nerthus-dataset-frames/Nerthus_mask_512_eq.png
#python match_features_demo_Nerthus_576_inliers_MIDL2022.py orb 1 2 --m /media/discoGordo/dataset_leon/nerthus-dataset-frames/Nerthus_mask_512_eq.png
#python match_features_demo_Nerthus_576_inliers_MIDL2022.py orb 1 25 --m /media/discoGordo/dataset_leon/nerthus-dataset-frames/Nerthus_mask_512_eq.png
#python match_features_demo_Nerthus_576_inliers_MIDL2022.py pretrained/superpoint_v1.pth 1 2 --m /media/discoGordo/dataset_leon/nerthus-dataset-frames/Nerthus_mask_512_eq.png
#python match_features_demo_Nerthus_576_inliers_MIDL2022.py pretrained/superpoint_v1.pth 1 25 --m /media/discoGordo/dataset_leon/nerthus-dataset-frames/Nerthus_mask_512_eq.png
#python match_features_demo_Nerthus_576_inliers_MIDL2022.py superpoint_ucluzlabel100_Bn/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 2 --m /media/discoGordo/dataset_leon/nerthus-dataset-frames/Nerthus_mask_512_eq.png
#python match_features_demo_Nerthus_576_inliers_MIDL2022.py superpoint_ucluzlabel100_Bn/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 25 --m /media/discoGordo/dataset_leon/nerthus-dataset-frames/Nerthus_mask_512_eq.png
#python match_features_demo_Nerthus_576_inliers_MIDL2022.py superpoint_ucluzlabel100_spec100pixels_Bn/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 2 --m /media/discoGordo/dataset_leon/nerthus-dataset-frames/Nerthus_mask_512_eq.png
#python match_features_demo_Nerthus_576_inliers_MIDL2022.py superpoint_ucluzlabel100_spec100pixels_Bn/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 25 --m /media/discoGordo/dataset_leon/nerthus-dataset-frames/Nerthus_mask_512_eq.png

#python match_features_demo_MIDL2022_H.py sift 1 2 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png
#python match_features_demo_MIDL2022_E.py sift 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png
#python match_features_demo_MIDL2022_H.py orb 1 2 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png
#python match_features_demo_MIDL2022_E.py orb 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png
#python match_features_demo_MIDL2022_H.py pretrained/superpoint_v1.pth 1 2 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png
#python match_features_demo_MIDL2022_E.py pretrained/superpoint_v1.pth 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png
#python match_features_demo_MIDL2022_H.py superpoint_ucluzlabel100_Bn/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 2 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png
#python match_features_demo_MIDL2022_E.py superpoint_ucluzlabel100_Bn/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png
#python match_features_demo_MIDL2022_H.py superpoint_ucluzlabel100_spec100pixels_Bn/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 2 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png
#python match_features_demo_MIDL2022_E.py superpoint_ucluzlabel100_spec100pixels_Bn/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 40 --m /media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png

#python match_features_demo_MICCAI_UZ_512.py pretrained/superpoint_v1.pth 1 2 --m /media/discoGordo/dataset_leon/UZ/HCULB_512_mask_eq.png
#python match_features_demo_MICCAI_UZ_512.py pretrained/superpoint_v1.pth 1 40 --m /media/discoGordo/dataset_leon/UZ/HCULB_512_mask_eq.png
#python match_features_demo_MICCAI_UZ_512.py superpoint_ucluzlabel100/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 2 --m /media/discoGordo/dataset_leon/UZ/HCULB_512_mask_eq.png
#python match_features_demo_MICCAI_UZ_512.py superpoint_ucluzlabel100/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 40 --m /media/discoGordo/dataset_leon/UZ/HCULB_512_mask_eq.png
#python match_features_demo_MICCAI_UZ_512.py superpoint_ucluzlabel100_spec100pixels/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 2 --m /media/discoGordo/dataset_leon/UZ/HCULB_512_mask_eq.png
#python match_features_demo_MICCAI_UZ_512.py superpoint_ucluzlabel100_spec100pixels/checkpoints/superPointNet_200000_checkpoint.pth.tar 1 40 --m /media/discoGordo/dataset_leon/UZ/HCULB_512_mask_eq.png

#python match_features_demo_calibration_basic.py sift /media/discoGordo/dataset_leon/BMVC2021/UZ_256/HCULB_00053_procedure_lossy_h264_0001.png /media/discoGordo/dataset_leon/BMVC2021/UZ_256/HCULB_00053_procedure_lossy_h264_0002.png --m /media/discoGordo/dataset_leon/UZ/HCULB_256_mask_eq.png --output sift_uz53_1_dist1.png
#python match_features_demo_calibration_basic.py sift /media/discoGordo/dataset_leon/BMVC2021/UZ_256/HCULB_00053_procedure_lossy_h264_0001.png /media/discoGordo/dataset_leon/BMVC2021/UZ_256/HCULB_00053_procedure_lossy_h264_0040.png --m /media/discoGordo/dataset_leon/UZ/HCULB_256_mask_eq.png --output sift_uz53_1_dist40.png
#python match_features_demo_calibration_basic.py sift /media/discoGordo/dataset_leon/BMVC2021/UZ_256/HCULB_00048_procedure_lossy_h264_0001.png /media/discoGordo/dataset_leon/BMVC2021/UZ_256/HCULB_00048_procedure_lossy_h264_0002.png --m /media/discoGordo/dataset_leon/UZ/HCULB_256_mask_eq.png --output sift_uz48_1_dist1.png
#python match_features_demo_calibration_basic.py sift /media/discoGordo/dataset_leon/BMVC2021/UZ_256/HCULB_00048_procedure_lossy_h264_0001.png /media/discoGordo/dataset_leon/BMVC2021/UZ_256/HCULB_00048_procedure_lossy_h264_0040.png --m /media/discoGordo/dataset_leon/UZ/HCULB_256_mask_eq.png --output sift_uz48_1_dist40.png
#python match_features_demo_calibration_basic.py sift /media/discoGordo/dataset_leon/BMVC2021/UZ_256/HCULB_00048_procedure_lossy_h264_0051.png /media/discoGordo/dataset_leon/BMVC2021/UZ_256/HCULB_00048_procedure_lossy_h264_0052.png --m /media/discoGordo/dataset_leon/UZ/HCULB_256_mask_eq.png --output sift_uz48_51_dist1.png
#python match_features_demo_calibration_basic.py sift /media/discoGordo/dataset_leon/BMVC2021/UZ_256/HCULB_00048_procedure_lossy_h264_0051.png /media/discoGordo/dataset_leon/BMVC2021/UZ_256/HCULB_00048_procedure_lossy_h264_0090.png --m /media/discoGordo/dataset_leon/UZ/HCULB_256_mask_eq.png --output sift_uz48_51_dist40.png
#python match_features_demo_calibration_basic.py orb /media/discoGordo/dataset_leon/BMVC2021/UZ_256/HCULB_00053_procedure_lossy_h264_0001.png /media/discoGordo/dataset_leon/BMVC2021/UZ_256/HCULB_00053_procedure_lossy_h264_0002.png --m /media/discoGordo/dataset_leon/UZ/HCULB_256_mask_eq.png --output orb_uz53_1_dist1.png
#python match_features_demo_calibration_basic.py orb /media/discoGordo/dataset_leon/BMVC2021/UZ_256/HCULB_00053_procedure_lossy_h264_0001.png /media/discoGordo/dataset_leon/BMVC2021/UZ_256/HCULB_00053_procedure_lossy_h264_0040.png --m /media/discoGordo/dataset_leon/UZ/HCULB_256_mask_eq.png --output orb_uz53_1_dist40.png
#python match_features_demo_calibration_basic.py orb /media/discoGordo/dataset_leon/BMVC2021/UZ_256/HCULB_00048_procedure_lossy_h264_0001.png /media/discoGordo/dataset_leon/BMVC2021/UZ_256/HCULB_00048_procedure_lossy_h264_0002.png --m /media/discoGordo/dataset_leon/UZ/HCULB_256_mask_eq.png --output orb_uz48_1_dist1.png
#python match_features_demo_calibration_basic.py orb /media/discoGordo/dataset_leon/BMVC2021/UZ_256/HCULB_00048_procedure_lossy_h264_0001.png /media/discoGordo/dataset_leon/BMVC2021/UZ_256/HCULB_00048_procedure_lossy_h264_0040.png --m /media/discoGordo/dataset_leon/UZ/HCULB_256_mask_eq.png --output orb_uz48_1_dist40.png
#python match_features_demo_calibration_basic.py orb /media/discoGordo/dataset_leon/BMVC2021/UZ_256/HCULB_00048_procedure_lossy_h264_0051.png /media/discoGordo/dataset_leon/BMVC2021/UZ_256/HCULB_00048_procedure_lossy_h264_0052.png --m /media/discoGordo/dataset_leon/UZ/HCULB_256_mask_eq.png --output orb_uz48_51_dist1.png
#python match_features_demo_calibration_basic.py orb /media/discoGordo/dataset_leon/BMVC2021/UZ_256/HCULB_00048_procedure_lossy_h264_0051.png /media/discoGordo/dataset_leon/BMVC2021/UZ_256/HCULB_00048_procedure_lossy_h264_0090.png --m /media/discoGordo/dataset_leon/UZ/HCULB_256_mask_eq.png --output orb_uz48_51_dist40.png
#python match_features_demo_calibration_basic.py pretrained/superpoint_v1.pth /media/discoGordo/dataset_leon/BMVC2021/UZ_256/HCULB_00053_procedure_lossy_h264_0001.png /media/discoGordo/dataset_leon/BMVC2021/UZ_256/HCULB_00053_procedure_lossy_h264_0002.png --m /media/discoGordo/dataset_leon/UZ/HCULB_256_mask_eq.png --output spv1_uz53_1_dist1.png
#python match_features_demo_calibration_basic.py pretrained/superpoint_v1.pth /media/discoGordo/dataset_leon/BMVC2021/UZ_256/HCULB_00053_procedure_lossy_h264_0001.png /media/discoGordo/dataset_leon/BMVC2021/UZ_256/HCULB_00053_procedure_lossy_h264_0040.png --m /media/discoGordo/dataset_leon/UZ/HCULB_256_mask_eq.png --output spv1_uz53_1_dist40.png
#python match_features_demo_calibration_basic.py pretrained/superpoint_v1.pth /media/discoGordo/dataset_leon/BMVC2021/UZ_256/HCULB_00048_procedure_lossy_h264_0001.png /media/discoGordo/dataset_leon/BMVC2021/UZ_256/HCULB_00048_procedure_lossy_h264_0002.png --m /media/discoGordo/dataset_leon/UZ/HCULB_256_mask_eq.png --output spv1_uz48_1_dist1.png
#python match_features_demo_calibration_basic.py pretrained/superpoint_v1.pth /media/discoGordo/dataset_leon/BMVC2021/UZ_256/HCULB_00048_procedure_lossy_h264_0001.png /media/discoGordo/dataset_leon/BMVC2021/UZ_256/HCULB_00048_procedure_lossy_h264_0040.png --m /media/discoGordo/dataset_leon/UZ/HCULB_256_mask_eq.png --output spv1_uz48_1_dist40.png
#python match_features_demo_calibration_basic.py pretrained/superpoint_v1.pth /media/discoGordo/dataset_leon/BMVC2021/UZ_256/HCULB_00048_procedure_lossy_h264_0051.png /media/discoGordo/dataset_leon/BMVC2021/UZ_256/HCULB_00048_procedure_lossy_h264_0052.png --m /media/discoGordo/dataset_leon/UZ/HCULB_256_mask_eq.png --output spv1_uz48_51_dist1.png
#python match_features_demo_calibration_basic.py pretrained/superpoint_v1.pth /media/discoGordo/dataset_leon/BMVC2021/UZ_256/HCULB_00048_procedure_lossy_h264_0051.png /media/discoGordo/dataset_leon/BMVC2021/UZ_256/HCULB_00048_procedure_lossy_h264_0090.png --m /media/discoGordo/dataset_leon/UZ/HCULB_256_mask_eq.png --output spv1_uz48_51_dist40.png
#python match_features_demo_calibration_basic.py superpoint_ucluzlabel100_spec100pixels/checkpoints/superPointNet_200000_checkpoint.pth.tar /media/discoGordo/dataset_leon/BMVC2021/UZ_256/HCULB_00053_procedure_lossy_h264_0001.png /media/discoGordo/dataset_leon/BMVC2021/UZ_256/HCULB_00053_procedure_lossy_h264_0002.png --m /media/discoGordo/dataset_leon/UZ/HCULB_256_mask_eq.png --output spspec_uz53_1_dist1.png
#python match_features_demo_calibration_basic.py superpoint_ucluzlabel100_spec100pixels/checkpoints/superPointNet_200000_checkpoint.pth.tar /media/discoGordo/dataset_leon/BMVC2021/UZ_256/HCULB_00053_procedure_lossy_h264_0001.png /media/discoGordo/dataset_leon/BMVC2021/UZ_256/HCULB_00053_procedure_lossy_h264_0040.png --m /media/discoGordo/dataset_leon/UZ/HCULB_256_mask_eq.png --output spspec_uz53_1_dist40.png
#python match_features_demo_calibration_basic.py superpoint_ucluzlabel100_spec100pixels/checkpoints/superPointNet_200000_checkpoint.pth.tar /media/discoGordo/dataset_leon/BMVC2021/UZ_256/HCULB_00048_procedure_lossy_h264_0001.png /media/discoGordo/dataset_leon/BMVC2021/UZ_256/HCULB_00048_procedure_lossy_h264_0002.png --m /media/discoGordo/dataset_leon/UZ/HCULB_256_mask_eq.png --output spspec_uz48_1_dist1.png
#python match_features_demo_calibration_basic.py superpoint_ucluzlabel100_spec100pixels/checkpoints/superPointNet_200000_checkpoint.pth.tar /media/discoGordo/dataset_leon/BMVC2021/UZ_256/HCULB_00048_procedure_lossy_h264_0001.png /media/discoGordo/dataset_leon/BMVC2021/UZ_256/HCULB_00048_procedure_lossy_h264_0040.png --m /media/discoGordo/dataset_leon/UZ/HCULB_256_mask_eq.png --output spspec_uz48_1_dist40.png
#python match_features_demo_calibration_basic.py superpoint_ucluzlabel100_spec100pixels/checkpoints/superPointNet_200000_checkpoint.pth.tar /media/discoGordo/dataset_leon/BMVC2021/UZ_256/HCULB_00048_procedure_lossy_h264_0051.png /media/discoGordo/dataset_leon/BMVC2021/UZ_256/HCULB_00048_procedure_lossy_h264_0052.png --m /media/discoGordo/dataset_leon/UZ/HCULB_256_mask_eq.png --output spspec_uz48_51_dist1.png
#python match_features_demo_calibration_basic.py superpoint_ucluzlabel100_spec100pixels/checkpoints/superPointNet_200000_checkpoint.pth.tar /media/discoGordo/dataset_leon/BMVC2021/UZ_256/HCULB_00048_procedure_lossy_h264_0051.png /media/discoGordo/dataset_leon/BMVC2021/UZ_256/HCULB_00048_procedure_lossy_h264_0090.png --m /media/discoGordo/dataset_leon/UZ/HCULB_256_mask_eq.png --output spspec_uz48_51_dist40.png
