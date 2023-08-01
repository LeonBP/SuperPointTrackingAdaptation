#python train4.py train_joint configs/superpoint_uz_train_heatmap.yaml superpoint_ucluzlabel100 --eval --debug
#python train4.py train_joint configs/superpoint_uz_train_heatmap_spec.yaml superpoint_ucluzlabel100_spec100pixels --eval --debug
python train4.py train_joint configs/superpoint_emcolmap_train_heatmap_tracking.yaml superpoint_tracking_v1 --eval --debug
python train4.py train_joint configs/superpoint_emcolmap_train_heatmap.yaml superpoint_homography_v1 --eval --debug
python train4.py train_joint configs/superpoint_em_train_heatmap.yaml superpoint_em_v1 --eval --debug
#sh match_eval.sh
