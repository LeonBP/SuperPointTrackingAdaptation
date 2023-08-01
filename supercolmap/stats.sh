#!/bin/bash

#python main.py 00033 19 >> out00033_19_10k.txt
#python main.py 00034 3 >> out00034_3.txt

#METHOD="em_disk"
#OUTPUT="/media/discoGordo/dataset_leon/reconstructions_MICCAI2023/"
#python /home/leon/repositories/supercolmap/main_stats.py 00364 18 $METHOD --featuretype superpointmod --superpoint superpoint_em_v1/checkpoints/superPointNet_500000_checkpoint.pth.tar --matchtype disk >> $OUTPUT/$METHOD/00364/18/stats.txt
#python /home/leon/repositories/supercolmap/main_stats.py 00033 15 $METHOD --featuretype superpointmod --superpoint superpoint_em_v1/checkpoints/superPointNet_500000_checkpoint.pth.tar --matchtype disk >> $OUTPUT/$METHOD/00033/15/stats.txt
#python /home/leon/repositories/supercolmap/main_stats.py 00034 38 $METHOD --featuretype superpointmod --superpoint superpoint_em_v1/checkpoints/superPointNet_500000_checkpoint.pth.tar --matchtype disk >> $OUTPUT/$METHOD/00034/38/stats.txt
#python /home/leon/repositories/supercolmap/main_stats.py 00364 14 $METHOD --featuretype superpointmod --superpoint superpoint_em_v1/checkpoints/superPointNet_500000_checkpoint.pth.tar --matchtype disk >> $OUTPUT/$METHOD/00364/14/stats.txt
#python /home/leon/repositories/supercolmap/main_stats.py 02001 39 $METHOD --featuretype superpointmod --superpoint superpoint_em_v1/checkpoints/superPointNet_500000_checkpoint.pth.tar --matchtype disk >> $OUTPUT/$METHOD/02001/39/stats.txt
#python /home/leon/repositories/supercolmap/main_stats.py 02003 86 $METHOD --featuretype superpointmod --superpoint superpoint_em_v1/checkpoints/superPointNet_500000_checkpoint.pth.tar --matchtype disk >> $OUTPUT/$METHOD/02003/86/stats.txt
#python /home/leon/repositories/supercolmap/main_stats.py 02005 28 $METHOD --featuretype superpointmod --superpoint superpoint_em_v1/checkpoints/superPointNet_500000_checkpoint.pth.tar --matchtype disk >> $OUTPUT/$METHOD/02005/28/stats.txt
#METHOD="sphomography1_disk"
#OUTPUT="/media/discoGordo/dataset_leon/reconstructions_MICCAI2023/"
#python /home/leon/repositories/supercolmap/main_stats.py 00364 18 $METHOD --featuretype superpointmod --superpoint superpoint_homography_v1/checkpoints/superPointNet_400000_checkpoint.pth.tar --matchtype disk >> $OUTPUT/$METHOD/00364/18/stats.txt
#python /home/leon/repositories/supercolmap/main_stats.py 00033 15 $METHOD --featuretype superpointmod --superpoint superpoint_homography_v1/checkpoints/superPointNet_400000_checkpoint.pth.tar --matchtype disk >> $OUTPUT/$METHOD/00033/15/stats.txt
#python /home/leon/repositories/supercolmap/main_stats.py 00034 38 $METHOD --featuretype superpointmod --superpoint superpoint_homography_v1/checkpoints/superPointNet_400000_checkpoint.pth.tar --matchtype disk >> $OUTPUT/$METHOD/00034/38/stats.txt
#python /home/leon/repositories/supercolmap/main_stats.py 00364 14 $METHOD --featuretype superpointmod --superpoint superpoint_homography_v1/checkpoints/superPointNet_400000_checkpoint.pth.tar --matchtype disk >> $OUTPUT/$METHOD/00364/14/stats.txt
#python /home/leon/repositories/supercolmap/main_stats.py 02001 39 $METHOD --featuretype superpointmod --superpoint superpoint_homography_v1/checkpoints/superPointNet_400000_checkpoint.pth.tar --matchtype disk >> $OUTPUT/$METHOD/02001/39/stats.txt
#python /home/leon/repositories/supercolmap/main_stats.py 02003 86 $METHOD --featuretype superpointmod --superpoint superpoint_homography_v1/checkpoints/superPointNet_400000_checkpoint.pth.tar --matchtype disk >> $OUTPUT/$METHOD/02003/86/stats.txt
#python /home/leon/repositories/supercolmap/main_stats.py 02005 28 $METHOD --featuretype superpointmod --superpoint superpoint_homography_v1/checkpoints/superPointNet_400000_checkpoint.pth.tar --matchtype disk >> $OUTPUT/$METHOD/02005/28/stats.txt
#METHOD="sptracking1_disk"
#OUTPUT="/media/discoGordo/dataset_leon/reconstructions_MICCAI2023/"
#python /home/leon/repositories/supercolmap/main_stats.py 00364 18 $METHOD --featuretype superpointmod --superpoint superpoint_tracking_v1/checkpoints/superPointNet_400000_checkpoint.pth.tar --matchtype disk >> $OUTPUT/$METHOD/00364/18/stats.txt
#python /home/leon/repositories/supercolmap/main_stats.py 00033 15 $METHOD --featuretype superpointmod --superpoint superpoint_tracking_v1/checkpoints/superPointNet_400000_checkpoint.pth.tar --matchtype disk >> $OUTPUT/$METHOD/00033/15/stats.txt
#python /home/leon/repositories/supercolmap/main_stats.py 00034 38 $METHOD --featuretype superpointmod --superpoint superpoint_tracking_v1/checkpoints/superPointNet_400000_checkpoint.pth.tar --matchtype disk >> $OUTPUT/$METHOD/00034/38/stats.txt
#python /home/leon/repositories/supercolmap/main_stats.py 00364 14 $METHOD --featuretype superpointmod --superpoint superpoint_tracking_v1/checkpoints/superPointNet_400000_checkpoint.pth.tar --matchtype disk >> $OUTPUT/$METHOD/00364/14/stats.txt
#python /home/leon/repositories/supercolmap/main_stats.py 02001 39 $METHOD --featuretype superpointmod --superpoint superpoint_tracking_v1/checkpoints/superPointNet_400000_checkpoint.pth.tar --matchtype disk >> $OUTPUT/$METHOD/02001/39/stats.txt
#python /home/leon/repositories/supercolmap/main_stats.py 02003 86 $METHOD --featuretype superpointmod --superpoint superpoint_tracking_v1/checkpoints/superPointNet_400000_checkpoint.pth.tar --matchtype disk >> $OUTPUT/$METHOD/02003/86/stats.txt
#python /home/leon/repositories/supercolmap/main_stats.py 02005 28 $METHOD --featuretype superpointmod --superpoint superpoint_tracking_v1/checkpoints/superPointNet_400000_checkpoint.pth.tar --matchtype disk >> $OUTPUT/$METHOD/02005/28/stats.txt
#METHOD="sptracking2_disk"
#OUTPUT="/media/discoGordo/dataset_leon/reconstructions_MICCAI2023/"
#python /home/leon/repositories/supercolmap/main_stats.py 00364 18 $METHOD --featuretype superpointmod --superpoint superpoint_tracking_v2/checkpoints/superPointNet_400000_checkpoint.pth.tar --matchtype disk >> $OUTPUT/$METHOD/00364/18/stats.txt
#python /home/leon/repositories/supercolmap/main_stats.py 00033 15 $METHOD --featuretype superpointmod --superpoint superpoint_tracking_v2/checkpoints/superPointNet_400000_checkpoint.pth.tar --matchtype disk >> $OUTPUT/$METHOD/00033/15/stats.txt
#python /home/leon/repositories/supercolmap/main_stats.py 00034 38 $METHOD --featuretype superpointmod --superpoint superpoint_tracking_v2/checkpoints/superPointNet_400000_checkpoint.pth.tar --matchtype disk >> $OUTPUT/$METHOD/00034/38/stats.txt
#python /home/leon/repositories/supercolmap/main_stats.py 00364 14 $METHOD --featuretype superpointmod --superpoint superpoint_tracking_v2/checkpoints/superPointNet_400000_checkpoint.pth.tar --matchtype disk >> $OUTPUT/$METHOD/00364/14/stats.txt
#python /home/leon/repositories/supercolmap/main_stats.py 02001 39 $METHOD --featuretype superpointmod --superpoint superpoint_tracking_v2/checkpoints/superPointNet_400000_checkpoint.pth.tar --matchtype disk >> $OUTPUT/$METHOD/02001/39/stats.txt
#python /home/leon/repositories/supercolmap/main_stats.py 02003 86 $METHOD --featuretype superpointmod --superpoint superpoint_tracking_v2/checkpoints/superPointNet_400000_checkpoint.pth.tar --matchtype disk >> $OUTPUT/$METHOD/02003/86/stats.txt
#python /home/leon/repositories/supercolmap/main_stats.py 02005 28 $METHOD --featuretype superpointmod --superpoint superpoint_tracking_v2/checkpoints/superPointNet_400000_checkpoint.pth.tar --matchtype disk >> $OUTPUT/$METHOD/02005/28/stats.txt
#METHOD="sptracking3_disk"
#OUTPUT="/media/discoGordo/dataset_leon/reconstructions_MICCAI2023/"
#python /home/leon/repositories/supercolmap/main_stats.py 00364 18 $METHOD --featuretype superpointmod --superpoint superpoint_tracking_v3/checkpoints/superPointNet_400000_checkpoint.pth.tar --matchtype disk >> $OUTPUT/$METHOD/00364/18/stats.txt
#python /home/leon/repositories/supercolmap/main_stats.py 00033 15 $METHOD --featuretype superpointmod --superpoint superpoint_tracking_v3/checkpoints/superPointNet_400000_checkpoint.pth.tar --matchtype disk >> $OUTPUT/$METHOD/00033/15/stats.txt
#python /home/leon/repositories/supercolmap/main_stats.py 00034 38 $METHOD --featuretype superpointmod --superpoint superpoint_tracking_v3/checkpoints/superPointNet_400000_checkpoint.pth.tar --matchtype disk >> $OUTPUT/$METHOD/00034/38/stats.txt
#python /home/leon/repositories/supercolmap/main_stats.py 00364 14 $METHOD --featuretype superpointmod --superpoint superpoint_tracking_v3/checkpoints/superPointNet_400000_checkpoint.pth.tar --matchtype disk >> $OUTPUT/$METHOD/00364/14/stats.txt
#python /home/leon/repositories/supercolmap/main_stats.py 02001 39 $METHOD --featuretype superpointmod --superpoint superpoint_tracking_v3/checkpoints/superPointNet_400000_checkpoint.pth.tar --matchtype disk >> $OUTPUT/$METHOD/02001/39/stats.txt
#python /home/leon/repositories/supercolmap/main_stats.py 02003 86 $METHOD --featuretype superpointmod --superpoint superpoint_tracking_v3/checkpoints/superPointNet_400000_checkpoint.pth.tar --matchtype disk >> $OUTPUT/$METHOD/02003/86/stats.txt
#python /home/leon/repositories/supercolmap/main_stats.py 02005 28 $METHOD --featuretype superpointmod --superpoint superpoint_tracking_v3/checkpoints/superPointNet_400000_checkpoint.pth.tar --matchtype disk >> $OUTPUT/$METHOD/02005/28/stats.txt
METHOD="sptracking3_sgh50"
OUTPUT="/media/discoGordo/dataset_leon/reconstructions_MICCAI2023"
python /home/leon/repositories/supercolmap/main_stats.py Seq_095 18 $METHOD --featuretype superpointmod --superpoint superpoint_tracking_v3/checkpoints/superPointNet_400000_checkpoint.pth.tar --matchtype disk >> $OUTPUT/$METHOD/Seq_095_err4/18/stats.txt
python /home/leon/repositories/supercolmap/main_stats.py Seq_001 15 $METHOD --featuretype superpointmod --superpoint superpoint_tracking_v3/checkpoints/superPointNet_400000_checkpoint.pth.tar --matchtype disk >> $OUTPUT/$METHOD/Seq_001_err4/15/stats.txt
python /home/leon/repositories/supercolmap/main_stats.py Seq_002 38 $METHOD --featuretype superpointmod --superpoint superpoint_tracking_v3/checkpoints/superPointNet_400000_checkpoint.pth.tar --matchtype disk >> $OUTPUT/$METHOD/Seq_002_err4/38/stats.txt
python /home/leon/repositories/supercolmap/main_stats.py Seq_095 14 $METHOD --featuretype superpointmod --superpoint superpoint_tracking_v3/checkpoints/superPointNet_400000_checkpoint.pth.tar --matchtype disk >> $OUTPUT/$METHOD/Seq_095_err4/14/stats.txt
python /home/leon/repositories/supercolmap/main_stats.py Seq_014 39 $METHOD --featuretype superpointmod --superpoint superpoint_tracking_v3/checkpoints/superPointNet_400000_checkpoint.pth.tar --matchtype disk >> $OUTPUT/$METHOD/Seq_014_err4/39/stats.txt
python /home/leon/repositories/supercolmap/main_stats.py Seq_016 86 $METHOD --featuretype superpointmod --superpoint superpoint_tracking_v3/checkpoints/superPointNet_400000_checkpoint.pth.tar --matchtype disk >> $OUTPUT/$METHOD/Seq_016_err4/86/stats.txt
python /home/leon/repositories/supercolmap/main_stats.py Seq_017 28 $METHOD --featuretype superpointmod --superpoint superpoint_tracking_v3/checkpoints/superPointNet_400000_checkpoint.pth.tar --matchtype disk >> $OUTPUT/$METHOD/Seq_017_err4/28/stats.txt
#METHOD="sptracking4_disk"
#OUTPUT="/media/discoGordo/dataset_leon/reconstructions_MICCAI2023/"
#python /home/leon/repositories/supercolmap/main_stats.py 00364 18 $METHOD --featuretype superpointmod --superpoint superpoint_tracking_v4/checkpoints/superPointNet_400000_checkpoint.pth.tar --matchtype disk >> $OUTPUT/$METHOD/00364/18/stats.txt
#python /home/leon/repositories/supercolmap/main_stats.py 00033 15 $METHOD --featuretype superpointmod --superpoint superpoint_tracking_v4/checkpoints/superPointNet_400000_checkpoint.pth.tar --matchtype disk >> $OUTPUT/$METHOD/00033/15/stats.txt
#python /home/leon/repositories/supercolmap/main_stats.py 00034 38 $METHOD --featuretype superpointmod --superpoint superpoint_tracking_v4/checkpoints/superPointNet_400000_checkpoint.pth.tar --matchtype disk >> $OUTPUT/$METHOD/00034/38/stats.txt
#python /home/leon/repositories/supercolmap/main_stats.py 00364 14 $METHOD --featuretype superpointmod --superpoint superpoint_tracking_v4/checkpoints/superPointNet_400000_checkpoint.pth.tar --matchtype disk >> $OUTPUT/$METHOD/00364/14/stats.txt
#python /home/leon/repositories/supercolmap/main_stats.py 02001 39 $METHOD --featuretype superpointmod --superpoint superpoint_tracking_v4/checkpoints/superPointNet_400000_checkpoint.pth.tar --matchtype disk >> $OUTPUT/$METHOD/02001/39/stats.txt
#python /home/leon/repositories/supercolmap/main_stats.py 02003 86 $METHOD --featuretype superpointmod --superpoint superpoint_tracking_v4/checkpoints/superPointNet_400000_checkpoint.pth.tar --matchtype disk >> $OUTPUT/$METHOD/02003/86/stats.txt
#python /home/leon/repositories/supercolmap/main_stats.py 02005 28 $METHOD --featuretype superpointmod --superpoint superpoint_tracking_v4/checkpoints/superPointNet_400000_checkpoint.pth.tar --matchtype disk >> $OUTPUT/$METHOD/02005/28/stats.txt
#METHOD="sp_sg"
#OUTPUT="/media/discoGordo/dataset_leon/reconstructions_MICCAI2023/"
#python /home/leon/repositories/supercolmap/main_stats.py 00364 18 $METHOD --featuretype superpoint --superpoint superpoint_tracking_v3/checkpoints/superPointNet_400000_checkpoint.pth.tar --matchtype superglue >> $OUTPUT/$METHOD/00364/18/stats.txt
#python /home/leon/repositories/supercolmap/main_stats.py 00033 15 $METHOD --featuretype superpoint --superpoint superpoint_tracking_v3/checkpoints/superPointNet_400000_checkpoint.pth.tar --matchtype superglue >> $OUTPUT/$METHOD/00033/15/stats.txt
#python /home/leon/repositories/supercolmap/main_stats.py 00034 38 $METHOD --featuretype superpoint --superpoint superpoint_tracking_v3/checkpoints/superPointNet_400000_checkpoint.pth.tar --matchtype superglue >> $OUTPUT/$METHOD/00034/38/stats.txt
#python /home/leon/repositories/supercolmap/main_stats.py 00364 14 $METHOD --featuretype superpoint --superpoint superpoint_tracking_v3/checkpoints/superPointNet_400000_checkpoint.pth.tar --matchtype superglue >> $OUTPUT/$METHOD/00364/14/stats.txt
#python /home/leon/repositories/supercolmap/main_stats.py 02001 39 $METHOD --featuretype superpoint --superpoint superpoint_tracking_v3/checkpoints/superPointNet_400000_checkpoint.pth.tar --matchtype superglue >> $OUTPUT/$METHOD/02001/39/stats.txt
#python /home/leon/repositories/supercolmap/main_stats.py 02003 86 $METHOD --featuretype superpoint --superpoint superpoint_tracking_v3/checkpoints/superPointNet_400000_checkpoint.pth.tar --matchtype superglue >> $OUTPUT/$METHOD/02003/86/stats.txt
#python /home/leon/repositories/supercolmap/main_stats.py 02005 28 $METHOD --featuretype superpoint --superpoint superpoint_tracking_v3/checkpoints/superPointNet_400000_checkpoint.pth.tar --matchtype superglue >> $OUTPUT/$METHOD/02005/28/stats.txt
#METHOD="sift"
#OUTPUT="/media/discoGordo/dataset_leon/colmap_MIDL/pseudoGT/"
#python /home/leon/repositories/supercolmap/main_stats.py 00364 18 $METHOD --featuretype superpoint --superpoint superpoint_tracking_v3/checkpoints/superPointNet_400000_checkpoint.pth.tar --matchtype superglue >> $OUTPUT/00364/18/stats.txt
#python /home/leon/repositories/supercolmap/main_stats.py 00033 15 $METHOD --featuretype superpoint --superpoint superpoint_tracking_v3/checkpoints/superPointNet_400000_checkpoint.pth.tar --matchtype superglue >> $OUTPUT/00033/15/stats.txt
#python /home/leon/repositories/supercolmap/main_stats.py 00034 38 $METHOD --featuretype superpoint --superpoint superpoint_tracking_v3/checkpoints/superPointNet_400000_checkpoint.pth.tar --matchtype superglue >> $OUTPUT/00034/38/stats.txt
#python /home/leon/repositories/supercolmap/main_stats.py 00364 14 $METHOD --featuretype superpoint --superpoint superpoint_tracking_v3/checkpoints/superPointNet_400000_checkpoint.pth.tar --matchtype superglue >> $OUTPUT/00364/14/stats.txt
#python /home/leon/repositories/supercolmap/main_stats.py 02001 39 $METHOD --featuretype superpoint --superpoint superpoint_tracking_v3/checkpoints/superPointNet_400000_checkpoint.pth.tar --matchtype superglue >> $OUTPUT/02001/39/stats.txt
#python /home/leon/repositories/supercolmap/main_stats.py 02003 86 $METHOD --featuretype superpoint --superpoint superpoint_tracking_v3/checkpoints/superPointNet_400000_checkpoint.pth.tar --matchtype superglue >> $OUTPUT/02003/86/stats.txt
#python /home/leon/repositories/supercolmap/main_stats.py 02005 28 $METHOD --featuretype superpoint --superpoint superpoint_tracking_v3/checkpoints/superPointNet_400000_checkpoint.pth.tar --matchtype superglue >> $OUTPUT/02005/28/stats.txt