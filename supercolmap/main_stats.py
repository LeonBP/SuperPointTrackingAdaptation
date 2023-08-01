import database_functions as dataFunctions
import change_format as cf
import glob, os
import subprocess
import camera_position as cm
import numpy as np
import argparse
from pathlib import Path


def triangulate(database_name, path_images, path_results):
    if os.path.exists(path_results + '/0'):
        subprocess.run([path_colmap, 'model_analyzer',
                        '--path', path_results + '/0'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Reconstruct with \
            SuperPoint feature matches.')
    parser.add_argument('video', type=str, default='00033')
    parser.add_argument('cluster', type=str, default='19')
    parser.add_argument('model', type=str, default='sp_bf')
    parser.add_argument('--featuretype', type=str, default='superpoint')
    parser.add_argument('--superpoint', type=str,
                        default='superpoint_ucluzlabel100_specga_9-4_d3/checkpoints/superPointNet_200000_checkpoint.pth.tar')
    parser.add_argument('--matchtype', type=str, default='bruteforce')
    parser.add_argument('--superglue', type=str,
                        default='superglue_pretrained/superglue_indoor.pth')
    parser.add_argument('--reperror', type=str, default='4')
    args = parser.parse_args()

    video = args.video
    cluster = args.cluster
    model = args.model

    path_colmap = '/home/leon/repositories/colmap/build/src/exe/colmap'  # 'COLMAP.app/Contents/MacOS/colmap'
    path_images = '/media/discoGordo/dataset_leon/UZ/colmap_benchmark_frames/' + video + '/' + cluster  # '/Users/mjimenez/pythonProject8/assets/alfombra'  # it has to be absolute route for colmap
    if model == "sift":
        path_results = "/media/discoGordo/dataset_leon/colmap_MIDL/pseudoGT/" + video + "/" + cluster
    else:
        path_results = '/media/discoGordo/dataset_leon/reconstructions_MICCAI2023/' + model + '/' + video + '_err' + args.reperror + '/' + cluster  # '/Users/mjimenez/pythonProject8/results'
    database_name = 'database.db'  # name for the databaase we will create. It overwrites an existing one if it already exists with that same name in the same directory

    triangulate(database_name, path_images, path_results)  #Triangulates point and camera positions with colmap

    # T = cm.print_camera_positions(path_results)  #Print results of camera positiosn