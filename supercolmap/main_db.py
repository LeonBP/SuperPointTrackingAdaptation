import database_functions as dataFunctions
import change_format as cf
import glob, os
import subprocess
import camera_position as cm
import numpy as np
import argparse
from pathlib import Path


def Save_database(path_database, path_images, path_results, seq, subseq):

    # Open the database.

    db = dataFunctions.COLMAPDatabase.connect(path_results + '/' + path_database)
    if False:
        print(db.get_table_names())
    else:
        # Take all images from the database
        images = db.get_all_images()
        # Update the names of each image in the database
        for image in images:
            image_name = seq+'_'+subseq.zfill(3)+'_'+image[1][-7:]
            db.update_image_name(image[0], image_name)

        db.commit()
    db.close()
    print(seq+'/'+subseq)


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
    args = parser.parse_args()

    video = args.video
    cluster = args.cluster
    model = args.model

    path_colmap = '/home/leon/repositories/colmap/build/src/exe/colmap'  # 'COLMAP.app/Contents/MacOS/colmap'
    path_dataset = '/media/discoGordo/dataset_leon//UZ/colmap_benchmark_frames'
    if model == 'sift':
        path_recons = '/media/discoGordo/dataset_leon/colmap_MIDL/pseudoGT'
    else:
        path_recons = '/media/discoGordo/dataset_leon/reconstructions_MICCAI2023' + '/' + model + '/'  # '/Users/mjimenez/pythonProject8/results'
    for video in os.listdir(path_dataset):
        for cluster in os.listdir(path_dataset + '/' + video):
            path_images = path_dataset + '/' + video + '/' + cluster  # '/Users/mjimenez/pythonProject8/assets/alfombra'  # it has to be absolute route for colmap
            path_results = path_recons + '/' + video + '/' + cluster  # '/Users/mjimenez/pythonProject8/results'
            database_name = 'database.db'  # name for the databaase we will create. It overwrites an existing one if it already exists with that same name in the same directory
            if os.path.exists(path_results):
                Save_database(database_name, path_images, path_results, video, cluster)
    #triangulate(database_name, path_images, path_results)  #Triangulates point and camera positions with colmap

    # T = cm.print_camera_positions(path_results)  #Print results of camera positiosn