import database_functions as dataFunctions
import change_format as cf
import glob, os
import subprocess
import camera_position as cm
import numpy as np
import argparse
from pathlib import Path


def triangulate(database_name, path_images, path_results):
    subprocess.run([path_colmap, 'mapper',
                    '--database_path', path_results + '/' + database_name,
                    '--image_path', path_images,
                    '--output_path', path_results,
                    '--Mapper.init_min_tri_angle', '8',
                    '--Mapper.ba_refine_focal_length', '0',
                    '--Mapper.ba_refine_extra_params', '0',
                    '--Mapper.min_model_size', '50'])
    if os.path.exists(path_results + '/0'):
        subprocess.run([path_colmap, 'model_converter',
                        '--input_path', path_results + '/0',
                        '--output_path', path_results,
                        '--output_type', 'TXT'])


def initialize_database(database_name, path_images, path_results):
    subprocess.run([path_colmap, 'feature_extractor',
                    '--database_path', path_results + '/' + database_name,
                    '--image_path', path_images,
                    '--ImageReader.camera_model', 'OPENCV_FISHEYE',
                    '--ImageReader.single_camera', '1',
                    '--ImageReader.camera_params',
                    '733.1061; 735.719; 739.2826; 539.6911; -0.1205372; -0.01179983; 0.00269742; -0.0001362505',
                    '--SiftExtraction.use_gpu', 'true',
                    '--SiftExtraction.gpu_index', '0'])
    subprocess.run([path_colmap, 'exhaustive_matcher',
                    '--database_path', path_results + '/' + database_name,
                    '--SiftMatching.guided_matching', '1'])


def matches_extraction(path_images, path_results):
    Path(path_results).mkdir(exist_ok=True, parents=True)
    f = open(path_results + '/images_to_match.txt', "w+")
    images_name = {}
    unordered = []
    for file in os.listdir(path_images):
        if file.endswith(".png") or file.endswith(".jpg"):
            termination = file[-4:]
            unordered.append(
                os.path.join(path_images, file).split('/')[-1].split('.')[
                    0])
    unordered.sort()

    for i in range(0, len(unordered)):
        images_name[i + 1] = unordered[i]

    for i in range(1, len(images_name) + 1):
        for j in range(i + 1, len(images_name) + 1):
            f.write(images_name[i] + termination)
            f.write(' ')
            f.write(images_name[j] + termination)
            f.write('\n')

    f.close()
    subprocess.run(['python', '/home/leon/repositories/supercolmap/match_pairs.py',
                    '--input_pairs', path_results + '/images_to_match.txt',
                    '--input_dir', path_images,
                    '--output_dir', path_results + '/dump_match_pairs/',
                    '--max_keypoints', '10000',
                    '--resize', '-1',
                    # '--viz',
                    ])


def Save_database(path_database, path_images, path_results):
    images_name = {}
    unordered = []
    for file in os.listdir(path_images):
        if file.endswith(".png") or file.endswith(".jpg"):
            unordered.append(os.path.join(path_images, file).split('/')[-1].split('.')[0])
    unordered.sort()

    for i in range(0, len(unordered)):
        images_name[i + 1] = unordered[i]

    path = {}
    for i in range(1, len(images_name)):
        for j in range(i + 1, len(images_name) + 1):
            num = (i, j)
            path[num] = path_results + '/dump_match_pairs/' + images_name[i] + '_' + images_name[j] + '_matches.npz'

    keypoints = {}
    matches = {}
    for num in path:
        keypoints[num[0]] = cf.load_keypoints0(path[num])
        matches[num] = cf.load_matches(path[num])
    keypoints[len(images_name)] = cf.load_keypoints1(path[(1, len(images_name))])

    # Open the database.

    db = dataFunctions.COLMAPDatabase.connect(path_results + '/' + path_database)

    for num in keypoints:
        db.update_keypoints(num, keypoints[num])

    for num in matches:
        if matches[num].shape[0] > 0:
            db.update_matches(num[0], num[1], matches[num])
        else:
            db.update_matches(num[0], num[1], np.array([[0, 0]]))

    db.delete_two_view_geometry()

    db.commit()
    db.close()
    print('finish')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Reconstruct with \
            SuperPoint feature matches.')
    parser.add_argument('video', type=str, default='00033')
    parser.add_argument('cluster', type=str, default='19')
    args = parser.parse_args()

    video = args.video
    cluster = args.cluster
    f = open('/media/discoGordo/dataset_leon/colmap_2023/reconstruction_2023_sg/log.txt', "a+")
    for video in os.listdir('/media/discoGordo/dataset_leon/UZ/training/training_colmap_frames/'):
        for cluster in os.listdir('/media/discoGordo/dataset_leon/UZ/training/training_colmap_frames/' + video):
            if os.path.exists('/media/discoGordo/dataset_leon/colmap_2023/reconstruction_2023_sg/' + video + '/' + cluster + '/0/'):
                continue
            f.write(video+" "+cluster)
            path_colmap = '/home/leon/repositories/colmap/build/src/exe/colmap'  # 'COLMAP.app/Contents/MacOS/colmap'
            path_images = '/media/discoGordo/dataset_leon/UZ/training/training_colmap_frames/' + video + '/' + cluster  # '/Users/mjimenez/pythonProject8/assets/alfombra'  # it has to be absolute route for colmap
            path_results = '/media/discoGordo/dataset_leon/colmap_2023/reconstruction_2023_sg/' + video + '/' + cluster  # '/Users/mjimenez/pythonProject8/results'
            database_name = 'database.db'  # name for the databaase we will create. It overwrites an existing one if it already exists with that same name in the same directory

            # matches_extraction(path_images,
            #                   path_results)  # creates the matches with superglue. Posible adjustment of parameters in the function definition
            # exit(1)
            initialize_database(database_name, path_images, path_results) #creates a database with colmap
            Save_database(database_name, path_images, path_results) #Saves the matches from superglue to colmap database
            subprocess.run([path_colmap, 'exhaustive_matcher', '--database_path', path_results+'/'+database_name])
            triangulate(database_name, path_images, path_results)  #Triangulates point and camera positions with colmap
            # exit(1)
    f.close()
    # T = cm.print_camera_positions(path_results)  #Print results of camera positiosn