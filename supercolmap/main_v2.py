import database_functions as dataFunctions
import change_format as cf
import glob, os
import subprocess
import camera_position as cm
import numpy as np
import argparse
from pathlib import Path


def triangulate(database_name, path_images, path_results, error, min_size='25'):
    subprocess.run([path_colmap, 'mapper',
                    '--database_path', path_results + '/' + database_name,
                    '--image_path', path_images,
                    '--output_path', path_results,
                    '--Mapper.init_min_tri_angle', '8',
                    '--Mapper.ba_refine_focal_length', '0',
                    '--Mapper.ba_refine_extra_params', '0',
                    '--Mapper.min_model_size', min_size,
                    '--Mapper.filter_max_reproj_error', error])
    # if os.path.exists(path_results + '/0'):
    #     subprocess.run([path_colmap, 'model_converter',
    #                     '--input_path', path_results + '/0',
    #                     '--output_path', path_results,
    #                     '--output_type', 'TXT'])
    for res in os.listdir(path_results):
        if res.isdigit():
            subprocess.run([path_colmap, 'model_converter',
                            '--input_path', path_results + '/' + res,
                            '--output_path', path_results + '/' + res,
                            '--output_type', 'TXT'])


def initialize_database(database_name, path_images, path_results, overlap):
    subprocess.run([path_colmap, 'feature_extractor',
                    '--database_path', path_results + '/' + database_name,
                    '--image_path', path_images,
                    '--ImageReader.camera_model', 'OPENCV_FISHEYE',
                    '--ImageReader.single_camera', '1',
                    '--ImageReader.camera_params',
                    '733.1061; 735.719; 739.2826; 539.6911; -0.1205372; -0.01179983; 0.00269742; -0.0001362505',
                    '--SiftExtraction.use_gpu', 'true',
                    '--SiftExtraction.gpu_index', '0'])
    # subprocess.run([path_colmap, 'exhaustive_matcher',
    #                 '--database_path', path_results + '/' + database_name,
    #                 '--SiftMatching.guided_matching', '1'])
    subprocess.run([path_colmap, 'sequential_matcher',
                    '--database_path', path_results + '/' + database_name,
                    '--SiftMatching.guided_matching', '1',
                    '--SequentialMatching.overlap', overlap,
                    '--SequentialMatching.quadratic_overlap', '1'])


def matches_extraction(path_images, path_results, featuretype, sp_weights_path, matchtype, sg_weights_path, overlap):
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

    # for i in range(1, len(images_name) + 1):
    #     for j in range(i + 1, min(len(images_name) + 1, i + int(overlap) + 1)):
    #         f.write(images_name[i] + termination)
    #         f.write(' ')
    #         f.write(images_name[j] + termination)
    #         f.write('\n')
    for i in range(1, len(images_name) + 1):
        for j in range(0, min(len(images_name) + 1-i, int(overlap) + 1)):
            f.write(images_name[i] + termination)
            f.write(' ')
            f.write(images_name[i+j] + termination)
            f.write('\n')
            if 2 ** j > int(overlap):
                i_quadratic = i + 2 ** j
                if i_quadratic < len(images_name) + 1:
                    f.write(images_name[i] + termination)
                    f.write(' ')
                    f.write(images_name[i_quadratic] + termination)
                    f.write('\n')

    f.close()
    subprocess.run(['python', '/home/leon/repositories/supercolmap/match_pairs.py',
                    '--input_pairs', path_results + '/images_to_match.txt',
                    '--input_dir', path_images,
                    '--output_dir', path_results + '/dump_match_pairs/',
                    '--featuretype', featuretype,
                    '--superpoint', sp_weights_path,
                    '--matchtype', matchtype,
                    '--superglue', sg_weights_path,
                    '--max_keypoints', '10000',
                    '--nms_radius', '4',
                    '--resize', '-1',
                    # '--viz',
                    ])

def Save_database(path_database, path_images, path_results, error, overlap):
    images_name = {}
    unordered = []
    for file in os.listdir(path_images):
        if file.endswith(".png") or file.endswith(".jpg"):
            unordered.append(os.path.join(path_images, file).split('/')[-1].split('.')[0])
    unordered.sort()
    print("PATHS DONE")

    for i in range(0, len(unordered)):
        images_name[i + 1] = unordered[i]

    path = {}
    if error != '4':
        if path_results.find("_err") != -1:
            path_results_src = path_results[:path_results.find("_err")]+path_results[path_results.find("_err")+5:]#+"_err"+'3'
        else:
            path_results_src = path_results+"_err"+error
    else:
        path_results_src = path_results
    # for i in range(1, len(images_name)):
    #     for j in range(i + 1, min(len(images_name) + 1, i + int(overlap) + 1)):
    #         num = (i, j)
    #         path[num] = path_results_src + '/dump_match_pairs/' + images_name[i] + '_' + images_name[j] + '_matches.npz'
    for i in range(1, len(images_name) + 1):
        for j in range(0, min(len(images_name) + 1 - i, int(overlap) + 1)):
            num = (i, i+j)
            path[num] = path_results_src + '/dump_match_pairs/' + images_name[i] + '_' + images_name[i+j] + '_matches.npz'
            if 2 ** j > int(overlap):
                i_quadratic = i + 2 ** j
                if i_quadratic < len(images_name) + 1:
                    num = (i, i_quadratic)
                    path[num] = path_results_src + '/dump_match_pairs/' + images_name[i] + '_' + images_name[i_quadratic] + '_matches.npz'
    print("LOADING DONE")


    # Open the database.

    db = dataFunctions.COLMAPDatabase.connect(path_results + '/' + path_database)
    keypoints = {}
    # matches = {}
    for num in path:
        if num[0] not in keypoints:
            keypoints[num[0]] = cf.load_keypoints0(path[num])
        matches = cf.load_matches(path[num])
        if matches.shape[0] > 0:
            db.update_matches(num[0], num[1], matches)
        else:
            db.update_matches(num[0], num[1], np.array([[0, 0]]))
    keypoints[len(images_name)] = cf.load_keypoints1(path[(len(images_name)-1, len(images_name))])

    for num in keypoints:
        db.update_keypoints(num, keypoints[num])

    # for num in matches:
    #     if matches[num].shape[0] > 0:
    #         db.update_matches(num[0], num[1], matches[num])
    #     else:
    #         db.update_matches(num[0], num[1], np.array([[0, 0]]))
    print("UPDATES DONE")

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
    parser.add_argument('model', type=str, default='sp_bf')
    parser.add_argument('--featuretype', type=str, default='superpoint')
    parser.add_argument('--superpoint', type=str,
                        default='superpoint_ucluzlabel100_specga_9-4_d3/checkpoints/superPointNet_200000_checkpoint.pth.tar')
    parser.add_argument('--matchtype', type=str, default='bruteforce')
    parser.add_argument('--superglue', type=str,
                        default='superglue_pretrained/superglue_indoor.pth')
    parser.add_argument('--reperror', type=str, default='4')
    parser.add_argument('--overlap', type=str, default='10')
    args = parser.parse_args()

    video = args.video
    cluster = args.cluster if args.cluster != '0' else args.overlap
    model = args.model

    path_colmap = '/home/leon/repositories/colmap/build/src/exe/colmap'  # 'COLMAP.app/Contents/MacOS/colmap'
    # path_images = '/media/discoGordo/dataset_leon/UZ/colmap_benchmark_frames/' + video + '/' + cluster  # '/Users/mjimenez/pythonProject8/assets/alfombra'  # it has to be absolute route for colmap
    # path_images = '/media/discoGordo/C3VD/' + video + '/' + cluster  # '/Users/mjimenez/pythonProject8/assets/alfombra'  # it has to be absolute route for colmap
    path_images = '/media/discoGordo/dataset_leon/UZ/test_frames' + '/' + video # it has to be absolute route for colmap
    if args.reperror != '4':
        path_results = '/media/discoGordo/dataset_leon/reconstructions_MICCAI2023/' + model + '/' + video + '_err' + args.reperror + '/' + cluster
    else:
        path_results = '/media/discoGordo/dataset_leon/reconstructions_MICCAI2023/' + model + '/' + video + '/' + cluster
    database_name = 'database.db'  # name for the databaase we will create. It overwrites an existing one if it already exists with that same name in the same directory
    print("NADA")
    Path(path_results).mkdir(exist_ok=True, parents=True)
    if "sift_gm" not in model:
        print("a")
        matches_extraction(path_images, path_results, args.featuretype, args.superpoint, args.matchtype, args.superglue, args.overlap)
    # creates the matches with superglue. Possible adjustment of parameters in the function definition
    # exit(1)
    initialize_database(database_name, path_images, path_results, args.overlap) #creates a database with colmap
    db = dataFunctions.COLMAPDatabase.connect(path_results + '/' + database_name)
    # print(db.get_matches(1, 2)[0][:3],
    #       db.get_keypoints(1)[0][:3], db.get_keypoints(2)[0][:3])
    print(len(db.get_all_cameras()), len(db.get_all_images()), len(db.get_all_keypoints()),
          len(db.get_all_descriptors()), len(db.get_all_matches()), len(db.get_all_two_view_geometries()))
    db.close()

    if "sift_gm" not in model:
        Save_database(database_name, path_images, path_results, args.reperror, args.overlap) #Saves the matches from superglue to colmap database
        db = dataFunctions.COLMAPDatabase.connect(path_results + '/' + database_name)
        # print(db.get_matches(1, 2)[0][:3],
        #       db.get_keypoints(1)[0][:3], db.get_keypoints(2)[0][:3])
        print(len(db.get_all_cameras()), len(db.get_all_images()), len(db.get_all_keypoints()),
              len(db.get_all_descriptors()), len(db.get_all_matches()), len(db.get_all_two_view_geometries()))
        db.close()

        # subprocess.run([path_colmap, 'exhaustive_matcher', '--database_path', path_results+'/'+database_name])
        subprocess.run([path_colmap, 'sequential_matcher',
                        '--database_path', path_results + '/' + database_name,
                        # '--SiftMatching.guided_matching', '1',
                        '--SequentialMatching.overlap', args.overlap,
                        '--SequentialMatching.quadratic_overlap', '1'])
        db = dataFunctions.COLMAPDatabase.connect(path_results + '/' + database_name)
        # print(db.get_matches(1, 2)[0][:3],
        #       db.get_keypoints(1)[0][:3], db.get_keypoints(2)[0][:3])
        print(len(db.get_all_cameras()), len(db.get_all_images()), len(db.get_all_keypoints()),
              len(db.get_all_descriptors()), len(db.get_all_matches()), len(db.get_all_two_view_geometries()))
        db.close()

    triangulate(database_name, path_images, path_results, args.reperror)  #Triangulates point and camera positions with colmap

    # T = cm.print_camera_positions(path_results)  #Print results of camera positiosn