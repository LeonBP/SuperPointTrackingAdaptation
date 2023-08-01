import database_functions as dataFunctions
import change_format as cf
import glob, os
import subprocess
import camera_position as cm
import numpy as np
import argparse
from pathlib import Path


def triangulate(database_name, path_images, path_results, error):
    subprocess.run([path_colmap, 'exhaustive_matcher',
                    '--database_path', path_results + '/' + database_name,
                    '--SiftMatching.use_gpu', 'false',
                    '--SiftMatching.max_ratio', '1.0',
                    '--SiftMatching.max_distance', '1.0',
                    '--SiftMatching.guided_matching', '1',
                    # '--SiftMatching.max_error', '8',
                    # '--SiftMatching.num_threads', '1',
                    ])
    db = dataFunctions.COLMAPDatabase.connect(path_results + '/' + database_name)
    # print(db.get_matches(1, 2)[0][:3],
    #       db.get_keypoints(1)[0][:3], db.get_keypoints(2)[0][:3])
    print(len(db.get_all_cameras()), len(db.get_all_images()), len(db.get_all_keypoints()),
          len(db.get_all_descriptors()), len(db.get_all_matches()), len(db.get_all_two_view_geometries()))
    db.close()
    print("GUIDED DONE")
    # subprocess.run([path_colmap, 'exhaustive_matcher', '--database_path', path_results + '/' + database_name])
    # db = dataFunctions.COLMAPDatabase.connect(path_results + '/' + database_name)
    # # print(db.get_matches(1, 2)[0][:3],
    # #       db.get_keypoints(1)[0][:3], db.get_keypoints(2)[0][:3])
    # print(len(db.get_all_cameras()), len(db.get_all_images()), len(db.get_all_keypoints()),
    #       len(db.get_all_descriptors()), len(db.get_all_matches()), len(db.get_all_two_view_geometries()))
    # db.close()
    # print("EXHAUSTIVE DONE")
    subprocess.run([path_colmap, 'mapper',
                    '--database_path', path_results + '/' + database_name,
                    '--image_path', path_images,
                    '--output_path', path_results,
                    '--Mapper.init_min_tri_angle', '8',
                    '--Mapper.ba_refine_focal_length', '0',
                    '--Mapper.ba_refine_extra_params', '0',
                    '--Mapper.min_model_size', '50',
                    '--Mapper.filter_max_reproj_error', error,
                    ])
    print("MAPPER DONE")
    if os.path.exists(path_results + '/0'):
        subprocess.run([path_colmap, 'model_converter',
                        '--input_path', path_results + '/0',
                        '--output_path', path_results,
                        # '--output_type', 'BIN'])
                        '--output_type', 'TXT'])
    print("CONVERTER DONE")


def initialize_database(database_name, path_images, path_results):
    print("INITIALIZE DATABASE")
    subprocess.run([path_colmap, 'feature_extractor',
                    '--database_path', path_results + '/' + database_name,
                    '--image_path', path_images,
                    '--ImageReader.camera_model', 'OPENCV_FISHEYE',
                    '--ImageReader.single_camera', '1',
                    '--ImageReader.camera_params',
                    '733.1061; 735.719; 739.2826; 539.6911; -0.1205372; -0.01179983; 0.00269742; -0.0001362505',
                    '--SiftExtraction.use_gpu', 'true',
                    '--SiftExtraction.gpu_index', '0'])
    print("FEATURE EXTRACTOR DONE")
    # subprocess.run([path_colmap, 'exhaustive_matcher',
    #                 '--database_path', path_results + '/' + database_name,
    #                 '--SiftMatching.guided_matching', '1'])
    # print("EXHAUSTIVE DONE")


def matches_extraction(path_images, path_results, featuretype, sp_weights_path, matchtype, sg_weights_path):
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
            if matchtype == 'colmap':
                break

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
                    # '--viz'
                    ])


def Save_database(path_database, path_images, path_results, error, matchtype):
    images_name = {}
    unordered = []
    for file in os.listdir(path_images):
        if file.endswith(".png") or file.endswith(".jpg"):
            unordered.append(os.path.join(path_images, file).split('/')[-1].split('.')[0])
    unordered.sort()

    for i in range(0, len(unordered)):
        images_name[i + 1] = unordered[i]

    path = {}
    # if error != '4':
    #     if path_results.find("_err") != -1:
    #         path_results_src = path_results[:path_results.find("_err")] + path_results[
    #                                                                       path_results.find("_err") + 5:]  # +"_err"+'3'
    #     else:
    #         path_results_src = path_results + "_err" + error
    # else:
    #     path_results_src = path_results
    for i in range(1, len(images_name)):
        for j in range(i + 1, len(images_name) + 1):
            num = (i, j)
            path[num] = path_results + '/dump_match_pairs/' + images_name[i] + '_' + images_name[j] + '_matches.npz'
            if matchtype == 'colmap':
                break
    print("PATHS DONE")
    keypoints = {}
    descriptors = {}
    matches = {}
    for num in path:
        if num[0] not in keypoints.keys():
            #print(num)
            keypoints[num[0]], descriptors[num[0]] = cf.load_kps_desc0(path[num])
        # matches[num] = cf.load_matches(path[num])
    keypoints[len(images_name)], descriptors[len(images_name)] = cf.load_kps_desc1(path[(len(images_name)-1, len(images_name))])
    # print(descriptors.keys(), descriptors[len(images_name)].shape)
    # Open the database.
    print("LOADING DONE")
    db = dataFunctions.COLMAPDatabase.connect(path_results + '/' + path_database)
    # db.add_camera(5, 1440, 1080,
    #               [733.1061, 735.719, 739.2826, 539.6911, -0.1205372, -0.01179983, 0.00269742, -0.0001362505], True, 0)
    for num in keypoints:
        # db.add_image(images_name[num] + '.png', 1, image_id=num)
        # print(db.get_image_id(num)[0],len(db.get_keypoints(num)[0]), len(db.get_descriptors(num)[0]), db.get_descriptors(num)[0][1])
        db.update_keypoints(num, keypoints[num])
        desc = descriptors[num].T#(descriptors[num].T + 0.5) * 255.
        db.update_descriptors_float(num, desc)
        # print(db.get_image_id(num)[0],len(db.get_keypoints(num)[0]), len(db.get_descriptors(num)[0]), db.get_descriptors(num)[0][1], len(dataFunctions.blob_to_array(db.get_descriptors(num)[0][3],dtype=np.float32))/256, np.sum(desc-dataFunctions.blob_to_array(db.get_descriptors(num)[0][3],dtype=np.float32).reshape(-1,256)))
        # exit(1)
    print("UPDATES DONE")
    # for num in path:
    #     if False and matches[num].shape[0] > 0:
    #         db.add_matches(num[0], num[1], matches[num])
    #     else:
    #         print(num)
    #         db.add_matches(num[0], num[1], np.array([[0, 0]]))
    #         print(db.get_matches(num[0], num[1])[0])
    # #         exit(1)
    #
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
    args = parser.parse_args()

    video = args.video
    cluster = args.cluster
    model = args.model

    path_colmap = '/home/leon/repositories/colmap/build/src/exe/colmap'  # 'COLMAP.app/Contents/MacOS/colmap'
    # path_images = '/media/discoGordo/dataset_leon/UZ/colmap_benchmark_frames/' + video + '/' + cluster  # '/Users/mjimenez/pythonProject8/assets/alfombra'  # it has to be absolute route for colmap
    path_images = '/media/discoGordo/C3VD/' + video + '/' + cluster  # '/Users/mjimenez/pythonProject8/assets/alfombra'  # it has to be absolute route for colmap
    if args.reperror != '4':
        path_results = '/media/discoGordo/dataset_leon/reconstructions_MICCAI2023/' + model + '/' + video + '_err' + args.reperror + '/' + cluster
    else:
        path_results = '/media/discoGordo/dataset_leon/reconstructions_MICCAI2023/' + model + '/' + video + '/' + cluster  #  + '_err' + args.reperror
    database_name = 'database.db'  # name for the databaase we will create. It overwrites an existing one if it already exists with that same name in the same directory
    print("NADA")
    Path(path_results).mkdir(exist_ok=True, parents=True)
    matches_extraction(path_images, path_results, args.featuretype, args.superpoint, args.matchtype, args.superglue)
    # creates the matches with superglue. Possible adjustment of parameters in the function definition
    # exit(1)
    initialize_database(database_name, path_images, path_results) #creates a database with colmap
    db = dataFunctions.COLMAPDatabase.connect(path_results + '/' + database_name)
    # print(db.get_matches(1, 2)[0][:3],
    #       db.get_keypoints(1)[0][:3], db.get_keypoints(2)[0][:3])
    print(len(db.get_all_cameras()), len(db.get_all_images()), len(db.get_all_keypoints()),
          len(db.get_all_descriptors()), len(db.get_all_matches()), len(db.get_all_two_view_geometries()))
    db.close()

    Save_database(database_name, path_images, path_results, args.reperror, args.matchtype) #Saves the matches from superglue to colmap database
    db = dataFunctions.COLMAPDatabase.connect(path_results + '/' + database_name)
    # print(db.get_matches(1, 2)[0][:3],
    #       db.get_keypoints(1)[0][:3], db.get_keypoints(2)[0][:3])
    print(len(db.get_all_cameras()), len(db.get_all_images()), len(db.get_all_keypoints()),
          len(db.get_all_descriptors()), len(db.get_all_matches()), len(db.get_all_two_view_geometries()))
    db.close()

    # subprocess.run([path_colmap, 'exhaustive_matcher', '--database_path', path_results+'/'+database_name])
    # db = dataFunctions.COLMAPDatabase.connect(path_results + '/' + database_name)
    # # print(db.get_matches(1, 2)[0][:3],
    # #       db.get_keypoints(1)[0][:3], db.get_keypoints(2)[0][:3])
    # print(len(db.get_all_cameras()), len(db.get_all_images()), len(db.get_all_keypoints()),
    #       len(db.get_all_descriptors()), len(db.get_all_matches()), len(db.get_all_two_view_geometries()))
    # db.close()

    triangulate(database_name, path_images, path_results, args.reperror)  #Triangulates point and camera positions with colmap

    T = cm.print_camera_positions(path_results)  #Print results of camera positiosn