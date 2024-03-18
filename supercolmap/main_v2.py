import database_functions as dataFunctions
import change_format as cf
import glob, os
import subprocess
import camera_position as cm
import numpy as np
import argparse
from pathlib import Path
import time


def triangulate(database_name, path_images, path_results, error, matchtype, max_distance, overlap, min_size='50'):
    if matchtype == 'colmap':
        if 'Full_' in path_results[path_results.rfind('/'):]:
            subprocess.run([path_colmap, 'sequential_matcher',
                            '--database_path', path_results + '/' + database_name,
                            '--SiftMatching.use_gpu', 'false',
                            '--SiftMatching.max_ratio', '1.0',
                            '--SiftMatching.max_distance', max_distance,
                            '--SiftMatching.guided_matching', '1',
                            '--SequentialMatching.overlap', overlap,
                            '--SequentialMatching.quadratic_overlap', '1'])
        else:
            subprocess.run([path_colmap, 'exhaustive_matcher',
                            '--database_path', path_results + '/' + database_name,
                            '--SiftMatching.use_gpu', 'false',
                            '--SiftMatching.max_ratio', '1.0',
                            '--SiftMatching.max_distance', max_distance,
                            '--SiftMatching.guided_matching', '1',
                            # '--SiftMatching.max_error', '4',
                            # '--SiftMatching.num_threads', '1',
                            ])
    elif 'sift_gm' not in path_results:
        if 'Full_' in path_results[path_results.rfind('/'):]:
            subprocess.run([path_colmap, 'sequential_matcher',
                            '--database_path', path_results + '/' + database_name,
                            # '--SiftMatching.guided_matching', '1',
                            '--SequentialMatching.overlap', args.overlap,
                            '--SequentialMatching.quadratic_overlap', '1'])
        else:
            subprocess.run([path_colmap, 'exhaustive_matcher', '--database_path', path_results + '/' + database_name])
    subprocess.run([path_colmap, 'mapper',
                    '--database_path', path_results + '/' + database_name,
                    '--image_path', path_images,
                    '--output_path', path_results,
                    '--Mapper.init_min_tri_angle', '8',
                    '--Mapper.ba_refine_focal_length', '0',
                    '--Mapper.ba_refine_extra_params', '0',
                    '--Mapper.min_model_size', min_size,
                    '--Mapper.filter_max_reproj_error', error])

    for res in os.listdir(path_results):
        if res.isdigit():
            subprocess.run([path_colmap, 'model_converter',
                            '--input_path', path_results + '/' + res,
                            '--output_path', path_results + '/' + res,
                            '--output_type', 'TXT'])
            f = open(path_results + '/' + res + '/stats.txt', "w+")
            subprocess.run([path_colmap, 'model_analyzer',
                            '--path', path_results + '/' + res], stdout=f)
            f.close()


def initialize_database(database_name, path_images, path_results, matchtype, overlap):
    subprocess.run([path_colmap, 'feature_extractor',
                    '--database_path', path_results + '/' + database_name,
                    '--image_path', path_images,
                    '--ImageReader.camera_model', 'OPENCV_FISHEYE',
                    '--ImageReader.single_camera', '1',
                    '--ImageReader.camera_params',
                    '733.1061; 735.719; 739.2826; 539.6911; -0.1205372; -0.01179983; 0.00269742; -0.0001362505',
                    '--SiftExtraction.use_gpu', 'true',
                    '--SiftExtraction.gpu_index', '0'])
    if matchtype != 'colmap':
        if 'Full_' in path_results[path_results.rfind('/'):]:
            subprocess.run([path_colmap, 'sequential_matcher',
                            '--database_path', path_results + '/' + database_name,
                            '--SiftMatching.guided_matching', '1',
                            # '--SiftMatching.use_gpu', 'false',
                            '--SequentialMatching.overlap', overlap,
                            '--SequentialMatching.quadratic_overlap', '1'])
        else:
            subprocess.run([path_colmap, 'exhaustive_matcher',
                            '--database_path', path_results + '/' + database_name,
                            '--SiftMatching.guided_matching', '1',
                            # '--SiftMatching.use_gpu', 'false'
                            ])
    print("DB INIT DONE")


def matches_extraction(path_images, path_results, featuretype, sp_weights_path, matchtype, sg_weights_path, keypoint_threshold, overlap):
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

    if 'Full_' in path_results[path_results.rfind('/'):]:
        for i in range(1, len(images_name) + 1):
            for j in range(0, min(len(images_name) + 1-i, int(overlap) + 1)):
                if j != 0:
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
    else:
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
                    '--keypoint_threshold', keypoint_threshold,
                    '--nms_radius', '4',
                    '--resize', '-1',
                    # '--viz',
                    ])
    print("EXTRACTION DONE")

def Save_database(path_database, path_images, path_results, error, matchtype, overlap):
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

    if 'Full_' in path_results[path_results.rfind('/'):]:
        for i in range(1, len(images_name) + 1):
            for j in range(0, min(len(images_name) + 1 - i, int(overlap) + 1)):
                if j != 0:
                    num = (i, i+j)
                    path[num] = path_results_src + '/dump_match_pairs/' + images_name[i] + '_' + images_name[i+j] + '_matches.npz'
                if 2 ** j > int(overlap):
                    i_quadratic = i + 2 ** j
                    if i_quadratic < len(images_name) + 1:
                        num = (i, i_quadratic)
                        path[num] = path_results_src + '/dump_match_pairs/' + images_name[i] + '_' + images_name[i_quadratic] + '_matches.npz'
    else:
        for i in range(1, len(images_name)):
            for j in range(i + 1, len(images_name) + 1):
                num = (i, j)
                path[num] = path_results_src + '/dump_match_pairs/' + images_name[i] + '_' + images_name[j] + '_matches.npz'
                if matchtype == 'colmap':
                    break
    print("LOADING DONE")


    # Open the database.

    db = dataFunctions.COLMAPDatabase.connect(path_results + '/' + path_database)
    keypoints = {}
    descriptors = {}
    if matchtype != 'colmap':
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
    else:
        for num in path:
            if num[0] not in keypoints:
                # print(num)
                keypoints[num[0]], descriptors[num[0]] = cf.load_kps_desc0(path[num])
        keypoints[len(images_name)], descriptors[len(images_name)] = cf.load_kps_desc1(
            path[(len(images_name) - 1, len(images_name))])
        for num in keypoints:
            db.update_keypoints(num, keypoints[num])
            desc = descriptors[num].T  # (descriptors[num].T + 0.5) * 255.
            db.update_descriptors_float(num, desc)
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
    parser.add_argument('--maxdistance', type=str, default='1.0')
    parser.add_argument('--keypoint_threshold', type=str, default='0.015')
    parser.add_argument('--overlap', type=str, default='10')
    parser.add_argument('--minsize', type=str, default='50')
    args = parser.parse_args()

    video = args.video
    cluster = args.cluster if args.cluster != 'None' else 'Full_'+args.overlap
    model = args.model
    if args.maxdistance != '1.0':
        model = model + '_md' + args.maxdistance.replace('.', '')
    if args.keypoint_threshold != '0.015':
        model = model + '_kt' + args.keypoint_threshold.replace('.', '')

    path_colmap = '/home/leon/repositories/colmap/build/src/exe/colmap'  # 'COLMAP.app/Contents/MacOS/colmap'
    if "Full_" in cluster:
        path_images = '/media/discoGordo/dataset_leon/UZ/test_frames' + '/' + video  # it has to be absolute route for colmap
    elif "color_" in video:
        path_images = '/media/discoGordo/C3VD/' + video + '/' + cluster  # '/Users/mjimenez/pythonProject8/assets/alfombra'  # it has to be absolute route for colmap
    else:
        path_images = '/media/discoGordo/dataset_leon/UZ/colmap_benchmark_frames/' + video + '/' + cluster  # '/Users/mjimenez/pythonProject8/assets/alfombra'  # it has to be absolute route for colmap

    if args.reperror != '4':
        path_results = '/media/discoGordo/dataset_leon/reconstructions_MICCAI2023/' + model + '/' + video + '_err' + args.reperror + '/' + cluster
    else:
        path_results = '/media/discoGordo/dataset_leon/reconstructions_MICCAI2023/' + model + '/' + video + '/' + cluster
    database_name = 'database_gm.db'  # name for the databaase we will create. It overwrites an existing one if it already exists with that same name in the same directory
    print("START")
    Path(path_results).mkdir(exist_ok=True, parents=True)
    # if os.path.exists(path_results + '/' + database_name):
    #     db = dataFunctions.COLMAPDatabase.connect(path_results + '/' + database_name)
    #     print(len(db.get_all_cameras()), len(db.get_all_images()), np.mean(db.get_num_all_keypoints()),
    #           np.mean(db.get_num_all_descriptors()), np.mean(db.get_num_all_matches()), len(db.get_all_two_view_geometries()))
    #     db.close()
    # exit(1)
    e0 = time.time()
    if "sift_gm" not in model:
        matches_extraction(path_images, path_results, args.featuretype, args.superpoint, args.matchtype, args.superglue, args.keypoint_threshold, args.overlap)
    # creates the matches with superglue. Possible adjustment of parameters in the function definition
    # exit(1)
    initialize_database(database_name, path_images, path_results, args.matchtype, args.overlap) #creates a database with colmap
    db = dataFunctions.COLMAPDatabase.connect(path_results + '/' + database_name)
    print(len(db.get_all_cameras()), len(db.get_all_images()), len(db.get_all_keypoints()),
          len(db.get_all_descriptors()), len(db.get_all_matches()), len(db.get_all_two_view_geometries()))
    db.close()
    # exit(1)

    if "sift_gm" not in model:
        Save_database(database_name, path_images, path_results, args.reperror, args.matchtype, args.overlap) #Saves the matches from superglue to colmap database
        db = dataFunctions.COLMAPDatabase.connect(path_results + '/' + database_name)
        print(len(db.get_all_cameras()), len(db.get_all_images()), len(db.get_all_keypoints()),
              len(db.get_all_descriptors()), len(db.get_all_matches()), len(db.get_all_two_view_geometries()))
        db.close()

    triangulate(database_name, path_images, path_results, args.reperror, args.matchtype, args.maxdistance, args.overlap, args.minsize)  #Triangulates point and camera positions with colmap
    print("Time COLMAP: " + str(time.time() - e0))
    with open('/media/discoGordo/dataset_leon/reconstructions_MICCAI2023/times_log.txt', 'a+') as f:
        f.write(model + " " + video + " " + cluster + " " + "Time COLMAP: " + str(time.time() - e0) + '\n')
    for res in os.listdir(path_results):
        if res.isdigit():
            T = cm.print_camera_positions(path_results + '/' + res)  # Print results of camera positions