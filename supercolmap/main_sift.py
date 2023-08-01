import glob, os
import subprocess
import camera_position as cm
import numpy as np
import argparse
from pathlib import Path

def matches_superglue(path_images, path_results):
    f = open(path_results+'/images_to_match.txt', "w+")
    images_name = {}
    unordered = [];
    for file in os.listdir(path_images):
        if file.endswith(".png") or file.endswith(".jpg"):
            if file.endswith(".png"):
                termination = '.png'
            else:
                if file.endswith(".jpg"):
                    termination = '.jpg'
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
            '--input_pairs', path_results+'/images_to_match.txt',
            '--input_dir', path_images,
            '--output_dir', path_results+'/dump_match_pairs/',
            '--max_keypoints', '10000',
            '--resize', '-1',
            '--viz'])

def generate_pseudoGT_EndoMapper(path_colmap,path_images,path_results,database_name,error):
    videos = [x for x in os.listdir(path_images) if os.path.isdir(path_images+"/"+x)]
    for v, s in [('Seq_001', '15'), ('Seq_002', '38'), ('Seq_014', '39'), ('Seq_016', '86'),
                 ('Seq_017', '28'), ('Seq_095', '14'), ('Seq_095', '18')]:#videos:
        #if "0200" not in v:
        #    continue
        print("Video "+v)
        video_results = path_results+"/"+v+'_err'+error
        if not os.path.exists(video_results):
            os.mkdir(video_results)
        video_path = path_images+"/"+v
        # sequences = [x for x in os.listdir(video_path) if os.path.isdir(video_path+"/"+x)]
        # for s in ['18']:#sequences:
        print("Sequence "+s)
        seq_results = video_results+"/"+s
        if not os.path.exists(seq_results):
            os.mkdir(seq_results)
        seq_path = video_path+"/"+s

        #matches_superglue(seq_path, seq_results)
        subprocess.run([path_colmap, 'feature_extractor',
                        '--database_path', seq_results + '/' + database_name,
                        '--image_path', seq_path,
                        '--ImageReader.camera_model', 'OPENCV_FISHEYE',
                        '--ImageReader.single_camera', '1',
                        '--ImageReader.camera_params',
                        '733.1061; 735.719; 739.2826; 539.6911; -0.1205372; -0.01179983; 0.00269742; -0.0001362505',
                        '--SiftExtraction.use_gpu', 'true',
                        '--SiftExtraction.gpu_index', '0'])
        subprocess.run([path_colmap, 'exhaustive_matcher',
                        '--database_path', seq_results + '/' + database_name,
                        '--SiftMatching.guided_matching', '1'])
        #Save_database(database_name, seq_path, seq_results)
        subprocess.run([path_colmap, 'exhaustive_matcher', '--database_path', seq_results+'/'+database_name])
        subprocess.run([path_colmap, 'mapper',
                        '--database_path', seq_results + '/' + database_name,
                        '--image_path', seq_path,
                        '--output_path', seq_results,
                        '--Mapper.init_min_tri_angle', '8',
                        '--Mapper.ba_refine_focal_length', '0',
                        '--Mapper.ba_refine_extra_params', '0',
                        '--Mapper.min_model_size', '50',
                        # '--Mapper.init_max_error', error,
                        # '--Mapper.tri_merge_max_reproj_error', error,
                        # '--Mapper.tri_complete_max_reproj_error', error,
                        '--Mapper.filter_max_reproj_error', error])
        if os.path.exists(seq_results + '/0'):
            subprocess.run([path_colmap, 'model_converter',
                            '--input_path', seq_results + '/0',
                            '--output_path', seq_results,
                            '--output_type', 'TXT'])

def generate_pseudoGT_VrCaps(path_colmap,path_images,path_results,database_name,overlap,error, min_size='25'):

    #matches_superglue(seq_path, seq_results)
    subprocess.run([path_colmap, 'feature_extractor',
                    '--database_path', path_results + '/' + database_name,
                    '--image_path', path_images,
                    '--ImageReader.camera_model', 'OPENCV_FISHEYE',
                    '--ImageReader.single_camera', '1',
                    '--ImageReader.camera_params',
                    '733.1061; 735.719; 739.2826; 539.6911; -0.1205372; -0.01179983; 0.00269742; -0.0001362505',
                    '--SiftExtraction.use_gpu', 'true',
                    '--SiftExtraction.gpu_index', '0'])
                    # '--ImageReader.camera_model', 'PINHOLE',
                    # '--ImageReader.single_camera', '1',
                    # '--ImageReader.camera_params',
                    # '1066.6667; 1066.6667; 960.0; 540.0',
                    # '--SiftExtraction.use_gpu', 'true',
                    # '--SiftExtraction.gpu_index', '0'])
    subprocess.run([path_colmap, 'sequential_matcher',
                    '--database_path', path_results + '/' + database_name,
                    '--SiftMatching.guided_matching', '1',
                    '--SequentialMatching.overlap', overlap,
                    '--SequentialMatching.quadratic_overlap', '1'])
    #Save_database(database_name, seq_path, seq_results)
    # subprocess.run([path_colmap, 'exhaustive_matcher', '--database_path', path_results+'/'+database_name])
    subprocess.run([path_colmap, 'mapper',
                    '--database_path', path_results + '/' + database_name,
                    '--image_path', path_images,
                    '--output_path', path_results,
                    '--Mapper.init_min_tri_angle', '8',
                    '--Mapper.ba_refine_focal_length', '0',
                    '--Mapper.ba_refine_extra_params', '0',
                    '--Mapper.min_model_size', min_size,
                    # '--Mapper.init_max_error', error,
                    # '--Mapper.tri_merge_max_reproj_error', error,
                    # '--Mapper.tri_complete_max_reproj_error', error,
                    '--Mapper.filter_max_reproj_error', error])
    if os.path.exists(path_results + '/0'):
        subprocess.run([path_colmap, 'model_converter',
                        '--input_path', path_results + '/0',
                        '--output_path', path_results,
                        '--output_type', 'TXT'])



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('video', type=str, default='vr_caps')
    parser.add_argument('cluster', type=str, default='images_fixed')
    parser.add_argument('model', type=str, default='sift_gm')
    parser.add_argument('--reperror', type=str, default='4')
    parser.add_argument('--overlap', type=str, default='10')
    args = parser.parse_args()

    video = args.video
    cluster = args.cluster
    model = args.model

    path_colmap = '/home/leon/repositories/colmap/build/src/exe/colmap'#'COLMAP.app/Contents/MacOS/colmap'
    # path_images = '/media/discoGordo/dataset_leon/UZ/colmap_benchmark_frames'# it has to be absolute route for colmap
    path_images = '/media/discoGordo/dataset_leon/UZ/test_frames' + '/' + video# it has to be absolute route for colmap
    # path_images = '/media/discoGordo/' + video + '/' + cluster # it has to be absolute route for colmap
    if args.reperror != '4':
        path_results = '/media/discoGordo/dataset_leon/reconstructions_MICCAI2023/' + model + '/' + video + '_err' + args.reperror #+ '/' + cluster
    else:
        path_results = '/media/discoGordo/dataset_leon/reconstructions_MICCAI2023/' + model + '/' + video #+ '/' + cluster #'/Users/mjimenez/pythonProject8/results'
    database_name = 'database.db'  #name for the databaase we will create. It overwrites an existing one if it already exists with that same name in the same directory
    Path(path_results).mkdir(exist_ok=True, parents=True)
    print("Start")
    generate_pseudoGT_VrCaps(path_colmap,path_images,path_results,database_name, args.overlap, args.reperror)

    T = cm.print_camera_positions(path_results)  #Print results of camera positiosn
