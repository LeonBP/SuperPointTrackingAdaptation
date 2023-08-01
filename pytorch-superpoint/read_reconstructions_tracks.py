import cv2
import numpy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image, ImageDraw, ImageFont
import os
import database_functions as dataFunctions

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

# Which sequence
seq = '00033'
subseq = '13'
suffix = 'trails_sg'
mode = "interactive"#"patches"#"draw"#
at_least = 10 #25
at_most = 300 #150
model = "sift_gm"#"sptracking3_sgh50"#
dataset = "/media/discoGordo/dataset_leon/UZ/colmap_benchmark_frames"
# dataset = "/media/discoGordo/dataset_leon/UZ/training/training_colmap_frames"
gt_src = "/media/discoGordo/dataset_leon/reconstructions_MICCAI2023/sptracking3_disk"
# gt_src = "/media/discoGordo/dataset_leon/reconstructions_MICCAI2023/sptracking3_sgh50"
# gt = "/media/discoGordo/dataset_leon/colmap_MIDL/pseudoGT_original"
# gt = "/media/discoGordo/dataset_leon/colmap_2023/reconstruction_2023_sg"
gt = "/media/discoGordo/dataset_leon/reconstructions_MICCAI2023/"+model
# gt = "/media/discoGordo/dataset_leon/reconstructions_MICCAI2023/sp_sg"
# gt = "/media/discoGordo/dataset_leon/reconstructions_MICCAI2023/sptracking3_disk"
num_images = []
num_points = []
length_tracks = []
num_detected = []
num_matched = []
num_green = []
num_blue = []
std_subseq = []
grid_subseq = []
bins = 16
specs_subseq = []
for seq in os.listdir(gt_src):#['Seq_020']:#["00364"]:#
    if '35_hd' not in seq:
        continue
    else:
        seq = seq+'/10'
    # if '_err' in seq or 'review' in seq:
    #     continue
    # # seq = seq[:-5]
    # if seq == "00033":
    #     new_seq = "Seq_001"
    # elif seq == "00034":
    #     new_seq = "Seq_002"
    # elif seq == "00364":
    #     new_seq = "Seq_095"
    # elif seq == "02001":
    #     new_seq = "Seq_014"
    # elif seq == "02003":
    #     new_seq = "Seq_016"
    # elif seq == "02005":
    #     new_seq = "Seq_017"
    if os.path.isdir(gt_src + "/" + seq):# and ("00033" in seq or "00034" in seq or "00364" in seq or "0200" in seq):
        print(seq)
        if '_256' in dataset:
            if seq == 'Seq_024':
                x_offset = 261
                y_offset = 56
                size = 966
                scale = 0.265
            else:
                x_offset = 233
                y_offset = 28
                size = 1024
                scale = 0.25
        else:
            x_offset = 0
            y_offset = 0
            size = 1080
            scale = 1
        for subseq in os.listdir(gt_src + "/" + seq):#['10']:#["18"]:#
            if ('00033' in seq and '13' in subseq) or '_' in subseq or not subseq.isdigit():# or int(subseq) <=25
                continue
            print(seq,subseq)
            src = dataset+"/"+seq+"/"+subseq
            dst = gt+"/"+seq+"/"+subseq
            if not os.path.exists(dst+"/images.txt"):
                continue
            # Read COLMAP files
            imagesf = open(dst+"/images.txt",'r')
            pointsf = open(dst+"/points3D.txt",'r')
            camerasf = open(dst+"/cameras.txt",'r')
            imagesl = imagesf.readlines()
            pointsl = pointsf.readlines()
            camerasl = camerasf.readlines()

            # Read camera intrinsics
            # print("Reading cameras.txt")
            line = camerasl[-1].split()
            W = int(line[2])
            H = int(line[3])
            fx = float(line[4])
            fy = float(line[5])
            cx = float(line[6])
            cy = float(line[7])
            k1 = float(line[8])
            k2 = float(line[9])
            k3 = float(line[10])
            k4 = float(line[11])

            # Save images with the 2D point coordinates
            print("Reading images.txt")
            image_id = None
            images_names = {}
            images = {}
            images_points = {}
            for i in range(4, len(imagesl)):
                line = imagesl[i].split()
                if i % 2 == 0:
                    image_id = int(line[0])
                    image_name = line[-1]
                    if "colmap_benchmark_frames" not in dataset:# and "_sg" not in gt:
                        images_names[image_id] = image_name[:-7]+subseq.zfill(3)+"_"+image_name[-7:]
                    else:
                        images_names[image_id] = image_name
                    images[image_id] = [float(x) for x in line[1:8]]
                else:
                    images_points[image_id] = [float(x) for x in line]
            # images_ids = sorted(images.keys())
            zipped = zip(images.keys(), images_names.values())
            sorted_zip = sorted(zipped)
            tuples = zip(*sorted_zip)
            images_ids, names = [list(tuple) for tuple in tuples]

            # Save the 3D points
            print("Reading points3D.txt")
            if os.path.exists(dst+"/points_projected.npz"):
                #continue
                loaded = np.load(dst+"/points_projected.npz")
                points2d = loaded["points2d"]
                visible = loaded["visible"]
                names = loaded["names"]
                names = [x[:-12]+x[-8:] for x in names]
                num_images = num_images + [points2d.shape[0]]
                num_points = num_points + [points2d.shape[1]]
                db = dataFunctions.COLMAPDatabase.connect(dst + "/database.db")
                db_ids = db.get_all_image_ids()
                std_img = []
                grid_img = []
                specs_img = []
                for i in range(len(names)):
                    img_name = names[i]
                    # print(img_name)
                    img_id = db.get_image_id_from_name(img_name)[0][0]
                    matched = []
                    for db_id in db_ids:
                        if db_id[0] != img_id:
                            mt = db.get_only_matches(img_id, db_id[0])
                            mt = mt[::2] if img_id < db_id[0] else mt[1::2]
                            matched = matched + list(mt)
                    matches = len(set(matched))
                    # print(img_name, img_id)
                    dst_matches = dst
                    if subseq+"_l2" in os.listdir(gt_src + "/" + seq):
                        dst_matches = dst+"_l2"
                    if "pseudoGT" in gt:
                        num_keypoints = db.get_num_keypoints_id(img_id)[0][0]
                    elif i<len(names)-1:
                        num_keypoints = np.load(dst_matches+"/dump_match_pairs/"+img_name[:-4]+"_"+names[i+1][:-4]+"_matches.npz")["keypoints0"].shape[0]
                    else:
                        num_keypoints = np.load(dst_matches+"/dump_match_pairs/"+names[i-1][:-4]+"_"+img_name[:-4]+"_matches.npz")["keypoints1"].shape[0]
                    num_detected = num_detected + [num_keypoints]
                    num_matched = num_matched + [100*matches/num_keypoints]
                    greens = 0
                    blues = 0
                    point_xs = []
                    point_ys = []
                    points_specs = 0
                    points_all = 0
                    new_image_name = img_name.replace("HCULB_","").replace("procedure_lossy_h264",subseq.zfill(3)).replace(seq,new_seq)
                    image = cv2.imread(src.replace(seq,new_seq) + "/" + new_image_name, cv2.IMREAD_GRAYSCALE)
                    # print(src.replace(seq,new_seq) + "/" + new_image_name)
                    for j in range(visible.shape[1]):
                        if visible[i, j]:
                            greens += 1
                            point_xs += [points2d[i, j, 0]]
                            point_ys += [points2d[i, j, 1]]
                            if 0<=int(point_ys[-1])<1080 and 0<=int(point_xs[-1])<1440 and \
                                    image[int(point_ys[-1]), int(point_xs[-1])] > 180:
                                points_specs += 1
                                print(point_xs[-1], point_ys[-1], new_image_name, image[int(point_ys[-1]), int(point_xs[-1])])
                            elif 0>int(point_ys[-1]) or int(point_ys[-1])>=1080 or 0>int(point_xs[-1]) or int(point_xs[-1])>=1440:
                                print("Out of bounds", point_xs[-1], point_ys[-1], new_image_name)
                            points_all += 1
                        elif i>0 and True in visible[:i, j] and i+1<visible.shape[0] and True in visible[i + 1:, j]:
                            blues += 1
                    num_green = num_green + [100*greens/num_keypoints]
                    num_blue = num_blue + [100*blues/num_keypoints]
                    # print(greens, blues, num_keypoints)
                    # print(np.max(point_xs), np.min(point_xs), np.max(point_ys), np.min(point_ys))
                    std_img += [np.std(point_xs),np.std(point_ys)]
                    hist_p, xedges, yedges, quadmesh = plt.hist2d(point_xs, point_ys, bins=(bins,bins),
                                                                  range=[[205, 205+1080 - 1], [0, 1080 - 1]])
                    grid_img += [len(np.where(hist_p > 0)[0]) / (bins * bins)]
                    specs_img += [points_specs/points_all]
                # exit(1)
                # for j in range(visible.shape[1]):
                #     start = False
                #     visibles = 0
                #     last_green = -1
                #     # print(visible[:,j])
                #     for i in range(visible.shape[0]):
                #         if visible[i,j]:
                #             last_green = i
                #             start = True
                #         if start:
                #             visibles += 1
                #     # print(visibles, visible.shape[0],last_green)
                #     if start:
                #         visibles -= (visible.shape[0] - last_green - 1)
                #     # print(visibles, visible.shape[0], last_green)
                length_tracks = length_tracks + [0]
                db.close()
            else:
                tope = len(pointsl)
                points2d = np.zeros((len(images_ids), tope-3, 3), dtype=float)  # x_image y_image z_3d_from_camera
                visible = np.full((len(images_ids), tope-3), False)
                for i in range(3, len(pointsl)):
                    if i == tope:
                        break
                    line = pointsl[i].split()
                    # point3d_id = int(line[0])
                    # 3D Point
                    X = [float(x) for x in line[1:4]]
                    for ind in range(len(images_ids)):
                        im = images_ids[ind]
                        # Rotation matrix
                        Q = images[im][0:4]
                        R = qvec2rotmat(Q)
                        # Translation vector
                        T = images[im][4:7]
                        Xc = R @ X + T
                        points2d[ind, i-3, 2] = Xc[2]
                        r = np.sqrt(Xc[0] ** 2 + Xc[1] ** 2)
                        phi = np.arctan2(Xc[1], Xc[0])
                        theta = np.arctan2(r, Xc[2])
                        d = theta + k1 * theta ** 3 + k2 * theta ** 5 + k3 * theta ** 7 + k4 * theta ** 9
                        K_c = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                        x_c = np.array([[d * np.cos(phi), d * np.sin(phi), 1]]).T
                        u = K_c @ x_c
                        points2d[ind, i-3, :2] = u[:2, 0]

                    img_index = -1
                    for ind in range(len(line[8:])):
                        if ind % 2 == 0:
                            img_id = int(line[8+ind])
                            img_index = images_ids.index(img_id)
                        else:
                            visible[img_index, i-3] = True

                np.savez_compressed(dst+"/points_projected",
                                    points2d=points2d,visible=visible,names=names)
            continue
            std_subseq += [np.mean(std_img)]
            grid_subseq += [np.mean(grid_img)]
            specs_subseq += [np.mean(specs_img)]
            print("std_subseq", std_subseq[-1], "grid_subseq", grid_subseq[-1], "specs_subseq", specs_subseq[-1])
            # Where to save the tracking of 3D points
            folder_track = "/home/leon/Experiments/Tracking_SP/track" + "_" + seq + "_" + subseq + "_" + suffix
            if not os.path.exists(folder_track):
                os.mkdir(folder_track)

            window = len(images_ids)
            if mode == "draw":
                # Paint the point in images to visualize
                print("Drawing points")

                window = 5
                for id in images_ids:
                    img_name = images_names[id]
                    image_src = cv2.imread(src+"/"+img_name)
                    index = images_ids.index(id)
                    for j in range(points2d.shape[1]):
                        if points2d[index, j, 2] > 0 and np.sum(visible[:,j]) >= at_least and np.sum(visible[:,j]) < at_most and \
                            (visible[index, j] or (True in visible[:index, j] and True in visible[index + 1:, j])):
                            for i in range(max(index-window,0),min(index+window,points2d.shape[0])):  #points2d.shape[0]):
                                # cv2.rectangle(images_src[img_name], (int(u[0]) - radius_rect, int(u[1]) - radius_rect),
                                #               (int(u[0]) + radius_rect, int(u[1]) + radius_rect), (255, 0, 0), 1)
                                color = (0, 255, 0) if visible[i, j] else (255, 0, 0)
                                # cv2.circle(image_src, (int(points2d[i, j, 0]), int(points2d[i, j, 1])), 1, color, lineType=1)
                                x = (points2d[i, j, 0] - x_offset) * scale
                                y = (points2d[i, j, 1] - y_offset) * scale
                                if i < points2d.shape[0]-1:
                                    x_plusone = (points2d[i+1, j, 0] - x_offset) * scale
                                    y_plusone = (points2d[i+1, j, 1] - y_offset) * scale
                                    cv2.line(image_src,
                                             (int(x), int(y)),
                                             (int(x_plusone), int(y_plusone)),
                                             color, thickness=1, lineType=1)
                                else:
                                    cv2.circle(image_src,
                                               (int(x), int(y)),
                                               1, color, lineType=1)
                                if i == index:
                                    cv2.drawMarker(image_src,
                                                   (int(x), int(y)),
                                                   color, markerSize=20, markerType=cv2.MARKER_DIAMOND, thickness=3)

                    # for i in range(int(len(images_points[id])/3)):
                    #     color = (0, 255, 0)
                    #     cv2.drawMarker(image_src, (int(images_points[id][3*i]), int(images_points[id][3*i+1])), color, markerSize=20)

                    cv2.imwrite(folder_track + "/" + img_name, image_src)
            elif mode == "patches":
                # Create mosaics with the patches surrounding the points
                print("Extracting patches")

                window = 10
                patch_half_size = 50
                patch_size = patch_half_size * 2 + 1
                row_height = patch_size + patch_half_size
                images_src = {}
                for id in images_ids:
                    img_name = images_names[id]
                    if img_name not in images_src.keys():
                        images_src[img_name] = Image.open(src+"/"+img_name)
                for j in range(points2d.shape[1]):
                    vis_sum = np.sum(visible[:, j])
                    vis_sum_not = points2d.shape[0] - vis_sum
                    if vis_sum >= at_least and vis_sum < at_most:
                        # vis = None
                        # changes = []
                        # for i in range(points2d.shape[0]): #max(index - window, 0),min(index + window, points2d.shape[0])):  #
                        #     if vis is None:
                        #         vis = visible[i,j]
                        #     elif vis != visible[i,j]:
                        #         changes = changes + [i]
                        #         vis = visible[i,j]
                        # print(changes)
                        im_vis = Image.new('RGB', (patch_size * window * 2, row_height * int(np.ceil(vis_sum/(window*2)))), color=(255, 255, 255))
                        im_vis_not = Image.new('RGB', (patch_size * window * 2, row_height * int(np.ceil(vis_sum_not/(window*2)))), color=(255, 255, 255))
                        draw_vis = ImageDraw.Draw(im_vis)
                        draw_vis_not = ImageDraw.Draw(im_vis_not)
                        # some = False
                        color1 = (0, 255, 0)  # green
                        color2 = (0, 0, 255)  # blue
                        print(vis_sum, vis_sum_not)
                        vis_index = -1
                        vis_index_not = -1
                        for i in range(points2d.shape[0]):
                            if visible[i, j]:
                                color = color1
                                draw = draw_vis
                                im = im_vis
                                vis_index = vis_index + 1
                                index = vis_index
                            else:
                                color = color2
                                draw = draw_vis_not
                                im = im_vis_not
                                vis_index_not = vis_index_not + 1
                                index = vis_index_not
                            img_name = images_names[images_ids[i]]
                            point_x = int(points2d[i, j, 0])
                            point_y = int(points2d[i, j, 1])
                            img = images_src[img_name]
                            left = max(0, point_x - patch_half_size)
                            top = max(0, point_y - patch_half_size)
                            right = min(img.size[0], point_x + patch_half_size + 1)
                            bottom = min(img.size[1], point_y + patch_half_size + 1)
                            patch = img.crop((left, top, right, bottom))
                            paste_left = (index % (window*2)) * patch_size
                            paste_top = patch_half_size + row_height * (index // (window*2))
                            draw.rectangle((paste_left,
                                            paste_top - int(patch_half_size/4),
                                            paste_left + patch_size,
                                            paste_top),
                                           fill=color, outline=color)
                            offset_x = left - (point_x - patch_half_size)
                            offset_y = top - (point_y - patch_half_size)
                            im.paste(patch, (paste_left + offset_x,
                                             paste_top + offset_y))
                            #print(visible[i, j], left, top, right, bottom, point_x, point_y)
                            # draw.rectangle((window*patch_size,
                            #                 int(patch_half_size*3/4) + row_height * row,
                            #                 im.size[0],
                            #                 row_height * row+row_height-1),
                            #                fill=color1, outline=color1)
                            # draw.rectangle((0,
                            #                 int(patch_half_size*3/4) + row_height * row,
                            #                 window*patch_size,
                            #                 row_height * row+row_height-1),
                            #                fill=color2, outline=color2)
                            # for col in range(max(0, ch - window), min(points2d.shape[0], ch + window)):
                            #     # print(ch,col,visible[ch,j], visible[col,j])
                            #     if ch - window < 0 or ch + window >= points2d.shape[0] or \
                            #         not (np.array_equal(visible[ch - window:ch,j], np.repeat([False],window)) or
                            #              np.array_equal(visible[ch - window:ch,j], np.repeat([True],window))) or \
                            #         not (np.array_equal(visible[ch:ch + window,j], np.repeat([True],window)) or
                            #              np.array_equal(visible[ch:ch + window,j], np.repeat([False],window))):
                            #         break
                            #     # print(visible[ch - window:ch + window,j])
                            #     some = True
                            #     # if (col < ch and visible[col,j] != visible[ch,j]) or \
                            #     #         (col >= ch and visible[col,j] == visible[ch,j]):
                            #     img_name = images_names[images_ids[col]]
                            #     point_x = int(points2d[col, j, 0])
                            #     point_y = int(points2d[col, j, 1])
                            #     img = images_src[img_name]
                            #     if point_y - patch_half_size >= 0 and \
                            #        point_y + patch_half_size + 1 < img.size[1] and \
                            #        point_x - patch_half_size >= 0 and \
                            #        point_x + patch_half_size + 1 < img.size[0]:
                            #         patch = img.crop((point_x-patch_half_size,
                            #                          point_y-patch_half_size,
                            #                          point_x+patch_half_size+1,
                            #                          point_y+patch_half_size+1))
                            #     else:
                            #         patch = img.crop((max(0,point_x - patch_half_size),
                            #                           max(0,point_y - patch_half_size),
                            #                           min(img.size[0], point_x + patch_half_size + 1),
                            #                           min(img.size[1], point_y + patch_half_size + 1)))
                            #     # print(point_x, point_y, ch, col, window, row, patch.size)
                            #     im.paste(patch, ((col - ch + window) * patch_size, patch_half_size + row_height * row))

                        # if some:
                        im_vis.save('/home/leon/Experiments/Tracking_SP/points_patches/point_' + str(j).zfill(6) + '_vis' + str(vis_sum).zfill(3) + '.png')
                        im_vis_not.save('/home/leon/Experiments/Tracking_SP/points_patches/point_' + str(j).zfill(6) + '_visnot' + str(vis_sum_not).zfill(3) + '.png')
                        # if j>1:
                        #     exit(1)
                    #(int(points2d[i, j, 0]), int(points2d[i, j, 1]))
            else:
                # Paint the points in images to visualize
                print("Visualizing")

                point_x = None
                point_y = None
                original = False

                def click_point(event, x, y, flags, param):
                    global point_x,point_y,original
                    # if the left mouse button was clicked, record the starting
                    # (x, y) coordinates and indicate that cropping is being
                    # performed
                    if event == cv2.EVENT_LBUTTONDOWN:
                        if point is None:
                            point_x = x
                            point_y = y
                            original = True

                key = None
                image_last = None
                point = None
                index = 0
                text = 'Point tracking'
                cv2.namedWindow(text)
                cv2.setMouseCallback(text, click_point)
                while True:
                    if key is not None:
                        if key == 97 and index > 0:  # a to go left (previous frame)
                            index = index - 1
                        elif key == 100 and index < len(images_ids)-1:  # d to go right (next frame)
                            index = index + 1
                        elif key == 113:  # q key to get out of single point view
                            original = False
                            point = None
                        elif key == 27:    # Esc key to stop
                            break

                    if not original:
                        id = images_ids[index]
                        img_name = images_names[id]
                        image_src = cv2.imread(folder_track+"/"+img_name)
                        text_new = 'Frame: '+img_name+' Index: '+str(index)
                        cv2.setWindowTitle(text, text_new)
                    else:
                        id = images_ids[index]
                        img_name = images_names[id]
                        image_src = cv2.imread(src+"/"+img_name)

                        if point_x is not None and point_y is not None:
                            px = point_x
                            py = point_y
                            point_x = None
                            point_y = None
                            pobj = np.array([px,py])
                            dist = 999999
                            for j in range(points2d.shape[1]):
                                if points2d[index, j, 2] > 0 and np.sum(visible[:,j]) >= at_least and np.sum(visible[:,j]) < at_most:
                                    d = np.linalg.norm(pobj-points2d[index, j, :2])
                                    if d < dist:
                                        dist = d
                                        point = j
                        if point is not None:
                            #occurrences of True in visible[:,point]
                            occurrences = np.where(visible[:,point])[0]
                            for i in range(max(occurrences[0],0),min(occurrences[-1]+1,points2d.shape[0])):  #points2d.shape[0]):
                                # cv2.rectangle(images_src[img_name], (int(u[0]) - radius_rect, int(u[1]) - radius_rect),
                                #               (int(u[0]) + radius_rect, int(u[1]) + radius_rect), (255, 0, 0), 1)
                                color = (0, 255, 0) if visible[i, point] else (255, 0, 0)
                                # cv2.circle(image_src, (int(points2d[i, point, 0]), int(points2d[i, point, 1])), 1, color, lineType=1)
                                if i < points2d.shape[0]-1:
                                    cv2.line(image_src,
                                             (int(points2d[i, point, 0]), int(points2d[i, point, 1])),
                                             (int(points2d[i+1, point, 0]), int(points2d[i+1, point, 1])),
                                             color, thickness=3, lineType=1)
                                else:
                                    cv2.circle(image_src,
                                               (int(points2d[i, point, 0]), int(points2d[i, point, 1])),
                                               1, color, lineType=1)
                                if i == index:
                                    cv2.drawMarker(image_src,
                                                   (int(points2d[i, point, 0]), int(points2d[i, point, 1])),
                                                   color, markerSize=40, markerType=cv2.MARKER_DIAMOND, thickness=5)
                        text_new = 'Frame: '+img_name+' Index: '+str(index)+' Point: '+str(point).zfill(6)
                        cv2.setWindowTitle(text, text_new)
                    image_last = image_src
                    cv2.imshow(text, image_last)
                    key = cv2.waitKey(500)
                    print(key)
                cv2.destroyAllWindows()
print("std:",np.mean(std_subseq))
print("grid:",np.mean(grid_subseq))
print("specs:",np.mean(specs_subseq))
exit(1)

# print("num_images", np.mean(num_images),"num_points", np.mean(num_points), "length_tracks", np.mean(length_tracks), "detected", np.mean(num_detected), "matched", np.mean(num_matched), "greens", np.mean(num_green), "blues", np.mean(num_blue))
# print("num_images", np.std(num_images),"num_points", np.std(num_points), "length_tracks", np.std(length_tracks), "detected", np.std(num_detected), "matched", np.std(num_matched), "greens", np.std(num_green), "blues", np.std(num_blue))
# print("num_images", np.median(num_images),"num_points", np.median(num_points), "length_tracks", np.median(length_tracks), "detected", np.median(num_detected), "matched", np.median(num_matched), "greens", np.median(num_green), "blues", np.median(num_blue))
# print("num_images", min(num_images),"num_points", min(num_points), "length_tracks", min(length_tracks), "detected", min(num_detected), "matched", min(num_matched), "greens", min(num_green), "blues", min(num_blue))
# print("num_images", max(num_images),"num_points", max(num_points), "length_tracks", max(length_tracks), "detected", max(num_detected), "matched", max(num_matched), "greens", max(num_green), "blues", max(num_blue))
print("num_images", np.mean(num_images), "detected", np.mean(num_detected), "matched", np.mean(num_matched), "greens", np.mean(num_green), "blues", np.mean(num_blue))
print("num_images", np.std(num_images), "detected", np.std(num_detected), "matched", np.std(num_matched), "greens", np.std(num_green), "blues", np.std(num_blue))
print("num_images", np.median(num_images), "detected", np.median(num_detected), "matched", np.median(num_matched), "greens", np.median(num_green), "blues", np.median(num_blue))
print("num_images", min(num_images), "detected", min(num_detected), "matched", min(num_matched), "greens", min(num_green), "blues", min(num_blue))
print("num_images", max(num_images), "detected", max(num_detected), "matched", max(num_matched), "greens", max(num_green), "blues", max(num_blue))
print(len(num_images),len(num_points), len(length_tracks))
n,bins,patches=plt.hist(num_detected,bins=100, range=(0,15000), cumulative=False, log=False)
plt.show()
plt.figure()
n,bins,patches=plt.hist(num_matched,bins=100, range=(0,15000), cumulative=False, log=False)
plt.show()
plt.figure()
n,bins,patches=plt.hist(num_green,bins=100, range=(0,100), cumulative=True, log=False)
np.savetxt('/home/leon/Experiments/Tracking_SP/histograms_detgreenblue/green_percdet_'+model+'.txt', n, delimiter=',')
plt.ylim(0,838)
plt.show()
plt.figure()
n,bins,patches=plt.hist(num_blue,bins=100, range=(0,500), cumulative=False, log=False)
plt.show()
# plt.figure()
# #plt.yticks(np.array([0,100000,1000000,10000000,100000000]))
# n,bins,patches=plt.hist(length_tracks,bins=500, cumulative=True, log=True)
# plt.show()