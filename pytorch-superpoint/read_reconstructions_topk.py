import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image, ImageDraw, ImageFont
import os


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
mode = "draw"#"interactive"#"patches"#
at_least = 25
at_most = 150
dataset = "/media/discoGordo/dataset_leon/UZ/colmap_benchmark_frames"
# dataset = "/media/discoGordo/dataset_leon/UZ/training/training_colmap_frames"
# gt = "/media/discoGordo/dataset_leon/colmap_2023/reconstruction_2023_sg"
gt_src = "/media/discoGordo/dataset_leon/reconstructions_MICCAI2023/sptracking3_disk"
# gt_src = "/media/discoGordo/dataset_leon/reconstructions_MICCAI2023/sptracking3_sgh50"
# gt = "/media/discoGordo/dataset_leon/colmap_MIDL/pseudoGT"
# gt = "/media/discoGordo/dataset_leon/reconstructions_MICCAI2023/sp_sg"
gt = "/media/discoGordo/dataset_leon/reconstructions_MICCAI2023/sptracking3_disk"
# gt = "/media/discoGordo/dataset_leon/reconstructions_MICCAI2023/sptracking3_sgh50"
# gt = "/media/discoGordo/dataset_leon/reconstructions_MICCAI2023/sift_gm"
num_images = []
num_points = []
length_tracks = []
errors_list = []
for seq in os.listdir(gt_src):#['Seq_020']:#["00033"]:#
    if '_err' in seq:
        continue
    # seq = seq[:-5]
    if os.path.isdir(gt + "/" + seq):# and ("00033" in seq or "00034" in seq or "00364" in seq or "0200" in seq):
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
        for subseq in os.listdir(gt_src + "/" + seq):#['10']:#["13"]:#
            if '00033' in seq and '13' in subseq:
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
            # print("Reading images.txt")
            image_id = None
            images_names = {}
            images = {}
            images_points = {}
            for i in range(4, len(imagesl)):
                line = imagesl[i].split()
                if i % 2 == 0:
                    image_id = int(line[0])
                    image_name = line[-1]
                    if "colmap_benchmark_frames" not in dataset and "_sg" not in gt:
                        images_names[image_id] = image_name[:-7]+subseq.zfill(3)+image_name[-8:]
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
            # print("Reading points3D.txt")
            if False and os.path.exists(dst+"/points_projected.npz"):
                continue
                loaded = np.load(dst+"/points_projected.npz")
                points2d = loaded["points2d"]
                visible = loaded["visible"]
                num_images = num_images + [points2d.shape[0]]
                num_points = num_points + [points2d.shape[1]]
                for j in range(visible.shape[1]):
                    start = False
                    visibles = 0
                    last_green = -1
                    # print(visible[:,j])
                    for i in range(visible.shape[0]):
                        if visible[i,j]:
                            last_green = i
                            start = True
                        if start:
                            visibles += 1
                    # print(visibles, visible.shape[0],last_green)
                    visibles -= (visible.shape[0] - last_green - 1)
                    # print(visibles, visible.shape[0], last_green)
                    length_tracks = length_tracks + [visibles]
            else:
                # print("Computing points_projected.npz")
                tope = len(pointsl)
                points2d = np.zeros((len(images_ids), tope-3, 3), dtype=float)  # x_image y_image z_3d_from_camera
                visible = np.full((len(images_ids), tope-3), False)
                errors = []
                tracks = []
                for i in range(3, len(pointsl)):
                    if i == tope:
                        break
                    line = pointsl[i].split()
                    errors.append(float(line[7]))
                    tracks.append(len(line[8:])/2)
                    length_tracks.append(len(line[8:])/2)
                    errors_list.append(float(line[7]))
                    continue
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
                #errors =sorted(errors)
                zipped = zip(errors, tracks)
                sorted_zip = sorted(zipped)
                tuples = zip(*sorted_zip)
                errors, tracks = [list(tuple) for tuple in tuples]
                # if len(errors) > 10000:
                # errors_list = errors_list + [np.mean(errors[:10000])]
                lessthan2 = np.where(np.array(errors) < 2)[0]
                print(len(lessthan2),np.mean(errors), np.mean(tracks), np.mean(np.array(errors)[lessthan2]), np.mean(np.array(tracks)[lessthan2]))
                # f = open(dst+"/errors.txt", "w")
                # for e in errors:
                #     f.write(str(e)+"\n")
                # f.close()
                # np.savez_compressed(dst+"/points_projected",
                #                     points2d=points2d,visible=visible,names=names)
# print(np.mean(errors_list), np.std(errors_list))
lessthan2 = np.where(np.array(length_tracks) > 50)[0]
print(len(errors_list), np.mean(np.array(errors_list)[lessthan2]))
fig, ax = plt.subplots()
# plt.scatter(errors_list, length_tracks, s=0.1)
bins = 50
max_error = 2
min_length = 10
max_length = 150
heatmap, xedges, yedges = np.histogram2d(errors_list, length_tracks,
                                         bins=(bins,bins),
                                         range=[[0, max_error], [min_length, max_length]],
                                         density=True)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
plt.imshow(heatmap.T, origin='lower')
step = 5
ax.set_xticks(np.arange(0, bins, step))
ax.set_xticklabels(np.arange(start=0, stop=max_error, step=max_error*step/bins).round(1))
ax.set_yticks(np.arange(0, bins, step))
ax.set_yticklabels(np.arange(start=min_length, stop=max_length, step=(max_length-min_length)*step/bins).round(2))
plt.colorbar()
# n,bins,patches=plt.hist(np.array(errors_list)[lessthan2],bins=1000, cumulative=True, log=True)
plt.show()

'''
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
                        if points2d[index, j, 2] > 0 and np.sum(visible[:,j]) >= at_least and np.sum(visible[:,j]) < at_most:
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
                            for i in range(max(index-window,0),min(index+window,points2d.shape[0])):  #points2d.shape[0]):
                                # cv2.rectangle(images_src[img_name], (int(u[0]) - radius_rect, int(u[1]) - radius_rect),
                                #               (int(u[0]) + radius_rect, int(u[1]) + radius_rect), (255, 0, 0), 1)
                                color = (0, 255, 0) if visible[i, point] else (255, 0, 0)
                                # cv2.circle(image_src, (int(points2d[i, point, 0]), int(points2d[i, point, 1])), 1, color, lineType=1)
                                if i < points2d.shape[0]-1:
                                    cv2.line(image_src,
                                             (int(points2d[i, point, 0]), int(points2d[i, point, 1])),
                                             (int(points2d[i+1, point, 0]), int(points2d[i+1, point, 1])),
                                             color, thickness=1, lineType=1)
                                else:
                                    cv2.circle(image_src,
                                               (int(points2d[i, point, 0]), int(points2d[i, point, 1])),
                                               1, color, lineType=1)
                                if i == index:
                                    cv2.drawMarker(image_src,
                                                   (int(points2d[i, point, 0]), int(points2d[i, point, 1])),
                                                   color, markerSize=20, markerType=cv2.MARKER_DIAMOND, thickness=3)
                        text_new = 'Frame: '+img_name+' Index: '+str(index)+' Point: '+str(point).zfill(6)
                        cv2.setWindowTitle(text, text_new)
                    image_last = image_src
                    cv2.imshow(text, image_last)
                    key = cv2.waitKey(500)
                    print(key)
                cv2.destroyAllWindows()

exit(1)

print("num_images",np.median(num_images),"num_points", np.median(num_points), "length_tracks", np.median(length_tracks))
print("num_images",min(num_images),"num_points", min(num_points), "length_tracks", min(length_tracks))
print("num_images",max(num_images),"num_points", max(num_points), "length_tracks", max(length_tracks))
print(len(num_images),len(num_points), len(length_tracks))
n,bins,patches=plt.hist(num_images,bins=len(num_images),range=(0,250), cumulative=False, log=False)
plt.show()
plt.figure()
n,bins,patches=plt.hist(num_points,bins=len(num_points), cumulative=False, log=False)
plt.show()
plt.figure()
n,bins,patches=plt.hist(length_tracks,bins=1000, cumulative=False, log=False)
plt.show()
plt.figure()
#plt.yticks(np.array([0,100000,1000000,10000000,100000000]))
n,bins,patches=plt.hist(length_tracks,bins=500, cumulative=True, log=True)
plt.show()'''