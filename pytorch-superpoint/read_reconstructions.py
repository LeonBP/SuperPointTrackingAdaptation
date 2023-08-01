import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
suffix = 'pro'
dataset = "/media/discoGordo/dataset_leon/UZ/colmap_benchmark_frames"
length = []
missing = []
present = []
for seq in ["00033"]:#os.listdir(dataset):
    if os.path.isdir(dataset + "/" + seq):# and ("00033" in seq or "00034" in seq or "00364" in seq or "0200" in seq):
        print(seq)
        for subseq in ["13"]:#os.listdir(dataset + "/" + seq):
            # Read COLMAP files
            imagesf = open("/media/discoGordo/dataset_leon/colmap_MIDL/pseudoGT/"+seq+"/"+subseq+"/images.txt",'r')
            pointsf = open("/media/discoGordo/dataset_leon/colmap_MIDL/pseudoGT/"+seq+"/"+subseq+"/points3D.txt",'r')
            camerasf = open("/media/discoGordo/dataset_leon/colmap_MIDL/pseudoGT/"+seq+"/"+subseq+"/cameras.txt",'r')
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
            images = {}
            image_id = None
            images_points = {}
            for i in range(4, len(imagesl)):
                line = imagesl[i].split()
                if i % 2 == 0:
                    image_id = line[0]
                    images[image_id] = line
                else:
                    images_points[image_id] = line

            # Save the 3D points with their appearances and choose some of them
            # print("Reading points3D.txt")
            points3d = None
            tracks = []
            mas = 8+2*120  # 8 info + 2 times ('img' 'point') * length_of_the_track
            for i in range(3, len(pointsl)):
                line = pointsl[i].split()
                if points3d is None:
                    points3d = [line]
                else:
                    points3d = points3d + [line]
                length = length + [(len(line)-8)/2]
                # if True or len(line) > mas:
                #     tracks = tracks + [i-3]

            # # Select the images that contain the chosen 3D points
            # print("Finding points to draw")
            # ts = []
            # image_num = []
            # point_num = []
            # for track in tracks:
            #     for i in range(8, len(points3d[track])):
            #         if i % 2 == 0:
            #             ts = ts + [track]
            #             image_num = image_num + [points3d[track][i]]
            #         else:
            #             point_num = point_num + [points3d[track][i]]

            # Where to save the tracking of 3D points
            # folder_track = "/home/leon/Experiments/Tracking_SP/track" + "_" + seq + "_" + subseq + "_" + suffix
            # if not os.path.exists(folder_track):
            #     os.mkdir(folder_track)

            # Paint the point in images to visualize
            # print("Drawing points")
            src = "/media/discoGordo/dataset_leon/UZ/colmap_benchmark_frames/"+seq+"/"+subseq
            images_src = {}

            for i in images.keys():
                img_name = images[i][-1]
                if img_name not in images_src.keys():
                    images_src[img_name] = cv2.imread(src+"/"+img_name)
                # Rotation matrix
                Q = np.zeros(4)
                Q[0] = float(images[i][1])
                Q[1] = float(images[i][2])
                Q[2] = float(images[i][3])
                Q[3] = float(images[i][4])
                # print(Q)
                R = qvec2rotmat(Q)
                # print(R)
                # Translation vector
                T = np.zeros(3)
                T[0] = float(images[i][5])
                T[1] = float(images[i][6])
                T[2] = float(images[i][7])

                # print("Drawing missing points.")
                miss = 0
                for j in range(len(points3d)):
                    # if j % 5000 == 0:
                    #     print("%2.1f" % ((j/len(points3d))*100), "%")
                    # 3D Point
                    X = np.zeros(3)
                    X[0] = float(points3d[j][1])
                    X[1] = float(points3d[j][2])
                    X[2] = float(points3d[j][3])
                    # print(X)
                    # The coordinate vector of P in the camera reference frame
                    Xc = R @ X + T
                    if Xc[2] < 0:
                        continue
                    # print(Xc)
                    # exit(1)
                    r = np.sqrt(Xc[0]**2 + Xc[1]**2)
                    phi = np.arctan2(Xc[1], Xc[0])
                    theta = np.arctan2(r, Xc[2])
                    d = theta + k1 * theta ** 3 + k2 * theta ** 5 + k3 * theta ** 7 + k4 * theta ** 9
                    K_c = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                    x_c = np.array([[d * np.cos(phi), d * np.sin(phi), 1]]).T
                    u = K_c @ x_c
                    # cv2.rectangle(images_src[img_name], (int(u[0]) - radius_rect, int(u[1]) - radius_rect),
                    #               (int(u[0]) + radius_rect, int(u[1]) + radius_rect), (255, 0, 0), 1)
                    cv2.circle(images_src[img_name], (int(u[0]), int(u[1])), 1, (255, 0, 0), lineType=1)
                    miss = miss + 1
                missing = missing + [miss]

                # print("Drawing present points.")
                pres = 0
                for j in range(int(len(images_points[i])/3)):
                    # if j % 5000 == 0:
                    #     print("%2.1f" % ((j/len(images_points[i]))*100),"%")
                    pointx = int(float(images_points[i][j * 3]))
                    pointy = int(float(images_points[i][j * 3 + 1]))
                    # cv2.circle(images_src[img_name], (pointx, pointy), 1, (255, 0, 0), lineType=16)
                    radius_rect = 10
                    cv2.rectangle(images_src[img_name],
                                  (pointx-radius_rect, pointy-radius_rect),
                                  (pointx+radius_rect, pointy+radius_rect), (0, 255, 0), 1)
                    pres = pres + 1
                present = present + [pres]

n,bins,patches=plt.hist(length,bins=250,range=(0,250), cumulative=False, log=True)
plt.show()
plt.figure()
n,bins,patches=plt.hist(missing,bins=1000, cumulative=False, log=True)
plt.show()
plt.figure()
n,bins,patches=plt.hist(present,bins=1000, cumulative=False, log=True)
plt.show()
exit(1)

# print("Saving images")
# for img_name in images_src.keys():
#     cv2.imwrite(folder_track + "/" + img_name, images_src[img_name])

'''fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111, projection='3d')
for p in points3d:
    ax.scatter(float(p[1]), float(p[2]), float(p[3]), marker='.', c=(float(p[4])/255, float(p[5])/255, float(p[6])/255))

plt.show()'''
