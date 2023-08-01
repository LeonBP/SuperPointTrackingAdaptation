import numpy as np
import os
import cv2


def read_image(path):
    cell = 8
    input_image = cv2.imread(path)
    # print(f"path: {path}, image: {image}")
    # print(f"path: {path}, image: {input_image.shape}")
    y_offset = int((input_image.shape[0] - 256) / 2.)
    x_offset = int((input_image.shape[1] - 256) / 2.)
    input_image_3 = input_image[y_offset:y_offset + 256,
                  x_offset:x_offset + 256]  # cv2.resize(input_image, (self.sizer[1], self.sizer[0]),
    # interpolation=cv2.INTER_AREA)
    H, W = input_image_3.shape[0], input_image_3.shape[1]
    # H = H//cell*cell
    # W = W//cell*cell
    # input_image = input_image[:H,:W,:]
    input_image = cv2.cvtColor(input_image_3, cv2.COLOR_RGB2GRAY)  # input_image[:,:,1]# BGR!!!!!!!!!!!!

    input_image = input_image.astype('float32') / 255.0
    return input_image, input_image_3

def points_to_2D(pnts, H, W):
    labels = np.zeros((H, W),dtype=np.int64)
    pnts = pnts.astype(int)
    labels[pnts[:, 1], pnts[:, 0]] = 1
    return labels


src = "/media/discoGordo/dataset_leon/training_SP/homoAdapt_superpoint_uz/predictions/"
dataset = "/media/discoGordo/dataset_leon/training_SP/256_label/"
dst = "/media/discoGordo/dataset_leon/training_SP/homoAdapt_superpoint_uz_spec/predictions/"

split = "train"

for filename in os.listdir(src+split):
    input_image, input_image_3 = read_image(dataset+filename[:-4]+".png")
    mask = np.where(input_image > 0.7, 1, 0)

    filepath = src+split+"/"+filename
    load = np.load(filepath)
    pts = load['pts']
    points = pts.astype(int)
    new_pts = None
    for p in range(points.shape[0]):
        if mask[points[p, 1], points[p, 0]] == 0:
            if new_pts is None:
                new_pts = np.array([pts[p, :]])
            else:
                new_pts = np.vstack((new_pts, pts[p, :]))
    if new_pts is None:
        new_pts = pts
    np.savez_compressed(dst+split+"/"+filename, pts=new_pts)
