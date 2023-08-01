import cv2
import os
import numpy as np

seq = '00033_13'
type = ""
debug = "_debug"
folder1 = "/home/leon/Experiments/Tracking_SP/SuperPoint_Tracking_v1/UZ_"+seq+type+debug
folder2 = "/home/leon/Experiments/Tracking_SP/SuperPoint_Tracking_v2/UZ_"+seq+type+debug

video_name = "/home/leon/Experiments/Tracking_SP/video_tracking_v1vs2"+type+debug+".avi"

images1 = [img for img in os.listdir(folder1) if img.endswith(".png")]
images1 = sorted(images1)
frame1 = cv2.imread(os.path.join(folder1, images1[0]))
images2 = [img for img in os.listdir(folder2) if img.endswith(".png")]
images2 = sorted(images2)
frame2 = cv2.imread(os.path.join(folder2, images2[0]))
vis = np.concatenate((frame1, frame2), axis=0)

height, width, layers = vis.shape

video = cv2.VideoWriter(video_name, 0, 1, (width,height))

for i in range(len(images1)):
    # Write some Text

    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,500)
    fontScale              = 1
    fontColor              = (255,255,255)
    thickness              = 1
    lineType               = 2

    im1 = cv2.imread(os.path.join(folder1, images1[i]))
    #cv2.resize(im1, (int(height/2), width))

    cv2.putText(im1,'SuperPoint Tracking_v1',
        bottomLeftCornerOfText,
        font,
        fontScale,
        fontColor,
        thickness,
        lineType)

    im2 = cv2.imread(os.path.join(folder2, images2[i]))
    #cv2.resize(im2, (int(height/2), width))

    cv2.putText(im2,'SuperPoint Tracking_v2',
        bottomLeftCornerOfText,
        font,
        fontScale,
        fontColor,
        thickness,
        lineType)

    vis = np.concatenate((im1, im2), axis=0)
    video.write(vis)

cv2.destroyAllWindows()
video.release()
