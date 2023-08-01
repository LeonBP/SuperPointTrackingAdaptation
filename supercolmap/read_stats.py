import os

import numpy as np

src = "/media/discoGordo/dataset_leon/UZ/colmap_benchmark_frames/"
models = ["em_disk", "sphomography1_disk", "sptracking1_disk", "sptracking2_disk", "sptracking3_disk", "sptracking4_disk"]#["sift", "sp_sg", "sptracking3_disk"]#
seqs = [("00033","15"),("00034","38"),("02001","39"),("02003","86"),("02005","28"),("00364","14"),("00364","18")]
seqs_text = ["Seq\_001\_1", "Seq\_002\_1", "Seq\_014\_1", "Seq\_016\_1", "Seq\_017\_1", "Seq\_095\_1", "Seq\_095\_2"]
cameras = []
images = []
registered_images = []
points = []
observations = []
mean_track_length = []
mean_observations_per_image = []
mean_reprojection_error = []
for model in models:
    cameras_model = []
    images_model = []
    registered_images_model = []
    points_model = []
    observations_model = []
    mean_track_length_model = []
    mean_observations_per_image_model = []
    mean_reprojection_error_model = []
    if model == "sift":
        path_results = "/media/discoGordo/dataset_leon/colmap_MIDL/pseudoGT/"# + seq + "/" + subseq
    else:
        path_results = '/media/discoGordo/dataset_leon/reconstructions_MICCAI2023/' + model + '/'# + seq + '/' + subseq
    for seq, subseq in seqs:
        f = open(path_results + seq + "/" + subseq + "/stats.txt", "r")
        lines = f.readlines()
        cameras_model.append(int(lines[0].split(" ")[-1]))
        images_model.append(os.listdir(src + seq + "/" + subseq).__len__())
        registered_images_model.append(int(lines[2].split(" ")[-1]))
        points_model.append(int(lines[3].split(" ")[-1]))
        observations_model.append(int(lines[4].split(" ")[-1]))
        mean_track_length_model.append(float(lines[5].split(" ")[-1]))
        mean_observations_per_image_model.append(float(lines[6].split(" ")[-1]))
        mean_reprojection_error_model.append(float(lines[7].split(" ")[-1][:-3]))
    cameras.append(cameras_model)
    images.append(images_model)
    registered_images.append(registered_images_model)
    points.append(points_model)
    observations.append(observations_model)
    mean_track_length.append(mean_track_length_model)
    mean_observations_per_image.append(mean_observations_per_image_model)
    mean_reprojection_error.append(mean_reprojection_error_model)
text = ""
for j in range(len(seqs)):
    text += " & " + seqs_text[j]
text += " \\\\ \n\\hline\n"
for i in range(len(models)):
    text += models[i].replace("_", "\\_")
    values = mean_track_length[i]
    for j in range(len(seqs_text)):
        value = values[j]
        if isinstance(value, float):
            text += " & $" + "{:0.2f}".format(round(value,2)) +"$"
        else:
            text += " & $" + str(value) +"$"
    # add the mean and the std values
    if True:
        text += " & $" + "{:0.2f}".format(round(np.mean(values),2)) +"$"
        text += " & ($" + "{:0.2f}".format(round(np.std(values),2)) +"$)"
    else:
        text += " & $" + "{:0.2f}".format(round(np.mean([i / j for i, j in zip(values, images[i])])*100,2)) +"$"
        text += " & ($" + "{:0.2f}".format(round(np.std([i / j for i, j in zip(values, images[i])])*100,2)) +"$)"
    text += " \\\\ \n"
print(text)