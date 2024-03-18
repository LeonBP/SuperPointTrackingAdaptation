import os

import numpy as np

src = "/media/discoGordo/dataset_leon/UZ/colmap_benchmark_frames"
# src = "/media/discoGordo/C3VD"
# src = "/media/discoGordo/dataset_leon/UZ/test_frames"

models = ["sift_gm", "sptracking2_disk", "sptracking3_disk", "sptracking3b4+_disk"]#["sift_gm", "sp_disk", "sp_gm", "sp_sg", "sptracking3_disk", "sptracking3_gm", "sptracking3_sgh50", "sptracking3b4+_gm"]
#"sptracking3_gm_md08", "sptracking3_gm_err2", "sptracking3_gm_kt0020", "sptracking3_gm_kt0025"
seqs_old = [("00033","15"),("00034","38"),("02001","39"),("02003","86"),("02005","28"),("00364","14"),("00364","18")]
seqs_new = [("Seq_001","15"),("Seq_002","38"),("Seq_014","39"),("Seq_016","86"),("Seq_017","28"),("Seq_095","14"),("Seq_095","18")]
seqs_text = ["Seq\_001\_1", "Seq\_002\_1", "Seq\_014\_1", "Seq\_016\_1", "Seq\_017\_1", "Seq\_095\_1", "Seq\_095\_2"]
# seqs_old = [("Seq_021","2"),("Seq_021","3"),("Seq_021","4"),("Seq_021","5"),("Seq_021","7")]
# seqs_new = [("Seq_021","2"),("Seq_021","3"),("Seq_021","4"),("Seq_021","5"),("Seq_021","7")]
# seqs_text = ["Seq\_021\_2", "Seq\_021\_3", "Seq\_021\_4", "Seq\_021\_5", "Seq\_021\_7"]
# seqs_old = [("cecum_t2_a_under_review","color_1440"),("trans_t1_a_under_review","color_1440"),("trans_t2_b_under_review","color_1440"),("sigmoid_t3_a_under_review","color_1440"),("desc_t4_a_under_review","color_1440")]
# seqs_new = [("cecum_t2_a_under_review","color_1440"),("trans_t1_a_under_review","color_1440"),("trans_t2_b_under_review","color_1440"),("sigmoid_t3_a_under_review","color_1440"),("desc_t4_a_under_review","color_1440")]
# seqs_text = ["cecum\_t2_a\_under\_review", "trans\_t1\_a\_under\_review", "trans\_t2\_b\_under\_review", "sigmoid\_t3\_a\_under\_review", "desc\_t4\_a\_under\_review"]
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
    if model == "sift_gm" and "test_frames" not in src:
        path_results = "/media/discoGordo/dataset_leon/colmap_MIDL/pseudoGT_original"# + seq + "/" + subseq
    else:
        path_results = '/media/discoGordo/dataset_leon/reconstructions_MICCAI2023' + '/' + model #+ '/' + seq + '/' + subseq
    for i in range(len(seqs_old)):
        seq, subseq = seqs_new[i]
        if "sift" in model and "test_frames" not in src:
            f = open(path_results + "/" + seqs_old[i][0] + "/" + seqs_old[i][1] + "/stats.txt", "r")
        elif model == "sptracking3_disk" or model == "sptracking2_disk" or model == "sp_sg":
            f = open(path_results + "/" + seqs_old[i][0] + "/" + seqs_old[i][1] + "/0/stats.txt", "r")
        elif (model == "sptracking3_gm" or model == "sptracking3_sgh50") and "test_frames" not in src:
            f = open(path_results + "/" + seq + "_err4" + "/" + subseq + "/stats.txt", "r")
        elif "_err" in model:
            f = open(path_results[:-5] + "/" + seq + "_err2" + "/" + subseq + "/stats.txt", "r")
        elif os.path.isdir(path_results + "/" + seq + "/" + subseq + "/0"):
            f = open(path_results + "/" + seq + "/" + subseq + "/0/stats.txt", "r")
        else:
            f = None
        if f is not None:
            lines = f.readlines()
        if len(lines) == 0 or f is None:
            cameras_model.append(1)
            images_model.append(os.listdir(src + "/" + seq + "/" + subseq).__len__())
            registered_images_model.append(0)
            points_model.append(0)
            observations_model.append(0)
            mean_track_length_model.append(0)
            mean_observations_per_image_model.append(0)
            mean_reprojection_error_model.append(10000)
        else:
            cameras_model.append(int(lines[0].split(" ")[-1]))
            images_model.append(os.listdir(src + "/" + seq + "/" + subseq).__len__())
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
for j in range(len(seqs_text)):
    text += " & " + seqs_text[j]
text += " \\\\ \n\\hline\n"
for i in range(len(models)):
    text += models[i].replace("_", "\\_")
    # values = mean_reprojection_error[i]
    values = 100 * np.array(registered_images[i]) / np.array(images[i])
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