#!/bin/bash

python evaluation.py /media/discoGordo/dataset_leon/BMVC2021/homestimation/spv1_homestimation/predictions --repeatibility --outputImg --homography --plotMatching >> spv1.txt
#python evaluation.py /media/discoGordo/dataset_leon/BMVC2021/homestimation/spft_homestimation/predictions --repeatibility --outputImg --homography --plotMatching >> spft.txt
python evaluation.py /media/discoGordo/dataset_leon/BMVC2021/homestimation/spspec_homestimation/predictions --repeatibility --outputImg --homography --plotMatching >> spspec.txt

