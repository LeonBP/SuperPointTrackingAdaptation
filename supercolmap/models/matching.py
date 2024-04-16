# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

import torch
import cv2

from .superpoint import SuperPoint
from .SuperPointNet import SuperPointNet
from .superglue import SuperGlue
from .bruteforce import BruteForce
from .bftorch import BFTorch
from .lightglue import LightGlue
from .superpoint_lg import SuperPoint_LG


class Matching(torch.nn.Module):
    """ Image Matching Frontend (SuperPoint + SuperGlue) """
    def __init__(self, config={}):
        super().__init__()
        config_sp = config.get('superpoint', {})
        if config_sp.get('featuretype') == 'superpoint':
            self.superpoint = SuperPoint(config_sp)
        elif config_sp.get('featuretype') == 'superpoint_lg':
            self.superpoint = SuperPoint_LG(**config_sp)
        else:
            self.superpoint = SuperPointNet(config_sp)

        config_sg = config.get('superglue', {})
        self.colmap = False
        if config_sg.get('matchtype') == 'superglue':
            self.superglue = SuperGlue(config_sg)
        elif config_sg.get('matchtype') == 'bruteforce'\
                or config_sg.get('matchtype') == 'ransac':
            self.superglue = BruteForce(config_sg)
        elif config_sg.get('matchtype') == 'bftorch':
            self.superglue = BFTorch(config_sg)
        elif config_sg.get('matchtype') == 'lightglue':
            if 'lightglue' in config_sg.get('superglue'):
                self.superglue = LightGlue(**config_sg)
            else:
                self.superglue = LightGlue(features=None, **config_sg)
        elif config_sg.get('matchtype') == 'colmap':
            self.colmap = True

    def forward(self, data, timer=None):
        """ Run SuperPoint (optionally) and SuperGlue
        SuperPoint is skipped if ['keypoints0', 'keypoints1'] exist in input
        Args:
          data: dictionary with minimal keys: ['image0', 'image1']
        """
        pred = {}

        # Extract SuperPoint (keypoints, scores, descriptors) if not provided
        if 'keypoints0' not in data:
            pred0 = self.superpoint({'image': data['image0'][:,:,:,205:-155]})
            pred = {**pred, **{k+'0': v for k, v in pred0.items()}}
            # for k in pred:
            #     print("sp0")
            #     print(k,len(pred[k]),pred[k][0].size(),pred[k][0])
            pred['keypoints0'][0]=pred['keypoints0'][0]+torch.tensor([205,0]).to('cuda:0')
            #print(pred['keypoints0'][0].size())
            timer.update('detector0')
        if 'keypoints1' not in data:
            pred1 = self.superpoint({'image': data['image1'][:,:,:,205:-155]})
            pred = {**pred, **{k+'1': v for k, v in pred1.items()}}
            pred['keypoints1'][0]=pred['keypoints1'][0]+torch.tensor([205,0]).to('cuda:0')
            #print(pred['keypoints1'])
            #print(pred['keypoints1'][0].size())
            timer.update('detector1')
        # print(pred['semi0'].detach().cpu().numpy().dtype)
        # dla = pred['semi0'].detach()[0,:-1,:,:]
        # _, h, w = dla.shape
        # dla = dla.permute(1, 2, 0).reshape(h, w, 8, 8)
        # dla = dla.permute(0, 2, 1, 3).reshape(h * 8, w * 8).cpu().numpy()
        # cv2.imwrite("/home/leon/Experiments/output/sp_semi.png",dla*255.)
        # exit(1)
        # Batch all features
        # We should either have i) one image per batch, or
        # ii) the same number of local features for all images in the batch.
        data = {**data, **pred}

        for k in data:
            if isinstance(data[k], (list, tuple)):
                data[k] = torch.stack(data[k])

        # Perform the matching
        if not self.colmap:
            if len(pred['keypoints0'][0]) > 0 and len(pred['keypoints1'][0]) > 0:
                pred = {**pred, **self.superglue(data)}
            else:
                shape0, shape1 = data['keypoints0'].shape[:-1], data['keypoints1'].shape[:-1]
                new_pred = {'matches0': data['keypoints0'].new_full(shape0, -1, dtype=torch.int),
                            'matches1': data['keypoints1'].new_full(shape1, -1, dtype=torch.int),
                            'matching_scores0': data['keypoints0'].new_zeros(shape0),
                            'matching_scores1': data['keypoints1'].new_zeros(shape1),}
                pred = {**pred, **new_pred}
            # for k in pred:
            #     print("sg")
            #     if k == "stop":
            #         print(k,pred[k])
            #     else:
            #         print(k,len(pred[k]),pred[k][0].size(),pred[k][0])

        return pred
