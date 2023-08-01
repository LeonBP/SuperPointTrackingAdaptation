import cv2
import numpy as np
import torch
from torch import nn

def compute_inliers_2(matched_kp1, matched_kp2, HorE='H'):
    matched_pts1 = cv2.KeyPoint_convert(matched_kp1)
    matched_pts2 = cv2.KeyPoint_convert(matched_kp2)

    dist = np.array([ -0.1205372, -0.01179983, 0.00269742, -0.0001362505]).reshape((4,1)) ##calibration from HCULB_00044,(45 and 48 are good choices too)
    mtx =  np.array([[733.1061, 0.0, 739.2826],
             [0.0, 735.719, 539.6911],
             [0.0, 0.0, 1.0]])
    und1 = np.float32(np.zeros(matched_pts1.shape)[:,np.newaxis,:])
    cv2.undistortPoints(src=matched_pts1[:,np.newaxis,:],cameraMatrix=mtx,distCoeffs=dist,dst=und1,P=mtx)
    und2 = np.float32(np.zeros(matched_pts2.shape)[:,np.newaxis,:])
    cv2.undistortPoints(src=matched_pts2[:,np.newaxis,:],cameraMatrix=mtx,distCoeffs=dist,dst=und2,P=mtx)
    
    # Estimate the essential matrix between the matches using RANSAC
    Es, inliers_E = cv2.findEssentialMat(und1[:, 0, [1, 0]],
                                         und2[:, 0, [1, 0]],
                                         mtx,
                                         cv2.RANSAC,
                                         prob=0.9999,
                                         threshold=3.)
    if inliers_E is None or Es is None:
        E = np.eye(3)
        inliers_E = np.array([0])
    else:
        E = Es[:3,:]
        n = 1
        while np.isnan(np.sum(E)):
            E = Es[n*3:(n+1)*3]
            n += 1
    
    inliers_E = inliers_E.flatten()
    return E, inliers_E
    
def compute_inliers(matched_kp1, matched_kp2, HorF='H'):
    matched_pts1 = cv2.KeyPoint_convert(matched_kp1)
    matched_pts2 = cv2.KeyPoint_convert(matched_kp2)

    dist = np.array([ -0.1389272, -0.001239606, 0.0009125824, -4.071615e-05]).reshape((4,1)) ##calibration from HCULB_00039,(44 and 48 are good choices too)
    mtx =  np.array([[717.2104, 0.0, 735.3566],
             [0.0, 717.4816, 552.7982],
             [0.0, 0.0, 1.0]])
    und1 = np.float32(np.zeros(matched_pts1.shape)[:,np.newaxis,:])
    cv2.undistortPoints(src=matched_pts1[:,np.newaxis,:],cameraMatrix=mtx,distCoeffs=dist,dst=und1,P=mtx)
    und2 = np.float32(np.zeros(matched_pts2.shape)[:,np.newaxis,:])
    cv2.undistortPoints(src=matched_pts2[:,np.newaxis,:],cameraMatrix=mtx,distCoeffs=dist,dst=und2,P=mtx)
    if 'E' in HorF or 'F' in HorF:
        if 'E' in HorF:
            # Estimate the essential matrix between the matches using RANSAC
            H, inliers = cv2.findEssentialMat(und1[:,0, [1, 0]],
                                            und2[:,0, [1, 0]],
                                            mtx,
                                            cv2.RANSAC,
                                            prob = 0.9999)
        else:
            # Estimate the fundamental matrix between the matches using RANSAC
            H, inliers = cv2.findFundamentalMat(und1[:,0, [1, 0]],
                                            und2[:,0, [1, 0]],
                                            cv2.RANSAC,
                                            confidence = 0.9999)
    else:
        # Estimate the homography between the matches using RANSAC
        H, inliers = cv2.findHomography(und1[:,0, [1, 0]],
                                        und2[:,0, [1, 0]],
                                        cv2.RANSAC,
                                        confidence = 0.9999)
                                        
    if inliers is None:
        inliers=[0]
    else:
        inliers = inliers.flatten()
    return H, inliers

class BruteForce(nn.Module):
    default_config = {
        'detector': 'superpoint',
        'matrix': 'E'
    }
    
    def __init__(self,config):
        super().__init__()
        self.config = {**self.default_config, **config}
        
        assert self.config['matrix'] in ['H', 'F', 'E']
        print('Loaded BruteForce model (fit to \"{}\" matrix)'.format(
            self.config['matrix']))
            
    def forward(self, data):
        desc0, desc1 = data['descriptors0'], data['descriptors1']
        kpts0, kpts1 = data['keypoints0'], data['keypoints1']

        if kpts0.shape[1] == 0 or kpts1.shape[1] == 0:  # no keypoints
            shape0, shape1 = kpts0.shape[:-1], kpts1.shape[:-1]
            return {
                'matches0': kpts0.new_full(shape0, -1, dtype=torch.int),
                'matches1': kpts1.new_full(shape1, -1, dtype=torch.int),
                'matching_scores0': kpts0.new_zeros(shape0),
                'matching_scores1': kpts1.new_zeros(shape1),
            }
        #indices0 = kpts0.new_full(shape0, -1, dtype=torch.int)
        #indices1 = kpts1.new_full(shape1, -1, dtype=torch.int)
        #mscores0 = kpts0.new_zeros(shape0)
        #mscores1 = kpts1.new_zeros(shape1)
        #m_kp1, m_kp2, matches = match_descriptors_cv2_BF(kp1, desc1.astype('float32'), kp2, desc2.astype('float32'), weights_name)
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        descriptors0 = desc0.cpu().numpy()[0].T
        descriptors1 = desc1.cpu().numpy()[0].T
        matches = bf.match(descriptors0, descriptors1)
        
        keypoints0 = kpts0.cpu().numpy()[0].astype(int)
        keypoints1 = kpts1.cpu().numpy()[0].astype(int)
        matches_idx = np.array([m.queryIdx for m in matches])
        m_kp0 = np.array([cv2.KeyPoint(keypoints0[idx,1], keypoints0[idx,0], 1) for idx in matches_idx])
        matches_idx = np.array([m.trainIdx for m in matches])
        m_kp1 = np.array([cv2.KeyPoint(keypoints1[idx,1], keypoints1[idx,0], 1) for idx in matches_idx])
        
        if len(matches)>0:
            if self.config['matchtype'] == 'ransac':
                E, inliers_E = compute_inliers_2(m_kp0, m_kp1, self.config['matrix'])
            else:
                inliers_E = np.ones_like(m_kp0)
        else:
            inliers_E=np.array([0])
        if np.amax(inliers_E)>1:
            inliers_E=np.array([0])
        # print(inliers_E)
        # exit(1)
        if len(matches)==len(inliers_E):
            matches = np.array(matches)[np.array(inliers_E).astype(bool)].tolist()
        else:
            matches=[]
        
        indices0 = np.repeat(-1,descriptors0.shape[0])
        for m in matches:
            indices0[m.queryIdx] = m.trainIdx
        indices0 = torch.from_numpy(indices0).unsqueeze(0).to('cuda:0')
        mscores0 = torch.where(indices0 > 0, 1, 0).to('cuda:0')

        return {
            'matches0': indices0, # use -1 for invalid match
            'matches1': kpts1.new_full(kpts1.shape[:-1], -1, dtype=torch.int), # use -1 for invalid match
            'matching_scores0': mscores0,
            'matching_scores1': kpts1.new_zeros(kpts1.shape[:-1]),
        }
            
        
