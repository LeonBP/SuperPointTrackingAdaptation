from pathlib import Path
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, zeros_
from models.loader import modelLoader, pretrainedLoader
import cv2
import numpy as np
from utils.utils import getPtsFromHeatmap, flattenDetection

# class SuperPointNet(torch.nn.Module):
#   """ Pytorch definition of SuperPoint Network. """

#   def vgg_block(self, filters_in, filters_out, kernel_size, padding, name, batch_normalization=True, relu=True, kernel_reg=0., **params):
#     modules = [nn.Conv2d(filters_in, filters_out, kernel_size=kernel_size, stride=1, padding=padding)]
#     if batch_normalization:
#         modules.append(nn.BatchNorm2d(filters_out))
#     if relu:
#         modules.append(self.relu)
#     else:
#         print('NO RELU!!!!!!!!!!!!!!!')
#     return nn.Sequential(*modules)

#     # x2 = conv_layer(inputs) ## HOW TO IMPLEMENT KERNEL REG?
#     # if relu:
#     #     x2 = self.relu(x2)
#     # else:
#     #     print('NO RELU!!!!!!!!!!!!!!!')
#     # if batch_normalization:
#     #     x2 = nn.BatchNorm2d(filters_out)(x2)
#     # return x2

#   def __init__(self):
#     super(SuperPointNet, self).__init__()
#     self.relu = nn.ReLU(inplace=True)
#     self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

#     params_conv = {'batch_normalization': False, 'relu': True, 'kernel_reg': 0.}

#     self.conv1_1 = self.vgg_block(1, 64, 3, 1, 'conv1_1', **params_conv)
#     self.conv1_2 = self.vgg_block(64, 64, 3, 1, 'conv1_2', **params_conv)

#     self.conv2_1 = self.vgg_block(64, 64, 3, 1, 'conv2_1', **params_conv)
#     self.conv2_2 = self.vgg_block(64, 64, 3, 1, 'conv2_2', **params_conv)

#     self.conv3_1 = self.vgg_block(64, 128, 3, 1, 'conv3_1', **params_conv)
#     self.conv3_2 = self.vgg_block(128, 128, 3, 1, 'conv3_2', **params_conv)

#     self.conv4_1 = self.vgg_block(128, 128, 3, 1, 'conv4_1', **params_conv)
#     self.conv4_2 = self.vgg_block(128, 128, 3, 1, 'conv4_2', **params_conv)

#     # Detector Head.
#     self.det_conv1_1 = self.vgg_block(128, 256, 3, 1, 'det_conv1_1', **params_conv)
#     self.det_conv1_2 = self.vgg_block(256, 65, 1, 0, 'det_conv1_2', batch_normalization=False, relu=False)

#     # Descriptor Head.
#     self.des_conv1_1 = self.vgg_block(128, 256, 3, 1, 'des_conv1_1', **params_conv)
#     self.des_conv1_2 = self.vgg_block(256, 256, 1, 0, 'des_conv1_2', batch_normalization=False, relu=False)
#     # desc = nn.functional.normalize(desc, p=2, dim=1)

#   def forward(self, x):
#     """ Forward pass that jointly computes unprocessed point and descriptor
#     tensors.
#     Input
#       x: Image pytorch tensor shaped N x 1 x H x W.
#     Output
#       semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
#       desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
#     """
#     # Shared Encoder.
#     # print(x.size(), '!!!!!!!!!!!')
#     x = self.pool(self.conv1_2(self.conv1_1(x)))
#     # print(x.size(), self.conv1_1(x).size(), '!!!!!!!!!!!')
#     x = self.pool(self.conv2_2(self.conv2_1(x)))
#     x = self.pool(self.conv3_2(self.conv3_1(x)))
#     x = self.conv4_2(self.conv4_1(x))
#     # Detector Head.
#     semi = self.det_conv1_2(self.det_conv1_1(x))
#     # print(semi.size(), '!!!!!!!!!!!')
#     # Descriptor Head.
#     desc = self.des_conv1_2(self.des_conv1_1(x))
#     desc = nn.functional.normalize(desc, p=2, dim=1)

#     return semi, desc

# ###############################

def toNumpy(tensor):
    return tensor.detach().cpu().numpy()

def sample_desc_from_points(coarse_desc, pts, cell=8, device='cpu'):
    # --- Process descriptor.
    H, W = coarse_desc.shape[2]*cell, coarse_desc.shape[3]*cell
    D = coarse_desc.shape[1]
    if pts.shape[1] == 0:
        desc = np.zeros((D, 0))
    else:
        # Interpolate into descriptor map using 2D point locations.
        samp_pts = torch.from_numpy(pts[:2, :].copy())
        samp_pts[0, :] = (samp_pts[0, :] / (float(W) / 2.)) - 1.
        samp_pts[1, :] = (samp_pts[1, :] / (float(H) / 2.)) - 1.
        samp_pts = samp_pts.transpose(0, 1).contiguous()
        samp_pts = samp_pts.view(1, 1, -1, 2)
        samp_pts = samp_pts.float()
        samp_pts = samp_pts.to(device)
        desc = torch.nn.functional.grid_sample(coarse_desc, samp_pts, align_corners=True)
        desc = desc.data.cpu().numpy().reshape(D, -1)
        desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]
    return desc

def extract_superpoint_kp_and_desc_pytorch(
        keypoint_map, descriptor_map, conf_thresh=0.015,
        nms_dist=4, cell=8, device='cpu',
        keep_k_points=10000, m=None, mask=None):

    def select_k_best(points, k, desc):
        """ Select the k most probable points (and strip their proba).
        points has shape (num_points, 3) where the last coordinate is the proba. """
        pointsdesc=np.append(points,desc,axis=1)
        sorted_prob = pointsdesc[pointsdesc[:, 2].argsort(), :]#:2
        start = min(k, points.shape[0])
        return sorted_prob[-start:, :2], sorted_prob[-start:, 3:]

    semi_flat_tensor = flattenDetection(keypoint_map[0, :, :, :]).detach()
    semi_flat = toNumpy(semi_flat_tensor)
    semi_thd = np.squeeze(semi_flat, 0)
    pts=getPtsFromHeatmap(
            semi_thd, conf_thresh=conf_thresh, nms_dist=nms_dist)
    desc_sparse_batch = sample_desc_from_points(
            descriptor_map, pts, cell=cell, device=device)
    #print(pts.shape, desc_sparse_batch.shape)

    # Extract keypoints
    keypoints = np.transpose(pts)[:,[1,0,2]]
    descriptors = np.transpose(desc_sparse_batch)
    if m!=None:
        keyp=np.array([])
        desc=np.array([])
        for i in range(keypoints.shape[0]):
            if mask[int(keypoints[i,0]),int(keypoints[i,1])]>127:
                if len(keyp)==0:
                    keyp=np.array([keypoints[i,:]])
                    desc=np.array([descriptors[i,:]])
                else:
                    keyp=np.append(keyp,np.array([keypoints[i,:]]),axis=0)
                    desc=np.append(desc,np.array([descriptors[i,:]]),axis=0)
        keypoints=keyp
        descriptors=desc
    if len(keypoints)>0:
        keypoints, descriptors = select_k_best(keypoints, keep_k_points, descriptors)
        keypoints = keypoints.astype(int)
    else:
        descriptors=np.array([])

    # Convert from just pts to cv2.KeyPoints
    #keypoints_cv2 = [cv2.KeyPoint(p[1], p[0], 1) for p in keypoints]

    return keypoints, descriptors

def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert(nms_radius >= 0)

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def remove_borders(keypoints, scores, border: int, height: int, width: int, bordermask):
    """ Removes keypoints too close to the border """
    mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
    mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
    kps=keypoints.cpu().numpy()
    mask_border = torch.tensor(bordermask[kps[:,0],kps[:,1]]>127).to('cuda:0')
    #print(mask_border.size(),mask_border)
    mask = mask_h & mask_w & mask_border
    return keypoints[mask], scores[mask]


def top_k_keypoints(keypoints, scores, k: int):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0)
    return keypoints[indices], scores


def sample_descriptors(keypoints, descriptors, s: int = 8):
    """ Interpolate descriptors at keypoint locations """
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5
    keypoints /= torch.tensor([(w*s - s/2 - 0.5), (h*s - s/2 - 0.5)],
                              ).to(keypoints)[None]
    keypoints = keypoints*2 - 1  # normalize to (-1, 1)
    args = {'align_corners': True} if int(torch.__version__[2]) > 2 else {}
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', **args)
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1)
    return descriptors



class SuperPointNet(nn.Module):
    default_config = {
        'descriptor_dim': 256,
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1,
        'remove_borders': 4,
    }
    
    """ Pytorch definition of SuperPoint Network. """
    def __init__(self, config):
        super(SuperPointNet, self).__init__()
        self.config = {**self.default_config, **config}

        self.relu = nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.unpool = torch.nn.MaxUnpool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        det_h = 65
        gn = 64
        useGn = False
        self.reBn = True
        if self.reBn:
            print ("model structure: relu - bn - conv")
        else:
            print ("model structure: bn - relu - conv")


        if useGn:
            print ("apply group norm!")
        else:
            print ("apply batch norm!")


        # Shared Encoder.
        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.bn1a = nn.GroupNorm(gn, c1) if useGn else nn.BatchNorm2d(c1, track_running_stats=False)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.bn1b = nn.GroupNorm(gn, c1) if useGn else nn.BatchNorm2d(c1, track_running_stats=False)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.bn2a = nn.GroupNorm(gn, c2) if useGn else nn.BatchNorm2d(c2, track_running_stats=False)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.bn2b = nn.GroupNorm(gn, c2) if useGn else nn.BatchNorm2d(c2, track_running_stats=False)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.bn3a = nn.GroupNorm(gn, c3) if useGn else nn.BatchNorm2d(c3, track_running_stats=False)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.bn3b = nn.GroupNorm(gn, c3) if useGn else nn.BatchNorm2d(c3, track_running_stats=False)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.bn4a = nn.GroupNorm(gn, c4) if useGn else nn.BatchNorm2d(c4, track_running_stats=False)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
        self.bn4b = nn.GroupNorm(gn, c4) if useGn else nn.BatchNorm2d(c4, track_running_stats=False)
        # Detector Head.
        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.bnPa = nn.GroupNorm(gn, c5) if useGn else nn.BatchNorm2d(c5, track_running_stats=False)
        self.convPb = nn.Conv2d(c5, det_h, kernel_size=1, stride=1, padding=0)
        self.bnPb = nn.GroupNorm(det_h, det_h) if useGn else nn.BatchNorm2d(65, track_running_stats=False)
        # Descriptor Head.
        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.bnDa = nn.GroupNorm(gn, c5) if useGn else nn.BatchNorm2d(c5, track_running_stats=False)
        self.convDb = nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)
        self.bnDb = nn.GroupNorm(gn, d1) if useGn else nn.BatchNorm2d(d1, track_running_stats=False)

        #path = Path(__file__).parent / 'weights/superpoint_v1.pth'
        #self.load_state_dict(torch.load(str(path)))

        path = Path('/home/leon/Experiments/SuperPoint_pytorch') / \
               self.config['weights']
               # 'superpoint_tracking_v2/checkpoints/superPointNet_400000_checkpoint.pth.tar'
               # 'superpoint_ucluzlabel100_Bn/checkpoints/superPointNet_200000_checkpoint.pth.tar'#_spec100pixels
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'], strict=False)

        self.bordermask = cv2.imread("/media/discoGordo/dataset_leon/UZ/masks/HCULB_1080_mask_eq.png",cv2.IMREAD_GRAYSCALE)

        mk = self.config['max_keypoints']
        if mk == 0 or mk < -1:
            raise ValueError('\"max_keypoints\" must be positive or \"-1\"')

        print('Loaded SuperPointNet model (\"{}\" weights)'.format(
            self.config['weights']))

    def forward(self, data):
        """ Forward pass that jointly computes unprocessed point and descriptor
        tensors.
        Input
            x: Image pytorch tensor shaped N x 1 x H x W.
        Output
            semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
            desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
        """

        # Shared Encoder.
        #print(data['image'].size())
        #cv2.imwrite("/home/leon/Experiments/output/endomapper/SEP2021/"+"source"+".png",255.*data['image'][0].permute(1,2,0).cpu().numpy())
        x = self.relu(self.bn1a(self.conv1a(data['image'])))
        conv1 = self.relu(self.bn1b(self.conv1b(x)))
        x, ind1 = self.pool(conv1)
        x = self.relu(self.bn2a(self.conv2a(x)))
        conv2 = self.relu(self.bn2b(self.conv2b(x)))
        x, ind2 = self.pool(conv2)
        x = self.relu(self.bn3a(self.conv3a(x)))
        conv3 = self.relu(self.bn3b(self.conv3b(x)))
        x, ind3 = self.pool(conv3)
        x = self.relu(self.bn4a(self.conv4a(x)))
        x = self.relu(self.bn4b(self.conv4b(x)))
        
        
        # Detector Head.
        cPa = self.relu(self.bnPa(self.convPa(x)))
        semi = self.bnPb(self.convPb(cPa)) #<<<<<<< THIS ONE IS THE ORIGINAL OUTPUT
        
        
        
        
        ##CHANGE?
        scores = torch.clone(semi)##
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h*8, w*8)
        scores = simple_nms(scores, self.config['nms_radius'])
        # Extract keypoints
        #print(np.amax(scores[0,:,:].cpu().numpy()))
        #cv2.imwrite("/home/leon/Experiments/output/endomapper/SEP2021/"+"sources"+str(np.amax(scores[0,:,:].cpu().numpy()))+".png",255.*scores[0,:,:].cpu().numpy()/np.amax(scores[0,:,:].cpu().numpy()))
        keypoints = [
            torch.nonzero(s > self.config['keypoint_threshold'])
            for s in scores]
        #print(keypoints)
        scores = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]
        # Discard keypoints near the image borders
        keypoints, scores = list(zip(*[
            remove_borders(k, s, self.config['remove_borders'], h*8, w*8, self.bordermask)
            for k, s in zip(keypoints, scores)]))
        # Keep the k keypoints with highest score
        if self.config['max_keypoints'] >= 0:
            keypoints, scores = list(zip(*[
                top_k_keypoints(k, s, self.config['max_keypoints'])
                for k, s in zip(keypoints, scores)]))
        #print(len(keypoints),keypoints[0].shape)
        # Convert (h, w) to (x, y)
        keypoints = [torch.flip(k, [1]).float() for k in keypoints]
        
        
        
        
        
        # Descriptor Head.
        cDa = self.relu(self.bnDa(self.convDa(x)))
        desc = self.bnDb(self.convDb(cDa))
        dn = torch.norm(desc, p=2, dim=1) # Compute the norm.
        descnorm = desc.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize. #<<<<<<< THIS ONE IS THE ORIGINAL OUTPUT
        #print(self.config['keypoint_threshold'],self.config['nms_radius'],self.config['max_keypoints'])
        '''kp1, desc1 = extract_superpoint_kp_and_desc_pytorch(
                    semi, descnorm, conf_thresh=self.config['keypoint_threshold'],
                    nms_dist=self.config['nms_radius'], cell=8, device='cuda:0',
                    keep_k_points=self.config['max_keypoints'], m=True, mask=self.bordermask)'''
        
        ##CHANGE?
        descriptors = torch.nn.functional.normalize(desc, p=2, dim=1)
        #desc = torch.clone(descriptors)##
        # Extract descriptors
        descriptors = [sample_descriptors(k[None], d[None], 8)[0]
                       for k, d in zip(keypoints, descriptors)]
        #print(semi.shape,desc.shape,keypoints[0].shape,scores[0].shape,descriptors[0].shape)
        #kp1, desc1 = extract_superpoint_kp_and_desc_pytorch(##
        #        semi, desc, conf_thresh=self.config['keypoint_threshold'],
        #        nms_dist=self.config['nms_radius'], cell=8, device='cuda:0',
        #        keep_k_points=self.config['max_keypoints'], m=True, mask=self.bordermask)
        #print(keypoints[0][0,0], kp1[0,0], descriptors[0][0,0], desc1[0,0])
        
        return {
            'keypoints': keypoints,
            'scores': scores,
            'descriptors': descriptors,
            'semi': semi,
            'desc': descnorm
        }

def forward_original(self, x):
    """ Forward pass that jointly computes unprocessed point and descriptor
    tensors.
    Input
      x: Image pytorch tensor shaped N x 1 x H x W.
    Output
      semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
      desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
    """
    # Shared Encoder.
    x = self.relu(self.conv1a(x))
    x = self.relu(self.conv1b(x))
    x = self.pool(x)
    x = self.relu(self.conv2a(x))
    x = self.relu(self.conv2b(x))
    x = self.pool(x)
    x = self.relu(self.conv3a(x))
    x = self.relu(self.conv3b(x))
    x = self.pool(x)
    x = self.relu(self.conv4a(x))
    x = self.relu(self.conv4b(x))
    # Detector Head.
    cPa = self.relu(self.convPa(x))
    semi = self.convPb(cPa)

    # Descriptor Head.
    cDa = self.relu(self.convDa(x))
    desc = self.convDb(cDa)
    dn = torch.norm(desc, p=2, dim=1) # Compute the norm.
    desc = desc.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.
    return semi, desc





###############################

def conv(in_planes, out_planes, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2, stride=2),
        nn.ReLU(inplace=True)
    )


def upconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True)
    )




if __name__ == '__main__':

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = SuperPointNet()
  model = model.to(device)

  # check keras-like model summary using torchsummary
  from torchsummary import summary
  summary(model, input_size=(1, 224, 224))
