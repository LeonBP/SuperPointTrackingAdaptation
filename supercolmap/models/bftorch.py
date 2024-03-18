import torch, os, argparse, h5py, warnings, imageio
from torch import nn
import numpy as np
from tqdm import tqdm

# from torch_dimcheck import dimchecked

# from disk.geom import distance_matrix

MAX_FULL_MATRIX = 10000**2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'h5_path',
        help=('Path to the .h5 artifacts directory (containing descriptors.h5 '
              'and keypoints.h5)')
    )
    parser.add_argument(
        '--f16', action='store_true',
        help=('Compute distance matrices in half precision (offers a '
              'substantial speedup with Turing and later GPUs).')
    )
    parser.add_argument(
        '--u16', action='store_true',
        help=('Store matches with as uin16. This won\'t work if you have '
              'more than ~65k features in an image, but otherwise saves '
              'disk space.')
    )
    parser.add_argument(
        '--rt', type=float, default=None,
        help='Ratio test value. Leave unspecified to perform no ratio test'
    )
    parser.add_argument(
        '--save-threshold', type=float, default=-float('inf'),
        help=('Don\'t save matches between a pair of images if less than '
              '--save-threshold were found.')
    )
    parser.add_argument(
        '--max-full-matrix', type=int, default=10000**2,
        help=('this is the biggest match matrix that will attempt to be '
              'computed allocated in memory. Matrices bigger than that will '
              'be split into chunks of at most this size. Reduce if your '
              'script runs out of memory.')
    )

    args = parser.parse_args()
    args.rt = args.rt if args.rt is not None else 1.

    MAX_FULL_MATRIX = args.max_full_matrix

    DEV   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Processing {args.h5_path} with DEV={DEV}')


class H5Store:
    def __init__(self, path, dtype=torch.float32):
        self.ds_file = h5py.File(os.path.join(path, 'descriptors.h5'), 'r')
        self.kp_file = h5py.File(os.path.join(path, 'keypoints.h5'), 'r')
        self.dtype   = dtype
    
    def keys(self):
        return list(self.ds_file.keys())

    def __getitem__(self, ix):
        descriptors = self.ds_file[ix][()]
        desc = torch.from_numpy(descriptors)
        if desc.dtype != self.dtype:
            warnings.warn(f'Type mismatch: converting {desc.dtype} to {self.dtype}')
            return desc.to(self.dtype)

        return desc

    def get_kp(self, ix):
        return self.kp_file[ix][()]

def pair_key(key_1, key_2):
    if key_2 > key_1:
        return key_1, key_2
    elif key_1 > key_2:
        return key_2, key_1
    else:
        raise ValueError(f'Equal keys {key_1}, {key_2}')

def _binary_to_index(binary_mask: ['N'], ix2: ['M']) -> [2, 'M']:
    return torch.stack([
        torch.nonzero(binary_mask, as_tuple=False)[:, 0],
        ix2
    ], dim=0)

def _ratio_one_way(dist_m: ['N', 'M'], rt) -> [2, 'K']:
    val, ix = torch.topk(dist_m, k=2, dim=1, largest=False)
    ratio = val[:, 0] / val[:, 1]
    passed_test = ratio < rt
    ix2 = ix[passed_test, 0]

    return _binary_to_index(passed_test, ix2)

def _match_chunkwise(ds1: ['N', 'F'], ds2: ['M', 'F'], rt) -> [2, 'K']:
    chunk_size = MAX_FULL_MATRIX // ds1.shape[0]
    matches = []
    start = 0

    while start < ds2.shape[0]:
        ds2_chunk = ds2[start:start+chunk_size]
        dist_m = distance_matrix(ds1, ds2_chunk)
        one_way = _ratio_one_way(dist_m, rt)
        one_way[1] += start
        matches.append(one_way)
        start += chunk_size

    return torch.cat(matches, dim=1)
    
def _match(ds1: ['N', 'F'], ds2: ['M', 'F'], rt) -> [2, 'K']:
    size = ds1.shape[0] * ds2.shape[0]

    fwd = _match_chunkwise(ds1, ds2, rt)
    bck = _match_chunkwise(ds2, ds1, rt)
    bck = torch.flip(bck, (0, ))

    # print(fwd.shape, fwd[:, :10])
    # print(bck.shape, bck[:, :10])
    merged = torch.cat([fwd, bck], dim=1)
    # print(merged.shape, merged[:, :10])
    if merged.shape[1] == 0:
        return torch.empty((2, 0), dtype=torch.int64)
    unique, counts = torch.unique(merged, dim=1, return_counts=True)
    # print(unique.shape, counts.shape, unique[:, :10], counts[:10])

    return unique[:, counts == 2]

def match(desc_1, desc_2, rt=1., u16=False):
    matched_pairs = _match(desc_1, desc_2, rt)
    matches = matched_pairs#.cpu().numpy()

    # if u16:
    #     matches = matches.astype(np.uint16)

    return matches

def brute_match(descriptors, hdf):
    keys = sorted(list(descriptors.keys()))

    n_total = (len(keys) * (len(keys) - 1)) // 2
    saved = 0
    pbar = tqdm(total=n_total)

    for i, key_1 in enumerate(keys):
        desc_1 = descriptors[key_1].to(DEV)
        group  = hdf.require_group(key_1)
        for key_2 in keys[i+1:]:
            if key_2 in group.keys():
                continue

            desc_2 = descriptors[key_2].to(DEV)
            
            try:
                matches = match(desc_1, desc_2, rt=args.rt, u16=args.u16)
                n = matches.shape[1]

                if n >= args.save_threshold:
                    group.create_dataset(key_2, data=matches)
                    saved += 1
            except RuntimeError:
                print('Error, skipping...')
                n = 0

            pbar.update(1)
            pbar.set_postfix(left=str(key_1), s=saved, n=n)

    pbar.close()

class MatcherWrapper:
    class InnerWrapper:
        def __init__(self):
            if args.rt is None:
                self._cycle_matcher = CycleMatcher()
            else:
                self._cycle_matcher = CycleRatioMatcher(args.rt)

        def raw_mle_match_pair(self, ds1: ['N', 'F'], ds2: ['M', 'F']) -> [2, 'K']:
            dist = distance_matrix(ds1, ds2, normalized=True)
            return self._cycle_matcher(dist)

    def __init__(self):
        self.matcher = MatcherWrapper.InnerWrapper()

SQRT_2 = 1.414213

def distance_matrix(fs1: ['N', 'F'], fs2: ['M', 'F']) -> ['N', 'M']:
    '''
    Assumes fs1 and fs2 are normalized!
    '''
    return SQRT_2 * (1. - fs1 @ fs2.T).clamp(min=1e-6).sqrt()

class Disk(nn.Module):
    default_config = {
        'detector': 'superpoint',
        'matrix': 'E'
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}
        self.rt = 1.0

    def forward(self, data):
        desc0, desc1 = data['descriptors0'], data['descriptors1']
        descriptors0 = desc0.squeeze(0).T
        descriptors1 = desc1.squeeze(0).T
        kpts0, kpts1 = data['keypoints0'], data['keypoints1']

        matches = match(descriptors0, descriptors1, rt=self.rt) # torch.Size([9034, 128]) torch.Size([11070, 128])
        # (2, 2644)
        # [[10   14   16   18   19   22   23   24   26   29]
        #  [172  155 3140 1387    4 1263  830 1897 2779   18]]

        indices0 = -(matches.new_ones(kpts0.shape[1]))
        indices0[matches[0,:]] = matches[1,:]
        indices0 = indices0.unsqueeze(0)
        mscores0 = torch.where(indices0 > 0, 1, 0).to('cuda:0')

        return {
            'matches0': indices0, # use -1 for invalid match
            'matches1': kpts1.new_full(kpts1.shape[:-1], -1, dtype=torch.int), # use -1 for invalid match
            'matching_scores0': mscores0,
            'matching_scores1': kpts1.new_zeros(kpts1.shape[:-1]),


        }

if __name__ == '__main__':
    dtype = torch.float16 if args.f16 else torch.float32
    described_samples = H5Store(args.h5_path, dtype=dtype)

    with h5py.File(os.path.join(args.h5_path, 'matches.h5'), 'a') as hdf:
        brute_match(described_samples, hdf)
