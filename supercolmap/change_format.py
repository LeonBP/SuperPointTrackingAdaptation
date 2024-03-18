import numpy as np

def load_matches(path):
    npz = np.load(path)
    matches_old = npz['matches']
    matches_new = []

    for i in range(len(matches_old)):
        if matches_old[i] > -1:
            matches_new.append([i, matches_old[i]])

    matches_new=np.array(matches_new)
    return matches_new

def load_keypoints0(path):
    npz=np.load(path)
    keypoints0=npz['keypoints0']
    return keypoints0

def load_keypoints1(path):
    npz=np.load(path)
    keypoints1=npz['keypoints1']
    return keypoints1

def load_kps_desc0(path):
    npz=np.load(path)
    keypoints0=npz['keypoints0']
    descriptors0=npz['descriptors0']
    return keypoints0, descriptors0

def load_kps_desc1(path):
    npz=np.load(path)
    keypoints1=npz['keypoints1']
    descriptors1=npz['descriptors1']
    return keypoints1, descriptors1

if __name__ == '__main__':
    path='/Users/mjimenez/pythonProject8/dump_match_pairs/0001_0001_2_matches.npz'
    npz=np.load(path)
    keypoints0=npz['keypoints0']
    keypoints1 = npz['keypoints1']
    matches_old= npz['matches']

    matches_new=[]

    for i in range(len(matches_old)):
        if matches_old[i]>-1:
            matches_new.append([i,matches_old[i]])

    print(matches_old)
    print(matches_new)

