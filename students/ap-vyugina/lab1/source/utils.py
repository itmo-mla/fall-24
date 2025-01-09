import numpy as np
from scipy.spatial.distance import cdist

def intercluster_distance(dataset, lbls):
    d = []
    for i1 in np.unique(lbls):
        for i2 in np.unique(lbls):
            if i2 > i1:
                cl_1 = dataset[lbls==i1]
                cl_2 = dataset[lbls==i2]
                d += [np.sum(cdist(cl_1, cl_2)) * len(cl_1) * len(cl_2) / (len(cl_1) + len(cl_2))]
    return np.mean(d)


def intracluster_distance(dataset, lbls):
    d = []
    for l in np.unique(lbls):
        mean = np.mean(dataset[lbls==l], axis=0).reshape(1, -1)
        d += [np.sum(cdist(dataset[lbls==l], mean))]
    return np.mean(d)
    