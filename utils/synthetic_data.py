import numpy as np

def generate_synthetic_normal(locs, sizes, scale=1.0):
    assert len(locs) == len(sizes)
    streams = []
    for loc, size in zip(locs, sizes):
        streams.append(np.random.normal(loc=loc, scale=scale, size=size))
    return np.concatenate(streams)
