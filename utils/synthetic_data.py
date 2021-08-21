import numpy as np

def generate_synthetic_normal(locs, sizes, scale=1.0, dims=1):
    assert len(locs) == len(sizes)
    if type(scale) == list and len(scale) != len(sizes):
        raise ValueError("scale and sizes lists must have equal length.")
    elif type(scale) in (int, float):
        scale = [scale]*len(sizes)
    streams = []
    for loc, scal, size in zip(locs, scale, sizes):
        streams.append(np.random.normal(loc=loc, scale=scal, size=(size, dims)))
    return np.concatenate(streams)
