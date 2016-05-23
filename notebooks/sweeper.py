import json
import numpy as np


def loadLog(filename):
    s = []
    for line in open(filename):
        r = line.find('json_stats')
        if r > -1:
            s.append(json.loads(line[r+12:]))
    return s


def findSweepParams(frames):
    def findConstants(frame):
        keys = dict()
        for key in frame.keys():
            v = np.asarray(frame[key])
            u = np.copy(v)
            u.fill(v[0])
            if np.array_equal(v, u):
                keys[key] = v[0]
        return keys
    changing = dict()
    for frame in frames:
        for k, v in findConstants(frame).items():
            if isinstance(v, list):
                v = json.dumps(v)
            if k not in changing:
                changing[k] = {v}
            else:
                changing[k].add(v)
    all_keys = []
    for k, v in changing.items():
        if len(v) > 1:
            all_keys.append(k)
    return sorted(all_keys)


def generateLegend(frame, sweeps):
    s = ''
    for key in sweeps:
        if key not in frame:
            s = s + key + '=not present, '
        else:
            s = s + key + '=' + str(frame[key][0]) + ', '
    return s

def generateLegends(frames):
    params = findSweepParams(frames)
    return [generateLegend(frame, params) for frame in frames]
