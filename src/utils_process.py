
from scipy.ndimage import gaussian_filter, maximum_filter
import cv2
import numpy as np

def post_process_heatmap(heatMap, kpConfidenceTh=0.2):
    kplst = list()
    for i in range(heatMap.shape[0]):
        _map = heatMap[i, :, :]
        _map = gaussian_filter(_map, sigma=1)
        _nmsPeaks = non_max_supression(_map, windowSize=3, threshold=1e-6)

        y, x = np.where(_nmsPeaks == _nmsPeaks.max())
        if len(x) > 0 and len(y) > 0:
            kplst.append((int(x[0]), int(y[0]), _nmsPeaks[y[0], x[0]]))
        else:
            kplst.append((0, 0, 0))

    kp = np.array(kplst)
    return kp


def non_max_supression(plain, windowSize=3, threshold=1e-6):
    # clear value less than threshold
    under_th_indices = plain < threshold
    plain[under_th_indices] = 0
    return plain * (plain == maximum_filter(plain, footprint=np.ones((windowSize, windowSize))))

def render_kps(cvmat, kps, scale_x, scale_y):
    for _kp in kps:
        _x, _y, _conf = _kp
        if _conf > 0.2:
            cv2.circle(cvmat, center=(int(_x*4*scale_x), int(_y*4*scale_y)), color=(0,0,255), radius=5)

    return cvmat
