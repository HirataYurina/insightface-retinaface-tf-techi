# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:utils.py
# software: PyCharm

import numpy as np

STEPS = [8, 16, 32]
ANCHOR_SHAPE = [[16, 32],
                [64, 128],
                [256, 512]]


def delta2box(delta, anchors):
    """delta to boxes
    center_x = tx * 0.1 * anchor_w + anchor_x
    center_y = ty * 0.1 * anchor_h + anchor_y
    w = exp(tw * 0.2) * anchor_w
    y = exp(ty * 0.2) * anchor_y

    Args:
        delta:   (batch, num_anchors, 4)
        anchors: (num_anchors, 4) (cx, cy, w, h)

    Returns:
        bboxes

    """
    center_x = delta[..., 0] * 0.1 * anchors[..., 2] + anchors[..., 0]
    center_y = delta[..., 1] * 0.1 * anchors[..., 3] + anchors[..., 1]

    w = np.exp(delta[..., 2] * 0.2) * anchors[..., 2]
    h = np.exp(delta[..., 3] * 0.2) * anchors[..., 3]

    tx = center_x - w / 2
    ty = center_y - h / 2
    bx = center_x + w / 2
    by = center_y + h / 2

    bboxes = np.stack([tx, ty, bx, by], axis=-1)

    # clip boxes to (0, 1)
    bboxes = np.clip(bboxes, a_min=0.0, a_max=1.0)

    return bboxes


def delta2landmarks(landmarks, anchors):
    """ delta to landmarks
    landmarks_x = delta_x * anchor_w * 0.1 + anchor_x
    landmarks_y = delta_y * anchor_h * 0.1 + anchor_h

    Args:
        landmarks: (num, 10)
        anchors:   (num, 4)

    Returns:
        landmarks

    """
    landmarks = np.reshape(landmarks, newshape=(-1, 5, 2))  # (num, 5, 2)
    x = landmarks[..., 0] * np.tile(anchors[..., 2:3], reps=[1, 5]) * 0.1 + \
        np.tile(anchors[..., 0:1], reps=[1, 5])  # (num, 5)
    y = landmarks[..., 1] * np.tile(anchors[..., 3:4], reps=[1, 5]) * 0.1 + \
        np.tile(anchors[..., 1:2], reps=[1, 5])
    return np.concatenate([x, y], axis=-1)


def generate_anchors(height, width):
    hw = np.array([height, width])
    hw = [np.ceil(hw / step) for step in STEPS]

    anchors = []

    for i, size in enumerate(hw):

        shapes = ANCHOR_SHAPE[i]

        h = size[0]
        w = size[1]
        x, y = np.meshgrid(np.arange(w), np.arange(h))  # (h, w)
        x = x.flatten()
        y = y.flatten()

        for j in range(len(x)):
            for shape in shapes:
                # normalize
                cx = (x[j] + 0.5) / w
                cy = (y[j] + 0.5) / h
                aw = shape / width
                ah = shape / height
                anchors.append([cx, cy, aw, ah])

    anchors = np.array(anchors)

    return anchors


def nms(scores, bboxes):
    pass


if __name__ == '__main__':
    anchors_ = generate_anchors(416, 416)
    shape_ = anchors_.shape
    print(shape_)
    print(anchors_)
