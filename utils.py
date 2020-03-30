import math
import numpy as np

def crop_hand(image, bounding_box, buffer=10):
    
    H, W, _ = image.shape
    left = int(max((bounding_box[1] * W) - buffer, 0))
    right = int(min((bounding_box[3] * W) + buffer, W))
    
    top = int(max((bounding_box[0]* H) - buffer, 0)) 
    bottom = int(min((bounding_box[2] * H) + buffer, H))
    
    print(bounding_box)
    print('({}, {}) ({}, {})'.format(top, bottom, left, right))

    return image[top:bottom, left:right]


def choose_best_bbox(prev_crop, new_crop):
    
    if type(prev_crop) == type(None):
        return new_crop
    
    prev_confidence = prev_crop['score'] * 0.8 
    new_confidence = new_crop['score'] - (np.sum(np.abs(new_crop['center'] - prev_crop['center'])) * prev_crop['score'])
    
    if new_confidence > prev_confidence:
        return new_crop
    else:
        return prev_crop