import math
import numpy as np


# NMS code from https://github.com/rbgirshick/fast-rcnn/
# Girshick et al.

def nms(dets, scores, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    #scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def choose_best_bbox(prev_crop, new_crop):
    
    """
    Method for choosing between a new crop proposal and previous proposal
    based on the confidence score + distance from previous proposal
    """
    
    if type(prev_crop) == type(None):
        return new_crop
    
    prev_confidence = prev_crop['score'] * 0.8
    new_confidence = new_crop['score'] - (np.sum(np.abs(new_crop['center'] - prev_crop['center'])) * prev_crop['score'])
    
    if new_confidence > prev_confidence:
        return new_crop
    else:
        #prev_crop['score'] *= 0.99
        return prev_crop
    
def handle_detection_proposals(prev_left_crop, new_left_crop, prev_right_crop, new_right_crop):
    """
    Method for selecting the next crop proposals or to keep the current crops (or some combination of)
    selection is based on the pair with the highest confidence. Pair confidence is calculated via a mixture of decay for 
    older proposals, distance from previous proposals and distances between left and right hands
    """
    if type(prev_left_crop) == type(None):
        return (new_left_crop, new_right_crop)
        
    decay = 0.95
    between_hands_factor = 0.25 
    
    prev_left_crop['score'] *= decay 
    prev_right_crop['score'] *= decay
    
    confidence = [0,0,0,0]
    
    proposals = [
        (prev_left_crop, prev_right_crop),
        (prev_left_crop, new_right_crop),
        (new_left_crop, prev_right_crop),
        (new_left_crop, new_right_crop)
            
    ]
    
    def calculate_distance(crops):
        crop1, crop2 = crops
        return crop1['score'] + crop2['score'] + (between_hands_factor * np.linalg.norm(crop1['center'] - crop2['center']))
    
    # Confidence for Old, Old pair
    confidence[0] = calculate_distance(proposals[0])
    # Condifence for Old, New pair
    confidence[1] = calculate_distance(proposals[1]) 
    - (np.linalg.norm(prev_right_crop['center'] - new_right_crop['center']) * prev_right_crop['score'])
    # Confidence for New, Old pair
    confidence[2] = calculate_distance(proposals[2]) 
    - (np.linalg.norm(prev_left_crop['center'] - new_left_crop['center']) * prev_right_crop['score'])
    # Confidence for New, New pair 
    confidence[3] = calculate_distance(proposals[0])
    - (np.linalg.norm(prev_left_crop['center'] - new_left_crop['center']) * prev_right_crop['score'])
    - (np.linalg.norm(prev_right_crop['center'] - new_right_crop['center']) * prev_right_crop['score'])

    best_pair = np.argmax(confidence) 
                 
    return proposals[best_pair]

def choose_initial_candidates(boxes, scores):
    
    print(boxes.shape)
    
    keep = nms(boxes, scores, 0.35)
    
    boxes = boxes[keep]
    scores = scores[keep]
    print(boxes)
    print(scores)
    
    hand1 = boxes[0]
    hand1_center = np.array([(hand1[1] + hand1[3]) / 2, (hand1[0] + hand1[2]) / 2])
    
    width_centers = (boxes[:,1] + boxes[:,3]) / 2
    height_centers = (boxes[:,0] + boxes[:,2]) / 2
    centers = np.concatenate((np.expand_dims(width_centers, axis=1), np.expand_dims(height_centers, axis=1)), axis=1)
    print(centers.shape) 
    
    distances = np.linalg.norm(centers - hand1_center, axis=1) 
    
    print(distances.shape) 
    
    #scores = 0.5 * distances +  scores 
    
    hand2_idx = np.argmax(scores[1:]) + 1 
    print("hand 2 index {}".format(hand2_idx))
    
    hand1_crop = {'bbox': hand1, 'center': hand1_center, 'score': scores[0]}
    hand2_crop = {'bbox': boxes[hand2_idx], 'center': centers[hand2_idx], 'score': scores[hand2_idx]}
        
    if hand1_crop['center'][0] > hand2_crop['center'][0]:
        left_crop = hand2_crop 
        right_crop = hand1_crop 
       
        if hand2_crop['score'] < 0.1:
            left_hand = {'bbox': [0.0, 0.0, 1.0, 0.5], 'center': [0.25, 0.5], 'score': 0.5}
    else:
        left_crop = hand1_crop
        right_crop = hand2_crop 
        
        if hand2_crop['score'] < 0.1:
            right_hand = {'bbox': [0.0, 0.5, 1.0, 1.0], 'center': [0.75, 0.5], 'score': 0.5}
     
    return left_crop, right_crop 

def choose_next_candidates_nms(boxes, scores, left_crop, right_crop):
    
    if type(left_crop) == type(None):
        left_crop, right_crop = choose_initial_candidates(boxes, scores)
        return left_crop, right_crop
    
    #width_centers = (boxes[:,1] + boxes[:,3]) / 2
    #height_centers = (boxes[:,0] + boxes[:,2]) / 2
    #centers = np.concatenate((np.expand_dims(width_centers, axis=1), np.expand_dims(height_centers, axis=1)), axis=1)
    
    #left_distances = np.linalg.norm(centers - left_crop['center'], axis=1)
    #right_distances = np.linalg.norm(centers - right_crop['center'], axis=1) 
    
    keep = nms(boxes, scores, 0.35)
    
    boxes = boxes[keep]
    scores = scores[keep]
    
    
    """
    
    left_distances = np.mean(np.abs(boxes - left_crop['bbox']), axis=1)
    right_distances = np.mean(np.abs(boxes - right_crop['bbox']), axis=1) 
    
    left_scores = scores + (0.5 - left_distances)  
    right_scores = scores + (0.5 - right_distances) 
    
    new_left_idx = np.argmax(left_scores)
    new_right_idx = np.argmax(right_scores)
    
    if new_left_idx == new_right_idx:
        if left_scores[new_left_idx] > right_scores[new_right_idx]:
            right_scores[new_right_idx] = 0 
            new_right_idx = np.argmax(right_scores)
        else:
            left_scores[new_left_idx] = 0 
            new_left_idx = np.argmax(left_scores)
            
    """
    
    
    hand_1_center = np.array([(boxes[0][1] + boxes[0][3]) / 2, (boxes[0][0] + boxes[0][2]) / 2]) # (x, y)
    hand_2_center = np.array([(boxes[1][1] + boxes[1][3]) / 2, (boxes[1][0] + boxes[1][2]) / 2])
   
    # arange new proposals based on center points
    if hand_1_center[0] > hand_2_center[0]:
        new_left_crop = {'bbox': boxes[1], 'center': hand_2_center, 'score': scores[1]}
        new_right_crop = {'bbox': boxes[0], 'center': hand_1_center, 'score': scores[0]}
    else:
        new_left_crop = {'bbox': boxes[1], 'center': hand_2_center, 'score': scores[1]}
        new_right_crop = {'bbox': boxes[0], 'center': hand_1_center, 'score': scores[0]}
            

    
    """
    new_left_box = boxes[new_left_idx]
    new_left_center = np.array([(new_left_box[1] + new_left_box[3]) / 2, (new_left_box[0] + new_left_box[2]) / 2])
    new_left_crop = {'bbox': new_left_box, 'center': new_left_center, 'score': scores[new_left_idx]}
    
    new_right_box = boxes[new_right_idx]
    new_right_center = np.array([(new_right_box[1] + new_right_box[3]) / 2, (new_right_box[0] + new_right_box[2]) / 2])
    new_right_crop = {'bbox': new_right_box, 'center': new_right_center, 'score': scores[new_right_idx]}
    """
    

    
    left_crop = choose_best_bbox(left_crop, new_left_crop)
    
    right_crop = choose_best_bbox(right_crop, new_right_crop)
    """
    
    if new_left_crop['score'] > 0.6:
        left_crop = new_left_crop 
    if new_right_crop['score'] > 0.6:
        right_crop = new_right_crop
        
    """
    """
    
    if scores[new_left_idx] > 0.3:
        
        left_box = boxes[new_left_idx]
        left_center = np.array([(left_box[1] + left_box[3]) / 2, (left_box[0] + left_box[2]) / 2])
        left_crop = {'bbox': left_box, 'center': left_center, 'score': scores[new_left_idx]}
        
    if scores[new_right_idx] > 0.3:
        
        right_box = boxes[new_right_idx]
        right_center = np.array([(right_box[1] + right_box[3]) / 2, (right_box[0] + right_box[2]) / 2])
        right_crop = {'bbox': right_box, 'center': right_center, 'score': scores[new_right_idx]}
    
    """
    return left_crop, right_crop

def choose_next_candidates_mean(boxes, scores, left_crop, right_crop):
    
    if type(left_crop) == type(None):
        left_crop, right_crop = choose_initial_candidates(boxes, scores)
        return left_crop, right_crop
    
    left_distances = np.mean(np.abs(boxes - left_crop['bbox']), axis=1)
    right_distances = np.mean(np.abs(boxes - right_crop['bbox']), axis=1) 
    
    
    left_scores = scores + (0.5 - left_distances)  
    right_scores = scores + (0.5 - right_distances) 
    
    left_idxs = np.argsort(left_scores) 
    right_idxs = np.argsort(left_scores)
    
    left_box = np.mean(boxes[left_idxs[:10]], axis=0)
    left_score = np.mean(scores[left_idxs[:10]], axis=0)
    print(left_box.shape) 
    
    right_box = np.mean(boxes[right_idxs[:10]], axis=0)
    right_score = np.mean(scores[right_idxs[:10]], axis=0)

    
    left_center = np.array([(left_box[1] + left_box[3]) / 2, (left_box[0] + left_box[2]) / 2])
    left_crop = {'bbox': left_box, 'center': left_center, 'score': left_score}
        
    right_center = np.array([(right_box[1] + right_box[3]) / 2, (right_box[0] + right_box[2]) / 2])
    right_crop = {'bbox': right_box, 'center': right_center, 'score': right_score}
    
    
    return left_crop, right_crop

class DetectionHandler():
    
    def __init__(self, configs):
        self.threshold = configs.score_threshold
        self.buffer = configs.buffer
        self.left_box = [0.0, 0.0, 1.0, 0.5] # generic intialization for hands; split screen into halves
        self.right_box = [0.0, 0.5, 1.0, 1.0] 
        
        self.left_score = 0
        self.right_score = 0
        
    def get_center(self, box):
        
        return (box[1] + box[3]) / 2.0, (box[0] + box[2]) / 2.0 #(x,y)
    
    def choose_next_candidates(self, boxes, scores):

        if scores[1] < self.threshold:
            # if the best two proposals do not have sufficiently high scores, just return previous 
            # proposals
            return self.left_box, self.right_box, self.left_score, self.right_score
        
        x0_center, _ = self.get_center(boxes[0])
        x1_center, _ = self.get_center(boxes[1])

        # arange new proposals based on center points
        if x0_center > x1_center:
            self.left_box = boxes[1]
            self.right_box = boxes[0]
            
            self.left_score = scores[1]
            self.right_score = scores[0]
        else:
            self.left_box = boxes[0] 
            self.right_box = boxes[1]
            
            self.left_score = scores[0]
            self.right_score = scores[1]

        return self.left_box, self.right_box, self.left_score, self.right_score
    
    def crop_hand(self, image, box):

        """
        Method for generating a crop around a detection, given the image and the bounding box 
        assumes bounding-box is (top, left, bottom, right) between 0 and 1
        adds a buffer of pixels on each side of the image # TODO: handle edges 
        """
        x_center, y_center = self.get_center(box)

        H, W, _ = image.shape
        left = int(max((box[1] * W) - self.buffer, 0))
        right = int(min((box[3] * W) + self.buffer, W))

        top = int(max((box[0]* H) - self.buffer, 0)) 
        bottom = int(min((box[2] * H) + self.buffer, H))

        # Conver to square crop 
        width = right - left 

        #top = int(max((y_center - (width / 2)), 0))
        #bottom = int(max((y_center + (width / 2)), 0))

        #print(bounding_box)
        #print('({}, {}) ({}, {})'.format(top, bottom, left, right))

        return image[top:bottom, left:right]
