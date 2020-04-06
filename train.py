import torch
import os
import numpy as np
import tensorflow as tf
import cv2
import time
import math

from config import Config
from dataset import Dataset
from convolutional_pose_machines_tensorflow.models.nets import cpm_hand_slim
from convolutional_pose_machines_tensorflow.utils import cpm_utils
from handtracking.utils import detector_utils
from visualize import Visualizer
from utils import DetectionHandler, get_global_pose

"""
Start of main method
"""

configs = Config()
visualizer = None
if configs.visualize_cropped_output or configs.visualize_full_output or configs.visualize_joint_positions or configs.visualize_stage_heatmaps:
    visualizer = Visualizer(configs)
detection_handler = DetectionHandler(configs)

"""
Initialize data loader
"""
train_set = Dataset(configs, label_path='labels/train.csv')

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=1,
    num_workers=0,
    pin_memory=False,
    shuffle=True,
    drop_last=True
)

"""
Initialize hand detector
"""

tf_device = '/gpu:0'

# load frozen tensorflow model into memory
print("> ====== loading HAND frozen graph into memory")


with tf.device(tf_device):
    detection_graph = tf.Graph()
    with detection_graph.as_default():

        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(configs.hand_detector_weights, 'rb') as file:

            serialized_graph = file.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
            
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.per_process_gpu_memory_fraction = configs.detector_gpu_alloc
        detect_sess = tf.Session(graph=detection_graph, config=tf_config)

    print(">  ====== Hand Inference graph loaded.")

#detection_graph, sess = detector_utils.load_inference_graph(configs.hand_detector_weights)

"""
Initialize hand pose estimator
"""

with tf.device(tf_device):
    """Build graph"""

    input_data = tf.placeholder(
        dtype=tf.float32,
        shape=[None, configs.input_size, configs.input_size, 3],
        name='input_image'
    )
    center_map = tf.placeholder(
        dtype=tf.float32,
        shape=[None, configs.input_size, configs.input_size, 1],
        name='center_map'
    )

    model = cpm_hand_slim.CPM_Model(6, 22) # 6 cpm stages, 21 + 1 joints
    model.build_model(input_data, center_map, 1)

    
tf_config = tf.ConfigProto()
tf_config.gpu_options.per_process_gpu_memory_fraction = configs.estimator_gpu_alloc
pose_sess = tf.Session(config=tf_config)

#print("GRAPH:")
#print(detection_graph)
#print("GLOBAL VARIABLES:")
#print(tf.global_variables())

pose_sess.run(tf.global_variables_initializer())
#print(pose_sess.graph)

model.load_weights_from_file(configs.hand_pose_estimator_weights, pose_sess, False)

#print(pose_sess.graph.get_tensor_by_name('image_tensor:0'))

test_center_map = cpm_utils.gaussian_img(
    configs.input_size,
    configs.input_size,
    configs.input_size / 2,
    configs.input_size / 2,
    21 #cmap_radius
)

test_center_map = np.reshape(
    test_center_map,
    [1, configs.input_size, configs.input_size, 1]
)

with tf.device(tf_device):

    for idx, batch in enumerate(train_loader):
        data, label = batch
        
        video_id, tensor = data 
        video_id = video_id.item()
        
        video = tensor.numpy().squeeze()
        #data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB).squeeze()
        
        print("Video {}".format(video_id))
        
        if visualizer: 
            
            visualizer.start_capture(video_id)
            
        for image in video:
            
            """
            Detect bounding boxes
            """
            # perform detections, model does not expect a normalized image, normalization leads to worse results
            
            # convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            boxes, scores = detector_utils.detect_objects(image, detection_graph, detect_sess)
            
            left_crop, right_crop, left_score, right_score = detection_handler.choose_next_candidates(image, boxes, scores)
           
            # generate crops
            left_hand_img = detection_handler.crop_hand(image, left_crop)
            right_hand_img = detection_handler.crop_hand(image, right_crop)
            
            # convert to BGR
            left_hand_img = cv2.cvtColor(left_hand_img, cv2.COLOR_RGB2BGR)
            right_hand_img = cv2.cvtColor(right_hand_img, cv2.COLOR_RGB2BGR)
            
            """
            Pose estimation
            """
            
            # perform estimation on left hand
            left_hand_img = cv2.resize(left_hand_img, (configs.input_size, configs.input_size))
            
            # normalize image
            left_hand_input = left_hand_img / 256.0 - 0.5
            left_hand_input = np.expand_dims(left_hand_input, axis=0)
            
            # predict
            predict_left_heatmap, stage_left_heatmap_np = pose_sess.run(
                [model.current_heatmap, model.stage_heatmap],
                feed_dict = {
                    'input_image:0': left_hand_input,
                    'center_map:0': test_center_map
                }
            )
            
            # perform estimation on right hand
            right_hand_img = cv2.resize(right_hand_img, (configs.input_size, configs.input_size))
            right_hand_img = cv2.flip(right_hand_img, 1)
            
            # normalize image
            right_hand_input= right_hand_img / 256.0 - 0.5
            # a bit of a hack, flip the orientation so the left hand looks like a right hand
            # model seems to be trained on right hand only 
            right_hand_input = np.expand_dims(right_hand_input, axis=0)
            
            # predict
            predict_right_heatmap, stage_right_heatmap_np = pose_sess.run(
                [model.current_heatmap, model.stage_heatmap],
                feed_dict = {
                    'input_image:0': right_hand_input,
                    'center_map:0': test_center_map
                }
            )
                        
            """
            Update visualizer 
            """
            if visualizer:
                
                left_data = (left_hand_img, left_crop, stage_left_heatmap_np) 
                right_data = (right_hand_img, right_crop, stage_right_heatmap_np)
                visualizer.update_capture(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), left_data, right_data)
        
        if visualizer:
            
            visualizer.end_capture(video_id)

#epochs = 1
#for epoch in range(epochs):
#    for batch in train_loader:
#        print(batch)
