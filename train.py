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
from utils import DetectionHandler

"""
Start of main method
"""

configs = Config()
visualizer = Visualizer(configs)
detection_handler = DetectionHandler(configs)

"""
Initialize data loader
"""
train_set = Dataset(configs)

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

        detect_sess = tf.Session(graph=detection_graph)

    print(">  ====== Hand Inference graph loaded.")

#detection_graph, sess = detector_utils.load_inference_graph(configs.hand_detector_weights)

"""
Initialize hand pose estimator
"""

print("IMAGE TENSOR:")
print(detection_graph.get_tensor_by_name('image_tensor:0'))

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

pose_sess = tf.Session()

print("GRAPH:")
print(detection_graph)
print("GLOBAL VARIABLES:")
print(tf.global_variables())

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

file_list = ['data/videos/0.mp4', 'data/videos/1.mp4', 'data/videos/2.mp4']
with tf.device(tf_device):

    for idx, batch in enumerate(train_loader):
        tensor, label = batch
        
        data = tensor.numpy().squeeze()
        #data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB).squeeze()

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        file_name = str(idx) + '.avi'
        print(file_name)
        out = cv2.VideoWriter(os.path.join('output', file_name),fourcc, 10, (640, 480))#configs.video_fps, (640,480))
        
        for image in data:
            
            """
            Detect bounding boxes
            """
            print(image.shape)
            
            # perform detections, model does not expect a normalized image, normalization leads to worse results
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            boxes, scores = detector_utils.detect_objects(image, detection_graph, detect_sess)
            
            left_crop, right_crop, left_score, right_score = detection_handler.choose_next_candidates(boxes, scores)
           
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
             
            print("PREDICTION DONE")

            """
            Visualize output 
            """
            
            #print(stage_heatmap_np.shape)
            
            # visualize joints
            
            left_hand_img = visualizer.visualize_result(left_hand_img, stage_left_heatmap_np, None)
            right_hand_img = visualizer.visualize_result(right_hand_img, stage_right_heatmap_np, None)
            
            
            left_hand_img = cv2.resize(np.squeeze(left_hand_img), (320, 480))
            right_hand_img = cv2.resize(np.squeeze(right_hand_img), (320, 480))
            right_hand_img = cv2.flip(right_hand_img, 1)
            
            # combine left and right hands into one output image
            
            both_hand_img = np.concatenate((left_hand_img, right_hand_img), axis=1)
            
            # constants for plotting
            font = cv2.FONT_HERSHEY_SIMPLEX
            right_of_screen = (550,200)
            left_of_screen = (10, 200)
            font_scale = 1
            font_color = (255,0,0)
            line_type = 2
            
            # plot left-hand detection scores
            both_hand_img = cv2.putText(both_hand_img, str(left_score), 
            left_of_screen, 
            font, 
            font_scale,
            font_color,
            line_type)
            
            # plot right-hand detection scores
            both_hand_img = cv2.putText(both_hand_img, str(right_score), 
            right_of_screen, 
            font, 
            font_scale,
            font_color,
            line_type)

            out.write(both_hand_img.astype(np.uint8))

        out.release()

        #file_name = os.path.basename(file)
        #cv2.imwrite(os.path.join('./output', file_name), demo_img.astype(np.uint8))

#epochs = 1
#for epoch in range(epochs):
#    for batch in train_loader:
#        print(batch)
