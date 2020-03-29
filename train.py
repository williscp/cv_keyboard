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
from visualize import Visualizer

"""
Start of main method
"""

configs = Config()
visualizer = Visualizer(configs)

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
Initialize hand pose detector
"""

tf_device = '/gpu:0'
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

sess = tf.Session()

sess.run(tf.global_variables_initializer())
model.load_weights_from_file(configs.hand_model_path, sess, False)

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
    #print(batch)
    #for file in file_list:
        #test_img = cpm_utils.read_image(file, [], configs.input_size, 'VIDEO')
        data = tensor.numpy().squeeze()

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        file_name = str(idx) + '.avi'
        print(file_name)
        out = cv2.VideoWriter(os.path.join('output', file_name),fourcc, configs.video_fps, (640,480))


        for image in data:

            print(image.shape)
            test_img_resize = cv2.resize(image, (configs.input_size, configs.input_size))

            test_img_input = test_img_resize / 256.0 - 0.5
            test_img_input = np.expand_dims(test_img_input, axis=0)

            predict_heatmap, stage_heatmap_np = sess.run(
                [model.current_heatmap, model.stage_heatmap],
                feed_dict = {
                    'input_image:0': test_img_input,
                    'center_map:0': test_center_map
                }
            )

            print("PREDICTION DONE")
            #print(stage_heatmap_np.shape)

            demo_img = visualizer.visualize_result(image, stage_heatmap_np, None)

            out.write(demo_img.astype(np.uint8))

        out.release()

        #file_name = os.path.basename(file)
        #cv2.imwrite(os.path.join('./output', file_name), demo_img.astype(np.uint8))

#epochs = 1
#for epoch in range(epochs):
#    for batch in train_loader:
#        print(batch)
