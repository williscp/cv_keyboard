import torch
import os
import numpy as np
import tensorflow as tf

from config import Config
from dataset import Dataset
from convolutional-pose-machines-tensorflow.models.nets import cpm_hand_slim
from convolutional-pose-machines-tensorflow.utils import cpm_utils

configs = Config()

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

saver = tf.Session()

sess.run(tf.global_variables_initializer())
model = load_weights_from_file(configs.model_path, sess, False)

test_center_map = cpm_utils.gaussian_img(
    configs.input_size,
    configs.input_size,
    configs.input_size / 2,
    configs.input_size / 2,
    configs.cmap_radius
)

test_center_map = np.reshape(
    test_center_map,
    [1, FLAGS.input_size, FLAGS.input_size, 1]
)

file_list = ['data/videos/0.mp4', 'data/videos/1.mp4', 'data/videos/2.mp4']
with tf.device(tf_device):
    for file in file_list:
        test_img = cpm_utils.read_image(file_path, [], configs.input_size, 'VIDEO')

        test_img_resize = cv2.resize(test_img, (configs.input_size, configs.input_size))

        test_img_input = test_img_resize / 256.0 - 0.5
        test_img_input = np.expand_dims(test_img_input, axis=0)

        predict_heatmap, stage_heatmap_np = sess.run(
            [model.current_heatmap, model.stage_heatmap],
            feed_disct = {
                'input_image:0': test_img_input,
                'center_map:0': test_center_map
            }
        )

        demo_img = visualize_result(test_img, stage_heatmap_np, None)

        file_name = os.path.basename(file)
        cv2.imwrite(os.path.join('./output', file_name), demo_img.astype(np.uint8))

epochs = 1
for epoch in range(epochs):
    for batch in train_loader:
        print(batch)

def visualize_result(test_img, stage_heatmap_np, kalman_filter_array):

    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_string('DEMO_TYPE',
                               default_value='image_list',
                               # default_value='SINGLE',
                               docstring='MULTI: show multiple stage,'
                                         'SINGLE: only last stage,'
                                         'HM: show last stage heatmap,'
                                         'paths to .jpg or .png image')
    tf.app.flags.DEFINE_string('model_path',
                               default_value='models/weights/cpm_hand.pkl',
                               docstring='Your model')
    tf.app.flags.DEFINE_integer('input_size',
                                default_value=368,
                                docstring='Input image size')
    tf.app.flags.DEFINE_integer('hmap_size',
                                default_value=46,
                                docstring='Output heatmap size')
    tf.app.flags.DEFINE_integer('cmap_radius',
                                default_value=21,
                                docstring='Center map gaussian variance')
    tf.app.flags.DEFINE_integer('joints',
                                default_value=21,
                                docstring='Number of joints')
    tf.app.flags.DEFINE_integer('stages',
                                default_value=6,
                                docstring='How many CPM stages')
    tf.app.flags.DEFINE_integer('cam_num',
                                default_value=0,
                                docstring='Webcam device number')
    tf.app.flags.DEFINE_bool('KALMAN_ON',
                             default_value=False,
                             docstring='enalbe kalman filter')
    tf.app.flags.DEFINE_float('kalman_noise',
                                default_value=3e-2,
                                docstring='Kalman filter noise value')
    tf.app.flags.DEFINE_string('color_channel',
                               default_value='RGB',
                               docstring='')
    t1 = time.time()
    demo_stage_heatmaps = []
    if FLAGS.DEMO_TYPE == 'MULTI':
        for stage in range(len(stage_heatmap_np)):
            demo_stage_heatmap = stage_heatmap_np[stage][0, :, :, 0:FLAGS.joints].reshape(
                (FLAGS.hmap_size, FLAGS.hmap_size, FLAGS.joints))
            demo_stage_heatmap = cv2.resize(demo_stage_heatmap, (test_img.shape[1], test_img.shape[0]))
            demo_stage_heatmap = np.amax(demo_stage_heatmap, axis=2)
            demo_stage_heatmap = np.reshape(demo_stage_heatmap, (test_img.shape[1], test_img.shape[0], 1))
            demo_stage_heatmap = np.repeat(demo_stage_heatmap, 3, axis=2)
            demo_stage_heatmap *= 255
            demo_stage_heatmaps.append(demo_stage_heatmap)

        last_heatmap = stage_heatmap_np[len(stage_heatmap_np) - 1][0, :, :, 0:FLAGS.joints].reshape(
            (FLAGS.hmap_size, FLAGS.hmap_size, FLAGS.joints))
        last_heatmap = cv2.resize(last_heatmap, (test_img.shape[1], test_img.shape[0]))
    else:
        last_heatmap = stage_heatmap_np[len(stage_heatmap_np) - 1][0, :, :, 0:FLAGS.joints].reshape(
            (FLAGS.hmap_size, FLAGS.hmap_size, FLAGS.joints))
        last_heatmap = cv2.resize(last_heatmap, (test_img.shape[1], test_img.shape[0]))
    print('hm resize time %f' % (time.time() - t1))

    t1 = time.time()
    joint_coord_set = np.zeros((FLAGS.joints, 2))

    # Plot joint colors
    if kalman_filter_array is not None:
        for joint_num in range(FLAGS.joints):
            joint_coord = np.unravel_index(np.argmax(last_heatmap[:, :, joint_num]),
                                           (test_img.shape[0], test_img.shape[1]))
            joint_coord = np.array(joint_coord).reshape((2, 1)).astype(np.float32)
            kalman_filter_array[joint_num].correct(joint_coord)
            kalman_pred = kalman_filter_array[joint_num].predict()
            joint_coord_set[joint_num, :] = np.array([kalman_pred[0], kalman_pred[1]]).reshape((2))

            color_code_num = (joint_num // 4)
            if joint_num in [0, 4, 8, 12, 16]:
                if PYTHON_VERSION == 3:
                    joint_color = list(map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num]))
                else:
                    joint_color = map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num])

                cv2.circle(test_img, center=(joint_coord[1], joint_coord[0]), radius=3, color=joint_color, thickness=-1)
            else:
                if PYTHON_VERSION == 3:
                    joint_color = list(map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num]))
                else:
                    joint_color = map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num])

                cv2.circle(test_img, center=(joint_coord[1], joint_coord[0]), radius=3, color=joint_color, thickness=-1)
    else:
        for joint_num in range(FLAGS.joints):
            joint_coord = np.unravel_index(np.argmax(last_heatmap[:, :, joint_num]),
                                           (test_img.shape[0], test_img.shape[1]))
            joint_coord_set[joint_num, :] = [joint_coord[0], joint_coord[1]]

            color_code_num = (joint_num // 4)
            if joint_num in [0, 4, 8, 12, 16]:
                if PYTHON_VERSION == 3:
                    joint_color = list(map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num]))
                else:
                    joint_color = map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num])

                cv2.circle(test_img, center=(joint_coord[1], joint_coord[0]), radius=3, color=joint_color, thickness=-1)
            else:
                if PYTHON_VERSION == 3:
                    joint_color = list(map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num]))
                else:
                    joint_color = map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num])

                cv2.circle(test_img, center=(joint_coord[1], joint_coord[0]), radius=3, color=joint_color, thickness=-1)
    print('plot joint time %f' % (time.time() - t1))

    t1 = time.time()
    # Plot limb colors
    for limb_num in range(len(limbs)):

        x1 = joint_coord_set[limbs[limb_num][0], 0]
        y1 = joint_coord_set[limbs[limb_num][0], 1]
        x2 = joint_coord_set[limbs[limb_num][1], 0]
        y2 = joint_coord_set[limbs[limb_num][1], 1]
        length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        if length < 150 and length > 5:
            deg = math.degrees(math.atan2(x1 - x2, y1 - y2))
            polygon = cv2.ellipse2Poly((int((y1 + y2) / 2), int((x1 + x2) / 2)),
                                       (int(length / 2), 3),
                                       int(deg),
                                       0, 360, 1)
            color_code_num = limb_num // 4
            if PYTHON_VERSION == 3:
                limb_color = list(map(lambda x: x + 35 * (limb_num % 4), joint_color_code[color_code_num]))
            else:
                limb_color = map(lambda x: x + 35 * (limb_num % 4), joint_color_code[color_code_num])

            cv2.fillConvexPoly(test_img, polygon, color=limb_color)
    print('plot limb time %f' % (time.time() - t1))

    if FLAGS.DEMO_TYPE == 'MULTI':
        upper_img = np.concatenate((demo_stage_heatmaps[0], demo_stage_heatmaps[1], demo_stage_heatmaps[2]), axis=1)
        lower_img = np.concatenate((demo_stage_heatmaps[3], demo_stage_heatmaps[len(stage_heatmap_np) - 1], test_img),
                                   axis=1)
        demo_img = np.concatenate((upper_img, lower_img), axis=0)
        return demo_img
    else:
        return test_img
