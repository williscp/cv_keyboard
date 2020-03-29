import cv2
import time
import math
import numpy as np 

class Visualizer():
    
    def __init__(self, configs):
        self.limb_model = configs.limb_model
        self.joints = configs.joints 
        self.joint_color_code = configs.joint_color_code 
        
        self.hmap_size = configs.hmap_size
        self.cmap_size = configs.cmap_size

    def visualize_result(self, test_img, stage_heatmap_np, kalman_filter_array):
        
        """
        Modified code from convolutional-pose-machines-tensorflow: 
        
            https://github.com/timctho/convolutional-pose-machines-tensorflow
            
        """
                
        #t1 = time.time()
        demo_stage_heatmaps = []
        #if FLAGS.DEMO_TYPE == 'MULTI':
        #    for stage in range(len(stage_heatmap_np)):
        #        demo_stage_heatmap = stage_heatmap_np[stage][0, :, :, 0:FLAGS.joints].reshape(
        #            (FLAGS.hmap_size, FLAGS.hmap_size, FLAGS.joints))
        #        demo_stage_heatmap = cv2.resize(demo_stage_heatmap, (test_img.shape[1], test_img.shape[0]))
        #        demo_stage_heatmap = np.amax(demo_stage_heatmap, axis=2)
        #        demo_stage_heatmap = np.reshape(demo_stage_heatmap, (test_img.shape[1], test_img.shape[0], 1))
        #        demo_stage_heatmap = np.repeat(demo_stage_heatmap, 3, axis=2)
        #        demo_stage_heatmap *= 255
        #        demo_stage_heatmaps.append(demo_stage_heatmap)
    #
    #        last_heatmap = stage_heatmap_np[len(stage_heatmap_np) - 1][0, :, :, 0:FLAGS.joints].reshape(
    #            (FLAGS.hmap_size, FLAGS.hmap_size, FLAGS.joints))
    #        last_heatmap = cv2.resize(last_heatmap, (test_img.shape[1], test_img.shape[0]))
    #    else:
        last_heatmap = stage_heatmap_np[len(stage_heatmap_np) - 1][0, :, :, 0:self.joints].reshape(
             (self.hmap_size, self.hmap_size, self.joints))
        last_heatmap = cv2.resize(last_heatmap, (test_img.shape[1], test_img.shape[0]))
        #print('hm resize time %f' % (time.time() - t1))

        #t1 = time.time()
        joint_coord_set = np.zeros((self.joints, 2))

        # Plot joint colors
        if kalman_filter_array is not None:
            for joint_num in range(self.joints):
                joint_coord = np.unravel_index(np.argmax(last_heatmap[:, :, joint_num]),
                                               (test_img.shape[0], test_img.shape[1]))
                joint_coord = np.array(joint_coord).reshape((2, 1)).astype(np.float32)
                kalman_filter_array[joint_num].correct(joint_coord)
                kalman_pred = kalman_filter_array[joint_num].predict()
                joint_coord_set[joint_num, :] = np.array([kalman_pred[0], kalman_pred[1]]).reshape((2))

                color_code_num = (joint_num // 4)
                if joint_num in [0, 4, 8, 12, 16]:
                    joint_color = list(map(lambda x: x + 35 * (joint_num % 4), self.joint_color_code[color_code_num]))
                    cv2.circle(test_img, center=(joint_coord[1], joint_coord[0]), radius=3, color=joint_color, thickness=-1)
                else:
                    joint_color = list(map(lambda x: x + 35 * (joint_num % 4), self.joint_color_code[color_code_num]))
                    cv2.circle(test_img, center=(joint_coord[1], joint_coord[0]), radius=3, color=joint_color, thickness=-1)
        else:
            for joint_num in range(self.joints):
                joint_coord = np.unravel_index(np.argmax(last_heatmap[:, :, joint_num]),
                                               (test_img.shape[0], test_img.shape[1]))
                joint_coord_set[joint_num, :] = [joint_coord[0], joint_coord[1]]

                color_code_num = (joint_num // 4)
                if joint_num in [0, 4, 8, 12, 16]:
                    joint_color = list(map(lambda x: x + 35 * (joint_num % 4), self.joint_color_code[color_code_num]))
                    cv2.circle(test_img, center=(joint_coord[1], joint_coord[0]), radius=3, color=joint_color, thickness=-1)
                else:
                    joint_color = list(map(lambda x: x + 35 * (joint_num % 4), self.joint_color_code[color_code_num]))
                    cv2.circle(test_img, center=(joint_coord[1], joint_coord[0]), radius=3, color=joint_color, thickness=-1)
        #print('plot joint time %f' % (time.time() - t1))

        #t1 = time.time()
        # Plot limb colors
        for limb_num in range(len(self.limb_model)):

            x1 = joint_coord_set[self.limb_model[limb_num][0], 0]
            y1 = joint_coord_set[self.limb_model[limb_num][0], 1]
            x2 = joint_coord_set[self.limb_model[limb_num][1], 0]
            y2 = joint_coord_set[self.limb_model[limb_num][1], 1]
            length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            if length < 150 and length > 5:
                deg = math.degrees(math.atan2(x1 - x2, y1 - y2))
                polygon = cv2.ellipse2Poly((int((y1 + y2) / 2), int((x1 + x2) / 2)),
                                           (int(length / 2), 3),
                                           int(deg),
                                           0, 360, 1)
                color_code_num = limb_num // 4
                limb_color = list(map(lambda x: x + 35 * (limb_num % 4), self.joint_color_code[color_code_num]))

                cv2.fillConvexPoly(test_img, polygon, color=limb_color)
        #print('plot limb time %f' % (time.time() - t1))

    #    if FLAGS.DEMO_TYPE == 'MULTI':
    #        upper_img = np.concatenate((demo_stage_heatmaps[0], demo_stage_heatmaps[1], demo_stage_heatmaps[2]), axis=1)
    #        lower_img = np.concatenate((demo_stage_heatmaps[3], demo_stage_heatmaps[len(stage_heatmap_np) - 1], test_img),
    #                                   axis=1)
    #        demo_img = np.concatenate((upper_img, lower_img), axis=0)
    #        return demo_img
    #    else:
        return test_img