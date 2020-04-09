import cv2
import time
import math
import numpy as np 
import torch
import os

from utils import get_global_pose
from dataset import CHAR_TO_CLASS
# constants for plotting
FONT = cv2.FONT_HERSHEY_SIMPLEX
RIGHT_OF_SCREEN = (550,200)
LEFT_OF_SCREEN = (10,200)
FONT_SCALE = 1
FONT_COLOR = (255,0,0)
LINE_TYPE = 2


def integral_heatmap_layer(heatmap):

    N, K, H, W,  = heatmap.shape
    
    # apply softmax:
    
    heatmap = heatmap.reshape(N, K, -1)
    #print(torch.argmax(heatmap, dim=2))
    probmap = torch.nn.functional.softmax(heatmap, dim=2)
    #print(torch.argmax(heatmap, dim=2))
    
    #print(torch.max(heatmap, dim=2))
    #print(torch.min(heatmap, dim=2))
    h_norm = probmap.reshape(N, K, H, W)

    # generate the integrals 

    x_linspace = torch.linspace(0, 1, W).repeat(N, K, H, 1)#.to("cuda")
    y_linspace = torch.linspace(0, 1, H).repeat(N, K, W, 1).permute(0,1,3,2)#.to("cuda")

    x_weights = x_linspace * h_norm 
    y_weights = y_linspace * h_norm 

    x_positions = torch.sum(torch.sum(x_weights, dim=3), dim=2).unsqueeze(-1)
    y_positions = torch.sum(torch.sum(y_weights, dim=3), dim=2).unsqueeze(-1)
    
    #print(x_positions)
    pose = torch.cat((x_positions, y_positions), dim=2)
    
    return pose

def visualize_predictions(input_path, out_path, preds, ground_truth, timestamps):
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottom_of_screen = (10,400)
    near_bottom_of_screen = (10,350)
    left_of_screen = (10, 200)
    font_scale = 3
    font_color = (255,255,255)
    display_color = (255,0,0)
    error_color = (0,0,255)
    line_type = 2
    
    cap = cv2.VideoCapture(input_path)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = 15
    
    out = cv2.VideoWriter(out_path,fourcc, fps, (640,480))

    timestamps = timestamps * fps

    frame_num = 0
    last_key = 0
    cur_gt = ''
    cur_preds = ''
    
    class_to_char = list(CHAR_TO_CLASS.keys())
        
    while(cap.isOpened()):
        ret, frame = cap.read()

        frame_num += 1

        if not ret:
            break

        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #frame = cv2.flip(frame,0)
        if (last_key < len(timestamps) - 1 and frame_num > timestamps[last_key + 1]) or last_key == 0:
            last_key += 1
            
            gt_char = class_to_char[int(ground_truth[last_key])].upper()
            pred_char = class_to_char[int(preds[last_key])].upper()
            
            if gt_char == pred_char:
                color = display_color 
            else: 
                color = error_color
                
            if gt_char != 'NONE':
                cur_gt = cur_gt + gt_char
            if pred_char != 'NONE':
                cur_preds = cur_preds + pred_char
                                
            if pred_char == ' ':
                pred_char = 'space'

        frame = cv2.putText(frame, pred_char, 
            left_of_screen, 
            font, 
            font_scale,
            color,
            line_type)
            
        frame = cv2.putText(frame, cur_preds, 
            near_bottom_of_screen, 
            font, 
            font_scale/4,
            font_color,
            line_type)
        
        frame = cv2.putText(frame, cur_gt, 
            bottom_of_screen, 
            font, 
            font_scale/4,
            font_color,
            line_type)
            

        #cv2.imshow('frame',frame)

        out.write(frame)


    cap.release()
    out.release()

class Visualizer():
    
    def __init__(self, configs):
        
        self.video_fps = configs.video_fps
        
        self.limb_model = configs.limb_model
        self.joints = configs.joints 
        self.joint_color_code = configs.joint_color_code
        
        self.hmap_size = configs.hmap_size
        self.cmap_size = configs.cmap_size
        self.input_size = configs.input_size
        
        self.visualize_cropped_output = configs.visualize_cropped_output
        self.visualize_full_output = configs.visualize_full_output
        self.visualize_joint_positions = configs.visualize_joint_positions 
        self.visualize_stage_heatmaps = configs.visualize_stage_heatmaps
        
        self.output_dir = configs.output_dir 
            
    def start_capture(self, idx):
            
        if self.visualize_cropped_output:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            file_name = str(idx) + '.avi'
            out_path = os.path.join(self.output_dir, 'cropped', file_name)
            self.cropped_out_stream = cv2.VideoWriter(out_path, fourcc, self.video_fps, (640, 480))
            
        if self.visualize_full_output:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            file_name = str(idx) + '.avi'
            out_path = os.path.join(self.output_dir, 'full', file_name)
            self.full_out_stream = cv2.VideoWriter(out_path, fourcc, self.video_fps, (self.input_size, self.input_size))
            
        if self.visualize_joint_positions:
            self.left_joint_signal = []
            self.right_joint_signal = []
        
        if self.visualize_stage_heatmaps:
            self.left_stage_heatmaps = []
            self.right_stage_heatmaps = []
                                              
    def update_capture(self, full_img, left_data, right_data):
        
        left_hand_img, left_crop, left_score, stage_left_heatmap_np = left_data
        right_hand_img, right_crop, right_score, stage_right_heatmap_np = right_data
        
        if self.visualize_cropped_output:
            # visualize joints

            left_hand_img = self.visualize_result(left_hand_img, stage_left_heatmap_np, None)
            right_hand_img = self.visualize_result(right_hand_img, stage_right_heatmap_np, None)
            left_hand_img = cv2.resize(np.squeeze(left_hand_img), (320, 480))
            right_hand_img = cv2.resize(np.squeeze(right_hand_img), (320, 480))
            right_hand_img = cv2.flip(right_hand_img, 1)

            # combine left and right hands into one output image

            both_hand_img = np.concatenate((left_hand_img, right_hand_img), axis=1)
            
            # plot left-hand detection scores
            both_hand_img = cv2.putText(both_hand_img, str(left_score), 
            LEFT_OF_SCREEN, 
            FONT, 
            FONT_SCALE,
            FONT_COLOR,
            LINE_TYPE)
            
            # plot right-hand detection scores
            both_hand_img = cv2.putText(both_hand_img, str(right_score), 
            RIGHT_OF_SCREEN, 
            FONT, 
            FONT_SCALE,
            FONT_COLOR,
            LINE_TYPE)
            
            self.cropped_out_stream.write(both_hand_img.astype(np.uint8))
            
        if self.visualize_full_output:
           
            x_joints, y_joints = get_global_pose(
                stage_left_heatmap_np,
                left_crop,
                self.input_size,
                self.hmap_size,
                self.joints,
                flip=False)
            
            left_joints = np.column_stack((x_joints, y_joints))
                    
            x_joints, y_joints = get_global_pose(
                stage_right_heatmap_np,
                right_crop,
                self.input_size,
                self.hmap_size,
                self.joints,
                flip=True)
            
            right_joints = np.column_stack((x_joints, y_joints))
                                                      
            frame = self.plot_joints(full_img, left_joints, right_joints, left_crop, right_crop)
            
            self.full_out_stream.write(frame.astype(np.uint8))
            
        if self.visualize_joint_positions:
            
            x_joints, y_joints = get_global_pose(
                stage_right_heatmap_np,
                right_crop,
                self.input_size,
                self.hmap_size,
                self.joints)
            
            self.right_joint_signal.append(np.column_stack((x_joints, y_joints)))
            
            x_joints, y_joints = get_global_pose(
                stage_left_heatmap_np,
                left_crop,
                self.input_size,
                self.hmap_size,
                self.joints)
            
            self.left_joint_signal.append(np.column_stack((x_joints, y_joints)))
            
        if self.visualize_stage_heatmaps:
            
            self.left_stage_heatmap.append(stage_left_heatmap_np[5][0])
            self.right_stage_heatmap.append(stage_right_heatmap_np[5][0])
            
    def end_capture(self, idx):
        
        if self.visualize_cropped_output:
            self.cropped_out_stream.release()
        
        if self.visualize_full_output:
            self.full_out_stream.release()
            
        if self.visualize_joint_positions:
            
            left_file_path = os.path.join(self.output_dir, 'signals','{}_left_joint_signal.npy'.format(str(idx)))
            right_file_path = os.path.join(self.output_dir, 'signals','{}_right_joint_signal.npy'.format(str(idx)))

            np.save(left_file_path, self.left_joint_signal)
            np.save(right_file_path, self.right_joint_signal)                                                 
            
        if self.visualize_stage_heatmaps:
            
            left_file_path = os.path.join(self.output_dir, 'heatmaps', '{}_left_stage_heatmap.npy'.format(str(idx)))
            right_file_path = os.path.join(self.output_dir, 'heatmaps', '{}_right_stage_heatmap.npy'.format(str(idx)))
            
            np.save(left_file_path, self.left_stage_heatmap)
            np.save(right_file_path, self.right_stage_heatmap)
            
    def plot_joints(self, img, left_joints, right_joints, left_crop, right_crop):
        
        img = cv2.resize(img, (self.input_size, self.input_size))
        
        for joint_num in range(self.joints):

            joint_coord = (int(left_joints[joint_num][0] * self.input_size), int(left_joints[joint_num][1] * self.input_size))

            color_code_num = (joint_num // 4)
            if joint_num in [0, 4, 8, 12, 16]:
                joint_color = list(map(lambda x: x + 35 * (joint_num % 4), self.joint_color_code[color_code_num]))
                cv2.circle(img, center=(joint_coord[0], joint_coord[1]), radius=1, color=joint_color, thickness=-1)
            else:
                joint_color = list(map(lambda x: x + 35 * (joint_num % 4), self.joint_color_code[color_code_num]))
                cv2.circle(img, center=(joint_coord[0], joint_coord[1]), radius=1, color=joint_color, thickness=-1)
                
        for joint_num in range(self.joints):

            joint_coord = (int(right_joints[joint_num][0] * self.input_size), int(right_joints[joint_num][1] * self.input_size))

            color_code_num = (joint_num // 4)
            if joint_num in [0, 4, 8, 12, 16]:
                joint_color = list(map(lambda x: x + 35 * (joint_num % 4), self.joint_color_code[color_code_num]))
                cv2.circle(img, center=(joint_coord[0], joint_coord[1]), radius=1, color=joint_color, thickness=-1)
            else:
                joint_color = list(map(lambda x: x + 35 * (joint_num % 4), self.joint_color_code[color_code_num]))
                cv2.circle(img, center=(joint_coord[0], joint_coord[1]), radius=1, color=joint_color, thickness=-1)
        
        left_crop = left_crop * self.input_size
        
        top, left, bottom, right = left_crop
        
        img = cv2.rectangle(img, (left, top), (right, bottom), (77, 255, 9), 3, 1)
        
        right_crop = right_crop * self.input_size
        
        top, left, bottom, right = right_crop
        
        img = cv2.rectangle(img, (left, top), (right, bottom), (77, 255, 9), 3, 1)


        #img = cv2.resize(img, (640, 480))
        return img
        
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
            
            #heatmap_tensor = torch.tensor(last_heatmap, dtype=torch.float)
            #heatmap_tensor = heatmap_tensor.permute(2, 1, 0).unsqueeze(0) # (W, H, K) -> (K, H, W) -> (N, K, H, W)
            
            #print(heatmap_tensor.shape)
            
            #joints_tensor = integral_heatmap_layer(heatmap_tensor)
            #joints = joints_tensor.detach().squeeze(0).numpy()
            
            for joint_num in range(self.joints):
                    
                    
                joint_coord = np.unravel_index(np.argmax(last_heatmap[:, :, joint_num]),
                                              (test_img.shape[0], test_img.shape[1]))
                joint_coord_set[joint_num, :] = [joint_coord[0], joint_coord[1]]
                
                #print("Joint coords")
                #print(joint_coord)
                #print((joints[joint_num][0]* test_img.shape[0], joints[joint_num][1]*test_img.shape[1]))
               
                #joint_coord = joints[joint_num]
                #joint_coord_set[joint_num, :] = [joint_coord[0], joint_coord[1]]

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