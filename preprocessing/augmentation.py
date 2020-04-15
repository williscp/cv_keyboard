import numpy as np

class Augmentor():

    def __init__(self, configs):
        
        self.max_angle = configs.augmentation_max_angle 
        self.max_scale = configs.augmentation_max_scale 
        
    def augment_joint_locations(self, joints):

        #print(joints.shape)
        
        #print(joints[0][0])

        # center
        joints = joints - 0.5

        scale_factor = 1 + ((np.random.random() - 0.5) * 2 * self.max_scale)
        scale_matrix = np.array([[1, 0], [0, 1]]) * scale_factor 

        angle = (np.random.random() - 0.5) * 2 * np.pi * self.max_angle / 180
        rot_matrix = np.array([[np.cos(angle), - np.sin(angle)], [np.sin(angle), np.cos(angle)]])

        joints = joints.dot(scale_matrix.T).dot(rot_matrix.T)
        
        joints = joints + 0.5
        
        #print(joints[0][0])
        #print(joints.shape)

        return joints
    