import torch
import torch.utils.data
import os
import csv
import bisect
import numpy as np
import cv2
import json

from preprocessing.spectrogram import SpectrogramGenerator
from preprocessing.augmentation import Augmentor

CHAR_TO_CLASS = {
    'None': 0,
    'a': 1,
    'b': 2,
    'c': 3,
    'd': 4,
    'e': 5,
    'f': 6,
    'g': 7,
    'h': 8,
    'i': 9,
    'j': 10,
    'k': 11,
    'l': 12,
    'm': 13,
    'n': 14,
    'o': 15,
    'p': 16,
    'q': 17,
    'r': 18,
    's': 19,
    't': 20,
    'u': 21,
    'v': 22,
    'w': 23,
    'x': 24,
    'y': 25,
    'z': 26,
    ' ': 27,
}

EPSILON = 0.01

class Dataset(torch.utils.data.Dataset):

    def __init__(self, configs, label_path):

        self.data_dir = configs.data_dir
        self.label_path = label_path
        self.video_sampling_rate = configs.video_sampling_rate

        with open(self.label_path) as csvfile:

            self.labels = {}
            self.data = {}
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')

            for row in reader:
                idx, video_path, labels, timestamps = row

                idx = int(idx)
                video_id = int(os.path.basename(video_path).split('.')[0])
                timestamps = json.loads(timestamps)
                print(timestamps)

                self.data[idx] = video_path
                self.labels[idx] = (video_id, labels, timestamps)



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_tensor =  self.video_to_tensor(os.path.join(self.data_dir, self.data[idx]))

        return video_tensor, self.labels[idx]


    def video_to_tensor(self, video_file):
        """ Converts a mp4 file into a pytorch tensor"""

        cap = cv2.VideoCapture(video_file)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.video_sampling_rate)
        frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

        fc = 0
        ret = True

        while (fc < frameCount  and ret):
            ret, frame = cap.read()
            if fc % self.video_sampling_rate == 0:
                buf[fc] = frame
            fc += 1

        cap.release()
        return buf #torch.tensor(buf)
      
class SpectrogramDataset(torch.utils.data.Dataset):
    
    def __init__(self, configs, label_path):
        
        self.data_dir = configs.data_dir
        self.label_path = label_path
        self.fps = configs.video_fps
        
        self.data_mean = configs.spectrogram_mean
        self.bucket_ratio = configs.spectrogram_bucket_ratio
        self.time_offset = configs.spectrogram_time_offset
        
        self.spectrogram_generator = SpectrogramGenerator(configs)
        self.left_path_template = '{}_left_joint_signal.npy'
        self.right_path_template = '{}_right_joint_signal.npy'
        
        self.augmentor = Augmentor(configs)
        
        with open(self.label_path) as csvfile:

            self.z = {} # basis coefficients
            self.t = {} # time buckets
            self.f = {} # frequency basis
            
            self.left_joints = {}
            self.right_joints = {}
                                        
            self.labels = {} # string labels
            self.label_tensor = {} # class-mapped tensor labels 
            
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')

            for row in reader:
                idx, video_path, labels, timestamps = row

                idx = int(idx)
                video_id = int(os.path.basename(video_path).split('.')[0])
                timestamps = json.loads(timestamps)
                print(timestamps)

                self.labels[idx] = (video_id, labels, timestamps)
                        
                # load data and generate spectrograms
        
                left_path = os.path.join(self.data_dir, self.left_path_template.format(video_id))
                right_path = os.path.join(self.data_dir, self.right_path_template.format(video_id))

                print("Loading joints from: {}".format(left_path))
                print("Loading joints from: {}".format(right_path))

                left_joints = np.load(left_path)
                right_joints = np.load(right_path)
                
                self.left_joints[idx] = left_joints
                self.right_joints[idx] = right_joints


    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        #return self.z[idx], self.label_tensor[idx]
        
        video_id, labels, timestamps = self.labels[idx]
        
        left_joints = self.augmentor.augment_joint_locations(self.left_joints[idx])
        right_joints = self.augmentor.augment_joint_locations(self.right_joints[idx])

        #print("Generating spectrograms")

        z, t, f = self.spectrogram_generator.process_signal(left_joints, right_joints)

        """
        # alternative for defining our own loss if needed 

        label_tensor = np.zeros((len(t), 27))

        for jdx, timestamp in enumerate(timestamps):

            time_slot = bisect.bisect_left(t, timestamp)
            label_tensor[time_slot][CHAR_TO_CLASS[labels[jdx]]] = 1.0

        label_tensor = torch.tensor(label_tensor, dtype=torch.float).to("cuda")
        """
        # create detection buckets 

        """

        num_buckets = int(len(t) / self.bucket_ratio)                
        total_time = left_joints.shape[0] / self.fps
        t = np.linspace(self.time_offset, total_time - self.time_offset, num_buckets)

        z = z[: self.bucket_ratio * num_buckets]

        """

        # normalize:

        z = z - self.data_mean

        # format labels:

        label_tensor = np.zeros(len(t))

        for jdx, timestamp in enumerate(timestamps):

            time_slot = bisect.bisect_left(t, timestamp)
            label_tensor[time_slot] = CHAR_TO_CLASS[labels[jdx]]

        label_tensor = torch.tensor(label_tensor, dtype=torch.long).to("cuda")

        data = torch.tensor(z, dtype=torch.float).to("cuda")

        self.z[idx] = data
        self.t[idx] = t 
        self.f[idx] = f
        self.label_tensor[idx] = label_tensor
        
        return self.z[idx], self.label_tensor[idx]
    
    def getitem__(self, idx):
        
        
        video_id, labels, timestamps = self.labels[idx]
              
        # load data and generate spectrograms
        
        left_path = os.path.join(self.data_dir, self.left_path_template.format(video_id))
        right_path = os.path.join(self.data_dir, self.right_path_template.format(video_id))

        print("Loading joints from: {}".format(left_path))
        print("Loading joints from: {}".format(right_path))

        left_joints = np.load(left_path)
        right_joints = np.load(right_path)

        print("Generating spectrograms")

        z, t, f = self.spectrogram_generator.process_signal(left_joints, right_joints)
        
        # normalize:
        
        z = z - self.data_mean
        
        """
        # alternative for defining our own loss if needed 
        
        label_tensor = np.zeros((len(t), 27))

        for jdx, timestamp in enumerate(timestamps):

            time_slot = bisect.bisect_left(t, timestamp)
            label_tensor[time_slot][CHAR_TO_CLASS[labels[jdx]]] = 1.0
                               
        label_tensor = torch.tensor(label_tensor, dtype=torch.float).to("cuda")
        """
       
        # format labels:
        
        label_tensor = np.zeros(len(t))
        
        for jdx, timestamp in enumerate(timestamps):

            time_slot = bisect.bisect_left(t, timestamp)
            label_tensor[time_slot] = CHAR_TO_CLASS[labels[jdx]]
                               
        label_tensor = torch.tensor(label_tensor, dtype=torch.long).to("cuda")
        
        data = torch.tensor(z, dtype=torch.float).to("cuda")
        
        return data, label_tensor
    
    def get_mean(self):
        
        data = []
        
        for idx in range(len(self.labels)):
            
            video_id, labels, timestamps = self.labels[idx]

            # load data and generate spectrograms
            
            z = self.z[idx].detach().numpy()
            
            z = np.moveaxis(z, -1, 0)
            
            data.append(z)
            
        data = np.concatenate(data)
        
        print(np.mean(data))
        print(np.std(data))

        
    def get_class_weights(self):
        
        occurences = np.zeros(len(CHAR_TO_CLASS.keys()))
        
        for idx in range(len(self.labels)):
            
            video_id, labels, timestamps = self.labels[idx]
            
            t = self.t[idx]
            
            occurences[0] += len(t) - len(labels)
            for char in labels:
                occurences[CHAR_TO_CLASS[char]] += 1
        
        print("TOTAL OCCURENCES: {}".format(np.sum(occurences)))
        occurences = (occurences / float(np.sum(occurences))) + EPSILON
        
        weights = np.ones(len(CHAR_TO_CLASS.keys())) / occurences 
        weights = weights / np.sum(weights)
        
        print(weights)
        