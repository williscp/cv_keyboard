import torch
import torch.utils.data
import os
import csv
import bisect
import numpy as np
import cv2
import json

CHAR_TO_CLASS = {
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

class Dataset(torch.utils.data.Dataset):

    def __init__(self, configs):

        self.data_dir = configs.data_dir
        self.label_path = configs.label_path
        self.video_sampling_rate = configs.video_sampling_rate

        with open(self.label_path) as csvfile:

            self.labels = {}
            self.data = {}
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')

            for row in reader:
                video_id, video_path, labels, timestamps = row

                video_id = int(video_id)
                timestamps = json.loads(timestamps)
                print(timestamps)

                self.data[video_id] = video_path
                self.labels[video_id] = (labels, timestamps)



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_tensor =  self.video_to_tensor(os.path.join(self.data_dir, self.data[idx]))

        return (idx, video_tensor), self.labels[idx]


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
    
    def __init__(self, configs):
        
        self.data_dir = configs.data_dir
        self.label_path = configs.label_path
        self.data_mean = configs.spectrogram_mean

        
        with open(self.label_path) as csvfile:

            self.labels = {}
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')

            for row in reader:
                video_id, _, labels, timestamps = row

                video_id = int(video_id)
                timestamps = json.loads(timestamps)
                self.labels[video_id] = (labels, timestamps)


    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        z = np.load(os.path.join(self.data_dir, 'spectrograms', '{}_z.npy'.format(idx)))
        f = np.load(os.path.join(self.data_dir, 'spectrograms', '{}_f.npy'.format(idx)))
        t = np.load(os.path.join(self.data_dir, 'spectrograms', '{}_t.npy'.format(idx)))
        
        # normalize:
        
        z = z - self.data_mean
        
        labels, timestamps = self.labels[idx]
        
        """
        # alternative for defining our own loss if needed 
        
        label_tensor = np.zeros((len(t), 27))

        for jdx, timestamp in enumerate(timestamps):

            time_slot = bisect.bisect_left(t, timestamp)
            label_tensor[time_slot][CHAR_TO_CLASS[labels[jdx]]] = 1.0
                               
        label_tensor = torch.tensor(label_tensor, dtype=torch.float).to("cuda")
        """
        label_tensor = np.zeros(len(t))
        
        for jdx, timestamp in enumerate(timestamps):

            time_slot = bisect.bisect_left(t, timestamp)
            label_tensor[time_slot] = CHAR_TO_CLASS[labels[jdx]]
            
        #print(timestamps)
        #print(t)
        #print(label_tensor)
                               
        label_tensor = torch.tensor(label_tensor, dtype=torch.long).to("cuda")
        
        data = torch.tensor(z, dtype=torch.float).to("cuda")
        
        return data, label_tensor
    
    def get_mean(self):
        
        data = []
        
        for idx, label in enumerate(self.labels):
            
            z = np.load(os.path.join(self.data_dir, 'spectrograms', '{}_z.npy'.format(idx)))
            
            z = np.moveaxis(z, -1, 0)
            
            data.append(z)
            
        data = np.concatenate(data)
        print(data.shape)
        
        print(np.mean(data))
        print(np.std(data))
        
            
            