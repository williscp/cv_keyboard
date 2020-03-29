import torch
import os
import csv
import numpy as np
import cv2
import json

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
        return torch.tensor(buf)
