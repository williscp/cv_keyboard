import numpy as np
import torch
import scipy.signal as signal


class SpectrogramGenerator():
    
    def __init__(self, configs):
        
        self.window_type = configs.spectrogram_window_type
        self.window_size = configs.spectrogram_window_size 
        self.window_overlap = configs.spectrogram_window_overlap 
        self.fs = configs.video_fps
        
    def process_signal(self, left_joints, right_joints):
        
        #left_joints is F x J X 2 (F = frames, J = joints) 
        
        F, J, _ = left_joints.shape
        
        spectrograms = []
        
        for joint in range(J):
            
            x_positions = left_joints[:, joint, 0]
            y_positions = left_joints[:, joint, 1]
            
            f, t, left_z_x = signal.spectrogram(x_positions, fs=self.fs, window=self.window_type, nperseg=self.window_size, noverlap=self.window_overlap)
            
            f, t, left_z_y = signal.spectrogram(y_positions, fs=self.fs, window=self.window_type, nperseg=self.window_size, noverlap=self.window_overlap)
                                    
            x_positions = right_joints[:, joint, 0]
            y_positions = right_joints[:, joint, 1]
            
            f, t, right_z_x = signal.spectrogram(x_positions, fs=self.fs, window=self.window_type, nperseg=self.window_size, noverlap=self.window_overlap)
            
            f, t, right_z_y = signal.spectrogram(y_positions, fs=self.fs, window=self.window_type, nperseg=self.window_size, noverlap=self.window_overlap)
                        
            spectrograms.append(np.stack((np.stack((left_z_x, left_z_y)), np.stack((right_z_x, right_z_y)))))
            
        
        # returns joints, left/right, x/y, spectrogram
        return np.array(spectrograms), t, f
            
