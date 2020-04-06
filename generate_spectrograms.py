import os 
import numpy as np

from dataset import Dataset
from config import Config
from preprocessing.spectrogram import SpectrogramGenerator 

configs = Config()
dataset = Dataset(configs)
spectrogram_generator = SpectrogramGenerator(configs)

signal_input_dir = 'output/signals'
signal_output_dir = 'data/spectrograms'

left_path_template = '{}_left_joint_signal.npy'
right_path_template = '{}_right_joint_signal.npy'

for idx in range(len(dataset)):
    
    left_path = os.path.join(signal_input_dir, left_path_template.format(idx))
    right_path = os.path.join(signal_input_dir, right_path_template.format(idx))
    
    print("Loading joints from: {}".format(left_path))
    print("Loading joints from: {}".format(right_path))
    
    left_joints = np.load(left_path)
    right_joints = np.load(right_path)

    print("Generating spectrograms")
    
    spectrogram_tensor, t, f = spectrogram_generator.process_signal(left_joints, right_joints)
    
    print("Saving to {}".format(signal_output_dir))
    
    np.save(os.path.join(signal_output_dir, '{}_z.npy'.format(idx)), spectrogram_tensor)
    np.save(os.path.join(signal_output_dir, '{}_t.npy'.format(idx)), t)
    np.save(os.path.join(signal_output_dir, '{}_f.npy'.format(idx)), f)
    