class Config():

    def __init__(self):

        # data loading

        self.data_dir = './data/joints'
        self.video_sampling_rate = 1
        self.video_fps = 15

        self.input_size = 368
        
        # hand detector
        
        self.detector_gpu_alloc = 0.3
        self.buffer = 50 #pixels around the crop 
        self.score_threshold = 0.2 

        # hand model
        
        self.estimator_gpu_alloc = 0.3
        self.hmap_size = 46
        self.cmap_size = 21
        self.joints = 21
        self.stage = 6 # number of cmp stages
        self.kalman = False
        self.kalman_noise = 3e-2
        self.color_channel = 'RGB'

        self.limb_model = [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [0, 5],
            [5, 6],
            [6, 7],
            [7, 8],
            [0, 9],
            [9, 10],
            [10, 11],
            [11, 12],
            [0, 13],
            [13, 14],
            [14, 15],
            [15, 16],
            [0, 17],
            [17, 18],
            [18, 19],
            [19, 20]
        ]

        self.joint_color_code = [
            [139, 53, 255],
            [0, 56, 255],
            [43, 140, 237],
            [37, 168, 36],
            [147, 147, 0],
            [70, 17, 145]
        ]

        # output
        
        self.visualize_cropped_output = True # output detection + joint estimates
        self.visualize_full_output = True
        self.visualize_joint_positions = True # output joint locations as npy files 
        self.visualize_stage_heatmaps = False # output last layer heatmaps
        
        self.output_dir = './output'
        self.model_dir = './models'
        self.hand_pose_estimator_weights = './convolutional_pose_machines_tensorflow/models/weights/cpm_hand.pkl'
        self.hand_detector_weights = 'handtracking/hand_inference_graph/frozen_inference_graph.pb'
        
        # data augmentation:
        
        self.augmentation_max_angle = 20
        self.augmentation_max_scale = 0.2

        # model
        
        self.epochs = 250
        self.lr = 1e-4
        self.input_freqs = 11 
        self.spectrogram_mean = 2.56440538185e-05
        #self.class_weights = [
        #    0.00157526, 0.01949247, 0.03721764, 0.03207116, 0.03721764, 0.01428861,
        #    0.03624818, 0.03445327, 0.02999715, 0.02343101, 0.0756698, 0.04296288,
        #    0.0260642, 0.03445327, 0.0192232, 0.01556738, 0.04167618, 0.08002114,
        #    0.02034753, 0.02230434, 0.01949247, 0.02936416, 0.05480771, 0.04901612,
        #    0.0756698, 0.03721764, 0.08002114, 0.0101286]
        
        self.class_weights = [ 
            0.0018884, 0.01736119, 0.04402892, 0.03586691, 0.0285816, 0.01299011,
            0.03282444, 0.03671775, 0.02375622, 0.01884044, 0.07679666, 0.04889706,
            0.02756365, 0.03466211, 0.0190726, 0.01716862, 0.0342783, 0.07875027,
            0.02019225, 0.01993222, 0.01726436, 0.03282444, 0.06153197, 0.05049214,
            0.07875027, 0.03715848, 0.08297167, 0.00883695]


        
        # spectrogram configs:
        
        self.spectrogram_window_type = 'hann'
        self.spectrogram_window_size = 20 #20
        self.spectrogram_window_overlap = 18 #15
        self.spectrogram_bucket_ratio = 3 # number of spectrogram buckets to consider as single detection bucket 
        self.spectrogram_time_offset = 0.666667 # offset due to bucket size
        
