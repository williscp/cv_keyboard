class Config():

    def __init__(self):

        # data loading

        self.data_dir = './data'
        self.label_path = './data/train.csv'
        self.video_sampling_rate = 1
        self.video_fps = 15

        self.input_size = 368
        
        # hand detector
        
        self.buffer = 50 #pixels around the crop 
        self.score_threshold = 0.2 

        # hand model

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

        self.output_dir = './output'
        self.model_dir = './models'
        self.hand_pose_estimator_weights = './convolutional_pose_machines_tensorflow/models/weights/cpm_hand.pkl'
        self.hand_detector_weights = 'handtracking/hand_inference_graph/frozen_inference_graph.pb'

        # model

        self.lr = 1e-3
