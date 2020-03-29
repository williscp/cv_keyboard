
class Config():

    def __init__(self):

        # data loading

        self.data_dir = './data'
        self.label_path = './data/train.csv'
        self.video_sampling_rate = 1

        self.input_size = 368

        # output

        self.output_dir = './output'
        self.model_dir = './models'
        self.hand_model_path = './convolutional-pose-machines-tensorflow/models/weights/cpm_hand.pkl'

        # model

        self.lr = 1e-3
