
class Config():

    def __init__(self):

        # data loading

        self.data_dir = './data'
        self.label_path = './data/train.csv'
        self.video_sampling_rate = 1

        # output

        self.output_dir = './output'
        self.model_dir = './models'

        # model

        self.lr = 1e-3
