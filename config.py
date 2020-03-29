
class Config():
    
    def __init__(self):
        
        # data loading
       
        self.data_dir = './data'
        self.label_path = './data/labels.csv'
        self.video_sampling_rate = 10
        
        # output 
        
        self.output_dir = './output'
        self.model_dir = './models'
        
        # model
        
        self.lr = 1e-3
       