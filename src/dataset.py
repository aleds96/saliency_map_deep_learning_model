#sistema 
import os
import torch
from torch.utils.data import Dataset,DataLoader
from PIL import Image
#utility 
import re
from utils import utility_fun as uf

class SalientDataset(Dataset):
    def __init__(self, source_path,dataset_mode,usecase,transform=None, target_transform=None,binary_target_transform=None,video_id=None):
        
        self.data_paths=uf.read_json(path=os.path.join(source_path,dataset_mode+'.json'))
        #filtriamo dati per uno specifico video id (per test)
        if video_id is not None: 
            #dal primo path(che è sempre presente) prendiamo il video_id
            self.data_paths['usecase'][usecase]=list(filter(lambda x: x[0].split('/')[-2]==str(video_id),self.data_paths['usecase'][usecase]))
            
        self.source_path = source_path
        self.dataset_mode = dataset_mode
        self.usecase = usecase 

        self.transform = transform
        self.target_transform = target_transform
        self.binary_target_transform = binary_target_transform

    def load_image(self,path,as_grayscale=False):
        channel_mode = 'L' if as_grayscale==True else 'RGB'
        try:
            im = Image.open(path).convert(channel_mode)
            return im
        except OSError:
            print(path)
        return Image.new(channel_mode, (512, 512), "white")
    def __getitem__(self, index):
        
        path_list =self.data_paths['usecase'][self.usecase][index]
        input_path = path_list[0] 
        CONTINUOUS_AND_BINARY = True if len(path_list)==3 else False
        if CONTINUOUS_AND_BINARY:  
            continuous_output_path = path_list[1]
            binary_output_path = path_list[2]
        else: 
            continuous_output_path = None
            binary_output_path = path_list[1]

        #leggi e trasforma input a
        input_frame = self.transform(self.load_image(input_path))
        #leggi e trasforma output continuo se esiste
        binary_output = [] if binary_output_path is None else self.binary_target_transform(self.load_image(binary_output_path,as_grayscale=True))

        
        if continuous_output_path is None:
            continuous_output = []
        else:
            # Applica la trasformazione (resize + ToTensor)
            continuous_output = self.target_transform(self.load_image(continuous_output_path, as_grayscale=True))
            # Clippa e normalizza in (0,1)
            continuous_output = torch.clamp(continuous_output, 0.0, 1.0)
            max_val = continuous_output.max()
            if max_val > 0:
                continuous_output = continuous_output / max_val

        
        if CONTINUOUS_AND_BINARY: 
            return input_frame, continuous_output, binary_output  
        return input_frame,[], binary_output
            
        #print('input path=>',input_path)
        #print('continuous path=>',continuous_output_path) 
        #print('binary path=>',binary_output_path) 
        #print('usecase==>',self.usecase)
    def __len__(self):
        return len(self.data_paths['usecase'][self.usecase])
