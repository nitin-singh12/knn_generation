import shutil
import os
import random
import pickle
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torch.nn.functional as F
import torchvision.transforms as transforms
# from backbone import Backbone


from model_irse import Backbone
import knn_model


root_path = "/media/sowjanya-yobi/hugeDrive3/nitin/ludex/segmaker/all_data/"
out_path = "/media/sowjanya-yobi/hugeDrive3/nitin/ludex/segmaker/all_knn/"
emb_model_path = "/media/sowjanya-yobi/hugeDrive3/nitin/ludex/segmaker/Backbone_IR_SE_152_Epoch_45_Batch_31806_Time_2022-05-08-21-51_checkpoint.pth"

seed = 10
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# Embedding models
EMBEDDING_MODEL_PATH = emb_model_path
EMBEDDING_SIZE = 512
INPUT_SIZE =[224, 224]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
backbone = Backbone(INPUT_SIZE, 152, 'ir_se') # NEW
backbone.load_state_dict(torch.load(EMBEDDING_MODEL_PATH, map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
backbone.to(device)
backbone.eval()

knn = knn_model.KNN(backbone,out_path) 
sports_category = sorted(os.listdir(root_path))
for i in range(len(sports_category)):
    years = sorted(os.listdir(root_path+sports_category[i]))
    for year in years:
        print(root_path+sports_category[i]+"/"+year)
        knn.generate_knn(root_path+sports_category[i]+"/"+year)



