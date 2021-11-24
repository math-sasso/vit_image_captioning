import os
import torch
import logging
from configs import configs
import pandas as pd
from data.data_module import MyDataModule
from model.caption_model import CaptionModel
import pytorch_lightning as pl
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

class Predictor():
    def __init__(self,model,device):
        self.model = model
        self.device = device

    def predict(self,data_module):
        df = pd.DataFrame(columns = ['trues','preds'])
        dict_df = {'trues':[],'preds':[]}

        i = 0
        for batch in tqdm(data_module.test_dataloader()):
            batch['image_tensor'] = batch['image_tensor'].to(self.device)
            batch['transformed_image_tensor'] = batch['transformed_image_tensor'].to(self.device)
            batch['tokenized_target_caption'] = batch['tokenized_target_caption'].to(self.device)
            rets = self.model._base_eval_step(batch)
            dict_df['trues'] += rets['trues']
            dict_df['preds'] += rets['preds']

            i += 1
            if i==5:
                break
            
        df = pd.DataFrame.from_dict(dict_df)

        return df

# Setting Paths
root_path = "/home/sasso158257/singularity/snlive/image_captioning/vit_image_captioning"
model_checkpoint = "epoch=0-val_bleu_score=0.14.ckpt"
checkpoint_path = os.path.join(root_path,"logs",model_checkpoint)
output_path = os.path.join(root_path,'results','model_results_v1.csv')

# Carrgando o modelo
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
model = CaptionModel.load_from_checkpoint(checkpoint_path = checkpoint_path,map_location=device,hparams=configs)
model.to(device)

# Caregando data modlule
data_module = MyDataModule(configs)

# Predictions
predictor = Predictor(model,device)
df = predictor.predict(data_module)
import pdb;pdb.set_trace()
df.to_csv(output_path,index=False)
