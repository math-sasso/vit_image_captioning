import torch
import numpy as np
import pandas as pd
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision import transforms
from torch.utils.data import Dataset

class MyDataset(Dataset):
    """Pytorch dataset para treinamento da rede

        h5_images_path (str): Arquivo hdf5 com as imagens
        json_captions_path (str): Json com as captions das imagens (5 por imagem)
        transforms (callable): pré-processamento de cada imagem
        return_dict (bool): se devemos retornar os resultados como um dict
        tokenizer: tokenizador a ser usado
        max_tokens_captions: tamanho máximo de tokens
        captions_per_image: número de legendas por imagem
    """

    def __init__(self,images_path: str, df_captions: pd.DataFrame,tokenizer,
                 max_tokens_captions, img_size, captions_per_image=5):
        self.images_path = images_path
        self.df_captions = df_captions
        self.tokenizer = tokenizer
        self.max_tokens_captions = max_tokens_captions
        self.captions_per_image = captions_per_image
        self.img_size = img_size
        #import pdb;pdb.set_trace()
    
    def __len__(self):
        return len(self.df_captions)

    def __transforms(self,image):
        transfors_func =  transforms.Compose([
        transforms.ToPILImage(),
        create_transform(**resolve_data_config(
            {},model='vit_base_patch16_224'))
        ])
        return transfors_func(image)

    def __tokenize(self, input_text):
        tokenized = self.tokenizer(
            input_text, return_tensors='pt',padding='max_length',
            max_length=self.max_tokens_captions,pad_to_max_length=True,
            return_attention_mask=False, truncation=True,
            add_special_tokens=True
            )
        return tokenized['input_ids'].squeeze()
      
    def __tensorize_image(self,img):
        img_tensor = torch.from_numpy(img).to(torch.uint8) # (H, W, C)
        return img_tensor.transpose(0,2).transpose(1,2) # (C, H, W)

    def __getitem__(self, idx):

        # Get dataframe idx
        batch_captions = self.df_captions.loc[idx]
        
        # Retrieve image
        img = Image.open(f"{self.images_path}/{batch_captions['img_id']}.jpg")
        img = img.resize((self.img_size,self.img_size), Image.ANTIALIAS)
        image_data = np.asarray( img )
        image_tensor = self.__tensorize_image(image_data)
        transformed_image_tensor = self.__transforms(image_tensor)
            
        # Tokenize Caption
        target_caption = batch_captions["caption"]
        tokenized_target_caption = self.__tokenize(target_caption)


        rets = {
            'image_tensor': image_tensor,
            'transformed_image_tensor': transformed_image_tensor,
            'target_caption': target_caption,
            'tokenized_target_caption': tokenized_target_caption
            }
        return rets
