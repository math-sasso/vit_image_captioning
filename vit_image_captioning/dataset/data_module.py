from transformers import T5Tokenizer
from torch.utils.data import DataLoader
from data.dataset import MyDataset
import pytorch_lightning as pl
from data.flickr30k_dataframe_builder import Flickr30kDFBuilder
from typing import List

class MyDataModule(pl.LightningDataModule):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.f30k_df_builder = Flickr30kDFBuilder(tokens_filepath=self.configs["paths"]["tokens_filepath"],
                                     train_val_list_filepath=self.configs["paths"]["train_val_list_filepath"],
                                     test_list_filepath=self.configs["paths"]["test_list_filepath"])
        
        self.tokenizer = T5Tokenizer.from_pretrained(self.configs['model_config']['t5_prefix'])
    
    def __get_dataset(self, partition):

        dataset = MyDataset(
            images_path=self.configs['paths']['images_path'],
            df_captions=self.f30k_df_builder.get_dataframe(partition),
            tokenizer=self.tokenizer,
            max_tokens_captions= self.configs['captions_config']['max_tokens_captions'],
            img_size = self.configs["images_config"]["img_size"]
            )

        return dataset
    
    
    def train_dataloader(self):
        """Dataloader de treino.

        """
        dataloader = DataLoader(
            self.__get_dataset('train'),
            batch_size=self.configs['dataloader']['batch_size']['train'],
            num_workers=self.configs['dataloader']['num_workers'],
            shuffle=True,
            )
        

        return dataloader 

    def val_dataloader(self):
        """Dataloader de validação

        """
        dataloader = DataLoader(
            self.__get_dataset('val'),
            batch_size=self.configs['dataloader']['batch_size']['val'],
            num_workers=self.configs['dataloader']['num_workers'],
            shuffle=False,
            )
        return dataloader

    def test_dataloader(self):
        """Dataloader de teste.

        """
        dataloader = DataLoader(
            self.__get_dataset('test'),
            batch_size=self.configs['dataloader']['batch_size']['test'],
            num_workers=self.configs['dataloader']['num_workers'],
            shuffle=False,
            )
        return dataloader
