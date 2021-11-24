import random
import pandas as pd
from typing import List

class Flickr30kDFBuilder():
  def __init__(self,tokens_filepath:str,train_val_list_filepath:str,test_list_filepath:str,selected_img_for_test_val:int=0):
    
    # setting variables
    self.selected_img_for_test_val = selected_img_for_test_val
    
    # compositions
    self.flickr30k_mapfile_reader = Flickr30kMapFileReader(tokens_filepath)
    self.snlive_reader = SnliveSplitsReader(train_val_list_filepath=train_val_list_filepath,
                                            test_list_filepath=test_list_filepath)
    
    # data organization
    self.splits = self.snlive_reader.get_splits()
    self.df_map = self.flickr30k_mapfile_reader.read_flickr30_map_as_df()
    self.datasets = self.__build_datasets()
    

  def __filter(self,step:str):
    samples_list = self.splits[step]
    filtered_df = self.df_map.loc[self.df_map['img_id'].isin(samples_list)].reset_index(drop=True)
    
    if step in ["train","val","test"]:
      filtered_df  = filtered_df.loc[filtered_df['img_example'] == str(self.selected_img_for_test_val)].reset_index(drop=True)
    

    return filtered_df
  
  def __build_datasets(self):
    datasets = {
        "train":self.__filter("train"),
        "val":self.__filter("val"),
        "test":self.__filter("test"),
    }

    return datasets
  
  def get_dataframe(self,step:str):
    if step not in self.datasets.keys():
      raise ValueError()
    
    return self.datasets[step]

class Flickr30kMapFileReader():
  def __init__(self,tokens_filepath):
    self.tokens_filepath = tokens_filepath


  def __get_line_first_element_split(self,line:List,str_spliter:str):
    line = line.split(str_spliter)
    elem = line[0]
    line = "".join(line[1:])
    return elem,line

  def read_flickr30_map_as_df(self):
    
    img_id_list = []
    img_example_list = []
    caption_list = []

    with open(self.tokens_filepath, 'r') as fin:          
          for line in fin:
            img_id,line = self.__get_line_first_element_split(line,"#")
            img_id,_ = img_id.split(".jpg")
            img_example,line = self.__get_line_first_element_split(line,"\t")
            caption = line.replace("\n","")

            img_id_list.append(img_id)
            img_example_list.append(img_example)
            caption_list.append(caption)

    df_dict = {
        "img_id":img_id_list,
        "img_example":img_example_list,
        "caption":caption_list
    }
    df = pd.DataFrame.from_dict(df_dict)
    return df

class SnliveSplitsReader():
  def __init__(self,train_val_list_filepath:str,test_list_filepath:str):
    self.train_val_list_filepath = train_val_list_filepath
    self.test_list_filepath = test_list_filepath
    self.train_val_list = self.__read_lines_file(self.train_val_list_filepath)
    self.test_list = self.__read_lines_file(self.test_list_filepath)
  
  def __read_lines_file(self,path:str):
    with open(path,'r+') as f:
      lines=f.read().splitlines()
      return lines
  
  def get_splits(self,train_val_proportion:float=0.95,seed:int=42):
    train_val_list = self.train_val_list.copy()
    random.Random(seed).shuffle(train_val_list)
    factor = int(len(train_val_list)*train_val_proportion)
    train_list = train_val_list[:factor]
    val_list = train_val_list[factor:]

    splits = {
        "train":train_list,
        "val":val_list,
        "test":self.test_list,
    }
    return splits


        
if __name__ == '__main__':
    tokens_filepath = "/content/data/flickr30k/results_20130124.token"
    train_val_list_filepath=os.path.join(FLICKR30K_DRIVE_DIRPATH,"snlive_flickr30k_splits","flickr30k_test.lst")
    test_list_filepath=os.path.join(FLICKR30K_DRIVE_DIRPATH,"snlive_flickr30k_splits","flickr30k_train_val.lst")

    f30k_df_builder = Flickr30kDFBuilder(tokens_filepath=tokens_filepath,
                                        train_val_list_filepath=train_val_list_filepath,
                                        test_list_filepath=test_list_filepath)
    f30k_df_builder.get_dataframe("test")
