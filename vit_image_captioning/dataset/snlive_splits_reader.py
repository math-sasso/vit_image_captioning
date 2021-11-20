import os
import random 

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
  
  def get_splits(self,train_val_proportion:float=0.9,seed:int=42):
    train_val_list = self.train_val_list.copy()
    random.Random(seed).shuffle(train_val_list)
    factor = int(len(train_val_list)*train_val_proportion)
    train_list = train_val_list[:factor]
    val_list = train_val_list[:factor]

    splits = {
        "train":train_list,
        "val":val_list,
        "test":self.test_list,
    }
    return splits
    
if __name__ == '__main__':


    snlive_reader = SnliveSplitsReader(train_val_list_filepath=os.path.join(FLICKR30K_DRIVE_DIRPATH,"snlive_flickr30k_splits","flickr30k_test.lst"),
                                   test_list_filepath=os.path.join(FLICKR30K_DRIVE_DIRPATH,"snlive_flickr30k_splits","flickr30k_train_val.lst"))

    splits = snlive_reader.get_splits() 