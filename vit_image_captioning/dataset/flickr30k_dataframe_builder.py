class Flickr30kDFBuilder():
  def __init__(self,tokens_filepath:str,train_val_list_filepath:str,test_list_filepath:str):
    #compositions
    self.flickr30k_mapfile_reader = Flickr30kMapFileReader(tokens_filepath)
    self.snlive_reader = SnliveSplitsReader(train_val_list_filepath,test_list_filepath)

    self.splits = snlive_reader.get_splits()
    self.df_map = self.flickr30k_mapfile_reader.read_flickr30_map_as_df()
    self.datasets = self.__build_datasets()

  def __filter(self,samples_list:List):
    return df_map.loc[df_map['img_id'].isin(samples_list)]
  
  def __build_datasets(self,):
    datasets = {
        "train":self.__filter(self.splits["train"]),
        "val":self.__filter(self.splits["val"]),
        "test":self.__filter(self.splits["test"]),
    }

    return datasets
  
  def get_dataframe(self,step:str):
    if step not in self.datasets.keys():
      raise ValueError()
    
    return self.datasets[step]

        
if __name__ == '__main__':
    tokens_filepath = "/content/data/flickr30k/results_20130124.token"
    train_val_list_filepath=os.path.join(FLICKR30K_DRIVE_DIRPATH,"snlive_flickr30k_splits","flickr30k_test.lst")
    test_list_filepath=os.path.join(FLICKR30K_DRIVE_DIRPATH,"snlive_flickr30k_splits","flickr30k_train_val.lst")

    f30k_df_builder = Flickr30kDFBuilder(tokens_filepath=tokens_filepath,
                                        train_val_list_filepath=train_val_list_filepath,
                                        test_list_filepath=test_list_filepath)
    f30k_df_builder.get_dataframe("test")