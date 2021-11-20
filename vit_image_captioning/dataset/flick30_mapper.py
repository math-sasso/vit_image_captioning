from typing import List

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

    with open(tokens_filepath, 'r') as fin:          
          for line in fin:
            try:
              img_id,line = self.__get_line_first_element_split(line,"#")
              img_example,line = self.__get_line_first_element_split(line,"\t")
              caption = line.replace("\n","")
              #caption,line = self.__get_line_first_element_split(line,"#")
            except:
              import pdb;pdb.set_trace()

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

if __name__ == '__main__':
    tokens_filepath = "/content/data/flickr30k/results_20130124.token"
    flickr30k_mapfile_reader = Flickr30kMapFileReader(tokens_filepath)
    df = flickr30k_mapfile_reader.read_flickr30_map_as_df()