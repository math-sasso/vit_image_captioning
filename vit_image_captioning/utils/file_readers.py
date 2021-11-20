import os
def read_lines_file(path:str):
  with open(path,'r+') as f:
    lines=f.read().splitlines()
    return lines


