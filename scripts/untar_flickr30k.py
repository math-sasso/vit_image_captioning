#from pathlib import Path
import os
import tarfile

def untar(fname:str):
    import pdb;pdb.set_trace()
    if fname.endswith("tar.gz"):
        tar = tarfile.open(fname, "r:gz")
        tar.extractall()
        tar.close()
    elif fname.endswith("tar"):
        tar = tarfile.open(fname, "r:")
        tar.extractall()
        tar.close()

current_path = os.path.dirname(os.path.abspath(__file__))
flickr30k_images_tar_path = os.path.join(current_path,"..","data","flickr30k","flickr30k-images.tar.gz")
# current_path = Path(__file__)
# flickr30k_images_tar_path = current_path.parent.parent / "data/flickr30k/flickr30k-images.tar.gz"
untar(flickr30k_images_tar_path)