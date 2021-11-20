import gdown


url = 'https://drive.google.com/drive/folders/1-2eHmVUqaaf3SSCGG1FUQltJ88bb2iOY?usp=sharing'
gdown.download_folder(url, quiet=True)

# url = 'https://drive.google.com/file/d/1-NL5VXGe2X7P8VEAyThseoRZKjTMXTg7/view?usp=sharing'
# output = 'flickr30k.tgz'
# gdown.download(url, output, quiet=False)


# url = 'https://drive.google.com/file/d/1-3IwC37-lQRlMhOy8F6YzGlxgkYyeXIb/view?usp=sharing'
# output = 'flickr30k_imgs.tgz'
# gdown.download(url, output, quiet=False)
