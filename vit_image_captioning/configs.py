import os

data_dirpath = "/home/sasso158257/singularity/snlive/image_captioning/vit_image_captioning/data"
configs = {
    "paths":{
        "images_path": os.path.join(data_dirpath,"flickr30k_img"),
        "tokens_filepath": os.path.join(data_dirpath,"flickr30k_txt", "results_20130124.token"),
        "train_val_list_filepath": os.path.join(data_dirpath,"snlive_flickr30k_splits","flickr30k_train_val.lst"),
         "test_list_filepath": os.path.join(data_dirpath,"snlive_flickr30k_splits","flickr30k_test.lst"),
    },
    "dataloader": {
        'batch_size': {'train': 10,'val': 10,'test': 10},
        'num_workers': 8,
        },
      "captions_config":{
          'max_tokens_captions': 64
      },
      "images_config":{
          "img_size" : 224
      },
      "model_config":{
          "t5_prefix":"t5-base",
          "vit_prefix":'vit_base_patch16_224',
          "learning_rate":1e-4,
          "use_t5_encoder":True
      },
      "freeze_conditions":{
          "freeze_t5_encoder":False,
          "freeze_t5_decoder":False,
          "freeze_t5_embeddings":False,
          "freeze_image_encoder":False,
      }
    
}
