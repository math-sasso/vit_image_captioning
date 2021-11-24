import os
import logging
from configs import configs
from data.data_module import MyDataModule
from model.caption_model import CaptionModel
import pytorch_lightning as pl
import warnings
warnings.filterwarnings("ignore")

# import neptune.new as neptune

# run = neptune.init(
#     project="m158257/vit-image-captioning",
#     api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxNWQ5MDhkZi1iMTRmLTRmZWEtOGQ2My1kM2E4ZTA2YmYyMDgifQ==",
# )  # your credentials


# parameters = {}
# parameters.update(configs["captions_config"])
# parameters.update(configs["images_config"])
# parameters.update(configs["model_config"])
# parameters.update(configs["freeze_conditions"])
# run["parameters"] = parameters

# configs["run"] = run



data_module = MyDataModule(configs)


root_path = "/home/sasso158257/singularity/snlive/image_captioning/vit_image_captioning"

debug = False

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath= os.path.join(root_path,"logs"), 
    filename="{epoch}-{val_bleu_score:.2f}",
    monitor="val_bleu_score", mode="max"
)

logger = pl.loggers.TensorBoardLogger(os.path.join(root_path,"logs"), version='versao_teste')

early_stop_callback = pl.callbacks.EarlyStopping(
    monitor='val_bleu_score',
    patience=1,
    mode='max',
    min_delta=0.01
)

# Não registra nada nas execuções de debug
if debug:
    logging.info("Debug, não vamos salvar logs no tensorboard nem checkpoints")
    checkpoint_callback = None
    logger = None
    early_stop_callback = None
    

# Initialize trainer passing callbacks
trainer = pl.Trainer(
    gpus=1,
    log_gpu_memory=True,
    logger=logger, 
    # checkpoint_callback=checkpoint_callback, 
    callbacks=[early_stop_callback, checkpoint_callback],
    max_epochs=5,
    progress_bar_refresh_rate=100
)

model = CaptionModel(configs)

trainer.fit(
    model=model,
    train_dataloader=data_module.train_dataloader(),
    val_dataloaders=data_module.val_dataloader()
    )
    
trainer.test(dataloaders=data_module.test_dataloader())

# run.stop()
