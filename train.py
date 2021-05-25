import torch

from pathlib import Path
from argparse import ArgumentParser, Namespace
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.model.net import CRMI
from src.task.pipeline import DataPipeline
from src.task.runner import ContrastiveRunner

def get_tensorboard_logger(args: Namespace) -> TensorBoardLogger:
    logger = TensorBoardLogger(save_dir=f"exp/{args.dataset}",
                               name=args.model,
                               version=args.tid + str(args.batch_size))
    return logger


def get_checkpoint_callback(args: Namespace) -> ModelCheckpoint:
    prefix = f"exp/{args.dataset}/{args.model}"
    suffix = "{epoch:02d}-{val_loss:.4f}"
    checkpoint_callback = ModelCheckpoint(
                                        monitor="val_loss",
                                        dirpath=prefix,
                                        filename= suffix,
                                        mode='min',
                                        save_top_k=1,
                                        save_weights_only=True,
                                        verbose=True)
    return checkpoint_callback

def get_early_stop_callback(args: Namespace) -> EarlyStopping:
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=5,
        verbose=False,
        mode='min',
    )
    return early_stop_callback


def main(args) -> None:
    logger = get_tensorboard_logger(args)
    checkpoint_callback = get_checkpoint_callback(args)
    early_stop_callback = get_early_stop_callback(args)

    pipeline = DataPipeline(args)
    model = CRMI(args)
    runner = ContrastiveRunner(model, args)
    
    trainer = Trainer(max_epochs= args.max_epochs,
                        gpus= [0],
                        distributed_backend= "dp",
                        benchmark= False,
                        deterministic= True,
                        logger=logger,
                        callbacks=[
                            early_stop_callback,
                            checkpoint_callback
                        ]
                      )
    trainer.fit(runner, datamodule=pipeline)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", default="CRMI", type=str)
    parser.add_argument("--dataset", default="baseline", type=str)
    parser.add_argument("--tid", default="0", type=str)

    # dataset
    parser.add_argument("--feature_path", default="./dataset", type=str)

    #dataloader
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--num_workers", default=8, type=int)

    # runner
    parser.add_argument("--learning_rate", default=1e-3, type=float)
    parser.add_argument("--weight_decay", default=1e-3, type=float)
    parser.add_argument("--verbose", default=2, type=float)

    # model
    parser.add_argument("--n_channels", default=128, type=float)
    parser.add_argument("--sample_rate", default=16000, type=float)
    parser.add_argument("--n_fft", default=512, type=float)
    parser.add_argument("--f_min", default=0, type=float)
    parser.add_argument("--f_max", default=8000, type=float)
    parser.add_argument("--n_mels", default=128, type=float)
    parser.add_argument("--feat_dim", default=1024, type=float)
    parser.add_argument("--num_proj_layers", default=1, type=float)

    parser.add_argument("--audio_pretrained", default=False, type=bool)
    parser.add_argument("--image_pretrained", default=False, type=bool)

    # trainer
    parser.add_argument("--max_epochs", default=200, type=int)
    parser.add_argument("--reproduce", default=False, action="store_true")
    args = parser.parse_args()
    main(args)