# Copyright (c) Megvii Inc. All rights reserved.
import os
from argparse import ArgumentParser, Namespace

import pytorch_lightning as pl
import torch
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
from pytorch_lightning.callbacks import ModelCheckpoint

from callbacks.ema import EMACallback
from exps.bev_depth_lss_r50_256x704_128x128_24e import \
    BEVDepthLightningModel as BaseBEVDepthLightningModel
    
from utils.backup_files import backup_codebase

class BEVDepthLightningModel(BaseBEVDepthLightningModel):
    def configure_optimizers(self):
        lr = self.basic_lr_per_img * \
            self.batch_size_per_device * self.gpus
        optimizer = torch.optim.AdamW(self.model.parameters(),
                                      lr=lr,
                                      weight_decay=1e-7)
        return [optimizer]


def main(args: Namespace) -> None:
    if args.seed is not None:
        pl.seed_everything(args.seed)

    model = BEVDepthLightningModel(**vars(args))
    checkpoint_callback = ModelCheckpoint(dirpath='./outputs/bev_depth_lss_r50_256x704_128x128_24e_ema/checkpoints', filename='{epoch}', every_n_epochs=5, save_last=True, save_top_k=-1)
    train_dataloader = model.train_dataloader()
    ema_callback = EMACallback(len(train_dataloader.dataset) * args.max_epochs)
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback, ema_callback])
    if args.evaluate:
        for ckpt_name in os.listdir(args.ckpt_path):
            model_pth = os.path.join(args.ckpt_path, ckpt_name)
            trainer.test(model, ckpt_path=model_pth)
    else:
        backup_codebase()
        trainer.fit(model)


def run_cli():
    parent_parser = ArgumentParser(add_help=False)
    parent_parser = pl.Trainer.add_argparse_args(parent_parser)
    parent_parser.add_argument('-e',
                               '--evaluate',
                               dest='evaluate',
                               action='store_true',
                               help='evaluate model on validation set')
    parent_parser.add_argument('-b', '--batch_size_per_device', type=int)
    parent_parser.add_argument('--seed',
                               type=int,
                               default=0,
                               help='seed for initializing training.')
    parent_parser.add_argument('--ckpt_path', type=str)
    parser = BEVDepthLightningModel.add_model_specific_args(parent_parser)
    parser.set_defaults(
        profiler='simple',
        deterministic=False,
        max_epochs=50,
        accelerator='ddp',
        num_sanity_val_steps=0,
        gradient_clip_val=5,
        # limit_val_batches=0,
        enable_checkpointing=True,
        precision=32,
        default_root_dir='./outputs/bev_depth_lss_r50_256x704_128x128_24e_ema')
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    run_cli()
