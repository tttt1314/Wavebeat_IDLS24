import os
import glob
import torch
import torchsummary
from itertools import product
import pytorch_lightning as pl
from argparse import ArgumentParser

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
#from lightning.pytorch.callbacks import ModelCheckpoint
#from wavebeat.tcn import TCNModel
from wavebeat.dstcn import dsTCNModel
#from wavebeat.lstm import LSTMModel
#from wavebeat.waveunet import WaveUNetModel
from wavebeat.data_new import DownbeatDataset
from lightning.pytorch.loggers import WandbLogger
import wandb
#import jukemirlib


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    #jukemirlib.setup_models(device="cuda:1")
    parser = ArgumentParser()

    # add PROGRAM level args
    parser.add_argument('--model_type', type=str, default='dstcn', help='tcn, lstm, waveunet, or dstcn')
    parser.add_argument('--dataset', type=str, default='ballroom')
    parser.add_argument('--beatles_audio_dir', type=str, default='./data')
    parser.add_argument('--beatles_annot_dir', type=str, default='./data')
    parser.add_argument('--ballroom_audio_dir', type=str, default='./dataset/BallroomData')
    parser.add_argument('--ballroom_annot_dir', type=str, default='./dataset/ISMIR2019/ballroom/annotations/beats')
    parser.add_argument('--hainsworth_audio_dir', type=str, default='./dataset/Hainsworth/wavs')
    parser.add_argument('--hainsworth_annot_dir', type=str, default='./dataset/ISMIR2019/hainsworth/annotations/beats')
    parser.add_argument('--rwc_popular_audio_dir', type=str, default='./data')
    parser.add_argument('--rwc_popular_annot_dir', type=str, default='./data')
    parser.add_argument('--preload', action="store_true")
    parser.add_argument('--audio_sample_rate', type=float, default=22050)
    parser.add_argument('--target_factor', type=int, default=1) #256
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--train_subset', type=str, default='train')
    parser.add_argument('--val_subset', type=str, default='val')
    parser.add_argument('--train_length', type=int, default=65536)
    parser.add_argument('--train_fraction', type=float, default=1.0)
    parser.add_argument('--eval_length', type=int, default=131072)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--rhythm_audio_dir', type=str, default='./dataset/rhythm/audio')
    parser.add_argument('--rhythm_annot_dir', type=str, default='./dataset/rhythm/annote_v2')
    #parser.add_argument('--accelerator', type=str, default='gpu')
    #parser.add_argument('--devices', type=int, default=2)
    parser.add_argument('--from_disk',  action="store_true")
    parser.add_argument('--dropout',  type=float, default = 0.2)
    parser.add_argument('--juke',  action="store_true")
    parser.add_argument('--ckpt_dir',   type=str, default = './')

    # add all the available trainer options to argparse
    parser = pl.Trainer.add_argparse_args(parser)
    
    # THIS LINE IS KEY TO PULL THE MODEL NAME
    temp_args, _ = parser.parse_known_args()

    if temp_args.model_type == 'dstcn':
        parser = dsTCNModel.add_model_specific_args(parser)
    else:
        raise RuntimeError(f"Invalid model_type: {temp_args.model_type}")
    
    # parse them args
    args = parser.parse_args()

    checkpoint_callback = ModelCheckpoint(
        verbose=True,
        monitor='val_loss/Joint F-measure',
        mode='max',
        dirpath=args.ckpt_dir
    )
    #######
    RESUME = 0
    LOAD = 0
    #######


    
    

    

    # let the model add what it wants
    #if temp_args.model_type == 'tcn':
    #    parser = TCNModel.add_model_specific_args(parser)
    #elif temp_args.model_type == 'lstm':
    #    parser = LSTMModel.add_model_specific_args(parser)
    #elif temp_args.model_type == 'waveunet':
    #    parser = WaveUNetModel.add_model_specific_args(parser)
    



    #datasets = ["beatles", "ballroom", "hainsworth", "rwc_popular"]
    datasets = ["hainsworth", "ballroom", "rhythm"  ]
    #datasets = [ "ballroom", "hainsworth"]
    #datasets = ["hainsworth"]
    #datasets = ["rhythm"]
    # set the seed
    pl.seed_everything(42)

    #
    args.default_root_dir = os.path.join("lightning_logs", "full")
    print(args.default_root_dir)

    # create the trainer
    wandb.login(key="c5af30930f4826bc232d6a9319422ed109b9e3c3")
    wandb_logger = WandbLogger(project="Wavebeat", log_model="all", name="no rhythm")
    #trainer = pl.Trainer.from_argparse_args(args, checkpoint_callback=checkpoint_callback, logger=wandb_logger)
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback], logger=wandb_logger) #1.8.6
    #trainer = pl.Trainer(**vars(args), checkpoint_callback=checkpoint_callback, logger=wandb_logger)
    # setup the dataloaders
    train_datasets = []
    val_datasets = []

    for dataset in datasets:
        if dataset == "beatles":
            audio_dir = args.beatles_audio_dir
            annot_dir = args.beatles_annot_dir
        elif dataset == "ballroom":
            audio_dir = args.ballroom_audio_dir
            annot_dir = args.ballroom_annot_dir
        elif dataset == "hainsworth":
            audio_dir = args.hainsworth_audio_dir
            annot_dir = args.hainsworth_annot_dir
        elif dataset == "rwc_popular":
            audio_dir = args.rwc_popular_audio_dir
            annot_dir = args.rwc_popular_annot_dir
        elif dataset == "rwc_popular":
            audio_dir = args.rwc_popular_audio_dir
            annot_dir = args.rwc_popular_annot_dir
        elif dataset == "rhythm":
            audio_dir = args.rhythm_audio_dir
            annot_dir = args.rhythm_annot_dir

        print("data loading...")
        print("juke: ", args.juke)
        train_dataset = DownbeatDataset(audio_dir,
                                        annot_dir,
                                        dataset=dataset,
                                        audio_sample_rate=args.audio_sample_rate,
                                        target_factor=args.target_factor,
                                        subset="train",
                                        fraction=args.train_fraction,
                                        augment=args.augment,
                                        half=True if args.precision == 16 else False,
                                        preload=args.preload,
                                        length=args.train_length,
                                        dry_run=args.dry_run,
                                        from_disk=args.from_disk,
                                        juke=args.juke)
        train_datasets.append(train_dataset)
        
        val_dataset = DownbeatDataset(audio_dir,
                                    annot_dir,
                                    dataset=dataset,
                                    audio_sample_rate=args.audio_sample_rate,
                                    target_factor=args.target_factor,
                                    subset="val",
                                    augment=False,
                                    half   =True if args.precision == 16 else False,
                                    preload=args.preload,
                                    length=args.eval_length,
                                    dry_run=args.dry_run,
                                    from_disk=args.from_disk,
                                    juke=args.juke)
        val_datasets.append(val_dataset)

    train_dataset_list = torch.utils.data.ConcatDataset(train_datasets)
    val_dataset_list = torch.utils.data.ConcatDataset(val_datasets)


    train_dataloader = torch.utils.data.DataLoader(train_dataset_list, 
                                                    shuffle=args.shuffle,
                                                    batch_size=args.batch_size,
                                                    num_workers=args.num_workers,
                                                    pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset_list, 
                                                shuffle=False,
                                                batch_size=1,
                                                num_workers=args.num_workers,
                                                pin_memory=True)    

    # create the model with args
    dict_args = vars(args)
    dict_args["nparams"] = 2
    dict_args["target_sample_rate"] = args.audio_sample_rate / args.target_factor

    if args.model_type == 'tcn':
        model = TCNModel(**dict_args)
        rf = model.compute_receptive_field()
        print(f"Model has receptive field of {(rf/args.sample_rate)*1e3:0.1f} ms ({rf}) samples")
    elif args.model_type == 'lstm':
        model = LSTMModel(**dict_args)
    elif args.model_type == 'waveunet':
        model = WaveUNetModel(**dict_args)
    elif args.model_type == 'dstcn':
        model = dsTCNModel(**dict_args)

    # summary 


    if args.juke:
        torchsummary.summary(model, [(4800,args.train_length)], device="cpu")
        print("Training!")
    else:
        torchsummary.summary(model, [(1,args.train_length)], device="cpu")
        print("Training!")
    # train!
    if LOAD: 
        print("resume from checkpoint!")
        trainer.fit(model, train_dataloader, val_dataloader, ckpt_path="./log/checkpoints/wavebeat_epoch=98-step=24749.ckpt")
    else:
        trainer.fit(model, train_dataloader, val_dataloader)
#trainer.validate(model=model, dataloaders=val_dataloader)


