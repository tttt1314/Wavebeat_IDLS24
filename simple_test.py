import os
import yaml
import glob
import json
import torch
import numpy as np
import torchsummary
from tqdm import tqdm
from itertools import product
import pytorch_lightning as pl
from argparse import ArgumentParser

#from wavebeat.tcn import TCNModel
from wavebeat.dstcn import dsTCNModel
#from wavebeat.waveunet import WaveUNetModel
from wavebeat.data_new import DownbeatDataset
from wavebeat.eval import evaluate

torch.backends.cudnn.benchmark = True

parser = ArgumentParser()

# add PROGRAM level args
parser.add_argument('--logdir', type=str, default='./', help='Path to pre-trained model log directory with checkpoint.')
parser.add_argument('--preload', action="store_true")
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--beatles_audio_dir', type=str, default='./data')
parser.add_argument('--beatles_annot_dir', type=str, default='./data')
parser.add_argument('--ballroom_audio_dir', type=str, default='./dataset/BallroomData')
parser.add_argument('--ballroom_annot_dir', type=str, default='./dataset/ISMIR2019/ballroom/annotations/beats')
parser.add_argument('--hainsworth_audio_dir', type=str, default='./dataset/Hainsworth/wavs')
parser.add_argument('--hainsworth_annot_dir', type=str, default='./dataset/ISMIR2019/hainsworth/annotations/beats')
parser.add_argument('--rwc_popular_audio_dir', type=str, default='./data')
parser.add_argument('--rwc_popular_annot_dir', type=str, default='./data')
parser.add_argument('--gtzan_audio_dir', type=str, default='./dataset/GTZAN/genres_original')
parser.add_argument('--gtzan_annot_dir', type=str, default='./dataset/ISMIR2019/gtzan/annotations/beats')
parser.add_argument('--smc_audio_dir', type=str, default='./dataset/SMC_MIREX/SMC_MIREX_Audio')
parser.add_argument('--smc_annot_dir', type=str, default='./dataset/ISMIR2019/smc/annotations/beats')
parser.add_argument('--rhythm_audio_dir', type=str, default='./dataset/rhythm/audio')
parser.add_argument('--rhythm_annot_dir', type=str, default='./dataset/rhythm/annote')
parser.add_argument('--ckpt_path', type=str, default='./')
parser.add_argument('--eval_length', type=int, default=2205000)
parser.add_argument('--juke',  action="store_true")

args = parser.parse_args()

# first out the model type from the yaml file
configfile = os.path.join(args.logdir, 'hparams.yaml')
print(configfile)
if os.path.isfile(configfile):
    with open(configfile) as fp:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        config = yaml.load(fp, Loader=yaml.FullLoader)
else:
    raise RuntimeError(f"No hparams.yaml file found in {args.logdir}.")

# find the checkpoint path
ckpts = glob.glob(os.path.join(args.logdir, "checkpoints", "*.ckpt"))
if len(ckpts) < 1:
    raise RuntimeError(f"No checkpoints found in {args.logdir}.")
else:
    ckpt_path = ckpts[-1]
ckpt_path = args.ckpt_path
# let the model add what it wants
#if config['model_type'] == 'tcn':
#    model = TCNModel.load_from_checkpoint(ckpt_path)
#elif config['model_type'] == 'lstm':
#    model = LSTMModel.load_from_checkpoint(ckpt_path)
#elif config['model_type'] == 'waveunet':
#    model = WaveUNetModel.load_from_checkpoint(ckpt_path)
if config['model_type'] == 'dstcn':
    model = dsTCNModel.load_from_checkpoint(ckpt_path)

# move model to GPU
model.to('cuda:0')
device = 'cuda:0'


#model= torch.nn.DataParallel(model)


# set model to eval mode
model.eval()

#datasets = ["beatles", "ballroom", "hainsworth", "rwc_popular", "gtzan", "smc"]
#datasets = ["ballroom", "hainsworth", "smc", "gtzan"]
#datasets = ["rhythm"]
#datasets = ["smc", "gtzan", "ballroom", "hainsworth", "rhythm"]
datasets = ["smc", "gtzan"]
results = {} # storage for our result metrics

# set the seed
pl.seed_everything(42)

# evaluate on each dataset using the test set
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
    elif dataset == "gtzan":
        audio_dir = args.gtzan_audio_dir
        annot_dir = args.gtzan_annot_dir
    elif dataset == "smc":
        audio_dir = args.smc_audio_dir
        annot_dir = args.smc_annot_dir
    elif dataset == "rhythm":
        audio_dir = args.rhythm_audio_dir
        annot_dir = args.rhythm_annot_dir

    test_dataset = DownbeatDataset(audio_dir,
                                    annot_dir,
                                    dataset=dataset,
                                    audio_sample_rate=config['audio_sample_rate'],
                                    target_factor=config['target_factor'],
                                    subset="test" if not dataset in ["gtzan", "smc"] else "full-val",
                                    #subset="full-val" if not dataset in ["gtzan", "smc"] else "full-val",
                                    augment=False,
                                    half=True if config['precision'] == 16 else False,
                                    preload=args.preload,
                                    length=args.eval_length,
                                    juke=args.juke
                                    )
    
    test_dataloader = torch.utils.data.DataLoader(test_dataset, 
                                                    shuffle=False,
                                                    batch_size=1,
                                                    num_workers=args.num_workers,
                                                    pin_memory=True)
    #print(test_dataset[113])

    #break
    # setup tracking of metrics
    results[dataset] = {
        "F-measure" : {
            "beat" : [],
            "dbn beat" : [],
            "downbeat" : [],    
            "dbn downbeat" : [],
        },
        "CMLt" : {
            "beat" : [],
            "dbn beat" : [],
            "downbeat" : [],
            "dbn downbeat" : [],
        },
        "AMLt" : {
            "beat" : [],
            "dbn beat" : [],
            "downbeat" : [],
            "dbn downbeat" : [],
        }
    }

    for example in tqdm(test_dataloader, ncols=80):
        audio, target, metadata = example
        # move data to GPU
        #audio = torch.nn.DataParallel(audio)
        #target = torch.nn.DataParallel(target)
        audio = audio.to(device)
        target = target.to(device)
        # print(audio.shape)
        # print(target.shape)

        with torch.no_grad():
            pred = torch.sigmoid(model(audio))

        # move data back to CPU
        pred = pred.cpu()
        target = target.cpu()
        
        beat_scores, downbeat_scores = evaluate(pred.view(2,-1),  
                                                target.view(2,-1), 
                                                model.hparams.target_sample_rate,
                                                use_dbn=False)
        
        # dbn_beat_scores, dbn_downbeat_scores = evaluate(pred.view(2,-1), 
        #                                         target.view(2,-1), 
        #                                         model.hparams.target_sample_rate,
        #                                         use_dbn=True)
        
        print()
        print(f"beat {beat_scores['F-measure']:0.3f} mean: {np.mean(results[dataset]['F-measure']['beat']):0.3f}  ")
        print(f"downbeat: {downbeat_scores['F-measure']:0.3f} mean: {np.mean(results[dataset]['F-measure']['downbeat']):0.3f}")

        results[dataset]['F-measure']['beat'].append(beat_scores['F-measure'])
        results[dataset]['CMLt']['beat'].append(beat_scores['Correct Metric Level Total'])
        results[dataset]['AMLt']['beat'].append(beat_scores['Any Metric Level Total'])

        # results[dataset]['F-measure']['dbn beat'].append(dbn_beat_scores['F-measure'])
        # results[dataset]['CMLt']['dbn beat'].append(dbn_beat_scores['Correct Metric Level Total'])
        # results[dataset]['AMLt']['dbn beat'].append(dbn_beat_scores['Any Metric Level Total'])

        results[dataset]['F-measure']['downbeat'].append(downbeat_scores['F-measure'])
        results[dataset]['CMLt']['downbeat'].append(downbeat_scores['Correct Metric Level Total'])
        results[dataset]['AMLt']['downbeat'].append(downbeat_scores['Any Metric Level Total'])

        # results[dataset]['F-measure']['dbn downbeat'].append(dbn_downbeat_scores['F-measure'])
        # results[dataset]['CMLt']['dbn downbeat'].append(dbn_downbeat_scores['Correct Metric Level Total'])
        # results[dataset]['AMLt']['dbn downbeat'].append(dbn_downbeat_scores['Any Metric Level Total'])
        del audio, target, metadata
        
        

    print()
    print(f"{dataset}")
    print(f"F1 beat: {np.mean(results[dataset]['F-measure']['beat'])}   F1 downbeat: {np.mean(results[dataset]['F-measure']['downbeat'])}")
    print(f"CMLt beat: {np.mean(results[dataset]['CMLt']['beat'])}   CMLt downbeat: {np.mean(results[dataset]['CMLt']['downbeat'])}")
    print(f"AMLt beat: {np.mean(results[dataset]['AMLt']['beat'])}   AMLt downbeat: {np.mean(results[dataset]['AMLt']['downbeat'])}")
    print()
    print(f"F1 dbn beat: {np.mean(results[dataset]['F-measure']['dbn beat'])}   F1 dbn downbeat: {np.mean(results[dataset]['F-measure']['dbn downbeat'])}")
    print(f"CMLt dbn  beat: {np.mean(results[dataset]['CMLt']['dbn beat'])}   CMLt dbn downbeat: {np.mean(results[dataset]['CMLt']['dbn downbeat'])}")
    print(f"AMLt dbn beat: {np.mean(results[dataset]['AMLt']['dbn beat'])}   AMLt dbn downbeat: {np.mean(results[dataset]['AMLt']['dbn downbeat'])}")
    print()

results_dir = 'results/test_retrain_juke.json'
with open(results_dir, 'w') as json_file:
    json.dump(results, json_file, sort_keys=True, indent=4) 
    print(f"Saved results to {results_dir}")