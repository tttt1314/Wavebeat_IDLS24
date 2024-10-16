import os
import sys
import glob
import torch 
import julius
import random
import torchaudio
import numpy as np
import scipy.signal
from tqdm import tqdm
import soxbindings as sox 
import librosa
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
#import jukemirlib


torchaudio.set_audio_backend("sox_io")

class DownbeatDataset(torch.utils.data.Dataset):
    """ Downbeat Dataset. """
    def __init__(self, 
                 audio_dir, 
                 annot_dir, 
                 audio_sample_rate=44100, 
                 target_factor=256,
                 dataset="ballroom",
                 subset="train", 
                 length=16384, 
                 preload=False, 
                 half=True, 
                 fraction=1.0,
                 augment=False,
                 dry_run=False,
                 pad_mode='constant',
                 examples_per_epoch=1000,
                 from_disk=False,
                 juke = False):
        """
        Args:
            audio_dir (str): Path to the root directory containing the audio (.wav) files.
            annot_dir (str): Path to the root directory containing the annotation (.beats) files.
            audio_sample_rate (float, optional): Sample rate of the audio files. (Default: 44100)
            target_factor (float, optional): Sample rate of the audio files. (Default: 256)
            subset (str, optional): Pull data either from "train", "val", "test", or "full-train", "full-val" subsets. (Default: "train")
            dataset (str, optional): Name of the dataset to be loaded "ballroom", "beatles", "hainsworth", "rwc_popular", "gtzan", "smc". (Default: "ballroom")
            length (int, optional): Number of samples in the returned examples. (Default: 40)
            preload (bool, optional): Read in all data into RAM during init. (Default: False)
            half (bool, optional): Store the float32 audio as float16. (Default: True)
            fraction (float, optional): Fraction of the data to load from the subset. (Default: 1.0)
            augment (bool, optional): Apply random data augmentations to input audio. (Default: False)
            dry_run (bool, optional): Train on a single example. (Default: False)
            pad_mode (str, optional): Padding type for inputs 'constant', 'reflect', 'replicate' or 'circular'. (Default: 'constant')
            examples_per_epoch (int, optional): Number of examples to sample from the dataset per epoch. (Default: 1000)

        Notes:
            - The SMC dataset contains only beats (no downbeats), so it should be used only for beat evaluation.
        """
        self.audio_dir = audio_dir
        self.annot_dir = annot_dir
        self.audio_sample_rate = audio_sample_rate
        self.target_factor = target_factor
        self.target_sample_rate = audio_sample_rate / target_factor
        self.subset = subset
        self.dataset = dataset
        self.length = length
        self.preload = preload
        self.half = half
        self.fraction = fraction
        self.augment = augment
        self.dry_run = dry_run
        self.pad_mode = pad_mode
        self.dataset = dataset
        self.examples_per_epoch = examples_per_epoch
        self.juke =juke

        self.target_length = int(self.length / self.target_factor)
        #print(f"Audio length: {self.length}")
        #print(f"Target length: {self.target_length}")

        # first get all of the audio files
        if self.dataset in ["beatles", "rwc_popular"]:
            file_ext = "*L+R.wav"
        elif self.dataset in ["ballroom", "hainsworth", "gtzan", "smc"]:
            file_ext = "*.wav"
        elif self.dataset in ["rhythm"]:
            file_ext = "*.mp3"
        else:
            raise ValueError(f"Invalid dataset: {self.dataset}")

        self.audio_files = glob.glob(os.path.join(self.audio_dir, "**", file_ext))
        if len(self.audio_files) == 0: # try from the root audio dir
            self.audio_files = glob.glob(os.path.join(self.audio_dir, file_ext))

        random.shuffle(self.audio_files) # shuffle them

        if self.subset == "train":
            start = 0
            stop = int(len(self.audio_files) * 0.8)
        elif self.subset == "val":
            start = int(len(self.audio_files) * 0.8)
            stop = int(len(self.audio_files) * 0.9)
        elif self.subset == "test":
            start = int(len(self.audio_files) * 0.9)
            stop = None
        elif self.subset in ["full-train", "full-val"]:
            start = 0
            stop = None

        # select one file for the dry run
        if self.dry_run: 
            self.audio_files = [self.audio_files[0]] * 50
            print(f"Selected 1 file for dry run.")
        else:
            # now pick out subset of audio files
            self.audio_files = self.audio_files[start:stop]
            print(f"Selected {len(self.audio_files)} files for {self.subset} set from {self.dataset} dataset.")

        self.annot_files = []
        self.audio_files_new = []
        self.annot_files_B = []
        self.annot_files_DB = []
        for audio_file in self.audio_files:
            # find the corresponding annot file
            if "jazz.00054" in audio_file:
                continue
            if self.dataset in ["rwc_popular", "beatles"]:
                replace = "_L+R.wav"
            elif self.dataset in ["ballroom", "hainsworth", "gtzan", "smc"]:
                replace = ".wav"
            elif self.dataset in ["rhythm"]:
                replace = ".mp3"

            
            filename = os.path.basename(audio_file).replace(replace, "")


            if self.dataset == "ballroom":
                path  = os.path.join(self.annot_dir, f"ballroom_{filename}.beats")
                if os.path.isfile(path):
                    self.annot_files.append(path)
                    self.audio_files_new.append(audio_file)
                else:
                    print("PATH not exist: ", path)
            elif self.dataset == "hainsworth":
                path  = os.path.join(self.annot_dir, f"hainsworth_{filename}.beats")
                if os.path.isfile(path):
                    self.annot_files.append(path)
                    self.audio_files_new.append(audio_file)
                else:
                    print("PATH not exist: ", path)
            elif self.dataset == "beatles":
                album_dir = os.path.basename(os.path.dirname(audio_file))
                annot_file = os.path.join(self.annot_dir, album_dir, f"{filename}.txt")
                self.annot_files.append(annot_file)

            elif self.dataset == "rwc_popular":
                album_dir = os.path.basename(os.path.dirname(audio_file))
                annot_file = os.path.join(self.annot_dir, album_dir, f"{filename}.BEAT.TXT")
                self.annot_files.append(annot_file)

            elif self.dataset == "gtzan":
                filename = filename.replace(".", "_")
                path = os.path.join(self.annot_dir, f"gtzan_{filename}.beats")
                if os.path.isfile(path) and os.path.isfile(audio_file):
                    self.annot_files.append(path)
                    self.audio_files_new.append(audio_file)
                elif not os.path.isfile(path):
                    print("PATH not exist: ", path)
                else: 
                    print("PATH not exist: ", audio_file)

                
            elif self.dataset == "smc":
                filename = filename.replace("SMC_", "")
                path = os.path.join(self.annot_dir, f"smc_{filename}.beats")
                if os.path.isfile(path):
                    self.annot_files.append(path)
                    self.audio_files_new.append(audio_file)
                else:
                    print("PATH not exist: ", path)

            elif self.dataset == "rhythm":
                #filename = filename.replace("SMC_", "")
                path_B = os.path.join(self.annot_dir, f"{filename}_beats.txt")
                path_DB = os.path.join(self.annot_dir, f"{filename}_downbeats.txt")
                if os.path.isfile(path_B):
                    self.annot_files_B.append(path_B)
                    self.audio_files_new.append(audio_file)
                else:
                    print("PATH not exist: ", path_B)

                if os.path.isfile(path_DB):
                    self.annot_files_DB.append(path_DB)
                else:
                    print("PATH not exist: ", path_DB)
                    assert(1, "no downbeat file")

        # when preloading store audio data and metadata
        RESUME = from_disk
        if self.preload and not RESUME:
            data = [] 
            if self.dataset == "rhythm":
                for audio_filename, annot_filename_B, annot_filename_DB in tqdm(zip(self.audio_files_new, self.annot_files_DB, self.annot_files_DB), 
                                                            total=len(self.audio_files_new), 
                                                            ncols=80):
                        
                        audio, target, metadata = self.load_data_rhythm(audio_filename, annot_filename_B, annot_filename_DB)
                        audio, target = self.resize(audio, target)
                        
                        if self.half:
                            audio = audio.half()
                            target = target.half()
                        data.append((audio, target, metadata))
                        #audios.append(audio)
                # audios = torch.vstack(audios)
                # print(audios)
                
                # torch.save(audios, "rhynme.pt")
            else: 
                for audio_filename, annot_filename in tqdm(zip(self.audio_files_new, self.annot_files), 
                                                            total=len(self.audio_files_new), 
                                                            ncols=80):
                        
                        audio, target, metadata = self.load_data(audio_filename, annot_filename)
                        audio, target = self.resize(audio, target)
                        if self.half:
                            audio = audio.half()
                            target = target.half()
                        data.append((audio, target, metadata))

            self.audios  = torch.stack([d[0]for d in data], dim=0)
            # audio_lengths = [len(d[0]) for d in data]
            self.targets = torch.stack([d[1] for d in data], dim=0)
            self.metadata = [d[2] for d in data]
            root = "./"
            if  not self.juke:
                torch.save(self.audios, root + self.dataset + "_audios_" + self.subset + ".pt")
                torch.save(self.targets, root + self.dataset + "_targets_" + self.subset + ".pt")
                torch.save(self.metadata,  root + self.dataset + "_metas_" + self.subset + ".pt")
            # target_lengths = [len(d[1]) for d in data]
            # self.audio_lengths = torch.Tensor(audio_lengths)
            # self.target_lengths = torch.Tensor(target_lengths)
        if RESUME:  
            root = "./dataset/data_32/"
            self.audios = torch.load(root + self.dataset + "_audios_" + self.subset + ".pt")
            self.targets = torch.load(root + self.dataset + "_targets_" + self.subset + ".pt")
            #print(self.targets[0])
            self.metadata = torch.load(root + self.dataset + "_metas_" + self.subset + ".pt")
            # audio_lengths = [len(d) for d in audios]
            # target_lengths = [len(d) for d in targets]
            # print("min_length_audio", min(audio_lengths))
            # print("min_length_target", min(target_lengths))
            # print("max_length_audio", max(audio_lengths))
            # print("max_length_target", max(target_lengths))
            # print("med_length_audio", np.median(np.array(audio_lengths)))
            # print("med_length_target", np.median(np.array(target_lengths)))
            #self.audios         = pad_sequence(self.audios, batch_first=True, padding_value=0)
            #self.targets = pad_sequence(self.targets, batch_first=True, padding_value=0)


            #self.metadata = [d[2] for d in data]
            # print("self.audios:", self.audios.shape)
            # print("self.targets:", self.targets.shape)
            # print("self.metadata:", len(self.metadata))
   


    def resize(self, audio, target):
        # do all processing in float32 not float16
        audio = audio.float()
        target = target.float()

        # apply augmentations 
        # if self.augment: 
        #     audio, target = self.apply_augmentations(audio, target)

        N_audio = audio.shape[-1]   # audio samples
        N_target = target.shape[-1] # target samples
        #N_audio = audio_length
        #N_target = target_length

        # random crop of the audio and target if larger than desired
        #print("N_audio:" ,  N_audio)
        #print("N_target:", N_target)
        if N_audio != N_target:
            if N_audio > N_target:
                pad_size = N_audio - N_target
                audio = torch.nn.functional.pad(target, 
                                            (0, pad_size), 
                                            mode=self.pad_mode)
            else: 
                pad_size = N_target - N_audio
                audio = torch.nn.functional.pad(audio, 
                                            (0, pad_size), 
                                            mode=self.pad_mode)              
        N_audio = audio.shape[-1]   # audio samples
        N_target = target.shape[-1] # target samples
        if (N_audio > self.length or N_target > self.target_length): # and self.subset not in ['val', 'test', 'full-val']:
            audio_start = np.random.randint(0, N_audio - self.length )
            audio_stop  = audio_start + self.length
            target_start = int(audio_start / self.target_factor)
            target_stop = int(audio_stop / self.target_factor)
            audio = audio[:,audio_start:audio_stop]
            target = target[:,target_start:target_stop]

        # pad the audio and target is shorter than desired
        if audio.shape[-1] < self.length: # and self.subset not in ['val', 'test', 'full-val']: 
            pad_size = self.length - audio.shape[-1]
            padl = pad_size - (pad_size // 2)
            padr = pad_size // 2
            audio = torch.nn.functional.pad(audio, 
                                            (padl, padr), 
                                            mode=self.pad_mode)
        if target.shape[-1] < self.target_length: # and self.subset not in ['val', 'test', 'full-val']: 
            pad_size = self.target_length - target.shape[-1]
            padl = pad_size - (pad_size // 2)
            padr = pad_size // 2
            target = torch.nn.functional.pad(target, 
                                             (padl, padr), 
                                             mode=self.pad_mode)
        assert(audio is not None)
        assert(target is not None)
        #assert(metadata is not None)
        return audio, target



    def __len__(self):
        if self.subset in ["test", "val", "full-val", "full-test"]:
            length = len(self.audio_files_new)
        else:
            length = self.examples_per_epoch
        return length

    def __getitem__(self, idx):

        if self.preload:
            #audio, target, metadata = self.data[idx % len(self.audio_files_new)]
            audio = self.audios[idx % (len(self.audio_files_new))]
            #audio_length = self.audio_lengths[idx % len(self.audio_files_new)]
            
            target = self.targets[idx % (len(self.audio_files_new))]
            #target_length = self.target_lengths[idx % len(self.audio_files_new)]
            
            metadata = self.metadata[idx % (len(self.audio_files_new))]
            if self.augment: 
                print("aug  ")
                audio, target = self.apply_augmentations(audio, target)
                audio, target = self.resize(audio, target)
            #audio = audio.transpose(0, 1)
            #target = target.transpose(0, 1)
            #print("audio:", audio.shape)
            #print("target:", target.shape)

            
            
        else:
            #eturn NotImplemented
            # get metadata of example
            if self.dataset == "rhythm":
                audio_filename = self.audio_files_new[(idx % len(self.audio_files_new))]
                annot_filename_B = self.annot_files_B[(idx % len(self.audio_files_new))]
                annot_filename_DB = self.annot_files_DB[(idx % len(self.audio_files_new))]
                audio, target, metadata = self.load_data_rhythm(audio_filename, annot_filename_B, annot_filename_DB)
                
            else:
                audio_filename = self.audio_files_new[(idx % len(self.audio_files_new))]
                annot_filename = self.annot_files[(idx % len(self.audio_files_new))]
                audio, target, metadata = self.load_data(audio_filename, annot_filename)

            audio, target = self.resize(audio, target)
        if self.half:
            audio = audio.half()
            target = target.half()
        #print("audio: ", audio)
        #print("target: ", target)
        if self.subset in ["train", "full-train"]:
            return audio, target
        elif self.subset in ["val", "test", "full-val"]:
            # this will only work with batch size = 1
            return audio, target, metadata
        else:
            raise RuntimeError(f"Invalid subset: `{self.subset}`")

 
        

    def load_data_rhythm(self, audio_filename, annot_filename_B, annot_filename_DB):
        # first load the audio file
        #print(audio_filename)
        #audio, sr = torchaudio.load(audio_filename, format="mp3") # audio = 2*N
        if self.juke:
            print("load from juke!!")

            if self.dataset in ["rwc_popular", "beatles"]:
                replace = "_L+R.wav"
            elif self.dataset in ["ballroom", "hainsworth", "gtzan", "smc"]:
                replace = ".wav"
            elif self.dataset in ["rhythm"]:
                replace = ".mp3"

            filename = os.path.basename(audio_filename).replace(replace, "")
            audio = np.load(os.path.join("./juke_feat", self.dataset, filename + ".npy"))
            audio = audio.transpose((1,0))
            audio = torch.Tensor(audio)
        else:
            audio, sr = librosa.load(audio_filename) # audio = 2*N
            print("rhythm:", audio.shape)
            audio = torch.Tensor(audio)
            if audio.shape[0] == 2:
                audio = torch.unsqueeze((audio[0, :] + audio[1, :])/2, 0)
            else:
                audio = torch.unsqueeze(audio, 0)
            #print(audio.shape)
            audio = audio.float()

            # resample if needed
            if sr != self.audio_sample_rate:
                audio = julius.resample_frac(audio, sr, self.audio_sample_rate)   

            # normalize all audio inputs -1 to 1
            audio /= audio.abs().max()

        # now get the annotation information
        annot_B = self.load_annot(annot_filename_B)
        beat_samples, _, _, _ = annot_B
        annot_DB = self.load_annot(annot_filename_DB)
        _, downbeat_samples, _, _ = annot_DB



        # get metadata
        genre = os.path.basename(os.path.dirname(audio_filename))

        # convert beat_samples to beat_seconds
        beat_sec = np.array(beat_samples) / self.audio_sample_rate
        downbeat_sec = np.array(downbeat_samples) / self.audio_sample_rate

        t = audio.shape[-1]/self.audio_sample_rate # audio length in sec
        N = int(t * self.target_sample_rate) + 1   # target length in samples
        target = torch.zeros(2,N)

        # now convert from seconds to new sample rate
        beat_samples = np.array(beat_sec * self.target_sample_rate)
        downbeat_samples = np.array(downbeat_sec * self.target_sample_rate)

        # check if there are any beats beyond the file end
        beat_samples = beat_samples[beat_samples < N]
        downbeat_samples = downbeat_samples[downbeat_samples < N]

        beat_samples = beat_samples.astype(int)
        downbeat_samples = downbeat_samples.astype(int)

        target[0,beat_samples] = 1  # first channel is beats
        target[1,downbeat_samples] = 1  # second channel is downbeats

        metadata = {
            "Filename" : audio_filename,
            "Genre" : genre,
            "Time signature" : "Nan"
        }

        return audio, target, metadata


    def load_data(self, audio_filename, annot_filename):
        # first load the audio file
        
        
        if self.juke:
            print("load from juke!!")

            if self.dataset in ["rwc_popular", "beatles"]:
                replace = "_L+R.wav"
            elif self.dataset in ["ballroom", "hainsworth", "gtzan", "smc"]:
                replace = ".wav"
            elif self.dataset in ["rhythm"]:
                replace = ".mp3"

            filename = os.path.basename(audio_filename).replace(replace, "")
            audio = np.load(os.path.join("./juke_feat", self.dataset, filename + ".npy"))
            audio = audio.transpose((1,0))
            #print("audio: ", audio.shape)
            audio = torch.Tensor(audio)
        else:
            audio, sr = torchaudio.load(audio_filename)
            audio = audio.float()

            # resample if needed
            if sr != self.audio_sample_rate:
                audio = julius.resample_frac(audio, sr, self.audio_sample_rate)   

            # normalize all audio inputs -1 to 1
            audio /= audio.abs().max()
        #print("non-rhythm:", audio.shape)
        # now get the annotation information
        annot = self.load_annot(annot_filename)
        beat_samples, downbeat_samples, beat_indices, time_signature = annot

        # get metadata
        genre = os.path.basename(os.path.dirname(audio_filename))

        # convert beat_samples to beat_seconds
        beat_sec = np.array(beat_samples) / self.audio_sample_rate
        downbeat_sec = np.array(downbeat_samples) / self.audio_sample_rate

        t = audio.shape[-1]/self.audio_sample_rate # audio length in sec
        N = int(t * self.target_sample_rate) + 1   # target length in samples
        target = torch.zeros(2,N)

        # now convert from seconds to new sample rate
        beat_samples = np.array(beat_sec * self.target_sample_rate)
        downbeat_samples = np.array(downbeat_sec * self.target_sample_rate)

        # check if there are any beats beyond the file end
        beat_samples = beat_samples[beat_samples < N]
        downbeat_samples = downbeat_samples[downbeat_samples < N]

        beat_samples = beat_samples.astype(int)
        downbeat_samples = downbeat_samples.astype(int)

        target[0,beat_samples] = 1  # first channel is beats
        target[1,downbeat_samples] = 1  # second channel is downbeats

        metadata = {
            "Filename" : audio_filename,
            "Genre" : genre,
            "Time signature" : time_signature
        }

        return audio, target, metadata

    def load_annot(self, filename):

        with open(filename, 'r') as fp:
            lines = fp.readlines()
        
        beat_samples = [] # array of samples containing beats
        downbeat_samples = [] # array of samples containing downbeats (1)
        beat_indices = [] # array of beat type one-hot encoded  
        time_signature = "?" # estimated time signature (only 3/4 or 4/4)

        for line in lines:
            if self.dataset == "ballroom":
                line = line.strip('\n')
                line = line.replace('\t', ' ')
                time_sec, beat = line.split(' ')
            elif self.dataset == "beatles":
                line = line.strip('\n')
                line = line.replace('\t', ' ')
                line = line.replace('  ', ' ')
                time_sec, beat = line.split(' ')
            elif self.dataset == "hainsworth":
                line = line.strip('\n')
                line = line.replace('\t', ' ')
                time_sec, beat = line.split(' ')
            elif self.dataset == "rwc_popular":
                line = line.strip('\n')
                line = line.split('\t')
                time_sec = int(line[0]) / 100.0
                beat = 1 if int(line[2]) == 384 else 2
            elif self.dataset == "gtzan":
                line = line.strip('\n')
                line = line.replace('\t', ' ')
                if len(line.split(' ')) == 2:
                    time_sec, beat = line.split(' ')
                else:
                    time_sec = line
                    beat = 1
            elif self.dataset == "smc":
                line = line.strip('\n')
                time_sec = line
                beat = 1
            elif self.dataset == "rhythm":
                line = line.strip('\n')
                time_sec = line
                time_sec = float(time_sec)/1000.0
                beat = 1

            # convert beat to one-hot
            beat = int(beat)
            if beat == 1:
                beat_one_hot = [1,0,0,0]
            elif beat == 2:
                beat_one_hot = [0,1,0,0]
            elif beat == 3:
                beat_one_hot = [0,0,1,0]    
            elif beat == 4:
                beat_one_hot = [0,0,0,1]

            # convert seconds to samples

            beat_time_samples = int(float(time_sec) * (self.audio_sample_rate))

            beat_samples.append(beat_time_samples)
            beat_indices.append(beat)

            if beat == 1:
                downbeat_time_samples = int(float(time_sec) * (self.audio_sample_rate))
                downbeat_samples.append(downbeat_time_samples)

        # guess at the time signature
        if len(beat_indices) == 0:
            time_signature = "?"
        elif np.max(beat_indices) == 2:
            time_signature = "2/4"
        elif np.max(beat_indices) == 3:
            time_signature = "3/4"
        elif np.max(beat_indices) == 4:
            time_signature = "4/4"

            

        return beat_samples, downbeat_samples, beat_indices, time_signature

    def apply_augmentations(self, audio, target):

        # random gain from 0dB to -6 dB
        #if np.random.rand() < 0.2:      
        #    #sgn = np.random.choice([-1,1])
        #    audio = audio * (10**((-1 * np.random.rand() * 6)/20))   
        #audio = audio.squeeze().numpy().astype('float32')
        # phase inversion
        if np.random.rand() < 0.5:      
            audio = -audio                              

        # drop continguous frames
        if np.random.rand() < 0.05:     
            zero_size = int(self.length*0.1)
            start = np.random.randint(audio.shape[-1] - zero_size - 1)
            stop = start + zero_size
            audio[:,start:stop] = 0
            target[:,start:stop] = 0

        # apply time stretching
        if np.random.rand() < 0.0:
            factor = np.random.normal(1.0, 0.5)  
            factor = np.clip(factor, a_min=0.6, a_max=1.8)

            tfm = sox.Transformer()        

            if abs(factor - 1.0) <= 0.1: # use stretch
                tfm.stretch(1/factor)
            else:   # use tempo
                tfm.tempo(factor, 'm')

            audio = tfm.build_array(input_array=audio.squeeze().numpy().astype('float32'), 
                                    sample_rate_in=self.audio_sample_rate)
            audio = torch.from_numpy(audio.astype('float32')).view(1,-1)

            # now we update the targets based on new tempo
            dbeat_ind = (target[1,:] == 1).nonzero(as_tuple=False)
            dbeat_sec = dbeat_ind / self.target_sample_rate
            new_dbeat_sec = (dbeat_sec / factor).squeeze()
            new_dbeat_ind = (new_dbeat_sec * self.target_sample_rate).long()

            beat_ind = (target[0,:] == 1).nonzero(as_tuple=False)
            beat_sec = beat_ind / self.target_sample_rate
            new_beat_sec = (beat_sec / factor).squeeze()
            new_beat_ind = (new_beat_sec * self.target_sample_rate).long()

            # now convert indices back to target vector
            new_size = int(np.ceil(target.shape[-1] / factor))
            streteched_target = torch.zeros(2,new_size)
            streteched_target[0,new_beat_ind] = 1
            streteched_target[1,new_dbeat_ind] = 1
            target = streteched_target

        if np.random.rand() < 0.0:
            # this is the old method (shift all beats)
            max_shift = int(0.070 * self.target_sample_rate)
            shift = np.random.randint(0, high=max_shift)
            direction = np.random.choice([-1,1])
            target = torch.roll(target, shift * direction)

        # shift targets forward/back max 70ms
        if np.random.rand() < 0.3:      
            
            # in this method we shift each beat and downbeat by a random amount
            max_shift = int(0.045 * self.target_sample_rate)

            beat_ind = torch.logical_and(target[0,:] == 1, target[1,:] != 1).nonzero(as_tuple=False) # all beats EXCEPT downbeats
            dbeat_ind = (target[1,:] == 1).nonzero(as_tuple=False)

            # shift just the downbeats
            dbeat_shifts = torch.normal(0.0, max_shift/2, size=(1,dbeat_ind.shape[-1]))
            dbeat_ind += dbeat_shifts.long()

            # now shift the non-downbeats 
            beat_shifts = torch.normal(0.0, max_shift/2, size=(1,beat_ind.shape[-1]))
            beat_ind += beat_shifts.long()

            # ensure we have no beats beyond max index
            beat_ind = beat_ind[beat_ind < target.shape[-1]]
            dbeat_ind = dbeat_ind[dbeat_ind < target.shape[-1]]  

            # now convert indices back to target vector
            shifted_target = torch.zeros(2,target.shape[-1])
            shifted_target[0,beat_ind] = 1
            shifted_target[0,dbeat_ind] = 1 # set also downbeats on first channel
            shifted_target[1,dbeat_ind] = 1

            target = shifted_target
    
        # apply pitch shifting
        if np.random.rand() < 0.5:
            sgn = np.random.choice([-1,1])
            factor = sgn * np.random.rand() * 8.0     
            tfm = sox.Transformer()        
            tfm.pitch(factor)
            audio = tfm.build_array(input_array=audio.squeeze().numpy().astype('float32'), 
                                    sample_rate_in=self.audio_sample_rate)
            audio = torch.from_numpy(audio.astype('float32')).view(1,-1)

        # apply a lowpass filter
        if np.random.rand() < 0.25:
            cutoff = (np.random.rand() * 4000) + 4000
            sos = scipy.signal.butter(2, 
                                      cutoff, 
                                      btype="lowpass", 
                                      fs=self.audio_sample_rate, 
                                      output='sos')
            audio_filtered = scipy.signal.sosfilt(sos, audio.numpy().astype('float32'))
            audio = torch.from_numpy(audio_filtered.astype('float32'))

        # apply a highpass filter
        if np.random.rand() < 0.25:
            cutoff = (np.random.rand() * 1000) + 20
            sos = scipy.signal.butter(2, 
                                      cutoff, 
                                      btype="highpass", 
                                      fs=self.audio_sample_rate, 
                                      output='sos')
            audio_filtered = scipy.signal.sosfilt(sos, audio.numpy().astype('float32'))
            audio = torch.from_numpy(audio_filtered.astype('float32'))

        # apply a chorus effect
        if np.random.rand() < 0:
            tfm = sox.Transformer()        
            tfm.chorus()
            audio = tfm.build_array(input_array=audio.squeeze().numpy().astype('float32'), 
                                    sample_rate_in=self.audio_sample_rate)
            audio = torch.from_numpy(audio.astype('float32')).view(1,-1)

        # apply a compressor effect
        if np.random.rand() < 0:
            attack = (np.random.rand() * 0.300) + 0.005
            release = (np.random.rand() * 1.000) + 0.3
            tfm = sox.Transformer()        
            tfm.compand(attack_time=attack, decay_time=release)
            audio = tfm.build_array(input_array=audio.squeeze().numpy().astype('float32'), 
                                    sample_rate_in=self.audio_sample_rate)
            audio = torch.from_numpy(audio.astype('float32')).view(1,-1)

        # apply an EQ effect
        if np.random.rand() < 0:
            freq = (np.random.rand() * 8000) + 60
            q = (np.random.rand() * 7.0) + 0.1
            g = np.random.normal(0.0, 6)  
            tfm = sox.Transformer()        
            tfm.equalizer(frequency=freq, width_q=q, gain_db=g)
            audio = tfm.build_array(input_array=audio.squeeze().numpy().astype('float32'), 
                                    sample_rate_in=self.audio_sample_rate)
            audio = torch.from_numpy(audio.astype('float32')).view(1,-1)

        # add white noise
        if np.random.rand() < 0.05:
            wn = (torch.rand(audio.shape) * 2) - 1
            g = 10**(-(np.random.rand() * 20) - 12)/20
            audio = audio + (g * wn)

        audio = audio.to(torch.float32)

        # apply nonlinear distortion 
        if np.random.rand() < 0.2:   
            g = 10**((np.random.rand() * 12)/20)   
            audio = torch.tanh(audio)    
        
        # normalize the audio
        audio /= audio.float().abs().max()

        return audio, target