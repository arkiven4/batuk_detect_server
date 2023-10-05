from flask import Flask, send_file, request, jsonify, render_template
import torch
from models import get_backbone_class
from copy import deepcopy

import os
import librosa
import torchaudio
from torchaudio import transforms as T

from util.icbhi_util import generate_fbank

from torchvision import transforms
from util.augmentation import SpecAugment

import math
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]=""

args = {"sample_rate": 16000, "desired_length" : 8, "pad_types": "repeat", "specaug_policy":'icbhi_ast_sup',"specaug_mask":'mean'}

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

args = dotdict(args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args.h, args.w = 532, 128
train_transform = [transforms.ToTensor(),
                    SpecAugment(args),
                    transforms.Resize(size=(int(args.h * 1), int(args.w * 1)))]
train_transform = transforms.Compose(train_transform)

def cut_pad_sample_torchaudio(data, args):
    fade_samples_ratio = 16
    fade_samples = int(args.sample_rate / fade_samples_ratio)
    fade_out = T.Fade(fade_in_len=0, fade_out_len=fade_samples, fade_shape='linear')
    target_duration = args.desired_length * args.sample_rate

    if data.shape[-1] > target_duration:
        data = data[..., :target_duration]
    else:
        if args.pad_types == 'zero':
            tmp = torch.zeros(1, target_duration, dtype=torch.float32)
            diff = target_duration - data.shape[-1]
            tmp[..., diff//2:data.shape[-1]+diff//2] = data
            data = tmp
        elif args.pad_types == 'repeat':
            ratio = math.ceil(target_duration / data.shape[-1])
            data = data.repeat(1, ratio)
            data = data[..., :target_duration]
            data = fade_out(data)
    
    return data

kwargs = {}
kwargs['input_fdim'] = int(532 * 1)
kwargs['input_tdim'] = int(128 * 1)
kwargs['label_dim'] = 3
kwargs['imagenet_pretrain'] = False
kwargs['audioset_pretrain'] = False
kwargs['mix_beta'] = 1.0

model = get_backbone_class("ast")(**kwargs)
ckpt = torch.load("best.pth", map_location='cpu')
model.load_state_dict(ckpt['model'], strict=False)

classifier = deepcopy(model.mlp_head)
classifier.load_state_dict(ckpt['classifier'], strict=True)
#classifier.to(device)

model.eval()
classifier.eval()

print("Ready.....")

app = Flask(__name__)

@app.route('/pred_audio/<uuid>')
def send_audio(uuid):
    fpath = os.path.join("/run/media/viblab/Markov2/Pras/PKM/TB/public_dataset", f'{uuid}.wav')
    
    sr = librosa.get_samplerate(fpath)
    data, _ = torchaudio.load(fpath)

    if sr != 16000:
        resample = T.Resample(sr, 16000)
        data = resample(data)

    fade_samples_ratio = 16
    fade_samples = int(16000 / fade_samples_ratio)

    fade = T.Fade(fade_in_len=fade_samples, fade_out_len=fade_samples, fade_shape='linear')
    data = fade(data)
    data = cut_pad_sample_torchaudio(data, args)

    image = generate_fbank(data, args.sample_rate, n_mels=128)

    audio_image = train_transform(image)
    audio_images = torch.stack([audio_image])

    with torch.no_grad():
        features = model(audio_images)
        output = classifier(features)

    class_name = ['COVID-19', 'healthy', 'symptomatic']
    return jsonify({"status": "success", "prediction": class_name[np.argmax(output.cpu().numpy())]})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")