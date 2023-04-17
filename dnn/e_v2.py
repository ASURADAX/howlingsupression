import torch as t
from pesq import pesq
from pystoi import stoi
from utils import spectrogram2wav
from scipy.io.wavfile import write
import hyperparams as hp
from text import text_to_sequence
import numpy as np
from network import ModelPostNet, Model
from collections import OrderedDict
from tqdm import tqdm
import argparse
from testdataset import get_testset, DataLoader, collate_fn_transformer_test

def load_checkpoint(step, model_name="transformer"):
    state_dict = t.load('./checkpoint/checkpoint_%s_%d.pth.tar'% (model_name, step))   
    new_state_dict = OrderedDict()
    for k, value in state_dict['model'].items():
        key = k[7:]
        new_state_dict[key] = value

    return new_state_dict

def synthesis():
    m = Model()
    m_post = ModelPostNet()

    m.load_state_dict(load_checkpoint(98650, "transformer"))
    m_post.load_state_dict(load_checkpoint(98340, "postnet"))

    dataset = get_testset()
    #batch_size must set 1
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn_transformer_test,drop_last=False, num_workers=1)

    pbar = tqdm(dataloader)
    m.eval()
    m_post.eval()
    total_stoi = 0.0
    total_snr = 0.0
    total_pesq = 0.0
    count, total_eval_loss = 0, 0.0
    with t.no_grad():
        for i, data in enumerate(pbar):
            #模型预测
            source_mel, source_mel_input, source_pos_mel, target_wav = data
            mel_pred, postnet_pred, attn_probs, stop_preds, attns_enc, attns_dec = m.inference(source_mel_input, source_pos_mel)
            mag_pred = m_post.forward(postnet_pred)
            wav = spectrogram2wav(mag_pred.squeeze(0).cpu().numpy())
            #将wav数据转化为tensor
            wav_tensor = t.FloatTensor(wav)

            wav_tensor = wav_tensor.squeeze()
            target_wav = target_wav.squeeze()
            # keep length same (output label)
            wavlen=min(wav_tensor.shape[-1],target_wav.shape[-1])
            wav_tensor = wav_tensor[:wavlen]
            target_wav = target_wav[:wavlen]
            # 估计tensor损失
            eval_loss = t.mean((wav_tensor - target_wav) ** 2)
            total_eval_loss += eval_loss.data.item()
            #将tensor还原为wav数据
            wav_restored = wav_tensor.numpy()
            target_wav_restored = target_wav.numpy()
            # 估计wav标准
            st = get_stoi(target_wav_restored, wav_restored, hp.sr)
            pe = get_pesq(target_wav_restored, wav_restored, hp.sr)
            sn = snr(target_wav_restored, wav_restored)
            total_pesq += pe
            total_snr += sn
            total_stoi += st

            count += 1
            #将wav数据保存为wav文件
            #write(hp.sample_path + "/test.wav", hp.sr, target_wav_restored)
        avg_eval_loss = total_eval_loss / count
        return avg_eval_loss, total_stoi / count, total_pesq / count, total_snr / count

###
#helper functions
###
def get_pesq(ref, deg, sr):

    score = pesq(sr, ref, deg, 'wb')

    return score

def get_stoi(ref, deg, sr):

    score = stoi(ref, deg, sr, extended=False)

    return score

def snr(s, s_p):
    r""" calculate signal-to-noise ratio (SNR)

        Parameters
        ----------
        s: clean speech
        s_p: processed speech
    """
    return 10.0 * np.log10(np.sum(s ** 2) / np.sum((s_p - s) ** 2))