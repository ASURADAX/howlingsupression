import torch as t
from pesq import pesq
from pystoi import stoi
from utils import spectrogram2wav
#from scipy.io.wavfile import write
import hyperparams as hp
import numpy as np
from network import ModelPostNet, Model
from collections import OrderedDict
from tqdm import tqdm
import resampy
import soundfile as sf
import os
from testdataset import get_testset, DataLoader, collate_fn_transformer_test
from tensorboardX import SummaryWriter

def load_checkpoint(step, model_name="transformer"):
    state_dict = t.load('./checkpoint/checkpoint_%s_%d.pth.tar'% (model_name, step),map_location=t.device('cpu'))   
    new_state_dict = OrderedDict()
    for k, value in state_dict['model'].items():
        key = k[7:]
        new_state_dict[key] = value
 
    return new_state_dict

def synthesis():
    writer = SummaryWriter('mse_e_logs')

    m = Model()
    m_post = ModelPostNet()
    #98650 l1loss
    #99910 mseloss
    m.load_state_dict(load_checkpoint(99910, "transformer"))
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
            writer.add_scalars('metrics',{
                    'stoi':st,
                    'pesq':pe,
                    'snr':sn,
                }, count)
            #将wav数据保存为wav文件
            sf.write(os.path.join("./dnn/samples", 'c_{}.wav'.format(i)),target_wav_restored.transpose(), hp.sr)
            sf.write(os.path.join("./dnn/samples", 'h_{}.wav'.format(i)), wav_restored.transpose(), hp.sr)
        avg_eval_loss = total_eval_loss / count
        return avg_eval_loss, total_stoi / count, total_pesq / count, total_snr / count

###
#helper functions
###
def get_pesq(ref, deg, sr):
    ref = resampy.resample(ref, sr_orig=hp.sr, sr_new=16000)
    deg = resampy.resample(deg, sr_orig=hp.sr, sr_new=16000)
    score = pesq(16000, ref, deg, 'wb')

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

if __name__ == '__main__':
    loss,st,pe,sn = synthesis()
    print('=======')
    print(loss) #0.008159851946402341 #0.006610008410643786
    print(st) #0.17373112025881976 #0.18716116640033464
    print(pe) #1.062848440806071 #1.1307512164115905
    print(sn) #-3.588075748334328 #-2.583868389328321
    print('=======')