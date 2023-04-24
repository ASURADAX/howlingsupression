import torch as t
import hyperparams as hp
import numpy as np
from network import ModelPostNet, Model
from collections import OrderedDict
from tqdm import tqdm
import soundfile as sf
import os
from testdataset import get_testset, DataLoader, collate_fn_transformer_test
from torchsummary import summary


def load_checkpoint(step, model_name="transformer"):
    state_dict = t.load('./checkpoint/checkpoint_%s_%d.pth.tar'% (model_name, step),map_location=t.device('cpu'))   
    new_state_dict = OrderedDict()
    for k, value in state_dict['model'].items():
        key = k[7:]
        new_state_dict[key] = value

    return new_state_dict

def model_vis():
    m = Model()
    m_post = ModelPostNet()

    #m.load_state_dict(load_checkpoint(98650, "transformer"))
    #m_post.load_state_dict(load_checkpoint(98340, "postnet"))
    #summary(m,input_size=[(2,80),(2,80)])
    summary(m_post,input_size=[(292,80)])
    
if __name__ == '__main__':
    model_vis()
