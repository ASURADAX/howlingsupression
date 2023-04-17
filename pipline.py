from torch.utils.data import DataLoader
import m2w
import howlinggenerator as hg
import dnn.prepare_data as w2pt
import os
import glob
import shutil

if __name__ == '__main__':
    #mp3 to wav
    wav_path = './wav_1'
    mp3_path = './mp3_1'
    dataset = m2w.PreprocessDataset(mp3_path,wav_path)
    dataloader = DataLoader(dataset, batch_size=1, drop_last=False, num_workers=1)
    from tqdm import tqdm
    pbar = tqdm(dataloader)
    for d in pbar:
        pass
    #generate howling
    source_path = './wav_1'
    target_path = './howl_1'
    hg.add_howling(source_path,target_path)
    #get mel and mag file for clean wav
    wav_path = './wav_1'
    dataset = w2pt.PrepareDataset(wav_path)
    dataloader = DataLoader(dataset, batch_size=1, drop_last=False, num_workers=8)
    from tqdm import tqdm
    pbar = tqdm(dataloader)
    for d in pbar:
        pass
    #get mel and mag file for howling wav
    wav_path = './howl_1'
    dataset = w2pt.PrepareDataset(wav_path)
    dataloader = DataLoader(dataset, batch_size=1, drop_last=False, num_workers=8)
    from tqdm import tqdm
    pbar = tqdm(dataloader)
    for d in pbar:
        pass
    #整理clean wav文件夹
    path = './wav_1'     #原始文件路径
    path_new_pt = './wav_1/pt'   #目标文件路径
    path_new_mag = './wav_1/mag'   #目标文件路径
    list_name = os.listdir(path)
    if not os.path.exists(path_new_pt):
        os.mkdir(path_new_pt)
    if not os.path.exists(path_new_mag):
        os.mkdir(path_new_mag)
    for f in os.listdir(path):
        filename = os.path.join(path,f)
        if f.split(".")[-1] == "npy":
            if f.split(".")[-2] == "pt":
                shutil.move(filename,path_new_pt)
            else :
                shutil.move(filename,path_new_mag)
    print("done")
    #整理howling wav文件夹
    path = './howl_1'     #原始文件路径
    path_new_pt = './howl_1/pt'   #目标文件路径
    path_new_mag = './howl_1/mag'   #目标文件路径
    list_name = os.listdir(path)
    if not os.path.exists(path_new_pt):
        os.mkdir(path_new_pt)
    if not os.path.exists(path_new_mag):
        os.mkdir(path_new_mag)
    for f in os.listdir(path):
        filename = os.path.join(path,f)
        if f.split(".")[-1] == "npy":
            if f.split(".")[-2] == "pt":
                shutil.move(filename,path_new_pt)
            else :
                shutil.move(filename,path_new_mag)
    print("done")
