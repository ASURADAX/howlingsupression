import os
import glob
import shutil

path = './wav'     #原始文件路径
path_new_pt = './wav/pt'   #目标文件路径
path_new_mag = './wav/mag'   #目标文件路径
list_name = os.listdir(path)

for f in os.listdir(path):
    filename = os.path.join(path,f)
    if f.split(".")[-1] == "npy":
        if f.split(".")[-2] == "pt":
            shutil.move(filename,path_new_pt)
        else :
            shutil.move(filename,path_new_mag)
print("done")
