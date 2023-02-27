import numpy as np
import glob, cv2, os, sys
from torchvision import transforms
sys.path.append('./func_transforms')
from func_transforms.transforms import *

DIR = "hero"
hero_names = "test_data/hero_names.txt"

file1 = open(hero_names, 'r')
Lines = file1.readlines()
map_hero = {}
for i,l in enumerate(Lines):
    l = l.strip().lower().split("_")
    map_hero["".join(l)] = i
print(map_hero)


my_transforms = transforms.Compose([
    SequenceRandomTransform(),
])

for file in glob.glob(DIR + "/*"):
    name = file.split("/")[-1].split("_")[0].lower()
    
    img_origin = cv2.imread(file)
    img_origin = cv2.resize(img_origin,(96,96))
    

    out_imgs = []
    out_name = []

    for i in range(500):
       
        img = my_transforms(img_origin.copy())
        # cv2.imwrite("a.png", img)
        # exit()
        out_imgs.append(img)
        out_name.append([map_hero[name]])
  
    np.savez("./data/"+name+".npz",image=np.array(out_imgs), name=np.array(out_name))