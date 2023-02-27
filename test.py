from itertools import count
from turtle import pos
import torch
import torch.nn.functional as F
import torchvision
import cv2
import os
from PIL import Image
from backbone.pplcnet import _pplcnet
from backbone.customnet import CustomNet
import torchvision.transforms as transforms


hero_names = "test_data/hero_names.txt"

file1 = open(hero_names, 'r')
Lines = file1.readlines()
map_hero = []
for i,l in enumerate(Lines):
    l = l.strip().lower().split("_")
    map_hero.append("".join(l))
print(map_hero)


# backbone = _pplcnet(width_mult = 2.0, class_num=64)
# net = backbone


backbone = torchvision.models.resnet34(pretrained = False)
net = CustomNet(backbone)

state_dict_ = torch.load("snapshots/res34_87.pt", map_location=torch.device('cuda'))
net.load_state_dict(state_dict_['state'])
net.to(torch.device('cuda'))

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])

preprocess = transforms.Compose([transforms.ToTensor(), normalize])


def test(test_file, net, preprocess, device):
    net.eval()
    file = open(test_file, 'r')
    Lines = file.readlines()
    count = 0
    for line in Lines:
        line = line.split("	")
        path, gt = line[0], line[1]
        frame = cv2.imread("test_data/test_images/" + path)
        if frame.shape[1] > 250:
            frame = frame[:,:frame.shape[1]//4,:]
        else:
            frame = frame[:,:frame.shape[1]//2,:]

        frame_origin = cv2.resize(frame, (224,192))

        frame = preprocess(frame_origin.copy())
        frame = frame.unsqueeze(0)
        pred = net(frame.to(device))
        _, predicted = torch.max(pred, 1)
        gt = gt.strip().lower().split("_")
        if map_hero[predicted] == "".join(gt):
            count +=1
        # else:
        #     cv2.imwrite("debug/"+ map_hero[predicted] + "_"+path, frame_origin)
    print(count, "/", len(Lines))
test("test_data/test.txt", net, preprocess, torch.device('cuda'))


