import torch
import torch.nn.functional as F
import torchvision
import cv2
import os, glob
from backbone.pplcnet import _pplcnet
from backbone.customnet import CustomNet
import torchvision.transforms as transforms

hero_names = "test_data/hero_names.txt"

file1 = open(hero_names, 'r')
Lines = file1.readlines()
map_hero = []
for i,l in enumerate(Lines):
    l = l.strip()
    map_hero.append("".join(l))

def infer(path, net, preprocess, device):
    net.eval()
    output = open("output.txt", "w")
    for file in glob.glob(path+"/*"):
        name = file.split("/")[-1]
        frame = cv2.imread(file)
        if frame is None:
            print("file is null ", file)
            return
        if frame.shape[1] > 250:
            frame = frame[:,:frame.shape[1]//4,:]
        else:
            frame = frame[:,:frame.shape[1]//2,:]

        frame = cv2.resize(frame, (224,192))

        frame = preprocess(frame.copy())
        frame = frame.unsqueeze(0)
        pred = net(frame.to(device))
        _, predicted = torch.max(pred, 1)
        output.write(name + "	" + map_hero[predicted] + "\n")
    output.close()
      

def main(*arg):
    if arg[2] == "cuda":
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if arg[1] == "small":
        backbone = _pplcnet(width_mult = 2.0, class_num=64)
        net = backbone
        state_dict_ = torch.load("snapshots/pplc_2.0_75.pt", map_location=device)
        net.load_state_dict(state_dict_['state'])
    else:
        backbone = torchvision.models.resnet34()
        net = CustomNet(backbone)
        state_dict_ = torch.load("snapshots/res34_87.pt", map_location=device)
        net.load_state_dict(state_dict_['state'])
    
    net.to(device)

    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])

    preprocess = transforms.Compose([transforms.ToTensor(), normalize])

    infer(arg[0], net, preprocess, device)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description = 'Hero recognition')
    parser.add_argument('--path', default="test_data/test_images", help='Path to image folder')
    parser.add_argument('--model', default="big", help='big or small model')
    parser.add_argument('--device', default="cuda", help='cuda or cpu')

    args = parser.parse_args()
    print(map_hero)

    main(args.path, args.model, args.device)