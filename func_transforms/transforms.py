
import torch
import cv2, random
import numpy as np
from zoom_transform import _apply_random_zoom

class Normalize(object):
    """Applies following normalization: out =  (img-mean)/std ."""
    def __init__(self, mean, std):

        self.mean = mean
        self.std = std

    def __call__(self, img):

        img = (img-self.mean)/self.std
        return img

class RandomCrop(object):
    """Select random crop portion from input image."""
    def __init__(self):
        pass

    def __call__(self, img):
        h = img.shape[0]
        w = img.shape[1]
        dn = np.random.randint(h/4,size=1)[0]+1

        dx = np.random.randint(dn,size=1)[0]
        dy = np.random.randint(dn,size=1)[0]
        
        out = img[0+dy:h-(dn-dy),0+dx:w-(dn-dx),:]

        out = cv2.resize(out, (h,w), interpolation=cv2.INTER_CUBIC)

        return out

class RandomCropBlack(object):
    """
    Select random crop portion from input image.
    Paste crop region on a black image having same shape as input image.
    """
    def __init__(self):
        pass

    def __call__(self, img):
        h = img.shape[0]
        w = img.shape[1]
        dn = np.random.randint(h/4,size=1)[0]+1

        dx = np.random.randint(dn,size=1)[0]
        dy = np.random.randint(dn,size=1)[0]
        
        

        dx_shift = np.random.randint(dn,size=1)[0]
        dy_shift = np.random.randint(dn,size=1)[0]
        out = np.zeros_like(img)
        out[0+dy_shift:h-(dn-dy_shift),0+dx_shift:w-(dn-dx_shift),:] = img[0+dy:h-(dn-dy),0+dx:w-(dn-dx),:]

        return out

class RandomOccludedBlack(object):
    def __init__(self):
        pass

    def __call__(self, img):
        h = img.shape[0]
        w = img.shape[1]

        dx = np.random.randint(w,size=1)[0]
        dy = np.random.randint(h,size=1)[0]
        
        

        dx_shift = np.random.randint(w/2,size=1)[0]
        dy_shift = np.random.randint(h/2,size=1)[0]
        
        cv2.rectangle(img, (dx, dy), (dx+dx_shift, dy+dy_shift), (0,0,0), -1)

        return img

class RandomOccludedWhite(object):
    def __init__(self):
        pass

    def __call__(self, img):
        h = img.shape[0]
        w = img.shape[1]

        dx = np.random.randint(w,size=1)[0]
        dy = np.random.randint(h,size=1)[0]
        
        

        dx_shift = np.random.randint(w/2,size=1)[0]
        dy_shift = np.random.randint(h/2,size=1)[0]
        
        cv2.rectangle(img, (dx, dy), (dx+dx_shift, dy+dy_shift), (255,255,255), -1)

        return img

class RandomCropWhite(object):
    """
    Select random crop portion from input image.
    Paste crop region on a white image having same shape as input image.
    """
    def __init__(self):
        pass

    def __call__(self, img):
        h = img.shape[0]
        w = img.shape[1]
        dn = np.random.randint(h/4,size=1)[0]+1

        dx = np.random.randint(dn,size=1)[0]
        dy = np.random.randint(dn,size=1)[0]

        

        dx_shift = np.random.randint(dn,size=1)[0]
        dy_shift = np.random.randint(dn,size=1)[0]
        out = np.ones_like(img)*255
        out[0+dy_shift:h-(dn-dy_shift),0+dx_shift:w-(dn-dx_shift),:] = img[0+dy:h-(dn-dy),0+dx:w-(dn-dx),:]

        return out

class RandomZoom(object):
    """Apply RandomZoom transformation."""
    def __init__(self,zoom_range=[0.8,1.2]):
        self.zoom_range = zoom_range

    def __call__(self,img):
        out = _apply_random_zoom(img,self.zoom_range)

        return out

class SequenceRandomTransform(object):
    """
    Apply Transformation in a sequenced random order
    similar to original author's implementation
    """
    def __init__(self,zoom_range=[0.5,1.2]):
        self.rc = RandomCrop()
        self.rcb = RandomCropBlack()
        self.rcw = RandomCropWhite()
        self.rz = RandomZoom(zoom_range=zoom_range)
        self.rob = RandomOccludedBlack()
        self.row = RandomOccludedWhite()
        file = open("test_data/test.txt", 'r')
        self.Lines = file.readlines()

    

    def __call__(self,img):
        # rand_r = np.random.random()
        # if  rand_r < 0.33:
        #     img = self.rc(img)

        # elif rand_r >= 0.33 and rand_r < 0.66:
        #     img = self.rcb(img)

        # elif rand_r >= 0.66:
        #     img = self.rcw(img)

        rand_r = np.random.random()
        if  rand_r < 0.3:
            img = self.rob(img)

        elif rand_r <0.6:
            img = self.row(img)
            

        if np.random.random() > 0.3:
            img = self.rz(img)
        

        # overlay = np.zeros_like(img)
        # overlay[:] = (random.randint(125,255), random.randint(125,255), 0) 
        # val = random.uniform(0.0, 0.5)
        # img = cv2.addWeighted(overlay, val, img, 1 - val, 0)

        rand_color = random.uniform(0.3, 0.8)
        img = img*rand_color

        rand_resize_w = np.random.randint(10,30)
        rand_resize_h = np.random.randint(10,30)
        img = cv2.resize(img, (rand_resize_w,rand_resize_h))

        # rand_blur = random.randrange(3,13,2)
        # img = cv2.GaussianBlur(img, (rand_blur,rand_blur),cv2.BORDER_DEFAULT)
        
        rand_resize_w = np.random.randint(100,150)
        rand_resize_h = np.random.randint(100,150)

        img = cv2.resize(img, (rand_resize_w,rand_resize_h))
        rand_shift_w = np.random.randint(20,170-rand_resize_w)
        rand_shift_h = np.random.randint(20,170-rand_resize_h)

        rand_r = np.random.randint(len(self.Lines))
        line = self.Lines[rand_r]
        line = line.split("	")
        path, gt = line[0], line[1]
        frame = cv2.imread("test_data/test_images/" + path)
        if frame.shape[1] > 250:
            frame = frame[:,:frame.shape[1]//4,:]
        else:
            frame = frame[:,:frame.shape[1]//2,:]
        background = cv2.resize(frame, (224,192))
        background = cv2.blur(background, (5,5))

        background[rand_shift_h:rand_shift_h+rand_resize_h, rand_shift_w:rand_shift_w+rand_resize_w,:] = img
        return background

class ToTensor(object):
    """Convert ndarrays to Tensors."""
    def __call__(self, img):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = img.transpose((2, 0, 1))

        return torch.from_numpy(img).float()
