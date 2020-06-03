import cv2
import glob
import tqdm
import numpy as np 
import pathlib

pathlib.Path("./data/train/Abnormal").mkdir(parents=True,exist_ok=True)
pathlib.Path("./data/train/Normal").mkdir(parents=True,exist_ok=True)
pathlib.Path("./data/valid/Abnormal").mkdir(parents=True,exist_ok=True)
pathlib.Path("./data/valid/Normal").mkdir(parents=True,exist_ok=True)

def bbox(img):
	
	rows = np.any(img, axis=1)
	cols = np.any(img, axis=0)

	row_indices = np.where(rows)[0]
	col_indices = np.where(cols)[0]

	rmin, rmax = row_indices[0], row_indices[-1]
	cmin, cmax = col_indices[0], col_indices[-1]

	return rmin, rmax, cmin, cmax
	

def crop_resize(img0, size=(450,450), pad=16):
	#crop a box around pixels large than the threshold 
	#some images contain line at the sides
	img0[img0 < 15] = 0
	ymin,ymax,xmin,xmax = bbox(img0[15:-15,15:-15] > 10)
	WIDTH = img0.shape[1]
	HEIGHT = img0.shape[0]

	#cropping may cut too much, so we need to add it back
	xmin = xmin - 10 if (xmin > 10) else 0
	ymin = ymin - 10 if (ymin > 15) else 0
	xmax = xmax + 10 if (xmax < WIDTH - 10) else WIDTH
	max = ymax + 10 if (ymax < HEIGHT - 15) else HEIGHT

	img = img0[ymin:ymax,xmin:xmax]
	lx, ly = xmax-xmin,ymax-ymin
	l = np.maximum(lx,ly) + pad

	# make sure that the aspect ratio is kept in rescaling
	img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')

	return cv2.resize(img,size)


for path_img in tqdm.tqdm(glob.glob('./MURA-v1.1/*/*/*/*/*')):
    img = cv2.imread(path_img,0)
    image = crop_resize(img)
	
    name = "_".join(path_img.split("\\")[2:])
    if "train" in path_img:
        if "positive" in path_img:
            cv2.imwrite("./data/train/Abnormal/" + name, image)
        else:
            cv2.imwrite("./data/train/Normal/" + name, image)
    elif "valid" in path_img:
        if "positive" in path_img:
            cv2.imwrite("./data/valid/Abnormal/" + name, image)
        else:
            cv2.imwrite("./data/valid/Normal/" + name, image)
