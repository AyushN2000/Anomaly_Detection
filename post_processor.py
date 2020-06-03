import cv2
import numpy as np
import scipy  as sp
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

def bbox(img):
	rows = np.any(img, axis=1)
	cols = np.any(img, axis=0)

	row_indices = np.where(rows)[0]
	col_indices = np.where(cols)[0]

	rmin, rmax = row_indices[0], row_indices[-1]
	cmin, cmax = col_indices[0], col_indices[-1]

	return rmin, rmax, cmin, cmax
	
def crop_resize(img0, size=(448,448), pad=16):
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

def plot_activation(img):
	img = img - 50.1428
	img = img/ 0.1812

	pred = model.predict(img[np.newaxis,:,:,np.newaxis])
	pred_class = np.argmax(pred)

	weights = model.layers[-1].get_weights()[0]
	class_weights = weights[:, pred_class]
	intermediate = tf.keras.models.Model(model.input,
                         model.get_layer("conv5_block32_concat").output)
	conv_output = intermediate.predict(img[np.newaxis,:,:,np.newaxis])
	conv_output = np.squeeze(conv_output)

	h = int(img.shape[0]/conv_output.shape[0])
	w = int(img.shape[1]/conv_output.shape[1])

	act_maps = sp.ndimage.zoom(conv_output, (h, w, 1), order=1)
	out = np.dot(act_maps.reshape((img.shape[0]*img.shape[1],1664)), 
                 class_weights).reshape(img.shape[0],img.shape[1])

	plt.imshow(img.astype('float32').reshape(img.shape[0],
               img.shape[1]),cmap='bone')
	plt.imshow(out, cmap='jet', alpha=0.35)

	plt.title('No Abnormality Detected' if pred_class == 1 else 'Abnormality Detected')
	plt.show()

path_image = 'Paste the path to test image'
img = cv2.imread(path_image,0)
img = crop_resize(img)

plt.imshow(img,cmap='bone')
plt.show()

model = tf.keras.models.load_model('model_trained.h5')
model.summary()

plot_activation(img)