import numpy as np
import cv2
from glob import glob


arg_num = 15
vid_list = glob('../../../dataset/fashion_video/train/*.mp4')

img_list = vid_list[arg_num].split('/')[-1].split('.')[0].split('\\')[1]
print(img_list)

img_list = glob('../../../dataset/fashion_video/train_orig_png/'+img_list+'/*.png')

step = 1
i1_ = 150
i2_ = i1_ + step
i3_ = i2_ + step

img_1 = cv2.imread(img_list[i1_])
img_2 = cv2.imread(img_list[i2_])
img_3 = cv2.imread(img_list[i3_])

cv2.imwrite('1.png', img_1)
cv2.imwrite('2.png', img_2)
cv2.imwrite('3.png', img_3)
print(img_1.shape, img_2.shape)
prev_gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
flow1 = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, pyr_scale = 0.5, levels = 5, winsize = 11, iterations = 5, poly_n = 5, poly_sigma = 1.1, flags = 0)

mag, ang = cv2.cartToPolar(flow1[...,0], flow1[...,1])
hsv = np.zeros_like(img_1)
hsv[...,1] = 255
hsv[...,0] = ang*180/np.pi/2
hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
cv2.imwrite('optical_flow_1_2.png',bgr)

next_gray = cv2.cvtColor(img_3, cv2.COLOR_BGR2GRAY)
flow2 = cv2.calcOpticalFlowFarneback(gray, next_gray, None, pyr_scale = 0.5, levels = 5, winsize = 11, iterations = 5, poly_n = 5, poly_sigma = 1.1, flags = 0)
print("flow2.shape", flow2.shape)
mag, ang = cv2.cartToPolar(flow2[...,0], flow2[...,1])
hsv = np.zeros_like(img_1)
hsv[...,1] = 255
hsv[...,0] = ang*180/np.pi/2
hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
cv2.imwrite('optical_flow_2_3.png',bgr)

flow1_3 = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, pyr_scale = 0.5, levels = 5, winsize = 11, iterations = 5, poly_n = 5, poly_sigma = 1.1, flags = 0)
print("flow1_3.shape", flow1_3.shape)
mag, ang = cv2.cartToPolar(flow1_3[...,0], flow1_3[...,1])
hsv = np.zeros_like(img_1)
hsv[...,1] = 255
hsv[...,0] = ang*180/np.pi/2
hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
cv2.imwrite('optical_flow_1_3.png',bgr)

flow3 = flow1 + flow2
print("flow3.shape", flow3.shape)
mag, ang = cv2.cartToPolar(flow3[...,0], flow3[...,1])
hsv = np.zeros_like(img_1)
hsv[...,1] = 255
hsv[...,0] = ang*180/np.pi/2
hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
cv2.imwrite('optical_flow_1_3_hat.png',bgr)


h = flow1.shape[0]
w = flow1.shape[1]

flow1[:,:,0] += np.arange(w)
flow1[:,:,1] += np.arange(h)[:,np.newaxis]
new_frame_1 = cv2.remap(img_1, flow1, None, cv2.INTER_LINEAR)
cv2.imwrite('new_1_2.png',new_frame_1)

flow2[:,:,0] += np.arange(w)
flow2[:,:,1] += np.arange(h)[:,np.newaxis]
new_frame_2 = cv2.remap(img_2, flow2, None, cv2.INTER_LINEAR)
cv2.imwrite('new_2_3.png',new_frame_2)

flow1_3[:,:,0] += np.arange(w)
flow1_3[:,:,1] += np.arange(h)[:,np.newaxis]
new_frame_2 = cv2.remap(img_1, flow1_3, None, cv2.INTER_LINEAR)
cv2.imwrite('new_1_3.png',new_frame_2)

# flow3= flow1 + flow2
flow3[:,:,0] += np.arange(w)
flow3[:,:,1] += np.arange(h)[:,np.newaxis]
new_frame_2_hat = cv2.remap(img_1, flow3, None, cv2.INTER_LINEAR)
cv2.imwrite('new_1_3_hat.png',new_frame_2_hat)