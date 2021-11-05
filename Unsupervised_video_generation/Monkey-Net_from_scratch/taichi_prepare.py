import os
import glob
import cv2

################################################################
# From Taichi videoset, extracts frames and delete video       #
# dataset/taichi/train/_64NZbqcISg#000000#000199/0.png         #
#                                                1.png         #
# For test:                                                    #
#   ../../../../../../../dataset/taichi/                       #
################################################################

train_video_list = glob.glob('../../../../../../../dataset/taichi/taichi/train/*.mp4')
test_video_list = glob.glob('../../../../../../../dataset/taichi/taichi/test/*.mp4')

# print(train_video_list[:3])

# train logic
for video in train_video_list:
    cap = cv2.VideoCapture(video)
    i=0
    video_name = video.split('\\')[-1].split('.')[0]
    directory = '../../../../../../../dataset/taichi/train_png/'+video_name

    if not os.path.exists(directory):
        os.makedirs(directory)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        cv2.imwrite('../../../../../../../dataset/taichi/train_png/' + video_name + '/' + str(i) + '.jpg',frame)
        i+=1
    
    cap.release()
    cv2.destroyAllWindows()

# test logic
for video in test_video_list:
    cap = cv2.VideoCapture(video)
    i=0
    video_name = video.split('\\')[-1].split('.')[0]
    directory = '../../../../../../../dataset/taichi/test_png/'+video_name

    if not os.path.exists(directory):
        os.makedirs(directory)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        cv2.imwrite('../../../../../../../dataset/taichi/test_png/' + video_name + '/' + str(i) + '.jpg',frame)
        i+=1
    
    cap.release()
    cv2.destroyAllWindows()
