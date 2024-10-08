import cv2
import os

def video2frame(video_path):
    vc = cv2.VideoCapture(video_path)
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False
    c = 1
    while rval:
        cv2.imwrite(os.path.join(out_path, f'{os.path.splitext(os.path.basename(video))[0]}_{c}.jpg'), frame)
        c = c + 1
        rval, frame = vc.read()
    vc.release()