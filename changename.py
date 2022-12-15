import dlib
import cv2
import os
import numpy as np
from PIL import Image

root_path = './'

orig_path = os.path.join(root_path, 'CACD2000')
out_path = os.path.join(root_path, 'CACD2000-centered')
#image = cv2.imread("18_Zoë_Kravitz_0017.jpg")
image = cv2.imread("18yrs.jpg")
cv2.imshow("im",image)
# for picture_name in os.listdir(orig_path):