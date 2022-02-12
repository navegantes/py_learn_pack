import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox

im = cv2.imread(r'img/traffic2.jpg')

conf = 0.5
nms_thresh = 0.6
bbox, label, confidence = cv.detect_common_objects(im,
                                                   confidence=conf,
                                                   nms_thresh=nms_thresh,
                                                   enable_gpu=True,
                                                   model='yolov4-tiny') # model='yolov3')
output_image = draw_bbox(im, bbox, label, confidence)

# Listing/Counting the number of objects detected
obj_list = pd.value_counts(np.array(label)) # return Pandas.Series
print("\nObjects list")
print(obj_list)
print("\nConfidence")
print(confidence)
num_objects = np.array(obj_list.to_list()).sum()
print(f"\nNumber of objects: {num_objects}")

# # Uncomment to show using Opencv
# scale = .5 # percent of original size
# rz_im = cv2.resize(output_image,
#                    None,
#                    fx=scale,
#                    fy=scale,
#                    interpolation = cv2.INTER_CUBIC)
# # infos
# print(output_image.shape[0], output_image.shape[1])
# print(rz_im.shape[0], rz_im.shape[1])

# cv2.imshow('Image', rz_im)
# print("\nPress any key to close.")
# cv2.waitKey(0)
# cv2.destroyAllWindows() 

#========================

# Uncomment to show using Matplotlib
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.title(f"Number of objects: {num_objects}")
plt.tight_layout()
plt.show()
