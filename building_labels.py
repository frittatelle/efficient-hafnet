import os

import matplotlib.pyplot as plt
import cv2
import numpy as np

if __name__ == '__main__':
    label_dir = "data/Potsdam/5_Labels_all"
    building_label_dir = "data/Potsdam/Building_Labels"
    labels_list = os.listdir(label_dir)

    # building_color = np.array([0, 0, 255])
    upper_building_color = np.array([0, 0, 255])
    lower_building_color = np.array([0, 0, 150])

    for label_image in labels_list:
        # cv2 reads image in BGR
        image = cv2.imread(os.path.join(label_dir, label_image))
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.inRange(rgb, lower_building_color, upper_building_color)
        result = cv2.bitwise_and(image, image, mask=mask)
        rgb_result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        result_filename = os.path.join(building_label_dir, label_image)
        cv2.imwrite(result_filename, mask)

    # plt.imshow(rgb_result)
    # plt.show()