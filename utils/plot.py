# from ITTI import get_saliency_map
# from MDC import get_sailency_map
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def getRandomIndex(n, x):
    index = np.random.choice(np.arange(n), size=x, replace=False)
    return index

def demo_plot(data_root : str, save_path : str, saillency_method):
    files = np.array(os.listdir(data_root))[getRandomIndex(5000, 3)]


    plt.figure(dpi=300,figsize=(24,8))
    for i, file_name in enumerate(files):
        image_test = cv2.imread(os.path.join(data_root, file_name))
        saliency_map = saillency_method.get_sailency_map(image_test)

        plt.subplot(2, 3, i+1), plt.imshow(cv2.cvtColor(image_test, cv2.COLOR_BGR2RGB))
        # plt.title('Original')
        plt.subplot(2, 3, i+4), plt.imshow(saliency_map, 'gray')
        # plt.title('Saliency Map')

    plt.tight_layout()
    plt.savefig(save_path)