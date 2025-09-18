import cv2
import numpy as np

from numpy.typing import NDArray

KERNEL_SIZE = 3
CENTER = int(KERNEL_SIZE // 2)

def create_original_filter() -> NDArray[np.float32]:
    kernel = np.zeros((KERNEL_SIZE, KERNEL_SIZE), dtype=np.float32)
    kernel[CENTER][CENTER] = 1

    return kernel

def create_blur_filter() -> NDArray[np.float32]:
    constant = float(1/9)
    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), dtype=np.float32)
    return constant * kernel

def create_gausian_blur_filter() -> NDArray[np.float32]:
    constant = float(1/16)
    kernel = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]], dtype=np.float32)
    
    return constant * kernel

def create_sharpen_filter() -> NDArray[np.float32]:
    kernel = np.array([[0 , -1,  0],
                       [-1,  5, -1],
                       [0 , -1,  0]], dtype=np.float32)

    return kernel

def create_sobel_x_filter() -> NDArray[np.float32]:
    kernel = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=np.float32)

    return kernel

def create_sobel_y_filter() -> NDArray[np.float32]:
    kernel = create_sobel_x_filter()
    return kernel.T
    
def create_edge_detection_filter() -> NDArray[np.float32]:
    kernel = np.full((KERNEL_SIZE, KERNEL_SIZE), -1, dtype=np.float32)
    kernel[CENTER][CENTER] = 8
    return kernel

def create_emboss_filter() -> NDArray[np.float32]:
    kernel = np.array([[-2, -1, 0],
                       [-1,  1, 1],
                       [ 0,  1, 2]], dtype=np.float32)
    return kernel

class Filters:
    # TODO: Image kernels
    Kernels = {
        "original"       : create_original_filter(),
        "blur"           : create_blur_filter(),
        "gaussian blur"  : create_gausian_blur_filter(),
        "sharpen"        : create_sharpen_filter(),
        "sobel (x)"      : create_sobel_x_filter(),
        "sobel (y)"      : create_sobel_y_filter(),
        "edge detection" : create_edge_detection_filter(),
        "emboss"         : create_emboss_filter(),
    }

    def __init__(self, kernels=Kernels):
        self.kernels = kernels
        # TODO: Implement internal variables
        kernel_keys = list(kernels.keys())
        self.cur_kernel = kernel_keys[0]

    def apply_filter(self, frame, filter_name) -> np.array:
        # TODO: Apply the selected filter kernel to the frame
        self.cur_kernel = filter_name
        return cv2.filter2D(frame, -1, self.kernels[filter_name])

    def get_current_filter_name(self) -> str:
        # TODO: Return currently set kernels's name
        return self.cur_kernel

    def switch_next_filter(self):
        # TODO: Update currently selected kernel to the next
        filter_list = list(self.kernels.keys())
        next_filter_idx = int(filter_list.index(self.cur_kernel) + 1)
        if (next_filter_idx >= len(filter_list)):
            next_filter_idx = 0

        # set new filter
        self.cur_kernel = filter_list[next_filter_idx]
        return self.cur_kernel

    def switch_previous_filter(self):
        # TODO: Update currently selected kernel to the previous
        filter_list = list(self.kernels.keys())
        prev_filter_idx = int(filter_list.index(self.cur_kernel) - 1)
        if (prev_filter_idx < 0):
            prev_filter_idx = len(filter_list) - 1

        # set new filter
        self.cur_kernel = filter_list[prev_filter_idx]
        return self.cur_kernel