import cv2
import numpy as np



def red_hsv(bgr_img: np.ndarray):
    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    # H -> 0->10
    # S -> 175-255
    # V -> 20-255
    lower_hsv_1 = np.array([0, 100, 20])
    higher_hsv_1 = np.array([10, 255, 255])

    # H -> 170 - 180
    # S -> 175 - 255
    # V -> 20 - 255
    lower_hsv_2 = np.array([170, 100, 20])
    higher_hsv_2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv_img, lower_hsv_1, higher_hsv_1)
    mask2 = cv2.inRange(hsv_img, lower_hsv_2, higher_hsv_2)
    return mask1 + mask2


def green_hsv(bgr_img: np.ndarray):
    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    # H -> 40 - 70
    # S -> 150 -255
    # V -> 20 - 255
    lower_hsv = np.array([40, 100, 20])
    higher_hsv = np.array([110, 255, 255])
    mask = cv2.inRange(hsv_img, lower_hsv, higher_hsv)
    return mask


def blue_hsv(bgr_img: np.ndarray):
    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

    # H -> 40 - 70
    # S -> 150 -255
    # V -> 20 - 255
    lower_hsv = np.array([100, 150, 0])
    higher_hsv = np.array([140, 255, 255])
    mask = cv2.inRange(hsv_img, lower_hsv, higher_hsv)
    return mask

def black_hsv(bgr_img: np.ndarray):
    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    # H -> 0 - 180
    # S -> 0 - 255
    # V -> 0 - 46
    lower_hsv = np.array([0, 0, 0])
    higher_hsv = np.array([180, 255, 46])
    mask = cv2.inRange(hsv_img, lower_hsv, higher_hsv)
    return mask


def get_green_red_image(ori_image: np.ndarray):
    red_hsv_mask = red_hsv(ori_image)
    green_hsv_mask = green_hsv(ori_image)

    red_green_mask = red_hsv_mask + green_hsv_mask

    bgr_r_g_image = cv2.bitwise_and(ori_image, ori_image, mask=red_green_mask)
    return bgr_r_g_image


def get_red_image(ori_image: np.ndarray):
    red_hsv_mask = red_hsv(ori_image)
    bgr_red_image = cv2.bitwise_and(ori_image, ori_image, mask=red_hsv_mask)
    return bgr_red_image


def get_blue_image(ori_image: np.ndarray):
    blue_hsv_mask = blue_hsv(ori_image)
    bgr_blue_image = cv2.bitwise_and(ori_image, ori_image, mask=blue_hsv_mask)
    return bgr_blue_image

def get_black_image(ori_image: np.ndarray):
    black_hsv_mask = black_hsv(ori_image)
    bgr_black_image = cv2.bitwise_and(ori_image, ori_image, mask=black_hsv_mask)
    return bgr_black_image

