from datetime import datetime

import cv2


def save_jpgs(image, name_prefix) -> str:
    now = datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M%S")
    ori_jpg_file_name = name_prefix + date_time + '.jpg'
    cv2.imwrite(ori_jpg_file_name, image)
    # undis_jpg_file_name = "undis" + date_time + '.jpg'
    # cv2.imwrite(undis_jpg_file_name, dst2)
    return ori_jpg_file_name