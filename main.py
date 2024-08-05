import cv2
import numpy as np
import platform

from color_method import get_green_red_image, get_black_image, red_hsv, green_hsv, get_blue_image, get_red_image
from cv2_key_event import cv2_key_event
from save_pics import save_jpgs
from c_contour import VArea
from v_math import get_angle_with_three_points, dart_points, get_angle_with_two_lines
from global_def import log
TEST_WITH_JPG = True


def get_white_area_list(src_img):
    area_list = []
    white_src_image = src_img.copy()
    ''' 檢測白色區塊 '''
    ''' 圖像直接二值化 '''
    gray_white_image = cv2.cvtColor(white_src_image, cv2.COLOR_BGR2GRAY)
    zret, white_image_binary = cv2.threshold(gray_white_image, 150, 255, cv2.THRESH_BINARY)
    # cv2.imshow("white_image_binary", white_image_binary)

    # erode & dilage, performance is better
    kernel_temp = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    white_image_binary = cv2.erode(white_image_binary, kernel_temp)
    white_image_binary = cv2.dilate(white_image_binary, kernel_temp)
    # cv2.imshow("white_image_binary erode", white_image_binary)

    ''' 找尋輪廓 '''
    white_cnts, hierarchy = cv2.findContours(white_image_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    new_contours = []
    for cnt in white_cnts:
        area = cv2.contourArea(cnt)
        if area > 1000:
            new_contours.append(cnt)

    contours = new_contours
    # print("len(contours): ", len(contours))
    # red_green_contour_image = src_img.copy()

    ''' contour 建立類別 '''
    v_id = 0
    for c in contours:
        M = cv2.moments(c)
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
        varea = VArea(c, v_id, center_x, center_y)
        v_id += 1
        area_list.append(varea)
    # print("len(contours): ", len(contours))
    # print("v_id: ", v_id)
    return area_list


def get_black_area_list(src_img):
    area_list = []
    ''' 檢測黑色區塊 '''
    ''' 先將紅色綠色區塊變成白色 '''
    black_src_image = src_img.copy()
    red_hsv_mask = red_hsv(black_src_image)
    green_hsv_mask = green_hsv(black_src_image)
    red_green_mask = red_hsv_mask + green_hsv_mask
    black_stage1_image = cv2.bitwise_not(black_src_image, black_src_image, mask=red_green_mask)
    # cv2.imshow("black stage1 Frame", black_stage1_image)

    black_image = get_black_image(black_stage1_image)

    # cv2.imshow("black Frame", black_image)

    ''' 灰階處理二值化 '''
    gray_black_image = cv2.cvtColor(black_stage1_image, cv2.COLOR_BGR2GRAY)
    '''cv2.imshow("gray black Frame 0", gray_black_image)
    zret, gray_black_image = cv2.threshold(gray_black_image, 170, 0, cv2.THRESH_TOZERO_INV)
    cv2.imshow("gray black Frame 1", gray_black_image)'''
    zret, black_image_gray = cv2.threshold(gray_black_image, 50, 255, cv2.THRESH_BINARY)
    # cv2.imshow("gray black Frame 2", gray_black_image)
    # erode & dilage, performance is better
    kernel_temp = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    black_image_gray = cv2.erode(black_image_gray, kernel_temp)
    black_image_gray = cv2.dilate(black_image_gray, kernel_temp)

    # cv2.imshow("black image output", black_image_gray)

    ''' 找尋輪廓 '''
    black_cnts, hierarchy = cv2.findContours(black_image_gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # black_cnts, hierarchy = cv2.findContours(black_image_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    new_contours = []
    for cnt in black_cnts:
        area = cv2.contourArea(cnt)
        if 10000 > area > 2000:
            new_contours.append(cnt)

    contours = new_contours
    # print("len(contours): ", len(contours))
    # red_green_contour_image = src_img.copy()

    ''' contour 建立類別 '''
    v_id = 0
    for c in contours:
        M = cv2.moments(c)
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
        varea = VArea(c, v_id, center_x, center_y)
        v_id += 1
        area_list.append(varea)
    # print("len(contours): ", len(contours))
    # print("v_id: ", v_id)
    return area_list


def get_red_green_area(src_img):
    area_list = []
    ''' 檢測紅色綠色區塊 '''
    red_green_image = get_green_red_image(src_img)
    # cv2.imshow("Red Green Frame", red_green_image)
    ''' 找出紅色綠色contour'''
    frame_sample = red_green_image.copy()
    ''' 先將圖片轉灰階 '''
    gray_red_green_image = cv2.cvtColor(frame_sample, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray_red_green_image", gray_red_green_image)
    ''' 灰階處理二值化 '''
    ret, gray_red_green_image = cv2.threshold(gray_red_green_image, 40, 255, cv2.THRESH_BINARY)

    # blur & canny
    # gray_red_green_image = cv2.GaussianBlur(gray_red_green_image, (7, 7), 0)
    # gray_red_green_image = cv2.Canny(gray_red_green_image, 20, 160)

    # erode & dilage, performance is better
    kernel_temp = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gray_red_green_image = cv2.erode(gray_red_green_image, kernel_temp)
    gray_red_green_image = cv2.dilate(gray_red_green_image, kernel_temp)

    # CLOSE & OPEN
    '''gray_red_green_image = cv2.morphologyEx(gray_red_green_image, cv2.MORPH_OPEN,
                                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1)))
    gray_red_green_image = cv2.morphologyEx(gray_red_green_image, cv2.MORPH_CLOSE,
                                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1)))'''

    # cv2.imshow("output", gray_red_green_image)

    ''' 找尋輪廓 '''
    red_green_cnts, hierarchy = cv2.findContours(gray_red_green_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # red_green_cnts, hierarchy = cv2.findContours(gray_red_green_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    new_contours = []
    for cnt in red_green_cnts:
        area = cv2.contourArea(cnt)
        if area > 5:
            new_contours.append(cnt)

    contours = new_contours
    # print("len(contours): ", len(contours))
    red_green_contour_image = src_img.copy()

    ''' contour 建立類別 '''
    v_id = 0
    for c in contours:
        M = cv2.moments(c)
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
        varea = VArea(c, v_id, center_x, center_y)
        v_id += 1
        area_list.append(varea)
    # print("len(contours): ", len(contours))
    # print("v_id: ", v_id)
    return area_list


if __name__ == '__main__':
    log.debug("Start ")
    double_triple_area_list = []
    if TEST_WITH_JPG is True:
        pass
    else:
        if 'aarch64' in platform.machine(): # RPI
            cap = cv2.VideoCapture(0)
        else:   # x86
            cap = cv2.VideoCapture(2)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        if TEST_WITH_JPG is True:
            image = cv2.imread("ori_frame_20240731_232718_b.jpg")
        else:
            ret, frame = cap.read()
            if ret is not True:
                print("no frame")
                break

        frame = image[110:710 , 275:960]

        cv2.imshow("Ori Frame", frame)
        double_triple_image = frame.copy()
        double_triple_area_list = get_red_green_area(double_triple_image)
        for area in double_triple_area_list:
            cv2.drawContours(double_triple_image, area.cnt, -1, (0, 255, 0), 2)
            cv2.putText(double_triple_image, str(area.id) + "/" + str(area.area_size), (area.center_x, area.center_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(double_triple_image, str(area.area_size), (area.center_x, area.center_y + 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("red_green_contour_image", double_triple_image)

        black_area_image = frame.copy()
        black_area_list = get_black_area_list(black_area_image)
        for area in black_area_list:
            cv2.drawContours(black_area_image, area.cnt, -1, (0, 255, 0), 2)
            cv2.putText(black_area_image, str(area.id) + "/" + str(area.area_size), (area.center_x, area.center_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 255), 1, cv2.LINE_AA)
            # cv2.putText(black_area_image, str(area.area_size), (area.center_x, area.center_y + 20), cv2.FONT_HERSHEY_SIMPLEX,
            #            0.5, (0, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("black_contour_image", black_area_image)

        white_area_image = frame.copy()
        white_area_list = get_white_area_list(white_area_image)
        for area in white_area_list:
            cv2.drawContours(white_area_image, area.cnt, -1, (0, 255, 0), 2)
            cv2.putText(white_area_image, str(area.id) + "/" + str(area.area_size), (area.center_x, area.center_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 255), 1, cv2.LINE_AA)
            # cv2.putText(white_area_image, str(area.area_size), (area.center_x, area.center_y + 20), cv2.FONT_HERSHEY_SIMPLEX,
            #             0.5, (0, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("white_contour_image", white_area_image)


        ''' get blue circle point for calculate angle '''
        blue_src_img = frame.copy()
        blue_image = get_blue_image(blue_src_img)
        cv2.imshow("Blue Frame", blue_image)

        gray_blue_image = cv2.cvtColor(blue_image, cv2.COLOR_BGR2GRAY)
        ret, gray_blue_image = cv2.threshold(gray_blue_image, 20, 255, cv2.THRESH_BINARY)

        # erode & dilage, performance is better
        kernel_temp = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gray_blue_image = cv2.erode(gray_blue_image, kernel_temp)
        gray_blue_image = cv2.dilate(gray_blue_image, kernel_temp)
        cv2.imshow("Blue Gray Frame", gray_blue_image)
        gray_blue_contour, hierarchy = cv2.findContours(gray_blue_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        print("len(contours): ", len(gray_blue_contour))
        if len(gray_blue_contour) == 1:
            blue_circle_contour = gray_blue_contour
            for c in blue_circle_contour:
                M = cv2.moments(c)
                blue_circle_center_x = int(M["m10"] / M["m00"])
                blue_circle_center_y = int(M["m01"] / M["m00"])
                print("blue_circle_center_x : ", blue_circle_center_x)
                print("blue_circle_center_y : ", blue_circle_center_y)

        ''' get red point circle '''
        red_src_img = frame.copy()
        red_image = get_red_image(red_src_img)
        cv2.imshow("Red Frame", red_image)
        gray_red_image = cv2.cvtColor(red_image, cv2.COLOR_BGR2GRAY)
        ret, gray_red_image = cv2.threshold(gray_red_image, 20, 255, cv2.THRESH_BINARY)
        cv2.imshow("Red Gray  Frame", gray_red_image)

        kernel_temp = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gray_red_image = cv2.erode(gray_red_image, kernel_temp)
        gray_red_image = cv2.dilate(gray_red_image, kernel_temp)
        gray_red_contour, hierarchy = cv2.findContours(gray_red_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        red_area_size_min = 0xffffff
        print("len(gray_red_contour):", len(gray_red_contour))
        for c in gray_red_contour:
            size_tmp = cv2.contourArea(c)
            if size_tmp < red_area_size_min:
                red_area_size_min = size_tmp
                M = cv2.moments(c)
                red_center_x = int(M["m10"] / int(M["m00"]))
                red_center_y = int(M["m01"] / int(M["m00"]))
        print("red_center_x :", red_center_x)
        print("red_center_y :", red_center_y)

        ''' 試算角度, 填入各area'''
        ''' 黑色區塊填入計分值 '''
        line1 = [(red_center_x, red_center_y), (blue_circle_center_x, blue_circle_center_y)]
        ''' 黑色區塊填入計分值 '''
        for black_area in black_area_list:
            line2 = []
            line2.append((red_center_x, red_center_y))
            line2.append((black_area.center_x, black_area.center_y))
            '''angle = get_angle((blue_circle_center_x, blue_circle_center_y),
                              (red_center_x, red_center_y),
                              (black_area.center_x, black_area.center_y))'''
            angle = get_angle_with_two_lines(line1, line2)
            '''if angle < 340:
                black_area.set_point(int(dart_points[int(angle/18) + 1]))
            else:
                black_area.set_point(int(dart_points[int(angle / 18)]))'''
            black_area.set_point(int(dart_points[int(angle / 18)]))
            log.debug("black_area id: %d", black_area.id)
            log.debug("angle : %f", angle)
            '''log.debug("int angle/20 : %d", int(angle/20))
            log.debug("black_area point: %d", black_area.point)'''
            cv2.putText(black_area_image, str(black_area.point), (black_area.center_x, black_area.center_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 1, cv2.LINE_AA)
        ''' 白色區塊填入計分值 '''
        for white_area in white_area_list:
            line2 = []
            line2.append((red_center_x, red_center_y))
            line2.append((white_area.center_x, white_area.center_y))

            angle = get_angle_with_two_lines(line1, line2)

            white_area.set_point(int(dart_points[int(angle / 18)]))
            log.debug("black_area id: %d", white_area.id)
            log.debug("angle : %f", angle)

            cv2.putText(white_area_image, str(white_area.point), (white_area.center_x, white_area.center_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 1, cv2.LINE_AA)

        cv2.imshow("white_contour_image_point", white_area_image)

        ret = cv2_key_event()
        if ret == -1:
            print("out")
            break
        else:
            # print("ret:", ret)
            if ret == 99:
                save_jpgs(frame, "ori_frame_")

    cv2.destroyAllWindows()