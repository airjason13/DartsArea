import cv2
import numpy as np
import platform

from color_method import get_green_red_image, get_black_image, red_hsv, green_hsv, get_blue_image, get_red_image
from cv2_key_event import cv2_key_event
from save_pics import save_jpgs
from c_contour import VArea
from v_math import get_angle_with_three_points, dart_points, get_angle_with_two_lines, get_distance
from global_def import log

TEST_WITH_JPG = True
TEST_SHOW_CV2_WINDOW = True

def get_white_area_list(src_img):
    area_list = []
    white_src_image = src_img.copy()
    ''' 檢測白色區塊 '''
    ''' 圖像直接二值化 '''
    gray_white_image = cv2.cvtColor(white_src_image, cv2.COLOR_BGR2GRAY)
    zret, white_image_binary = cv2.threshold(gray_white_image, 150, 255, cv2.THRESH_BINARY)

    # erode & dilage, performance is better
    kernel_temp = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    white_image_binary = cv2.erode(white_image_binary, kernel_temp)
    white_image_binary = cv2.dilate(white_image_binary, kernel_temp)

    ''' 找尋輪廓 '''
    white_cnts, hierarchy = cv2.findContours(white_image_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    new_contours = []
    for cnt in white_cnts:
        area = cv2.contourArea(cnt)
        if area > 1000:
            new_contours.append(cnt)

    contours = new_contours

    ''' contour 建立類別 '''
    v_id = 0
    for c in contours:
        M = cv2.moments(c)
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
        varea = VArea(c, v_id, center_x, center_y)
        v_id += 1
        area_list.append(varea)

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

    black_image = get_black_image(black_stage1_image)


    ''' 灰階處理二值化 '''
    gray_black_image = cv2.cvtColor(black_stage1_image, cv2.COLOR_BGR2GRAY)

    zret, black_image_gray = cv2.threshold(gray_black_image, 50, 255, cv2.THRESH_BINARY)
    # cv2.imshow("gray black Frame 2", gray_black_image)
    # erode & dilage, performance is better
    kernel_temp = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    black_image_gray = cv2.erode(black_image_gray, kernel_temp)
    black_image_gray = cv2.dilate(black_image_gray, kernel_temp)


    ''' 找尋輪廓 '''
    black_cnts, hierarchy = cv2.findContours(black_image_gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # black_cnts, hierarchy = cv2.findContours(black_image_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    new_contours = []
    for cnt in black_cnts:
        area = cv2.contourArea(cnt)
        if 10000 > area > 2000:
            new_contours.append(cnt)

    contours = new_contours


    ''' contour 建立類別 '''
    v_id = 0
    for contour in contours:
        M = cv2.moments(contour)
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
        varea = VArea(contour, v_id, center_x, center_y)
        v_id += 1
        area_list.append(varea)

    return area_list


def get_red_green_area(src_img):
    area_list = []

    img_copy = src_img.copy()
    # erode & dilage, performance is better
    kernel_temp = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    img_copy = cv2.erode(img_copy, kernel_temp)
    img_copy = cv2.dilate(img_copy, kernel_temp)

    ''' 檢測紅色綠色區塊 '''
    red_green_image = get_green_red_image(img_copy)

    ''' 找出紅色綠色contour'''
    frame_sample = red_green_image.copy()
    ''' 先將圖片轉灰階 '''
    gray_red_green_image = cv2.cvtColor(frame_sample, cv2.COLOR_BGR2GRAY)

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



    ''' 找尋輪廓 '''
    red_green_cnts, hierarchy = cv2.findContours(gray_red_green_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # red_green_cnts, hierarchy = cv2.findContours(gray_red_green_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    new_contours = []
    for cnt in red_green_cnts:
        area = cv2.contourArea(cnt)
        if area > 170:
            new_contours.append(cnt)

    contours = new_contours


    ''' contour 建立類別 '''
    v_id = 0
    for c in contours:
        M = cv2.moments(c)
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
        varea = VArea(c, v_id, center_x, center_y)
        v_id += 1
        area_list.append(varea)

    return area_list


def get_double_area_list(s_double_triple_area_list: []):
    log.debug("get_double_area_list start")
    t_double_area_list = []
    tmp_double_area_list = s_double_triple_area_list.copy()
    for area in s_double_triple_area_list:
        for tmp_area in tmp_double_area_list:
            if tmp_area.id == area.id:
                continue
                # log.debug("id matched")
            else:
                '''if area.id == 27 :
                    log.debug("area.id: %d", area.id)
                    log.debug("tmp_area.id: %d", tmp_area.id)
                    log.debug("area.angle: %d", area.angle)
                    log.debug("tmp_area.angle: %d", tmp_area.angle)
                    log.debug("area.dist: %d", area.dist)
                    log.debug("tmp_area.dist: %d", tmp_area.dist)
                    log.debug("abs(area.angle - tmp_area.angle): %d", abs(area.angle - tmp_area.angle))'''
                if 0 <= abs(area.angle - tmp_area.angle) <= 2:
                    if area.dist < tmp_area.dist:
                        log.debug("area.id : %d is appand to list", area.id)
                        log.debug("tmp_area.id : %d", tmp_area.id)
                        t_double_area_list.append(area)
    log.debug("get_double_area_list end")
    '''for double_area in t_double_area_list:
        log.debug("double_area.id :%d", double_area.id)'''

    return t_double_area_list

def get_triple_area_list(double_triple_area_list: []):
    t_triple_area_list = []
    tmp_triple_area_list = double_triple_area_list.copy()
    for area in double_triple_area_list:
        for tmp_area in tmp_triple_area_list:
            if tmp_area.id == area.id:
                pass
                # log.debug("id matched")
            else:
                if area.id == 34 and tmp_area.id == 27:
                    log.debug("area.id: %d", area.id)
                    log.debug("tmp_area.id: %d", tmp_area.id)
                    log.debug("area.angle: %d", area.angle)
                    log.debug("tmp_area.angle: %d", tmp_area.angle)
                    log.debug("area.dist: %d", area.dist)
                    log.debug("tmp_area.dist: %d", tmp_area.dist)
                if 0 <= abs(area.angle - tmp_area.angle) <= 2:
                    if area.dist > tmp_area.dist:
                        t_triple_area_list.append(area)


    return t_triple_area_list

def show_score(score: int):
    image1 = np.zeros((640, 640, 3), dtype='uint8')
    cv2.putText(image1, str(score), (0, 320), cv2.FONT_ITALIC, 10,
                (255, 255, 255), 5, cv2.LINE_AA )
    cv2.imshow("Score", image1)

class DartAreas:
    NUM_OF_ANGLE_SECTION = 18
    def __init__(self, ori_image: np.ndarray):
        self.src_image = ori_image
        self.double_triple_image = self.src_image.copy()
        self.double_triple_area_list = get_red_green_area(self.double_triple_image)

        self.black_area_image = self.src_image.copy()
        self.black_area_list = get_black_area_list(self.black_area_image)
        log.debug("len(self.black_area_list): %d", len(self.black_area_list))
        for area in self.black_area_list:
            cv2.drawContours(self.black_area_image, area.cnt, -1, (0, 255, 0), 2)
            cv2.putText(self.black_area_image, str(area.id) + "/" + str(area.area_size), (area.center_x, area.center_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("black_area_image", self.black_area_image)

        self.white_area_image = self.src_image.copy()
        self.white_area_list = get_white_area_list(self.white_area_image)
        log.debug("len(self.white_area_list): %d", len(self.white_area_list))

        self.blue_circle_center_x, self.blue_circle_center_y = self.get_blue_point()
        self.red_center_x, self.red_center_y = self.get_red_bulleye()

        self.set_black_area_data()
        self.set_white_area_data()

        self.set_double_triple_area_default_data()

        ''' 獲取雙倍區 '''
        self.double_area_list = get_double_area_list(self.double_triple_area_list)
        self.double_image_with_data = self.src_image.copy()
        self.set_double_area_data()

        ''' 獲取三倍區 '''
        self.triple_area_list = get_triple_area_list(self.double_triple_area_list)
        self.triple_image_with_data = frame.copy()
        self.set_triple_area_data()


    def get_blue_point(self) -> (int, int) :
        """ get blue circle point for calculate angle """
        blue_src_img = self.src_image.copy()
        blue_image = get_blue_image(blue_src_img)
        gray_blue_image = cv2.cvtColor(blue_image, cv2.COLOR_BGR2GRAY)
        i_ret, gray_blue_image = cv2.threshold(gray_blue_image, 20, 255, cv2.THRESH_BINARY)

        # erode & dilage, performance is better
        kernel_temp = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gray_blue_image = cv2.erode(gray_blue_image, kernel_temp)
        gray_blue_image = cv2.dilate(gray_blue_image, kernel_temp)
        gray_blue_contour, hierarchy = cv2.findContours(gray_blue_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        if len(gray_blue_contour) == 1:
            blue_circle_contour = gray_blue_contour
            for c in blue_circle_contour:
                M = cv2.moments(c)
                blue_circle_center_x = int(M["m10"] / M["m00"])
                blue_circle_center_y = int(M["m01"] / M["m00"])

        return (blue_circle_center_x, blue_circle_center_y)

    def get_red_bulleye(self) -> (int, int):
        """ get red point circle """
        red_src_img = self.src_image.copy()
        red_image = get_red_image(red_src_img)

        gray_red_image = cv2.cvtColor(red_image, cv2.COLOR_BGR2GRAY)
        ret, gray_red_image = cv2.threshold(gray_red_image, 20, 255, cv2.THRESH_BINARY)

        kernel_temp = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gray_red_image = cv2.erode(gray_red_image, kernel_temp)
        gray_red_image = cv2.dilate(gray_red_image, kernel_temp)
        gray_red_contour, hierarchy = cv2.findContours(gray_red_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        red_area_size_min = 0xffffff

        for c in gray_red_contour:
            size_tmp = cv2.contourArea(c)
            if size_tmp < red_area_size_min:
                red_area_size_min = size_tmp
                M = cv2.moments(c)
                red_center_x = int(M["m10"] / int(M["m00"]))
                red_center_y = int(M["m01"] / int(M["m00"]))

        return (red_center_x, red_center_y)


    def get_double_triple_image_with_data(self):
        tmp_image = self.double_triple_image.copy()
        for area in self.double_triple_area_list:
            cv2.drawContours(tmp_image, area.cnt, -1, (0, 255, 0), 2)
            cv2.putText(tmp_image, str(area.id) + "/" + str(area.area_size), (area.center_x, area.center_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3, (0, 255, 255), 1, cv2.LINE_AA)
            # cv2.putText(tmp_image, str(area.id) + "/" + str(area.area_size),
            #            (area.center_x, area.center_y),
            #             cv2.FONT_HERSHEY_SIMPLEX,
            #           0.5, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(tmp_image,
                        str(area.dist) + "/" + str(area.angle) + "/" + str(area.point),
                        (area.center_x, area.center_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3, (0, 0, 255), 1, cv2.LINE_AA)
        return tmp_image


    def set_black_area_data(self):
        """ 試算角度, 填入各area"""
        ' 黑色區塊填入計分值 '
        line1 = [(self.red_center_x, self.red_center_y), (self.blue_circle_center_x, self.blue_circle_center_y)]

        ' 黑色區塊填入計分值 '
        for black_area in self.black_area_list:
            line2 = [(self.red_center_x, self.red_center_y), (black_area.center_x, black_area.center_y)]
            angle = get_angle_with_two_lines(line1, line2)
            black_area.set_angle(angle)
            dist = get_distance((self.red_center_x, self.red_center_y),
                                (black_area.center_x, black_area.center_y))
            black_area.set_dist(dist)
            black_area.set_point(int(dart_points[int(angle / self.NUM_OF_ANGLE_SECTION)]))

    def set_white_area_data(self):
        line1 = [(self.red_center_x, self.red_center_y), (self.blue_circle_center_x, self.blue_circle_center_y)]
        for white_area in self.white_area_list:
            line2 = [(self.red_center_x, self.red_center_y), (white_area.center_x, white_area.center_y)]
            angle = get_angle_with_two_lines(line1, line2)
            white_area.set_angle(angle)
            dist = get_distance((self.red_center_x, self.red_center_y),
                                (white_area.center_x, white_area.center_y))
            white_area.set_dist(dist)
            white_area.set_point(int(dart_points[int(angle / self.NUM_OF_ANGLE_SECTION)]))


    def set_double_triple_area_default_data(self):
        line1 = [(self.red_center_x, self.red_center_y), (self.blue_circle_center_x, self.blue_circle_center_y)]
        ''' 紅色綠色區塊填入計分資料 '''
        for red_green_area in self.double_triple_area_list:
            line2 = [(self.red_center_x, self.red_center_y), (red_green_area.center_x, red_green_area.center_y)]
            angle = get_angle_with_two_lines(line1, line2)
            red_green_area.set_angle(angle)
            dist = get_distance((self.red_center_x, self.red_center_y),
                                (red_green_area.center_x, red_green_area.center_y))
            red_green_area.set_dist(dist)

    def set_double_area_data(self):
        line1 = [(self.red_center_x, self.red_center_y), (self.blue_circle_center_x, self.blue_circle_center_y)]
        for area in self.double_area_list:
            cv2.drawContours(self.double_image_with_data, area.cnt, -1, (0, 255, 0), 2)
            line2 = [(self.red_center_x, self.red_center_y), (area.center_x, area.center_y)]
            angle = get_angle_with_two_lines(line1, line2)

            area.set_point(2 * int(dart_points[int(angle / self.NUM_OF_ANGLE_SECTION)]))
            cv2.putText(self.double_image_with_data, str(area.id) + "/" + str(area.area_size), (area.center_x, area.center_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(self.double_image_with_data,
                        str(area.dist) + "/" + str(area.angle) + "/" + str(area.point),
                        (area.center_x, area.center_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 1, cv2.LINE_AA)


    def set_triple_area_data(self):
        line1 = [(self.red_center_x, self.red_center_y), (self.blue_circle_center_x, self.blue_circle_center_y)]
        for area in self.triple_area_list:
            line2 = [(self.red_center_x, self.red_center_y), (area.center_x, area.center_y)]
            angle = get_angle_with_two_lines(line1, line2)
            area.set_point( 3 * int(dart_points[int(angle / self.NUM_OF_ANGLE_SECTION)]))
            cv2.drawContours(self.triple_image_with_data, area.cnt, -1, (0, 255, 0), 2)
            cv2.putText(self.triple_image_with_data, str(area.id) + "/" + str(area.area_size), (area.center_x, area.center_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(self.triple_image_with_data,
                        str(area.dist) + "/" + str(area.angle) + "/" + str(area.point),
                        (area.center_x, area.center_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 1, cv2.LINE_AA)


    def get_double_area_image_with_data(self):
        return self.double_image_with_data

    def get_triple_area_image_with_data(self):
        return self.triple_image_with_data

    def get_score_with_point(self, x, y):
        point = (x, y)
        ''' 測試 點在那一個區塊內 '''
        # point_in_contour_result = 0
        for area in self.triple_area_list:
            point_in_contour_result = cv2.pointPolygonTest(area.cnt, point, False)
            # log.debug("point_in_contour_result :%d", point_in_contour_result)
            if point_in_contour_result > 0:
                log.debug("area id : %d", area.id)
                return area.point

        for area in self.double_area_list:
            point_in_contour_result = cv2.pointPolygonTest(area.cnt, point, False)
            # log.debug("point_in_contour_result :%d", point_in_contour_result)
            if point_in_contour_result > 0:
                log.debug("area id : %d", area.id)
                return area.point

        for area in self.black_area_list:
            point_in_contour_result = cv2.pointPolygonTest(area.cnt, point, False)
            # log.debug("point_in_contour_result :%d", point_in_contour_result)
            if point_in_contour_result > 0:
                log.debug("area id : %d", area.id)
                return area.point

        for area in self.white_area_list:
            point_in_contour_result = cv2.pointPolygonTest(area.cnt, point, False)
            # log.debug("point_in_contour_result :%d", point_in_contour_result)
            if point_in_contour_result > 0:
                log.debug("area id : %d", area.id)
                return area.point

        return 0



if __name__ == '__main__':
    cap = cv2.VideoCapture(2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FPS, 30)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # 建立 VideoWriter 物件，輸出影片至 output.avi
    # FPS 值為 20.0，解析度為 640x360
    out = cv2.VideoWriter('output_test.avi', fourcc, 20.0, (1280, 720))
    while True:
        ret, image = cap.read()
        if ret is not True:
            print("no image")
            break
        cv2.imshow("Ori Image", image)
        out.write(image)
        ret = cv2_key_event()
        if ret == -1:
            print("out")
            break
        else:
            # print("ret:", ret)
            if ret == 99:
                save_jpgs(image, "ori_frame_")
    cap.release()
    out.release()
    cv2.destroyAllWindows()

""" Test with JPG """
if __name__ == '__main__jpg__':
    image = cv2.imread("ori_frame_20240821_074058.jpg")
    cv2.imshow("ORI_IMAGE", image)
    frame = image[20:675, 195:915]
    cv2.imshow("SRC_IMAGE", frame)
    dart_area = DartAreas(frame)
    cv2.imshow("Double Triple Areas Image", dart_area.get_double_triple_image_with_data())
    cv2.imshow("Double Areas Image", dart_area.get_double_area_image_with_data())
    cv2.imshow("Triple Areas Image", dart_area.get_triple_area_image_with_data())
    while True:
        ret = cv2_key_event()
        if ret == -1:
            print("out")
            break
        else:
            # print("ret:", ret)
            if ret == 99:
                save_jpgs(image, "ori_frame_")

        ''' 測試 點在那一個區塊內 '''
        score = dart_area.get_score_with_point(484, 212)
        log.debug("score : %d", score)
        show_score(score)

    cv2.destroyAllWindows()


""" Test with Video """
if __name__ == '__main__video':
    cap = cv2.VideoCapture("output_test.avi")
    log.debug("cap.isOpened() : %d", cap.isOpened())
    while cap.isOpened() is True:
        iret, image = cap.read()
        if iret is not True:
            continue
        cv2.imshow("ORI_IMAGE", image)
        frame = image[20:675, 195:915]
        cv2.imshow("SRC_IMAGE", frame)
        dart_area = DartAreas(frame)
        cv2.imshow("Double Triple Areas Image", dart_area.get_double_triple_image_with_data())
        cv2.imshow("Double Areas Image", dart_area.get_double_area_image_with_data())
        cv2.imshow("Triple Areas Image", dart_area.get_triple_area_image_with_data())

        ret = cv2_key_event()
        if ret == -1:
            print("out")
            break
        else:
            # print("ret:", ret)
            if ret == 99:
                save_jpgs(image, "ori_frame_")

        ''' 測試 點在那一個區塊內 '''
        # score = dart_area.get_score_with_point(484, 212)
        # log.debug("score : %d", score)

    cv2.destroyAllWindows()



