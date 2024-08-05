import cv2


class VArea:
    def __init__(self, cnt, id, center_x, center_y, **kwargs):
        super(VArea, self).__init__(**kwargs)

        self.cnt = cnt
        self.id = id
        self.center_x = center_x
        self.center_y = center_y
        self.area_size = cv2.contourArea(self.cnt)
        self.point = 0

    def set_point(self, point):
        self.point = point


