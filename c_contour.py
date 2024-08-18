import cv2
from global_def import log

class VArea:
    def __init__(self, cnt, id, center_x, center_y, **kwargs):
        super(VArea, self).__init__(**kwargs)

        self.cnt = cnt
        self.id = id
        self.center_x = center_x
        self.center_y = center_y
        self.area_size = cv2.contourArea(self.cnt)
        self.point = 0
        self.angle = 0
        self.dist_from_bulleye = 0

    def set_point(self, point):
        self.point = point

    def set_angle(self, angle):
        self.angle = angle

    def set_dist(self, dist):
        self.dist = dist
        # log.debug("area id: %d, dist: %d", self.id, self.dist)



