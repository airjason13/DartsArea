import math
from global_def import log

dart_points = [
               6,
               13,
               4,
               18,
               1,
               20,
               5,
               12,
               9,
               14,
               11,
               8,
               16,
               7,
               19,
               3,
               17,
               2,
               15,
               10,]


def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1])*(p1[1] - p2[1]))



''' 三點求角度 '''
def get_angle_with_three_points(p1, p2, p3):
    a = dist(p1, p2)
    b = dist(p2, p3)
    c = dist(p1, p3)

    angle = math.acos((a*a + b*b - c*c)/(2*a*b))
    return angle/math.pi*180


''' 兩條線求角度 '''
def get_angle_with_two_lines(line1, line2):
    '''log.debug("line1[0][0] : %d", line1[0][0])
    log.debug("line1[1][0] : %d", line1[1][0])
    log.debug("line2[0][0] : %d", line2[0][0])
    log.debug("line2[1][0] : %d", line2[1][0])'''
    dx1 = line1[0][0] - line1[1][0]
    dy1 = line1[0][1] - line1[1][1]
    dx2 = line2[0][0] - line2[1][0]
    dy2 = line2[0][1] - line2[1][1]
    angle1 = math.atan2(dy1, dx1)
    angle1 = int(angle1 * 180/math.pi)
    angle2 = math.atan2(dy2, dx2)
    angle2 = int(angle2 * 180/math.pi)



    log.debug("angle1 : %d", angle1)
    log.debug("angle2 : %d", angle2)
    if dx2 < 0:
        if angle1*angle2 >= 0:
            inside_angle = abs(angle1 - angle2)
        else:
            inside_angle = abs(angle1) + abs(angle2)
            if inside_angle > 180:
                inside_angle = 360 - inside_angle
        inside_angle = inside_angle % 180
    else:
        if angle1 * angle2 >= 0:
            inside_angle = abs(angle1 - angle2)
            inside_angle = 360 - inside_angle
        else:
            inside_angle = abs(angle1) + abs(angle2)
            inside_angle = 360 - inside_angle

    log.debug("inside_angle : %d", inside_angle)
    return inside_angle


def get_angle_with_two_lines_method1(line1, line2):
    '''log.debug("line1[0][0] : %d", line1[0][0])
    log.debug("line1[1][0] : %d", line1[1][0])
    log.debug("line2[0][0] : %d", line2[0][0])
    log.debug("line2[1][0] : %d", line2[1][0])'''
    dx1 = line1[0][0] - line1[1][0]
    dy1 = line1[0][1] - line1[1][1]
    dx2 = line2[0][0] - line2[1][0]
    dy2 = line2[0][1] - line2[1][1]
    m1 = dy1/dx1
    m2 = dy2/dx2
    inside_angle = math.atan(abs((m2-m1)/(1 + (m1*m2))))
    angle = inside_angle/math.pi*180
    log.debug("angle: %d", angle)
    if 370 > angle > -370:
        angle = int(angle)
    return angle

