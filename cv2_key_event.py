import cv2

def cv2_key_event():
    global selected_point_index
    key = cv2.waitKey(1)
    if key == ord('q'):
        return -1

    if key == ord('p'):
        while True:
            key2 = cv2.waitKey(1)
            if key2 == ord('p'):
                break
    if key == ord('1'):
        print('got btn 1 press')
        selected_point_index = 0
    elif key == ord('2'):
        print('got btn 2 press')
        selected_point_index = 1
    elif key == ord('3'):
        print('got btn 3 press')
        selected_point_index = 2
    elif key == ord('4'):
        print('got btn 4 press')
        selected_point_index = 3

    '''if key == 82:  # 'up' key
        print('got btn up press')
        x, y = warp_points[selected_point_index]
        if y == 0:
            pass
        else:
            y -= 1
        warp_points[selected_point_index] = (x, y)
    if key == 81:  # 'left' key
        print('got btn left press')
        x, y = warp_points[selected_point_index]
        if x == 0:
            pass
        else:
            x -= 1
        warp_points[selected_point_index] = (x, y)

    if key == 84:  # 'down' key
        print('got btn down press')
        x, y = warp_points[selected_point_index]
        if y >= 720:
            pass
        else:
            y += 1
        warp_points[selected_point_index] = (x, y)

    if key == 83:  # 'right' key
        print('got btn right press')
        x, y = warp_points[selected_point_index]
        if x >= 1280:
            pass
        else:
            x += 1
        warp_points[selected_point_index] = (x, y)

    # get the warp point
    if key == ord('i'):
        print('got warp point')
        print('warp_points : ', warp_points)'''

    if key == -1:
        return 0

    return key