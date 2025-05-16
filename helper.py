import numpy as np
import cv2
import math
index_map = {
    0: 0,    # Nose
    2: 15,    # Left Eye
    5: 14,    # Right Eye
    7: 17,    # Left Ear
    8: 16,    # Right Ear
    11: 5,   # Left Shoulder
    12: 2,   # Right Shoulder
    13: 6,   # Left Elbow
    14: 3,   # Right Elbow
    15: 7,   # Left Wrist
    16: 4,  # Right Wrist
    23: 11,  # Left Hip
    24: 8,  # Right Hip
    25: 12,  # Left Knee
    26: 9,  # Right Knee
    27: 13,  # Left Ankle
    28: 10   # Right Ankle
}


def convert_mediapipe_to_openpose(landmarks, image_width, image_height, visibility_threshold=0.5):

    keypoints = np.zeros((18, 3), dtype=np.float32)


    for mp_idx, op_idx in index_map.items():
        lm = landmarks[mp_idx]
        if lm.visibility > visibility_threshold:
            keypoints[op_idx] = [
                int(lm.x * image_width), int(lm.y * image_height), lm.visibility]

    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    if left_shoulder.visibility > visibility_threshold and right_shoulder.visibility > visibility_threshold:
        neck_x = int((left_shoulder.x + right_shoulder.x) / 2 * image_width)
        neck_y = int((left_shoulder.y + right_shoulder.y) / 2 * image_height)
        keypoints[1] = [neck_x, neck_y,
                        (left_shoulder.visibility+right_shoulder.visibility)/2]

    return keypoints

def resize_image(image,size=368):
    scale =size/ image.shape[0]
    image = cv2.resize(image,None,fx=scale,fy=scale)
    return image

limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
    [10, 11], [2, 12], [12, 13], [13, 14], [2, 1]]
colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
        [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
        [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

def draw_bodypose(canvas, candidate):
    stickwidth = 4

    # Draw keypoints
    for i in range(14):
        x, y = candidate[i][0:2]
        if x == 0 and y == 0:
            continue
        cv2.circle(canvas, (int(x), int(y)), 4, colors[i % len(colors)], thickness=-1)

    # Draw limbs
    for idx, (i, j) in enumerate(limbSeq):
        if (candidate[i - 1][0] == 0 and candidate[i - 1][1] == 0) or \
           (candidate[j - 1][0] == 0 and candidate[j - 1][1] == 0):
            continue

        x1, y1 = candidate[i - 1][0:2]
        x2, y2 = candidate[j - 1][0:2]

        # midpoint and length
        mX = int((x1 + x2) / 2)
        mY = int((y1 + y2) / 2)
        length = math.hypot(x2 - x1, y2 - y1)
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))

        # ellipse as limb
        polygon = cv2.ellipse2Poly((mX, mY), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        cur_canvas = canvas.copy()
        cv2.fillConvexPoly(cur_canvas, polygon, colors[idx % len(colors)])
        canvas = cv2.addWeighted(canvas, 0.6, cur_canvas, 0.4, 0)

    return canvas