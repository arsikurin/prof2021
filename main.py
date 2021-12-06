import collections

import cv2
import mediapipe as mp
import numpy as np
import screen_brightness_control as sbc

Results = collections.namedtuple("Results", field_names=("multi_hand_landmarks", "multi_handedness"))


def get_points(landmark, shape):
    points = []
    for mark in landmark:
        points.append([mark.x * shape[1], mark.y * shape[0]])
    return np.array(points, dtype=np.int32)


def palm_size(landmark, shape):
    x1, y1 = landmark[0].x * shape[1], landmark[0].y * shape[0]
    x2, y2 = landmark[5].x * shape[1], landmark[5].y * shape[0]
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** .5


with mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
) as handsDetector:
    cap = cv2.VideoCapture(0)
    prev_fist = False
    while cap.isOpened():
        ret, frame = cap.read()
        if cv2.waitKey(1) & 0xFF == ord("q") or not ret:
            break
        flipped = np.fliplr(frame)
        flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
        results: Results = handsDetector.process(flippedRGB)

        if results.multi_hand_landmarks is not None:
            cv2.drawContours(
                flippedRGB, (get_points(results.multi_hand_landmarks[0].landmark, flippedRGB.shape),),
                0, (255, 0, 0), 2
            )
            (x, y), r = cv2.minEnclosingCircle(get_points(results.multi_hand_landmarks[0].landmark, flippedRGB.shape))
            ws = palm_size(results.multi_hand_landmarks[0].landmark, flippedRGB.shape)
            if 2 * r / ws > 1.5:
                cv2.circle(flippedRGB, (int(x), int(y)), int(r), (0, 0, 255), 2)
                if prev_fist:
                    sbc.fade_brightness(finish=100, start=1, blocking=False, increment=10)
                    prev_fist = False
                else:
                    pinky_tip = results.multi_hand_landmarks[0].landmark[20]
                    thumb_tip = results.multi_hand_landmarks[0].landmark[4]
                    sbc.set_brightness((pinky_tip.x - thumb_tip.x + .2) * 200)
                    print(round((pinky_tip.x - thumb_tip.x + .2) * 200), "%")
            else:
                cv2.circle(flippedRGB, (int(x), int(y)), int(r), (0, 255, 0), 2)
                if not prev_fist:
                    sbc.fade_brightness(finish=1, start=100, blocking=False, increment=10)
                    prev_fist = True

        res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
        cv2.imshow("Res", res_image)
