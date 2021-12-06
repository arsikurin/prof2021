import collections

import cv2
import mediapipe as mp
import numpy as np

from dependencies import Vector

Results = collections.namedtuple("Results", field_names=("multi_hand_landmarks", "multi_handedness"))


def gen_vectors(img, mcp_num: int, tip_num: int, i):
    mcp = results.multi_hand_landmarks[i].landmark[mcp_num]
    tip = results.multi_hand_landmarks[i].landmark[tip_num]
    cv2.line(
        img,
        (round(mcp.x * img.shape[1]), round(mcp.y * img.shape[0])),
        (round(tip.x * img.shape[1]), round(tip.y * img.shape[0])),
        (255, 0, 0), 5
    )
    return Vector(
        _x=mcp.x * img.shape[1],
        _y=mcp.y * img.shape[0],
        _a=tip.x * img.shape[1],
        _b=tip.y * img.shape[0]
    )


with mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
) as handsDetector:
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if cv2.waitKey(1) & 0xFF == ord("q") or not ret:
            break
        flippedRGB = cv2.cvtColor(np.fliplr(frame), cv2.COLOR_BGR2RGB)

        results: Results = handsDetector.process(flippedRGB)
        if results.multi_hand_landmarks is not None:
            for i, hand in enumerate(results.multi_hand_landmarks):
                thumb_vec = gen_vectors(flippedRGB, 2, 4, i)
                index_vec = gen_vectors(flippedRGB, 5, 8, i)
                middle_vec = gen_vectors(flippedRGB, 9, 12, i)
                ring_vec = gen_vectors(flippedRGB, 13, 16, i)
                pinky_vec = gen_vectors(flippedRGB, 17, 20, i)

                flags = (
                    1.745 > thumb_vec.angle > 1.308,
                    1.745 > index_vec.angle > 1.308,
                    1.745 > middle_vec.angle > 1.308,
                    1.745 > ring_vec.angle > 1.308,
                    1.745 > pinky_vec.angle > 1.308
                )
                if all(flags):
                    print("STOP!")

                # for finger in results.multi_hand_landmarks[i].landmark:
                #     x_tip = int(
                #         finger.x * flippedRGB.shape[1]
                #     )
                #     y_tip = int(
                #         finger.y * flippedRGB.shape[0]
                #     )
                #
                #     cv2.circle(flippedRGB, (x_tip, y_tip), 5, (255, 0, 0), -1)
                # print(results.multi_handedness)
                # print(results.multi_hand_landmarks[i])
        res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
        cv2.imshow("Hands", res_image)
