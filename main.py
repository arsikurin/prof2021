import collections
import math

import cv2
import mediapipe as mp
import numpy as np
import screen_brightness_control as sbc

Results = collections.namedtuple("Results", field_names=("multi_hand_landmarks", "multi_handedness"))


def get_points(landmark, shape):
    points = []
    for mark in landmark:
        points.append((mark.x * shape[1], mark.y * shape[0]))
    return np.array(points, dtype=np.int32)


def palm_size(landmark, shape):
    """
    :return: palm enclosing circle radius
    """

    x1, y1 = landmark[0].x * shape[1], landmark[0].y * shape[0]
    x2, y2 = landmark[5].x * shape[1], landmark[5].y * shape[0]
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def help_():
    with mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=.2
    ) as handsDetector:
        flags = [0, 0, 0]
        cap = cv2.VideoCapture(0)
        prev_fist = False
        while cap.isOpened():
            ret, frame = cap.read()
            if cv2.waitKey(1) & 0xFF == ord("q") or not ret:
                break
            flipped = np.fliplr(frame)
            flippedRGB = cv2.cvtColor(flipped, code=cv2.COLOR_BGR2RGB)
            results: Results = handsDetector.process(flippedRGB)
            if results.multi_hand_landmarks is not None:
                cv2.circle(
                    flippedRGB,
                    center=(int(results.multi_hand_landmarks[0].landmark[4].x * flippedRGB.shape[1]),
                            int(results.multi_hand_landmarks[0].landmark[4].y * flippedRGB.shape[0])),
                    radius=5, color=(255, 0, 0), thickness=-1
                )
                cv2.circle(
                    flippedRGB, center=(int(results.multi_hand_landmarks[0].landmark[20].x * flippedRGB.shape[1]),
                                        int(results.multi_hand_landmarks[0].landmark[20].y * flippedRGB.shape[0])),
                    radius=5, color=(255, 0, 0), thickness=-1
                )
                cv2.line(
                    flippedRGB,
                    pt1=(int(results.multi_hand_landmarks[0].landmark[4].x * flippedRGB.shape[1]),
                         int(results.multi_hand_landmarks[0].landmark[4].y * flippedRGB.shape[0])),
                    pt2=(int(results.multi_hand_landmarks[0].landmark[20].x * flippedRGB.shape[1]),
                         int(results.multi_hand_landmarks[0].landmark[20].y * flippedRGB.shape[0])),
                    color=(255, 0, 0), thickness=2
                )

                (x, y), r = cv2.minEnclosingCircle(
                    get_points(results.multi_hand_landmarks[0].landmark, flippedRGB.shape))
                ws = palm_size(results.multi_hand_landmarks[0].landmark, flippedRGB.shape)

                if flags[0] < 7:
                    cv2.putText(
                        flippedRGB, text=f"Try clenching your fist ({flags[0]} / 6)",
                        org=(10, flippedRGB.shape[0] - 10),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(255, 255, 255), thickness=1
                    )
                    if 2 * r / ws > 1.6:
                        cv2.circle(flippedRGB, center=(int(x), int(y)), radius=int(r), color=(0, 0, 255), thickness=2)
                        if prev_fist:
                            prev_fist = False
                            flags[0] += 1
                        cv2.putText(
                            flippedRGB, text="status: [PALM]", org=(10, 50), fontFace=cv2.FONT_HERSHEY_PLAIN,
                            fontScale=3, color=(255, 255, 255), thickness=2
                        )
                    else:
                        cv2.circle(flippedRGB, (int(x), int(y)), int(r), (0, 255, 0), 2)
                        if not prev_fist:
                            prev_fist = True
                            flags[0] += 1
                        cv2.putText(
                            flippedRGB, text="status: [FIST]", org=(10, 50), fontFace=cv2.FONT_HERSHEY_PLAIN,
                            fontScale=3, color=(255, 255, 255), thickness=2
                        )

                elif flags[1] < 7:
                    cv2.putText(
                        flippedRGB, text="Try rotating your unclenched fist", org=(10, flippedRGB.shape[0] - 40),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(255, 255, 255), thickness=1
                    )
                    cv2.putText(
                        flippedRGB, text=f"around axis of the ordinate ({flags[1]} / 6)",
                        org=(10, flippedRGB.shape[0] - 10),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(255, 255, 255), thickness=1
                    )
                    if 2 * r / ws > 1.6:
                        cv2.circle(flippedRGB, center=(int(x), int(y)), radius=int(r), color=(0, 0, 255), thickness=2)
                        pinky_tip = results.multi_hand_landmarks[0].landmark[20]
                        thumb_tip = results.multi_hand_landmarks[0].landmark[4]
                        if (pinky_tip.x - thumb_tip.x + .2) * 200 > 60 and flags[2] == 0:
                            flags[1] += 1
                            flags[2] = 1
                        elif (pinky_tip.x - thumb_tip.x + .2) * 200 < 20 and flags[2] == 1:
                            flags[1] += 1
                            flags[2] = 0

                        cv2.putText(
                            flippedRGB, text="status: [ROTATING]", org=(10, 50), fontFace=cv2.FONT_HERSHEY_PLAIN,
                            fontScale=3, color=(255, 255, 255), thickness=2
                        )
                else:
                    with open("service.arsikurin", "w") as f:
                        f.write("1")
                    cv2.putText(
                        flippedRGB, text=f"Congrats! You've completed tutorial.", org=(10, flippedRGB.shape[0] - 40),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(255, 255, 255), thickness=1
                    )
                    cv2.putText(
                        flippedRGB, text=f"You should rerun the script", org=(10, flippedRGB.shape[0] - 10),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(255, 255, 255), thickness=1
                    )

            res_image = cv2.cvtColor(flippedRGB, code=cv2.COLOR_RGB2BGR)
            cv2.imshow("Result", res_image)


def main():
    with mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=.2
    ) as handsDetector:
        cap = cv2.VideoCapture(0)
        prev_fist = False
        while cap.isOpened():
            ret, frame = cap.read()
            if cv2.waitKey(1) & 0xFF == ord("q") or not ret:
                break
            flipped = np.fliplr(frame)
            flippedRGB = cv2.cvtColor(flipped, code=cv2.COLOR_BGR2RGB)
            results: Results = handsDetector.process(flippedRGB)

            if results.multi_hand_landmarks is not None:
                cv2.circle(
                    flippedRGB,
                    center=(int(results.multi_hand_landmarks[0].landmark[4].x * flippedRGB.shape[1]),
                            int(results.multi_hand_landmarks[0].landmark[4].y * flippedRGB.shape[0])),
                    radius=5, color=(255, 0, 0), thickness=-1
                )
                cv2.circle(
                    flippedRGB, center=(int(results.multi_hand_landmarks[0].landmark[20].x * flippedRGB.shape[1]),
                                        int(results.multi_hand_landmarks[0].landmark[20].y * flippedRGB.shape[0])),
                    radius=5, color=(255, 0, 0), thickness=-1
                )
                cv2.line(
                    flippedRGB,
                    pt1=(int(results.multi_hand_landmarks[0].landmark[4].x * flippedRGB.shape[1]),
                         int(results.multi_hand_landmarks[0].landmark[4].y * flippedRGB.shape[0])),
                    pt2=(int(results.multi_hand_landmarks[0].landmark[20].x * flippedRGB.shape[1]),
                         int(results.multi_hand_landmarks[0].landmark[20].y * flippedRGB.shape[0])),
                    color=(255, 0, 0), thickness=2
                )

                (x, y), r = cv2.minEnclosingCircle(
                    get_points(results.multi_hand_landmarks[0].landmark, flippedRGB.shape))
                ws = palm_size(results.multi_hand_landmarks[0].landmark, flippedRGB.shape)
                if 2 * r / ws > 1.6:
                    cv2.circle(flippedRGB, center=(int(x), int(y)), radius=int(r), color=(0, 0, 255), thickness=2)
                    if prev_fist:
                        sbc.fade_brightness(finish=100, start=1, blocking=False, increment=10)
                        prev_fist = False
                        cv2.putText(
                            flippedRGB, text="status: [PALM]", org=(10, 50), fontFace=cv2.FONT_HERSHEY_PLAIN,
                            fontScale=3, color=(255, 255, 255), thickness=2
                        )
                    else:
                        pinky_tip = results.multi_hand_landmarks[0].landmark[20]
                        thumb_tip = results.multi_hand_landmarks[0].landmark[4]
                        sbc.set_brightness((pinky_tip.x - thumb_tip.x + .2) * 200)
                        cv2.putText(
                            flippedRGB, text="status: [ROTATING]", org=(10, 50), fontFace=cv2.FONT_HERSHEY_PLAIN,
                            fontScale=3, color=(255, 255, 255), thickness=2
                        )
                else:
                    cv2.circle(flippedRGB, (int(x), int(y)), int(r), (0, 255, 0), 2)
                    if not prev_fist:
                        sbc.fade_brightness(finish=1, start=100, blocking=False, increment=10)
                        prev_fist = True
                    cv2.putText(
                        flippedRGB, text="status: [FIST]", org=(10, 50), fontFace=cv2.FONT_HERSHEY_PLAIN,
                        fontScale=3, color=(255, 255, 255), thickness=2
                    )

            res_image = cv2.cvtColor(flippedRGB, code=cv2.COLOR_RGB2BGR)
            cv2.imshow("Result", res_image)


if __name__ == "__main__":
    try:
        with open("service.arsikurin", "r") as f:
            if f.read() == "1":
                main()
            else:
                help_()
    except FileNotFoundError:
        help_()
