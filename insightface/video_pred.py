# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:video_pred.py
# software: PyCharm

import cv2
from insightface.recognize import InsightFace
import time

insightface = InsightFace()
capture = cv2.VideoCapture('./video/shu.mp4')

success = True
counter = 0

while success:
    counter += 1
    success, frame = capture.read()

    if counter % 4 == 0:
        # start = time.time()
        results = insightface.recognize(frame, detect_vis=False)
        # end = time.time()
        # print('total inference time: {}'.format(end - start))

        if results:
            face_ids, boxes, total_landmarks, distance_list = results
            # draw boxes and names
            # start2 = time.time()
            for i, box in enumerate(boxes):
                if face_ids[i] != 'stranger':
                    cv2.rectangle(frame,
                                  (int(box[0]), int(box[1])),
                                  (int(box[2]), int(box[3])),
                                  color=(0, 255, 255),
                                  thickness=2)
                    cv2.putText(frame, face_ids[i], (int(box[0]), int(box[1]) - 7),
                                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0))
                    cv2.imshow('shu', frame)
                # cv2.waitKey(100)
            # end2 = time.time()
            # print('drawing picture spends {}'.format(end2 - start2))
        else:
            cv2.imshow('shu', frame)
            # cv2.waitKey(100)
    else:
        cv2.imshow('shu', frame)

    # if press esc, stop.
    if (cv2.waitKey(35) & 0xff) == 27:
        capture.release()
        break
