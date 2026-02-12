import cv2
import mediapipe

capture = cv2.VideoCapture(0)
mediapipehand = mediapipe.solutions.hands
tangan = mediapipehand.Hands(max_num_hands=2)
mpdraw = mediapipe.solutions.drawing_utils

while True:
    success, img = capture.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = tangan.process(imgRGB)
    if results.multi_hand_landmarks:
        for jalurtangan in results.multi_hand_landmarks:
            mpdraw.draw_landmarks(img,jalurtangan,mediapipehand.HAND_CONNECTIONS)
        for id, titik in enumerate(jalurtangan.landmark):
            print (id)
            print(titik.x)
            print(titik.y)
    cv2.imshow('webcam',img)
    cv2.waitKey(10)
    if cv2.waitKey(10) & 0xff == ord('q'):
        break
capture.release()
cv2.destroyAllWindows()