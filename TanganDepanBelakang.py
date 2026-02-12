import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            h, w, _ = img.shape
            index_x = int(hand_landmarks.landmark[5].x * w)
            pinky_x = int(hand_landmarks.landmark[17].x * w)
            label = results.multi_handedness[idx].classification[0].label
            if label == "Right":
                if index_x < pinky_x:
                    arah = "Front End"
                    warna = (255, 0, 0)
                else:
                    arah = "Back End"
                    warna = (0, 0, 255)
            elif label == "Left":
                if index_x < pinky_x:
                    arah = "Back End"
                    warna = (0, 0, 255)
                else:
                    arah = "Front End"
                    warna = (255, 0, 0)
            cv2.putText(img, f"{label} - {arah}", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, warna, 3, cv2.LINE_AA)
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    cv2.imshow("webcam", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()