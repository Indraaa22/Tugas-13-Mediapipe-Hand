import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    h, w, c = img.shape
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        print("Tangan terdeteksi")
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            for id, titik in enumerate(hand_landmarks.landmark):
                print(f"ID: {id} | x: {titik.x:.3f} | y: {titik.y:.3f}")
            label = results.multi_handedness[idx].classification[0].label
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 1.2
            thickness_outline = 6
            thickness_text = 2

            if label == "Right":
                posisi_text = (w - 250, 60)
                warna_lr = (255, 0, 0)
            else:
                posisi_text = (20, 60)
                warna_lr = (0, 0, 255)
            cv2.putText(img, label.upper(), posisi_text,
                        font, scale, (0, 0, 0), thickness_outline, cv2.LINE_AA)
            cv2.putText(img, label.upper(), posisi_text,
                        font, scale, warna_lr, thickness_text, cv2.LINE_AA)

            index_x = int(hand_landmarks.landmark[5].x * w)
            pinky_x = int(hand_landmarks.landmark[17].x * w)

            if label == "Right":
                if index_x < pinky_x:
                    arah = "Front End"
                    warna_fb = (255, 0, 0)
                else:
                    arah = "Back End"
                    warna_fb = (0, 0, 255)
            else:  # Left
                if index_x < pinky_x:
                    arah = "Back End"
                    warna_fb = (0, 0, 255)
                else:
                    arah = "Front End"
                    warna_fb = (255, 0, 0)
            y_pos = h - 30 - (idx * 40)  # tiap tangan turun 40px
            cv2.putText(img, f"{label} - {arah}", (20, y_pos),
                        font, 1, warna_fb, 3, cv2.LINE_AA)
    else:
        print("Tidak ada tangan")
    cv2.imshow("WEB CAM - FULL SYSTEM", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()