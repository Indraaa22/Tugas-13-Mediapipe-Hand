import cv2
import mediapipe

cap = cv2.VideoCapture(0)
mediapipehand = mediapipe.solutions.hands
hands = mediapipehand.Hands()

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    h, w, c = img.shape
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_handedness:
        for idx, hand in enumerate(results.multi_handedness):
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 2
            thickness_outline = 8
            thickness_text = 4
            if hand.classification[0].index == 1:
                text = "RIGHT"
                (text_width, text_height), _ = cv2.getTextSize(text, font, scale, thickness_text)
                x = w - text_width - 20
                y = text_height + 20
                cv2.putText(img, text, (x, y),
                            font, scale, (0, 0, 0), thickness_outline, cv2.LINE_AA)
                cv2.putText(img, text, (x, y),
                            font, scale, (255, 0, 0), thickness_text, cv2.LINE_AA)
            elif hand.classification[0].index == 0:
                text = "LEFT"
                x = 20
                y = 60
                cv2.putText(img, text, (x, y),
                            font, scale, (0, 0, 0), thickness_outline, cv2.LINE_AA)
                cv2.putText(img, text, (x, y),
                            font, scale, (0, 0, 255), thickness_text, cv2.LINE_AA)
    cv2.imshow("webcam", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()