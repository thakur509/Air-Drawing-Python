import cv2 as cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe and variables
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
canvas = None
prev_x, prev_y = 0, 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip for mirror view
    h, w, c = frame.shape
    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # Convert to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            # Get index finger tip coordinates (landmark 8)
            lm = handLms.landmark[8]
            x, y = int(lm.x * w), int(lm.y * h)

            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = x, y

            # Draw line on the canvas
            cv2.line(canvas, (prev_x, prev_y), (x, y), (255, 0, 0), 5)

            prev_x, prev_y = x, y

            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    # Combine canvas and original frame
    combined = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    # Show the final output
    cv2.imshow("Air Drawing", combined)

    key = cv2.waitKey(1)
    if key == ord('c'):  # Clear drawing
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        prev_x, prev_y = 0, 0
    elif key == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()

