import cv2
import mediapipe as mp
import pyautogui
import time
import math

# Volume control libraries
from pycaw.pycaw import AudioUtilities, ISimpleAudioVolume

# Get system audio session for volume control
sessions = AudioUtilities.GetAllSessions()
volume_obj = None
for session in sessions:
    if session.SimpleAudioVolume:
        volume_obj = session.SimpleAudioVolume
        break

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

# Screen size
screen_width, screen_height = pyautogui.size()

# Webcam
cap = cv2.VideoCapture(0)

# Variables
prev_x, prev_y = 0, 0
smooth_factor = 0.2
last_click_time = 0
click_delay = 0.5
last_screenshot_time = 0
screenshot_delay = 1
prev_palm_distance = None
zoom_threshold = 30

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    hand_centers = []

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Coordinates
            x_index = int(hand_landmarks.landmark[8].x * w)
            y_index = int(hand_landmarks.landmark[8].y * h)
            x_middle = int(hand_landmarks.landmark[12].x * w)
            y_middle = int(hand_landmarks.landmark[12].y * h)
            x_thumb = int(hand_landmarks.landmark[4].x * w)
            y_thumb = int(hand_landmarks.landmark[4].y * h)
            x_wrist = int(hand_landmarks.landmark[0].x * w)
            y_wrist = int(hand_landmarks.landmark[0].y * h)

            # Draw points
            cv2.circle(frame, (x_index, y_index), 10, (0, 255, 0), cv2.FILLED)
            cv2.circle(frame, (x_thumb, y_thumb), 10, (0, 0, 255), cv2.FILLED)

            # Smooth cursor movement
            screen_x = int(hand_landmarks.landmark[8].x * screen_width)
            screen_y = int(hand_landmarks.landmark[8].y * screen_height)
            screen_x = int(prev_x + (screen_x - prev_x) * smooth_factor)
            screen_y = int(prev_y + (screen_y - prev_y) * smooth_factor)
            pyautogui.moveTo(screen_x, screen_y)
            prev_x, prev_y = screen_x, screen_y

            # CLICK GESTURE (index + thumb)
            if abs(x_index - x_thumb) < 30 and abs(y_index - y_thumb) < 30:
                if time.time() - last_click_time > click_delay:
                    pyautogui.click()
                    last_click_time = time.time()
                    cv2.putText(frame, "Click!", (x_index, y_index - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 255, 255), 2)

            # SCREENSHOT GESTURE (middle + thumb)
            if abs(x_middle - x_thumb) < 30 and abs(y_middle - y_thumb) < 30:
                if time.time() - last_screenshot_time > screenshot_delay:
                    filename = f"screenshot_{int(time.time())}.png"
                    pyautogui.screenshot(filename)
                    last_screenshot_time = time.time()
                    cv2.putText(frame, "Screenshot!", (x_middle, y_middle - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            # TWO-FINGER SCROLL
            if y_middle - y_index > 45:
                pyautogui.scroll(-100)
                cv2.putText(frame, "Scroll Down", (x_index, y_index - 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 200, 255), 2)
            if y_index - y_middle > 45:
                pyautogui.scroll(100)
                cv2.putText(frame, "Scroll Up", (x_index, y_index - 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 200, 255), 2)

            # VOLUME CONTROL (Thumb Up / Down)
            if volume_obj:
                # Thumb Up
                if y_thumb < y_wrist - 40:
                    vol = volume_obj.GetMasterVolume()
                    volume_obj.SetMasterVolume(min(vol + 0.05, 1.0), None)
                    cv2.putText(frame, "Volume UP", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

                # Thumb Down
                elif y_thumb > y_wrist + 40:
                    vol = volume_obj.GetMasterVolume()
                    volume_obj.SetMasterVolume(max(vol - 0.05, 0.0), None)
                    cv2.putText(frame, "Volume DOWN", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # Save palm center for zoom
            cx = int(hand_landmarks.landmark[0].x * w)
            cy = int(hand_landmarks.landmark[0].y * h)
            hand_centers.append((cx, cy))

    # ZOOM WITH 2 HANDS
    if len(hand_centers) == 2:
        (x1, y1), (x2, y2) = hand_centers
        distance = math.hypot(x2 - x1, y2 - y1)

        if prev_palm_distance is not None:
            if distance - prev_palm_distance > zoom_threshold:
                pyautogui.hotkey('ctrl', '+')
                cv2.putText(frame, "Zoom In", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
            elif prev_palm_distance - distance > zoom_threshold:
                pyautogui.hotkey('ctrl', '-')
                cv2.putText(frame, "Zoom Out", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

        prev_palm_distance = distance

    cv2.imshow("Virtual Mouse", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



