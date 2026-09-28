import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time
import math
from pycaw.pycaw import AudioUtilities

# --- SYSTEM OPTIMIZATIONS ---
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

screen_width, screen_height = pyautogui.size()

# --- STATE VARIABLES ---
current_mode = "mouse"
last_switch_time = 0
SWITCH_COOLDOWN = 0.8

cloc_x, cloc_y = 0, 0
ploc_x, ploc_y = 0, 0
smoothing = 4

last_click_time = 0
last_screenshot_time = 0

color_palette = [(0, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (200, 0, 200)]
color_index = 1
thickness = 8
tools = ["PEN", "LINE", "RECT", "CIRCLE"]
current_tool = 0
canvas = None
px, py = 0, 0
ix, iy = 0, 0
is_drawing_shape = False

sessions = AudioUtilities.GetAllSessions()
volume_obj = next((s.SimpleAudioVolume for s in sessions if s.SimpleAudioVolume), None)


def get_fingers_up(lmList):
    fingers = []
    # Thumb logic
    if lmList[4][1] > lmList[3][1]:
        fingers.append(1)
    else:
        fingers.append(0)
    for id in [8, 12, 16, 20]:
        if lmList[id][2] < lmList[id - 2][2]:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers


def draw_ui(frame):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (160, 720), (45, 45, 45), -1)
    cv2.rectangle(overlay, (160, 0), (1280, 80), (230, 230, 230), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    for i, t in enumerate(tools):
        y = 120 + (i * 100)
        c = (140, 140, 140) if current_tool == i else (70, 70, 70)
        cv2.rectangle(frame, (15, y - 40), (145, y + 20), c, -1)
        cv2.putText(frame, t, (45, y), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
    for i, col in enumerate(color_palette):
        x = 280 + (i * 100)
        if color_index == i:
            cv2.circle(frame, (x, 40), 30, (200, 200, 200), 2)
        cv2.circle(frame, (x, 40), 22, col, -1)


while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    if canvas is None:
        canvas = np.zeros_like(frame)

    preview = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    if res.multi_hand_landmarks:
        hand_lms = res.multi_hand_landmarks[0]
        lmList = [[id, int(lm.x * w), int(lm.y * h)] for id, lm in enumerate(hand_lms.landmark)]
        fingers = get_fingers_up(lmList)

        xt, yt = lmList[4][1], lmList[4][2]  # Thumb
        xi, yi = lmList[8][1], lmList[8][2]  # Index
        xm, ym = lmList[12][1], lmList[12][2]  # Middle
        xp, yp = lmList[20][1], lmList[20][2]  # Pinky
        xw, yw = lmList[0][1], lmList[0][2]  # Wrist

        # --- 1. MODE SWITCH (Thumb to Pinky) ---
        if math.hypot(xt - xp, yt - yp) < 30:
            if (time.time() - last_switch_time > SWITCH_COOLDOWN):
                current_mode = "canvas" if current_mode == "mouse" else "mouse"
                last_switch_time = time.time()
                px, py = 0, 0

                # --- 2. SCREENSHOT (Thumb to Middle) ---
        # FIXED: Removed from inside the Mode Switch block and fixed indentation
        if math.hypot(xt - xm, yt - ym) < 35:
            if time.time() - last_screenshot_time > 2.0:
                # Capture visual feedback
                cv2.circle(frame, (xm, ym), 30, (0, 255, 0), cv2.FILLED)
                pyautogui.screenshot(f"aura_snap_{int(time.time())}.png")
                last_screenshot_time = time.time()
                frame[:, :, :] = 255  # Flash visual

        # --- 3. MODE LOGIC ---
        if current_mode == "mouse":
            # --- MOUSE MOVEMENT ---
            frameR = 150
            val_x = np.interp(xi, (frameR, w - frameR), (0, screen_width))
            val_y = np.interp(yi, (frameR, h - frameR), (0, screen_height))
            cloc_x = ploc_x + (val_x - ploc_x) / smoothing
            cloc_y = ploc_y + (val_y - ploc_y) / smoothing
            pyautogui.moveTo(cloc_x, cloc_y)
            ploc_x, ploc_y = cloc_x, cloc_y

            # --- CLICK ---
            if math.hypot(xi - xt, yi - yt) < 30:
                if time.time() - last_click_time > 0.3:
                    pyautogui.click()
                    last_click_time = time.time()

                    # --- LIQUID SCROLLING ---
            if fingers[1] and fingers[2]:
                scroll_distance = yi - ym
                if abs(scroll_distance) > 25:
                    scroll_val = int((scroll_distance) * 0.8)
                    pyautogui.scroll(scroll_val)
                    cv2.line(frame, (xi, yi), (xm, ym), (255, 0, 255), 2)

                    # --- VOLUME ---
            if volume_obj:
                if yt < yw - 100:
                    volume_obj.SetMasterVolume(min(volume_obj.GetMasterVolume() + 0.02, 1.0), None)
                elif yt > yw + 60:
                    volume_obj.SetMasterVolume(max(volume_obj.GetMasterVolume() - 0.02, 0.0), None)

        else:
            # --- AIR CANVAS ---
            draw_ui(frame)

            if sum(fingers) == 5:
                e_rad = 65
                cv2.circle(frame, (xi, yi), e_rad, (180, 180, 180), 2)
                cv2.circle(canvas, (xi, yi), e_rad, (0, 0, 0), -1)
                px, py = 0, 0

            elif xi < 160 or yi < 80:
                px, py = 0, 0

                if math.hypot(xt - xi, yt - yi) < 35:

                    if xi < 160:

                        if 80 < yi < 140:
                            current_tool = 0

                        elif 180 < yi < 240:
                            current_tool = 1

                        elif 280 < yi < 340:
                            current_tool = 2

                        elif 380 < yi < 440:
                            current_tool = 3

                        elif 460 < yi < 520:
                            canvas = np.zeros_like(frame)

                    elif yi < 80:
                        for i in range(len(color_palette)):
                            if abs(xi - (280 + (i * 100))) < 40:
                                color_index = i

            else:

                if fingers[1] and not fingers[2]:
                    col = color_palette[color_index]

                    if current_tool == 0:

                        if px == 0:
                            px, py = xi, yi

                        cv2.line(
                            canvas,
                            (px, py),
                            (xi, yi),
                            col,
                            thickness,
                            cv2.LINE_AA
                        )

                        px, py = xi, yi

                    else:

                        if not is_drawing_shape:
                            is_drawing_shape = True
                            ix, iy = xi, yi

                        if current_tool == 1:
                            cv2.line(
                                preview,
                                (ix, iy),
                                (xi, yi),
                                col,
                                thickness,
                                cv2.LINE_AA
                            )

                        elif current_tool == 2:
                            cv2.rectangle(
                                preview,
                                (ix, iy),
                                (xi, yi),
                                col,
                                thickness,
                                cv2.LINE_AA
                            )

                        elif current_tool == 3:
                            cv2.circle(
                                preview,
                                (ix, iy),
                                int(math.hypot(xi - ix, yi - iy)),
                                col,
                                thickness,
                                cv2.LINE_AA
                            )

                elif fingers[1] and fingers[2]:

                    if is_drawing_shape:
                        is_drawing_shape = False
                        canvas = cv2.add(canvas, preview)

                    px, py = 0, 0

                    # Composite Layers
    canvas[0:80, :] = 0
    canvas[:, 0:160] = 0

    res_draw = cv2.add(canvas, preview)

    gray = cv2.cvtColor(
        res_draw,
        cv2.COLOR_BGR2GRAY
    )

    _, mask = cv2.threshold(
        gray,
        1,
        255,
        cv2.THRESH_BINARY_INV
    )

    frame = cv2.bitwise_and(
        frame,
        cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    )

    frame = cv2.bitwise_or(
        frame,
        res_draw
    )

    cv2.putText(
        frame,
        f"MODE: {current_mode.upper()}",
        (200, 55),
        cv2.FONT_HERSHEY_DUPLEX,
        1,
        (0, 255, 0) if current_mode == "mouse" else (0, 0, 255),
        2
    )

    cv2.imshow(
        "Aura Control Hub",
        frame
    )

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()