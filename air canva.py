import cv2
import numpy as np
import mediapipe as mp
import math

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Open webcam and set resolution
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Colors available
# Palettes: Black, Red, Green, Blue, Yellow, Purple
color_palette = [
    (0, 0, 0),  # Eraser/Black
    (0, 0, 255),  # Red
    (0, 255, 0),  # Green
    (255, 0, 0),  # Blue
    (0, 255, 255),  # Yellow
    (200, 0, 200)  # Purple
]
color_index = 1  # Default to Red
thickness = 6

# Tools: 0=Freehand, 1=Line, 2=Rectangle, 3=Circle
tools = ["FREEHAND", "LINE", "RECTANGLE", "CIRCLE"]
current_tool = 0

# App State Variables
canvas = None
preview_canvas = None  # Used to show shapes before confirming
px, py = 0, 0
ix, iy = 0, 0  # Initial x, y for shapes
is_drawing_shape = False

smooth_cx, smooth_cy = 0, 0

print("Pro Air Canvas Started! Press 'ESC' to exit.")


def get_fingers_up(lmList):
    """Returns a list of 5 active states [Thumb, Index, Middle, Ring, Pinky]"""
    fingers = []
    # Thumb (Check if tip is further to the right/left than PIP joint)
    # This varies if flipped, so we use distance heuristically or just check x vs x.
    # Hand is flipped, so x-axis is inverted.
    if lmList[4][1] > lmList[3][1]:
        fingers.append(1)
    else:
        fingers.append(0)

    # 4 Fingers (Check if tip y is lower (higher in image) than PIP y)
    for id in [8, 12, 16, 20]:
        if lmList[id][2] < lmList[id - 2][2]:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers


def draw_ui(frame):
    """Draws a professional Canva-like interface"""
    # 1. Left Sidebar (Tools) - Dark Gray
    cv2.rectangle(frame, (0, 0), (160, 720), (30, 30, 30), cv2.FILLED)

    # 2. Top Sidebar (Colors) - Light Gray
    cv2.rectangle(frame, (160, 0), (1280, 80), (240, 240, 240), cv2.FILLED)

    # Draw Tool Buttons
    for i, tool_name in enumerate(tools):
        y_pos = 100 + (i * 100)
        # Highlight active tool
        if current_tool == i:
            cv2.rectangle(frame, (10, y_pos - 35), (150, y_pos + 15), (70, 70, 70), cv2.FILLED)
        else:
            cv2.rectangle(frame, (10, y_pos - 35), (150, y_pos + 15), (50, 50, 50), cv2.FILLED)

        cv2.putText(frame, tool_name, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Clear Canvas Button
    cv2.rectangle(frame, (10, 500 - 35), (150, 500 + 15), (50, 50, 150), cv2.FILLED)
    cv2.putText(frame, "CLEAR ALL", (25, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Draw Color Swatches
    for i, color in enumerate(color_palette):
        x_pos = 250 + (i * 100)
        # Highlight active color
        if color_index == i:
            cv2.circle(frame, (x_pos, 40), 28, (100, 100, 100), cv2.FILLED)
            cv2.circle(frame, (x_pos, 40), 24, color, cv2.FILLED)
        else:
            cv2.circle(frame, (x_pos, 40), 24, color, cv2.FILLED)


while True:
    success, frame = cap.read()
    if not success:
        continue

    # Resize to guarantee 720p if cam defaulted smaller
    frame = cv2.resize(frame, (1280, 720))
    # Flip frame horizontally for intuitive "mirror" movement
    frame = cv2.flip(frame, 1)

    # Initialize canvases
    if canvas is None:
        canvas = np.zeros_like(frame)
        preview_canvas = np.zeros_like(frame)
    else:
        preview_canvas = np.zeros_like(frame)  # reset preview every frame

    # Draw the UI
    draw_ui(frame)

    # Process Hand Landmarks
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            h, w, c = frame.shape

            # Create a list of normalized coordinates (id, x, y)
            lmList = []
            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

            fingers = get_fingers_up(lmList)

            # Index Finger Tip
            raw_cx, raw_cy = lmList[8][1], lmList[8][2]

            # Smooth cursor movement
            if smooth_cx == 0 and smooth_cy == 0:
                smooth_cx, smooth_cy = raw_cx, raw_cy
            else:
                smooth_factor = 0.4
                smooth_cx = int(smooth_cx + smooth_factor * (raw_cx - smooth_cx))
                smooth_cy = int(smooth_cy + smooth_factor * (raw_cy - smooth_cy))

            cx, cy = smooth_cx, smooth_cy

            # Draw Custom Cursor
            cursor_color = color_palette[color_index]
            cv2.circle(frame, (cx, cy), 8, cursor_color, cv2.FILLED, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 12, (255, 255, 255), 2, cv2.LINE_AA)

            thumb_x, thumb_y = lmList[4][1], lmList[4][2]
            pinch_dist = math.hypot(thumb_x - raw_cx, thumb_y - raw_cy)

            # --- UI INTERACTIONS (Prioritized over Drawing) ---
            if cx < 160 or (cy < 80 and cx > 160):
                px, py = 0, 0  # Prevent drawing strokes from bleeding
                cv2.circle(frame, (cx, cy), 20, (200, 200, 200), 2, cv2.LINE_AA)  # Hover ring

                # Click function: Pinching thumb and index finger together
                if pinch_dist < 40:
                    cv2.circle(frame, (cx, cy), 15, (0, 255, 0), cv2.FILLED, cv2.LINE_AA)  # Green Click indicator

                    if cx < 160:  # Left Sidebar Clicks
                        y_zone = cy
                        if 65 < y_zone < 115:
                            current_tool = 0
                        elif 165 < y_zone < 215:
                            current_tool = 1
                        elif 265 < y_zone < 315:
                            current_tool = 2
                        elif 365 < y_zone < 415:
                            current_tool = 3
                        elif 465 < y_zone < 515:
                            canvas = np.zeros_like(frame)  # CLEAR

                    elif cy < 80 and cx > 160:  # Top Sidebar Clicks
                        for i in range(len(color_palette)):
                            x_center = 250 + (i * 100)
                            if abs(cx - x_center) < 30:
                                color_index = i

            else:
                # --- CANVAS INTERACTIONS ---
                # --- SELECT / MOVE MODE --- (Index and Middle fingers are UP)
                if fingers[1] and fingers[2] and sum(fingers) <= 3:
                    # Commit the shape to main canvas
                    if is_drawing_shape:
                        is_drawing_shape = False
                        canvas = cv2.add(canvas, preview_canvas)
                    px, py = 0, 0

                    # --- DRAW MODE --- (Only Index finger is UP)
                elif fingers[1] and not fingers[2] and sum(fingers) <= 2:
                    if current_tool == 0:
                        # FREEHAND
                        if px == 0 and py == 0:
                            px, py = cx, cy
                        cv2.line(canvas, (px, py), (cx, cy), cursor_color, thickness, cv2.LINE_AA)
                        px, py = cx, cy
                    else:
                        # SHAPES (Line, Rect, Circle)
                        if not is_drawing_shape:
                            is_drawing_shape = True
                            ix, iy = cx, cy  # Set starting point

                        # Live preview of the shape
                        if current_tool == 1:  # LINE
                            cv2.line(preview_canvas, (ix, iy), (cx, cy), cursor_color, thickness, cv2.LINE_AA)
                        elif current_tool == 2:  # RECTANGLE
                            cv2.rectangle(preview_canvas, (ix, iy), (cx, cy), cursor_color, thickness, cv2.LINE_AA)
                        elif current_tool == 3:  # CIRCLE
                            radius = int(math.hypot(cx - ix, cy - iy))
                            cv2.circle(preview_canvas, (ix, iy), radius, cursor_color, thickness, cv2.LINE_AA)

                # --- PALM ERASER MODE --- (All fingers extended)
                elif sum(fingers) >= 4:
                    is_drawing_shape = False
                    px, py = 0, 0

                    # Erase lines on canvas with big black circle
                    cv2.circle(canvas, (cx, cy), 60, (0, 0, 0), cv2.FILLED)
                    cv2.circle(frame, (cx, cy), 60, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(frame, "ERASING", (cx - 35, cy - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    else:  # No Hands
        px, py = 0, 0
        smooth_cx, smooth_cy = 0, 0

    # Overlay Canvas on Frame
    # We combine regular canvas and preview canvas
    combined_canvas = cv2.add(canvas, preview_canvas)

    canvas_gray = cv2.cvtColor(combined_canvas, cv2.COLOR_BGR2GRAY)
    _, inv_mask = cv2.threshold(canvas_gray, 5, 255, cv2.THRESH_BINARY_INV)
    inv_mask = cv2.cvtColor(inv_mask, cv2.COLOR_GRAY2BGR)

    # The drawing area mask (don't paint lines over the UI sidebars)
    # We can enforce this by blanking the top and left of the combined_canvas
    combined_canvas[0:80, 0:1280] = 0
    combined_canvas[0:720, 0:160] = 0

    # Recalculate inv_mask after enforcing UI borders
    canvas_gray = cv2.cvtColor(combined_canvas, cv2.COLOR_BGR2GRAY)
    _, inv_mask = cv2.threshold(canvas_gray, 5, 255, cv2.THRESH_BINARY_INV)
    inv_mask = cv2.cvtColor(inv_mask, cv2.COLOR_GRAY2BGR)

    frame = cv2.bitwise_and(frame, inv_mask)
    frame = cv2.bitwise_or(frame, combined_canvas)

    cv2.imshow("Air Canvas", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()