import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import io
import requests
from PIL import Image
import base64
import google.generativeai as genai

# ===================== API KEYS =====================
GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"
STABILITY_API_KEY = "YOUR_STABILITY_API_KEY"

genai.configure(api_key=GOOGLE_API_KEY)

# ===================== CANVAS SETUP =====================
bpoints = [deque(maxlen=512)]
gpoints = [deque(maxlen=512)]
rpoints = [deque(maxlen=512)]
ypoints = [deque(maxlen=512)]

blue_i = green_i = red_i = yellow_i = 0

# OpenCV uses BGR
colors = [
    (255, 0, 0),    # BLUE
    (0, 255, 0),    # GREEN
    (0, 0, 255),    # RED
    (0, 255, 255)   # YELLOW
]

colorIndex = 0

paintWindow = np.ones((471, 636, 3), dtype=np.uint8) * 255
cv2.namedWindow("Paint", cv2.WINDOW_AUTOSIZE)

# ===================== GEMINI =====================
def recognize_drawing():
    try:
        canvas = paintWindow[67:, :, :]
        ok, buffer = cv2.imencode(".png", canvas)
        if not ok:
            return None

        image = Image.open(io.BytesIO(buffer))
        model = genai.GenerativeModel("gemini-pro-vision")

        response = model.generate_content(
            ["Describe this drawing in a short, simple phrase.", image]
        )
        print("Gemini:", response.text)
        return response.text

    except Exception as e:
        print("Gemini Error:", e)
        return None

# ===================== STABILITY AI =====================
def generate_image(prompt):
    try:
        url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"

        headers = {
            "Authorization": f"Bearer {STABILITY_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        payload = {
            "text_prompts": [{"text": prompt}],
            "cfg_scale": 7,
            "height": 512,
            "width": 512,
            "samples": 1,
            "steps": 30
        }

        response = requests.post(url, headers=headers, json=payload)

        if response.status_code == 200:
            img_base64 = response.json()["artifacts"][0]["base64"]
            img_bytes = base64.b64decode(img_base64)
            img_np = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            cv2.imshow("Generated Image", img_np)
        else:
            print("Stability Error:", response.text)

    except Exception as e:
        print("Generation Error:", e)

# ===================== MEDIAPIPE =====================
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.8)
mpDraw = mp.solutions.drawing_utils

# ===================== CAMERA =====================
cap = cv2.VideoCapture(0)
print("✋ Draw | Press 'r' for AI | 'q' to quit")

# ===================== MAIN LOOP =====================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ---------- Buttons ----------
    buttons = [
        ("CLEAR", 40, 140),
        ("BLUE", 160, 255),
        ("GREEN", 275, 370),
        ("RED", 390, 485),
        ("YELLOW", 505, 600)
    ]

    for i, (txt, x1, x2) in enumerate(buttons):
        col = (255, 255, 255) if txt == "CLEAR" else colors[i - 1]
        cv2.rectangle(frame, (x1, 1), (x2, 65), col, -1)
        cv2.rectangle(frame, (x1, 1), (x2, 65), (0, 0, 0), 2)
        cv2.putText(frame, txt, (x1 + 10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, hand, mpHands.HAND_CONNECTIONS)
            lm = hand.landmark

            cx, cy = int(lm[8].x * 640), int(lm[8].y * 480)
            tx, ty = int(lm[4].x * 640), int(lm[4].y * 480)

            # ✋ Pinch → new stroke
            if abs(ty - cy) < 30:
                if colorIndex == 0:
                    bpoints.append(deque(maxlen=512))
                    blue_i += 1
                elif colorIndex == 1:
                    gpoints.append(deque(maxlen=512))
                    green_i += 1
                elif colorIndex == 2:
                    rpoints.append(deque(maxlen=512))
                    red_i += 1
                elif colorIndex == 3:
                    ypoints.append(deque(maxlen=512))
                    yellow_i += 1

            # 🎨 Button click
            elif cy <= 65:
                if 40 <= cx <= 140:
                    bpoints = [deque(maxlen=512)]
                    gpoints = [deque(maxlen=512)]
                    rpoints = [deque(maxlen=512)]
                    ypoints = [deque(maxlen=512)]
                    blue_i = green_i = red_i = yellow_i = 0
                    paintWindow[67:, :, :] = 255

                elif 160 <= cx <= 255:
                    colorIndex = 0
                elif 275 <= cx <= 370:
                    colorIndex = 1
                elif 390 <= cx <= 485:
                    colorIndex = 2
                elif 505 <= cx <= 600:
                    colorIndex = 3

            # ✏️ Drawing
            else:
                if colorIndex == 0:
                    bpoints[blue_i].appendleft((cx, cy))
                elif colorIndex == 1:
                    gpoints[green_i].appendleft((cx, cy))
                elif colorIndex == 2:
                    rpoints[red_i].appendleft((cx, cy))
                elif colorIndex == 3:
                    ypoints[yellow_i].appendleft((cx, cy))

    # ---------- Draw strokes ----------
    for pts, col in zip([bpoints, gpoints, rpoints, ypoints], colors):
        for stroke in pts:
            for i in range(1, len(stroke)):
                cv2.line(frame, stroke[i - 1], stroke[i], col, 2)
                cv2.line(paintWindow, stroke[i - 1], stroke[i], col, 2)

    cv2.imshow("Output", frame)
    cv2.imshow("Paint", paintWindow)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('r'):
        desc = recognize_drawing()
        if desc:
            generate_image(desc)

cap.release()
cv2.destroyAllWindows()

