import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode
color = (255, 255, 255)


def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_gesture_result
    latest_gesture_result = result


latest_gesture_result = None

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='./gesture_recognizer.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

recognizer = GestureRecognizer.create_from_options(options)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)


def drawGestureBoundingBox(frame, hand_landmarks, gesture_name, confidence, color=color):
    h, w, _ = frame.shape
    x_coords = [int(landmark.x * w) for landmark in hand_landmarks.landmark]
    y_coords = [int(landmark.y * h) for landmark in hand_landmarks.landmark]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    padding = 20
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w, x_max + padding)
    y_max = min(h, y_max + padding)
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 10)
    label = f"{gesture_name}: {confidence:.2f}"
    cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def showTextOnFrame(frame, text, position=(50, 50), color=(255, 255, 255), font_scale=1, thickness=2):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    return frame


def displayPredictionText(frame, gesture_name, confidence, color=color):
    h, w, _ = frame.shape
    if gesture_name.lower() == "thumbs_up":
        text = f"Acting real cool"
    elif gesture_name.lower() == "open_palm":
        text = f"5 second intervals.."
    elif gesture_name.lower() == "cross":
        text = f"divine intervention?"
    elif gesture_name.lower() == "little":
        text = f"thats some miniscule chances"
    elif gesture_name.lower() == "me":
        text = f"I"
    elif gesture_name.lower() == "memorize":
        text = f"memorized each line on your face"
    elif gesture_name.lower() == "middle_finger":
        text = f"used to read you like a beckett play"
    elif gesture_name.lower() == "none":
        text = f":)"
    elif gesture_name.lower() == "number_one":
        text = f"same place"
    elif gesture_name.lower() == "passing":
        text = f"saw you in passing"
    elif gesture_name.lower() == "pinky_up":
        text = f"crying under my glasses"
    elif gesture_name.lower() == "rock":
        text = f"or another hard lesson"
    elif gesture_name.lower() == "slide":
        text = f"let you slide in at least 10 million ways"
    elif gesture_name.lower() == "speak":
        text = f"tell me what you're too scared to say"
    elif gesture_name.lower() == "pointing" or gesture_name.lower() == "point":
        text = f"same place"
    elif gesture_name.lower() == "victory":
        text = f"same time"
    else:
        text = f""
        color = (255, 255, 255)
    font_scale = 1.5
    thickness = 3
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    text_x = (w - text_size[0]) // 2
    text_y = h - 50
    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)


def isThumbsUp(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
    thumb_extended = thumb_tip.y < thumb_ip.y < thumb_mcp.y
    index_curled = index_tip.y > index_pip.y
    middle_curled = middle_tip.y > middle_pip.y
    ring_curled = ring_tip.y > ring_pip.y
    pinky_curled = pinky_tip.y > pinky_pip.y
    return thumb_extended and index_curled and middle_curled and ring_curled and pinky_curled


def isOpenPalm(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
    ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
    thumb_extended = thumb_tip.y < thumb_ip.y
    index_extended = index_tip.y < index_pip.y < index_mcp.y
    middle_extended = middle_tip.y < middle_pip.y < middle_mcp.y
    ring_extended = ring_tip.y < ring_pip.y < ring_mcp.y
    pinky_extended = pinky_tip.y < pinky_pip.y < pinky_mcp.y
    index_middle_distance = abs(index_tip.x - middle_tip.x)
    middle_ring_distance = abs(middle_tip.x - ring_tip.x)
    ring_pinky_distance = abs(ring_tip.x - pinky_tip.x)
    thumb_index_distance = abs(thumb_tip.x - index_tip.x)
    min_spread_distance = 0.03
    fingers_spread = (index_middle_distance > min_spread_distance and
                     middle_ring_distance > min_spread_distance and
                     ring_pinky_distance > min_spread_distance and
                     thumb_index_distance > min_spread_distance)
    return (thumb_extended and index_extended and middle_extended and
            ring_extended and pinky_extended and fingers_spread)


def isVictory(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_pip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
    index_extended = index_tip.y < index_pip.y < index_mcp.y
    middle_extended = middle_tip.y < middle_pip.y < middle_mcp.y
    thumb_curled = thumb_tip.y > thumb_pip.y
    ring_curled = ring_tip.y > ring_pip.y
    pinky_curled = pinky_tip.y > pinky_pip.y
    finger_spread = abs(index_tip.x - middle_tip.x) > 0.02
    return (index_extended and middle_extended and thumb_curled and
            ring_curled and pinky_curled and finger_spread)


def drawBoundingBoxWithLabel(frame, face_landmarks, label, color=color):
    h, w, _ = frame.shape
    x_min = min([int(landmark.x * w) for landmark in face_landmarks.landmark])
    y_min = min([int(landmark.y * h) for landmark in face_landmarks.landmark])
    x_max = max([int(landmark.x * w) for landmark in face_landmarks.landmark])
    y_max = max([int(landmark.y * h) for landmark in face_landmarks.landmark])
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 10)
    cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 5)


def isMouthOpen(face_landmarks):
    upper_lip = face_landmarks.landmark[13]
    lower_lip = face_landmarks.landmark[14]
    mouth_open_distance = abs(upper_lip.y - lower_lip.y)
    return mouth_open_distance > 0.03


def isWink(face_landmarks):
    left_eye_upper = face_landmarks.landmark[159]
    left_eye_lower = face_landmarks.landmark[145]
    right_eye_upper = face_landmarks.landmark[386]
    right_eye_lower = face_landmarks.landmark[374]
    left_eye_distance = abs(left_eye_upper.y - left_eye_lower.y)
    right_eye_distance = abs(right_eye_upper.y - right_eye_lower.y)
    return (left_eye_distance < 0.02 and right_eye_distance > 0.03) or \
           (right_eye_distance < 0.02 and left_eye_distance > 0.03)


def drawMouthBoundingBox(frame, face_landmarks, label, color=(0, 255, 0)):
    h, w, _ = frame.shape
    mouth_landmarks = [face_landmarks.landmark[i] for i in [13, 14]]
    x_min = min([int(landmark.x * w) for landmark in mouth_landmarks])
    y_min = min([int(landmark.y * h) for landmark in mouth_landmarks])
    x_max = max([int(landmark.x * w) for landmark in mouth_landmarks])
    y_max = max([int(landmark.y * h) for landmark in mouth_landmarks])
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
    cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


frame_timestamp = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    recognizer.recognize_async(mp_image, frame_timestamp)
    frame_timestamp += 1
    face_results = face_mesh.process(rgb_frame)
    hand_results = hands.process(rgb_frame)
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            if isThumbsUp(hand_landmarks):
                drawGestureBoundingBox(frame, hand_landmarks, "Thumbs Up", 1, color)
                displayPredictionText(frame, "Thumbs_Up", 10)
            elif isOpenPalm(hand_landmarks):
                drawGestureBoundingBox(frame, hand_landmarks, "Open Palm", 1, color)
                displayPredictionText(frame, "open_palm", 10)
            elif isVictory(hand_landmarks):
                drawGestureBoundingBox(frame, hand_landmarks, "Victory", 1, color)
                displayPredictionText(frame, "Victory", 10)
            elif latest_gesture_result and latest_gesture_result.gestures:
                top_gesture = latest_gesture_result.gestures[0][0]
                confidence = top_gesture.score
                gesture_name = top_gesture.category_name
                if confidence > 0.5:
                    drawGestureBoundingBox(frame, hand_landmarks, gesture_name, confidence, color)
                    displayPredictionText(frame, gesture_name, confidence)
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            if isWink(face_landmarks):
                drawBoundingBoxWithLabel(frame, face_landmarks, "Wink Detected", color=(0, 255, 0))
    cv2.imshow("anythingatall.mp3", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()