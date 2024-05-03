import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import numpy as np
import cv2
import mediapipe as mp
from PIL import ImageGrab
import argparse
from collections import deque
from src.model import CLASS_IDS
from src.utils import get_overlay, get_images
# import tensorflow as tf
from src.utils import CLASS_IDS
from src.model import create_model 

HAND_GESTURES = ["Open", "Closed"]
WHITE_RGB = (255, 255, 255)
GREEN_RGB = (0, 255, 0)
PURPLE_RGB = (255, 0, 127)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)


def draw_color_palette(frame):
    colors = [
        ("CLEAR", (255, 255, 255)),
        ("RED", (0, 0, 255)),
        ("YELLOW", (0, 255, 255)),
        ("GREEN", (0, 255, 0)),
        ("BLUE", (255, 0, 0)),
        ("ERASER", (0, 0, 0))
    ]
    button_width = 74
    button_height = 40
    start_x = 10
    start_y = 10
    for i, (text, color) in enumerate(colors):
        x = start_x + i * (button_width + 10)  # 10 pixels space between buttons
        y = start_y
        cv2.rectangle(frame, (x, y), (x + button_width, y + button_height), color, -1)
        cv2.putText(frame, text, (x + 5, y + button_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255) if color != (255, 255, 255) else (0, 0, 0), 1)

    return colors, start_x, start_y, button_width, button_height


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Google's Quick Draw Project (https://quickdraw.withgoogle.com/#)""")
    parser.add_argument("-a", "--area", type=int, default=3000, help="Minimum area of captured object")
    parser.add_argument("-l", "--load_path", type=str, default="data/trained_models")
    parser.add_argument("-s", "--save_video", type=str, default="data/output.mp4")
    args = parser.parse_args()
    return args


def load_graph(path):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    InteractiveSession(config=config)
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(path, 'rb') as fid:
            graph_def.ParseFromString(fid.read())
            tf.import_graph_def(graph_def, name='')
        sess = tf.compat.v1.Session(graph=detection_graph)
    return detection_graph, sess


def detect_hands(image, graph, sess):
    input_image = graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = graph.get_tensor_by_name('detection_scores:0')
    detection_classes = graph.get_tensor_by_name('detection_classes:0')
    image = image[None, :, :, :]
    boxes, scores, classes = sess.run([detection_boxes, detection_scores, detection_classes],
                                      feed_dict={input_image: image})
    return np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes)


def predict(boxes, scores, classes, threshold, width, height, num_hands=2):
    count = 0
    results = {}
    for box, score, class_ in zip(boxes[:num_hands], scores[:num_hands], classes[:num_hands]):
        if score > threshold:
            y_min = int(box[0] * height)
            x_min = int(box[1] * width)
            y_max = int(box[2] * height)
            x_max = int(box[3] * width)
            category = HAND_GESTURES[int(class_) - 1]
            results[count] = [x_min, x_max, y_min, y_max, category]
            count += 1
    return results

def write_fingertip_coordinates(x, y):
    with open("fingertip_coordinates.txt", "w") as f:
        f.write(f"{x},{y}")

import keras
def run(opt):
    global current_color
    current_color = WHITE_RGB  # Default drawing color
    graph, sess = load_graph("data/pretrained_model.pb")
    # model = create_model()

    model = tf.saved_model.load(opt.load_path)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    out = cv2.VideoWriter(opt.save_video, cv2.VideoWriter_fourcc(*"MJPG"), int(cap.get(cv2.CAP_PROP_FPS)),
                          (640, 480))
    points = deque(maxlen=1024)
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    is_drawing = False
    is_shown = False
    predicted_class = None
    class_images = get_images("images", CLASS_IDS.values())

    while True:
        key = cv2.waitKey(10)
        if key == ord("q"):
            break
        elif key == ord(" "):
            is_drawing = not is_drawing
            if is_drawing:
                if is_shown:
                    points = deque(maxlen=1024)
                    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
                is_shown = False

        if not is_drawing and not is_shown:
            if len(points):
                canvas_gs = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
                median = cv2.medianBlur(canvas_gs, 9)
                gaussian = cv2.GaussianBlur(median, (5, 5), 0)
                _, thresh = cv2.threshold(gaussian, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                contour_gs, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                if len(contour_gs):
                    contour = sorted(contour_gs, key=cv2.contourArea, reverse=True)[0]
                    if cv2.contourArea(contour) > opt.area:
                        x, y, w, h = cv2.boundingRect(contour)
                        image = canvas_gs[y:y + h, x:x + w]
                        image = cv2.resize(image, (28, 28))
                        image = np.array(image, dtype=np.float32)[None, :, :, None] / 255
                        image = tf.convert_to_tensor(image)
                        predictions = model(image)
                        score = tf.nn.softmax(predictions[0])
                        predicted_class = np.argmax(score)
                        is_shown = True
                    else:
                        print("The object drawn is too small. Please draw a bigger one!")
                        points = deque(maxlen=1024)
                        canvas = np.zeros((480, 640, 3), dtype=np.uint8)

        ret, frame = cap.read()
        if frame is None:
            continue
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        color_buttons, palette_start_x, palette_start_y, btn_width, btn_height = draw_color_palette(frame_bgr)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame_bgr,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style())
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x, y = int(index_finger_tip.x * frame.shape[1]), int(index_finger_tip.y * frame.shape[0])
                for i, (color_name, color_value) in enumerate(color_buttons):
                    btn_x = palette_start_x + i * (btn_width + 10)
                    btn_y = palette_start_y
                    if btn_x <= x <= btn_x + btn_width and btn_y <= y <= btn_y + btn_height:
                        current_color = color_value
                        if color_name == "CLEAR":
                            canvas = np.zeros((480, 640, 3), dtype=np.uint8)
                        break
                if is_drawing:
                    write_fingertip_coordinates(x,y)
                    points.appendleft((x, y))
                    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                for i in range(1, len(points)):
                    if points[i - 1] is None or points[i] is None:
                        continue
                    cv2.line(canvas, points[i - 1], points[i], current_color, 5)

        frame_bgr = cv2.addWeighted(frame_bgr, 1, canvas, 0.5, 0)
        if is_shown:
            cv2.putText(frame_bgr, 'You are drawing', (100, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, PURPLE_RGB, 5,
                        cv2.LINE_AA)
            image_start_x = 400
            image_start_y = 420
            image_end_x = image_start_x + 60
            image_end_y = image_start_y + 60
            frame_bgr[image_start_y:image_end_y, image_start_x:image_end_x] = get_overlay(frame_bgr[image_start_y:image_end_y, image_start_x:image_end_x], class_images[predicted_class], (60, 60))
        cv2.imshow("Camera", frame_bgr)
        out.write(frame_bgr)
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    opt = get_args()
    run(opt)
