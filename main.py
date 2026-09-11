import time
import ctypes
import threading
import atexit
from collections import defaultdict, deque, Counter

import cv2
import mediapipe as mp
import numpy as np
from pynput.keyboard import Controller as KeyboardController
from pynput.keyboard import Key
from pynput.keyboard import Listener as KeyboardListener
from pynput.mouse import Controller as MouseController

from controls.actions import ActionController
from controls.movement import MovementController
from config import *
from gestures.actions import RightHandActionClassifier
from gestures.common import classify_hand as classify_basic_hand
from gestures.hands import get_physical_hand_name as resolve_physical_hand_name
from gestures.hands import is_action_hand as is_physical_action_hand
from gestures.hands import is_control_hand as is_physical_control_hand
from gestures.movement import classify_movement_state as classify_left_hand_movement
from gestures.movement import describe_movement_state as describe_left_hand_movement


# ==============================
# MEDIAPIPE
# ==============================

BaseOptions = mp.tasks.BaseOptions

PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

RunningMode = mp.tasks.vision.RunningMode


# ==============================
# CONFIG POSE
# ==============================

pose_options = PoseLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=POSE_MODEL_PATH
    ),

    running_mode=RunningMode.VIDEO,

    num_poses=1,

    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)


# ==============================
# CONFIG MANOS
# ==============================

hand_options = HandLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=HAND_MODEL_PATH
    ),

    running_mode=RunningMode.VIDEO,

    # Queremos detectar las dos manos
    num_hands=2,

    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)


# ==============================
# CÁMARA
# ==============================

face_options = FaceLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=FACE_MODEL_PATH
    ),

    running_mode=RunningMode.VIDEO,

    num_faces=1,

    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,

    output_facial_transformation_matrixes=True,
)


cap = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)


if not cap.isOpened():

    raise RuntimeError(
        "No se pudo abrir la webcam"
    )


start_time = time.perf_counter()


# ==============================
# CREAR MODELOS
# ==============================

gesture_history = defaultdict(
    lambda: deque(maxlen=5)
)

keyboard = KeyboardController()
mouse = MouseController()
movement_controller = MovementController(
    keyboard
)
action_classifier = RightHandActionClassifier()
action_controller = ActionController(
    keyboard,
    mouse
)

head_state_lock = threading.Lock()
head_control_enabled = HEAD_CONTROL_STARTS_ON
head_recenter_requested = HEAD_CONTROL_STARTS_ON
head_status_message = ""
head_status_message_until = 0.0
debug_overlay_enabled = False
neutral_yaw = None
neutral_pitch = None
smoothed_relative_yaw = 0.0
smoothed_relative_pitch = 0.0
head_yaw_moving = False
head_pitch_moving = False


def cleanup_controls():

    action_controller.cleanup()
    movement_controller.release_movement_keys()


atexit.register(
    cleanup_controls
)


def clamp(value, minimum, maximum):

    return max(
        minimum,
        min(
            maximum,
            value
        )
    )


def get_default_head_tracking():

    with head_state_lock:
        control_enabled = head_control_enabled

    return {
        "head_status": "HEAD: NOT DETECTED",
        "message": get_head_status_message(),
        "control_status": (
            "HEAD CONTROL: ON"
            if control_enabled
            else "HEAD CONTROL: OFF"
        ),
        "yaw": None,
        "pitch": None,
        "relative_yaw": None,
        "relative_pitch": None,
        "camera": "CENTER",
        "mouse_dx": 0,
        "mouse_dy": 0,
        "face_landmarks": None,
    }


def get_head_direction(offset_x, offset_y):

    horizontal = "CENTER"
    vertical = "CENTER"

    if offset_x < 0:
        horizontal = "LEFT"
    elif offset_x > 0:
        horizontal = "RIGHT"

    if offset_y < 0:
        vertical = "UP"
    elif offset_y > 0:
        vertical = "DOWN"

    if (
        horizontal == "CENTER"
        and vertical == "CENTER"
    ):
        return "CENTER"

    if horizontal == "CENTER":
        return vertical

    if vertical == "CENTER":
        return horizontal

    return f"{vertical}_{horizontal}"


def set_head_status_message(message, duration=1.2):

    global head_status_message
    global head_status_message_until

    with head_state_lock:
        head_status_message = message
        head_status_message_until = (
            time.perf_counter()
            + duration
        )


def get_head_status_message():

    with head_state_lock:
        if time.perf_counter() > head_status_message_until:
            return ""

        return head_status_message


def toggle_head_control():

    global head_control_enabled
    global head_recenter_requested

    with head_state_lock:
        head_control_enabled = not head_control_enabled
        enabled = head_control_enabled
        if enabled:
            head_recenter_requested = True

    set_head_status_message(
        "HEAD CONTROL ON - CENTER HEAD"
        if enabled
        else "HEAD CONTROL OFF"
    )


def request_recenter_head():

    global head_recenter_requested

    with head_state_lock:
        head_recenter_requested = True


def toggle_debug_overlay():

    global debug_overlay_enabled

    debug_overlay_enabled = not debug_overlay_enabled


def on_global_key_press(key):

    if key == Key.f8:
        toggle_head_control()

    if key == Key.f9:
        request_recenter_head()

    if key == Key.f12:
        toggle_debug_overlay()


def matrix_to_head_pose(matrix):

    rotation = np.array(
        matrix[:3, :3],
        dtype=np.float64
    )

    u, _, vt = np.linalg.svd(
        rotation
    )

    rotation = u @ vt

    if np.linalg.det(rotation) < 0:
        rotation *= -1

    angles = cv2.RQDecomp3x3(
        rotation
    )[0]

    pitch = float(
        angles[0]
    )

    yaw = float(
        angles[1]
    )

    roll = float(
        angles[2]
    )

    if INVERT_HEAD_YAW:
        yaw = -yaw

    if INVERT_HEAD_PITCH:
        pitch = -pitch

    return {
        "yaw": yaw,
        "pitch": pitch,
        "roll": roll,
    }


def get_head_pose(face_result):

    if (
        not face_result.face_landmarks
        or not face_result.facial_transformation_matrixes
    ):
        return None

    head_pose = matrix_to_head_pose(
        face_result.facial_transformation_matrixes[0]
    )

    head_pose["face_landmarks"] = (
        face_result.face_landmarks[0]
    )

    return head_pose


def recenter_head(yaw, pitch):

    global neutral_yaw
    global neutral_pitch
    global smoothed_relative_yaw
    global smoothed_relative_pitch
    global head_yaw_moving
    global head_pitch_moving

    neutral_yaw = yaw
    neutral_pitch = pitch
    smoothed_relative_yaw = 0.0
    smoothed_relative_pitch = 0.0
    head_yaw_moving = False
    head_pitch_moving = False

    set_head_status_message(
        "HEAD RECENTERED"
    )


def update_hysteresis(value, is_moving, enter_degrees, stop_degrees):

    magnitude = abs(
        value
    )

    if is_moving:
        if magnitude < stop_degrees:
            return False

        return True

    if magnitude > enter_degrees:
        return True

    return False


def calculate_axis_mouse(value, deadzone, sensitivity, maximum):

    magnitude = abs(
        value
    )

    excess = max(
        0.0,
        magnitude - deadzone
    )

    movement = int(
        round(
            clamp(
                excess * sensitivity,
                0,
                maximum
            )
        )
    )

    if value < 0:
        return -movement

    return movement


def calculate_camera_movement(relative_yaw, relative_pitch):

    global smoothed_relative_yaw
    global smoothed_relative_pitch
    global head_yaw_moving
    global head_pitch_moving

    relative_yaw = clamp(
        relative_yaw,
        -HEAD_MAX_RELATIVE_YAW,
        HEAD_MAX_RELATIVE_YAW
    )

    relative_pitch = clamp(
        relative_pitch,
        -HEAD_MAX_RELATIVE_PITCH,
        HEAD_MAX_RELATIVE_PITCH
    )

    smoothed_relative_yaw = (
        HEAD_SMOOTHING_ALPHA
        * relative_yaw
        + (1 - HEAD_SMOOTHING_ALPHA)
        * smoothed_relative_yaw
    )

    smoothed_relative_pitch = (
        HEAD_SMOOTHING_ALPHA
        * relative_pitch
        + (1 - HEAD_SMOOTHING_ALPHA)
        * smoothed_relative_pitch
    )

    head_yaw_moving = update_hysteresis(
        smoothed_relative_yaw,
        head_yaw_moving,
        HEAD_YAW_DEADZONE_DEGREES,
        HEAD_YAW_STOP_DEGREES
    )

    head_pitch_moving = update_hysteresis(
        smoothed_relative_pitch,
        head_pitch_moving,
        HEAD_PITCH_DEADZONE_DEGREES,
        HEAD_PITCH_STOP_DEGREES
    )

    mouse_dx = 0
    mouse_dy = 0

    if head_yaw_moving:
        mouse_dx = calculate_axis_mouse(
            smoothed_relative_yaw,
            HEAD_YAW_DEADZONE_DEGREES,
            HEAD_YAW_SENSITIVITY,
            MAX_MOUSE_X
        )

    if head_pitch_moving:
        mouse_dy = calculate_axis_mouse(
            smoothed_relative_pitch,
            HEAD_PITCH_DEADZONE_DEGREES,
            HEAD_PITCH_SENSITIVITY,
            MAX_MOUSE_Y
        )

    return {
        "relative_yaw": relative_yaw,
        "relative_pitch": relative_pitch,
        "smoothed_yaw": smoothed_relative_yaw,
        "smoothed_pitch": smoothed_relative_pitch,
        "mouse_dx": mouse_dx,
        "mouse_dy": mouse_dy,
        "camera": get_head_direction(
            mouse_dx,
            mouse_dy
        ),
    }


def move_game_mouse(dx, dy):

    if (
        dx == 0
        and dy == 0
    ):
        return

    if USE_WINDOWS_GAME_MOUSE:
        ctypes.windll.user32.mouse_event(
            MOUSEEVENTF_MOVE,
            dx,
            dy,
            0,
            0
        )
        return

    mouse.move(
        dx,
        dy
    )


def apply_camera_movement(dx, dy):

    with head_state_lock:
        control_enabled = head_control_enabled

    if not control_enabled:
        return

    move_game_mouse(
        dx,
        dy
    )


def update_head_tracking(face_result):

    global head_recenter_requested
    global smoothed_relative_yaw
    global smoothed_relative_pitch
    global head_yaw_moving
    global head_pitch_moving

    tracking = get_default_head_tracking()

    head_pose = get_head_pose(
        face_result
    )

    if head_pose is None:
        smoothed_relative_yaw = 0.0
        smoothed_relative_pitch = 0.0
        head_yaw_moving = False
        head_pitch_moving = False
        show_not_detected_message = False

        with head_state_lock:
            if head_recenter_requested:
                show_not_detected_message = True

        if show_not_detected_message:
            set_head_status_message(
                "HEAD NOT DETECTED - WAITING"
            )

        tracking["message"] = get_head_status_message()
        return tracking

    yaw = head_pose["yaw"]
    pitch = head_pose["pitch"]

    tracking["head_status"] = "HEAD: DETECTED"
    tracking["yaw"] = yaw
    tracking["pitch"] = pitch
    tracking["face_landmarks"] = head_pose["face_landmarks"]

    with head_state_lock:
        should_recenter = head_recenter_requested
        head_recenter_requested = False

    if should_recenter:
        recenter_head(
            yaw,
            pitch
        )

    if (
        neutral_yaw is None
        or neutral_pitch is None
    ):
        tracking["head_status"] = "HEAD: NOT CALIBRATED"
        tracking["message"] = get_head_status_message()
        return tracking

    relative_yaw = (
        yaw
        - neutral_yaw
    )

    relative_pitch = (
        pitch
        - neutral_pitch
    )

    movement = calculate_camera_movement(
        relative_yaw,
        relative_pitch
    )

    apply_camera_movement(
        movement["mouse_dx"],
        movement["mouse_dy"]
    )

    tracking.update(
        movement
    )

    tracking["message"] = get_head_status_message()

    return tracking


def format_angle(value):

    if value is None:
        return "--"

    return f"{value:.1f}"


def format_ratio(value):

    if value is None:
        return "--"

    return f"{value:.2f}"


def format_pressed_state(pressed):

    return (
        "DOWN"
        if pressed
        else "UP"
    )


def draw_head_pose_vector(frame, tracking, width, height):

    face_landmarks = tracking["face_landmarks"]

    if not face_landmarks:
        return

    face_center_x = sum(
        landmark.x
        for landmark in face_landmarks
    ) / len(face_landmarks)

    face_center_y = sum(
        landmark.y
        for landmark in face_landmarks
    ) / len(face_landmarks)

    start_x = int(
        face_center_x
        * width
    )

    start_y = int(
        face_center_y
        * height
    )

    relative_yaw = tracking["relative_yaw"]
    relative_pitch = tracking["relative_pitch"]

    if (
        relative_yaw is None
        or relative_pitch is None
    ):
        relative_yaw = 0.0
        relative_pitch = 0.0

    end_x = int(
        start_x
        + clamp(
            relative_yaw * 4,
            -120,
            120
        )
    )

    end_y = int(
        start_y
        + clamp(
            relative_pitch * 4,
            -100,
            100
        )
    )

    cv2.circle(
        frame,
        (
            start_x,
            start_y
        ),
        5,
        (255, 120, 0),
        -1
    )

    cv2.arrowedLine(
        frame,
        (
            start_x,
            start_y
        ),
        (
            end_x,
            end_y
        ),
        (255, 120, 0),
        3,
        tipLength=0.25
    )


hotkey_listener = KeyboardListener(
    on_press=on_global_key_press
)

hotkey_listener.start()


with PoseLandmarker.create_from_options(
    pose_options
) as pose_landmarker, HandLandmarker.create_from_options(
    hand_options
) as hand_landmarker, FaceLandmarker.create_from_options(
    face_options
) as face_landmarker:


    while True:

        ok, frame = cap.read()

        if not ok:
            break


        # Webcam modo espejo
        frame = cv2.flip(
            frame,
            1
        )


        # BGR → RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )


        timestamp_ms = int(
            (
                time.perf_counter()
                - start_time
            )
            * 1000
        )


        # ==============================
        # DETECTAR POSE
        # ==============================

        pose_result = (
            pose_landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )
        )


        # ==============================
        # DETECTAR MANOS
        # ==============================

        hand_result = (
            hand_landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )
        )


        # ==============================
        # DETECTAR ROSTRO
        # ==============================

        face_result = (
            face_landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )
        )


        height, width, _ = frame.shape

        head_tracking = update_head_tracking(
            face_result
        )

        draw_head_pose_vector(
            frame,
            head_tracking,
            width,
            height
        )


        # ====================================
        # DIBUJAR TREN SUPERIOR
        # ====================================

        if pose_result.pose_landmarks:

            landmarks = (
                pose_result.pose_landmarks[0]
            )


            for start_idx, end_idx in (
                UPPER_BODY_CONNECTIONS
            ):

                start = landmarks[start_idx]
                end = landmarks[end_idx]


                if start.visibility < 0.5:
                    continue

                if end.visibility < 0.5:
                    continue


                x1 = int(
                    start.x * width
                )

                y1 = int(
                    start.y * height
                )


                x2 = int(
                    end.x * width
                )

                y2 = int(
                    end.y * height
                )


                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    3
                )


            for index in (
                UPPER_BODY_LANDMARKS
            ):

                landmark = landmarks[index]


                if landmark.visibility < 0.5:
                    continue


                x = int(
                    landmark.x * width
                )

                y = int(
                    landmark.y * height
                )


                cv2.circle(
                    frame,
                    (x, y),
                    7,
                    (0, 0, 255),
                    -1
                )


                if debug_overlay_enabled:
                    cv2.putText(
                        frame,
                        str(index),
                        (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 255, 255),
                        1
                    )


        # ====================================
        # DIBUJAR MANOS
        # ====================================

        control_hand_detected = False
        raw_movement_state = "NEUTRAL"
        movement_gesture = "NONE"
        movement_direction = "NONE"
        movement_status = "NEUTRAL"
        left_hand_status = "LEFT HAND: NOT DETECTED"
        right_hand_detected = False
        right_hand_status = "RIGHT HAND: NOT DETECTED"
        right_gesture = "NONE"
        raw_right_action = "NEUTRAL"
        right_action = "NEUTRAL"
        right_pinch_ratio = None
        
        if hand_result.hand_landmarks:

            for hand_index, hand_landmarks in enumerate(
                hand_result.hand_landmarks
            ):


                # --------------------------
                # Líneas de la mano
                # --------------------------

                for start_idx, end_idx in (
                    HAND_CONNECTIONS
                ):

                    start = (
                        hand_landmarks[
                            start_idx
                        ]
                    )

                    end = (
                        hand_landmarks[
                            end_idx
                        ]
                    )


                    x1 = int(
                        start.x * width
                    )

                    y1 = int(
                        start.y * height
                    )


                    x2 = int(
                        end.x * width
                    )

                    y2 = int(
                        end.y * height
                    )


                    cv2.line(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (255, 255, 0),
                        2
                    )


                # --------------------------
                # 21 landmarks
                # --------------------------

                for index, landmark in enumerate(
                    hand_landmarks
                ):

                    x = int(
                        landmark.x * width
                    )

                    y = int(
                        landmark.y * height
                    )


                    cv2.circle(
                        frame,
                        (x, y),
                        5,
                        (255, 0, 255),
                        -1
                    )


                    if debug_overlay_enabled:
                        cv2.putText(
                            frame,
                            str(index),
                            (x + 5, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.35,
                            (255, 255, 255),
                            1
                        )


                # --------------------------
                # LEFT / RIGHT
                # --------------------------

                if (
                    hand_index
                    < len(
                        hand_result.handedness
                    )
                ):

                    category = (
                        hand_result
                        .handedness[
                            hand_index
                        ][0]
                    )


                    hand_name = (
                        category.category_name
                    )

                    physical_hand_name = resolve_physical_hand_name(
                        hand_name
                    )
                    
                    raw_gesture = classify_basic_hand(
                        hand_landmarks
                    )

                    if is_physical_action_hand(
                        physical_hand_name
                    ):

                        right_hand_detected = True
                        right_hand_status = "RIGHT HAND: DETECTED"

                        (
                            right_gesture,
                            raw_right_action,
                            right_pinch_ratio
                        ) = action_classifier.classify(
                            hand_landmarks
                        )

                        raw_gesture = right_gesture

                    if is_physical_control_hand(
                        physical_hand_name
                    ):

                        control_hand_detected = True
                        left_hand_status = "LEFT HAND: DETECTED"

                        (
                            raw_movement_state,
                            movement_gesture,
                            movement_direction
                        ) = classify_left_hand_movement(
                            hand_landmarks
                        )

                        raw_gesture = (
                            movement_gesture
                        )
                    
                    gesture_history[
                        physical_hand_name
                    ].append(
                        raw_gesture
                    )
                    
                    # Obtener el gesto que más se repite
                    # entre los últimos frames
                    
                    gesture = Counter(
                        gesture_history[
                            physical_hand_name
                        ]
                    ).most_common(1)[0][0]
                    
                    hand_score = (
                        category.score
                    )
                    

                    wrist = (
                        hand_landmarks[0]
                    )


                    wrist_x = int(
                        wrist.x * width
                    )

                    wrist_y = int(
                        wrist.y * height
                    )


                    if debug_overlay_enabled:
                        cv2.putText(
                            frame,
                            f"{physical_hand_name} "
                            f"{hand_score:.2f}",
                            (
                                wrist_x,
                                wrist_y + 30
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 0),
                            2
                        )
                    
                    # --------------------------
                    # MOSTRAR GESTO
                    # --------------------------

                    if debug_overlay_enabled:
                        cv2.putText(
                            frame,
                            gesture,
                            (
                                wrist_x,
                                wrist_y + 60
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 255),
                            2
                        )

            if right_hand_detected:

                stable_action_state = action_controller.stabilize_action(
                    right_gesture,
                    raw_right_action
                )

                right_action = action_controller.apply_action_state(
                    stable_action_state
                )

            else:

                action_classifier.reset()
                action_controller.reset_tracking()
                right_action = "NEUTRAL"

            if control_hand_detected:

                stable_movement_state = movement_controller.get_stable_movement_state(
                    raw_movement_state
                )

                movement_status = movement_controller.apply_movement_state(
                    stable_movement_state
                )

                if stable_movement_state != "NEUTRAL":
                    (
                        movement_gesture,
                        movement_direction
                    ) = describe_left_hand_movement(
                        stable_movement_state
                    )

            else:
                movement_controller.clear_history()
                movement_controller.release_movement_keys()

        else:

            movement_controller.clear_history()
            movement_controller.release_movement_keys()
            action_classifier.reset()
            action_controller.reset_tracking()
            movement_status = "NEUTRAL"
            movement_gesture = "NONE"
            movement_direction = "NONE"



        # ====================================
        # INFORMACIÓN
        # ====================================

        hands_detected = len(
            hand_result.hand_landmarks
        )


        if debug_overlay_enabled:
            overlay_rows = [
                (
                    f"HANDS: {hands_detected}",
                    (30, 50),
                    1,
                    (0, 255, 0)
                ),
                (
                    f"MOVE: {movement_status}",
                    (30, 90),
                    0.9,
                    (0, 255, 255)
                ),
                (
                    "CONTROL HAND: LEFT",
                    (30, 125),
                    0.7,
                    (255, 255, 0)
                ),
                (
                    left_hand_status,
                    (30, 160),
                    0.7,
                    (255, 255, 0)
                ),
                (
                    f"GESTURE: {movement_gesture}",
                    (30, 195),
                    0.7,
                    (255, 255, 255)
                ),
                (
                    f"DIRECTION: {movement_direction}",
                    (30, 230),
                    0.7,
                    (255, 255, 255)
                ),
                (
                    right_hand_status,
                    (30, 265),
                    0.7,
                    (0, 180, 255)
                ),
                (
                    f"RIGHT GESTURE: {right_gesture}",
                    (30, 300),
                    0.7,
                    (0, 180, 255)
                ),
                (
                    f"ACTION: {right_action}",
                    (30, 335),
                    0.7,
                    (0, 180, 255)
                ),
                (
                    f"PINCH RATIO: {format_ratio(right_pinch_ratio)}",
                    (30, 370),
                    0.7,
                    (0, 180, 255)
                ),
                (
                    f"BUTTONS: LEFT {format_pressed_state(action_controller.attack_pressed)} "
                    f"| RIGHT {format_pressed_state(action_controller.use_pressed)}",
                    (30, 405),
                    0.7,
                    (0, 180, 255)
                ),
                (
                    head_tracking["head_status"],
                    (30, 440),
                    0.7,
                    (255, 120, 0)
                ),
                (
                    f"YAW: {format_angle(head_tracking['yaw'])} "
                    f"PITCH: {format_angle(head_tracking['pitch'])}",
                    (30, 475),
                    0.7,
                    (255, 120, 0)
                ),
                (
                    f"REL YAW: {format_angle(head_tracking['relative_yaw'])} "
                    f"REL PITCH: {format_angle(head_tracking['relative_pitch'])}",
                    (30, 510),
                    0.7,
                    (255, 120, 0)
                ),
                (
                    head_tracking["control_status"],
                    (30, 545),
                    0.7,
                    (255, 120, 0)
                ),
                (
                    f"CAMERA: {head_tracking['camera']}",
                    (30, 580),
                    0.7,
                    (255, 120, 0)
                ),
                (
                    "F8: POV ON/OFF | F9: RECENTER | F12: DEBUG | Q: QUIT",
                    (30, 615),
                    0.58,
                    (200, 200, 200)
                ),
                (
                    f"MOUSE: {head_tracking['mouse_dx']}, "
                    f"{head_tracking['mouse_dy']}",
                    (30, 685),
                    0.62,
                    (200, 200, 200)
                )
            ]

            if head_tracking["message"]:
                overlay_rows.insert(
                    -1,
                    (
                        head_tracking["message"],
                        (30, 650),
                        0.72,
                        (0, 255, 255)
                    )
                )

        else:
            overlay_rows = [
                (
                    f"HANDS: {hands_detected}",
                    (30, 50),
                    1,
                    (0, 255, 0)
                ),
                (
                    f"MOVE: {movement_status}",
                    (30, 90),
                    0.9,
                    (0, 255, 255)
                ),
                (
                    "CONTROL HAND: LEFT",
                    (30, 125),
                    0.7,
                    (255, 255, 0)
                ),
                (
                    f"ACTION: {right_action}",
                    (30, 160),
                    0.7,
                    (0, 180, 255)
                ),
                (
                    head_tracking["control_status"],
                    (30, 195),
                    0.7,
                    (255, 120, 0)
                ),
                (
                    f"CAMERA: {head_tracking['camera']}",
                    (30, 230),
                    0.7,
                    (255, 120, 0)
                ),
                (
                    "F8: POV ON/OFF | F9: RECENTER | F12: DEBUG | Q: QUIT",
                    (30, 265),
                    0.58,
                    (200, 200, 200)
                )
            ]

            if head_tracking["message"]:
                overlay_rows.append(
                    (
                        head_tracking["message"],
                        (30, 300),
                        0.72,
                        (0, 255, 255)
                    )
                )

        for text, position, scale, color in overlay_rows:
            cv2.putText(
                frame,
                text,
                position,
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                color,
                2
            )
        
        
        cv2.imshow(
            "Minecraft Motion Controller",
            frame
        )

        key = cv2.waitKey(1) & 0xFF


         # ==================================
        # SALIR
        # ==================================

        if key == ord("q"):
            movement_controller.release_movement_keys()
            break


action_controller.cleanup()
movement_controller.release_movement_keys()
hotkey_listener.stop()

cap.release()
cv2.destroyAllWindows()
