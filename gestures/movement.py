import math

from config import INDEX_DIAGONAL_RATIO
from config import INDEX_MIN_VECTOR_LENGTH
from config import INDEX_UP_LEFT_MAX_ANGLE
from config import INDEX_UP_LEFT_MIN_ANGLE
from config import INDEX_UP_MAX_ANGLE
from config import INDEX_UP_MIN_ANGLE
from config import INDEX_UP_RIGHT_MAX_ANGLE
from config import INDEX_UP_RIGHT_MIN_ANGLE
from config import INVERT_HORIZONTAL_MOVEMENT
from config import THUMB_AWAY_FROM_FINGERS_RATIO
from config import THUMB_AWAY_FROM_INDEX_RATIO
from config import THUMB_EXTENDED_DISTANCE_RATIO
from config import THUMB_EXTENDED_MIN_ANGLE
from gestures.common import calculate_angle
from gestures.common import classify_hand
from gestures.common import get_finger_states
from gestures.common import get_hand_scale
from gestures.common import is_pinching
from gestures.common import landmark_distance


def get_index_direction_sector(landmarks):
    index_mcp = landmarks[5]
    index_tip = landmarks[8]

    dx = index_tip.x - index_mcp.x
    dy = index_tip.y - index_mcp.y

    if INVERT_HORIZONTAL_MOVEMENT:
        dx = -dx

    if math.hypot(dx, dy) < INDEX_MIN_VECTOR_LENGTH:
        return None

    angle = math.degrees(
        math.atan2(
            -dy,
            dx,
        )
    )

    if angle < 0:
        angle += 360

    if dy >= 0:
        return None

    if (
        INDEX_UP_LEFT_MIN_ANGLE
        <= angle
        <= INDEX_UP_LEFT_MAX_ANGLE
        and abs(dx) > abs(dy) * INDEX_DIAGONAL_RATIO
    ):
        return "UP_LEFT"

    if (
        INDEX_UP_RIGHT_MIN_ANGLE
        <= angle
        <= INDEX_UP_RIGHT_MAX_ANGLE
        and abs(dx) > abs(dy) * INDEX_DIAGONAL_RATIO
    ):
        return "UP_RIGHT"

    if (
        INDEX_UP_MIN_ANGLE
        <= angle
        <= INDEX_UP_MAX_ANGLE
    ):
        return "UP"

    return None


def is_thumb_extended(landmarks):
    hand_scale = get_hand_scale(
        landmarks
    )

    thumb_mcp_angle = calculate_angle(
        landmarks[1],
        landmarks[2],
        landmarks[3],
    )
    thumb_ip_angle = calculate_angle(
        landmarks[2],
        landmarks[3],
        landmarks[4],
    )
    thumb_distance = landmark_distance(
        landmarks[0],
        landmarks[4],
    )
    thumb_index_gap = landmark_distance(
        landmarks[4],
        landmarks[5],
    )
    thumb_finger_gaps = [
        landmark_distance(
            landmarks[4],
            landmarks[index],
        )
        for index in (8, 12, 16, 20)
    ]

    return (
        thumb_mcp_angle > THUMB_EXTENDED_MIN_ANGLE
        and thumb_ip_angle > THUMB_EXTENDED_MIN_ANGLE
        and thumb_distance
        > hand_scale * THUMB_EXTENDED_DISTANCE_RATIO
        and thumb_index_gap
        > hand_scale * THUMB_AWAY_FROM_INDEX_RATIO
        and min(thumb_finger_gaps)
        > hand_scale * THUMB_AWAY_FROM_FINGERS_RATIO
    )


def classify_movement_state(landmarks):
    extended, curled = get_finger_states(
        landmarks
    )

    open_hand = (
        extended["index"]
        and extended["middle"]
        and extended["ring"]
        and extended["pinky"]
    )

    if open_hand:
        return "NEUTRAL", "OPEN", "NONE"

    rock_gesture = (
        extended["index"]
        and extended["pinky"]
        and curled["middle"]
        and curled["ring"]
    )

    if rock_gesture:
        return "SPRINT", "ROCK", "FORWARD"

    ok_gesture = (
        is_pinching(
            landmarks
        )
        and curled["index"]
        and extended["middle"]
        and extended["ring"]
        and extended["pinky"]
    )

    if ok_gesture:
        return "S", "OK", "NONE"

    index_only = (
        extended["index"]
        and curled["middle"]
        and curled["ring"]
        and curled["pinky"]
    )

    if index_only:
        direction = get_index_direction_sector(
            landmarks
        )

        if direction == "UP":
            return "W", "INDEX", "UP"

        if direction == "UP_LEFT":
            return "W+A", "INDEX", "UP_LEFT"

        if direction == "UP_RIGHT":
            return "W+D", "INDEX", "UP_RIGHT"

        return "NEUTRAL", "INDEX", "NONE"

    pinky_only = (
        extended["pinky"]
        and curled["index"]
        and curled["middle"]
        and curled["ring"]
    )

    if pinky_only:
        return "A", "PINKY", "LEFT"

    thumb_only = (
        is_thumb_extended(
            landmarks
        )
        and curled["index"]
        and curled["middle"]
        and curled["ring"]
        and curled["pinky"]
    )

    if thumb_only:
        return "D", "THUMB", "RIGHT"

    if classify_hand(
        landmarks
    ) == "FIST":
        return "NEUTRAL", "FIST", "NONE"

    return "NEUTRAL", "OTHER", "NONE"


def describe_movement_state(movement_state):
    if movement_state == "W":
        return "INDEX", "UP"

    if movement_state == "W+A":
        return "INDEX", "UP_LEFT"

    if movement_state == "W+D":
        return "INDEX", "UP_RIGHT"

    if movement_state == "A":
        return "PINKY", "LEFT"

    if movement_state == "D":
        return "THUMB", "RIGHT"

    if movement_state == "S":
        return "OK", "NONE"

    if movement_state == "SPRINT":
        return "ROCK", "FORWARD"

    return "NEUTRAL", "NONE"
