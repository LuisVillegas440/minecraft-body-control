import math

from config import FINGER_LANDMARKS
from config import PINCH_DISTANCE_RATIO


def calculate_angle(a, b, c):
    ba = (
        a.x - b.x,
        a.y - b.y,
        a.z - b.z,
    )

    bc = (
        c.x - b.x,
        c.y - b.y,
        c.z - b.z,
    )

    dot_product = (
        ba[0] * bc[0]
        + ba[1] * bc[1]
        + ba[2] * bc[2]
    )

    magnitude_ba = math.sqrt(
        ba[0] ** 2
        + ba[1] ** 2
        + ba[2] ** 2
    )

    magnitude_bc = math.sqrt(
        bc[0] ** 2
        + bc[1] ** 2
        + bc[2] ** 2
    )

    if magnitude_ba == 0 or magnitude_bc == 0:
        return 0

    cosine = dot_product / (
        magnitude_ba * magnitude_bc
    )
    cosine = max(
        -1.0,
        min(1.0, cosine),
    )

    return math.degrees(
        math.acos(cosine)
    )


def classify_hand(landmarks):
    fingers = [
        (5, 6, 7, 8),
        (9, 10, 11, 12),
        (13, 14, 15, 16),
        (17, 18, 19, 20),
    ]

    extended_fingers = 0
    curled_fingers = 0

    for mcp, pip, dip, tip in fingers:
        pip_angle = calculate_angle(
            landmarks[mcp],
            landmarks[pip],
            landmarks[dip],
        )
        dip_angle = calculate_angle(
            landmarks[pip],
            landmarks[dip],
            landmarks[tip],
        )

        if (
            pip_angle > 150
            and dip_angle > 145
        ):
            extended_fingers += 1

        if pip_angle < 125:
            curled_fingers += 1

    if extended_fingers >= 4:
        return "OPEN"

    if (
        curled_fingers >= 3
        and extended_fingers == 0
    ):
        return "FIST"

    return "OTHER"


def get_finger_states(
    landmarks,
    extended_min_angle=150,
    curled_max_angle=125,
):
    extended = {}
    curled = {}

    for name, (mcp, pip, dip, tip) in FINGER_LANDMARKS.items():
        pip_angle = calculate_angle(
            landmarks[mcp],
            landmarks[pip],
            landmarks[dip],
        )
        dip_angle = calculate_angle(
            landmarks[pip],
            landmarks[dip],
            landmarks[tip],
        )

        extended[name] = (
            pip_angle > extended_min_angle
            and dip_angle > extended_min_angle - 5
        )
        curled[name] = pip_angle < curled_max_angle

    return extended, curled


def landmark_distance(a, b):
    return math.sqrt(
        (a.x - b.x) ** 2
        + (a.y - b.y) ** 2
        + (a.z - b.z) ** 2
    )


def get_hand_scale(landmarks):
    return max(
        landmark_distance(
            landmarks[0],
            landmarks[9],
        ),
        0.001,
    )


def is_pinching(landmarks):
    pinch_distance = landmark_distance(
        landmarks[4],
        landmarks[8],
    )

    return (
        pinch_distance
        < get_hand_scale(landmarks) * PINCH_DISTANCE_RATIO
    )
