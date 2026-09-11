from config import ACTION_HAND
from config import CONTROL_HAND
from config import MEDIAPIPE_LABELS_MATCH_PHYSICAL_HANDS


def get_physical_hand_name(hand_name):
    if MEDIAPIPE_LABELS_MATCH_PHYSICAL_HANDS:
        return hand_name

    if hand_name == "Left":
        return "Right"

    if hand_name == "Right":
        return "Left"

    return hand_name


def is_control_hand(physical_hand_name):
    return (
        CONTROL_HAND == "PHYSICAL_LEFT"
        and physical_hand_name == "Left"
    )


def is_action_hand(physical_hand_name):
    return (
        ACTION_HAND == "PHYSICAL_RIGHT"
        and physical_hand_name == "Right"
    )
