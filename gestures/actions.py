from config import ACTION_FINGER_CURLED_MAX_ANGLE
from config import ACTION_FINGER_EXTENDED_MIN_ANGLE
from config import ACTION_PINCH_ENTER_THRESHOLD
from config import ACTION_PINCH_EXIT_THRESHOLD
from gestures.common import get_finger_states
from gestures.common import is_pinching
from gestures.common import landmark_distance


class RightHandActionClassifier:
    def __init__(self):
        self.action_pinching = False

    def reset(self):
        self.action_pinching = False

    def get_action_hand_finger_states(self, landmarks):
        return get_finger_states(
            landmarks,
            ACTION_FINGER_EXTENDED_MIN_ANGLE,
            ACTION_FINGER_CURLED_MAX_ANGLE,
        )

    def calculate_pinch_ratio(self, landmarks):
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]

        palm_size = max(
            landmark_distance(
                landmarks[0],
                landmarks[9],
            ),
            landmark_distance(
                landmarks[5],
                landmarks[17],
            ),
            0.001,
        )

        return (
            landmark_distance(
                thumb_tip,
                index_tip,
            )
            / palm_size
        )

    def classify(self, landmarks):
        extended, curled = self.get_action_hand_finger_states(
            landmarks
        )

        open_hand = (
            extended["index"]
            and extended["middle"]
            and extended["ring"]
            and extended["pinky"]
        )

        pinch_ratio = self.calculate_pinch_ratio(
            landmarks
        )

        if open_hand:
            self.action_pinching = False
            return "OPEN", "NEUTRAL", pinch_ratio

        index_only = (
            extended["index"]
            and curled["middle"]
            and curled["ring"]
            and curled["pinky"]
        )

        if index_only:
            self.action_pinching = False
            return "INDEX", "JUMP", pinch_ratio

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
            self.action_pinching = True
            return "OK", "USE", pinch_ratio

        fist = (
            curled["index"]
            and curled["middle"]
            and curled["ring"]
            and curled["pinky"]
        )

        if fist:
            self.action_pinching = False
            return "FIST", "ATTACK", pinch_ratio

        if self.action_pinching:
            is_pinch = (
                pinch_ratio
                < ACTION_PINCH_EXIT_THRESHOLD
            )
        else:
            is_pinch = (
                pinch_ratio
                < ACTION_PINCH_ENTER_THRESHOLD
            )

        if is_pinch:
            self.action_pinching = True
            return "PINCH", "USE", pinch_ratio

        self.action_pinching = False

        return "UNKNOWN", "NEUTRAL", pinch_ratio
