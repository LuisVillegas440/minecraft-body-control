from collections import Counter
from collections import deque

from pynput.keyboard import Key

from config import MOVEMENT_STATES_TO_KEYS


class MovementController:
    def __init__(self, keyboard):
        self.keyboard = keyboard
        self.movement_history = deque(
            maxlen=5
        )
        self.key_states = {
            "w": False,
            "a": False,
            "s": False,
            "d": False,
            Key.ctrl_l: False,
        }

    def set_key(self, key, pressed):
        if self.key_states[key] == pressed:
            return

        self.key_states[key] = pressed

        if pressed:
            self.keyboard.press(key)
        else:
            self.keyboard.release(key)

    def release_movement_keys(self):
        for key in self.key_states:
            self.set_key(key, False)

    def clear_history(self):
        self.movement_history.clear()

    def format_movement_status(self, movement_state):
        if movement_state == "NEUTRAL":
            return "NEUTRAL"

        return movement_state.replace(
            "+",
            " + ",
        )

    def apply_movement_state(self, movement_state):
        movement_keys = MOVEMENT_STATES_TO_KEYS.get(
            movement_state,
            set(),
        )

        for key in self.key_states:
            if key not in movement_keys:
                self.set_key(
                    key,
                    False,
                )

        for key in self.key_states:
            self.set_key(
                key,
                key in movement_keys,
            )

        return self.format_movement_status(
            movement_state
        )

    def get_stable_movement_state(self, raw_state):
        if raw_state == "NEUTRAL":
            self.movement_history.clear()
            return "NEUTRAL"

        self.movement_history.append(
            raw_state
        )

        recent_states = list(
            self.movement_history
        )[-3:]

        if (
            len(recent_states) >= 3
            and len(set(recent_states)) == 1
        ):
            return raw_state

        return Counter(
            self.movement_history
        ).most_common(1)[0][0]
