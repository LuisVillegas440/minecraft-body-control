import threading
import time
from collections import deque

from pynput.keyboard import Key
from pynput.mouse import Button

from config import ACTION_ENTER_FRAMES
from config import ACTION_HISTORY_LENGTH
from config import ACTION_UNKNOWN_EXIT_FRAMES
from config import ENABLE_ACTION_OUTPUT
from config import JUMP_COOLDOWN_SECONDS
from config import JUMP_HOLD_SECONDS


class ActionController:
    def __init__(self, keyboard, mouse):
        self.keyboard = keyboard
        self.mouse = mouse
        self.action_history = deque(
            maxlen=ACTION_HISTORY_LENGTH
        )
        self.attack_pressed = False
        self.use_pressed = False
        self.last_action_state = "NEUTRAL"
        self.action_unknown_frames = 0
        self.jump_armed = True
        self.last_jump_time = 0.0
        self.space_pressed = False

    def stabilize_action(self, raw_gesture, raw_action):
        if raw_gesture == "OPEN":
            self.action_history.clear()
            self.action_unknown_frames = 0
            self.last_action_state = "NEUTRAL"
            return "NEUTRAL"

        if raw_gesture == "UNKNOWN":
            self.action_unknown_frames += 1
            self.action_history.clear()

            if (
                self.action_unknown_frames
                >= ACTION_UNKNOWN_EXIT_FRAMES
            ):
                self.action_history.clear()
                self.last_action_state = "NEUTRAL"

            return self.last_action_state

        if raw_action == "JUMP":
            self.action_unknown_frames = 0
            self.action_history.clear()
            self.last_action_state = "JUMP"
            return "JUMP"

        self.action_unknown_frames = 0
        self.action_history.append(
            raw_action
        )

        recent_actions = list(
            self.action_history
        )[-ACTION_ENTER_FRAMES:]

        if (
            len(recent_actions) == ACTION_ENTER_FRAMES
            and all(
                action == raw_action
                for action in recent_actions
            )
        ):
            self.last_action_state = raw_action

        return self.last_action_state

    def release_action_buttons(self):
        if self.attack_pressed:
            if ENABLE_ACTION_OUTPUT:
                self.mouse.release(
                    Button.left
                )

            self.attack_pressed = False

        if self.use_pressed:
            if ENABLE_ACTION_OUTPUT:
                self.mouse.release(
                    Button.right
                )

            self.use_pressed = False

    def set_action_button(self, button, pressed):
        if button == Button.left:
            if self.attack_pressed == pressed:
                return

            self.attack_pressed = pressed

        if button == Button.right:
            if self.use_pressed == pressed:
                return

            self.use_pressed = pressed

        if not ENABLE_ACTION_OUTPUT:
            return

        if pressed:
            self.mouse.press(
                button
            )
        else:
            self.mouse.release(
                button
            )

    def trigger_jump(self):
        now = time.perf_counter()

        if (
            not self.jump_armed
            or now - self.last_jump_time < JUMP_COOLDOWN_SECONDS
        ):
            return

        self.last_jump_time = now
        self.jump_armed = False

        if ENABLE_ACTION_OUTPUT:
            if self.space_pressed:
                self.keyboard.release(
                    Key.space
                )

            self.keyboard.press(
                Key.space
            )
            self.space_pressed = True

            threading.Timer(
                JUMP_HOLD_SECONDS,
                self.release_jump_key,
            ).start()

    def release_jump_key(self):
        if not self.space_pressed:
            return

        if ENABLE_ACTION_OUTPUT:
            self.keyboard.release(
                Key.space
            )

        self.space_pressed = False

    def rearm_jump(self):
        self.jump_armed = True

    def apply_action_state(self, action_state):
        if action_state == "ATTACK":
            self.rearm_jump()
            self.set_action_button(
                Button.right,
                False,
            )
            self.set_action_button(
                Button.left,
                True,
            )
            return "ATTACK"

        if action_state == "USE":
            self.rearm_jump()
            self.set_action_button(
                Button.left,
                False,
            )
            self.set_action_button(
                Button.right,
                True,
            )
            return "USE"

        if action_state == "JUMP":
            self.release_action_buttons()
            self.trigger_jump()
            return "JUMP"

        self.rearm_jump()
        self.release_action_buttons()
        return "NEUTRAL"

    def reset_tracking(self):
        self.action_history.clear()
        self.last_action_state = "NEUTRAL"
        self.action_unknown_frames = 0
        self.jump_armed = True
        self.release_action_buttons()

    def cleanup(self):
        self.release_jump_key()
        self.release_action_buttons()
