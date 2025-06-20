import msgParser
import carState
import carControl
from pynput import keyboard

class Driver(object):
    def __init__(self, stage):
        self.WARM_UP = 0
        self.QUALIFYING = 1
        self.RACE = 2
        self.UNKNOWN = 3
        self.stage = stage
        self.parser = msgParser.MsgParser()
        self.state = carState.CarState()
        self.control = carControl.CarControl()
        self.steer_lock = 0.785398
        self.steer_sensitivity = 0.5  # Lower for smoother steering
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()
        self.steer = 0.0
        self.accel = 0.0
        self.brake = 0.0
        self.gear = 1

    def init(self):
        self.angles = [0 for _ in range(19)]
        for i in range(5):
            self.angles[i] = -90 + i * 15
            self.angles[18 - i] = 90 - i * 15
        for i in range(5, 9):
            self.angles[i] = -20 + (i-5) * 5
            self.angles[18 - i] = 20 - (i-5) * 5
        return self.parser.stringify({'init': self.angles})

    def on_press(self, key):
        try:
            if key == keyboard.Key.left:
                self.steer = min(self.steer + 0.1, 1.0)  # Gradual left
            elif key == keyboard.Key.right:
                self.steer = max(self.steer - 0.1, -1.0)  # Gradual right
            elif key == keyboard.Key.up:
                self.accel = min(self.accel + 0.2, 1.0)  # Gradual accel
            elif key == keyboard.Key.space:
                self.brake = min(self.brake + 0.2, 1.0)  # Gradual brake
            elif hasattr(key, 'char') and key.char == 'q' and self.gear < 6:
                self.gear += 1
            elif hasattr(key, 'char') and key.char == 'e' and self.gear > -1:
                self.gear -= 1
        except AttributeError:
            pass

    def on_release(self, key):
        try:
            if key in (keyboard.Key.left, keyboard.Key.right):
                self.steer *= 0.5  # Decay steering
            elif key == keyboard.Key.up:
                self.accel *= 0.5  # Decay accel
            elif key == keyboard.Key.space:
                self.brake = 0.0  # Reset brake to 0
        except AttributeError:
            pass

    def drive(self, msg):
        self.state.setFromMsg(msg)
        # Auto-shift to gear 1 if stopped
        if float(self.state.getSpeedX()) < 1.0 and self.gear > 1:
            self.gear = 1
        self.control.setSteer(self.steer * self.steer_sensitivity / self.steer_lock)
        self.control.setAccel(self.accel)
        self.control.setBrake(self.brake)
        self.control.setGear(self.gear)
        # Print feedback
        print(f"Speed: {self.state.getSpeedX():.1f}, TrackPos: {self.state.getTrackPos():.2f}, "
              f"Steer: {self.steer:.2f}, Accel: {self.accel:.2f}, Brake: {self.brake:.2f}, Gear: {self.gear}")
        return self.control.toMsg()

    def onShutDown(self):
        self.listener.stop()

    def onRestart(self):
        pass