import msgParser
import carState
import carControl
from pynput import keyboard
import time
import os
import json
from improvedCarPredict import ImprovedCarPredict

class EnhancedDriver(object):
    def __init__(self, stage, track_type='none', car_type='toyota_corolla_wrc', model_dir='model', manual_override=False, record_data=False, model_type='xgboost', use_unified_model=False):
        """
        Enhanced AI driver for TORCS with model-based control and manual override
        
        Parameters:
        - stage: Race stage (0=warm-up, 1=qualifying, 2=race, 3=unknown)
        - track_type: Type of track ('oval', 'road', 'dirt')
        - car_type: Type of car
        - model_dir: Directory containing trained models
        - manual_override: Whether to enable manual control override
        - record_data: Whether to record driving data for training
        - model_type: Type of model to use ('xgboost', 'gradient_boosting', 'random_forest', 'decision_tree')
        - use_unified_model: Whether to use a unified model for all track types
        """

        print("Initializing Enhanced Driver...")
        print(f"Stage: {stage}, Track Type: {track_type}, Car Type: {car_type}")
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
        
        # Manual control variables
        self.manual_override = manual_override
        self.manual_steer = 0.0
        self.manual_accel = 0.0
        self.manual_brake = 0.0
        self.manual_gear = 1
        
        # Set up model-based control
        self.track_type = track_type.lower()  # Normalize to lowercase
        self.car_type = car_type
        
        # Force unified model if track_type is 'none'
        if self.track_type == 'none':
            self.use_unified_model = True
            print("Track type is 'none', forcing unified model usage")
        else:
            self.use_unified_model = use_unified_model
        
        # Initialize predictor
        self.predictor = ImprovedCarPredict()
        
        
        # Try to load models
        try:
            self.predictor.load_models(model_dir)
            print(f"Loaded {model_type} models from {model_dir}")
            
            # Verify the requested model type exists
            if self.use_unified_model:
                if 'unified' not in self.predictor.models:
                    print("Warning: Unified model not found. Using track-specific models.")
                    self.use_unified_model = False
            
            if not self.use_unified_model and self.track_type not in self.predictor.models:
                if 'unified' in self.predictor.models:
                    print("Falling back to unified model")
                    self.use_unified_model = True
                else:
                    available = list(self.predictor.models.keys())
                    if available:
                        print(f"Falling back to first available model: {available[0]}")
                        self.track_type = available[0]
                    
            self.model_control_enabled = True
        except Exception as e:
            self.model_control_enabled = False

        
        # Data recording
        self.record_data = record_data
        if self.record_data:
            self.data_file = open(f'torcs_data_{car_type}_{track_type}_{int(time.time())}.jsonl', 'w')
            print(f"Recording driving data to {self.data_file.name}")
        
        # Set up keyboard listener if manual override is enabled
        if self.manual_override:
            self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
            self.listener.start()
            print("Manual override enabled. Use arrow keys to control.")

    def init(self):
        """Initialize angles for track sensors"""
        self.angles = [0 for _ in range(19)]
        for i in range(5):
            self.angles[i] = -90 + i * 15
            self.angles[18 - i] = 90 - i * 15
        for i in range(5, 9):
            self.angles[i] = -20 + (i-5) * 5
            self.angles[18 - i] = 20 - (i-5) * 5
        return self.parser.stringify({'init': self.angles})

    def on_press(self, key):
        """Handle keyboard press events for manual control"""
        try:
            if key == keyboard.Key.left:
                self.manual_steer = min(self.manual_steer + 0.1, 1.0)  # Gradual left
            elif key == keyboard.Key.right:
                self.manual_steer = max(self.manual_steer - 0.1, -1.0)  # Gradual right
            elif key == keyboard.Key.up:
                self.manual_accel = min(self.manual_accel + 0.2, 1.0)  # Gradual accel
                self.manual_brake = 0.0  # Can't accelerate and brake at the same time
            elif key == keyboard.Key.down:
                self.manual_brake = min(self.manual_brake + 0.2, 1.0)  # Gradual brake
                self.manual_accel = 0.0  # Can't accelerate and brake at the same time
            elif key == keyboard.Key.space:
                self.manual_brake = min(self.manual_brake + 0.5, 1.0)  # Hard brake
                self.manual_accel = 0.0  # Can't accelerate and brake at the same time
            elif hasattr(key, 'char'):
                if key.char == 'q' and self.manual_gear < 6:
                    self.manual_gear += 1
                elif key.char == 'e' and self.manual_gear > -1:
                    self.manual_gear -= 1
                elif key.char == 'm':  # Toggle between manual and model control
                    self.manual_override = not self.manual_override
                    print(f"Manual override: {'ON' if self.manual_override else 'OFF'}")
        except AttributeError:
            pass

    def on_release(self, key):
        """Handle keyboard release events for manual control"""
        try:
            if key in (keyboard.Key.left, keyboard.Key.right):
                self.manual_steer *= 0.5  # Decay steering
            elif key == keyboard.Key.up:
                self.manual_accel *= 0.5  # Decay accel
            elif key == keyboard.Key.down or key == keyboard.Key.space:
                self.manual_brake = 0.0  # Reset brake
        except AttributeError:
            pass

    def trial(self):
        """
        Simple rule-based control when model is not available
        Uses track sensors to determine steering and acceleration
        """
        track = self.state.getTrack()
        # if track: 
        #     track = track[::-1]
        speed = self.state.getSpeedX()
        
        if not track:
            return
            
        # Basic collision avoidance
        mid_range_sensors = track[8:11]  # Center sensors
        
        # Find the farthest distance sensor
        max_distance = max(mid_range_sensors)
        max_index = mid_range_sensors.index(max_distance) + 8
        
        # Calculate desired angle towards max distance
        desired_angle = (max_index - 9) * 10  # Each sensor is roughly 10 degrees apart
        current_angle = self.state.getAngle() * 180 / 3.14159  # Convert to degrees
        
        # Calculate steering based on angle error
        angle_error = desired_angle - current_angle
        steer = angle_error / 45.0  # Normalize
        steer = max(-1.0, min(1.0, steer))  # Clamp
        
        # Speed control based on track visibility
        if max_distance > 100:
            target_speed = 200  # Fast on straight
        elif max_distance > 50:
            target_speed = 100  # Medium on gentle curves
        elif max_distance > 20:
            target_speed = 50  # Slow on tight curves
        else:
            target_speed = 20  # Very slow when close to walls
            
        # Apply acceleration or braking based on target speed
        if speed < target_speed - 10:
            accel = 1.0
            brake = 0.0
        elif speed > target_speed + 10:
            accel = 0.0
            brake = min(1.0, (speed - target_speed) / 10.0)  # Progressive braking
        else:
            accel = 0.5
            brake = 0.0
            
        # Automatic gear shifting based on RPM
        rpm = self.state.getRpm()
        gear = self.state.getGear()
        
        # if gear == 0:
        #     gear = 1
        # elif speed > 5:  # Only shift if moving
        #     if rpm > 7000 and gear < 6:
        #         gear += 1
        #     elif rpm < 2500 and gear > 1:
        #         gear -= 1
        # elif speed < 0.5:  # Only go to gear 1 when nearly stopped
        #     gear = 1

        # if speed > 10 and gear == 1:  # Shouldn't be in 1st at high speed
        #     gear = 2

        if rpm > 7000 and gear < 6:
                gear += 1
        elif rpm < 2500 and gear > 1:
            gear -= 1
            
        return steer, accel, brake, gear

    def drive(self, msg):
        """
        Main driving function that processes sensor data and returns control commands
        """
            
        if not hasattr(self, 'last_shift_time'):
            self.last_shift_time = 0
            self.shift_delay = 0.5  # seconds between shifts
            self.last_gear = 1
        
        # Update car state from message
        self.state.setFromMsg(msg)
        current_time = time.time()
        
        # Decide between manual, model-based, or rule-based control
        if self.manual_override:
            # Use manual control values
            steer = self.manual_steer
            accel = self.manual_accel
            brake = self.manual_brake
            gear = self.manual_gear
            control_type = "MANUAL"
        elif self.model_control_enabled:
            # Use ML model for prediction
            control_type = "MODEL" + ("-UNIFIED" if self.use_unified_model else "")
            try:
                # If using unified model, track_type will be handled internally by predict
                predictions = self.predictor.predict(self.state, self.track_type, self.car_type)
                steer = -(predictions.get('steer', 0.0))
                accel = predictions.get('accel', 0.0)
                brake = predictions.get('brake', 0.0)
                gear = int(predictions.get('gear', 1))
            except Exception as e:
                print(f"Model prediction error: {e}")
                steer, accel, brake, gear = self.trial()
                control_type = "RULE (fallback)"
        else:
            # Fall back to rule-based control
            steer, accel, brake, gear = self.trial()
            control_type = "RULE"
            
        # Auto-shift to gear 1 if stopped
        speed_x = self.state.getSpeedX()
        if speed_x < 1.0 and gear > 1:
            gear = 1

        if not self.manual_override:  # Only flip AI steering
            steer = -steer

        # Only allow gear changes if enough time has passed since last shift
        if current_time - self.last_shift_time > self.shift_delay:
            # Auto-shift to gear 1 if stopped (using averaged speed)
            if speed_x < 1.0 and gear > 1:
                gear = 1
        else:
            # Maintain current gear during shift delay
            gear = self.last_gear
            
        # Update shift time if gear changed
        if gear != self.last_gear:
            self.last_shift_time = current_time
            #Soften acceleration briefly after shift
            accel *= 0.7 if (current_time - self.last_shift_time < 0.3) else 1.0
        
        self.last_gear = gear  # Remember current gear for next cycle
        if gear == 0:
            gear = 1
        
        # Apply controls to the car
        self.control.setSteer(steer)
        self.control.setAccel(accel)
        self.control.setBrake(brake)
        self.control.setGear(gear)
        
        # Record data if enabled
        if self.record_data and self.data_file:
            state_dict = self.state.to_dict()
            action_dict = {
                'steer': steer,
                'accel': accel,
                'brake': brake,
                'gear': gear
            }
            data_entry = {'state': state_dict, 'action': action_dict}
            self.data_file.write(json.dumps(data_entry) + '\n')
            self.data_file.flush()  # Ensure data is written immediately
            
        # Print feedback
        print(f"[{control_type}] Speed: {speed_x:.1f}, TrackPos: {self.state.getTrackPos():.2f}, "
            f"Steer: {steer:.2f}, Accel: {accel:.2f}, Brake: {brake:.2f}, Gear: {gear}")
            
        return self.control.toMsg()
 
    def onShutDown(self):
        """Clean up resources on shutdown"""
        if hasattr(self, 'listener') and self.listener:
            self.listener.stop()
            
        if self.record_data and hasattr(self, 'data_file') and self.data_file:
            self.data_file.close()
            print(f"Closed data recording file: {self.data_file.name}")

    def onRestart(self):
        """Handle restart events"""
        if self.manual_override:
            # Reset manual control variables
            self.manual_steer = 0.0
            self.manual_accel = 0.0
            self.manual_brake = 0.0
            self.manual_gear = 1
            
        # Reset control values
        self.control = carControl.CarControl()
        
        # Reset state
        self.state = carState.CarState()
        
        # If data recording is enabled, close the current file and open a new one
        if self.record_data and hasattr(self, 'data_file') and self.data_file:
            old_filename = self.data_file.name
            self.data_file.close()
            self.data_file = open(f'torcs_data_{self.car_type}_{self.track_type}_{int(time.time())}.jsonl', 'w')
            print(f"Closed data file {old_filename} and opened new file {self.data_file.name} for restart")
            
        print("Driver restarted and ready for new race")


class Driver(EnhancedDriver):
    def __init__(self, stage, track_type='none', car_type='toyota_corolla_wrc', 
                 model_dir='model', manual_override=False, record_data=False,
                 model_type='xgboost', use_unified_model=False):
        """
        Legacy driver class for backward compatibility
        """
        super().__init__(stage, track_type, car_type, model_dir, manual_override, 
                         record_data, model_type, use_unified_model)