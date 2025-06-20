import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

class CarPredict:
    def __init__(self, filename):
        # Load CSV
        data = pd.read_csv(filename)
        
        # Drop irrelevant or redundant columns
        drop_cols = ['timestamp_episode', 'step', 'curLapTime', 'damage', 'distRaced',
                     'fuel', 'lastLapTime', 'racePos', 'z', 'opponents', 'action']
        data = data.drop(columns=[col for col in drop_cols if col in data.columns])
        
        # Track is already split as track_0 to track_18
        track_cols = [f'track_{i}' for i in range(19)]
        
        # wheelSpinVel is already split as wheelSpinVel_0 to wheelSpinVel_3
        wheel_cols = [f'wheelSpinVel_{i}' for i in range(4)]
        
        # Features and targets
        feature_cols = ['angle', 'speedX', 'speedY', 'speedZ', 'rpm', 'trackPos'] + track_cols + wheel_cols
        target_cols = ['steer', 'accel', 'brake', 'gear_action']  # Use gear_action as target
        
        # Drop NaN and ensure all columns exist
        missing_cols = [col for col in feature_cols + target_cols if col not in data.columns]
        if missing_cols:
            raise KeyError(f"Missing columns in CSV: {missing_cols}")
        data = data[feature_cols + target_cols].dropna()
        
        # Split features and targets
        X = data[feature_cols]
        y = data[target_cols]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train models
        self.models = {}
        for action in target_cols:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train[action])
            self.models[action] = model
            score = model.score(X_test_scaled, y_test[action])
            print(f"{action} model R^2 score: {score:.4f}")
        
        self.feature_cols = feature_cols

    def newPrediction(self, state):
        # Convert CarState to DataFrame
        track = state.getTrack()
        wheel = state.getWheelSpinVel()
        features = {
            'angle': state.getAngle(),
            'speedX': state.getSpeedX(),
            'speedY': state.getSpeedY(),
            'speedZ': state.getSpeedZ(),
            'rpm': state.getRpm(),
            'trackPos': state.getTrackPos()
        }
        for i in range(19):
            features[f'track_{i}'] = track[i]
        for i in range(4):
            features[f'wheelSpinVel_{i}'] = wheel[i]  # Match CSV column names
        
        features_df = pd.DataFrame([features], columns=self.feature_cols)
        
        # Scale features
        features_scaled = self.scaler.transform(features_df)
        
        # Predict actions
        predictions = {}
        for action, model in self.models.items():
            pred = model.predict(features_scaled)[0]
            if action == 'gear_action':
                pred = round(pred)
                pred = max(-1, min(6, pred))
            else:
                pred = max(0, min(1, pred)) if action in ['accel', 'brake'] else max(-1, min(1, pred))
            predictions[action] = pred
        
        return predictions