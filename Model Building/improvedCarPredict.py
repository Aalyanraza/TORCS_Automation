import pandas as pd
import numpy as np
import os
import json
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

class ImprovedCarPredict:
    """
    Improved prediction system for TORCS car control.
    Supports:
    1. Training separate models for different track types
    2. Including car type as a feature
    3. Multiple model types (Random Forest, Gradient Boosting)
    4. Model evaluation and comparison
    """
    def __init__(self):
        self.models = {}  # Dictionary to store models for each track
        self.scalers = {}  # Dictionary to store feature scalers for each track
        self.track_types = ['oval', 'road', 'dirt', 'unified']
        self.target_columns = ['steer', 'accel', 'brake', 'gear']
        
    def preprocess_data(self, data_files, track_mapping=None):
        import pandas as pd
        import json

        if isinstance(data_files, list):
            if track_mapping is None:
                raise ValueError("track_mapping must be provided if data_files is a list")
            file_track_map = {file: track_mapping.get(file, 'unknown') for file in data_files}
        else:
            file_track_map = data_files

        track_data = {track: [] for track in self.track_types}

        for file_path, track_type in file_track_map.items():
            if track_type not in self.track_types:
                print(f"Warning: Unknown track type {track_type} for {file_path}, skipping")
                continue

            car_type = 'unknown'
            for car in self.car_types:
                if car.lower() in file_path.lower():
                    car_type = car
                    break

            with open(file_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                try:
                    entry = json.loads(line.strip())
                    state = entry.get('state', {})
                    action = entry.get('action', {})

                    if not state or not action:
                        continue

                    angle = state.get('angle', 0.0)

                    # Skip if car is facing backwards (> ~145 degrees)
                    if abs(angle) > 2.5:
                        continue

                    features = {
                        'car_type': car_type,
                        'angle': angle,
                        'speedX': state.get('speedX', 0.0),
                        'speedY': state.get('speedY', 0.0),
                        'speedZ': state.get('speedZ', 0.0),
                        'rpm': state.get('rpm', 0.0),
                        'trackPos': state.get('trackPos', 0.0),
                        'damage': state.get('damage', 0.0),
                        'fuel': state.get('fuel', 0.0),
                        'gear': state.get('gear', 0),
                        'forward_aligned': int(abs(angle) < 1.0)  # Add new feature
                    }

                    for i, sensor in enumerate(state.get('track', [])[:19]):
                        features[f'track_{i}'] = sensor

                    for i, sensor in enumerate(state.get('wheelSpinVel', [])[:4]):
                        features[f'wheelSpinVel_{i}'] = sensor

                    features['steer'] = action.get('steer', 0.0)
                    features['accel'] = action.get('accel', 0.0)
                    features['brake'] = action.get('brake', 0.0)
                    features['gear'] = action.get('gear', 1)

                    track_data[track_type].append(features)

                except json.JSONDecodeError:
                    print(f"Warning: Could not parse line in {file_path}")
                except Exception as e:
                    print(f"Error processing line in {file_path}: {e}")

        for track_type in self.track_types:
            if track_data[track_type]:
                track_data[track_type] = pd.DataFrame(track_data[track_type])
                print(f"Processed {len(track_data[track_type])} samples for {track_type} tracks")
            else:
                track_data[track_type] = pd.DataFrame()
                print(f"No data for {track_type} tracks")

        return track_data
    
    def train(self, track_data, model_type='xgboost', unified_model=False):
        """
        Train models for each track type or a unified model for all tracks
        
        Parameters:
        - track_data: Dictionary of DataFrames organized by track type
        - model_type: 'random_forest', 'gradient_boosting', 'xgboost', or 'decision_tree'
        - unified_model: If True, train a single model for all track types
        """
        if unified_model:
            # Combine all track data into one dataframe
            combined_data = pd.DataFrame()
            for track_type, data in track_data.items():
                if not data.empty:
                    # Add track_type as a feature
                    data = data.copy()
                    data['track_type'] = track_type
                    combined_data = pd.concat([combined_data, data], ignore_index=True)
            
            if combined_data.empty:
                print("No data available for training unified model")
                return
                
            print(f"\nTraining unified model using {model_type}...")
            
            feature_cols = [col for col in combined_data.columns if col not in self.target_columns]
            X = combined_data[feature_cols]
            y = combined_data[self.target_columns]
            
            categorical_cols = ['car_type', 'track_type']
            categorical_cols = [col for col in categorical_cols if col in X.columns]
            numerical_cols = [col for col in X.columns if col not in categorical_cols]
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_cols),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
                ]
            )
            
            # Create model based on type
            if model_type == 'xgboost':
                models = {
                    target: Pipeline([
                        ('preprocessor', preprocessor),
                        ('regressor', xgb.XGBRegressor(
                            n_estimators=200,
                            max_depth=6,
                            learning_rate=0.1,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            random_state=42,
                            verbosity=0
                        ))
                    ]) for target in self.target_columns
                }
            elif model_type == 'random_forest':
                models = {
                    target: Pipeline([
                        ('preprocessor', preprocessor),
                        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
                    ]) for target in self.target_columns
                }
            elif model_type == 'gradient_boosting':
                models = {
                    target: Pipeline([
                        ('preprocessor', preprocessor),
                        ('regressor', GradientBoostingRegressor(n_estimators=100, random_state=42))
                    ]) for target in self.target_columns
                }
            elif model_type == 'decision_tree':
                models = {
                    target: Pipeline([
                        ('preprocessor', preprocessor),
                        ('regressor', DecisionTreeRegressor(max_depth=10, random_state=42))
                    ]) for target in self.target_columns
                }
            else:
                raise ValueError(f"Unknown model type: {model_type}")
                
            for target, model in models.items():
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y[target], test_size=0.2, random_state=42
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                print(f"{target} model - MSE: {mse:.4f}, R²: {r2:.4f}")
                
            self.models['unified'] = models
        else:
            # Original per-track training logic
            for track_type, data in track_data.items():
                if data.empty:
                    print(f"No data for {track_type}, skipping model training")
                    continue

                print(f"\nTraining model for {track_type} tracks using {model_type}...")

                missing_targets = [col for col in self.target_columns if col not in data.columns]
                if missing_targets:
                    print(f"Missing target columns {missing_targets} in {track_type} data, skipping")
                    continue

                feature_cols = [col for col in data.columns if col not in self.target_columns]
                X = data[feature_cols]
                y = data[self.target_columns]

                categorical_cols = ['car_type']
                categorical_cols = [col for col in categorical_cols if col in X.columns]
                numerical_cols = [col for col in X.columns if col not in categorical_cols]

                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', StandardScaler(), numerical_cols),
                        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
                    ]
                )

                # Create model based on type
                if model_type == 'xgboost':
                    models = {
                        target: Pipeline([
                            ('preprocessor', preprocessor),
                            ('regressor', xgb.XGBRegressor(
                                n_estimators=200,
                                max_depth=6,
                                learning_rate=0.1,
                                subsample=0.8,
                                colsample_bytree=0.8,
                                random_state=42,
                                verbosity=0
                            ))
                        ]) for target in self.target_columns
                    }
                elif model_type == 'random_forest':
                    models = {
                        target: Pipeline([
                            ('preprocessor', preprocessor),
                            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
                        ]) for target in self.target_columns
                    }
                elif model_type == 'gradient_boosting':
                    models = {
                        target: Pipeline([
                            ('preprocessor', preprocessor),
                            ('regressor', GradientBoostingRegressor(n_estimators=100, random_state=42))
                        ]) for target in self.target_columns
                    }
                elif model_type == 'decision_tree':
                    models = {
                        target: Pipeline([
                            ('preprocessor', preprocessor),
                            ('regressor', DecisionTreeRegressor(max_depth=10, random_state=42))
                        ]) for target in self.target_columns
                    }
                else:
                    raise ValueError(f"Unknown model type: {model_type}")

                for target, model in models.items():
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y[target], test_size=0.2, random_state=42
                    )
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    print(f"{target} model - MSE: {mse:.4f}, R²: {r2:.4f}")

                self.models[track_type] = models
           
    def save_models(self, directory='models'):
        """Save trained models to disk"""
        os.makedirs(directory, exist_ok=True)
        
        for track_type, models in self.models.items():
            track_dir = os.path.join(directory, track_type)
            os.makedirs(track_dir, exist_ok=True)
            
            for target, model in models.items():
                model_path = os.path.join(track_dir, f"{target}_model.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                    
            print(f"Saved models for {track_type} to {track_dir}")
    
    def load_models(self, directory='models'):
        """Load trained models from disk"""
        self.models = {}
        
        for track_type in self.track_types:
            track_dir = os.path.join(directory, track_type)
            if not os.path.exists(track_dir):
                continue
                
            models = {}
            for target in self.target_columns:
                model_path = os.path.join(track_dir, f"{target}_model.pkl")
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        models[target] = pickle.load(f)
                else:
                    print(f"No model found for {target} in {track_type} tracks")
                    
            if models:
                self.models[track_type] = models
                print(f"Loaded models for {track_type} tracks")
    
    def predict(self, state, track_type='road', car_type='toyota_corolla_wrc'):
        """
        Predict control actions for a given car state
        
        Parameters:
        - state: CarState object or dict with state values
        - track_type: Type of track ('oval', 'road', 'dirt')
        - car_type: Type of car
        
        Returns:
        - Dictionary with predicted actions
        """
        # Check if using unified model
        if 'unified' in self.models:
            track_type = 'unified'
        elif track_type not in self.models:
            track_type = next(iter(self.models)) if self.models else None
            
        if not track_type or not self.models.get(track_type):
            raise ValueError("No models available for prediction")
            
        # Extract features from CarState object or use dict directly
        if hasattr(state, 'getAngle'):
            # It's a CarState object
            features = {
                'car_type': car_type,
                'angle': state.getAngle(),
                'speedX': state.getSpeedX(),
                'speedY': state.getSpeedY(),
                'speedZ': state.getSpeedZ(),
                'rpm': state.getRpm(),
                'trackPos': state.getTrackPos(),
                'damage': state.getDamage() if hasattr(state, 'getDamage') else 0.0,
                'fuel': state.getFuel() if hasattr(state, 'getFuel') else 100.0,
                'gear': state.getGear(),
                'forward_aligned': int(abs(state.getAngle()) < 1.0)
            }
            
            # Add track distances
            track = state.getTrack() or []
            for i in range(min(19, len(track))):
                features[f'track_{i}'] = track[i]
                
            # Add wheel spin velocities
            wheel = state.getWheelSpinVel() or []
            for i in range(min(4, len(wheel))):
                features[f'wheelSpinVel_{i}'] = wheel[i]
        else:
            # It's a dict
            angle = state.get('angle', 0.0)
            features = {
                'car_type': car_type,
                'angle': angle,
                'speedX': state.get('speedX', 0.0),
                'speedY': state.get('speedY', 0.0),
                'speedZ': state.get('speedZ', 0.0),
                'rpm': state.get('rpm', 0.0),
                'trackPos': state.get('trackPos', 0.0),
                'damage': state.get('damage', 0.0),
                'fuel': state.get('fuel', 100.0),
                'gear': state.get('gear', 0),
                'forward_aligned': int(abs(angle) < 1.0)
            }
            
            # Add track distances
            track = state.get('track', [])
            for i in range(min(19, len(track))):
                features[f'track_{i}'] = track[i] if i < len(track) else 200.0
                
            # Add wheel spin velocities
            wheel = state.get('wheelSpinVel', [])
            for i in range(min(4, len(wheel))):
                features[f'wheelSpinVel_{i}'] = wheel[i] if i < len(wheel) else 0.0

        # Make sure all required columns are present
        # This ensures we have all columns needed for prediction,
        # handling cases where new features were added during training
        for i in range(19):
            if f'track_{i}' not in features:
                features[f'track_{i}'] = 200.0  # Default to large distance
                
        for i in range(4):
            if f'wheelSpinVel_{i}' not in features:
                features[f'wheelSpinVel_{i}'] = 0.0
        
        # Add track_type for unified model
        if track_type == 'unified':
            features['track_type'] = track_type if track_type in self.track_types else 'road'
        
        # Create feature DataFrame
        X = pd.DataFrame([features])

        # Make predictions
        predictions = {}
        for target, model in self.models[track_type].items():
            try:
                pred = model.predict(X)[0]

                # Clamp outputs
                if target == 'gear':
                    pred = round(pred)
                    pred = max(-1, min(6, pred))
                elif target in ['accel', 'brake']:
                    pred = max(0, min(1, pred))
                elif target == 'steer':
                    pred = max(-1, min(1, pred))

                predictions[target] = pred
            except Exception as e:
                print(f"Error predicting {target}: {e}")
                predictions[target] = {'steer': 0.0, 'accel': 0.0, 'brake': 0.0, 'gear': 1}[target]

        # Extra safety check: if car is reversed, correct it
        if abs(features['angle']) > 2.0:
            print("Warning: Car appears to be facing backwards. Applying emergency stop.")
            predictions['steer'] = 0.0
            predictions['accel'] = 0.0
            predictions['brake'] = 1.0
            predictions['gear'] = -1

        return predictions


def demo():
    """Demonstration of training and using the improved prediction system"""
    # Example mapping of filenames to track types
    data_files = [
        # Peugeot 406
        "data/p406/Dirt2/torcs_data_1746808577.jsonl",
        "data/p406/Dirt2/torcs_data_1746809073.jsonl",
        "data/p406/Dirt2/torcs_data_1746809512.jsonl",
        "data/p406/E-Track3/torcs_data_1746809956.jsonl",
        "data/p406/E-Track3/torcs_data_1746810453.jsonl",
        "data/p406/G-Speedway/torcs_data_1746806221.jsonl",
        "data/p406/G-Speedway/torcs_data_1746806496.jsonl",
        "data/p406/G-Speedway/torcs_data_1746806685.jsonl",

        # Toyota Corolla WRC
        "data/pw-corollawrc/Dirt2/pw-corollawrc track2.jsonl",
        "data/pw-corollawrc/Dirt2/torcs_data_1746803937.jsonl",
        "data/pw-corollawrc/Dirt2/torcs_data_1746804366.jsonl",
        "data/pw-corollawrc/E-Track3/torcs_data_1746804968.jsonl",
        "data/pw-corollawrc/E-Track3/torcs_data_1746805585.jsonl",
        "data/pw-corollawrc/G-Speedway/pw-corollawrc track1.jsonl",

        # Mitsubishi Lancer
        "data/pw-evoviwrc/Dirt2/torcs_data_1746811832.jsonl",
        "data/pw-evoviwrc/Dirt2/torcs_data_1746812235.jsonl",
        "data/pw-evoviwrc/E-Track3/torcs_data_1746813329.jsonl",
        "data/pw-evoviwrc/E-Track3/torcs_data_1746813532.jsonl",
        "data/pw-evoviwrc/E-Track3/torcs_data_1746814012.jsonl",
        "data/pw-evoviwrc/G-Speedway/torcs_data_1746811249.jsonl",
        "data/pw-evoviwrc/G-Speedway/torcs_data_1746811461.jsonl",
        "data/pw-evoviwrc/G-Speedway/torcs_data_1746811628.jsonl",
    ]
    
    track_mapping = {
        # Map each file to its track type
        "data/p406/Dirt2/torcs_data_1746808577.jsonl": "dirt",
        "data/p406/Dirt2/torcs_data_1746809073.jsonl": "dirt",
        "data/p406/Dirt2/torcs_data_1746809512.jsonl": "dirt",
        "data/p406/E-Track3/torcs_data_1746809956.jsonl": "road",
        "data/p406/E-Track3/torcs_data_1746810453.jsonl": "road",
        "data/p406/G-Speedway/torcs_data_1746806221.jsonl": "oval",
        "data/p406/G-Speedway/torcs_data_1746806496.jsonl": "oval",
        "data/p406/G-Speedway/torcs_data_1746806685.jsonl": "oval",

        "data/pw-corollawrc/Dirt2/pw-corollawrc track2.jsonl": "dirt",
        "data/pw-corollawrc/Dirt2/torcs_data_1746803937.jsonl": "dirt", 
        "data/pw-corollawrc/Dirt2/torcs_data_1746804366.jsonl": "dirt",
        "data/pw-corollawrc/E-Track3/torcs_data_1746804968.jsonl": "road",
        "data/pw-corollawrc/E-Track3/torcs_data_1746805585.jsonl": "road",
        "data/pw-corollawrc/G-Speedway/pw-corollawrc track1.jsonl": "oval",

        "data/pw-evoviwrc/Dirt2/torcs_data_1746811832.jsonl": "dirt",
        "data/pw-evoviwrc/Dirt2/torcs_data_1746812235.jsonl": "dirt",
        "data/pw-evoviwrc/E-Track3/torcs_data_1746813329.jsonl": "road",
        "data/pw-evoviwrc/E-Track3/torcs_data_1746813532.jsonl": "road",
        "data/pw-evoviwrc/E-Track3/torcs_data_1746814012.jsonl": "road",
        "data/pw-evoviwrc/G-Speedway/torcs_data_1746811249.jsonl": "oval",
        "data/pw-evoviwrc/G-Speedway/torcs_data_1746811461.jsonl": "oval",
        "data/pw-evoviwrc/G-Speedway/torcs_data_1746811628.jsonl": "oval",

    }
    
    # Initialize predictor
    predictor = ImprovedCarPredict()
    track_data = predictor.preprocess_data(data_files, track_mapping)
    
    # Train with unified model
    print("\n=== Training unified model with XGBoost ===")
    predictor.train(track_data, model_type='xgboost', unified_model=True)
    
    # Save models
    predictor.save_models()
    
    # Train with decision tree (per track type)
    print("\n=== Training per-track models with Decision Tree ===")
    predictor.train(track_data, model_type='decision_tree')
    
    # Save models
    predictor.save_models(directory='models_dt')
    
    # Load models
    predictor.load_models()
    
    # Example prediction - with all required features included
    sample_state = {
        'angle': 0.0022394,
        'speedX': 10.5,
        'speedY': -0.007833,
        'speedZ': -0.001917,
        'rpm': 3500,
        'trackPos': 0.33287,
        'gear': 3,
        'damage': 0.0,
        'fuel': 80.0,
        'forward_aligned': 1,
        'track': [3.33566, 3.4554, 3.85667, 4.72792, 6.69729, 9.81319, 12.9966, 19.4564, 
                 56.3064, 49.2058, 41.694, 33.8028, 25.5357, 19.3662, 13.2772, 9.40378, 
                 7.68541, 6.89532, 6.66437],
        'wheelSpinVel': [53.2, 53.1, 51.4, 51.5]
    }
    
    # Predict using unified model
    print("\n=== Predictions with unified model ===")
    for car in ['toyota_corolla_wrc', 'peugeot_406', 'mitsubishi_lancer']:
        print(f"\nPredictions for {car}:")
        predictions = predictor.predict(sample_state, 'road', car)
        for action, value in predictions.items():
            print(f"  {action}: {value:.4f}")
            
    # Load track-specific decision tree models for comparison
    predictor.load_models(directory='models_dt')
    
    # Predict using decision tree models
    print("\n=== Predictions with decision tree model ===")
    for track in ['oval', 'road', 'dirt']:
        for car in ['toyota_corolla_wrc']:
            if track in predictor.models:
                print(f"\nPredictions for {car} on {track} track:")
                predictions = predictor.predict(sample_state, track, car)
                for action, value in predictions.items():
                    print(f"  {action}: {value:.4f}")


if __name__ == "__main__":
    demo()