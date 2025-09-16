import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
import pickle
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm
import time
import warnings
warnings.filterwarnings('ignore')

class EnhancedBayesianOptimizationAngleSelector:
    def __init__(self, predictor, feature_columns, angle_bounds=(1, 360)):
        self.predictor = predictor
        self.feature_columns = feature_columns
        self.angle_bounds = angle_bounds
        
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=30.0, length_scale_bounds=(1, 100), nu=2.5)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-5,
            normalize_y=True,
            n_restarts_optimizer=20
        )
        
        self.X_evaluated = []
        self.y_evaluated = []
        self.use_periodic_mapping = True
        
    def _periodic_angle_transform(self, angle):
        if self.use_periodic_mapping:
            angle_rad = np.radians(angle)
            return np.array([np.sin(angle_rad), np.cos(angle_rad)])
        return np.array([angle])
    
    def _inverse_periodic_transform(self, transformed):
        if self.use_periodic_mapping:
            angle_rad = np.arctan2(transformed[0], transformed[1])
            angle = np.degrees(angle_rad)
            if angle < 0:
                angle += 360
            return angle
        return transformed[0]
        
    def _create_sample_dataframe(self, base_features, angle):
        angle_rad = np.radians(angle)
        angle_sin = np.sin(angle_rad)
        angle_cos = np.cos(angle_rad)
        
        feature_dict = {}
        base_feature_names = [col for col in self.feature_columns 
                            if col not in ['angle_sin', 'angle_cos']]
        
        for i, col in enumerate(base_feature_names):
            feature_dict[col] = [base_features[i]]
        
        feature_dict['angle_sin'] = [angle_sin]
        feature_dict['angle_cos'] = [angle_cos]
        
        sample_df = pd.DataFrame(feature_dict)
        sample_df = sample_df[self.feature_columns]
        
        return sample_df
        
    def _evaluate_objective(self, angle, base_features):
        sample_df = self._create_sample_dataframe(base_features, angle)
        current_pred = self.predictor.predict(sample_df)
        return current_pred.iloc[0] if hasattr(current_pred, 'iloc') else current_pred[0]
    
    def _evaluate_objective_batch(self, angles, base_features):
        predictions = []
        for angle in angles:
            pred = self._evaluate_objective(angle, base_features)
            predictions.append(pred)
        return np.array(predictions)
    
    def _acquisition_function_ucb(self, angle, base_features, kappa=2.0):
        if len(self.X_evaluated) == 0:
            return float('inf')
            
        X_test = self._periodic_angle_transform(angle).reshape(1, -1)
        mu, sigma = self.gp.predict(X_test, return_std=True)
        
        return mu[0] + kappa * sigma[0]
    
    def _acquisition_function_ei(self, angle, base_features, xi=0.01):
        if len(self.X_evaluated) == 0:
            return float('inf')
            
        X_test = self._periodic_angle_transform(angle).reshape(1, -1)
        mu, sigma = self.gp.predict(X_test, return_std=True)
        
        f_best = max(self.y_evaluated)
        
        if sigma[0] == 0:
            return 0
            
        z = (mu[0] - f_best - xi) / sigma[0]
        ei = (mu[0] - f_best - xi) * norm.cdf(z) + sigma[0] * norm.pdf(z)
        
        return ei
    
    def _acquisition_function_poi(self, angle, base_features, xi=0.01):
        if len(self.X_evaluated) == 0:
            return float('inf')
            
        X_test = self._periodic_angle_transform(angle).reshape(1, -1)
        mu, sigma = self.gp.predict(X_test, return_std=True)
        
        f_best = max(self.y_evaluated)
        
        if sigma[0] == 0:
            return 0
            
        z = (mu[0] - f_best - xi) / sigma[0]
        return norm.cdf(z)
    
    def _multi_start_optimization(self, acq_func, n_starts=20):
        best_value = float('-inf')
        best_angle = None
        
        start_angles = self._latin_hypercube_sampling(n_starts)
            
        for start_angle in start_angles:
            try:
                result = minimize(
                    lambda x: -acq_func(x[0]),
                    x0=[start_angle],
                    bounds=[self.angle_bounds],
                    method='L-BFGS-B',
                    options={'maxiter': 100}
                )
                
                if -result.fun > best_value:
                    best_value = -result.fun
                    best_angle = result.x[0]
            except:
                continue
        
        if best_angle is None:
            try:
                result = differential_evolution(
                    lambda x: -acq_func(x[0]),
                    bounds=[self.angle_bounds],
                    maxiter=50,
                    popsize=15
                )
                best_angle = result.x[0]
            except:
                best_angle = np.random.uniform(self.angle_bounds[0], self.angle_bounds[1])
        
        return best_angle
    
    def _latin_hypercube_sampling(self, n_samples):
        segments = np.linspace(self.angle_bounds[0], self.angle_bounds[1], n_samples + 1)
        samples = []
        
        for i in range(n_samples):
            sample = np.random.uniform(segments[i], segments[i + 1])
            samples.append(sample)
        
        np.random.shuffle(samples)
        return samples
    
    def _local_search(self, angle, base_features, radius=10, n_samples=10):
        best_angle = angle
        best_value = self._evaluate_objective(angle, base_features)
        
        for _ in range(n_samples):
            delta = np.random.uniform(-radius, radius)
            new_angle = (angle + delta) % 360
            if new_angle < self.angle_bounds[0]:
                new_angle = self.angle_bounds[0]
            elif new_angle > self.angle_bounds[1]:
                new_angle = self.angle_bounds[1]
                
            new_value = self._evaluate_objective(new_angle, base_features)
            
            if new_value > best_value:
                best_value = new_value
                best_angle = new_angle
        
        return best_angle, best_value
    
    def optimize_angle(self, base_features, n_initial=10, n_iterations=20, use_local_search=True):
        self.X_evaluated = []
        self.y_evaluated = []
        self.X_transformed = []
        start_time = time.time()
        
        initial_angles = self._latin_hypercube_sampling(n_initial)
        
        key_angles = [0, 45, 90, 135, 180, 225, 270, 315]
        for angle in key_angles:
            if len(initial_angles) < n_initial * 1.5 and angle not in initial_angles:
                initial_angles.append(angle)
        
        for angle in initial_angles[:n_initial]:
            current = self._evaluate_objective(angle, base_features)
            self.X_evaluated.append([angle])
            self.y_evaluated.append(current)
            self.X_transformed.append(self._periodic_angle_transform(angle))
        
        acquisition_functions = [self._acquisition_function_ei, self._acquisition_function_ucb, self._acquisition_function_poi]
        acq_weights = [0.5, 0.3, 0.2]
        
        for i in range(n_iterations):
            if len(self.X_transformed) > 0:
                X = np.array(self.X_transformed)
                y = np.array(self.y_evaluated)
                
                y_mean = np.mean(y)
                y_std = np.std(y) if np.std(y) > 0 else 1.0
                y_normalized = (y - y_mean) / y_std
                
                try:
                    self.gp.fit(X, y_normalized)
                except:
                    simple_kernel = ConstantKernel(1.0) * RBF(length_scale=30.0)
                    self.gp.kernel = simple_kernel
                    self.gp.fit(X, y_normalized)
            
            def combined_acquisition(angle):
                score = 0
                for acq_func, weight in zip(acquisition_functions, acq_weights):
                    score += weight * acq_func(angle, base_features)
                return score
            
            best_next_angle = self._multi_start_optimization(combined_acquisition, n_starts=20)
            
            current = self._evaluate_objective(best_next_angle, base_features)
            self.X_evaluated.append([best_next_angle])
            self.y_evaluated.append(current)
            self.X_transformed.append(self._periodic_angle_transform(best_next_angle))
            
            if i > n_iterations // 2:
                acq_weights = [0.3, 0.5, 0.2]
        
        if use_local_search:
            best_idx = np.argmax(self.y_evaluated)
            best_angle = self.X_evaluated[best_idx][0]
            
            refined_angle, refined_value = self._local_search(
                best_angle, base_features, radius=5, n_samples=10
            )
            
            if refined_value > self.y_evaluated[best_idx]:
                self.X_evaluated.append([refined_angle])
                self.y_evaluated.append(refined_value)
                self.X_transformed.append(self._periodic_angle_transform(refined_angle))
                best_angle = refined_angle
        
        best_idx = np.argmax(self.y_evaluated)
        best_angle = self.X_evaluated[best_idx][0]
        best_current = self.y_evaluated[best_idx]
        total_time = time.time() - start_time
        
        X_best = self._periodic_angle_transform(best_angle).reshape(1, -1)
        _, uncertainty = self.gp.predict(X_best, return_std=True)
        
        top_k = 5
        if len(self.y_evaluated) >= top_k:
            top_indices = np.argsort(self.y_evaluated)[-top_k:]
            top_angles = [self.X_evaluated[idx][0] for idx in top_indices]
            top_values = [self.y_evaluated[idx] for idx in top_indices]
            
            angle_std = np.std(top_angles)
            if angle_std < 10:
                refined_best_angle = np.mean(top_angles)
                refined_value = self._evaluate_objective(refined_best_angle, base_features)
                if refined_value > best_current:
                    best_angle = refined_best_angle
                    best_current = refined_value
        
        evaluation_history = {
            'angles': [x[0] for x in self.X_evaluated],
            'currents': self.y_evaluated,
            'best_angle': best_angle,
            'best_current': best_current,
            'total_time': total_time,
            'total_evaluations': len(self.X_evaluated),
            'uncertainty': uncertainty[0]
        }
        
        return best_angle, best_current, uncertainty[0], evaluation_history

def safe_load_predictor(model_dir):
    try:
        predictor = TabularPredictor.load(model_dir)
        return predictor
    except Exception as e:
        raise e

def create_scaled_predictor_wrapper(predictor):
    class ScaledPredictorWrapper:
        def __init__(self, original_predictor):
            self.predictor = original_predictor
            self.model_best = getattr(original_predictor, 'model_best', 'ScaledWrapper')
            self.stats = self._calculate_scale_params()
        
        def _calculate_scale_params(self):
            try:
                data = pd.read_csv("data.csv")
                
                base_stats = {
                    'mean': data['current'].mean(),
                    'std': data['current'].std(),
                    'min': data['current'].min(),
                    'max': data['current'].max()
                }
                
                feature_columns = ['pos1', 'pos2', 'pos3', 'pos4', 
                                  'sunlight_x', 'sunlight_y', 'sunlight_z', 'time', 
                                  'angle_sin', 'angle_cos', 'source']
                
                scale_factors = []
                
                for i in range(min(100, len(data))):
                    sample = data.iloc[i]
                    true_current = sample['current']
                    
                    if abs(true_current) < 1e-6:
                        continue
                        
                    test_data = {}
                    for col in feature_columns:
                        if col == 'angle_sin':
                            test_data[col] = [0.5]
                        elif col == 'angle_cos':
                            test_data[col] = [0.866]
                        elif col in sample:
                            test_data[col] = [sample[col]]
                        else:
                            test_data[col] = [0.0]
                    
                    try:
                        test_df = pd.DataFrame(test_data)
                        raw_pred = self.predictor.predict(test_df)[0]
                        
                        standard_pred = raw_pred * base_stats['std'] + base_stats['mean']
                        
                        if abs(standard_pred) > 1e-6:
                            scale_factor = true_current / standard_pred
                            if 0.1 < scale_factor < 10:
                                scale_factors.append(scale_factor)
                    except:
                        continue
                
                if scale_factors:
                    avg_scale_factor = np.median(scale_factors)
                    filtered_factors = [f for f in scale_factors if 0.5 * avg_scale_factor < f < 2 * avg_scale_factor]
                    if filtered_factors:
                        avg_scale_factor = np.mean(filtered_factors)
                    
                    base_stats['scale_factor'] = avg_scale_factor
                else:
                    base_stats['scale_factor'] = 1.0
                
                return base_stats
                
            except Exception as e:
                return {
                    'mean': 0,
                    'std': 1, 
                    'scale_factor': 1.0,
                    'min': 0,
                    'max': 1
                }
        
        def predict(self, X):
            raw_pred = self.predictor.predict(X)
            
            if hasattr(raw_pred, '__iter__') and len(raw_pred) > 1:
                fixed_pred = []
                for pred in raw_pred:
                    fixed = pred * self.stats['std'] + self.stats['mean']
                    fixed = fixed * self.stats['scale_factor']
                    fixed_pred.append(max(0, fixed))
                return pd.Series(fixed_pred) if hasattr(raw_pred, 'index') else np.array(fixed_pred)
            else:
                if hasattr(raw_pred, 'iloc'):
                    raw_value = raw_pred.iloc[0]
                elif hasattr(raw_pred, '__getitem__'):
                    raw_value = raw_pred[0]
                else:
                    raw_value = raw_pred
                
                fixed = raw_value * self.stats['std'] + self.stats['mean']
                fixed = fixed * self.stats['scale_factor']
                result = max(0, fixed)
                
                if hasattr(raw_pred, 'iloc'):
                    return pd.Series([result])
                else:
                    return np.array([result])
        
        def __getattr__(self, name):
            return getattr(self.predictor, name)
    
    return ScaledPredictorWrapper(predictor)

def load_trained_model(model_dir="./satellite_model"):
    if not os.path.exists(model_dir):
        print(f"Error: Model directory does not exist: {model_dir}")
        return None, None, None
    
    try:
        predictor = safe_load_predictor(model_dir)
        
        feature_columns = None
        feature_path = os.path.join(model_dir, 'feature_columns.pkl')
        
        if os.path.exists(feature_path):
            with open(feature_path, 'rb') as f:
                feature_columns = pickle.load(f)
        else:
            feature_columns = [
                'pos1', 'pos2', 'pos3', 'pos4', 
                'sunlight_x', 'sunlight_y', 'sunlight_z', 'time', 
                'angle_sin', 'angle_cos', 'source'
            ]
        
        wrapped_predictor = create_scaled_predictor_wrapper(predictor)
        
        training_info = {
            'feature_columns': feature_columns,
            'model_path': model_dir,
            'is_wrapped': True,
            'wrapper_type': 'scaled'
        }
        
        return wrapped_predictor, feature_columns, training_info
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None

def run_enhanced_batch_optimization(n_samples=5, error_threshold=100):
    print("=" * 70)
    print(f"Enhanced Bayesian Optimization for Solar Panel Angle Prediction")
    print(f"Number of samples: {n_samples}")
    print(f"Error threshold: {error_threshold}°")
    print("=" * 70)
    
    predictor, feature_columns, training_info = load_trained_model()
    if predictor is None:
        print("Failed to load model")
        return None
    
    try:
        data = pd.read_csv("data.csv")
        print(f"Data loaded: {data.shape[0]} samples available")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    if n_samples > len(data):
        n_samples = len(data)
        print(f"Adjusted sample size to {n_samples}")
    
    samples = data.head(n_samples)
    base_feature_names = [col for col in feature_columns 
                         if col not in ['angle_sin', 'angle_cos']]
    
    results = []
    valid_results = []
    anomalies = []
    total_start_time = time.time()
    
    print("\nProcessing samples...")
    print("-" * 80)
    print(f"{'Sample':<8} {'Predicted':<12} {'True':<12} {'Error':<12} {'Status':<12} {'Evals':<8} {'Time(s)':<10}")
    print("-" * 80)
    
    for idx, (_, sample) in enumerate(samples.iterrows()):
        base_features = [sample[col] if col in sample else 0 for col in base_feature_names]
        
        optimizer = EnhancedBayesianOptimizationAngleSelector(
            predictor=predictor,
            feature_columns=feature_columns
        )
        
        best_angle, best_current, uncertainty, history = optimizer.optimize_angle(
            base_features=base_features,
            n_initial=10,
            n_iterations=20,
            use_local_search=True
        )
        
        result = {
            'sample_index': idx + 1,
            'original_index': sample.name,
            'predicted_angle': best_angle,
            'evaluations_used': history['total_evaluations'],
            'optimization_time': history['total_time'],
            'uncertainty': uncertainty
        }
        
        if 'best_angle' in sample:
            result['true_angle'] = sample['best_angle']
            result['angle_error'] = abs(best_angle - sample['best_angle'])
            
            if result['angle_error'] > 180:
                result['angle_error'] = 360 - result['angle_error']
            
            if result['angle_error'] > error_threshold:
                result['status'] = 'ANOMALY'
                anomalies.append(result)
                status = 'ANOMALY'
            else:
                result['status'] = 'VALID'
                valid_results.append(result)
                status = 'VALID'
            
            print(f"{idx+1:<8} {best_angle:<12.1f} {sample['best_angle']:<12.1f} {result['angle_error']:<12.1f} {status:<12} {result['evaluations_used']:<8} {result['optimization_time']:<10.3f}")
        else:
            result['true_angle'] = None
            result['angle_error'] = None
            result['status'] = 'NO_REF'
            valid_results.append(result)
            
            print(f"{idx+1:<8} {best_angle:<12.1f} {'N/A':<12} {'N/A':<12} {'NO_REF':<12} {result['evaluations_used']:<8} {result['optimization_time']:<10.3f}")
        
        results.append(result)
    
    total_time = time.time() - total_start_time
    
    print("-" * 80)
    print("\nSummary Statistics:")
    print("-" * 70)
    
    valid_with_ref = [r for r in valid_results if r['angle_error'] is not None]
    
    print(f"Total samples processed: {n_samples}")
    print(f"Valid predictions: {len(valid_results)} ({len(valid_results)/n_samples*100:.1f}%)")
    print(f"Anomalies detected: {len(anomalies)} ({len(anomalies)/n_samples*100:.1f}%)")
    
    if valid_with_ref:
        angle_errors = [r['angle_error'] for r in valid_with_ref]
        
        print(f"\nValid Results Statistics:")
        print(f"Mean angle error: {np.mean(angle_errors):.2f}°")
        print(f"Std angle error: {np.std(angle_errors):.2f}°")
        print(f"Max angle error: {np.max(angle_errors):.2f}°")
        print(f"Min angle error: {np.min(angle_errors):.2f}°")
        print(f"Median angle error: {np.median(angle_errors):.2f}°")
        
        accuracy_5 = np.mean([1 if err <= 5 else 0 for err in angle_errors])
        accuracy_10 = np.mean([1 if err <= 10 else 0 for err in angle_errors])
        accuracy_15 = np.mean([1 if err <= 15 else 0 for err in angle_errors])
        accuracy_20 = np.mean([1 if err <= 20 else 0 for err in angle_errors])
        
        print(f"\nPrediction Accuracy (Valid Samples):")
        print(f"Accuracy (±5°): {accuracy_5*100:.1f}%")
        print(f"Accuracy (±10°): {accuracy_10*100:.1f}%")
        print(f"Accuracy (±15°): {accuracy_15*100:.1f}%")
        print(f"Accuracy (±20°): {accuracy_20*100:.1f}%")
    
    if anomalies:
        print(f"\nAnomaly Analysis:")
        anomaly_errors = [r['angle_error'] for r in anomalies]
        print(f"Mean anomaly error: {np.mean(anomaly_errors):.2f}°")
        print(f"Anomaly indices: {[r['sample_index'] for r in anomalies]}")
    
    total_evaluations = sum(r['evaluations_used'] for r in results)
    traditional_evaluations = n_samples * 360
    efficiency = (1 - total_evaluations/traditional_evaluations) * 100
    
    print(f"\nEfficiency Analysis:")
    print(f"Total evaluations: {total_evaluations}")
    print(f"Traditional method: {traditional_evaluations}")
    print(f"Efficiency improvement: {efficiency:.1f}%")
    print(f"Total processing time: {total_time:.3f}s")
    print(f"Average time per sample: {total_time/n_samples:.3f}s")
    print(f"Average evaluations per sample: {total_evaluations/n_samples:.1f}")
    
    results_df = pd.DataFrame(results)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    all_results_path = f'angle_optimization_enhanced_all_results_{timestamp}.csv'
    results_df.to_csv(all_results_path, index=False)
    print(f"\nAll results saved to: {all_results_path}")
    
    if valid_results:
        valid_df = pd.DataFrame(valid_results)
        valid_results_path = f'angle_optimization_enhanced_valid_results_{timestamp}.csv'
        valid_df.to_csv(valid_results_path, index=False)
        print(f"Valid results saved to: {valid_results_path}")
    
    return results

def main():
    print("Enhanced Solar Panel Angle Optimization using Bayesian Optimization")
    print("=" * 70)
    
    try:
        n_samples = input("\nEnter number of samples to process: ").strip()
        n_samples = int(n_samples)
        
        if n_samples <= 0:
            print("Number of samples must be positive")
            return
        
        threshold_input = input("Enter angle error threshold in degrees (default=100): ").strip()
        if threshold_input:
            try:
                error_threshold = float(threshold_input)
                if error_threshold <= 0:
                    print("Threshold must be positive, using default (100°)")
                    error_threshold = 100
            except ValueError:
                print("Invalid threshold, using default (100°)")
                error_threshold = 100
        else:
            error_threshold = 100
            
        results = run_enhanced_batch_optimization(n_samples, error_threshold)
        
        if results is None:
            print("Optimization failed")
            return
            
        print("\nOptimization completed successfully")
        
    except ValueError:
        print("Please enter a valid number")
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
