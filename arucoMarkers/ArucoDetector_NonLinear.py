import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import optimize
import json
from datetime import datetime

class EnhancedArucoDetector:
    def __init__(self, camera_id, dictionary_id, marker_length, camera_matrix, dist_coeffs):
        """
        Enhanced ArUco detector with calibration capabilities
        """
        self.camera_id = camera_id
        self.dictionary_id = dictionary_id
        self.marker_length = marker_length
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        
        # Set up detector
        self.detector_params = cv2.aruco.DetectorParameters()
        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.detector_params)
        
        # Set up video capture
        self.cap = cv2.VideoCapture(camera_id)
        
        # Use your rosbag resolution (640x480) for calibration
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.image_width = 640
        self.image_height = 480
        
        # Calculate focal length from camera matrix
        self.focal_length_x = camera_matrix[0, 0]
        self.focal_length_y = camera_matrix[1, 1]
        self.focal_length = (self.focal_length_x + self.focal_length_y) / 2.0
        
        # Define marker corner points in 3D
        self.obj_points = np.zeros((4, 1, 3), dtype=np.float32)
        self.obj_points[0, 0] = [-marker_length/2.0, marker_length/2.0, 0]
        self.obj_points[1, 0] = [marker_length/2.0, marker_length/2.0, 0]
        self.obj_points[2, 0] = [marker_length/2.0, -marker_length/2.0, 0]
        self.obj_points[3, 0] = [-marker_length/2.0, -marker_length/2.0, 0]
        
        # Data storage for calibration
        self.range_measurements = []
        self.pixel_measurements = []
        
        print(f"Enhanced ArUco Detector initialized:")
        print(f"Resolution: {self.image_width}x{self.image_height}")
        print(f"Focal length: {self.focal_length:.1f} pixels")
        print(f"Marker size: {marker_length*100}cm")
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open camera at index {camera_id}")
    
    def detect_markers_with_pose(self, image):
        """
        Detect ArUco markers and estimate their pose
        Returns: corners, ids, rvecs, tvecs, distances, marker_pixel_sizes
        """
        corners, ids, rejected = self.detector.detectMarkers(image)
        
        rvecs = tvecs = distances = marker_pixel_sizes = None
        if ids is not None and len(ids) > 0:
            rvecs = []
            tvecs = []
            distances = []
            marker_pixel_sizes = []
            
            for i in range(len(ids)):
                # Use solvePnP for pose estimation
                success, rvec, tvec = cv2.solvePnP(
                    self.obj_points, corners[i], self.camera_matrix, self.dist_coeffs
                )
                if success:
                    rvecs.append(rvec)
                    tvecs.append(tvec)
                    # Calculate distance from translation vector
                    distance = np.linalg.norm(tvec)
                    distances.append(distance)
                    
                    # Calculate marker pixel size (average side length)
                    corner_points = corners[i].reshape(-1, 2)
                    side_lengths = []
                    for j in range(4):
                        p1 = corner_points[j]
                        p2 = corner_points[(j + 1) % 4]
                        side_length = np.linalg.norm(p2 - p1)
                        side_lengths.append(side_length)
                    avg_pixel_size = np.mean(side_lengths)
                    marker_pixel_sizes.append(avg_pixel_size)
        
        return corners, ids, rvecs, tvecs, distances, marker_pixel_sizes
    
    def run_detection(self):
        """
        Original detection functionality
        """
        print("Starting ArUco marker detection...")
        print("Press ESC to exit, 'c' for calibration mode")
        
        while self.cap.grab():
            ret, image = self.cap.retrieve()
            if not ret:
                break
            
            corners, ids, rvecs, tvecs, distances, marker_pixel_sizes = self.detect_markers_with_pose(image)
            
            # Draw results
            image_copy = image.copy()
            if ids is not None and len(ids) > 0:
                cv2.aruco.drawDetectedMarkers(image_copy, corners, ids)
                
                # Draw axes and distance info
                for i in range(len(ids)):
                    if i < len(rvecs) and i < len(tvecs):
                        cv2.drawFrameAxes(image_copy, self.camera_matrix, self.dist_coeffs, 
                                          rvecs[i], tvecs[i], self.marker_length * 1.5, 2)
                        
                        # Display distance and pixel size
                        if i < len(distances):
                            distance_text = f"ID{ids[i][0]}: {distances[i]:.2f}m"
                            cv2.putText(image_copy, distance_text, 
                                       (int(corners[i][0][0][0]), int(corners[i][0][0][1]) - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        if i < len(marker_pixel_sizes):
                            pixel_text = f"{marker_pixel_sizes[i]:.1f}px"
                            cv2.putText(image_copy, pixel_text, 
                                       (int(corners[i][0][0][0]), int(corners[i][0][0][1]) + 20),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            cv2.putText(image_copy, "Press 'c' for calibration mode", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("ArUco Detection", image_copy)
            
            key = cv2.waitKey(10) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('c'):
                self.run_calibration_mode()
                break
        
        self.cap.release()
        cv2.destroyAllWindows()
    
    def run_calibration_mode(self):
        """
        Interactive calibration mode
        """
        print("\n" + "="*50)
        print("CALIBRATION MODE")
        print("="*50)
        print("Choose calibration type:")
        print("1. Range calibration (distance accuracy)")
        print("2. Pixel calibration (detection accuracy)")
        print("3. Complete calibration (both)")
        print("4. Return to detection")
        
        choice = input("Enter choice (1-4): ")
        
        if choice == '1':
            self.collect_range_calibration_data()
        elif choice == '2':
            self.collect_pixel_calibration_data()
        elif choice == '3':
            self.collect_range_calibration_data()
            self.collect_pixel_calibration_data()
            self.process_calibration_results()
        elif choice == '4':
            self.run_detection()
        else:
            print("Invalid choice")
            self.run_calibration_mode()
    
    def collect_range_calibration_data(self):
        """
        Collect range calibration data using pose estimation
        """
        print("\n=== RANGE CALIBRATION ===")
        print("1. Place ArUco marker at known distances")
        print("2. Measure actual distance with tape measure")
        print("3. Press 's' to save measurement when marker is detected")
        print("4. Press 'q' to finish, 'r' to restart")
        print("5. Collect at: 0.5m, 1m, 1.5m, 2m, 3m, 4m, 5m")
        
        while True:
            ret, image = self.cap.read()
            if not ret:
                break
            
            corners, ids, rvecs, tvecs, distances, marker_pixel_sizes = self.detect_markers_with_pose(image)
            
            image_copy = image.copy()
            if ids is not None and len(ids) > 0:
                cv2.aruco.drawDetectedMarkers(image_copy, corners, ids)
                
                for i in range(len(ids)):
                    if i < len(distances):
                        # Draw distance info
                        distance_text = f"ID{ids[i][0]}: {distances[i]:.3f}m"
                        cv2.putText(image_copy, distance_text, 
                                   (int(corners[i][0][0][0]), int(corners[i][0][0][1]) - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Draw pose axes
                        if i < len(rvecs) and i < len(tvecs):
                            cv2.drawFrameAxes(image_copy, self.camera_matrix, self.dist_coeffs, 
                                              rvecs[i], tvecs[i], self.marker_length * 1.5, 3)
            else:
                cv2.putText(image_copy, "No markers detected", 
                           (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display instructions
            cv2.putText(image_copy, f"Samples collected: {len(self.range_measurements)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image_copy, "Press 's' to save, 'q' to quit", 
                       (10, image_copy.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Range Calibration", image_copy)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') and ids is not None and len(ids) > 0:
                try:
                    # Use first detected marker
                    estimated_distance = distances[0]
                    marker_id = ids[0][0]
                    
                    actual_distance = float(input(f"\nMarker ID {marker_id} - Estimated: {estimated_distance:.3f}m\nEnter ACTUAL distance (meters): "))
                    
                    print("Collecting 30 measurements... keep marker steady!")
                    measurements = []
                    pixel_sizes = []
                    
                    for i in range(30):
                        ret, img = self.cap.read()
                        if ret:
                            corners_temp, ids_temp, rvecs_temp, tvecs_temp, distances_temp, pixel_sizes_temp = self.detect_markers_with_pose(img)
                            if ids_temp is not None and len(ids_temp) > 0:
                                # Find the same marker ID
                                for j, detected_id in enumerate(ids_temp):
                                    if detected_id[0] == marker_id and j < len(distances_temp):
                                        measurements.append(distances_temp[j])
                                        if j < len(pixel_sizes_temp):
                                            pixel_sizes.append(pixel_sizes_temp[j])
                                        break
                    
                    if len(measurements) >= 15:
                        measured_mean = np.mean(measurements)
                        measured_std = np.std(measurements)
                        avg_pixel_size = np.mean(pixel_sizes) if pixel_sizes else 0
                        
                        self.range_measurements.append({
                            'marker_id': int(marker_id),
                            'actual_distance': float(actual_distance),
                            'measured_distance': float(measured_mean),
                            'measurement_std': float(measured_std),
                            'avg_pixel_size': float(avg_pixel_size),
                            'num_samples': len(measurements),
                            'raw_measurements': [float(x) for x in measurements]
                        })
                        
                        print(f"Saved: {actual_distance}m actual, {measured_mean:.3f}±{measured_std:.4f}m measured")
                        print(f"Error: {abs(actual_distance - measured_mean):.3f}m ({100*abs(actual_distance - measured_mean)/actual_distance:.1f}%)")
                        print(f"Pixel size: {avg_pixel_size:.1f}px")
                    else:
                        print("Not enough valid measurements - try again")
                        
                except ValueError:
                    print("Invalid input - enter a number")
                    
            elif key == ord('q'):
                break
            elif key == ord('r'):
                self.range_measurements = []
                print("Range measurements cleared")
        
        cv2.destroyAllWindows()
        print(f"Range calibration complete. Collected {len(self.range_measurements)} measurements.")
    
    def collect_pixel_calibration_data(self):
        """
        Collect pixel accuracy calibration data with different marker sizes
        """
        print("\n=== PIXEL ACCURACY CALIBRATION ===")
        print("IMPORTANT: Collect measurements at DIFFERENT DISTANCES to vary pixel size:")
        print("- 0.5m distance → ~260 pixel marker")
        print("- 1.0m distance → ~130 pixel marker")
        print("- 2.0m distance → ~65 pixel marker")
        print("- 3.0m distance → ~43 pixel marker")
        print("- 4.0m distance → ~32 pixel marker")
        print("Press 's' to collect at current distance, 'q' to finish")
        print("Recommended: Collect at least 3 different distances")
        
        distances_collected = []
        
        while True:
            ret, image = self.cap.read()
            if not ret:
                break
            
            corners, ids, rvecs, tvecs, distances, marker_pixel_sizes = self.detect_markers_with_pose(image)
            
            image_copy = image.copy()
            if ids is not None and len(ids) > 0:
                cv2.aruco.drawDetectedMarkers(image_copy, corners, ids)
                
                # Show center and corners
                for i, corner_set in enumerate(corners):
                    corner_points = corner_set.reshape(-1, 2)
                    center = np.mean(corner_points, axis=0)
                    
                    # Draw center
                    cv2.circle(image_copy, tuple(center.astype(int)), 5, (0, 255, 0), -1)
                    
                    # Draw corner indices
                    for j, corner in enumerate(corner_points):
                        cv2.circle(image_copy, tuple(corner.astype(int)), 3, (255, 0, 0), -1)
                        cv2.putText(image_copy, str(j), tuple(corner.astype(int) + 5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    
                    if i < len(marker_pixel_sizes) and i < len(distances):
                        pixel_size = marker_pixel_sizes[i]
                        distance = distances[i]
                        cv2.putText(image_copy, f"Distance: {distance:.2f}m, Size: {pixel_size:.1f}px", 
                                   (10, 30 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show collected distances
            y_offset = 100
            cv2.putText(image_copy, "Collected at distances:", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            for dist_info in distances_collected:
                y_offset += 20
                cv2.putText(image_copy, f"  {dist_info:.2f}m", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.putText(image_copy, "Press 's' to collect at this distance", 
                       (10, image_copy.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(image_copy, f"Total samples: {len(self.pixel_measurements)}", 
                       (10, image_copy.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow("Pixel Calibration", image_copy)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') and ids is not None and len(ids) > 0:
                current_distance = distances[0] if distances else 0
                print(f"\nCollecting at {current_distance:.2f}m distance...")
                print("IMPORTANT: Keep camera and marker PERFECTLY STILL!")
                cv2.waitKey(1000)  # Give user time to stabilize
                
                print("Collecting 100 pixel measurements...")
                center_positions = []
                corner_positions = []
                pixel_sizes_collected = []
                successful_detections = 0
                
                for i in range(100):
                    ret, img = self.cap.read()
                    if ret:
                        corners_temp, ids_temp, _, _, _, pixel_sizes_temp = self.detect_markers_with_pose(img)
                        if ids_temp is not None and len(ids_temp) > 0:
                            # Use first marker
                            corner_points = corners_temp[0].reshape(-1, 2)
                            center = np.mean(corner_points, axis=0)
                            center_positions.append(center)
                            corner_positions.append(corner_points)
                            if len(pixel_sizes_temp) > 0:
                                pixel_sizes_collected.append(pixel_sizes_temp[0])
                            successful_detections += 1
                
                if successful_detections >= 50:
                    centers = np.array(center_positions)
                    center_std_x = np.std(centers[:, 0])
                    center_std_y = np.std(centers[:, 1])
                    center_std_combined = np.sqrt(center_std_x**2 + center_std_y**2)
                    
                    # Analyze individual corner accuracy
                    corners_array = np.array(corner_positions)
                    corner_stds = []
                    for corner_idx in range(4):
                        corner_x_std = np.std(corners_array[:, corner_idx, 0])
                        corner_y_std = np.std(corners_array[:, corner_idx, 1])
                        corner_combined_std = np.sqrt(corner_x_std**2 + corner_y_std**2)
                        corner_stds.append(corner_combined_std)
                    
                    avg_corner_std = np.mean(corner_stds)
                    avg_pixel_size = np.mean(pixel_sizes_collected) if pixel_sizes_collected else 0
                    
                    # Ensure we have actual variation (not perfectly stable)
                    if center_std_combined < 0.01:  # Less than 0.01 pixel variation is suspicious
                        print("WARNING: Extremely low variation detected. Ensure natural camera shake is present.")
                        center_std_combined = max(center_std_combined, 0.1)  # Set minimum reasonable value
                        avg_corner_std = max(avg_corner_std, 0.1)
                    
                    result = {
                        'distance': float(current_distance),
                        'center_std_x': float(center_std_x),
                        'center_std_y': float(center_std_y),
                        'center_std_combined': float(center_std_combined),
                        'corner_stds': [float(x) for x in corner_stds],
                        'avg_corner_std': float(avg_corner_std),
                        'avg_pixel_size': float(avg_pixel_size),
                        'num_samples': successful_detections
                    }
                    
                    self.pixel_measurements.append(result)
                    distances_collected.append(current_distance)
                    
                    print(f"Distance: {current_distance:.2f}m")
                    print(f"Center detection accuracy: σ_u = {center_std_combined:.3f} pixels")
                    print(f"Corner detection accuracy: σ_u = {avg_corner_std:.3f} pixels")
                    print(f"Marker pixel size: {avg_pixel_size:.1f} pixels")
                    print(f"Detection rate: {successful_detections}/100")
                    
                    if len(self.pixel_measurements) >= 3:
                        print("\nYou have enough data for bearing model fitting.")
                        print("Press 'q' to finish or continue collecting at other distances.")
                else:
                    print(f"Not enough detections ({successful_detections}/100), try again")
                    
            elif key == ord('q'):
                if len(self.pixel_measurements) < 2:
                    print("\nWARNING: Need at least 2 measurements at different distances!")
                    print("Continue? (y/n)")
                    if cv2.waitKey(0) & 0xFF != ord('y'):
                        continue
                break
        
        cv2.destroyAllWindows()
        print(f"Pixel calibration complete. Collected {len(self.pixel_measurements)} measurements.")
    
    def fit_range_error_model(self):
        """
        Fit σ_r(d) = α·d² + β model to range measurements
        """
        if len(self.range_measurements) < 3:
            print("Need at least 3 range measurements")
            return None
        
        distances = [m['actual_distance'] for m in self.range_measurements]
        stds = [m['measurement_std'] for m in self.range_measurements]
        
        # Fit quadratic model: σ_r(d) = α·d² + β
        def quadratic_model(d, alpha, beta):
            return alpha * d**2 + beta
        
        try:
            # Provide reasonable initial guesses
            p0 = [0.001, 0.001]  # Initial guess for alpha and beta
            popt, pcov = optimize.curve_fit(quadratic_model, distances, stds, p0=p0)
            alpha, beta = popt
            
            # Calculate goodness of fit
            predicted_stds = quadratic_model(np.array(distances), alpha, beta)
            ss_res = np.sum((np.array(stds) - predicted_stds) ** 2)
            ss_tot = np.sum((np.array(stds) - np.mean(stds)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            return {
                'alpha': float(alpha),
                'beta': float(beta),
                'r_squared': float(r_squared),
                'covariance': [[float(pcov[i,j]) for j in range(pcov.shape[1])] for i in range(pcov.shape[0])]
            }
        except Exception as e:
            print(f"Error fitting range model: {e}")
            return None
    
    def fit_bearing_error_model(self):
        """
        Fit σ_θ = α + β/s model to pixel measurements
        """
        if len(self.pixel_measurements) < 2:
            print("Need at least 2 pixel measurements with different marker sizes")
            return None
        
        # Extract data
        pixel_sizes = []
        angular_stds = []
        
        for measurement in self.pixel_measurements:
            if measurement['avg_pixel_size'] > 0 and measurement['center_std_combined'] > 0:
                pixel_sizes.append(measurement['avg_pixel_size'])
                # Convert pixel std to angular std: σ_θ ≈ σ_u / f
                sigma_u = measurement['center_std_combined']
                sigma_theta = sigma_u / self.focal_length
                angular_stds.append(sigma_theta)
        
        if len(pixel_sizes) < 2:
            print("Need measurements with different marker pixel sizes")
            return None
        
        print(f"\nBearing model fitting data:")
        print(f"Pixel sizes: {pixel_sizes}")
        print(f"Angular stds (rad): {angular_stds}")
        print(f"Angular stds (deg): {[np.degrees(x) for x in angular_stds]}")
        
        # Fit model: σ_θ = α + β/s
        def bearing_model(s, alpha, beta):
            return alpha + beta / s
        
        try:
            # Provide reasonable initial guesses
            p0 = [min(angular_stds), 0.01]
            popt, pcov = optimize.curve_fit(bearing_model, pixel_sizes, angular_stds, p0=p0)
            alpha, beta = popt
            
            # Calculate goodness of fit
            predicted_stds = bearing_model(np.array(pixel_sizes), alpha, beta)
            ss_res = np.sum((np.array(angular_stds) - predicted_stds) ** 2)
            ss_tot = np.sum((np.array(angular_stds) - np.mean(angular_stds)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            return {
                'alpha': float(alpha),
                'beta': float(beta),
                'r_squared': float(r_squared),
                'covariance': [[float(pcov[i,j]) for j in range(pcov.shape[1])] for i in range(pcov.shape[0])],
                'data_points': {
                    'pixel_sizes': [float(x) for x in pixel_sizes],
                    'angular_stds_rad': [float(x) for x in angular_stds],
                    'angular_stds_deg': [float(np.degrees(x)) for x in angular_stds]
                }
            }
        except Exception as e:
            print(f"Error fitting bearing model: {e}")
            # Fall back to average if fitting fails
            if len(angular_stds) > 0:
                avg_angular_std = np.mean(angular_stds)
                return {
                    'alpha': float(avg_angular_std),
                    'beta': 0.0,
                    'r_squared': 0.0,
                    'covariance': [[0.0, 0.0], [0.0, 0.0]],
                    'data_points': {
                        'pixel_sizes': [float(x) for x in pixel_sizes],
                        'angular_stds_rad': [float(x) for x in angular_stds],
                        'angular_stds_deg': [float(np.degrees(x)) for x in angular_stds]
                    }
                }
            return None
    
    def process_calibration_results(self):
        """
        Process and save calibration results
        """
        print("\n" + "="*50)
        print("PROCESSING CALIBRATION RESULTS")
        print("="*50)
        
        # Fit range model
        range_params = None
        if self.range_measurements:
            range_params = self.fit_range_error_model()
            if range_params:
                print(f"\nRange Error Model: σ_r(d) = {range_params['alpha']:.6f}·d² + {range_params['beta']:.6f}")
                print(f"R² = {range_params['r_squared']:.3f}")
                print(f"α = {range_params['alpha']:.6f} (quadratic coefficient)")
                print(f"β = {range_params['beta']:.6f} (constant offset)")
                
                # Check for negative values at typical distances
                test_distances = [0.5, 1.0, 2.0, 3.0, 5.0]
                print("\nPredicted range errors:")
                for d in test_distances:
                    sigma_r = range_params['alpha'] * d**2 + range_params['beta']
                    print(f"  At {d}m: σ_r = {sigma_r:.4f}m")
                    if sigma_r < 0:
                        print(f"    WARNING: Negative error at {d}m!")
        
        # Fit bearing model
        bearing_params = None
        if self.pixel_measurements:
            bearing_params = self.fit_bearing_error_model()
            if bearing_params:
                print(f"\nBearing Error Model: σ_θ = {bearing_params['alpha']:.6f} + {bearing_params['beta']:.6f}/s")
                print(f"R² = {bearing_params['r_squared']:.3f}")
                print(f"α = {bearing_params['alpha']:.6f} rad = {np.degrees(bearing_params['alpha']):.3f}° (constant angular error)")
                print(f"β = {bearing_params['beta']:.6f} (pixel-dependent component)")
                if 'data_points' in bearing_params:
                    avg_angular_error_deg = np.mean(bearing_params['data_points']['angular_stds_deg'])
                    print(f"Average angular error: {avg_angular_error_deg:.3f}°")
        
        # Save results
        if range_params or bearing_params:
            self.save_calibration_results(range_params, bearing_params)
        
        # Plot results if we have data
        if (range_params and len(self.range_measurements) >= 3) or (bearing_params and len(self.pixel_measurements) >= 2):
            self.plot_calibration_results(range_params, bearing_params)
    
    def save_calibration_results(self, range_params, bearing_params):
        """
        Save calibration results for ROS integration
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'camera_setup': {
                'model': 'Logitech C505',
                'resolution': [self.image_width, self.image_height],
                'focal_length_pixels': float(self.focal_length),
                'camera_matrix': [[float(self.camera_matrix[i,j]) for j in range(self.camera_matrix.shape[1])] 
                                 for i in range(self.camera_matrix.shape[0])],
                'dist_coeffs': [float(x) for x in self.dist_coeffs.flatten()]
            },
            'aruco_setup': {
                'marker_size_m': float(self.marker_length),
                'dictionary': str(self.dictionary_id)
            },
            'calibration_results': {
                'range_model': range_params,
                'bearing_model': bearing_params
            },
            'usage_example': {
                'python_code': self.generate_usage_code(range_params, bearing_params)
            },
            'raw_data': {
                'range_measurements': self.range_measurements,
                'pixel_measurements': self.pixel_measurements
            }
        }
        
        filename = f'aruco_calibration_{self.image_width}x{self.image_height}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nCalibration results saved to: {filename}")
        except Exception as e:
            print(f"Error saving calibration results: {e}")
    
    def generate_usage_code(self, range_params, bearing_params):
        """
        Generate Python code for using the calibration results
        """
        code = "import numpy as np\n\n"
        code += "def get_measurement_covariance_matrix(distance_meters, marker_pixel_size):\n"
        code += "    \"\"\"\n"
        code += "    Calculate measurement covariance matrix for range-bearing measurements\n"
        code += "    Args: \n"
        code += "        distance_meters - estimated distance to target\n"
        code += "        marker_pixel_size - size of marker in pixels (average side length)\n"
        code += "    Returns:\n"
        code += "        2x2 covariance matrix [[σ_r², 0], [0, σ_θ²]]\n"
        code += "    \"\"\"\n"
        
        if range_params:
            code += f"    # Range error model: σ_r(d) = {range_params['alpha']:.6f}·d² + {range_params['beta']:.6f}\n"
            code += f"    sigma_r = {range_params['alpha']:.6f} * distance_meters**2 + {range_params['beta']:.6f}\n"
            code += "    # Ensure positive value (minimum 1mm error)\n"
            code += "    sigma_r = max(sigma_r, 0.001)\n"
        else:
            code += "    # Default range error (no calibration data)\n"
            code += "    sigma_r = 0.01 * distance_meters  # 1% of distance\n"
        
        if bearing_params:
            code += f"    \n"
            code += f"    # Bearing error model: σ_θ = {bearing_params['alpha']:.6f} + {bearing_params['beta']:.6f}/s\n"
            code += f"    sigma_theta = {bearing_params['alpha']:.6f} + {bearing_params['beta']:.6f} / marker_pixel_size\n"
            code += "    # Ensure positive value\n"
            code += "    sigma_theta = max(sigma_theta, 1e-6)\n"
        else:
            code += "    \n"
            code += "    # Default bearing error (no calibration data)\n"
            code += f"    focal_length = {self.focal_length:.1f}  # pixels\n"
            code += "    sigma_u = 0.5  # pixel error\n"
            code += "    sigma_theta = sigma_u / focal_length  # radians\n"
        
        code += "    \n"
        code += "    # Create covariance matrix\n"
        code += "    R = np.array([[sigma_r**2, 0.0],\n"
        code += "                  [0.0, sigma_theta**2]])\n"
        code += "    return R\n\n"
        code += "# Example usage:\n"
        code += "# R = get_measurement_covariance_matrix(2.5, 45.0)\n"
        code += "# print('Measurement covariance matrix:')\n"
        code += "# print(f'Range variance: {R[0,0]:.6f} m²')\n"
        code += "# print(f'Bearing variance: {R[1,1]:.9f} rad²')\n"
        
        return code
    
    def plot_calibration_results(self, range_params, bearing_params):
        """
        Plot calibration results
        """
        n_plots = sum([range_params is not None, bearing_params is not None])
        if n_plots == 0:
            return
        
        fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
        if n_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # Plot range calibration
        if range_params and self.range_measurements:
            ax = axes[plot_idx]
            plot_idx += 1
            
            distances = [m['actual_distance'] for m in self.range_measurements]
            measured_distances = [m['measured_distance'] for m in self.range_measurements]
            stds = [m['measurement_std'] for m in self.range_measurements]
            
            # Plot actual vs measured
            ax.scatter(distances, measured_distances, alpha=0.7, label='Measurements', s=60)
            ax.plot([0, max(distances)*1.1], [0, max(distances)*1.1], 'r--', label='Perfect accuracy')
            
            # Add error bars
            ax.errorbar(distances, measured_distances, yerr=stds, fmt='o', alpha=0.5, capsize=5)
            
            ax.set_xlabel('Actual Distance (m)')
            ax.set_ylabel('Measured Distance (m)')
            ax.set_title('Range Calibration Results')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add fitted model curve on secondary y-axis
            d_fit = np.linspace(0, max(distances)*1.1, 100)
            sigma_fit = range_params['alpha'] * d_fit**2 + range_params['beta']
            sigma_fit = np.maximum(sigma_fit, 0)  # Ensure non-negative
            
            ax_twin = ax.twinx()
            ax_twin.plot(d_fit, sigma_fit*1000, 'g-', linewidth=2, 
                        label=f'σ_r(d) = {range_params["alpha"]:.4f}d² + {range_params["beta"]:.4f}')
            ax_twin.set_ylabel('Range Error σ_r (mm)', color='g')
            ax_twin.tick_params(axis='y', labelcolor='g')
            ax_twin.legend(loc='upper left')
        
        # Plot bearing calibration
        if bearing_params and 'data_points' in bearing_params and len(bearing_params['data_points']['pixel_sizes']) > 0:
            ax = axes[plot_idx]
            pixel_sizes = bearing_params['data_points']['pixel_sizes']
            angular_errors_deg = bearing_params['data_points']['angular_stds_deg']
            
            ax.scatter(pixel_sizes, angular_errors_deg, alpha=0.7, s=80, label='Measurements')
            
            # Plot fitted model
            if len(pixel_sizes) > 1:
                s_fit = np.linspace(min(pixel_sizes)*0.8, max(pixel_sizes)*1.2, 100)
                theta_fit = np.degrees(bearing_params['alpha'] + bearing_params['beta'] / s_fit)
                ax.plot(s_fit, theta_fit, 'r-', linewidth=2,
                        label=f'σ_θ = {bearing_params["alpha"]:.4f} + {bearing_params["beta"]:.4f}/s')
            
            ax.set_xlabel('Marker Size (pixels)')
            ax.set_ylabel('Angular Error (degrees)')
            ax.set_title('Bearing Calibration Results')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add secondary axis for radians
            ax_twin = ax.twinx()
            ax_twin.set_ylabel('Angular Error (radians)')
            ax_twin.set_ylim(np.radians(ax.get_ylim()[0]), np.radians(ax.get_ylim()[1]))
        
        plt.tight_layout()
        filename = f'calibration_plots_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(filename, dpi=300)
        print(f"\nCalibration plots saved to: {filename}")
        plt.show()
    
    def __del__(self):
        """
        Cleanup resources
        """
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

def load_calibration(calibration_file):
    """Load camera calibration parameters from a .npz file"""
    print(f"Loading calibration from: {calibration_file}")
    try:
        data = np.load(calibration_file)
        camera_matrix = data['camera_matrix']
        dist_coeffs = data['dist_coeffs']
        print("Calibration loaded successfully!")
        print(f"Camera matrix:\n{camera_matrix}")
        print(f"Distortion coefficients:\n{dist_coeffs}")
        return camera_matrix, dist_coeffs
    except Exception as e:
        print(f"Error loading calibration file: {e}")
        exit(1)

if __name__ == "__main__":
   # Load camera calibration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    calibration_path = os.path.join(os.path.dirname(script_dir), "calibration", "camera_calibration.npz")
    
    camera_matrix, dist_coeffs = load_calibration(calibration_path)

    # Initialize detector
    detector = EnhancedArucoDetector(
        camera_id=0,
        dictionary_id=cv2.aruco.DICT_5X5_100,
        marker_length=0.16,  # 16cm marker
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs
    )

    # Run detection
    try:
        detector.run_detection()
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        del detector