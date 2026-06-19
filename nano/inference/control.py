import numpy as np
import cv2

class LaneController:
    """Processes segmentation masks, fits lanes, and computes steering commands."""
    def __init__(self, img_width=320, img_height=180, look_ahead_ratio=0.6, max_steer_angle=30.0):
        self.width = img_width
        self.height = img_height
        self.max_steer_angle = max_steer_angle
        
        # Look-ahead line height (y-coordinate)
        # Measured from the top of the image (0) to the bottom (height)
        self.look_ahead_y = int(self.height * look_ahead_ratio)
        
        # Default lane width in pixel space at bottom of screen
        self.nominal_lane_width = 200.0
        self.half_lane_width = self.nominal_lane_width / 2.0
        
        # Keep track of previous polynomial coefficients for smoothing / temporal consistency
        self.left_fit = None
        self.right_fit = None
        
        # Low-pass filter coefficient (0 = no update, 1 = instant update)
        self.alpha_smooth = 0.7

    def fit_lanes(self, mask):
        """
        Fits 2nd-order polynomials to left and right lanes.
        Input: Binary mask (180x320) where 255 = lane.
        """
        # Crop mask to road region (ignore top horizon/background)
        # Only process rows below y=70 (typical road horizon in our synthetic model)
        road_mask = mask.copy()
        road_mask[0:70, :] = 0
        
        # Extract coordinates of lane pixels
        y_indices, x_indices = np.where(road_mask > 127)
        
        left_x, left_y = [], []
        right_x, right_y = [], []
        
        # Split pixels into left and right halves
        center_x = self.width // 2
        for x, y in zip(x_indices, y_indices):
            if x < center_x:
                left_x.append(x)
                left_y.append(y)
            else:
                right_x.append(x)
                right_y.append(y)
                
        # Fit 2nd-order polynomial: x = a*y^2 + b*y + c
        # We fit x as function of y because lane lines are close to vertical
        fit_left_curr = None
        fit_right_curr = None
        
        if len(left_y) > 30: # Need enough points to fit
            try:
                fit_left_curr = np.polyfit(left_y, left_x, 2)
            except np.RankWarning:
                pass
                
        if len(right_y) > 30:
            try:
                fit_right_curr = np.polyfit(right_y, right_x, 2)
            except np.RankWarning:
                pass
                
        # Temporal smoothing
        if fit_left_curr is not None:
            if self.left_fit is None:
                self.left_fit = fit_left_curr
            else:
                self.left_fit = self.alpha_smooth * fit_left_curr + (1 - self.alpha_smooth) * self.left_fit
        
        if fit_right_curr is not None:
            if self.right_fit is None:
                self.right_fit = fit_right_curr
            else:
                self.right_fit = self.alpha_smooth * fit_right_curr + (1 - self.alpha_smooth) * self.right_fit
                
        return self.left_fit is not None, self.right_fit is not None

    def calculate_steering(self):
        """
        Computes steering angle based on fitted lanes and look-ahead distance.
        Returns: target_steering_angle (float in degrees), target_x (pixel), status string
        """
        img_center_x = self.width / 2.0
        car_bottom_y = float(self.height)
        
        x_left_target = None
        x_right_target = None
        
        # Evaluate polynomials at the look-ahead row
        y = self.look_ahead_y
        
        if self.left_fit is not None:
            x_left_target = self.left_fit[0] * (y**2) + self.left_fit[1] * y + self.left_fit[2]
            # Sanity limit to prevent outlier divergence
            x_left_target = np.clip(x_left_target, -50, img_center_x)
            
        if self.right_fit is not None:
            x_right_target = self.right_fit[0] * (y**2) + self.right_fit[1] * y + self.right_fit[2]
            # Sanity limit
            x_right_target = np.clip(x_right_target, img_center_x, self.width + 50)
            
        # Determine lane center target point
        status = "No lanes"
        if x_left_target is not None and x_right_target is not None:
            # Both lanes detected
            target_x = (x_left_target + x_right_target) / 2.0
            status = "Dual lanes"
        elif x_left_target is not None:
            # Only left lane detected: project right boundary
            target_x = x_left_target + self.half_lane_width
            status = "Left lane only"
        elif x_right_target is not None:
            # Only right lane detected: project left boundary
            target_x = x_right_target - self.half_lane_width
            status = "Right lane only"
        else:
            # No lanes detected: default to center-steer
            target_x = img_center_x
            status = "Lost - Blind steer"
            
        # Calculate geometric heading angle to target point
        # Center bottom of image is the car's current position: (img_center_x, car_bottom_y)
        dx = target_x - img_center_x
        dy = car_bottom_y - y
        
        # heading_error in radians
        heading_error = np.arctan2(dx, dy)
        
        # Convert to degrees and scale with proportional gain
        # Gain can be adjusted depending on lateral speed and camera field-of-view
        kp = 1.0
        steering_angle = np.degrees(heading_error) * kp
        
        # Apply mechanical hard limit
        steering_angle = np.clip(steering_angle, -self.max_steer_angle, self.max_steer_angle)
        
        return float(steering_angle), int(target_x), status

    def draw_overlay(self, image, mask):
        """Generates a visualization overlay on the original frame."""
        overlay = image.copy()
        
        # Draw lane mask in green with transparency
        color_mask = np.zeros_like(image)
        color_mask[mask > 127] = [0, 255, 0]
        overlay = cv2.addWeighted(overlay, 1.0, color_mask, 0.4, 0)
        
        y_range = np.linspace(70, self.height - 1, 30)
        
        # Draw left polynomial curve (Blue)
        if self.left_fit is not None:
            pts_left = np.array([
                [int(self.left_fit[0] * (y**2) + self.left_fit[1] * y + self.left_fit[2]), int(y)]
                for y in y_range
            ], dtype=np.int32)
            pts_left = pts_left[(pts_left[:, 0] >= 0) & (pts_left[:, 0] < self.width)]
            if len(pts_left) > 1:
                cv2.polylines(overlay, [pts_left], isClosed=False, color=(255, 0, 0), thickness=2)
                
        # Draw right polynomial curve (Red)
        if self.right_fit is not None:
            pts_right = np.array([
                [int(self.right_fit[0] * (y**2) + self.right_fit[1] * y + self.right_fit[2]), int(y)]
                for y in y_range
            ], dtype=np.int32)
            pts_right = pts_right[(pts_right[:, 0] >= 0) & (pts_right[:, 0] < self.width)]
            if len(pts_right) > 1:
                cv2.polylines(overlay, [pts_right], isClosed=False, color=(0, 0, 255), thickness=2)
                
        # Compute current steering values for drawing
        steer, target_x, status = self.calculate_steering()
        
        # Draw look-ahead line (Yellow dashed-like line)
        cv2.line(overlay, (0, self.look_ahead_y), (self.width, self.look_ahead_y), (0, 255, 255), 1)
        
        # Draw target point (Yellow dot)
        cv2.circle(overlay, (target_x, self.look_ahead_y), 5, (0, 255, 255), -1)
        
        # Draw heading direction vector (Yellow line)
        cv2.line(overlay, (self.width // 2, self.height), (target_x, self.look_ahead_y), (0, 255, 255), 2)
        
        # Draw text readouts
        cv2.putText(overlay, f"Steer: {steer:.2f} deg", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(overlay, f"Status: {status}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        return overlay

if __name__ == "__main__":
    # Test controller mechanics
    controller = LaneController()
    # Create simple binary mask representing direct straight lines
    mask = np.zeros((180, 320), dtype=np.uint8)
    cv2.line(mask, (60, 180), (140, 70), 255, 4)
    cv2.line(mask, (260, 180), (180, 70), 255, 4)
    
    left_ok, right_ok = controller.fit_lanes(mask)
    steer, target_x, status = controller.calculate_steering()
    
    print(f"Lanes fitted - Left: {left_ok}, Right: {right_ok}")
    print(f"Computed steer angle: {steer:.2f} degrees (Target X: {target_x}, Status: {status})")
    assert abs(steer) < 5.0, "Steer angle for straight lane should be near 0!"
    print("LaneController verification test passed successfully.")
