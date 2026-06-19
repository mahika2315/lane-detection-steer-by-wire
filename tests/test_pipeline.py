import sys
import os
import numpy as np
import cv2

# Add the nano/inference folder to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'nano', 'inference')))

from control import LaneController
from serial_tx import SerialTX

def generate_test_lane_mask(curve_type="straight"):
    """Generates synthetic binary lane masks for testing."""
    mask = np.zeros((180, 320), dtype=np.uint8)
    
    # Bottom lane points (starting points)
    l_start_x = 60
    r_start_x = 260
    
    # Determine offset points based on curvature
    if curve_type == "straight":
        l_end_x = 130
        r_end_x = 190
    elif curve_type == "left":
        l_end_x = 90
        r_end_x = 150
    elif curve_type == "right":
        l_end_x = 170
        r_end_x = 230
    else:
        raise ValueError(f"Unknown curve type: {curve_type}")
        
    # Draw simple lines to represent lanes
    cv2.line(mask, (l_start_x, 180), (l_end_x, 70), 255, 4)
    cv2.line(mask, (r_start_x, 180), (r_end_x, 70), 255, 4)
    
    return mask

def test_straight_lane():
    print("\n--- Running Test: Straight Lane ---")
    controller = LaneController()
    mask = generate_test_lane_mask("straight")
    
    left_ok, right_ok = controller.fit_lanes(mask)
    steer, target_x, status = controller.calculate_steering()
    
    print(f"Lanes fitted - Left: {left_ok}, Right: {right_ok}")
    print(f"Computed steer angle: {steer:.2f} deg | Target X: {target_x} | Status: {status}")
    
    assert left_ok and right_ok, "Failing to fit straight lanes!"
    assert abs(steer) < 5.0, f"Straight lane should have near zero steering, got {steer:.2f}"
    print("Test passed.")

def test_left_lane():
    print("\n--- Running Test: Left Curve ---")
    controller = LaneController()
    mask = generate_test_lane_mask("left")
    
    left_ok, right_ok = controller.fit_lanes(mask)
    steer, target_x, status = controller.calculate_steering()
    
    print(f"Lanes fitted - Left: {left_ok}, Right: {right_ok}")
    print(f"Computed steer angle: {steer:.2f} deg | Target X: {target_x} | Status: {status}")
    
    assert left_ok and right_ok, "Failing to fit left lanes!"
    assert steer < -5.0, f"Left-turning lane should have negative steering, got {steer:.2f}"
    print("Test passed.")

def test_right_lane():
    print("\n--- Running Test: Right Curve ---")
    controller = LaneController()
    mask = generate_test_lane_mask("right")
    
    left_ok, right_ok = controller.fit_lanes(mask)
    steer, target_x, status = controller.calculate_steering()
    
    print(f"Lanes fitted - Left: {left_ok}, Right: {right_ok}")
    print(f"Computed steer angle: {steer:.2f} deg | Target X: {target_x} | Status: {status}")
    
    assert left_ok and right_ok, "Failing to fit right lanes!"
    assert steer > 5.0, f"Right-turning lane should have positive steering, got {steer:.2f}"
    print("Test passed.")

def test_serial_packaging():
    print("\n--- Running Test: Serial Packaging ---")
    tx = SerialTX(mock=True)
    test_angle = -14.25
    packet = tx.send_steering_angle(test_angle)
    
    # Check start bytes
    assert packet[0] == 0xAA and packet[1] == 0x55, "Incorrect start bytes!"
    # Check Command ID
    assert packet[2] == 0x10, "Incorrect Command ID!"
    # Check Data Length
    assert packet[3] == 0x04, "Incorrect Payload Length!"
    
    # Check Checksum: XOR of indices 2 through 7 should match index 8
    calculated_cs = packet[2] ^ packet[3]
    for i in range(4, 8):
        calculated_cs ^= packet[i]
    assert packet[8] == calculated_cs, f"Checksum mismatch: expected {packet[8]}, calculated {calculated_cs}"
    
    print("Serial packaging matches protocol specs. Test passed.")

def run_all_tests():
    print("Starting Lane Detection and Controller Integration Tests...")
    try:
        test_straight_lane()
        test_left_lane()
        test_right_lane()
        test_serial_packaging()
        print("\nAll integration tests passed successfully!")
    except AssertionError as e:
        print(f"\nTEST FAILURE: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_all_tests()
