import cv2
import numpy as np

def create_synthetic_video(filename="tests/test_video.mp4", num_frames=100):
    # Setup VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, 20.0, (320, 180))
    
    print(f"Generating synthetic road video at {filename}...")
    
    for i in range(num_frames):
        # Create road background (grayish)
        img = np.ones((180, 320, 3), dtype=np.uint8) * 40
        img[0:70, :] = [120, 100, 90] # sky
        
        # Road polygon
        road_pts = np.array([[0, 180], [130, 70], [190, 70], [320, 180]], dtype=np.int32)
        cv2.fillPoly(img, [road_pts], (80, 80, 80))
        
        # Simulate curves using sine wave
        curve_offset = np.sin(i / 10.0) * 35.0
        
        # Lane lines curves
        pts = np.linspace(0, 1, 20)
        
        l_start = (int(40 + curve_offset), 180)
        l_end = (int(145 + curve_offset), 70)
        l_control = (int(90 + curve_offset), 125)
        
        r_start = (int(280 + curve_offset), 180)
        r_end = (int(175 + curve_offset), 70)
        r_control = (int(230 + curve_offset), 125)
        
        l_curve_pts = np.array([
            (1 - t)**2 * np.array(l_start) + 2 * (1 - t) * t * np.array(l_control) + t**2 * np.array(l_end)
            for t in pts
        ], dtype=np.int32)
        
        r_curve_pts = np.array([
            (1 - t)**2 * np.array(r_start) + 2 * (1 - t) * t * np.array(r_control) + t**2 * np.array(r_end)
            for t in pts
        ], dtype=np.int32)
        
        # Draw white lanes
        cv2.polylines(img, [l_curve_pts], isClosed=False, color=(255, 255, 255), thickness=4)
        cv2.polylines(img, [r_curve_pts], isClosed=False, color=(255, 255, 255), thickness=4)
        
        # Add slight Gaussian noise
        noise = np.random.randint(-10, 10, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        out.write(img)
        
    out.release()
    print("Video generation complete.")

if __name__ == "__main__":
    create_synthetic_video()
