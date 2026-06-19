import os
import time
import argparse
import cv2
import numpy as np
import onnxruntime as ort

from serial_tx import SerialTX
from control import LaneController

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280,
    capture_height=720,
    display_width=320,
    display_height=180,
    framerate=30,
    flip_method=0,
):
    """Generates an optimized GStreamer camera capture pipeline for CSI cameras on Jetson Nano."""
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def main():
    parser = argparse.ArgumentParser(description="Real-Time Lane Detection and Steering Controller for Jetson Nano")
    parser.add_argument("--model", type=str, default=None, help="Path to the trained ONNX model")
    parser.add_argument("--video", type=str, default=None, help="Path to a test video file for simulation")
    parser.add_argument("--camera", type=int, default=0, help="USB Camera index (if not using CSI)")
    parser.add_argument("--csi", action="store_true", help="Use Jetson onboard CSI camera")
    parser.add_argument("--port", type=str, default="/dev/ttyTHS1", help="Serial port to STM32")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baudrate")
    parser.add_argument("--no-serial", action="store_true", help="Run without sending serial commands")
    parser.add_argument("--show", action="store_true", help="Display diagnostic GUI window")
    parser.add_argument("--output", type=str, default=None, help="Path to save the processed video with overlays")
    args = parser.parse_args()

    # Determine paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if args.model is None:
        model_path = os.path.join(current_dir, "..", "model", "model.onnx")
    else:
        model_path = args.model

    # Check for model existence
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}. Please train and export model first.")
        return

    # Initialize ONNX Runtime Session
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    print(f"Initializing ONNX session with model: {model_path}")
    try:
        session = ort.InferenceSession(model_path, providers=providers)
        print(f"ONNX session loaded successfully. Active Provider: {session.get_providers()}")
    except Exception as e:
        print(f"Failed to load ONNX session: {e}")
        return

    input_name = session.get_inputs()[0].name

    # Initialize Serial Communication
    ser = SerialTX(port=args.port, baudrate=args.baud, mock=args.no_serial)

    # Initialize Lane and Steering Controller
    controller = LaneController(img_width=320, img_height=180)

    # Open Camera or Video Source
    if args.csi:
        gst_str = gstreamer_pipeline(display_width=320, display_height=180, framerate=30)
        print(f"Opening CSI camera with pipeline: {gst_str}")
        cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
    elif args.video:
        print(f"Opening test video file: {args.video}")
        cap = cv2.VideoCapture(args.video)
    else:
        print(f"Opening USB camera index: {args.camera}")
        cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        print("Error: Could not open video/camera source.")
        return

    # Initialize Video Writer if output path is set
    video_writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(args.output, fourcc, 20.0, (320, 180))
        print(f"Saving output video to: {args.output}")

    print("Pipeline running. Press 'q' in visualizer to quit.")

    
    # FPS tracking variables
    prev_time = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if args.video:
                    print("Reached end of video file.")
                else:
                    print("Failed to capture frame from camera.")
                break

            # 1. Preprocessing
            # Convert to target dimensions: 320x180
            h, w = frame.shape[:2]
            if w != 320 or h != 180:
                resized_frame = cv2.resize(frame, (320, 180))
            else:
                resized_frame = frame.copy()

            # BGR to RGB (ONNX model trained on standard RGB tensors)
            rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0.0, 1.0] and transpose to (CHW)
            normalized = rgb.astype(np.float32) / 255.0
            input_tensor = np.transpose(normalized, (2, 0, 1))
            
            # Add batch dimension: (1, 3, 180, 320)
            input_batch = np.expand_dims(input_tensor, axis=0)

            # 2. Model Inference
            start_infer = time.time()
            outputs = session.run(None, {input_name: input_batch})
            infer_time_ms = (time.time() - start_infer) * 1000

            # Extraction and thresholding
            prob_mask = outputs[0][0, 0] # Shape: (180, 320)
            binary_mask = (prob_mask > 0.5).astype(np.uint8) * 255

            # 3. Path Control Calculation
            controller.fit_lanes(binary_mask)
            steer_angle, target_x, status = controller.calculate_steering()

            # 4. Transmit commands to STM32
            ser.send_steering_angle(steer_angle)

            # Calculate actual loop FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if prev_time != 0 else 0.0
            prev_time = curr_time

            # 5. Visualizer overlay & Output video saving
            if args.show or video_writer is not None:
                overlay = controller.draw_overlay(resized_frame, binary_mask)
                
                # Overlay inference specs & loop rates
                cv2.putText(overlay, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(overlay, f"Inference: {infer_time_ms:.1f} ms", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                
                if video_writer is not None:
                    video_writer.write(overlay)
                
                if args.show:
                    cv2.imshow("Jetson Nano Lane Detection", overlay)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            
            if not args.show:
                # Log to console in headless mode
                print(f"FPS: {fps:.1f} | Latency: {infer_time_ms:.1f}ms | Angle: {steer_angle:6.2f} | Status: {status}")

    except KeyboardInterrupt:
        print("\nInterrupt received. Stopping pipeline...")

    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()
            print("Output video writer released.")
        ser.close()
        if args.show:
            cv2.destroyAllWindows()
        print("Shutdown completed.")


if __name__ == "__main__":
    main()
