# Autonomous Lane Detection & Steer-by-Wire System

A real-time, closed-loop autonomous driving sub-system that combines high-level computer vision and trajectory planning with low-level actuator control and hardware safety monitoring.

The system is split across two main components:
1. **Jetson Nano (High-Level AI Brain)**: Runs a custom, lightweight convolutional neural network (Tiny U-Net) to detect lane lines from a camera feed, fits polynomial curves, and calculates target steering angles.
2. **STM32 Microcontroller (Low-Level Actuator Control)**: Parses steering commands from the Jetson Nano, monitors serial connection safety, reads wheel feedback via an ADC, and runs a PID controller to drive the steering motor.

---

## Repository Structure

```text
lanedetection/
├── nano/
│   ├── requirements.txt      # Python library dependencies
│   ├── model/
│   │   ├── net.py            # Tiny U-Net neural network definition
│   │   ├── train.py          # Training loop with synthetic data generator
│   │   └── export_onnx.py    # ONNX export and verification script
│   └── inference/
│       ├── control.py        # Polynomial fitting & look-ahead control math
│       ├── serial_tx.py      # Binary serial packaging and transmitter (UART)
│       └── main.py           # Main camera frame loop & inference script
├── stm32/
│   ├── Core/
│   │   ├── Inc/
│   │   │   └── steering_control.h  # Configurations, structs, and prototypes
│   │   └── Src/
│   │       ├── steering_control.c  # Packet parser, PID, and watchdog
│   │       └── main.c              # Peripheral template and 100Hz loop
│   └── README.md             # Microcontroller pin configuration guide
└── tests/
    ├── generate_test_video.py# Creates a synthetic video for offline testing
    └── test_pipeline.py      # Integration test validating geometry and packaging
```

---

## Hardware Connections (Pinout)

Ensure that all connected UART pins share a common ground and operate at **3.3V logic levels**.

### Jetson Nano to STM32 UART Interface
| Jetson Nano Pin (J41 Header) | STM32 Microcontroller Pin | Description |
| :--- | :--- | :--- |
| **Pin 8 (TXD)** | **PA3 (USART2 RX)** | Transmits target steering angle |
| **Pin 10 (RXD)** | **PA2 (USART2 TX)** | Receives telemetry feedback (optional) |
| **Pin 14 (GND)** | **GND (Signal Ground)** | Shared common ground reference |

### STM32 to Driver & Sensor Interface
| STM32 Microcontroller Pin | Driver/Sensor Interface | Description |
| :--- | :--- | :--- |
| **PA8 (TIM1 PWM)** | H-Bridge PWM Speed Pin | Speed/torque duty cycle signal |
| **PB0 (GPIO Output)** | H-Bridge Direction Pin | Controls motor direction (Polarity) |
| **PA0 (ADC1 IN0)** | Potentiometer Center Pin | Reads analog feedback voltage |

---

## Getting Started

### 1. High-Level Pipeline Setup (Jetson Nano / Laptop)
Clone the repository and install the dependencies:
```bash
cd lanedetection
pip install -r nano/requirements.txt
```

Train the model using the synthetic road generator:
```bash
python nano/model/train.py
```

Export the trained model to the optimized ONNX format:
```bash
python nano/model/export_onnx.py
```

Run the pipeline on a test video (runs headless and compiles an output video file with overlays):
```bash
python nano/inference/main.py --video tests/test_video.mp4 --output output.mp4 --no-serial
```

Run the pipeline with a camera feed:
```bash
# USB Camera
python nano/inference/main.py --camera 0 --port /dev/ttyTHS1
# CSI Camera
python nano/inference/main.py --csi --port /dev/ttyTHS1
```

### 2. Low-Level Controller Setup (STM32)
* Copy `steering_control.h` and `steering_control.c` from the `stm32/Core/` subdirectories into your STM32 IDE workspace.
* Configure USART2 (115200 Baud, global interrupt enabled), TIM1 (PWM generation), and ADC1 (potentiometer feedback reading) in STM32CubeMX.
* Integrate the interrupt callback (`HAL_UART_RxCpltCallback`) and the 100Hz controller update loop in `main.c` (refer to the templates in `stm32/Core/Src/main.c`).
* Flash the controller.

---

## Safety Features

* **Loss-of-Signal Watchdog**: If the Jetson Nano stops sending commands (e.g., software crash or camera unplugged) for more than **500 ms**, the STM32 automatically cuts motor power to prevent runaway steering.
* **XOR Checksum Verification**: Every serial data packet is verified using an XOR checksum byte to prevent erroneous steering movements caused by serial line noise.
* **Mechanical Angle Clamping**: Target steering commands are hard-clamped to a safe mechanical range ($\pm 30.0^\circ$) at both the Python planning and C execution layers.
