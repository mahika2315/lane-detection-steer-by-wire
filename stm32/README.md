# STM32 Steer-by-Wire Integration Guide

This directory contains the embedded C source files to implement the low-level motor controller. The modules are structured to be compatible with **STM32CubeMX** / **STM32CubeIDE** and standard **HAL (Hardware Abstraction Layer)** drivers.

---

## 1. Peripheral Setup in STM32CubeMX

To compile and execute this code, configure the following peripherals in your STM32 configuration (`.ioc` file):

### A. USART2 (Communication with Jetson)
* **Mode**: Asynchronous
* **Baud Rate**: `115200 Bits/s`
* **Word Length**: `8 Bits`
* **Parity**: `None`
* **Stop Bits**: `1`
* **NVIC Settings**: Enable `USART2 global interrupt` (Crucial for receiving bytes in the background).

### B. TIM1 (PWM Actuator Command)
* **Channel 1**: `PWM Generation CH1`
* **Counter Settings**:
  * **For DC Motor (H-Bridge)**: Aim for a frequency of $10\text{kHz} - 20\text{kHz}$ to avoid audible high-pitch noise. (e.g. Prescaler = 0, Counter Period = 4000 on an 80MHz clock).
  * **For RC Servo**: Must run at $50\text{Hz}$ (20ms period). Set Prescaler and Period accordingly (e.g. Prescaler = 83, Counter Period = 19999 on an 84MHz clock).

### C. ADC1 (Steering Angle Position Feedback)
* **Channel**: Configure any analog channel (e.g. `IN0` on `PA0`) to read a feedback potentiometer attached to the steering rack.
* **Resolution**: `12-bit` (standard).

---

## 2. Hardware Wiring Diagram

Always connect matching grounds between the boards. **Note that the Jetson Nano uses 3.3V logic levels, which are compatible with STM32.**

```text
  Jetson Nano J41 Header                 STM32 MCU (e.g. Nucleo)
 +-----------------------+              +-----------------------+
 |  Pin 8 (UART TX)      | ------------>|  PA3 (USART2 RX)      |
 |  Pin 10 (UART RX)     | <------------|  PA2 (USART2 TX)      |
 |  Pin 14 (GND)         | ------------>|  GND (Signal Ground)  |
 +-----------------------+              +-----------------------+
                                        +-----------------------+
                                        |  PA0 (ADC1 IN0) <==== Feedback Potentiometer
                                        |  PA8 (TIM1 PWM) =====> Motor Driver Speed Pin
                                        |  PB0 (GPIO Out) =====> Motor Driver Direction Pin
                                        +-----------------------+
```

---

## 3. Step-by-Step Code Integration

1. Copy [steering_control.h](file:///c:/rvce/helios/lanedetection/stm32/Core/Inc/steering_control.h) into your project's `Core/Inc/` directory.
2. Copy [steering_control.c](file:///c:/rvce/helios/lanedetection/stm32/Core/Src/steering_control.c) into your project's `Core/Src/` directory.
3. Open your project's `Core/Src/main.c` and make the following changes:

### A. Includes & Variables
In `main.c`, under `/* USER CODE BEGIN Includes */`:
```c
#include "steering_control.h"
```

Under `/* USER CODE BEGIN PV */` (Private Variables):
```c
extern steering_system_t steering_sys;
uint8_t rx_byte = 0;
```

### B. Initialization & Startup
Under `/* USER CODE BEGIN 2 */` (inside main before the while loop):
```c
steering_init(&steering_sys);
HAL_TIM_PWM_Start(&htim1, TIM_CHANNEL_1);
HAL_UART_Receive_IT(&huart2, &rx_byte, 1); // Start interrupt listening
```

### C. The Control Loop
Inside the `while (1)` loop under `/* USER CODE BEGIN 3 */`:
```c
uint32_t current_time = HAL_GetTick();
static uint32_t last_loop_time = 0;

if ((current_time - last_loop_time) >= 10) // 100Hz
{
  float dt = (float)(current_time - last_loop_time) / 1000.0f;
  last_loop_time = current_time;
  
  // Read physical angle from ADC
  float actual_angle = Read_Steering_Feedback_Angle();
  steering_update_feedback(&steering_sys, actual_angle);
  
  // Supervision & control computation
  steering_check_watchdog(&steering_sys, current_time);
  steering_run_pid(&steering_sys, dt);
  steering_apply_output(&steering_sys);
}
```

### D. Interrupt Callback Function
At the bottom of `main.c` under `/* USER CODE BEGIN 4 */`:
```c
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
  if (huart->Instance == USART2)
  {
    steering_parse_byte(&steering_sys, rx_byte);
    HAL_UART_Receive_IT(&huart2, &rx_byte, 1); // Listen for next byte
  }
}
```
