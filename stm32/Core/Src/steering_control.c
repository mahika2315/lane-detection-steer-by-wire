#include "steering_control.h"
#include <string.h>

// External reference to STM32 HAL system tick function
extern uint32_t HAL_GetTick(void);

// External helper to set PWM or GPIO pins. 
// Can be customized for specific hardware.
void steering_set_actuator_hardware(float motor_output_pct);

void steering_init(steering_system_t *sys) {
    sys->target_angle = 0.0f;
    sys->actual_angle = 0.0f;
    sys->motor_out = 0.0f;
    
    sys->pid_kp = PID_KP;
    sys->pid_ki = PID_KI;
    sys->pid_kd = PID_KD;
    sys->integral = 0.0f;
    sys->prev_error = 0.0f;
    
    sys->last_packet_time = 0;
    sys->watchdog_tripped = 1; // Start in safe, tripped state until first valid packet
    
    sys->rx_state = RX_STATE_START1;
    sys->rx_cmd = 0;
    sys->rx_len = 0;
    sys->rx_data_index = 0;
    sys->rx_checksum = 0;
}

void steering_parse_byte(steering_system_t *sys, uint8_t byte) {
    switch (sys->rx_state) {
        case RX_STATE_START1:
            if (byte == SERIAL_START_BYTE1) {
                sys->rx_state = RX_STATE_START2;
            }
            break;
            
        case RX_STATE_START2:
            if (byte == SERIAL_START_BYTE2) {
                sys->rx_state = RX_STATE_CMD;
            } else {
                sys->rx_state = RX_STATE_START1;
            }
            break;
            
        case RX_STATE_CMD:
            if (byte == CMD_STEERING_CONTROL) {
                sys->rx_cmd = byte;
                sys->rx_checksum = byte; // Start checksum calculation
                sys->rx_state = RX_STATE_LEN;
            } else {
                sys->rx_state = RX_STATE_START1;
            }
            break;
            
        case RX_STATE_LEN:
            if (byte == PAYLOAD_LENGTH_EXPECTED) {
                sys->rx_len = byte;
                sys->rx_checksum ^= byte;
                sys->rx_data_index = 0;
                sys->rx_state = RX_STATE_DATA;
            } else {
                sys->rx_state = RX_STATE_START1;
            }
            break;
            
        case RX_STATE_DATA:
            sys->rx_data_buf[sys->rx_data_index++] = byte;
            sys->rx_checksum ^= byte;
            if (sys->rx_data_index >= sys->rx_len) {
                sys->rx_state = RX_STATE_CHECKSUM;
            }
            break;
            
        case RX_STATE_CHECKSUM:
            if (byte == sys->rx_checksum) {
                // Packet successfully parsed!
                float parsed_angle = 0.0f;
                memcpy(&parsed_angle, sys->rx_data_buf, 4);
                
                // Clamp target angle to hardware-safe boundaries
                if (parsed_angle > MAX_STEERING_ANGLE_DEG) {
                    parsed_angle = MAX_STEERING_ANGLE_DEG;
                } else if (parsed_angle < MIN_STEERING_ANGLE_DEG) {
                    parsed_angle = MIN_STEERING_ANGLE_DEG;
                }
                
                sys->target_angle = parsed_angle;
                sys->last_packet_time = HAL_GetTick();
                sys->watchdog_tripped = 0; // Clear watchdog
            }
            // Always return to hunt for start bytes
            sys->rx_state = RX_STATE_START1;
            break;
            
        default:
            sys->rx_state = RX_STATE_START1;
            break;
    }
}

void steering_update_feedback(steering_system_t *sys, float actual) {
    sys->actual_angle = actual;
}

void steering_run_pid(steering_system_t *sys, float dt) {
    // If the safety watchdog is tripped, shut off motor and clear states
    if (sys->watchdog_tripped) {
        sys->motor_out = 0.0f;
        sys->integral = 0.0f;
        sys->prev_error = 0.0f;
        return;
    }
    
    // Ensure dt is valid to prevent division by zero
    if (dt <= 0.0001f) {
        dt = 0.01f; // Default to 100Hz step
    }
    
    float error = sys->target_angle - sys->actual_angle;
    
    // Proportional Term
    float p_term = sys->pid_kp * error;
    
    // Integral Term with simple anti-windup clamping
    sys->integral += error * dt;
    float i_term = sys->pid_ki * sys->integral;
    if (i_term > PID_OUTPUT_LIMIT) {
        i_term = PID_OUTPUT_LIMIT;
        sys->integral = PID_OUTPUT_LIMIT / sys->pid_ki;
    } else if (i_term < -PID_OUTPUT_LIMIT) {
        i_term = -PID_OUTPUT_LIMIT;
        sys->integral = -PID_OUTPUT_LIMIT / sys->pid_ki;
    }
    
    // Derivative Term
    float d_term = sys->pid_kd * (error - sys->prev_error) / dt;
    sys->prev_error = error;
    
    // Total controller output
    float output = p_term + i_term + d_term;
    
    // Clamp to motor controller limits (-100% to 100%)
    if (output > PID_OUTPUT_LIMIT) {
        output = PID_OUTPUT_LIMIT;
    } else if (output < -PID_OUTPUT_LIMIT) {
        output = -PID_OUTPUT_LIMIT;
    }
    
    sys->motor_out = output;
}

void steering_check_watchdog(steering_system_t *sys, uint32_t current_time_ms) {
    // Check if time elapsed exceeds timeout threshold
    if ((current_time_ms - sys->last_packet_time) > WATCHDOG_TIMEOUT_MS) {
        sys->watchdog_tripped = 1;
        sys->motor_out = 0.0f;
        sys->target_angle = 0.0f;
    }
}

void steering_apply_output(steering_system_t *sys) {
    // Send PWM command to hardware driver
    steering_set_actuator_hardware(sys->motor_out);
}

// ============================================================================
// LOW-LEVEL ACTUATOR DRIVER EXAMPLE (TO BE CUSTOMIZED BY USER)
// ============================================================================
// Define motor driver pins and parameters. These can be adjusted for:
// - H-Bridge (PWM duty cycle + Direction GPIO)
// - RC Servo (50Hz PWM signal, 1ms - 2ms pulse width)
// ============================================================================

#define MOTOR_DRIVER_H_BRIDGE   1  // Set to 1 for H-Bridge, 2 for RC Servo
#define STEERING_PWM_TIMER_REG  TIM1->CCR1 // Target Timer Channel Compare Register

void steering_set_actuator_hardware(float motor_output_pct) {
    // Map control output (-100.0f to +100.0f) to registers
#if MOTOR_DRIVER_H_BRIDGE == 1
    // DC MOTOR H-BRIDGE DRIVER:
    // Direction GPIO Pin (e.g. DIR_Pin on Port B)
    // Speed PWM Timer Channel
    
    if (motor_output_pct >= 0.0f) {
        // Turn Right: DIR Pin HIGH, set PWM speed
        // HAL_GPIO_WritePin(GPIOB, GPIO_PIN_0, GPIO_PIN_SET);
        float speed = motor_output_pct;
        uint32_t compare_val = (uint32_t)((speed / 100.0f) * 1000.0f); // Assuming Timer period is 1000
        // STEERING_PWM_TIMER_REG = compare_val;
        (void)compare_val; // Silence warning in simulation
    } else {
        // Turn Left: DIR Pin LOW, set PWM speed
        // HAL_GPIO_WritePin(GPIOB, GPIO_PIN_0, GPIO_PIN_RESET);
        float speed = -motor_output_pct;
        uint32_t compare_val = (uint32_t)((speed / 100.0f) * 1000.0f);
        // STEERING_PWM_TIMER_REG = compare_val;
        (void)compare_val;
    }
    
#elif MOTOR_DRIVER_H_BRIDGE == 2
    // RC SERVO DRIVER (50Hz, 1.0ms - 2.0ms):
    // 0 deg steering center is 1.5ms pulse.
    // -30 deg left is 1.0ms pulse.
    // +30 deg right is 2.0ms pulse.
    // Assuming Timer Clock = 1MHz, Auto-Reload = 20000 (20ms / 50Hz period)
    // CCR duty ranges from 1000 (1ms) to 2000 (2ms)
    
    // Scale motor output to RC pulse width
    // Center steering (0.0f output) = 1500 comparison limit
    float pulse_width_us = 1500.0f + (motor_output_pct / 100.0f) * 500.0f;
    uint32_t compare_val = (uint32_t)pulse_width_us;
    
    // STEERING_PWM_TIMER_REG = compare_val;
    (void)compare_val;
#endif
}
