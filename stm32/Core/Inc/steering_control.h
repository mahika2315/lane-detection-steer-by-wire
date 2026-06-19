#ifndef STEERING_CONTROL_H
#define STEERING_CONTROL_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/* --- Configuration Constants --- */
#define SERIAL_START_BYTE1      0xAA
#define SERIAL_START_BYTE2      0x55
#define CMD_STEERING_CONTROL    0x10
#define PAYLOAD_LENGTH_EXPECTED 4

#define WATCHDOG_TIMEOUT_MS     500   // Disengage steering motor if no frame for 500ms
#define MAX_STEERING_ANGLE_DEG  30.0f // Hard mechanical limit
#define MIN_STEERING_ANGLE_DEG  -30.0f

#define PID_KP                  2.5f  // Tune based on actuator response
#define PID_KI                  0.1f
#define PID_KD                  0.05f
#define PID_OUTPUT_LIMIT        100.0f // PWM maximum percentage

/* --- Types and Enums --- */

typedef enum {
    RX_STATE_START1 = 0,
    RX_STATE_START2,
    RX_STATE_CMD,
    RX_STATE_LEN,
    RX_STATE_DATA,
    RX_STATE_CHECKSUM
} rx_state_t;

typedef struct {
    // Controller Variables
    float target_angle;       // Setpoint in degrees
    float actual_angle;       // Current reading in degrees
    float motor_out;          // Actuator command output (-100.0f to 100.0f percent)
    
    // PID State
    float pid_kp;
    float pid_ki;
    float pid_kd;
    float integral;
    float prev_error;
    
    // Safety & Monitoring
    uint32_t last_packet_time; // Timestamp (ms) of last valid command packet
    uint8_t watchdog_tripped;  // 1 = Safe state active, 0 = Operating normally
    
    // Serial Receiver State Machine
    rx_state_t rx_state;
    uint8_t rx_cmd;
    uint8_t rx_len;
    uint8_t rx_data_buf[8];
    uint8_t rx_data_index;
    uint8_t rx_checksum;
} steering_system_t;

/* --- Function Prototypes --- */

/**
 * @brief Initializes the steering system control variables and state.
 */
void steering_init(steering_system_t *sys);

/**
 * @brief Feeds a single byte received over UART into the packet parser state machine.
 */
void steering_parse_byte(steering_system_t *sys, uint8_t byte);

/**
 * @brief Updates the actual steering angle reading from sensor feedback.
 * @param actual Measured angle in degrees.
 */
void steering_update_feedback(steering_system_t *sys, float actual);

/**
 * @brief Computes PID controller command based on error. Should be called periodically.
 * @param dt Time delta since last call in seconds.
 */
void steering_run_pid(steering_system_t *sys, float dt);

/**
 * @brief Safety watchdog checker to monitor transmission heartbeats.
 * @param current_time_ms Current system tick in milliseconds.
 */
void steering_check_watchdog(steering_system_t *sys, uint32_t current_time_ms);

/**
 * @brief Interfaces with low-level STM32 hardware registers (PWM/DAC/GPIO) to set output.
 */
void steering_apply_output(steering_system_t *sys);

#ifdef __cplusplus
}
#endif

#endif // STEERING_CONTROL_H
