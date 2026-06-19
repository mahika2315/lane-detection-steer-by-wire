import struct
import serial
import time

class SerialTX:
    """Handles communication between Jetson Nano and STM32 using a custom binary protocol."""
    def __init__(self, port='/dev/ttyTHS1', baudrate=115200, timeout=0.1, mock=False):
        self.mock = mock
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        
        if not mock:
            try:
                self.ser = serial.Serial(port, baudrate, timeout=timeout)
                print(f"Serial port {port} initialized at {baudrate} baud.")
            except Exception as e:
                print(f"Failed to open serial port {port}: {e}. Running in MOCK mode.")
                self.mock = True
        else:
            print("Running serial in MOCK mode.")
            
    def send_steering_angle(self, angle: float):
        """
        Sends target steering angle to the STM32.
        Frame Format:
        [0xAA] [0x55] [CMD_ID] [LEN] [DATA_0] [DATA_1] [DATA_2] [DATA_3] [CHECKSUM]
        * START_BYTES: 0xAA, 0x55 (2 bytes)
        * CMD_ID: 0x10 (Steering control, 1 byte)
        * LEN: 0x04 (4 bytes payload, 1 byte)
        * DATA: IEEE 754 float32 steering angle in degrees (little-endian, 4 bytes)
        * CHECKSUM: XOR of CMD_ID, LEN, and 4 payload bytes (1 byte)
        """
        # Pack float value as little-endian float (compatible with STM32 ARM)
        payload = struct.pack('<f', float(angle))
        
        cmd_id = 0x10
        data_len = 4
        
        # Calculate XOR checksum
        checksum = cmd_id ^ data_len
        for b in payload:
            checksum ^= b
            
        # Build packet frame
        frame = bytearray([0xAA, 0x55, cmd_id, data_len]) + payload + bytearray([checksum])
        
        if not self.mock and self.ser:
            try:
                self.ser.write(frame)
                self.ser.flush()
            except Exception as e:
                print(f"Serial write error: {e}")
                self.mock = True
        return frame

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("Serial port closed.")

if __name__ == "__main__":
    # Self-test code
    tx = SerialTX(mock=True)
    test_angle = 12.54
    packet = tx.send_steering_angle(test_angle)
    print(f"Generated packet for steering angle {test_angle}: {[hex(b) for b in packet]}")
    # Unpack to verify
    unpacked = struct.unpack('<f', packet[4:8])[0]
    print(f"Verification unpacked float: {unpacked}")
    assert abs(unpacked - test_angle) < 1e-4, "Serialization error!"
    print("SerialTX verification test passed.")
