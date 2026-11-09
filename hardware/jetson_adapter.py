"""
NavMind AI - Jetson Nano Adapter (Future Hardware Integration)
Interface for deploying the trained RL agent on real hardware.

This module provides the same interface as SimulationAdapter but
communicates with physical sensors and motors on the Jetson Nano.

Usage (future):
    from hardware.jetson_adapter import JetsonAdapter
    adapter = JetsonAdapter(serial_port="/dev/ttyUSB0")
    # Then use identically to SimulationAdapter
"""

import numpy as np
from typing import Optional


class JetsonAdapter:
    """
    Hardware adapter for Jetson Nano deployment.

    Maps simulation observation/action format to real robot hardware:
    - Reads from physical distance sensors (HC-SR04 or similar)
    - Reads from camera or LiDAR
    - Sends velocity commands to motor controller

    NOTE: This class is a scaffold for future implementation.
    Physical robot commands require careful safety validation before use.
    """

    def __init__(
        self,
        serial_port: str = "/dev/ttyUSB0",
        baud_rate: int = 115200,
        sensor_config: Optional[dict] = None,
    ):
        """
        Args:
            serial_port: Serial port for motor controller communication
            baud_rate: Serial baud rate
            sensor_config: Sensor configuration overrides
        """
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.sensor_config = sensor_config or {}
        self._connected = False
        self._emergency_stop = False

        # Sensor readings (populated from hardware)
        self._sensor_readings = np.ones(5, dtype=np.float32)
        self._robot_x = 0.0
        self._robot_y = 0.0
        self._robot_theta = 0.0

    def connect(self) -> bool:
        """
        Connect to hardware (motors + sensors).

        Returns:
            True if connection successful
        """
        # TODO: Implement serial/GPIO connection
        # import serial
        # self._serial = serial.Serial(self.serial_port, self.baud_rate)
        print("[Jetson] Hardware connection — not yet implemented.")
        print("[Jetson] Running in simulation-passthrough mode.")
        return False

    def get_observation(self) -> np.ndarray:
        """
        Read observation from physical sensors.

        Hardware pipeline:
        1. Read distance sensors (front, left, right, front-left, front-right)
        2. Read odometry for robot position/heading
        3. Compute distance and angle to goal
        4. Return normalized observation vector

        Returns:
            Observation array matching RobotNavEnv format
        """
        # TODO: Replace with actual sensor reads
        # dist_front = self._read_ultrasonic(pin=18)
        # dist_left = self._read_ultrasonic(pin=23)
        # ...
        obs = np.array([
            0.5,  # dist_to_goal (placeholder)
            0.0,  # angle_to_goal
            self._sensor_readings[0],  # front
            self._sensor_readings[1],  # front-left
            self._sensor_readings[2],  # front-right
            self._sensor_readings[3],  # left
            self._sensor_readings[4],  # right
        ], dtype=np.float32)
        return obs

    def execute_action(self, action: int):
        """
        Send action command to motor controller.

        Action mapping to motor velocities:
        0 (Forward):    left_motor=+v,  right_motor=+v
        1 (Turn Left):  left_motor=-v/2, right_motor=+v
        2 (Turn Right): left_motor=+v,  right_motor=-v/2
        3 (Stop):       left_motor=0,   right_motor=0
        """
        if self._emergency_stop:
            self._send_motor_command(0, 0)
            return

        # TODO: Implement motor commands via serial
        action_map = {
            0: "FORWARD",
            1: "TURN_LEFT",
            2: "TURN_RIGHT",
            3: "STOP",
        }
        cmd = action_map.get(action, "STOP")
        # self._serial.write(f"{cmd}\n".encode())

    def emergency_stop(self):
        """Immediately halt all motors."""
        self._emergency_stop = True
        self._send_motor_command(0, 0)
        print("[Jetson] EMERGENCY STOP activated!")

    def resume(self):
        """Resume operation after emergency stop."""
        self._emergency_stop = False

    def _send_motor_command(self, left_vel: float, right_vel: float):
        """Send velocity command to differential drive controller."""
        # TODO: Implement serial command
        pass

    def is_simulation(self) -> bool:
        """Always False for hardware adapter."""
        return False

    def disconnect(self):
        """Safely disconnect from hardware."""
        self.emergency_stop()
        self._connected = False
        print("[Jetson] Disconnected from hardware.")

    # ─── Sensor Integration Notes ─────────────────────────────────────────
    #
    # Sensor               | Simulation equiv.      | Jetson GPIO/Interface
    # ---------------------|------------------------|----------------------
    # Front distance       | Ray-cast sensor[0]     | HC-SR04 / VL53L0X
    # Left/Right distance  | Ray-cast sensors       | HC-SR04 array
    # 2D LiDAR             | Multi-ray sensors      | RPLiDAR A1 (UART)
    # Camera               | Pygame render()        | CSI camera / USB cam
    # Odometry             | robot.x, y, theta      | Encoder feedback
    # Motor control        | env.step(action)       | L298N / TB6612 driver
    #
    # Power budget for RTX 3050 policy inference: negligible (< 1ms per step)
    # Jetson Nano compute: runs PyTorch inference at 30+ Hz

