# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 00:37:58 2024

@author: jayni
"""
import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter

# Define simulation parameters
h = 0.1  # Time step
initial_position = 10
initial_velocity = -5
acceleration = 0.5
noise_std = 1  # Measurement noise standard deviation
num_steps = 100

# System matrices for a Newtonian system
A = np.array([[1, h, 0.5 * h**2], [0, 1, h], [0, 0, 1]])
B = np.array([0, 0, 0])  # Not used in this setup
C = np.array([[1, 0, 0]])
R = np.array([[1]])  # Measurement noise covariance
Q = np.zeros((3, 3))  # Process noise covariance

# Initial guesses for the state and covariance
x0 = np.array([[0], [0], [0]])
P0 = np.eye(3)

# Generate simulation data
time_vector = np.linspace(0, (num_steps - 1) * h, num_steps)
ideal_position = initial_position + initial_velocity * time_vector + 0.5 * acceleration * time_vector**2
ideal_velocity = initial_velocity + acceleration * time_vector
position_noisy = ideal_position + noise_std * np.random.randn(num_steps)

# Plotting the observed vs. ideal positions
plt.figure(figsize=(10, 6))
plt.plot(time_vector, ideal_position, 'g-', linewidth=2, label='Ideal position')
plt.plot(time_vector, position_noisy, 'r.', label='Observed position')
plt.xlabel('Time')
plt.ylabel('Position')
plt.legend()
plt.grid(True)
plt.show()

# Initialize the Kalman filter
kf = KalmanFilter(dim_x=3, dim_z=1)
kf.x = x0  # Initial state estimate
kf.P = P0  # Initial covariance estimate
kf.F = A  # State transition matrix
kf.H = C  # Measurement function
kf.R = R  # Measurement uncertainty
kf.Q = Q  # Process uncertainty

# Perform Kalman filtering
estimates = np.zeros((num_steps, 3))
for i in range(num_steps):
    kf.predict()
    kf.update(position_noisy[i])
    estimates[i, :] = kf.x.T

# Extract estimates for plotting
estimated_position = estimates[:, 0]
estimated_velocity = estimates[:, 1]
estimated_acceleration = estimates[:, 2]

# Plot results
plt.figure(figsize=(15, 10))

# Position
plt.subplot(3, 1, 1)
plt.plot(time_vector, ideal_position, 'g-', label='True position')
plt.plot(time_vector, estimated_position, 'b--', label='Estimated position')
plt.xlabel('Time')
plt.ylabel('Position')
plt.legend()
plt.grid(True)

# Velocity
plt.subplot(3, 1, 2)
plt.plot(time_vector, ideal_velocity, 'g-', label='True velocity')
plt.plot(time_vector, estimated_velocity, 'b--', label='Estimated velocity')
plt.xlabel('Time')
plt.ylabel('Velocity')
plt.legend()
plt.grid(True)

# Acceleration
plt.subplot(3, 1, 3)
plt.plot(time_vector, acceleration * np.ones(num_steps), 'g-', label='True acceleration')
plt.plot(time_vector, estimated_acceleration, 'b--', label='Estimated acceleration')
plt.xlabel('Time')
plt.ylabel('Acceleration')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

