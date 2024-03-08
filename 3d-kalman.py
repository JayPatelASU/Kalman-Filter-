import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Adjusted for 3D movement with reduced trajectory overlap
dt = 1.0
A = np.array([[1, dt, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0],
              [0, 0, 1, dt, 0, 0],
              [0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 1, dt],
              [0, 0, 0, 0, 0, 1]])
H = np.array([[1, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 1, 0]])
Q = 0.05 * np.eye(6)
R = 0.1 * np.eye(3)
x_est = np.array([[0], [1], [0], [1], [0], [1]])
P = np.eye(6)
I = np.eye(6)
x_true = np.array([[0], [1], [0], [1], [0], [1]])
n_timesteps = 1000

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# For fading effect
history_length = 100

true_positions_x = []
true_positions_y = []
true_positions_z = []
estimated_positions_x = []
estimated_positions_y = []
estimated_positions_z = []
total_difference = 0
max_difference = 0

def init():
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    return []

def update(frame):
    global x_est, P, x_true, total_difference, max_difference
    global true_positions_x, true_positions_y, true_positions_z
    global estimated_positions_x, estimated_positions_y, estimated_positions_z

    # Adjust velocity for reduced trajectory overlap using a sine wave pattern
    if frame % 100 == 0:
        x_true[1, 0] *= -1  # Reverse x velocity every 100 frames
    x_true[3, 0] = np.cos(frame / 20.0)  # Smooth y velocity change
    x_true[5, 0] = np.sin(frame / 20.0)  # Smooth z velocity change

    x_true = A @ x_true + np.random.normal(0, 0.1, (6, 1))
    true_positions_x.append(x_true[0, 0])
    true_positions_y.append(x_true[2, 0])
    true_positions_z.append(x_true[4, 0])

    true_positions_x = true_positions_x[-history_length:]
    true_positions_y = true_positions_y[-history_length:]
    true_positions_z = true_positions_z[-history_length:]

    z = H @ x_true + np.random.normal(0, np.sqrt(0.1), (3, 1))

    x_pred = A @ x_est
    P_pred = A @ P @ A.T + Q
    K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)
    x_est = x_pred + K @ (z - H @ x_pred)
    P = (I - K @ H) @ P_pred

    estimated_positions_x.append(x_est[0, 0])
    estimated_positions_y.append(x_est[2, 0])
    estimated_positions_z.append(x_est[4, 0])

    estimated_positions_x = estimated_positions_x[-history_length:]
    estimated_positions_y = estimated_positions_y[-history_length:]
    estimated_positions_z = estimated_positions_z[-history_length:]

    # Clear previous lines and replot to differentiate the leading part
    ax.cla()
    ax.plot(true_positions_x[:-1], true_positions_y[:-1], true_positions_z[:-1], 'r-', lw=2, alpha=0.5, label='True Position (Trail)')
    ax.plot(estimated_positions_x[:-1], estimated_positions_y[:-1], estimated_positions_z[:-1], 'b-', lw=2, alpha=0.5, label='Estimated Position (Trail)')

    # Highlight the leading part of the paths with different colors
    if len(true_positions_x) > 1 and len(estimated_positions_x) > 1:
        ax.plot(true_positions_x[-2:], true_positions_y[-2:], true_positions_z[-2:], 'k-', lw=4, label='True Position (Leading)')
        ax.plot(estimated_positions_x[-2:], estimated_positions_y[-2:], estimated_positions_z[-2:], 'orange', lw=4, label='Estimated Position (Leading)')

    ax.legend()
    
    difference = np.sqrt(
        (x_true[0, 0] - x_est[0, 0]) ** 2 + (x_true[2, 0] - x_est[2, 0]) ** 2 + (x_true[4, 0] - x_est[4, 0]) ** 2)
    total_difference += difference
    average_difference = total_difference / (frame + 1)
    max_difference = max(max_difference, difference)

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_xlim([min(true_positions_x + estimated_positions_x), max(true_positions_x + estimated_positions_x)])
    ax.set_ylim([min(true_positions_y + estimated_positions_y), max(true_positions_y + estimated_positions_y)])
    ax.set_zlim([min(true_positions_z + estimated_positions_z), max(true_positions_z + estimated_positions_z)])
    ax.set_title(f'Frame: {frame}, Live Difference: {difference:.2f}, Avg. Difference: {average_difference:.2f}, Max Difference: {max_difference:.2f}')

    return []

ani = FuncAnimation(fig, update, frames=n_timesteps, init_func=init, blit=False, interval=10, repeat=False)

plt.show()
