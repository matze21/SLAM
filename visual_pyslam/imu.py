import numpy as np

def euler_to_rotation_matrix(roll, pitch, yaw):
    # Convert angles to radians
    roll, pitch, yaw = np.radians([roll, pitch, yaw])
    
    # Rotation matrices
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])
    
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])
    
    # Combine rotations
    R = np.dot(Rz, np.dot(Ry, Rx))
    return R

def integrate_acceleration(ax, ay, az, dt):
    # First integration: acceleration to velocity
    vx = ax * dt
    vy = ay * dt
    vz = az * dt
    
    # Second integration: velocity to displacement
    dx = vx * dt
    dy = vy * dt
    dz = az * dt
    
    return np.array([dx, dy, dz])

def imu_to_essential_matrix(roll, pitch, yaw, ax, ay, az):
    # Get rotation matrix from IMU orientation
    R = euler_to_rotation_matrix(roll, pitch, yaw)
    
    # Estimate translation from accelerometer data
    t = np.array([ax, ay, az])
    
    # Create skew-symmetric matrix from translation
    t_x = np.array([[0, -t[2], t[1]],
                    [t[2], 0, -t[0]],
                    [-t[1], t[0], 0]])
    
    # Compute essential matrix
    E = np.dot(t_x, R)
    
    return E