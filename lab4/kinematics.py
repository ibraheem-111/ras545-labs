import numpy as np
import math

def DH(theta, d, a, alpha):

    return np.array([[np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
                     [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
                     [0, np.sin(alpha), np.cos(alpha), d],
                     [0, 0, 0, 1]])

def forward_kinematics(theta1, theta2, theta3):
    T01 = DH(theta1, 107.2, 12.5, np.pi/2)
    T12 = DH(theta2, 150, 0, 0)
    T23 = DH(theta3, 0, 0, 0)

    T03 = T01 @ T12 @ T23
    return T03

def apply_transform(T, point):
    return T @ np.array([point[0], point[1], point[2], 1])
    


if __name__ == "__main__":
    print(forward_kinematics(0, 0, 0))
    print(apply_transform(forward_kinematics(0, 0, 0), (0, 0, 0)))