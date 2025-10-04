import numpy as np
import math
import sympy as sp
from sympy import symbols, cos, sin, Matrix, simplify, pi

def DH(theta, d, a, alpha):

    return np.array([[np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
                     [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
                     [0, np.sin(alpha), np.cos(alpha), d],
                     [0, 0, 0, 1]])

def forward_kinematics(theta1, theta2, theta3, theta4):
    T01 = DH(theta1, 107.2, -12.5, np.pi/2)
    T12 = DH(theta2, 0, 150, 0)
    # T23 = DH(theta3, 76-35, 150+25+40, 0)
    T23 = DH(theta3, -(76-35), 150+50+40, -np.pi/2)
    T34 = DH(theta4, 0, 0, 0)

    T04 = T01 @ T12 @ T23 @ T34
    return T04

def apply_transform(T, point):
    return T @ np.array([point[0], point[1], point[2], 1])
    


def forward_kinematics_sympy(theta1_val=None, theta2_val=None, theta3_val=None):
    """
    Forward kinematics using SymPy based on MATLAB code.
    
    Args:
        theta1_val: Joint 1 angle in degrees (optional)
        theta2_val: Joint 2 angle in degrees (optional) 
        theta3_val: Joint 3 angle in degrees (optional)
    
    Returns:
        Dictionary with transformation matrix, position, and adjusted position
    """
    # Define symbolic variables
    theta1, theta2, theta3 = symbols('theta1 theta2 theta3')
    
    # Robot parameters
    a1 = 53.5
    l1 = 150
    l2 = 150
    
    # Joint angles
    ta1 = theta1
    ta2 = theta2
    ta3 = theta3
    
    # Transformation matrix from base to joint 1: T01
    R01 = Matrix([[cos(ta1), -sin(ta1), 0],
                  [sin(ta1), cos(ta1), 0],
                  [0, 0, 1]])
    
    P01 = Matrix([[1, 0, 0],
                  [0, 0, 1],
                  [0, -1, 0]])
    
    # Combine rotation and pose matrices, then add translation
    R01P01 = R01 * P01
    T01 = Matrix.hstack(Matrix.vstack(R01P01, Matrix([[0, 0, 0]])),
                        Matrix([[0], [0], [a1], [1]]))
    
    # Transformation matrix from joint 1 to joint 2: T12
    R12 = Matrix([[cos(ta2), -sin(ta2), 0],
                  [sin(ta2), cos(ta2), 0],
                  [0, 0, 1]])
    
    P12 = Matrix([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])
    
    # Combine rotation and pose matrices, then add translation
    R12P12 = R12 * P12
    T12 = Matrix.hstack(Matrix.vstack(R12P12, Matrix([[0, 0, 0]])),
                        Matrix([[90 + l1*sin(ta2)], [-l1*cos(ta2)], [0], [1]]))
    
    # Transformation matrix from joint 2 to joint 3: T23
    R23 = Matrix([[cos(ta3-ta2), -sin(ta3-ta2), 0],
                  [sin(ta3-ta2), cos(ta3-ta2), 0],
                  [0, 0, 1]])
    
    P23 = Matrix([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])
    
    # Combine rotation and pose matrices, then add translation
    R23P23 = R23 * P23
    T23 = Matrix.hstack(Matrix.vstack(R23P23, Matrix([[0, 0, 0]])),
                        Matrix([[l2*sin(ta3-ta2)], [-l2*cos(ta3-ta2)], [0], [1]]))
    
    # Forward kinematics: T03 = T01 * T12 * T23
    T03 = simplify(T01 * T12 * T23)
    
    result = {
        'T03_symbolic': T03,
        'position_symbolic': T03[0:3, 3],  # Extract position (translation part)
        'orientation_symbolic': T03[0:3, 0:3]  # Extract orientation (rotation part)
    }
    
    # If joint angles are provided, substitute and calculate numerical values
    if theta1_val is not None and theta2_val is not None and theta3_val is not None:
        # Convert degrees to radians
        t1 = theta1_val * pi / 180  # Convert degrees to radians
        t2 = theta2_val * pi / 180
        t3 = theta3_val * pi / 180
        
        # Substitute joint angles into transformation matrices
        T01_substituted = T01.subs([(theta1, t1), (theta2, t2), (theta3, t3)])
        T12_substituted = T12.subs([(theta1, t1), (theta2, t2), (theta3, t3)])
        T23_substituted = T23.subs([(theta1, t1), (theta2, t2), (theta3, t3)])
        T03_substituted = T03.subs([(theta1, t1), (theta2, t2), (theta3, t3)])
        
        # Extract position and orientation
        position_substituted = T03_substituted[0:3, 3]
        orientation_substituted = T03_substituted[0:3, 0:3]
        
        # End Effector Position after adjustment
        offset_vector = Matrix([0, 0, -58.5])
        adjusted_position = position_substituted + offset_vector
        
        result.update({
            'T01_numerical': T01_substituted,
            'T12_numerical': T12_substituted,
            'T23_numerical': T23_substituted,
            'T03_numerical': T03_substituted,
            'position': position_substituted,
            'orientation': orientation_substituted,
            'adjusted_position': adjusted_position
        })
    
    return result


def print_results_numeric(result):
    """
    Helper function to print SymPy results with numeric conversion for easier reading.
    """
    print("=== INTERMEDIATE TRANSFORMATION MATRICES ===")
    print("\nT01 matrix (Base to Joint 1):")
    T01_numeric = result['T01_numerical'].evalf()
    print(T01_numeric)
    
    print("\nT12 matrix (Joint 1 to Joint 2):")
    T12_numeric = result['T12_numerical'].evalf()
    print(T12_numeric)
    
    print("\nT23 matrix (Joint 2 to Joint 3):")
    T23_numeric = result['T23_numerical'].evalf()
    print(T23_numeric)
    
    print("\n=== FINAL RESULT ===")
    print("\nT03 matrix (numeric):")
    T03_numeric = result['T03_numerical'].evalf()
    print(T03_numeric)
    
    print("\nPosition (numeric):")
    position_numeric = result['position'].evalf()
    print(position_numeric)
    
    print("\nAdjusted Position (numeric):")
    adjusted_position_numeric = result['adjusted_position'].evalf()
    print(adjusted_position_numeric)
    
    return T01_numeric, T12_numeric, T23_numeric, T03_numeric, position_numeric, adjusted_position_numeric


if __name__ == "__main__":
    print("Original DH method:")
    print(forward_kinematics(0, 0, 0, 0))
    print(apply_transform(forward_kinematics(0, 0, 0, 0), (0, 0, 0)))
    
    print("\nSymPy method:")
    # Test with joint angles j1=0, j2=0, j3=30 degrees
    result = forward_kinematics_sympy(theta1_val=0, theta2_val=0, theta3_val=30)
    print_results_numeric(result)