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


def print_symbolic_equations():
    """
    Print the complete symbolic forward kinematics equations.
    """
    # Define symbolic variables
    theta1, theta2, theta3 = symbols('theta1 theta2 theta3')
    
    # Robot parameters
    a1 = 53.5
    l1 = 150
    l2 = 150
    
    print("="*80)
    print("COMPLETE SYMBOLIC FORWARD KINEMATICS EQUATIONS")
    print("="*80)
    
    print(f"\nRobot Parameters:")
    print(f"  a1 = {a1}")
    print(f"  l1 = {l1}")
    print(f"  l2 = {l2}")
    
    print(f"\nSymbolic Variables:")
    print(f"  θ1 = {theta1}")
    print(f"  θ2 = {theta2}")
    print(f"  θ3 = {theta3}")
    
    # Joint angles
    ta1 = theta1
    ta2 = theta2
    ta3 = theta3
    
    print("\n" + "="*50)
    print("TRANSFORMATION MATRICES CONSTRUCTION")
    print("="*50)
    
    # Transform 1: Base to Joint 1
    print("\n1. T01 Matrix (Base to Joint 1):")
    print("R01 rotation matrix:")
    R01 = Matrix([[cos(ta1), -sin(ta1), 0],
                  [sin(ta1), cos(ta1), 0],
                  [0, 0, 1]])
    print(R01)
    
    print("\nP01 pose matrix:")
    P01 = Matrix([[1, 0, 0],
                  [0, 0, 1],
                  [0, -1, 0]])
    print(P01)
    
    print("\nR01 * P01:")
    R01P01 = R01 * P01
    print(R01P01)
    
    # Transform 2: Joint 1 to Joint 2
    print("\n2. T12 Matrix (Joint 1 to Joint 2):")
    print("R12 rotation matrix:")
    R12 = Matrix([[cos(ta2), -sin(ta2), 0],
                  [sin(ta2), cos(ta2), 0],
                  [0, 0, 1]])
    print(R12)
    
    print("\nP12 pose matrix:")
    P12 = Matrix([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])
    print(P12)
    
    print("\nR12 * P12:")
    R12P12 = R12 * P12
    print(R12P12)
    
    # Transform 3: Joint 2 to Joint 3
    print("\n3. T23 Matrix (Joint 2 to Joint 3):")
    print("R23 rotation matrix:")
    R23 = Matrix([[cos(ta3-ta2), -sin(ta3-ta2), 0],
                  [sin(ta3-ta2), cos(ta3-ta2), 0],
                  [0, 0, 1]])
    print(R23)
    
    print("\nP23 pose matrix:")
    P23 = Matrix([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])
    print(P23)
    
    print("\nR23 * P23:")
    R23P23 = R23 * P23
    print(R23P23)
    
    print("\n" + "="*50)
    print("COMPLETE TRANSFORMATION MATRICES")
    print("="*50)
    
    # Build complete transformation matrices
    T01 = Matrix.hstack(Matrix.vstack(R01P01, Matrix([[0, 0, 0]])),
                        Matrix([[0], [0], [a1], [1]]))
    
    T12 = Matrix.hstack(Matrix.vstack(R12P12, Matrix([[0, 0, 0]])),
                        Matrix([[90 + l1*sin(ta2)], [-l1*cos(ta2)], [0], [1]]))
    
    T23 = Matrix.hstack(Matrix.vstack(R23P23, Matrix([[0, 0, 0]])),
                        Matrix([[l2*sin(ta3-ta2)], [-l2*cos(ta3-ta2)], [0], [1]]))
    
    print("\nT01 (Base to Joint 1):")
    print(T01)
    
    print("\nT12 (Joint 1 to Joint 2):")
    print(T12)
    
    print("\nT23 (Joint 2 to Joint 3):")
    print(T23)
    
    print("\n" + "="*50)
    print("FORWARD KINEMATICS EQUATIONS")
    print("="*50)
    
    # Forward kinematics: T03 = T01 * T12 * T23
    print("\nT03 = T01 * T12 * T23:")
    T03_raw = T01 * T12 * T23
    print("T03 (before simplification):")
    print(T03_raw)
    
    T03 = simplify(T03_raw)
    print("\nT03 (after simplification):")
    print(T03)
    
    print("\n" + "="*50)
    print("POSITION EQUATIONS")
    print("="*50)
    
    position_symbolic = T03[0:3, 3]
    print("\nEnd-effector position:")
    print("P_x =", position_symbolic[0])
    print("P_y =", position_symbolic[1])  
    print("P_z =", position_symbolic[2])
    
    print("\nOrientation matrix:")
    orientation_symbolic = T03[0:3, 0:3]
    print(orientation_symbolic)
    
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    print("\nForward kinematics equations:")
    print("P_x =", position_symbolic[0])
    print("P_y =", position_symbolic[1])
    print("P_z =", position_symbolic[2])
    
    # Calculate adjusted position with offset
    offset_vector = Matrix([0, 0, -58.5])
    adjusted_position_symbolic = position_symbolic + offset_vector
    
    print("\nAdjusted end-effector position (with offset [0, 0, -58.5]):")
    print("P_adj_x =", adjusted_position_symbolic[0])
    print("P_adj_y =", adjusted_position_symbolic[1])
    print("P_adj_z =", adjusted_position_symbolic[2])
    
    return {
        'T01': T01,
        'T12': T12, 
        'T23': T23,
        'T03': T03,
        'position': position_symbolic,
        'adjusted_position': adjusted_position_symbolic,
        'orientation': orientation_symbolic
    }


def inverse_kinematics_simple(target_x, target_y, target_z, show_symbolic=True):
    """
    Solve inverse kinematics symbolically for the target end-effector position.
    
    Args:
        target_x: Target x position for end-effector
        target_y: Target y position for end-effector  
        target_z: Target z position for end-effector
        show_symbolic: Whether to show symbolic equations during solving
    
    Returns:
        Dictionary with solutions and analysis
    """
    # Define symbolic variables
    theta1, theta2, theta3 = symbols('theta1 theta2 theta3')
    
    # Robot parameters
    a1 = 53.5
    l1 = 150
    l2 = 150
    
    # Apply offset to get actual target position (reverse the adjustment)
    offset_vector = Matrix([0, 0, -58.5])
    adjusted_target = Matrix([target_x, target_y, target_z])
    actual_target = adjusted_target - offset_vector
    
    if show_symbolic:
        print(f"=== INVERSE KINEMATICS PROBLEM ===")
        print(f"Target end-effector position: [{target_x}, {target_y}, {target_z}]")
        print(f"Target after removing offset: {actual_target}")
        print(f"Robot parameters: a1={a1}, l1={l1}, l2={l2}")
        print("\n=== SOLVING FOR JOINT ANGLES ===")
    
    # Get the symbolic position equations from forward kinematics
    # We'll use the position components from T03
    R01 = Matrix([[cos(theta1), -sin(theta1), 0],
                  [sin(theta1), cos(theta1), 0],
                  [0, 0, 1]])
    
    P01 = Matrix([[1, 0, 0],
                  [0, 0, 1],
                  [0, -1, 0]])
    
    R01P01 = R01 * P01
    T01 = Matrix.hstack(Matrix.vstack(R01P01, Matrix([[0, 0, 0]])),
                        Matrix([[0], [0], [a1], [1]]))
    
    R12 = Matrix([[cos(theta2), -sin(theta2), 0],
                  [sin(theta2), cos(theta2), 0],
                  [0, 0, 1]])
    
    P12 = Matrix([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])
    
    R12P12 = R12 * P12
    T12 = Matrix.hstack(Matrix.vstack(R12P12, Matrix([[0, 0, 0]])),
                        Matrix([[90 + l1*sin(theta2)], [-l1*cos(theta2)], [0], [1]]))
    
    R23 = Matrix([[cos(theta3-theta2), -sin(theta3-theta2), 0],
                  [sin(theta3-theta2), cos(theta3-theta2), 0],
                  [0, 0, 1]])
    
    P23 = Matrix([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])
    
    R23P23 = R23 * P23
    T23 = Matrix.hstack(Matrix.vstack(R23P23, Matrix([[0, 0, 0]])),
                        Matrix([[l2*sin(theta3-theta2)], [-l2*cos(theta3-theta2)], [0], [1]]))
    
    T03 = simplify(T01 * T12 * T23)
    
    # Extract position equations
    eq_x = T03[0, 3] - actual_target.x
    eq_y = T03[1, 3] - actual_target.y  
    eq_z = T03[2, 3] - actual_target.z
    
    if show_symbolic:
        print("\nPosition constraint equations:")
        print(f"X constraint: {eq_x} = 0")
        print(f"Y constraint: {eq_y} = 0")
        print(f"Z constraint: {eq_z} = 0")
        print("\n=== SOLUTION ATTEMPT ===")
    
    # Try to solve the system of equations
    try:
        solutions = sp.solve([eq_x, eq_y, eq_z], [theta1, theta2, theta3])
        
        if show_symbolic:
            if solutions:
                print(f"Found {len(solutions)} solution(s):")
                for i, sol in enumerate(solutions):
                    print(f"Solution {i+1}:")
                    print(f"  theta1 = {sol.get(theta1, 'Not determined')}")
                    print(f"  theta2 = {sol.get(theta2, 'Not determined')}")
                    print(f"  theta3 = {sol.get(theta3, 'Not determined')}")
            else:
                print("No symbolic solutions found!")
        
        return {
            'success': len(solutions) > 0,
            'solutions': solutions,
            'equations': [eq_x, eq_y, eq_z],
            'target_position': actual_target
        }
        
    except Exception as e:
        if show_symbolic:
            print(f"Error in symbolic solving: {e}")
            print("The equations are too complex for symbolic solution.")
            print("Consider using numerical methods or geometric approach.")
        
        return {
            'success': False,
            'error': str(e),
            'equations': [eq_x, eq_y, eq_z],
            'target_position': actual_target
        }


def geometric_inverse_kinematics(target_x, target_y, target_z):
    """
    Solve inverse kinematics using geometric approach.
    This is often more practical than symbolic solving.
    """
    print(f"=== GEOMETRIC INVERSE KINEMATICS ===")
    print(f"Target end-effector position: [{target_x}, {target_y}, {target_z}]")
    
    # Robot parameters
    a1 = 53.5
    l1 = 150
    l2 = 150
    
    # Apply offset correction
    actual_z = target_z + 58.5
    
    print(f"Actual target (no offset): [{target_x}, {target_y}, {actual_z}]")
    
    results = []
    
    # For a 2D planar arm, theta1 can be solved first
    theta1 = theta1_sym = symbols('theta1')
    
    # Then theta2 and theta3 can be solved geometrically
    # This is a simplified approach - you may need to adapt based on your specific robot
    
    # Check workspace reachability roughly
    from sympy import sqrt
    r2_z2 = target_x**2 + actual_z**2  # Distance in XZ plane
    max_reach = l1 + l2  # Maximum possible reach
    
    print(f"Required reach: {sqrt(r2_z2)}")
    print(f"Maximum reach: {max_reach}")
    
    try:
        if sqrt(r2_z2) > max_reach:
            print("WARNING: Target position is outside workspace!")
            return {'success': False, 'reason': 'Outside workspace'}
        else:
            print("Target is reachable.")
            return {'success': True, 'geometry_check': True}
    except:
        print("Could not perform geometric analysis.")
        return {'success': False, 'reason': 'Calculation error'}


if __name__ == "__main__":
    print("Original DH method:")
    print(forward_kinematics(0, 0, 0, 0))
    print(apply_transform(forward_kinematics(0, 0, 0, 0), (0, 0, 0)))
    
    print("\nSymPy method:")
    # Test with joint angles j1=0, j2=0, j3=30 degrees
    result = forward_kinematics_sympy(theta1_val=0, theta2_val=0, theta3_val=30)
    print_results_numeric(result)
    
    print("\n" + "="*80)
    print("SYMBOLIC EQUATIONS")
    print("="*80)
    
    # Print complete symbolic forward kinematics equations
    symbolic_result = print_symbolic_equations()
    
    print("\n" + "="*60)
    print("INVERSE KINEMATICS TEST")
    print("="*60)
    
    # Test inverse kinematics with the position we found from forward kinematics
    target_pos = result['adjusted_position'].evalf()  # [219.904, 0, 70.0]
    
    print(f"\nTesting inverse kinematics for corrected position: {target_pos}")
    ik_result = inverse_kinematics_simple(float(target_pos[0]), float(target_pos[1]), float(target_pos[2]))
    
    print("\nGeometric approach:")
    geometric_result = geometric_inverse_kinematics(float(target_pos[0]), float(target_pos[1]), float(target_pos[2]))