import numpy as np
import math
import sympy as sp
from sympy import symbols, cos, sin, Matrix, simplify, pi
import argparse
import sys

offset_vector = Matrix([0, 0, -53.5])

    
def solve_forward_kinematics_symbolically():
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

    return T03, T01, T12, T23, theta1, theta2, theta3


def forward_kinematics_sympy(theta1_val=None, theta2_val=None, theta3_val=None, theta4_val= None):
    """
    Forward kinematics using SymPy based on MATLAB code.
    
    Args:
        theta1_val: Joint 1 angle in degrees (optional)
        theta2_val: Joint 2 angle in degrees (optional) 
        theta3_val: Joint 3 angle in degrees (optional)
    
    Returns:
        Dictionary with transformation matrix, position, and adjusted position
    """
    T03, T01, T12, T23, theta1, theta2, theta3 = solve_forward_kinematics_symbolically()
    r = None
    if theta4_val is not None: 
        r = theta4_val +theta1_val
    
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
        
        
        adjusted_position = position_substituted + offset_vector
        
        result.update({
            'T01_numerical': T01_substituted,
            'T12_numerical': T12_substituted,
            'T23_numerical': T23_substituted,
            'T03_numerical': T03_substituted,
            'position': position_substituted,
            'orientation': orientation_substituted,
            'adjusted_position': adjusted_position,
            'r': r
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
    T03, T01, T12, T23, theta1, theta2, theta3 = solve_forward_kinematics_symbolically()
    
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

    adjusted_position_symbolic = position_symbolic + offset_vector
    
    print("\nAdjusted end-effector position (with offset [0, 0, -58.5]):")
    print("P_adj_x =", adjusted_position_symbolic[0])
    print("P_adj_y =", adjusted_position_symbolic[1])
    print("P_adj_z =", adjusted_position_symbolic[2])
    
    return {
        'T03': T03,
        'position': position_symbolic,
        'adjusted_position': adjusted_position_symbolic,
        'orientation': orientation_symbolic
    }


def inverse_kinematics_simple(target_x, target_y, target_z, show_symbolic=True):
    """
    Solve inverse kinematics analytically using the symbolic forward kinematics equations.
    
    Args:
        target_x: Target x position for end-effector
        target_y: Target y position for end-effector  
        target_z: Target z position for end-effector
        show_symbolic: Whether to show symbolic equations during solving
    
    Returns:
        Dictionary with solutions and analysis
    """
    # Get symbolic forward kinematics
    T03_symbolic, T01, T12, T23, theta1, theta2, theta3 = solve_forward_kinematics_symbolically()
    
    if show_symbolic:
        print(f"=== INVERSE KINEMATICS PROBLEM ===")
        print(f"Target end-effector position: [{target_x}, {target_y}, {target_z}]")
        print("Robot parameters: a1=53.5, l1=150, l2=150")
        print("\n=== SOLVING FOR JOINT ANGLES ===")
    
    # Remove offset to get actual target position
    actual_target = Matrix([target_x, target_y, target_z]) - offset_vector
    target_x_actual, target_y_actual, target_z_actual = actual_target[0], actual_target[1], actual_target[2]
    
    if show_symbolic:
        print(f"Target after removing offset: [{target_x_actual}, {target_y_actual}, {target_z_actual}]")
    
    # Extract symbolic position equations from T03
    pos_x_symbolic = T03_symbolic[0, 3]  # P_x = 30*(5*sin(θ2) + 5*cos(θ3) + 3)*cos(θ1)
    pos_y_symbolic = T03_symbolic[1, 3]  # P_y = 30*(asint(θ2) + 5*cos(θ3) + 3)*sin(θ1)
    pos_z_symbolic = T03_symbolic[2, 3]  # P_z = -150.0*sin(θ3) + 150.0*cos(θ2) + 53.5
    
    if show_symbolic:
        print(f"\nSymbolic position equations:")
        print(f"P_x = {pos_x_symbolic}")
        print(f"P_y = {pos_y_symbolic}")
        print(f"P_z = {pos_z_symbolic}")
    
    # Analytical solution based on the kinematics equations
    
    solutions = []
    
    try:
        # From position equations, we know:
        # P_x = 30*(5*sin(θ2) + 5*cos(θ3) + 3)*cos(θ1) = target_x_actual
        # P_y = 30*(5*sin(θ2) + 5*cos(θ3) + 3)*sin(θ1) = target_y_actual
        # P_z = -150.0*sin(θ3) + 150.0*cos(θ2) + 53.5 = target_z_actual
        
        # Strategy: Use the structure of the equations to solve step by step
        
        # Step 1: From x and y equations, we can solve for θ1
        # If P_x^2 + P_y^2 = [30*(5*sin(θ2) + 5*cos(θ3) + 3)]^2 * (cos^2(θ1) + sin^2(θ1))
        #                     = [30*(5*sin(θ2) + 5*cos(θ3) + 3)]^2
        
        radial_distance_squared = target_x_actual**2 + target_y_actual**2
        radial_distance = sp.sqrt(radial_distance_squared)
        
        if show_symbolic:
            print(f"\nAnalytical solution:")
            print(f"Radial distance from robot base: {radial_distance}")
        
        # From: 30*(5*sin(θ2) + 5*cos(θ3) + 3) = radial_distance
        # We get: 5*sin(θ2) + 5*cos(θ3) + 3 = radial_distance/30
        
        from sympy import atan2, asin, acos, pi
        
        # Solve for θ1 using atan2
        if target_x_actual != 0 or target_y_actual != 0:
            theta1_sol = sp.atan2(target_y_actual, target_x_actual)
        else:
            theta1_sol = 0  # If at origin, θ1 can be arbitrary
        
        # Now solve for θ2 and θ3 from z equation and the radial constraint
        # From z equation: -150*sin(θ3) + 150*cos(θ2) = target_z_actual - 53.5
        # From radial: 5*sin(θ2) + 5*cos(θ3) = radial_distance/30 - 3
        
        # This gives us two equations:
        # cos(θ2) - sin(θ3) = (target_z_actual - 53.5)/150
        # sin(θ2) + cos(θ3) = radial_distance/30 - 3
        
        # For now, let's try a numerical approach since the analytical solution is complex
        
        if show_symbolic:
            print(f"θ1 = {theta1_sol} ({theta1_sol*180/pi:.2f}°)")
        
        # Try multiple θ2 values and solve for θ3
        best_solution = None
        min_error = float('inf')
        
        # Search through θ2 values from -π to π with higher resolution
        import numpy as np
        theta2_test_values = np.linspace(-np.pi, np.pi, 500)
        for theta2_test in theta2_test_values:
            # From the position equations:
            # 30*(5*sin(θ2) + 5*cos(θ3) + 3) = radial_distance
            # So: sin(θ2) + cos(θ3) = radial_distance/30/5 - 3/5
            
            radial_term = float((radial_distance/30/5 - 3/5).evalf())
            cos_theta3_target = radial_term - np.sin(theta2_test)
            
            # Check if cos(θ3) is in valid range [-1, 1]
            if -1 <= cos_theta3_target <= 1:
                theta3_candidates = [np.arccos(cos_theta3_target), -np.arccos(cos_theta3_target)]
                
                for theta3_test in theta3_candidates:
                    # Check if this solution satisfies the z equation
                    z_calculated = float((-150*np.sin(theta3_test) + 150*np.cos(theta2_test) + 53.5))
                    z_error = abs(z_calculated - float(target_z_actual.evalf()))
                    
                    if z_error < min_error:
                        min_error = z_error
                        best_solution = {
                            'theta1': theta1_sol,
                            'theta2': theta2_test,
                            'theta3': theta3_test,
                            'error': z_error
                        }
        
        if best_solution and best_solution['error'] < 1.0:  # Tolerance of 1mm
            solution = best_solution
            theta1_deg = float(solution['theta1'] * 180/pi)
            theta2_deg = float(solution['theta2'] * 180/pi)
            theta3_deg = float(solution['theta3'] * 180/pi)
            
            if show_symbolic:
                print(f"Solution found:")
                print(f"θ1 = {theta1_deg:.2f}°")
                print(f"θ2 = {theta2_deg:.2f}°")
                print(f"θ3 = {theta3_deg:.2f}°")
                print(f"Position error: ~{min_error:.3f} mm")
            
            # Verify solution by plugging back into forward kinematics
            test_result = forward_kinematics_sympy(theta1_val=theta1_deg, theta2_val=theta2_deg, theta3_val=theta3_deg)
            test_pos = test_result['adjusted_position'].evalf()
            
            if show_symbolic:
                print(f"Verification - calculated position: [{float(test_pos[0]):.1f}, {float(test_pos[1]):.1f}, {float(test_pos[2]):.1f}]")
                print(f"Target position: [{target_x}, {target_y}, {target_z}]")
            
            return {
                'success': True,
                'solutions': [solution],
                'theta1_deg': theta1_deg,
                'theta2_deg': theta2_deg,
                'theta3_deg': theta3_deg,
                'position_error': min_error,
                'verified_position': test_pos
            }
        
        else:
            if show_symbolic:
                print("No suitable solution found!")
                if best_solution:
                    print(f"Best attempt had error: {min_error:.3f} mm")
            
            return {'success': False, 'reason': 'No solution found within tolerance'}
    
    except Exception as e:
        if show_symbolic:
            print(f"Error in inverse kinematics: {e}")
        
        return {'success': False, 'error': str(e)}



def run_cli():
    """
    Command-line interface for kinematics calculations.
    
    Usage examples:
    Forward kinematics: python kinematics.py --forward --angles 0 0 30
    Inverse kinematics: python kinematics.py --inverse --position 219.9 0 70.0
    Symbolic equations: python kinematics.py --symbolic
    """
    parser = argparse.ArgumentParser(
        description='Robot Kinematics Calculator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Forward kinematics:
    python kinematics.py --forward --angles 0 0 30
    
  Inverse kinematics:
    python kinematics.py --inverse --position 219.9 0 70.0
    
  Show symbolic equations:
    python kinematics.py --symbolic
    
  Show intermediate matrices:
    python kinematics.py --forward --angles 0 0 30 --intermediate
        """
    )
    
    # Create mutually exclusive group for operation type
    operation_group = parser.add_mutually_exclusive_group(required=True)
    operation_group.add_argument('--forward', '-f', action='store_true',
                                 help=f'Run forward kinematics (requires --angles)')
    operation_group.add_argument('--inverse', '-i', action='store_true',
                                 help=f'Run inverse kinematics (requires --position)')
    operation_group.add_argument('--symbolic', '-s', action='store_true',
                                 help=f'Show complete symbolic equations')
    
    # Arguments for forward kinematics
    parser.add_argument('--angles', '-a', nargs=3, type=float, metavar=('THETA1', 'THETA2', 'THETA3'),
                        help=f'Joint angles in degrees for forward kinematics')
    parser.add_argument('--intermediate', action='store_true',
                        help=f'Show intermediate transformation matrices')
    
    # Arguments for inverse kinematics
    parser.add_argument('--position', '-p', nargs=3, type=float, metavar=('X', 'Y', 'Z'),
                        help=f'Target end-effector position for inverse kinematics')
    
    # Additional options
    parser.add_argument('--verbose', '-v', action='store_true',
                        help=f'Show detailed output')
    
    args = parser.parse_args()
    
    print("="*60)
    print("ROBOT KINEMATICS CALCULATOR")
    print("="*60)
    
    # Handle symbolic equations request
    if args.symbolic:
        print("\nGenerating complete symbolic forward kinematics equations...")
        symbolic_result = print_symbolic_equations()
        return symbolic_result
    
    # Handle forward kinematics
    if args.forward:
        if args.angles is None:
            print("Error: Forward kinematics requires --angles argument")
            print("Example: python kinematics.py --forward --angles 0 0 30")
            return
        
        theta1, theta2, theta3 = args.angles
        print(f"\n=== FORWARD KINEMATICS ===")
        print(f"Joint angles: θ1={theta1}°, θ2={theta2}°, θ3={theta3}°")
        
        # Run forward kinematics
        result = forward_kinematics_sympy(theta1_val=theta1, theta2_val=theta2, theta3_val=theta3)
        
        if args.intermediate:
            # Show intermediate matrices
            print_results_numeric(result)
        else:
            # Just show final results
            print("\nFinal Results:")
            print(f"T03 matrix:")
            print(result['T03_numerical'].evalf())
            print(f"\nPosition: {result['position'].evalf()}")
            print(f"Adjusted Position: {result['adjusted_position'].evalf()}")
        
        return result
    
    # Handle inverse kinematics
    if args.inverse:
        if args.position is None:
            print("Error: Inverse kinematics requires --position argument")
            print("Example: python kinematics.py --inverse --position 219.9 0 70.0")
            return
        
        x, y, z = args.position
        print(f"\n=== INVERSE KINEMATICS ===")
        print(f"Target position: [{x}, {y}, {z}]")
        
        # Try symbolic inverse kinematics
        if args.verbose:
            ik_result = inverse_kinematics_simple(x, y, z, show_symbolic=True)
        else:
            ik_result = inverse_kinematics_simple(x, y, z, show_symbolic=False)
        
        print(f"\nResults:")
        print(f"Inverse kinematics success: {ik_result['success']}")
        if not ik_result['success']:
            print(f"Error: {ik_result.get('error', 'No specific error reported')}")
        
        return ik_result


if __name__ == "__main__":
    # Check if command line arguments were provided
    if len(sys.argv) > 1:
        # Run CLI version
        run_cli()
    else:
        # Run demonstration version
        print("DEMO MODE - No CLI arguments provided")
        print("Use --help to see command-line options")
        print("\n" + "="*60)
        print("DEMONSTRATION")
        print("="*60)
        
        print("\nOriginal DH method:")
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
        
        print("\n" + "="*60)
        print("CLI USAGE EXAMPLES")
        print("="*60)
        print("\nForward kinematics:")
        print("  python kinematics.py --forward --angles 0 0 30")
        print("  python kinematics.py --forward --angles 0 0 30 --intermediate")
        
        print("\nInverse kinematics:")
        print("  python kinematics.py --inverse --position 219.9 0 70.0")
        print("  python kinematics.py --inverse --position 219.9 0 70.0 --verbose")
        
        print("\nSymbolic equations:")
        print("  python kinematics.py --symbolic")
        
        print("\nHelp:")
        print("  python kinematics.py --help")