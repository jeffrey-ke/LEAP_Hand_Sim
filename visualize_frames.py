#!/usr/bin/env python3
"""
Visualize URDF Frame Relationships
This script creates a 3D visualization of the LEAP Hand coordinate frames.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from urdf_frame_parser import URDFFrameParser
import os


def plot_coordinate_frame(ax, T, frame_name, scale=0.01, alpha=0.8):
    """
    Plot a coordinate frame as RGB axes (X=Red, Y=Green, Z=Blue).
    
    Args:
        ax: matplotlib 3D axis
        T: 4x4 transformation matrix
        frame_name: name of the frame
        scale: scale of the axis arrows
        alpha: transparency
    """
    origin = T[:3, 3]
    
    # X-axis (red)
    x_axis = T[:3, 0] * scale
    ax.quiver(origin[0], origin[1], origin[2], 
              x_axis[0], x_axis[1], x_axis[2], 
              color='red', alpha=alpha, arrow_length_ratio=0.1)
    
    # Y-axis (green)
    y_axis = T[:3, 1] * scale
    ax.quiver(origin[0], origin[1], origin[2], 
              y_axis[0], y_axis[1], y_axis[2], 
              color='green', alpha=alpha, arrow_length_ratio=0.1)
    
    # Z-axis (blue)
    z_axis = T[:3, 2] * scale
    ax.quiver(origin[0], origin[1], origin[2], 
              z_axis[0], z_axis[1], z_axis[2], 
              color='blue', alpha=alpha, arrow_length_ratio=0.1)
    
    # Add frame label
    ax.text(origin[0], origin[1], origin[2], frame_name, 
            fontsize=8, alpha=0.7)


def visualize_hand_frames():
    """Visualize all coordinate frames in the LEAP Hand."""
    
    # Parse URDF
    urdf_path = "/Users/jke/repo/LEAP_Hand_Sim/assets/leap_hand/robot.urdf"
    if not os.path.exists(urdf_path):
        print(f"URDF file not found: {urdf_path}")
        return
    
    parser = URDFFrameParser(urdf_path)
    
    # Create 3D plot
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Calculate cumulative transformations from palm_lower (root) to all frames
    root_frame = "palm_lower"
    all_frames = parser.get_all_frames()
    
    # Plot root frame
    T_root = np.eye(4)  # Identity matrix for root
    plot_coordinate_frame(ax, T_root, root_frame, scale=0.02, alpha=1.0)
    
    # Track frame positions for connecting lines
    frame_positions = {root_frame: T_root[:3, 3]}
    
    # Calculate and plot all other frames
    for frame in all_frames:
        if frame != root_frame:
            T_frame = parser.get_transformation_chain(root_frame, frame)
            if T_frame is not None:
                plot_coordinate_frame(ax, T_frame, frame, scale=0.015)
                frame_positions[frame] = T_frame[:3, 3]
    
    # Draw kinematic chain connections
    for joint_name, joint_data in parser.joints.items():
        parent = joint_data['parent']
        child = joint_data['child']
        
        if parent in frame_positions and child in frame_positions:
            parent_pos = frame_positions[parent]
            child_pos = frame_positions[child]
            
            ax.plot([parent_pos[0], child_pos[0]],
                   [parent_pos[1], child_pos[1]], 
                   [parent_pos[2], child_pos[2]], 
                   'k-', alpha=0.5, linewidth=1)
    
    # Set equal aspect ratio and labels
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12) 
    ax.set_zlabel('Z (meters)', fontsize=12)
    ax.set_title('LEAP Hand Coordinate Frames\n(X=Red, Y=Green, Z=Blue)', fontsize=14)
    
    # Set axis limits for better viewing
    all_positions = np.array(list(frame_positions.values()))
    margin = 0.02
    ax.set_xlim(all_positions[:, 0].min() - margin, all_positions[:, 0].max() + margin)
    ax.set_ylim(all_positions[:, 1].min() - margin, all_positions[:, 1].max() + margin)
    ax.set_zlim(all_positions[:, 2].min() - margin, all_positions[:, 2].max() + margin)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='red', lw=2, label='X-axis'),
                      Line2D([0], [0], color='green', lw=2, label='Y-axis'),
                      Line2D([0], [0], color='blue', lw=2, label='Z-axis'),
                      Line2D([0], [0], color='black', lw=1, label='Kinematic Chain')]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.show()
    
    return parser, frame_positions


def print_frame_summary(parser, frame_positions):
    """Print a summary of frame positions relative to palm_lower."""
    print("\n=== FRAME POSITIONS RELATIVE TO PALM_LOWER ===")
    print(f"{'Frame Name':<20} {'X (mm)':<10} {'Y (mm)':<10} {'Z (mm)':<10}")
    print("-" * 60)
    
    for frame, position in frame_positions.items():
        x, y, z = position * 1000  # Convert to mm
        print(f"{frame:<20} {x:8.2f}   {y:8.2f}   {z:8.2f}")
    
    # Calculate fingertip positions
    print(f"\n=== FINGERTIP POSITIONS ===")
    fingertips = ["fingertip", "fingertip_2", "fingertip_3", "thumb_fingertip"]
    fingertip_names = ["Index", "Middle", "Ring", "Thumb"]
    
    for i, fingertip in enumerate(fingertips):
        if fingertip in frame_positions:
            x, y, z = frame_positions[fingertip] * 1000  # Convert to mm
            print(f"{fingertip_names[i]:<10} fingertip: ({x:7.2f}, {y:7.2f}, {z:7.2f}) mm")
    
    # Calculate finger spans
    print(f"\n=== FINGER SPAN ANALYSIS ===")
    if "fingertip" in frame_positions and "thumb_fingertip" in frame_positions:
        thumb_pos = frame_positions["thumb_fingertip"]
        index_pos = frame_positions["fingertip"]
        span = np.linalg.norm(thumb_pos - index_pos) * 1000
        print(f"Thumb-to-Index span: {span:.2f} mm")
    
    if "fingertip" in frame_positions and "fingertip_3" in frame_positions:
        index_pos = frame_positions["fingertip"]
        ring_pos = frame_positions["fingertip_3"]
        span = np.linalg.norm(index_pos - ring_pos) * 1000
        print(f"Index-to-Ring span: {span:.2f} mm")


def demonstrate_transformation_calculation():
    """Demonstrate step-by-step transformation calculation."""
    print("\n=== TRANSFORMATION MATRIX CALCULATION EXAMPLE ===")
    print("Let's trace the transformation from palm_lower to fingertip (index finger)")
    
    urdf_path = "/Users/jke/repo/LEAP_Hand_Sim/assets/leap_hand/robot.urdf"
    parser = URDFFrameParser(urdf_path)
    
    # Manual step-by-step calculation
    print("\nStep-by-step calculation:")
    print("Path: palm_lower -> mcp_joint -> pip -> dip -> fingertip")
    
    T_cumulative = np.eye(4)
    path = ["palm_lower", "mcp_joint", "pip", "dip", "fingertip"]
    
    for i in range(len(path) - 1):
        parent = path[i]
        child = path[i + 1]
        
        if (parent, child) in parser.transformations:
            T_joint = parser.transformations[(parent, child)]
            T_cumulative = T_cumulative @ T_joint
            
            print(f"\nStep {i+1}: {parent} -> {child}")
            print(f"Joint translation: {T_joint[:3, 3]}")
            print(f"Cumulative position: {T_cumulative[:3, 3]}")
    
    print(f"\nFinal transformation matrix:")
    print(T_cumulative)
    
    # Compare with direct calculation
    T_direct = parser.get_transformation_chain("palm_lower", "fingertip")
    print(f"\nDirect calculation result:")
    print(T_direct)
    
    print(f"\nDifference (should be near zero):")
    print(np.abs(T_cumulative - T_direct).max())


if __name__ == "__main__":
    print("=== LEAP HAND FRAME VISUALIZATION ===")
    
    # Create visualization
    parser, positions = visualize_hand_frames()
    
    # Print summary
    print_frame_summary(parser, positions)
    
    # Demonstrate calculation
    demonstrate_transformation_calculation()