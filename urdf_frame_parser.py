#!/usr/bin/env python3
"""
URDF Frame Parser - Extract coordinate frames and compute transformation matrices
This script parses the LEAP Hand URDF and creates transformation matrices between frames.
"""

import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Dict, List, Tuple, Optional
import os


class URDFFrameParser:
    def __init__(self, urdf_path: str):
        """
        Initialize the URDF frame parser.
        
        Args:
            urdf_path: Path to the URDF file
        """
        self.urdf_path = urdf_path
        self.tree = ET.parse(urdf_path)
        self.root = self.tree.getroot()
        
        # Data structures to store frame information
        self.links = {}  # link_name -> link_data
        self.joints = {}  # joint_name -> joint_data
        self.frame_tree = {}  # child_frame -> parent_frame
        self.transformations = {}  # (parent, child) -> 4x4 transformation matrix
        
        # Parse the URDF
        self._parse_links()
        self._parse_joints()
        self._build_frame_tree()
        self._calculate_transformations()
    
    def _parse_links(self):
        """Parse all links from the URDF."""
        for link in self.root.findall('link'):
            link_name = link.get('name')
            self.links[link_name] = {
                'name': link_name,
                'visual': self._parse_origin(link.find('visual')),
                'collision': self._parse_origin(link.find('collision')),
                'inertial': self._parse_origin(link.find('inertial'))
            }
            print(f"Found link: {link_name}")
    
    def _parse_joints(self):
        """Parse all joints from the URDF."""
        for joint in self.root.findall('joint'):
            joint_name = joint.get('name')
            joint_type = joint.get('type')
            
            parent = joint.find('parent').get('link')
            child = joint.find('child').get('link')
            origin = self._parse_origin(joint)
            
            axis_elem = joint.find('axis')
            axis = [0, 0, 1]  # default axis
            if axis_elem is not None:
                axis_str = axis_elem.get('xyz')
                axis = [float(x) for x in axis_str.split()]
            
            limit_elem = joint.find('limit')
            limits = None
            if limit_elem is not None:
                limits = {
                    'lower': float(limit_elem.get('lower', '0')),
                    'upper': float(limit_elem.get('upper', '0')),
                    'effort': float(limit_elem.get('effort', '0')),
                    'velocity': float(limit_elem.get('velocity', '0'))
                }
            
            self.joints[joint_name] = {
                'name': joint_name,
                'type': joint_type,
                'parent': parent,
                'child': child,
                'origin': origin,
                'axis': axis,
                'limits': limits
            }
            print(f"Found joint: {joint_name} ({parent} -> {child})")
    
    def _parse_origin(self, element) -> Dict:
        """Parse origin element (xyz and rpy)."""
        if element is None:
            return {'xyz': [0, 0, 0], 'rpy': [0, 0, 0]}
        
        origin_elem = element.find('origin')
        if origin_elem is None:
            return {'xyz': [0, 0, 0], 'rpy': [0, 0, 0]}
        
        xyz_str = origin_elem.get('xyz', '0 0 0')
        rpy_str = origin_elem.get('rpy', '0 0 0')
        
        xyz = [float(x) for x in xyz_str.split()]
        rpy = [float(x) for x in rpy_str.split()]
        
        return {'xyz': xyz, 'rpy': rpy}
    
    def _build_frame_tree(self):
        """Build the frame tree showing parent-child relationships."""
        for joint_name, joint_data in self.joints.items():
            parent = joint_data['parent']
            child = joint_data['child']
            self.frame_tree[child] = parent
            print(f"Frame relationship: {parent} -> {child} (via joint {joint_name})")
    
    def xyz_rpy_to_matrix(self, xyz: List[float], rpy: List[float]) -> np.ndarray:
        """
        Convert xyz translation and rpy rotation to 4x4 transformation matrix.
        
        Args:
            xyz: [x, y, z] translation in meters
            rpy: [roll, pitch, yaw] rotation in radians
            
        Returns:
            4x4 transformation matrix
        """
        # Create rotation matrix from roll, pitch, yaw
        rotation = R.from_euler('xyz', rpy, degrees=False)
        rotation_matrix = rotation.as_matrix()
        
        # Create 4x4 transformation matrix
        T = np.eye(4)
        T[:3, :3] = rotation_matrix
        T[:3, 3] = xyz
        
        return T
    
    def _calculate_transformations(self):
        """Calculate transformation matrices between connected frames."""
        for joint_name, joint_data in self.joints.items():
            parent = joint_data['parent']
            child = joint_data['child']
            origin = joint_data['origin']
            
            # Calculate transformation matrix from parent to child
            T = self.xyz_rpy_to_matrix(origin['xyz'], origin['rpy'])
            self.transformations[(parent, child)] = T
            
            print(f"Transformation {parent} -> {child}:")
            print(f"  Translation: {origin['xyz']}")
            print(f"  Rotation (RPY): {origin['rpy']}")
            print(f"  Matrix:\n{T}\n")
    
    def get_transformation_chain(self, from_frame: str, to_frame: str) -> Optional[np.ndarray]:
        """
        Calculate transformation matrix from one frame to another through the kinematic chain.
        
        Args:
            from_frame: Source frame name
            to_frame: Target frame name
            
        Returns:
            4x4 transformation matrix or None if no path exists
        """
        # Find path from from_frame to to_frame
        path = self._find_path(from_frame, to_frame)
        if not path:
            return None
        
        # Compute cumulative transformation
        T_cumulative = np.eye(4)
        
        for i in range(len(path) - 1):
            parent_frame = path[i]
            child_frame = path[i + 1]
            
            if (parent_frame, child_frame) in self.transformations:
                T = self.transformations[(parent_frame, child_frame)]
                T_cumulative = T_cumulative @ T
            elif (child_frame, parent_frame) in self.transformations:
                # Reverse transformation (inverse)
                T = self.transformations[(child_frame, parent_frame)]
                T_inv = np.linalg.inv(T)
                T_cumulative = T_cumulative @ T_inv
            else:
                print(f"No transformation found between {parent_frame} and {child_frame}")
                return None
        
        return T_cumulative
    
    def _find_path(self, from_frame: str, to_frame: str) -> List[str]:
        """
        Find path between two frames in the kinematic tree.
        
        Args:
            from_frame: Source frame
            to_frame: Target frame
            
        Returns:
            List of frame names forming the path, or empty list if no path
        """
        # Simple implementation: find path to root, then from root to target
        # This could be optimized for direct paths
        
        def path_to_root(frame):
            path = [frame]
            current = frame
            while current in self.frame_tree:
                parent = self.frame_tree[current]
                path.append(parent)
                current = parent
            return path
        
        path_from = path_to_root(from_frame)
        path_to = path_to_root(to_frame)
        
        # Find common ancestor
        common_ancestor = None
        for frame in path_from:
            if frame in path_to:
                common_ancestor = frame
                break
        
        if common_ancestor is None:
            return []
        
        # Build path: from_frame -> common_ancestor -> to_frame
        path_up = []
        current = from_frame
        while current != common_ancestor:
            path_up.append(current)
            current = self.frame_tree[current]
        path_up.append(common_ancestor)
        
        path_down = []
        current = to_frame
        while current != common_ancestor:
            path_down.append(current)
            current = self.frame_tree[current]
        
        # Reverse path_down and combine
        path_down.reverse()
        full_path = path_up + path_down
        
        return full_path
    
    def get_all_frames(self) -> List[str]:
        """Get list of all frame names."""
        return list(self.links.keys())
    
    def print_frame_tree(self):
        """Print the frame tree structure."""
        print("\n=== FRAME TREE STRUCTURE ===")
        
        # Find root frame (no parent)
        all_children = set(self.frame_tree.keys())
        all_parents = set(self.frame_tree.values())
        roots = all_parents - all_children
        
        def print_subtree(frame, indent=0):
            prefix = "  " * indent
            print(f"{prefix}{frame}")
            
            # Find children
            children = [child for child, parent in self.frame_tree.items() if parent == frame]
            for child in children:
                print_subtree(child, indent + 1)
        
        for root in roots:
            print_subtree(root)
    
    def get_frame_data_dict(self) -> Dict:
        """
        Get all frame data as a comprehensive dictionary.
        
        Returns:
            Dictionary containing all frame relationships and transformations
        """
        return {
            'links': self.links,
            'joints': self.joints,
            'frame_tree': self.frame_tree,
            'transformations': {f"{k[0]}_to_{k[1]}": v.tolist() for k, v in self.transformations.items()},
            'root_frames': list(set(self.frame_tree.values()) - set(self.frame_tree.keys()))
        }


def demonstrate_transformations():
    """Demonstrate the URDF frame parser with the LEAP Hand."""
    
    # Path to URDF file
    urdf_path = "/Users/jke/repo/LEAP_Hand_Sim/assets/leap_hand/robot.urdf"
    
    if not os.path.exists(urdf_path):
        print(f"URDF file not found: {urdf_path}")
        return
    
    print("=== LEAP HAND URDF FRAME ANALYSIS ===\n")
    
    # Parse URDF
    parser = URDFFrameParser(urdf_path)
    
    # Print frame tree
    parser.print_frame_tree()
    
    # Get frame data dictionary
    frame_data = parser.get_frame_data_dict()
    print(f"\n=== FRAME DATA SUMMARY ===")
    print(f"Total links: {len(frame_data['links'])}")
    print(f"Total joints: {len(frame_data['joints'])}")
    print(f"Root frames: {frame_data['root_frames']}")
    
    # Example: Calculate transformation from palm to fingertip
    print(f"\n=== EXAMPLE TRANSFORMATIONS ===")
    
    # Get all frames
    frames = parser.get_all_frames()
    palm_frame = "palm_lower"
    fingertip_frame = "fingertip"
    
    if palm_frame in frames and fingertip_frame in frames:
        T = parser.get_transformation_chain(palm_frame, fingertip_frame)
        if T is not None:
            print(f"\nTransformation from {palm_frame} to {fingertip_frame}:")
            print(f"Translation: {T[:3, 3]}")
            print(f"Rotation matrix:\n{T[:3, :3]}")
            print(f"Full transformation matrix:\n{T}")
        else:
            print(f"Could not find path from {palm_frame} to {fingertip_frame}")
    
    # Show direct joint transformations
    print(f"\n=== DIRECT JOINT TRANSFORMATIONS ===")
    for joint_name, joint_data in parser.joints.items():
        parent = joint_data['parent']
        child = joint_data['child']
        if (parent, child) in parser.transformations:
            T = parser.transformations[(parent, child)]
            print(f"\nJoint {joint_name}: {parent} -> {child}")
            print(f"Translation: {T[:3, 3]}")
            print(f"Rotation (Euler angles): {R.from_matrix(T[:3, :3]).as_euler('xyz', degrees=True)} degrees")
    
    return parser


if __name__ == "__main__":
    parser = demonstrate_transformations()