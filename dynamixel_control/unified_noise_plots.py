#!/usr/bin/env python3
"""
Unified noise data plotter that combines data from two different experiments:
1. Original noise experiments (from data/noise/results_tables/)
2. Gripper noise experiments (from data/noise_gripper/experiment_master_log.txt)

Creates combined visualizations showing both datasets together.
"""

from matplotlib.transforms import Affine2D
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.path import Path
import matplotlib.patches as mpatches
from scipy.spatial import ConvexHull
from scipy.interpolate import splprep, splev
import warnings

# Set up matplotlib for better plots
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Gyre Termes"],
    'font.size': 24,               # Base font size
    'axes.titlesize': 24,          # Title size
    'axes.labelsize': 24,          # X/Y axis labels
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 24})


def read_original_noise_data(base_path="data/noise/results_tables/"):
    """
    Read the original noise experiment data from text files.
    
    Args:
        base_path (str): Base path to the noise data directory
        
    Returns:
        pandas.DataFrame: DataFrame with noise experiment data
    """
    directions_file = os.path.join(base_path, "directions.txt")
    success_file = os.path.join(base_path, "success.txt")
    
    if not os.path.exists(directions_file) or not os.path.exists(success_file):
        print(f"Warning: Original noise data files not found at {base_path}")
        return pd.DataFrame()
    
    # Read angles and success data
    with open(directions_file) as f:
        noise_vectors_angles = f.readlines()
    with open(success_file) as f:
        success_list = f.readlines()
    
    # Parse data
    noise_vectors_angles_array = np.empty((10, 3))
    for i, line in enumerate(noise_vectors_angles):
        noise_vectors_angles_array[i, :] = line.split()
    
    success_array = np.empty((10, 3))
    for i, line in enumerate(success_list):
        success_array[i, :] = line.split()
    
    # Define magnitudes for each column
    magnitudes = [0.05, 0.1, 0.15]
    
    # Convert to DataFrame format
    data = []
    for col_idx, magnitude in enumerate(magnitudes):
        for row_idx in range(10):
            angle_rad = noise_vectors_angles_array[row_idx, col_idx]
            angle_deg = np.degrees(angle_rad) % 360
            x_noise = magnitude * np.cos(angle_rad)
            y_noise = magnitude * np.sin(angle_rad)
            success = bool(success_array[row_idx, col_idx])
            
            data.append({
                'experiment_id': f"orig_{col_idx}_{row_idx}",
                'source': 'original',
                'random_direction_rad': angle_rad,
                'random_direction_deg': angle_deg,
                'noise_magnitude': magnitude,
                'x_noise': x_noise,
                'y_noise': y_noise,
                'success': success
            })
    
    return pd.DataFrame(data)


def read_gripper_noise_data(log_file_path="data/noise_gripper/experiment_master_log.txt"):
    """
    Read experiment data from the gripper noise master log file.
    
    Args:
        log_file_path (str): Path to the master log file
        
    Returns:
        pandas.DataFrame: DataFrame with experiment data
    """
    if not os.path.exists(log_file_path):
        print(f"Warning: Gripper noise data file not found at {log_file_path}")
        return pd.DataFrame()
    
    data = []
    with open(log_file_path, 'r') as f:
        lines = f.readlines()
    
    # Filter out comment lines
    data_lines = [line.strip() for line in lines if not line.startswith('#') and line.strip()]
    
    for line in data_lines:
        parts = [part.strip() for part in line.split(',')]
        if len(parts) >= 8:
            try:
                experiment_id = int(parts[0])
                timestamp = parts[1]
                random_direction_rad = float(parts[2])
                random_direction_deg = float(parts[3])
                noise_magnitude = float(parts[4])
                x_noise = float(parts[5])
                y_noise = float(parts[6])
                success = parts[7].lower() == 'true'
                
                data.append({
                    'experiment_id': experiment_id,
                    'source': 'gripper',
                    'timestamp': timestamp,
                    'random_direction_rad': random_direction_rad,
                    'random_direction_deg': random_direction_deg,
                    'noise_magnitude': noise_magnitude,
                    'x_noise': x_noise,
                    'y_noise': y_noise,
                    'success': success
                })
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse line: {line}")
                continue
    
    return pd.DataFrame(data)


def plot_combined_scatter(df_original, df_gripper, save_path=None, show_plot=True):
    """
    Create a combined scatter plot showing both datasets.
    
    Args:
        df_original (pandas.DataFrame): Original noise experiment data
        df_gripper (pandas.DataFrame): Gripper noise experiment data
        save_path (str, optional): Path to save the plot
        show_plot (bool): Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot original data
    if not df_original.empty:
        orig_success = df_original[df_original['success'] == True]
        orig_failure = df_original[df_original['success'] == False]
        
        if not orig_success.empty:
            ax.scatter(orig_success['x_noise'], orig_success['y_noise'], 
                      color='darkgreen', s=80, marker='o', alpha=0.7, 
                      label=f'Original Success ({len(orig_success)})', 
                      edgecolors='black', linewidth=1.5)
        
        if not orig_failure.empty:
            ax.scatter(orig_failure['x_noise'], orig_failure['y_noise'], 
                      color='darkred', s=80, marker='x', alpha=0.7, 
                      label=f'Original Failure ({len(orig_failure)})', linewidth=2.5)
    
    # Plot gripper data
    if not df_gripper.empty:
        grip_success = df_gripper[df_gripper['success'] == True]
        grip_failure = df_gripper[df_gripper['success'] == False]
        
        if not grip_success.empty:
            ax.scatter(grip_success['x_noise'], grip_success['y_noise'], 
                      color='lightgreen', s=60, marker='o', alpha=0.6, 
                      label=f'Gripper Success ({len(grip_success)})', 
                      edgecolors='green', linewidth=1)
        
        if not grip_failure.empty:
            ax.scatter(grip_failure['x_noise'], grip_failure['y_noise'], 
                      color='lightcoral', s=60, marker='x', alpha=0.6, 
                      label=f'Gripper Failure ({len(grip_failure)})', linewidth=2)
    
    # Get all data for determining radius ranges
    df_combined = pd.concat([df_original, df_gripper], ignore_index=True)
    
    if not df_combined.empty:
        # Add concentric circles for magnitude levels
        magnitudes = sorted(df_combined['noise_magnitude'].unique())
        for magnitude in magnitudes:
            circle = Circle((0, 0), magnitude, color='black', alpha=0.3, 
                          fill=False, linestyle='--', linewidth=1)
            ax.add_patch(circle)
            # Add label for each circle
            ax.text(magnitude * 0.707, magnitude * 0.707, f'{magnitude:.3f}m', 
                   fontsize=9, alpha=0.6, ha='left')
    
    # Add grid lines
    ax.axhline(0, color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.axvline(0, color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # Formatting
    ax.set_xlabel("X Noise [m]", fontsize=14, fontweight='bold')
    ax.set_ylabel("Y Noise [m]", fontsize=14, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_title("Combined Noise Experiments: All Data", fontsize=16, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Set axis limits
    if not df_combined.empty:
        max_range = max(abs(df_combined['x_noise'].max()), abs(df_combined['x_noise'].min()),
                       abs(df_combined['y_noise'].max()), abs(df_combined['y_noise'].min()))
        padding = max_range * 0.15
        ax.set_xlim(-max_range - padding, max_range + padding)
        ax.set_ylim(-max_range - padding, max_range + padding)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Combined scatter plot saved to: {save_path}")
        # Also save as PDF
        pdf_path = save_path.replace('.png', '.pdf')
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"Combined scatter plot saved to: {pdf_path}")
    
    if show_plot:
        plt.show()
    
    return fig, ax


def plot_combined_polar_sectors(df_original, df_gripper, save_path=None, show_plot=True, n_sectors=12):
    """
    Create a combined polar sector plot showing both datasets.
    
    Args:
        df_original (pandas.DataFrame): Original noise experiment data
        df_gripper (pandas.DataFrame): Gripper noise experiment data
        save_path (str, optional): Path to save the plot
        show_plot (bool): Whether to display the plot
        n_sectors (int): Number of angular sectors
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Combine datasets
    df_combined = pd.concat([df_original, df_gripper], ignore_index=True)
    
    if df_combined.empty:
        print("No data to plot")
        return
    
    # Get magnitude ranges
    magnitudes = sorted(df_combined['noise_magnitude'].unique())
    
    # Create magnitude bins
    if len(magnitudes) > 3:
        min_mag = df_combined['noise_magnitude'].min()
        max_mag = df_combined['noise_magnitude'].max()
        mag_ranges = np.linspace(min_mag, max_mag, 4)
    else:
        mag_ranges = [0] + magnitudes
    
    sector_angle = 360 / n_sectors
    
    # Create sector visualization for combined data
    for i in range(len(mag_ranges) - 1):
        inner_radius = mag_ranges[i]
        outer_radius = mag_ranges[i + 1]
        
        for sector in range(n_sectors):
            start_angle = sector * sector_angle
            end_angle = (sector + 1) * sector_angle
            
            # Find points in this sector and magnitude range
            sector_data = df_combined[
                (df_combined['noise_magnitude'] >= inner_radius) & 
                (df_combined['noise_magnitude'] < outer_radius) &
                (df_combined['random_direction_deg'] >= start_angle) & 
                (df_combined['random_direction_deg'] < end_angle)
            ]
            
            if not sector_data.empty:
                success_rate = sector_data['success'].mean()
                
                # Color based on success rate
                if success_rate == 1.0:
                    color = 'darkgreen'
                    alpha = 0.4
                elif success_rate >= 0.75:
                    color = 'green'
                    alpha = 0.3
                elif success_rate >= 0.5:
                    color = 'yellow'
                    alpha = 0.3
                elif success_rate >= 0.25:
                    color = 'orange'
                    alpha = 0.3
                else:
                    color = 'red'
                    alpha = 0.4
                
                # Create wedge
                wedge = Wedge(
                    center=(0, 0),
                    r=outer_radius,
                    theta1=start_angle,
                    theta2=end_angle,
                    width=outer_radius - inner_radius,
                    facecolor=color,
                    edgecolor='black',
                    alpha=alpha,
                    linewidth=0.5,
                    zorder=1
                )
                ax.add_patch(wedge)
    
    # Plot individual points on top
    if not df_original.empty:
        orig_success = df_original[df_original['success'] == True]
        orig_failure = df_original[df_original['success'] == False]
        
        if not orig_success.empty:
            ax.scatter(orig_success['x_noise'], orig_success['y_noise'], 
                      color='darkgreen', s=80, marker='o', alpha=0.8, 
                      label=f'Original Success ({len(orig_success)})', 
                      edgecolors='black', linewidth=1.5, zorder=5)
        
        if not orig_failure.empty:
            ax.scatter(orig_failure['x_noise'], orig_failure['y_noise'], 
                      color='darkred', s=80, marker='x', alpha=0.8, 
                      label=f'Original Failure ({len(orig_failure)})', 
                      linewidth=2.5, zorder=5)
    
    if not df_gripper.empty:
        grip_success = df_gripper[df_gripper['success'] == True]
        grip_failure = df_gripper[df_gripper['success'] == False]
        
        if not grip_success.empty:
            ax.scatter(grip_success['x_noise'], grip_success['y_noise'], 
                      color='lightgreen', s=60, marker='o', alpha=0.7, 
                      label=f'Gripper Success ({len(grip_success)})', 
                      edgecolors='green', linewidth=1, zorder=5)
        
        if not grip_failure.empty:
            ax.scatter(grip_failure['x_noise'], grip_failure['y_noise'], 
                      color='lightcoral', s=60, marker='x', alpha=0.7, 
                      label=f'Gripper Failure ({len(grip_failure)})', 
                      linewidth=2, zorder=5)
    
    # Add concentric circles
    for radius in mag_ranges[1:]:
        circle = Circle((0, 0), radius, color='black', alpha=0.5, 
                      fill=False, linestyle='--', linewidth=1.5, zorder=2)
        ax.add_patch(circle)
    
    # Add grid lines
    ax.axhline(0, color='grey', linestyle='-', linewidth=0.5, alpha=0.5, zorder=3)
    ax.axvline(0, color='grey', linestyle='-', linewidth=0.5, alpha=0.5, zorder=3)
    
    # Formatting
    ax.set_xlabel("X Noise [m]", fontsize=14, fontweight='bold')
    ax.set_ylabel("Y Noise [m]", fontsize=14, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_title("Combined Noise Experiments: Polar Sector Analysis", fontsize=16, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, zorder=0)
    ax.set_aspect('equal')
    
    # Set axis limits
    max_radius = max(mag_ranges) * 1.15
    ax.set_xlim(-max_radius, max_radius)
    ax.set_ylim(-max_radius, max_radius)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Combined polar plot saved to: {save_path}")
        # Also save as PDF
        pdf_path = save_path.replace('.png', '.pdf')
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"Combined polar plot saved to: {pdf_path}")
    
    if show_plot:
        plt.show()
    
    return fig, ax


def plot_side_by_side_comparison(df_original, df_gripper, save_path=None, show_plot=True):
    """
    Create side-by-side plots comparing the two datasets with the same scale.
    
    Args:
        df_original (pandas.DataFrame): Original noise experiment data
        df_gripper (pandas.DataFrame): Gripper noise experiment data
        save_path (str, optional): Path to save the plot
        show_plot (bool): Whether to display the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Determine common scale
    df_combined = pd.concat([df_original, df_gripper], ignore_index=True)
    if not df_combined.empty:
        max_range = max(abs(df_combined['x_noise']).max(), 
                       abs(df_combined['y_noise']).max()) * 1.15
    else:
        max_range = 0.1
    
    # Plot 1: Original data
    if not df_original.empty:
        orig_success = df_original[df_original['success'] == True]
        orig_failure = df_original[df_original['success'] == False]
        
        if not orig_success.empty:
            ax1.scatter(orig_success['x_noise'], orig_success['y_noise'], 
                       color='green', s=100, marker='o', alpha=0.7, 
                       label=f'Success ({len(orig_success)})', 
                       edgecolors='darkgreen', linewidth=1.5)
        
        if not orig_failure.empty:
            ax1.scatter(orig_failure['x_noise'], orig_failure['y_noise'], 
                       color='red', s=100, marker='x', alpha=0.7, 
                       label=f'Failure ({len(orig_failure)})', linewidth=2.5)
        
        # Add circles for original data
        for magnitude in [0.05, 0.1, 0.15]:
            circle = Circle((0, 0), magnitude, color='black', alpha=0.3, 
                          fill=False, linestyle='--', linewidth=1)
            ax1.add_patch(circle)
    
    ax1.axhline(0, color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
    ax1.axvline(0, color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
    ax1.set_xlabel("X Noise [m]", fontsize=13, fontweight='bold')
    ax1.set_ylabel("Y Noise [m]", fontsize=13, fontweight='bold')
    ax1.set_title("Original Noise Experiments", fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    ax1.set_xlim(-max_range, max_range)
    ax1.set_ylim(-max_range, max_range)
    
    # Plot 2: Gripper data (no radius circles)
    if not df_gripper.empty:
        grip_success = df_gripper[df_gripper['success'] == True]
        grip_failure = df_gripper[df_gripper['success'] == False]
        
        if not grip_success.empty:
            ax2.scatter(grip_success['x_noise'], grip_success['y_noise'], 
                       color='green', s=100, marker='o', alpha=0.7, 
                       label=f'Success ({len(grip_success)})', 
                       edgecolors='darkgreen', linewidth=1.5)
        
        if not grip_failure.empty:
            ax2.scatter(grip_failure['x_noise'], grip_failure['y_noise'], 
                       color='red', s=100, marker='x', alpha=0.7, 
                       label=f'Failure ({len(grip_failure)})', linewidth=2.5)
    
    ax2.axhline(0, color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
    ax2.axvline(0, color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
    ax2.set_xlabel("X Noise [m]", fontsize=13, fontweight='bold')
    ax2.set_ylabel("Y Noise [m]", fontsize=13, fontweight='bold')
    ax2.set_title("Gripper Noise Experiments", fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    ax2.set_xlim(-max_range, max_range)
    ax2.set_ylim(-max_range, max_range)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Side-by-side comparison saved to: {save_path}")
        # Also save as PDF
        pdf_path = save_path.replace('.png', '.pdf')
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"Side-by-side comparison saved to: {pdf_path}")
    
    if show_plot:
        plt.show()
    
    return fig, (ax1, ax2)


def find_success_regions(df, min_points=3, shift_vertices=None):
    """
    Find contiguous regions of successful experiments.
    Uses a greedy approach to find the largest polygon of successes with no failures inside.
    
    Args:
        df (pandas.DataFrame): Experiment data
        min_points (int): Minimum points needed to form a polygon
        shift_vertices (dict): Optional dict mapping vertex indices to shift amounts (e.g., {1: 0.01, 4: 0.01})
        
    Returns:
        list: List of arrays containing vertices of success regions
    """
    if df.empty:
        return []
    
    success_df = df[df['success'] == True].copy()
    failure_df = df[df['success'] == False].copy()
    
    if len(success_df) < min_points:
        return []
    
    # Get success points
    success_points = success_df[['x_noise', 'y_noise']].values
    failure_points = failure_df[['x_noise', 'y_noise']].values if not failure_df.empty else np.array([])
    
    regions = []
    
    # Try to find convex hull of all successes first
    if len(success_points) >= min_points:
        try:
            hull = ConvexHull(success_points)
            # Get hull points in the correct order
            # ConvexHull.vertices gives indices, but not necessarily in order
            # We need to use hull.simplices to get the correct ordering
            hull_points = success_points[hull.vertices]
            
            # Sort vertices by angle from centroid to ensure proper ordering
            centroid = np.mean(hull_points, axis=0)
            angles = np.arctan2(hull_points[:, 1] - centroid[1], 
                               hull_points[:, 0] - centroid[0])
            sorted_indices = np.argsort(angles)
            hull_points = hull_points[sorted_indices]
            
            # Apply vertex shifts if specified (for drawing purposes)
            if shift_vertices is not None:
                for vertex_idx, shift_amount in shift_vertices.items():
                    if vertex_idx < len(hull_points):
                        # Shift outward from centroid
                        direction = hull_points[vertex_idx] - centroid
                        direction_norm = np.linalg.norm(direction)
                        if direction_norm > 0:
                            direction = direction / direction_norm
                            hull_points[vertex_idx] = hull_points[vertex_idx] + direction * shift_amount
            
            # Debug: print which points are on the hull
            print(f"  Convex hull has {len(hull_points)} vertices out of {len(success_points)} success points")
            print(f"  Hull vertices (x, y) in order:")
            for i, pt in enumerate(hull_points):
                print(f"    {i}: ({pt[0]:.6f}, {pt[1]:.6f})")
            
            # Check if any failures are inside this hull
            if len(failure_points) > 0:
                from matplotlib.path import Path
                hull_path = Path(hull_points)
                failures_inside = hull_path.contains_points(failure_points)
                
                print(f"  Checking {len(failure_points)} failure points...")
                print(f"  Failures inside hull: {np.sum(failures_inside)}")
                if np.any(failures_inside):
                    print(f"  Failure points inside hull:")
                    for i, (fail_pt, is_inside) in enumerate(zip(failure_points, failures_inside)):
                        if is_inside:
                            print(f"    ({fail_pt[0]:.6f}, {fail_pt[1]:.6f})")
                
                if not np.any(failures_inside):
                    # No failures inside - this is a valid region
                    regions.append(hull_points)
                    return regions
                else:
                    print(f"  WARNING: Hull contains failures, not using as success region")
            else:
                # No failures at all - hull is valid
                regions.append(hull_points)
                return regions
        except Exception as e:
            warnings.warn(f"Could not compute convex hull: {e}")
    
    # If global hull contains failures, try to find smaller regions
    # Group success points by angular sectors and magnitude ranges
    if len(success_points) >= min_points:
        # Convert to polar coordinates
        success_r = np.sqrt(success_points[:, 0]**2 + success_points[:, 1]**2)
        success_theta = np.arctan2(success_points[:, 1], success_points[:, 0])
        
        # Try different sector configurations
        for n_sectors in [8, 12, 16]:
            sector_size = 2 * np.pi / n_sectors
            
            for sector_i in range(n_sectors):
                sector_start = sector_i * sector_size - np.pi
                sector_end = (sector_i + 1) * sector_size - np.pi
                
                # Find points in this sector
                in_sector = ((success_theta >= sector_start) & (success_theta < sector_end))
                sector_points = success_points[in_sector]
                
                if len(sector_points) >= min_points:
                    try:
                        hull = ConvexHull(sector_points)
                        hull_points = sector_points[hull.vertices]
                        
                        # Check if any failures are inside
                        if len(failure_points) > 0:
                            from matplotlib.path import Path
                            hull_path = Path(hull_points)
                            failures_inside = hull_path.contains_points(failure_points)
                            
                            if not np.any(failures_inside):
                                regions.append(hull_points)
                        else:
                            regions.append(hull_points)
                    except:
                        continue
    
    return regions


def create_rounded_polygon_with_straight_edge(vertices, straight_edge_indices=None):
    """
    Create a polygon with straight edges and rounded corners using circular arcs.
    The arcs smoothly connect to straight edges with proper tangent continuity.
    
    Args:
        vertices (np.array): Array of (x, y) coordinates (convex hull vertices)
        straight_edge_indices (list): Not used, kept for compatibility
        
    Returns:
        np.array: Polygon with straight edges and circular arc corners
    """
    if len(vertices) < 3:
        return vertices
    
    n = len(vertices)
    result_points = []
    arc_radius = 0.003  # Arc radius in meters (radius of the circular arc)
    points_per_corner = 16  # Points to create each circular arc corner
    
    for i in range(n):
        prev_vertex = vertices[(i - 1) % n]
        curr_vertex = vertices[i]
        next_vertex = vertices[(i + 1) % n]
        
        # Vectors from current vertex to neighbors
        vec_to_prev = prev_vertex - curr_vertex
        vec_to_next = next_vertex - curr_vertex
        
        # Normalize
        len_to_prev = np.linalg.norm(vec_to_prev)
        len_to_next = np.linalg.norm(vec_to_next)
        
        if len_to_prev > 0:
            vec_to_prev = vec_to_prev / len_to_prev
        if len_to_next > 0:
            vec_to_next = vec_to_next / len_to_next
        
        # Calculate the angle between the two edges
        dot_product = np.dot(vec_to_prev, vec_to_next)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle_between = np.arccos(dot_product)
        half_angle = angle_between / 2
        
        # Calculate tangent distance: how far from vertex the arc touches the edges
        # Using tangent_distance = arc_radius / tan(half_angle)
        if np.sin(half_angle) > 0.01:  # avoid division by zero
            tangent_distance = arc_radius / np.tan(half_angle)
        else:
            tangent_distance = arc_radius * 10  # large distance for nearly straight edge
        
        # Limit tangent distance to avoid overlapping with other corners
        max_tangent = min(len_to_prev * 0.4, len_to_next * 0.4)
        tangent_distance = min(tangent_distance, max_tangent)
        
        # If we had to limit it, recalculate the actual arc radius
        actual_arc_radius = tangent_distance * np.tan(half_angle)
        
        # Points on edges where the arc touches (tangent points)
        tangent_prev = curr_vertex + vec_to_prev * tangent_distance
        tangent_next = curr_vertex + vec_to_next * tangent_distance
        
        # Add the straight edge endpoint (tangent point)
        result_points.append(tangent_prev)
        
        # Find arc center: it's perpendicular to the edge at the tangent point
        # The bisector points outward from the corner
        bisector = vec_to_prev + vec_to_next
        bisector_len = np.linalg.norm(bisector)
        if bisector_len > 0:
            bisector = bisector / bisector_len
        else:
            # Handle 180-degree angle case
            bisector = np.array([-vec_to_prev[1], vec_to_prev[0]])
        
        # Arc center is at distance actual_arc_radius along the bisector from vertex
        # But more precisely, it's at distance actual_arc_radius/sin(half_angle) from vertex
        if np.sin(half_angle) > 0.01:
            center_distance = actual_arc_radius / np.sin(half_angle)
        else:
            center_distance = actual_arc_radius * 10
            
        arc_center = curr_vertex + bisector * center_distance
        
        # Generate points along the circular arc
        angle_start = np.arctan2((tangent_prev - arc_center)[1], (tangent_prev - arc_center)[0])
        angle_end = np.arctan2((tangent_next - arc_center)[1], (tangent_next - arc_center)[0])
        
        # Ensure we take the shorter arc
        angle_diff = angle_end - angle_start
        if angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        elif angle_diff < -np.pi:
            angle_diff += 2 * np.pi
        
        for j in range(1, points_per_corner):
            t = j / points_per_corner
            angle = angle_start + t * angle_diff
            arc_point = arc_center + actual_arc_radius * np.array([np.cos(angle), np.sin(angle)])
            result_points.append(arc_point)
    
    return np.array(result_points)


def create_rounded_polygon_safe(vertices, failure_points=None):
    """
    Create polygon with straight edges and slightly rounded corners.
    """
    return create_rounded_polygon_with_straight_edge(vertices)


def create_rounded_polygon_simple(vertices):
    """
    Create a smooth polygon that passes through all vertices using spline interpolation.
    
    Args:
        vertices (np.array): Array of (x, y) coordinates
        
    Returns:
        np.array: Smoothed vertices
    """
    if len(vertices) < 3:
        return vertices
    
    # Close the polygon by appending the first few points
    vertices_closed = np.vstack([vertices, vertices[:2]])
    
    try:
        tck, u = splprep([vertices_closed[:, 0], vertices_closed[:, 1]], 
                         s=0,  # Pass through all points exactly
                         k=min(3, len(vertices) - 1),
                         per=False)
        
        # Evaluate the spline at many points for smooth curve
        u_new = np.linspace(0, 1, len(vertices) * 20)
        x_new, y_new = splev(u_new, tck)
        
        result = np.column_stack([x_new, y_new])
        
        # Trim to avoid the closing duplication
        n_trim = int(len(result) * 2 / len(vertices))
        return result[:-n_trim]
        
    except Exception as e:
        warnings.warn(f"Spline interpolation failed: {e}")
        return vertices


def create_rounded_polygon(vertices, radius=0.01):
    """
    Wrapper for backward compatibility.
    """
    return create_rounded_polygon_simple(vertices)


def plot_success_polygons(df_original, df_gripper, save_path=None, show_plot=True,bg_img_path=None):
    """
    Create a plot showing the largest polygons of connected successes with no failures inside.
    
    Args:
        df_original (pandas.DataFrame): Original noise experiment data
        df_gripper (pandas.DataFrame): Gripper noise experiment data
        save_path (str, optional): Path to save the plot
        show_plot (bool): Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Get failure points for both datasets
    orig_failure_points = df_original[df_original['success'] == False][['x_noise', 'y_noise']].values if not df_original.empty else np.array([])
    grip_failure_points = df_gripper[df_gripper['success'] == False][['x_noise', 'y_noise']].values if not df_gripper.empty else np.array([])
    
    # Find success regions for both datasets
    print("  Finding success regions for original data...")
    # Shift vertices 1 and 4 outward by 0.01m for drawing purposes
    original_regions = find_success_regions(df_original, shift_vertices={0: 0.0025, 3: 0.0025}) if not df_original.empty else []
    
    print("  Finding success regions for gripper data...")
    gripper_regions = find_success_regions(df_gripper) if not df_gripper.empty else []
    
    # Plot original success regions first (bottom layer) with lower alpha
    for i, region in enumerate(original_regions):
        # Create smooth curve through vertices, with straight lines where curves would enclose failures
        rounded_region = create_rounded_polygon_safe(region, orig_failure_points)
        rounded_region_swapped = [(y, x) for (x, y) in rounded_region]
        polygon = Polygon(rounded_region_swapped, alpha=0.25, facecolor='darkgreen', 
                         edgecolor='darkgreen', linewidth=3, 
                         label='Hybrid success region' if i == 0 else None,
                         zorder=1)
        ax.add_patch(polygon)
    
    # Plot gripper success regions on top with stronger alpha (more prominent)
    for i, region in enumerate(gripper_regions):
        # Create smooth curve through vertices, with straight lines where curves would enclose failures
        rounded_region = create_rounded_polygon_safe(region, grip_failure_points)
        rounded_region_swapped = [(y, x) for (x, y) in rounded_region]
        polygon = Polygon(rounded_region_swapped, alpha=0.4, facecolor='dodgerblue', 
                         edgecolor='blue', linewidth=3, linestyle='--',
                         label='Gripper success region' if i == 0 else None,
                         zorder=2)
        ax.add_patch(polygon)
    
    # Plot individual points
    if not df_original.empty:
        orig_success = df_original[df_original['success'] == True]
        orig_failure = df_original[df_original['success'] == False]
        
        if not orig_success.empty:
            ax.scatter(orig_success['y_noise'], orig_success['x_noise'], 
                      color='green', s=300, marker='*', alpha=1.0, 
                      label=f'Hybrid success', 
                      edgecolors='darkgreen', linewidths=1.5, zorder=6)
        
        if not orig_failure.empty:
            ax.scatter(orig_failure['y_noise'], orig_failure['x_noise'], 
                      color='red', s=150, marker='x', alpha=1.0, 
                      label=f'Hybrid failure', 
                      linewidths=3, zorder=6)
    
    if not df_gripper.empty:
        grip_success = df_gripper[df_gripper['success'] == True]
        grip_failure = df_gripper[df_gripper['success'] == False]
        
        if not grip_success.empty:
            ax.scatter(grip_success['y_noise'], grip_success['x_noise'], 
                      color='blue', s=100, marker='o', alpha=1.0, 
                      label=f'Gripper success', 
                      edgecolors='darkblue', linewidth=2, zorder=5)
        
        if not grip_failure.empty:
            ax.scatter(grip_failure['y_noise'], grip_failure['x_noise'], 
                      color='darkorange', s=300, marker='+', alpha=1.0, 
                      label=f'Gripper failure', 
                      linewidths=3.5, zorder=5)
    
    # No reference circles for the polygon plot
    df_combined = pd.concat([df_original, df_gripper], ignore_index=True)
    
    # Add grid lines
    ax.axhline(0, color='grey', linestyle='-', linewidth=0.5, alpha=0.5, zorder=0)
    ax.axvline(0, color='grey', linestyle='-', linewidth=0.5, alpha=0.5, zorder=0)
    
    # Formatting
    ax.set_xlabel("y noise [m]", fontsize=22, fontweight='bold')
    ax.set_ylabel("x noise [m]", fontsize=22, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=18)
    # ax.set_title("Success Region Analysis: Largest Polygons with No Failures Inside", 
                # fontsize=16, fontweight='bold')
    ax.legend(fontsize=16, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, zorder=0)
    ax.set_aspect('equal')
    
    # Set axis limits
    if not df_combined.empty:
        max_range = max(abs(df_combined['x_noise']).max(), 
                       abs(df_combined['y_noise']).max()) * 1.15
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)

    # background setting
    if bg_img_path and os.path.isfile(bg_img_path):
        bg_scale = 1.2  # Scale factor for background image size
        bg_img = plt.imread(bg_img_path)
        bg_img = np.flipud(bg_img)  # Flip image vertically if needed
        img_h, img_w = bg_img.shape[:2]
        # Scale to fit axes
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        scale = 0.0001*0.9
        # Optional shift
        dx, dy = 30, 70
        # Rotation in radians
        import math
        # angle = math.radians(180)  # rotate 10 degrees counterclockwise

        # Center of image in data coordinates
        cx = (xlim[0] + xlim[1]) / 2
        cy = (ylim[0] + ylim[1]) / 2

        # Affine transform: scale, then rotate around center, then translate if needed
        trans_data = (Affine2D()
                    .translate(-img_w/2+dx, -img_h/2+dy)  # move center to origin
                    .scale(scale, scale))
        trans_data += ax.transData
        ax.imshow(bg_img, origin='upper', transform=trans_data, alpha=0.8, zorder=0)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Success polygon plot saved to: {save_path}")
        # Also save as PDF
        pdf_path = save_path.replace('.png', '.pdf')
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"Success polygon plot saved to: {pdf_path}")
    
    if show_plot:
        plt.show()
    
    # Print statistics about regions found
    print(f"\n  Found {len(original_regions)} success region(s) for original data")
    print(f"  Found {len(gripper_regions)} success region(s) for gripper data")
    
    return fig, ax


def print_combined_statistics(df_original, df_gripper):
    """
    Print summary statistics comparing both datasets.
    
    Args:
        df_original (pandas.DataFrame): Original noise experiment data
        df_gripper (pandas.DataFrame): Gripper noise experiment data
    """
    print("\n" + "=" * 70)
    print(" " * 20 + "COMBINED STATISTICS")
    print("=" * 70)
    
    # Original data statistics
    if not df_original.empty:
        print("\n--- ORIGINAL NOISE EXPERIMENTS ---")
        total = len(df_original)
        success = df_original['success'].sum()
        print(f"Total Experiments: {total}")
        print(f"Successful: {success}")
        print(f"Failed: {total - success}")
        print(f"Success Rate: {success/total:.2%}")
        print(f"Noise Magnitudes: {sorted(df_original['noise_magnitude'].unique())}")
    else:
        print("\n--- ORIGINAL NOISE EXPERIMENTS ---")
        print("No data available")
    
    # Gripper data statistics
    if not df_gripper.empty:
        print("\n--- GRIPPER NOISE EXPERIMENTS ---")
        total = len(df_gripper)
        success = df_gripper['success'].sum()
        print(f"Total Experiments: {total}")
        print(f"Successful: {success}")
        print(f"Failed: {total - success}")
        print(f"Success Rate: {success/total:.2%}")
        print(f"Noise Magnitude Range: [{df_gripper['noise_magnitude'].min():.6f}, "
              f"{df_gripper['noise_magnitude'].max():.6f}]")
    else:
        print("\n--- GRIPPER NOISE EXPERIMENTS ---")
        print("No data available")
    
    # Combined statistics
    df_combined = pd.concat([df_original, df_gripper], ignore_index=True)
    if not df_combined.empty:
        print("\n--- COMBINED (ALL EXPERIMENTS) ---")
        total = len(df_combined)
        success = df_combined['success'].sum()
        print(f"Total Experiments: {total}")
        print(f"Successful: {success}")
        print(f"Failed: {total - success}")
        print(f"Overall Success Rate: {success/total:.2%}")
        
        print(f"\nCombined X Noise Range: [{df_combined['x_noise'].min():.6f}, "
              f"{df_combined['x_noise'].max():.6f}]")
        print(f"Combined Y Noise Range: [{df_combined['y_noise'].min():.6f}, "
              f"{df_combined['y_noise'].max():.6f}]")
    
    print("\n" + "=" * 70 + "\n")


def main():
    """
    Main function to run the unified visualization script.
    """
    # File paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Original noise data paths
    original_data_path = os.path.join(script_dir, "data/noise/results_tables/")
    
    # Gripper noise data paths
    gripper_log_file = os.path.join(script_dir, "data/noise_gripper/experiment_master_log.txt")
    
    # Output directory
    output_dir = os.path.join(script_dir, "data/unified_noise_plots/")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Read both datasets
        print("Reading original noise experiment data...")
        df_original = read_original_noise_data(original_data_path)
        if not df_original.empty:
            print(f"  Loaded {len(df_original)} original experiments")
        else:
            print("  No original data found")
        
        print("\nReading gripper noise experiment data...")
        df_gripper = read_gripper_noise_data(gripper_log_file)
        if not df_gripper.empty:
            print(f"  Loaded {len(df_gripper)} gripper experiments")
        else:
            print("  No gripper data found")
        
        if df_original.empty and df_gripper.empty:
            print("\nError: No data found from either source!")
            return
        
        # Print statistics
        print_combined_statistics(df_original, df_gripper)
        
        # Create plots
        print("Creating visualizations...")
        
        # # 1. Combined scatter plot
        # print("  - Combined scatter plot...")
        # scatter_path = os.path.join(output_dir, "combined_scatter_plot.png")
        # plot_combined_scatter(df_original, df_gripper, save_path=scatter_path, show_plot=False)
        
        # # 2. Combined polar sector plot
        # print("  - Combined polar sector plot...")
        # polar_path = os.path.join(output_dir, "combined_polar_plot.png")
        # plot_combined_polar_sectors(df_original, df_gripper, save_path=polar_path, show_plot=False)
        
        # 3. Success polygon plot
        print("  - Success polygon plot...")
        polygon_path = os.path.join(output_dir, "success_polygon_plot.png")
        bg_img_path = "/home/mariano/Documents/hybrid_ws/src/dynamixel_control/dynamixel_control/merged_shifted_blended.png"
        plot_success_polygons(df_original, df_gripper, save_path=polygon_path, show_plot=False, bg_img_path=bg_img_path)
        
        # # 4. Side-by-side comparison
        # print("  - Side-by-side comparison...")
        # comparison_path = os.path.join(output_dir, "side_by_side_comparison.png")
        # plot_side_by_side_comparison(df_original, df_gripper, save_path=comparison_path, show_plot=False)
        
        print(f"\n✓ All plots saved to: {output_dir}")
        # print("\nGenerated files:")
        # print(f"  - {os.path.basename(scatter_path)}")
        # print(f"  - {os.path.basename(polar_path)}")
        # print(f"  - {os.path.basename(polygon_path)}")
        # print(f"  - {os.path.basename(comparison_path)}")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
