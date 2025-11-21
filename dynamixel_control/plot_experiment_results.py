#!/usr/bin/env python3
"""
Script to visualize experiment results from the master log file.
Creates scatter plots showing XNoise and YNoise positions with success/failure colors.

Takes inspiration from noise_plots.py but adapted for the new experiment metadata format.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid display issues
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib.patches import Circle, Wedge

# Set up matplotlib for better plots (disable LaTeX to avoid compatibility issues)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "font.size": 12
})
print("Using default matplotlib formatting")

def read_experiment_data(log_file_path="data/noise_gripper/experiment_master_log.txt"):
    """
    Read experiment data from the master log file.
    
    Args:
        log_file_path (str): Path to the master log file
        
    Returns:
        pandas.DataFrame: DataFrame with experiment data
    """
    if not os.path.exists(log_file_path):
        raise FileNotFoundError(f"Log file not found: {log_file_path}")
    
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

def plot_noise_scatter(df, save_path=None, show_plot=True):
    """
    Create a scatter plot of XNoise vs YNoise with success/failure colors.
    
    Args:
        df (pandas.DataFrame): DataFrame with experiment data
        save_path (str, optional): Path to save the plot
        show_plot (bool): Whether to display the plot
    """
    if df.empty:
        print("No data to plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Separate successful and failed experiments
    success_df = df[df['success'] == True]
    failure_df = df[df['success'] == False]
    
    # Plot successful experiments in green
    if not success_df.empty:
        ax.scatter(success_df['x_noise'], success_df['y_noise'], 
                  color='green', s=60, marker='o', alpha=0.7, 
                  label=f'Success ({len(success_df)})', edgecolors='darkgreen', linewidth=1)
    
    # Plot failed experiments in red
    if not failure_df.empty:
        ax.scatter(failure_df['x_noise'], failure_df['y_noise'], 
                  color='red', s=60, marker='x', alpha=0.7, 
                  label=f'Failure ({len(failure_df)})', linewidth=2)
    
    # Add concentric circles to show noise magnitude levels
    noise_magnitudes = df['noise_magnitude'].unique()
    if len(noise_magnitudes) > 1:
        # If multiple noise magnitudes, show circles for each
        for magnitude in sorted(noise_magnitudes):
            circle = Circle((0, 0), magnitude, color='black', alpha=0.3, 
                          fill=False, linestyle='--', linewidth=1)
            ax.add_patch(circle)
    else:
        # If single noise magnitude, show a few reference circles
        max_magnitude = df['noise_magnitude'].max()
        for radius in [max_magnitude * 0.5, max_magnitude, max_magnitude * 1.5]:
            circle = Circle((0, 0), radius, color='gray', alpha=0.2, 
                          fill=False, linestyle='--', linewidth=0.8)
            ax.add_patch(circle)
    
    # Add grid lines
    ax.axhline(0, color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.axvline(0, color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # Formatting
    ax.set_xlabel("X Noise [m]", fontsize=13)
    ax.set_ylabel("Y Noise [m]", fontsize=13)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_title("Experiment Results: Noise Positions", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Set axis limits with some padding
    max_range = max(abs(df['x_noise'].max()), abs(df['x_noise'].min()),
                   abs(df['y_noise'].max()), abs(df['y_noise'].min()))
    padding = max_range * 0.1
    ax.set_xlim(-max_range - padding, max_range + padding)
    ax.set_ylim(-max_range - padding, max_range + padding)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    if show_plot:
        plt.show()
    
    return fig, ax

def plot_polar_sectors(df, save_path=None, show_plot=True, n_sectors=12):
    """
    Create a polar sector plot similar to the original noise_plots.py.
    
    Args:
        df (pandas.DataFrame): DataFrame with experiment data
        save_path (str, optional): Path to save the plot
        show_plot (bool): Whether to display the plot
        n_sectors (int): Number of angular sectors
    """
    if df.empty:
        print("No data to plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot individual points first
    success_df = df[df['success'] == True]
    failure_df = df[df['success'] == False]
    
    if not success_df.empty:
        ax.scatter(success_df['x_noise'], success_df['y_noise'], 
                  color='green', s=60, marker='o', alpha=0.8, 
                  label=f'Success ({len(success_df)})', zorder=5)
    
    if not failure_df.empty:
        ax.scatter(failure_df['x_noise'], failure_df['y_noise'], 
                  color='red', s=60, marker='x', alpha=0.8, 
                  label=f'Failure ({len(failure_df)})', linewidth=2, zorder=5)
    
    # Create sectors based on noise magnitude ranges
    noise_magnitudes = sorted(df['noise_magnitude'].unique())
    
    if len(noise_magnitudes) > 3:
        # If many different magnitudes, create ranges
        min_mag = df['noise_magnitude'].min()
        max_mag = df['noise_magnitude'].max()
        mag_ranges = np.linspace(min_mag, max_mag, 4)  # 3 ranges
    else:
        # Use existing magnitudes
        mag_ranges = [0] + sorted(noise_magnitudes)
    
    sector_angle = 360 / n_sectors
    
    # Create sector visualization
    for i in range(len(mag_ranges) - 1):
        inner_radius = mag_ranges[i]
        outer_radius = mag_ranges[i + 1]
        
        for sector in range(n_sectors):
            start_angle = sector * sector_angle
            end_angle = (sector + 1) * sector_angle
            
            # Find points in this sector and magnitude range
            sector_data = df[
                (df['noise_magnitude'] >= inner_radius) & 
                (df['noise_magnitude'] < outer_radius) &
                (df['random_direction_deg'] >= start_angle) & 
                (df['random_direction_deg'] < end_angle)
            ]
            
            if not sector_data.empty:
                success_rate = sector_data['success'].mean()
                
                # Color based on success rate
                if success_rate >= 0.75:
                    color = 'green'
                elif success_rate >= 0.25:
                    color = 'orange'
                else:
                    color = 'red'
                
                # Create wedge
                wedge = Wedge(
                    center=(0, 0),
                    r=outer_radius,
                    theta1=start_angle,
                    theta2=end_angle,
                    width=outer_radius - inner_radius,
                    facecolor=color,
                    edgecolor='black',
                    alpha=0.3,
                    linewidth=0.5,
                    zorder=1
                )
                ax.add_patch(wedge)
    
    # Add concentric circles
    for radius in mag_ranges[1:]:
        circle = Circle((0, 0), radius, color='black', alpha=0.4, 
                      fill=False, linestyle='--', linewidth=1, zorder=2)
        ax.add_patch(circle)
    
    # Add grid lines
    ax.axhline(0, color='grey', linestyle='-', linewidth=0.5, alpha=0.5, zorder=3)
    ax.axvline(0, color='grey', linestyle='-', linewidth=0.5, alpha=0.5, zorder=3)
    
    # Formatting
    ax.set_xlabel("X Noise [m]", fontsize=13)
    ax.set_ylabel("Y Noise [m]", fontsize=13)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_title("Experiment Results: Polar Sector Analysis", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, zorder=0)
    ax.set_aspect('equal')
    
    # Set axis limits
    max_radius = max(mag_ranges) * 1.1
    ax.set_xlim(-max_radius, max_radius)
    ax.set_ylim(-max_radius, max_radius)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Polar plot saved to: {save_path}")
    
    if show_plot:
        plt.show()
    
    return fig, ax

def print_statistics(df):
    """
    Print summary statistics of the experiments.
    
    Args:
        df (pandas.DataFrame): DataFrame with experiment data
    """
    if df.empty:
        print("No data available for statistics")
        return
    
    total_experiments = len(df)
    successful_experiments = df['success'].sum()
    failed_experiments = total_experiments - successful_experiments
    success_rate = successful_experiments / total_experiments if total_experiments > 0 else 0
    
    print(f"\n=== Experiment Statistics ===")
    print(f"Total Experiments: {total_experiments}")
    print(f"Successful: {successful_experiments}")
    print(f"Failed: {failed_experiments}")
    print(f"Success Rate: {success_rate:.2%}")
    
    print(f"\nNoise Magnitude Statistics:")
    print(f"  Min: {df['noise_magnitude'].min():.6f}")
    print(f"  Max: {df['noise_magnitude'].max():.6f}")
    print(f"  Mean: {df['noise_magnitude'].mean():.6f}")
    print(f"  Std: {df['noise_magnitude'].std():.6f}")
    
    print(f"\nX Noise Statistics:")
    print(f"  Min: {df['x_noise'].min():.6f}")
    print(f"  Max: {df['x_noise'].max():.6f}")
    print(f"  Mean: {df['x_noise'].mean():.6f}")
    print(f"  Std: {df['x_noise'].std():.6f}")
    
    print(f"\nY Noise Statistics:")
    print(f"  Min: {df['y_noise'].min():.6f}")
    print(f"  Max: {df['y_noise'].max():.6f}")
    print(f"  Mean: {df['y_noise'].mean():.6f}")
    print(f"  Std: {df['y_noise'].std():.6f}")
    
    # Success rate by noise magnitude ranges
    if len(df['noise_magnitude'].unique()) > 1:
        print(f"\nSuccess Rate by Noise Magnitude:")
        for magnitude in sorted(df['noise_magnitude'].unique()):
            subset = df[df['noise_magnitude'] == magnitude]
            if not subset.empty:
                rate = subset['success'].mean()
                count = len(subset)
                print(f"  {magnitude:.6f}: {rate:.2%} ({subset['success'].sum()}/{count})")
    
    print(f"===============================\n")

def main():
    """
    Main function to run the visualization script.
    """
    # File paths - use absolute paths to avoid issues
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(script_dir, "data/noise_gripper/experiment_master_log.txt")
    output_dir = os.path.join(script_dir, "data/noise_gripper/plots/")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Check if log file exists
        if not os.path.exists(log_file):
            print(f"Log file not found at: {log_file}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Script directory: {script_dir}")
            # Try alternative paths
            alt_paths = [
                "data/noise_gripper/experiment_master_log.txt",
                "/home/mariano/Documents/hybrid_ws/src/dynamixel_control/dynamixel_control/data/noise_gripper/experiment_master_log.txt"
            ]
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    print(f"Found log file at alternative path: {alt_path}")
                    log_file = alt_path
                    break
            else:
                print("Could not find log file at any expected location")
                return
        
        # Read data
        print(f"Reading experiment data from: {log_file}")
        df = read_experiment_data(log_file)
        
        if df.empty:
            print("No data found in the log file.")
            return
        
        print(f"Loaded {len(df)} experiments")
        
        # Print statistics
        print_statistics(df)
        
        # Create scatter plot
        print("Creating scatter plot...")
        scatter_save_path = os.path.join(output_dir, "noise_scatter_plot.png")
        plot_noise_scatter(df, save_path=scatter_save_path, show_plot=False)
        
        # Create polar sector plot
        print("Creating polar sector plot...")
        polar_save_path = os.path.join(output_dir, "noise_polar_plot.png")
        plot_polar_sectors(df, save_path=polar_save_path, show_plot=False)
        
        print(f"\nPlots saved to: {output_dir}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Make sure the log file exists at: {log_file}")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
