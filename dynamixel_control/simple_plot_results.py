#!/usr/bin/env python3
"""
Simple script to visualize experiment results from the master log file.
Creates scatter plots showing XNoise and YNoise positions with success/failure colors.

Simplified version that avoids LaTeX and complex dependencies.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os

def read_experiment_data(log_file_path):
    """
    Read experiment data from the master log file.
    
    Args:
        log_file_path (str): Path to the master log file
        
    Returns:
        dict: Dictionary with experiment data arrays
    """
    if not os.path.exists(log_file_path):
        raise FileNotFoundError(f"Log file not found: {log_file_path}")
    
    data = {
        'experiment_id': [],
        'x_noise': [],
        'y_noise': [],
        'noise_magnitude': [],
        'random_direction_deg': [],
        'success': []
    }
    
    with open(log_file_path, 'r') as f:
        lines = f.readlines()
    
    # Filter out comment lines
    data_lines = [line.strip() for line in lines if not line.startswith('#') and line.strip()]
    
    for line in data_lines:
        parts = [part.strip() for part in line.split(',')]
        if len(parts) >= 8:
            try:
                experiment_id = int(parts[0])
                random_direction_deg = float(parts[3])
                noise_magnitude = float(parts[4])
                x_noise = float(parts[5])
                y_noise = float(parts[6])
                success = parts[7].lower() == 'true'
                
                data['experiment_id'].append(experiment_id)
                data['x_noise'].append(x_noise)
                data['y_noise'].append(y_noise)
                data['noise_magnitude'].append(noise_magnitude)
                data['random_direction_deg'].append(random_direction_deg)
                data['success'].append(success)
                
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse line: {line}")
                continue
    
    # Convert to numpy arrays
    for key in data:
        data[key] = np.array(data[key])
    
    return data

def plot_simple_scatter(data, save_path):
    """
    Create a simple scatter plot of XNoise vs YNoise with success/failure colors.
    
    Args:
        data (dict): Dictionary with experiment data arrays
        save_path (str): Path to save the plot
    """
    if len(data['x_noise']) == 0:
        print("No data to plot")
        return
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Separate successful and failed experiments
    success_mask = data['success']
    failure_mask = ~data['success']
    
    # Plot successful experiments in green
    if np.any(success_mask):
        plt.scatter(data['x_noise'][success_mask], data['y_noise'][success_mask], 
                   color='green', s=60, marker='o', alpha=0.7, 
                   label=f'Success ({np.sum(success_mask)})', edgecolors='darkgreen', linewidth=1)
    
    # Plot failed experiments in red
    if np.any(failure_mask):
        plt.scatter(data['x_noise'][failure_mask], data['y_noise'][failure_mask], 
                   color='red', s=60, marker='x', alpha=0.7, 
                   label=f'Failure ({np.sum(failure_mask)})', linewidth=2)
    
    # Add concentric circles to show noise magnitude levels
    unique_magnitudes = np.unique(data['noise_magnitude'])
    if len(unique_magnitudes) > 1:
        for magnitude in sorted(unique_magnitudes):
            circle = plt.Circle((0, 0), magnitude, color='black', alpha=0.3, 
                              fill=False, linestyle='--', linewidth=1)
            plt.gca().add_patch(circle)
    else:
        # If single noise magnitude, show a few reference circles
        max_magnitude = np.max(data['noise_magnitude'])
        for radius in [max_magnitude * 0.5, max_magnitude, max_magnitude * 1.5]:
            circle = plt.Circle((0, 0), radius, color='gray', alpha=0.2, 
                              fill=False, linestyle='--', linewidth=0.8)
            plt.gca().add_patch(circle)
    
    # Add grid lines
    plt.axhline(0, color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
    plt.axvline(0, color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # Formatting
    plt.xlabel("X Noise [m]", fontsize=13)
    plt.ylabel("Y Noise [m]", fontsize=13)
    plt.title("Experiment Results: Noise Positions", fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Set axis limits with some padding
    max_range = max(abs(np.max(data['x_noise'])), abs(np.min(data['x_noise'])),
                   abs(np.max(data['y_noise'])), abs(np.min(data['y_noise'])))
    padding = max_range * 0.1
    plt.xlim(-max_range - padding, max_range + padding)
    plt.ylim(-max_range - padding, max_range + padding)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close figure to free memory
    print(f"Scatter plot saved to: {save_path}")

def print_statistics(data):
    """
    Print summary statistics of the experiments.
    
    Args:
        data (dict): Dictionary with experiment data arrays
    """
    if len(data['x_noise']) == 0:
        print("No data available for statistics")
        return
    
    total_experiments = len(data['x_noise'])
    successful_experiments = np.sum(data['success'])
    failed_experiments = total_experiments - successful_experiments
    success_rate = successful_experiments / total_experiments if total_experiments > 0 else 0
    
    print(f"\n=== Experiment Statistics ===")
    print(f"Total Experiments: {total_experiments}")
    print(f"Successful: {successful_experiments}")
    print(f"Failed: {failed_experiments}")
    print(f"Success Rate: {success_rate:.2%}")
    
    print(f"\nNoise Magnitude Statistics:")
    print(f"  Min: {np.min(data['noise_magnitude']):.6f}")
    print(f"  Max: {np.max(data['noise_magnitude']):.6f}")
    print(f"  Mean: {np.mean(data['noise_magnitude']):.6f}")
    print(f"  Std: {np.std(data['noise_magnitude']):.6f}")
    
    print(f"\nX Noise Statistics:")
    print(f"  Min: {np.min(data['x_noise']):.6f}")
    print(f"  Max: {np.max(data['x_noise']):.6f}")
    print(f"  Mean: {np.mean(data['x_noise']):.6f}")
    print(f"  Std: {np.std(data['x_noise']):.6f}")
    
    print(f"\nY Noise Statistics:")
    print(f"  Min: {np.min(data['y_noise']):.6f}")
    print(f"  Max: {np.max(data['y_noise']):.6f}")
    print(f"  Mean: {np.mean(data['y_noise']):.6f}")
    print(f"  Std: {np.std(data['y_noise']):.6f}")
    
    print(f"===============================\n")

def main():
    """
    Main function to run the visualization script.
    """
    # Try different possible paths for the log file
    possible_paths = [
        "data/noise_gripper/experiment_master_log.txt",
        "/home/mariano/Documents/hybrid_ws/src/dynamixel_control/dynamixel_control/data/noise_gripper/experiment_master_log.txt",
        os.path.join(os.path.dirname(__file__), "data/noise_gripper/experiment_master_log.txt")
    ]
    
    log_file = None
    for path in possible_paths:
        if os.path.exists(path):
            log_file = path
            break
    
    if log_file is None:
        print("Error: Could not find experiment_master_log.txt in any of these locations:")
        for path in possible_paths:
            print(f"  - {path}")
        print(f"Current working directory: {os.getcwd()}")
        return
    
    # Output directory
    output_dir = os.path.join(os.path.dirname(log_file), "plots")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Read data
        print(f"Reading experiment data from: {log_file}")
        data = read_experiment_data(log_file)
        
        if len(data['x_noise']) == 0:
            print("No data found in the log file.")
            return
        
        print(f"Loaded {len(data['x_noise'])} experiments")
        
        # Print statistics
        print_statistics(data)
        
        # Create scatter plot
        print("Creating scatter plot...")
        scatter_save_path = os.path.join(output_dir, "noise_scatter_plot.png")
        plot_simple_scatter(data, scatter_save_path)
        
        print(f"\nPlot saved to: {output_dir}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
