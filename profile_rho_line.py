import yt
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from datetime import datetime

# Turn off yt INFO logging messages
import logging
yt.funcs.mylog.setLevel(logging.WARNING)


parser = argparse.ArgumentParser(description='Extract RHO_ENERGY values along a line at x=source_x, y=[-source_y,+source_y].')
parser.add_argument('--simname', type=str, required=True, help='Name of the simulation directory')
parser.add_argument('--outplot', action='store_true', help='Output plots to files')
parser.add_argument('--source_x', type=float, default=650, help='x-coordinate of the line (default: 700)')
parser.add_argument('--source_y', type=float, default=600, help='Maximum y-coordinate (line extends from -source_y to +source_y) (default: 600)')
parser.add_argument('--skipevery', type=int, default=1, help='Skip every n iterations')
parser.add_argument('--maxframes', type=int, default=10000, help='Maximum number of frames to process')
parser.add_argument('--tmpfile', type=str, default='/tmp/rho_avg_results.txt', help='Temporary file to write results')
parser.add_argument('--time_threshold', type=float, default=200.0, help='Time threshold for calculating average (default: 200.0)')

args = parser.parse_args()

simname = args.simname
outplot = args.outplot
source_x = args.source_x
source_y = args.source_y
skipevery = args.skipevery
maxframes = args.maxframes
tmpfile = args.tmpfile
time_threshold = args.time_threshold

basedir = "/pscratch/sd/x/xinshuo/runGReX/"
plotdir = "/pscratch/sd/x/xinshuo/plotGReX/"
rundir = basedir + simname + "/"

# Create output directories if they don't exist
if outplot:
    outputdir = os.path.join(plotdir, simname)
    framesdir = os.path.join(outputdir, "frames")
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    if not os.path.exists(framesdir):
        os.makedirs(framesdir)

# Find all plt directories and sort them
plt_dirs = sorted([d for d in os.listdir(rundir) if d.startswith('plt') and d[3:].isdigit()], key=lambda x: int(x[3:]))
plt_dirs = plt_dirs[::skipevery]  # Skip iterations as specified

if len(plt_dirs) > maxframes:
    plt_dirs = plt_dirs[:maxframes]

# Arrays to store results
times = []
line_averages = []

print(f"Frame,Time,AverageRHO_ENERGY")
print("-" * 30)

for frame_idx, plt_dir in enumerate(plt_dirs):
    try:
        # Load dataset
        ds = yt.load(os.path.join(rundir, plt_dir))
        current_time = float(ds.current_time)
        
        # Find the finest level that contains our line of interest
        finest_level = 0
        for level in range(ds.index.max_level, -1, -1):
            level_grids = ds.index.select_grids(level)
            if len(level_grids) == 0:
                continue
                
            level_left_edges = np.array([g.LeftEdge for g in level_grids])
            level_right_edges = np.array([g.RightEdge for g in level_grids])
            
            min_x = level_left_edges[:, 0].min()
            max_x = level_right_edges[:, 0].max()
            min_y = level_left_edges[:, 1].min()
            max_y = level_right_edges[:, 1].max()
            
            if min_x <= source_x <= max_x and min_y <= -source_y and max_y >= source_y:
                finest_level = level
                break
        
        # Create ray along the y-axis at the specified x-coordinate
        start_point = ds.arr([source_x, -source_y, 0], "code_length")
        end_point = ds.arr([source_x, source_y, 0], "code_length")
        ray = ds.ray(start_point, end_point)
        
        # Extract RHO_ENERGY values
        rho_energy = ray['RHO_ENERGY']
        
        # Get the y-coordinates and sort everything by y
        y_positions = ray['y']
        sort_idx = np.argsort(y_positions)
        y_sorted = y_positions[sort_idx]
        rho_sorted = rho_energy[sort_idx]
        
        # Calculate average
        avg_rho = np.mean(rho_sorted)
        
        # Store results
        times.append(current_time)
        line_averages.append(avg_rho)
        
        # Print result
        print(f"{frame_idx},{current_time},{avg_rho}")
        
        # Generate plot of the distribution if requested
        if outplot:
            plt.figure(figsize=(10, 6))
            plt.plot(y_sorted, rho_sorted, 'b-')
            plt.grid(True)
            plt.xlabel('y position')
            plt.ylabel('RHO_ENERGY')
            plt.title(f'RHO_ENERGY at x={source_x}, t={current_time:.2f}')
            plt.savefig(os.path.join(framesdir, f"rho_profile_frame{frame_idx:05d}.png"))
            plt.close()
    
    except Exception as e:
        print(f"Error processing frame {frame_idx} ({plt_dir}): {str(e)}")

# Calculate average of rho_avg after t=200
times = np.array(times)
line_averages = np.array(line_averages)

# Get final time
final_time = times[-1] if len(times) > 0 else 0.0

# Calculate average of rho_avg after time_threshold
mask_after_threshold = times >= time_threshold
if np.any(mask_after_threshold):
    avg_after_threshold = np.mean(line_averages[mask_after_threshold])
    print(f"\nAverage RHO_ENERGY after t={time_threshold}: {avg_after_threshold:.6e}")
else:
    avg_after_threshold = 0.0
    print(f"\nNo data points after t={time_threshold}")

# Write results to temporary file
with open(tmpfile, 'w') as f:
    f.write(f"simname: {simname}\n")
    f.write(f"final_time: {final_time}\n")
    f.write(f"avg_after_{time_threshold}: {avg_after_threshold:.6e}\n")

# Plot average over time if requested
if outplot and len(times) > 0:
    plt.figure(figsize=(10, 6))
    plt.plot(times, line_averages, 'r-')
    
    # Highlight the average after time_threshold if applicable
    if np.any(mask_after_threshold):
        plt.axhline(y=avg_after_threshold, color='blue', linestyle='--', 
                   label=f'Avg after t={time_threshold}: {avg_after_threshold:.6e}')
        plt.axvline(x=time_threshold, color='green', linestyle='--', label=f't={time_threshold}')
        plt.legend()
    
    plt.grid(True)
    plt.xlabel('Time')
    plt.ylabel('Average RHO_ENERGY')
    plt.title(f'Average RHO_ENERGY at x={source_x}, y=±{source_y}')
    timestamp = datetime.now().strftime("%Y%m%d")
    plt.savefig(os.path.join(plotdir, simname, f"rho_average_x{source_x}_y{source_y}_{timestamp}.png"))
    
    # Also save the data as CSV and NumPy format
    data = np.column_stack((times, line_averages))
    np.savetxt(os.path.join(plotdir, simname, f"rho_average_x{source_x}_y{source_y}_{timestamp}.csv"), 
               data, delimiter=',', header='Time,AverageRHO_ENERGY', comments='')
    np.save(os.path.join(plotdir, simname, f"rho_average_x{source_x}_y{source_y}_{timestamp}.npy"), data)
    
    print(f"Results saved to {plotdir}{simname}/") 