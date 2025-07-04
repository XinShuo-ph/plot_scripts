#!/usr/bin/env python
import yt
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from matplotlib.colors import LogNorm, Normalize
import logging
import matplotlib.colors as colors
import copy
from matplotlib import cm

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Compare RHO_ENERGY across different simulations')
    parser.add_argument('--time_snapshot', type=float, default=8000, help='Time snapshot to compare (in M)')
    parser.add_argument('--extent_x', type=float, default=140, help='Extent of the plot in x direction')
    parser.add_argument('--extent_y', type=float, default=140, help='Extent of the plot in y direction')
    parser.add_argument('--plot_cmap_max', type=float, default=0.01, help='Maximum value for colormap')
    parser.add_argument('--plot_cmap_min', type=float, default=0.00001, help='Minimum value for colormap')
    parser.add_argument('--only_velocity', action='store_true', help='Generate only velocity comparison plot')
    parser.add_argument('--only_resolution', action='store_true', help='Generate only resolution comparison plot')
    parser.add_argument('--verbose', action='store_true', help='Show verbose output')
    return parser.parse_args()

def setup_logger(verbose=False):
    """Configure logging based on verbosity."""
    # Set YT logger to only show warnings
    yt_logger = logging.getLogger('yt')
    yt_logger.setLevel(logging.WARNING)
    
    # Set root logger level
    log_level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    return logging.getLogger(__name__)

def list_datasets(simname, basedir):
    """Find all datasets in the simulation directory."""
    rundir = os.path.join(basedir, simname)
    plt_dirs = sorted([d for d in os.listdir(rundir) if d.startswith('plt') and d[3:].isdigit()], 
                     key=lambda x: int(x[3:]))
    return plt_dirs

def find_closest_dataset_time(simname, target_time, basedir):
    """Find the dataset with time closest to target_time using an efficient approach."""
    rundir = os.path.join(basedir, simname)
    plt_dirs = list_datasets(simname, basedir)
    
    closest_path = None
    closest_time = None
    min_time_diff = float('inf')
    times = []
    
    logger.info(f"Found {len(plt_dirs)} datasets for {simname}, using first 2 to determine time step...")
    
    # Load only the first 2 datasets to determine the time step
    if len(plt_dirs) >= 2:
        # Get times from first two datasets
        ds1 = yt.load(os.path.join(rundir, plt_dirs[0]))
        ds2 = yt.load(os.path.join(rundir, plt_dirs[1]))
        
        time1 = float(ds1.current_time)
        time2 = float(ds2.current_time)
        
        # Calculate time step
        dt = time2 - time1
        
        # Calculate the expected index for target time
        target_idx = round((target_time - time1) / dt)
        target_idx = max(0, min(target_idx, len(plt_dirs) - 1))  # Ensure within bounds
        
        # Get the closest dataset
        plt_dir = plt_dirs[target_idx]
        ds_path = os.path.join(rundir, plt_dir)
        ds = yt.load(ds_path)
        
        closest_time = float(ds.current_time)
        closest_path = ds_path
        min_time_diff = abs(closest_time - target_time)
        times.append((plt_dir, closest_time))
        
        logger.info(f"Using time step dt = {dt:.2f}M, estimated index: {target_idx}")
    else:
        logger.warning("Not enough datasets to determine time step")
    
    if closest_path:
        logger.info(f"Found dataset at time {closest_time:.2f}M (diff: {min_time_diff:.2f}M)")
    
    return closest_path, closest_time, times

def load_datasets_by_time(simnames, target_time, extent_x, extent_y, basedir, plotdir):
    """Load datasets for multiple simulations at the specified time."""
    datasets = {}
    
    for param_value, simname in simnames:
        logger.info(f"\nProcessing simulation: {simname}")
        if isinstance(param_value, list):
            logger.info(f"Parameters: v={param_value[0]}, μ={param_value[1]}, d_BBH={param_value[2]}")
            # Convert list to tuple for using as dictionary key
            param_key = tuple(param_value)
        else:
            logger.info(f"Parameter: {param_value}")
            param_key = param_value
            
        # Find the closest dataset
        closest_path, closest_time, _ = find_closest_dataset_time(simname, target_time, basedir)
        
        if closest_path and closest_time:
            # Load the dataset
            ds = yt.load(closest_path)
            
            # Check if RHO_ENERGY is available
            if ('boxlib', 'RHO_ENERGY') in ds.field_list:
                # Create a slice plot to get the data
                slc = yt.SlicePlot(ds, 'z', 'RHO_ENERGY')
                slc.set_width((extent_x, extent_y))
                
                # Get the raw data
                frb = slc.frb
                data = frb['RHO_ENERGY'].d
                
                # Get spatial coordinates
                x_left = -extent_x/2
                x_right = extent_x/2
                y_bottom = -extent_y/2
                y_top = extent_y/2
                
                x = np.linspace(x_left, x_right, data.shape[0])
                y = np.linspace(y_bottom, y_top, data.shape[1])
                
                # Store the data - use the param_key instead of param_value
                datasets[param_key] = {
                    'data': data,
                    'x': x,
                    'y': y,
                    'time': closest_time,
                    'path': closest_path,
                    'original_param': param_value  # Store the original parameter for reference
                }
                
                logger.info(f"  ✓ Loaded dataset with time {closest_time:.2f}M")
                logger.info(f"  ✓ Data shape: {data.shape}, range: {data.min():.8f} to {data.max():.8f}")
            else:
                logger.error(f"  ✗ RHO_ENERGY field not available in {closest_path}")
        else:
            logger.error(f"  ✗ No suitable dataset found for {simname}")
    
    return datasets

# Create a custom colormap with black for masked values
def create_masked_colormap(cmap_name='jet'):
    """Create a colormap that shows masked values as black."""
    cmap = copy.deepcopy(plt.get_cmap(cmap_name))
    cmap.set_bad('black', 1.0)  # Set the color for masked values to black
    return cmap

def create_multipanel_plot(datasets, title, filename, vmin, vmax, plotdir):
    """Create a multi-panel plot comparing datasets."""
    if not datasets:
        logger.error("No datasets to plot!")
        return
        
    # Use the existing order from the input data
    sorted_params = list(datasets.keys())  # Convert to list but don't sort
    n_plots = len(sorted_params)
    
    logger.info(f"Creating {n_plots}-panel plot: {title}")
    
    # Determine grid layout
    cols = min(4, n_plots)
    rows = (n_plots + cols - 1) // cols
    
    # Create figure and axes with extra space for colorbar
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols+1, 4*rows), 
                           sharex=True, sharey=True, constrained_layout=False)
    
    # Convert axes to array for consistent indexing
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    elif rows == 1 or cols == 1:
        axes = axes.reshape(-1)
    
    # Create a custom colormap that handles masked values
    masked_cmap = create_masked_colormap('jet')
    
    # Create the plots
    for i, param in enumerate(sorted_params):
        logger.info(f"  Plotting panel {i+1}/{n_plots}: {param}")
        data_dict = datasets[param]
        data = data_dict['data']
        x = data_dict['x']
        y = data_dict['y']
        time = data_dict['time']
        
        # Mask zero or negative values (these will appear black)
        masked_data = np.ma.masked_less_equal(data, 0)
        
        # Calculate extent for imshow
        extent = [min(x), max(x), min(y), max(y)]
        
        # Get subplot
        ax = axes.flat[i] if len(axes.flat) > 1 else axes
        
        # Create plot with imshow for better performance, using masked data
        im = ax.imshow(masked_data, norm=LogNorm(vmin=vmin, vmax=vmax),
                     cmap=masked_cmap, origin='lower', extent=extent, 
                     interpolation='nearest', aspect='auto')
        
        # Add parameters in text box
        if isinstance(param, int):
            # For resolution plots
            ax.text(0.03, 0.97, f'N = {param}', transform=ax.transAxes,
              fontsize=12, va='top', ha='left', 
              bbox=dict(facecolor='white', alpha=0.7))
        elif isinstance(param, tuple):
            # For velocity, mass, radius plots - unpack the parameters
            v, mu, r_orbit = param
            # Multi-line label with all three parameters
            label = f'$v = {v:.1f}$\n$\\mu = {mu:.2f} \\, M^{{-1}}$\n$d_{{BBH}} = {r_orbit} \\, M$'
            ax.text(0.03, 0.97, label, transform=ax.transAxes,
              fontsize=10, va='top', ha='left', 
              bbox=dict(facecolor='white', alpha=0.7))
        else:
            # For single value parameters
            ax.text(0.03, 0.97, f'v = {param:.2f}', transform=ax.transAxes,
              fontsize=12, va='top', ha='left', 
              bbox=dict(facecolor='white', alpha=0.7))
            
        # Set labels on outer plots - make sure units are included
        if i >= n_plots - cols:  # Bottom row
            ax.set_xlabel('$x/M$', fontsize=14)
        if i % cols == 0:  # First column
            ax.set_ylabel('$y/M$', fontsize=14)
            
        # Set equal aspect ratio
        ax.set_aspect('equal')
    
    # Hide any unused subplots
    for j in range(n_plots, rows*cols):
        if j < len(axes.flat):
            logger.info(f"  Hiding unused panel {j+1}")
            axes.flat[j].axis('off')
    
    logger.info("  Adding colorbar...")
    # Add colorbar with improved styling and positioning
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    # Create a single colorbar for the entire figure
    cax = fig.add_axes([0.92, 0.15, 0.01, 0.7])  # [x, y, width, height]
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('$\\rho$', fontsize=16, rotation=0, labelpad=15)
    
    # Format the ticks for logarithmic scale
    import matplotlib.ticker as ticker
    cbar.ax.tick_params(labelsize=12)
    cbar.ax.minorticks_off()  # Turn off minor ticks for cleaner look
    
    # Adjust layout
    logger.info("  Adjusting layout...")
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])  # Left, bottom, right, top
    
    # Save the figure
    logger.info("  Saving figure...")
    output_path = os.path.join(plotdir, filename)
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    logger.info(f"  ✓ Saved multi-panel plot to {output_path}")
    
    return fig

def main():
    """Main function to run the script."""
    args = parse_arguments()
    
    # Setup logging
    global logger
    logger = setup_logger(args.verbose)
    
    # Directory paths
    basedir = "/pscratch/sd/x/xinshuo/runGReX/"
    plotdir = "/pscratch/sd/x/xinshuo/plotGReX/"
    
    # Configure matplotlib for better plots
    plt.rcParams['font.size'] = 14
    
    # Define the sets of simulations
    velocity_simnames = [
        ([0.7,0.2,26], '20250622_tune_damping_n256_v07'),
        ([0.6,0.2,26], '20250621_tune_damping_n256_v06'),
        # (0.55, '20250623_tune_damping_n256_v055'),
        ([0.5,0.2,26], '20250620_tune_damping_n256_1'),
        # (0.45, '20250623_tune_damping_n256_v045'),
        ([0.4,0.2,26], '20250622_tune_damping_n256_v04')
        # ,        (0.3, '20250622_tune_damping_n256_v03') 
        # ,'20250627_mu02_v04_r35',
        # '20250627_mu02_v05_r35',
        # '20250627_mu02_v06_r35',
        # '20250627_mu02_v07_r35'
        , ([0.7,0.2,13], '20250627_mu02_v07_r35'),
        ([0.6,0.2,13], '20250627_mu02_v06_r35'),
        ([0.5,0.2,13], '20250627_mu02_v05_r35'),
        ([0.4,0.2,13], '20250627_mu02_v04_r35')
        , ([0.7,0.8,26], '20250628_ncell320_v07_mu08'),
        ([0.6,0.8,26], '20250628_ncell320_v06_mu08'),
        ([0.5,0.8,26], '20250627_ncell256_v05_mu08'),
        ([0.4,0.8,26], '20250628_ncell320_v04_mu08')
        ,
        ([0.4,0.05,26], '20250630_ncell416_v04_mu005'),
        ([0.5,0.05,26], '20250630_ncell416_v05_mu005'),
        ([0.6,0.05,26], '20250630_ncell416_v06_mu005'),
        ([0.7,0.05,26], '20250630_ncell416_v07_mu005'),
    ]

    ncell_simnames = [
        (256, '20250620_tune_damping_n256_1'),
        (192, '20250621_tune_damping_n192'),
        (128, '20250621_tune_damping_n128')
    ]
    
    # Generate velocity comparison plot
    if not args.only_resolution:
        logger.info("Loading velocity comparison datasets...")
        velocity_datasets = load_datasets_by_time(
            velocity_simnames, 
            args.time_snapshot, 
            args.extent_x, 
            args.extent_y,
            basedir,
            plotdir
        )
        
        if velocity_datasets:
            create_multipanel_plot(
                velocity_datasets,
                f'Comparison of $\\rho$ across different velocities at t ≈ {args.time_snapshot}M',
                f'rho_energy_velocity_comparison_t{args.time_snapshot:.0f}.pdf',
                args.plot_cmap_min,
                args.plot_cmap_max,
                plotdir
            )
        else:
            logger.error("No velocity datasets were loaded, skipping velocity plot")
    
    # Generate resolution comparison plot
    if not args.only_velocity:
        logger.info("\nLoading resolution comparison datasets...")
        ncell_datasets = load_datasets_by_time(
            ncell_simnames, 
            args.time_snapshot, 
            args.extent_x, 
            args.extent_y,
            basedir,
            plotdir
        )
        
        if ncell_datasets:
            create_multipanel_plot(
                ncell_datasets,
                f'Comparison of $\\rho$ across different resolutions at t ≈ {args.time_snapshot}M',
                f'rho_energy_resolution_comparison_t{args.time_snapshot:.0f}.pdf',
                args.plot_cmap_min,
                args.plot_cmap_max,
                plotdir
            )
        else:
            logger.error("No resolution datasets were loaded, skipping resolution plot")
    
    logger.info("Script completed")

if __name__ == "__main__":
    logger = None  # Will be set in main()
    main() 