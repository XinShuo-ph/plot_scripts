#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import argparse
from matplotlib.colors import LogNorm, SymLogNorm
from matplotlib.ticker import LogLocator, MultipleLocator, FormatStrFormatter

# Set matplotlib rcParams for consistent fonts like in the notebook
plt.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": 'cm',  # Computer Modern font - looks like LaTeX
    "font.family": 'STIXGeneral',
    "font.size": 18,          # Further increased base font size
    "axes.labelsize": 20,     # Further increased label font size
    "axes.titlesize": 22,     # Further increased title font size
    "xtick.labelsize": 18,    # Further increased tick label font size
    "ytick.labelsize": 18,    # Further increased tick label font size
    "legend.fontsize": 18,    # Further increased legend font size
    "figure.titlesize": 24    # Further increased figure title font size
})

def load_data_for_all_radii(simname, excise_factor=1.5, suffix="parallel", start_radius=None, end_radius=None):
    """Load all surface integral data files for different radii."""
    # Find all surface integral files for this simulation
    pattern = f"{simname}_2d_integrals_surface_outR*_excise{excise_factor}_{suffix}.npy"
    files = glob.glob(pattern)
    
    if not files:
        print(f"No files found matching pattern: {pattern}")
        return None, None, None
    
    # Dictionary to store data by radius
    data_by_radius = {}
    
    for file in files:
        # Extract radius from filename
        parts = file.split("_")
        outR_part = [p for p in parts if p.startswith("outR")][0]
        radius = float(outR_part[4:])
        
        # Filter by radius range if specified
        if (start_radius is not None and radius < start_radius) or \
           (end_radius is not None and radius > end_radius):
            continue
        
        # Load data
        data = np.load(file)
        data_by_radius[radius] = data
    
    # Get list of all times across all files
    all_times = set()
    for data in data_by_radius.values():
        all_times.update(data[:, 0])
    all_times = sorted(all_times)
    
    # Get list of all radii
    all_radii = sorted(data_by_radius.keys())
    
    print(f"Loaded data for {len(all_radii)} radii and {len(all_times)} time points")
    
    return data_by_radius, all_radii, all_times

def generate_log_spaced_radii(start_radius, end_radius, num_points):
    """Generate log-spaced radius values"""
    return np.logspace(np.log10(start_radius), np.log10(end_radius), num_points)

def filter_to_closest_radii(all_radii, target_radii):
    """Filter the available radii to the closest ones to the target values"""
    closest_radii = []
    
    for target in target_radii:
        closest = min(all_radii, key=lambda x: abs(x - target))
        closest_radii.append(closest)
    
    return sorted(list(set(closest_radii)))  # Remove duplicates and sort

def create_heatmap_data(data_by_radius, selected_radii, all_times, quantity_index):
    """Create a 2D array for heatmap plotting with selected radii."""
    # Initialize 2D array for heatmap
    heatmap = np.zeros((len(selected_radii), len(all_times)))
    heatmap.fill(np.nan)  # Fill with NaN for missing data
    
    # Create mapping from time to column index
    time_to_idx = {t: i for i, t in enumerate(all_times)}
    
    # Fill heatmap with data
    for r_idx, radius in enumerate(selected_radii):
        data = data_by_radius[radius]
        for row in data:
            time = row[0]
            value = row[quantity_index]
            t_idx = time_to_idx[time]
            heatmap[r_idx, t_idx] = value
    
    return heatmap

def main():
    # Constants for orbital parameters
    binary_mass = +2.71811e+00
    binary_omega = -0.002657634562418009 * binary_mass
    T_orbit = np.abs(2*np.pi/binary_omega)
    R_orbit = 70.49764373525885/binary_mass /2
    
    parser = argparse.ArgumentParser(description="Create heatmaps from radial scan results")
    parser.add_argument("--simname", type=str, default="250514_BBH_r70", help="Simulation name")
    parser.add_argument("--excise_factor", type=float, default=1.5, help="Excision factor")
    parser.add_argument("--suffix", type=str, default="parallel", help="Output file suffix")
    parser.add_argument("--output_dir", type=str, default="heatmaps", help="Output directory for heatmaps")
    parser.add_argument("--startR", type=float, default=20.0, help="Start radius")
    parser.add_argument("--endR", type=float, default=320.0, help="End radius")
    parser.add_argument("--numR", type=int, default=80, help="Number of radius points")
    parser.add_argument("--add_horizontal_lines", type=str, default="", help="Comma-separated list of radii to mark with horizontal lines")
    parser.add_argument("--cmap", type=str, default="viridis", help="Colormap to use (e.g., viridis, plasma, RdBu)")
    parser.add_argument("--symmetric_cmap", action="store_true", help="Use symmetric colormap centered at zero")
    parser.add_argument("--time_range", type=str, default="", help="Time range to plot (min,max) - adjusts plot view, not data filtering")
    parser.add_argument("--radius_range", type=str, default="", help="Radius range to plot (min,max) - adjusts plot view, not data filtering")
    parser.add_argument("--no_title", action="store_true", help="Remove titles from plots")
    parser.add_argument("--vmax", type=str, default="", help="Custom vmax values for each plot as comma-separated list: 'surface_x,surface_y,surface_torque,flux_magnitude'")
    parser.add_argument("--r_ticks", type=int, default=10, help="Number of ticks on radius axis")
    parser.add_argument("--use_orbital_units", action="store_true", help="Use orbital units for time and radius")
    parser.add_argument("--binary_mass", type=float, default=binary_mass, help="Binary mass")
    parser.add_argument("--binary_omega", type=float, default=binary_omega, help="Binary angular velocity")
    parser.add_argument("--r_orbit", type=float, default=R_orbit, help="Orbital radius")
    args = parser.parse_args()
    
    # Update orbital parameters if provided
    binary_mass = args.binary_mass
    binary_omega = args.binary_omega
    R_orbit = args.r_orbit
    T_orbit = np.abs(2*np.pi/binary_omega)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data for all radii within the specified range
    data_by_radius, all_radii, all_times = load_data_for_all_radii(
        args.simname, args.excise_factor, args.suffix, args.startR, args.endR
    )
    
    if data_by_radius is None:
        return
    
    # Generate logarithmically spaced target radii
    target_radii = generate_log_spaced_radii(args.startR, args.endR, args.numR)
    
    # Filter available radii to closest matches to the target radii
    selected_radii = filter_to_closest_radii(all_radii, target_radii)
    print(f"Selected {len(selected_radii)} radii for plotting")
    
    # Process horizontal lines if specified
    horizontal_lines = []
    if args.add_horizontal_lines:
        try:
            horizontal_lines = [float(r) for r in args.add_horizontal_lines.split(",")]
        except ValueError:
            print("Warning: Could not parse horizontal lines argument. Format should be comma-separated numbers.")
    
    # Process time range if specified
    time_min, time_max = None, None
    if args.time_range:
        try:
            time_parts = args.time_range.split(",")
            if len(time_parts) == 2:
                time_min = float(time_parts[0]) if time_parts[0].strip() else None
                time_max = float(time_parts[1]) if time_parts[1].strip() else None
        except ValueError:
            print("Warning: Could not parse time range. Format should be min,max")
    
    # Process radius range if specified
    radius_min, radius_max = None, None
    if args.radius_range:
        try:
            radius_parts = args.radius_range.split(",")
            if len(radius_parts) == 2:
                radius_min = float(radius_parts[0]) if radius_parts[0].strip() else None
                radius_max = float(radius_parts[1]) if radius_parts[1].strip() else None
        except ValueError:
            print("Warning: Could not parse radius range. Format should be min,max")
    
    # Parse custom vmax values if provided
    custom_vmax = {}
    if args.vmax:
        try:
            vmax_values = args.vmax.split(",")
            keys = ["surface_x", "surface_y", "surface_torque", "flux_magnitude"]
            for i, key in enumerate(keys):
                if i < len(vmax_values) and vmax_values[i].strip():
                    custom_vmax[key] = float(vmax_values[i])
        except ValueError:
            print("Warning: Could not parse vmax values. Format should be comma-separated numbers.")
    
    # Convert to orbital units if requested
    plot_times = all_times
    plot_radii = selected_radii
    
    if args.use_orbital_units:
        plot_times = [t / T_orbit for t in all_times]
        plot_radii = [r / R_orbit for r in selected_radii]
        horizontal_lines = [r / R_orbit for r in horizontal_lines]
    
    # Based on test_integrate_2d_surface_parallel.py, these are the indices we care about:
    # results.append([ds.current_time,
    #             sur_x, sur_y, sur_z,          # outer surface (indices 1,2,3)
    #             sur_bh1_x, sur_bh1_y, sur_bh1_z, # BH 1 (indices 4,5,6)
    #             sur_bh2_x, sur_bh2_y, sur_bh2_z, # BH 2 (indices 7,8,9)
    #             rhoavg, sur_torque, sur_bh1_torque, sur_bh2_torque]) # (indices 10,11,12,13)

    # Define quantities to plot with their indices and pretty names
    quantities = {
        "surface_x": {
            "index": 1,
            "name": r"$\dot{P}_x$", 
            "description": "SURFACE_X flux through outer boundary"
        },
        "surface_y": {
            "index": 2, 
            "name": r"$\dot{P}_y$", 
            "description": "SURFACE_Y flux through outer boundary"
        },
        "surface_torque": {
            "index": 11, 
            "name": r"$\dot{L}_z$", 
            "description": "Torque on outer boundary (from -y*Fx + x*Fy)"
        }
    }
    
    # Create heatmaps for each quantity
    for key, info in quantities.items():
        idx = info["index"]
        name = info["name"]
        
        # Create heatmap data
        heatmap = create_heatmap_data(data_by_radius, selected_radii, all_times, idx)
        
        # Create figure with appropriate colormap
        plt.figure(figsize=(12, 8))
        
        # Determine colormap and scale
        if args.symmetric_cmap:
            # For symmetric data like torque that can be positive or negative
            if key in custom_vmax:
                vmax = custom_vmax[key]
            else:
                vmax = max(abs(np.nanmin(heatmap)), abs(np.nanmax(heatmap)))
            vmin = -vmax
            cmap = plt.get_cmap(args.cmap if args.cmap != "viridis" else "RdBu_r")
        else:
            # For data that's primarily positive
            vmin = np.nanmin(heatmap)
            if key in custom_vmax:
                vmax = custom_vmax[key]
            else:
                vmax = np.nanmax(heatmap)
            cmap = plt.get_cmap(args.cmap)
        
        # Plot heatmap
        im = plt.pcolormesh(plot_times, plot_radii, heatmap, 
                            shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        
        # Add colorbar with horizontal label
        cbar = plt.colorbar(im)
        cbar.set_label(name, rotation=0, labelpad=15)
        
        # Set axis labels based on units
        if args.use_orbital_units:
            plt.xlabel(r'$t/T_{\rm orbit}$')
            plt.ylabel(r'$r/R_{\rm orbit}$')
        else:
            plt.xlabel(r'$t/M$')
            plt.ylabel(r'$r/M$')
            
        plt.yscale('log')  # Use logarithmic scale for radius
        
        # Get current axis
        ax = plt.gca()
        
        # Set plot limits based on time_range and radius_range if specified
        if time_min is not None or time_max is not None:
            if args.use_orbital_units:
                if time_min is not None:
                    time_min_plot = time_min / T_orbit
                if time_max is not None:
                    time_max_plot = time_max / T_orbit
            else:
                time_min_plot = time_min
                time_max_plot = time_max
            
            if time_min is not None:
                plt.xlim(left=time_min_plot)
            if time_max is not None:
                plt.xlim(right=time_max_plot)
        
        if radius_min is not None or radius_max is not None:
            if args.use_orbital_units:
                if radius_min is not None:
                    radius_min_plot = radius_min / R_orbit
                if radius_max is not None:
                    radius_max_plot = radius_max / R_orbit
            else:
                radius_min_plot = radius_min
                radius_max_plot = radius_max
            
            if radius_min is not None:
                plt.ylim(bottom=radius_min_plot)
            if radius_max is not None:
                plt.ylim(top=radius_max_plot)
        
        # Set more radius ticks for log scale - use both major and minor ticks
        ax.yaxis.set_major_locator(LogLocator(numticks=args.r_ticks))
        ax.yaxis.set_minor_locator(LogLocator(numticks=args.r_ticks*2, subs=np.arange(2, 10)))
        ax.yaxis.set_minor_formatter(FormatStrFormatter("%.0f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
        ax.yaxis.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.2)
        
        # Add title only if not disabled
        if not args.no_title:
            plt.title(f'{name} - {args.simname}')
        
        # Add horizontal lines if specified
        for r in horizontal_lines:
            plt.axhline(y=r, color='white', linestyle='-', alpha=0.7)
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f"{args.output_dir}/{args.simname}_{key}_heatmap.png", dpi=300)
        plt.savefig(f"{args.output_dir}/{args.simname}_{key}_heatmap.pdf", bbox_inches='tight')
        plt.close()
    
    # Create a combined visualization for SURFACE_X and SURFACE_Y fluxes
    plt.figure(figsize=(12, 8))
    surface_x_heatmap = create_heatmap_data(data_by_radius, selected_radii, all_times, 1)
    surface_y_heatmap = create_heatmap_data(data_by_radius, selected_radii, all_times, 2)
    
    # Calculate magnitude of the flux vector
    flux_magnitude = np.sqrt(surface_x_heatmap**2 + surface_y_heatmap**2)
    
    # Determine vmax for flux magnitude
    if "flux_magnitude" in custom_vmax:
        vmax = custom_vmax["flux_magnitude"]
    else:
        vmax = np.nanmax(flux_magnitude)
    
    # Plot the magnitude
    im = plt.pcolormesh(plot_times, plot_radii, flux_magnitude, 
                         shading='auto', cmap=args.cmap, vmin=0, vmax=vmax)
    
    # Add colorbar with horizontal label
    cbar = plt.colorbar(im)
    cbar.set_label(r'$|\dot{\vec{P}}|$', rotation=0, labelpad=15)
    
    # Set axis labels based on units
    if args.use_orbital_units:
        plt.xlabel(r'$t/T_{\rm orbit}$')
        plt.ylabel(r'$r/R_{\rm orbit}$')
    else:
        plt.xlabel(r'$t/M$')
        plt.ylabel(r'$r/M$')
    
    plt.yscale('log')
    
    # Get current axis
    ax = plt.gca()
    
    # Set plot limits based on time_range and radius_range if specified
    if time_min is not None or time_max is not None:
        if args.use_orbital_units:
            if time_min is not None:
                time_min_plot = time_min / T_orbit
            if time_max is not None:
                time_max_plot = time_max / T_orbit
        else:
            time_min_plot = time_min
            time_max_plot = time_max
        
        if time_min is not None:
            plt.xlim(left=time_min_plot)
        if time_max is not None:
            plt.xlim(right=time_max_plot)
    
    if radius_min is not None or radius_max is not None:
        if args.use_orbital_units:
            if radius_min is not None:
                radius_min_plot = radius_min / R_orbit
            if radius_max is not None:
                radius_max_plot = radius_max / R_orbit
        else:
            radius_min_plot = radius_min
            radius_max_plot = radius_max
        
        if radius_min is not None:
            plt.ylim(bottom=radius_min_plot)
        if radius_max is not None:
            plt.ylim(top=radius_max_plot)
    
    # Set more radius ticks for log scale - use both major and minor ticks
    ax.yaxis.set_major_locator(LogLocator(numticks=args.r_ticks))
    ax.yaxis.set_minor_locator(LogLocator(numticks=args.r_ticks*2, subs=np.arange(2, 10)))
    ax.yaxis.set_minor_formatter(FormatStrFormatter("%.0f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    ax.yaxis.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.2)
    
    # Add title only if not disabled
    if not args.no_title:
        plt.title(f'Surface Flux Magnitude - {args.simname}')
    
    # Add horizontal lines if specified
    for r in horizontal_lines:
        plt.axhline(y=r, color='white', linestyle='-', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/{args.simname}_surface_flux_magnitude_heatmap.png", dpi=300)
    plt.savefig(f"{args.output_dir}/{args.simname}_surface_flux_magnitude_heatmap.pdf", bbox_inches='tight')
    plt.close()
    
    print(f"Heatmaps saved to {args.output_dir}/")

if __name__ == "__main__":
    main() 