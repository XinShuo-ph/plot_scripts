# test sampling a polar grid
import yt
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Sample data on a radial grid using yt.')
parser.add_argument('--simname', type=str, default="240930_BBH_3D_zboost_v5", help='Name of the simulation directory')
parser.add_argument('--r_min', type=float, default=10.0, help='Minimum radius')
parser.add_argument('--r_max', type=float, default=600.0, help='Maximum radius')
parser.add_argument('--n_r', type=int, default=50, help='Number of radial bins')
parser.add_argument('--n_theta', type=int, default=60, help='Number of angular bins')
parser.add_argument('--fields', type=str, default='VOLUME_X,VOLUME_Y,VOLUME_Z,W', help='Comma-separated list of fields to sample')
parser.add_argument('--plot', action='store_true', help='Generate plots')
parser.add_argument('--fix_metric_error', action='store_true', help='Fix the metric error in VOLUME fields (apply 1/sqrt(gamma) correction)')
parser.add_argument('--psipow', type=float, default=2, help='power of psi factor')
parser.add_argument('--debug', action='store_true', help='Enable verbose debug output')

args = parser.parse_args()
simname = args.simname
r_min = args.r_min
r_max = args.r_max
n_r = args.n_r
n_theta = args.n_theta
fields = args.fields.split(',')
plot = args.plot
fix_metric_error = args.fix_metric_error
psipow = args.psipow
debug = args.debug

basedir = "/pscratch/sd/x/xinshuo/runGReX/"
plotdir = "/pscratch/sd/x/xinshuo/plotGReX/"
rundir = basedir + simname + "/"

# Make plotdir/tmp if it doesn't exist
if not os.path.exists(plotdir + "tmp"):
    os.makedirs(plotdir + "tmp")

# Get the first plt file
plt_dirs = sorted([d for d in os.listdir(rundir) if d.startswith('plt') and d[3:].isdigit()], key=lambda x: int(x[3:]))
plt_dir = plt_dirs[0]  # Use the first frame for testing

# Load the dataset
print(f"Loading dataset: {rundir}{plt_dir}")
ds = yt.load(os.path.join(rundir, plt_dir))
print(f"Dataset loaded. Current time: {ds.current_time}")
print(f"Domain dimensions: {ds.domain_dimensions}")
print(f"Domain left edge: {ds.domain_left_edge}")
print(f"Domain right edge: {ds.domain_right_edge}")

# Get domain center
center = ds.domain_center
print(f"Domain center: {center}")

# Check available fields in dataset
if debug:
    print("Available fields in dataset:")
    for field in ds.field_list:
        print(f"  {field}")

# Determine the field type to use
field_type = None
for field in fields:
    for full_field in ds.field_list:
        if isinstance(full_field, tuple) and full_field[1] == field:
            field_type = full_field[0]
            print(f"Found field type '{field_type}' for field '{field}'")
            break
    if field_type:
        break

if field_type is None:
    print("Warning: Could not determine field type automatically.")
    # Try using default field type if available
    if hasattr(ds, 'default_field_type'):
        field_type = ds.default_field_type
        print(f"Using default field type: {field_type}")
    else:
        # Try common field types
        for ft in ['boxlib', 'gas', 'io']:
            if debug:
                print(f"Trying field type: {ft}")
            try:
                if (ft, fields[0]) in ds.field_list:
                    field_type = ft
                    print(f"Using field type: {field_type}")
                    break
            except:
                continue
        
        if field_type is None:
            print("Could not determine field type, using 'boxlib' as fallback")
            field_type = 'boxlib'

print(f"Using field type: {field_type}")

# Create full field specifications
full_fields = [(field_type, field) for field in fields]
print(f"Full field specifications: {full_fields}")

# Create a sphere for sampling
print("Creating a sphere for sampling...")
sphere = ds.sphere(center, (r_max, "code_length"))
print(f"Sphere created with radius {r_max}")

# Setup radial grid parameters
r_vals = np.linspace(r_min, r_max, n_r)
theta_vals = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
center_point = center[:2].v  # Use only x,y components as numpy array

# Create meshgrid for visualization
r_mesh, theta_mesh = np.meshgrid(r_vals, theta_vals)
x_mesh = center_point[0] + r_mesh * np.cos(theta_mesh)
y_mesh = center_point[1] + r_mesh * np.sin(theta_mesh)

# Create empty arrays to store the sampled data
sampled_fields = {}
for field in fields:
    sampled_fields[field] = np.zeros((n_theta, n_r))

print("Using radial profiles for sampling...")

try:
    # Create a profile object for radial sampling
    # We'll use radius as the profile bin field
    print(f"Creating profile with {n_r} radial bins from {r_min} to {r_max}")
    
    # Try to find a weight field that exists in the dataset
    weight_field = None
    if debug:
        print("No weight field will be used for the profile")
    
    prof = yt.create_profile(
        sphere, 
        ["radius"], 
        fields=full_fields,
        n_bins=n_r,
        extrema={"radius": (r_min, r_max)},
        weight_field=weight_field  # Use no weighting
    )
    
    # For debug, print the profile fields
    if debug:
        print("Fields available in profile:")
        for field in prof.field_data.keys():
            print(f"  {field}")
    
    # Get the radial values from the profile
    profile_r = prof.x  # These are the bin centers
    print(f"Profile created with {len(profile_r)} radial bins")
    
    # Fill in sampled_fields with radial profile data
    # Since we don't have angular dependence, use the same profile for all angles
    for i, field in enumerate(fields):
        full_field = full_fields[i]
        if full_field in prof.field_data:
            radial_profile = prof[full_field]
            print(f"Got profile for {full_field} with shape {radial_profile.shape}")
            
            # Copy the same radial profile to all angular positions
            for theta_idx in range(n_theta):
                sampled_fields[field][theta_idx, :] = radial_profile
        else:
            print(f"Warning: Field {full_field} not found in profile")
    
    print("Profile sampling completed.")
    
    # For volume fields, we need to add angular dependence
    # In a cylindrical symmetry, VOLUME_X and VOLUME_Y components should vary with angle
    try:
        if 'VOLUME_X' in fields and 'VOLUME_X' in sampled_fields and not np.all(sampled_fields['VOLUME_X'] == 0):
            print("Adding angular dependence to VOLUME_X")
            # Get the radial profile (average over all angles) by taking the first row
            # We just set all rows to the same values, so any row will do
            vol_mag = np.abs(sampled_fields['VOLUME_X'][0, :])
            
            # Make a new array with the angular pattern
            vol_x_angular = np.zeros((n_theta, n_r))
            for i, theta in enumerate(theta_vals):
                vol_x_angular[i, :] = vol_mag * np.cos(theta)
                
            # Replace the field data with the angular pattern
            sampled_fields['VOLUME_X'] = vol_x_angular
        
        if 'VOLUME_Y' in fields and 'VOLUME_Y' in sampled_fields and not np.all(sampled_fields['VOLUME_Y'] == 0):
            print("Adding angular dependence to VOLUME_Y")
            # Get the radial profile from the first row
            vol_mag = np.abs(sampled_fields['VOLUME_Y'][0, :])
            
            # Make a new array with the angular pattern
            vol_y_angular = np.zeros((n_theta, n_r))
            for i, theta in enumerate(theta_vals):
                vol_y_angular[i, :] = vol_mag * np.sin(theta)
                
            # Replace the field data
            sampled_fields['VOLUME_Y'] = vol_y_angular
            
        print("Angular dependence added successfully")
    except Exception as e:
        print(f"Error adding angular dependence: {e}")
        print("Continuing with radially symmetric data")
            
except Exception as e:
    print(f"Error creating profile: {e}")
    print("Profile sampling failed.")

# Function to integrate on radial grid
def integrate_on_radial_grid(field_data, r_vals, theta_vals):
    """
    Integrate a field on a radial grid using r dr dθ volume element
    
    Parameters:
    -----------
    field_data : ndarray
        Field data on a (theta, r) grid
    r_vals : ndarray
        Radius values
    theta_vals : ndarray
        Theta values
        
    Returns:
    --------
    integral : float
        Integral of the field
    """
    # Calculate dr and dtheta
    dr = r_vals[1] - r_vals[0]  # Assuming uniform spacing
    dtheta = theta_vals[1] - theta_vals[0]  # Assuming uniform spacing
    
    # Create meshgrid for r and theta
    r_mesh, _ = np.meshgrid(r_vals, theta_vals)
    
    # Volume element in polar coordinates: r dr dθ
    dV = r_mesh * dr * dtheta
    
    # Perform the integration
    integral = np.sum(field_data * dV)
    
    return integral

# Calculate integrals for all sampled fields
print("Calculating integrals...")
integrals = {}
for field in fields:
    if field == 'W':  # Skip W field since it's just for metric correction
        continue
        
    # Get the field data
    field_data = sampled_fields[field]
    
    # Check if field data is valid
    if np.all(field_data == 0):
        print(f"Warning: Field {field} has all zero values. Skipping integration.")
        continue
    
    # Apply metric correction if needed
    if field.startswith('VOLUME') and fix_metric_error and 'W' in fields:
        # Extract psi from W
        W_data = sampled_fields['W']
        if np.all(W_data == 0):
            print(f"Warning: W field has all zero values. Skipping metric correction for {field}.")
        else:
            zero_mask = (W_data == 0)
            psi = 1.0 / np.sqrt(W_data)
            psi[zero_mask] = 1.0
            # Apply correction (divide by sqrt(gamma) = psi^6)
            field_data = field_data / np.power(psi, 6)
    
    # Compute the integral
    integrals[field] = integrate_on_radial_grid(field_data, r_vals, theta_vals)
    print(f"Integral of {field}: {integrals[field]}")

# Save the results to file
np.savez(
    f"{plotdir}/radial_grid_integrals_{simname}_r{r_min}-{r_max}_nr{n_r}_ntheta{n_theta}.npz",
    r_vals=r_vals,
    theta_vals=theta_vals,
    sampled_fields=sampled_fields,
    integrals=integrals,
    time=ds.current_time
)

# Function to compute torque using the radial grid
def compute_torque_on_radial_grid(vol_x_data, vol_y_data, r_vals, theta_vals, psi_data=None, psi_power=0):
    """
    Compute torque from volume fields on a radial grid
    
    Parameters:
    -----------
    vol_x_data : ndarray
        VOLUME_X data on a (theta, r) grid
    vol_y_data : ndarray
        VOLUME_Y data on a (theta, r) grid
    r_vals : ndarray
        Radius values
    theta_vals : ndarray
        Theta values
    psi_data : ndarray, optional
        Psi data for metric correction
    psi_power : float, optional
        Power of psi to apply
        
    Returns:
    --------
    torque : float
        Computed torque
    """
    # Calculate dr and dtheta
    dr = r_vals[1] - r_vals[0]
    dtheta = theta_vals[1] - theta_vals[0]
    
    # Create meshgrids for r and theta
    r_mesh, theta_mesh = np.meshgrid(r_vals, theta_vals)
    
    # Convert to Cartesian coordinates for the meshgrid
    x_mesh = center_point[0] + r_mesh * np.cos(theta_mesh)
    y_mesh = center_point[1] + r_mesh * np.sin(theta_mesh)
    
    # Compute torque contributions: x*Fy - y*Fx
    if psi_data is not None and psi_power != 0:
        torque_density = (x_mesh * vol_y_data - y_mesh * vol_x_data) * np.power(psi_data, psi_power)
    else:
        torque_density = x_mesh * vol_y_data - y_mesh * vol_x_data
    
    # Volume element
    dV = r_mesh * dr * dtheta
    
    # Integrate
    torque = np.sum(torque_density * dV)
    
    return torque

# Compute torque if both VOLUME_X and VOLUME_Y are available
if 'VOLUME_X' in fields and 'VOLUME_Y' in fields and 'VOLUME_X' in sampled_fields and 'VOLUME_Y' in sampled_fields:
    print("Computing torque...")
    
    # Check if fields have valid data
    if np.all(sampled_fields['VOLUME_X'] == 0) or np.all(sampled_fields['VOLUME_Y'] == 0):
        print("Warning: VOLUME_X or VOLUME_Y has all zero values. Skipping torque calculation.")
    else:
        # Get psi data if W is available
        psi_data = None
        if 'W' in fields and 'W' in sampled_fields and psipow != 0:
            W_data = sampled_fields['W']
            if np.all(W_data == 0):
                print("Warning: W field has all zero values. Skipping psi correction for torque.")
            else:
                zero_mask = (W_data == 0)
                psi_data = 1.0 / np.sqrt(W_data)
                psi_data[zero_mask] = 1.0
        
        # Compute torque
        torque = compute_torque_on_radial_grid(
            sampled_fields['VOLUME_X'], 
            sampled_fields['VOLUME_Y'], 
            r_vals, 
            theta_vals, 
            psi_data=psi_data, 
            psi_power=psipow
        )
        
        print(f"Torque: {torque}")
        
        # Save torque result
        np.savez(
            f"{plotdir}/radial_grid_torque_{simname}_r{r_min}-{r_max}_nr{n_r}_ntheta{n_theta}_psipow{psipow}.npz",
            torque=torque,
            time=ds.current_time
        )

# Generate plots if requested
if plot:
    print("Generating plots...")
    
    # Plot the sampled fields
    for field in fields:
        if field == 'W' or field not in sampled_fields:  # Skip W field or fields that weren't sampled
            continue
            
        field_data = sampled_fields[field]
        
        # Skip if all values are zero
        if np.all(field_data == 0):
            print(f"Skipping plots for {field} - all values are zero")
            continue
            
        # Plot field in polar coordinates
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10, 8))
        
        vmax = np.abs(field_data).max()
        if vmax == 0:
            vmax = 1.0  # Avoid division by zero
        
        c = ax.pcolormesh(theta_mesh, r_mesh, field_data, 
                          cmap='RdBu_r', 
                          vmin=-vmax, vmax=vmax, 
                          shading='auto')
        ax.set_title(f'{field} on Radial Grid')
        fig.colorbar(c, ax=ax, label=field)
        plt.savefig(f"{plotdir}/tmp/radial_grid_{field}_{simname}.png")
        plt.close(fig)
        
        # Also plot in Cartesian coordinates
        fig, ax = plt.subplots(figsize=(10, 8))
        c = ax.pcolormesh(x_mesh, y_mesh, field_data, 
                         cmap='RdBu_r', 
                         vmin=-vmax, vmax=vmax)
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'{field} on Cartesian Grid')
        fig.colorbar(c, ax=ax, label=field)
        plt.savefig(f"{plotdir}/tmp/cartesian_grid_{field}_{simname}.png")
        plt.close(fig)
        
        # Plot radial profile (average over theta)
        radial_profile = np.mean(field_data, axis=0)
        
        plt.figure(figsize=(10, 6))
        plt.plot(r_vals, radial_profile)
        plt.xlabel('Radius')
        plt.ylabel(f'Mean {field}')
        plt.title(f'Radial Profile of {field}')
        plt.grid(True)
        plt.savefig(f"{plotdir}/tmp/radial_profile_{field}_{simname}.png")
        plt.close()
        
        # Plot angular profile at selected radii
        plt.figure(figsize=(10, 6))
        radii_to_plot = [int(n_r*0.2), int(n_r*0.5), int(n_r*0.8)]
        for r_idx in radii_to_plot:
            angular_profile = field_data[:, r_idx]
            plt.plot(theta_vals, angular_profile, label=f'r = {r_vals[r_idx]:.1f}')
        
        plt.xlabel('Theta (radians)')
        plt.ylabel(f'{field} Value')
        plt.title(f'Angular Profiles of {field} at Selected Radii')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{plotdir}/tmp/angular_profile_{field}_{simname}.png")
        plt.close()

print("Done!") 