import yt
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse


binary_mass = +2.71811e+00
max_excise_factor = 2.0

parser = argparse.ArgumentParser(description='Plot scalar field rho from simulation data using polar mesh sampling.')
parser.add_argument('--simname', type=str, default="240930_BBH_3D_zboost_v5", help='Name of the simulation directory')
parser.add_argument('--skipevery', type=int, default=1, help='Skip every n iterations')
parser.add_argument('--maxframes', type=int, default=400, help='max num of frames (to avoid OOM)')
parser.add_argument('--outR', type=float, default=650, help='outer radius of the integration sphere')
parser.add_argument('--innerR', type=float, default=-1, help='inner radius of the integration sphere')
parser.add_argument('--r_min', type=float, default=0.2, help='minimum radius for polar mesh')
parser.add_argument('--num_r', type=int, default=100, help='number of radial bins')
parser.add_argument('--num_theta', type=int, default=200, help='number of angular bins')
parser.add_argument('--bbh1_x', type=float, default=-(70.49764373525885/2-2.926860031395978)/binary_mass, help='x coordinate of the first black hole') 
parser.add_argument('--bbh2_x', type=float, default= (70.49764373525885/2+2.926860031395978)/binary_mass, help='x coordinate of the second black hole')
parser.add_argument('--bbh1_r', type=float, default=3.98070/binary_mass, help='radius of the first black hole')
parser.add_argument('--bbh2_r', type=float, default=3.98070/binary_mass, help='radius of the second black hole')
parser.add_argument('--binary_omega', type=float, default=- 0.002657634562418009 * 2.71811, help='orbital frequency of the binary')
parser.add_argument('--excise_factor', type=float, default=1.5, help='factor to excise the black hole')
parser.add_argument('--outplot', action='store_true', help='Output the plot')
parser.add_argument('--fix_metric_error', action='store_true', help='Fix the metric error in VOLUME fields (apply 1/sqrt(gamma) correction)')
parser.add_argument('--psipow', type=float, default=2, help='power of psi factor')
parser.add_argument('--debug', action='store_true', help='Enable debug output')

args = parser.parse_args()
simname = args.simname
skipevery = args.skipevery
maxframes = args.maxframes
outR = args.outR
innerR = args.innerR
r_min = args.r_min
num_r = args.num_r
num_theta = args.num_theta
bbh1_x = args.bbh1_x
bbh2_x = args.bbh2_x
bbh1_r = args.bbh1_r
bbh2_r = args.bbh2_r
binary_omega = args.binary_omega
excise_factor = args.excise_factor
outplot = args.outplot
fix_metric_error = args.fix_metric_error
psipow = args.psipow
debug = args.debug

basedir = "/pscratch/sd/x/xinshuo/runGReX/"
plotdir = "/pscratch/sd/x/xinshuo/plotGReX/"
rundir = basedir + simname +"/"

# make /pscratch/sd/x/xinshuo/plotGReX/tmp if it doesn't exist
if not os.path.exists(plotdir + "tmp"):
    os.makedirs(plotdir + "tmp")

plt_dirs = sorted([d for d in os.listdir(rundir) if d.startswith('plt') and d[3:].isdigit()], key=lambda x: int(x[3:]))
plt_dirs = plt_dirs[::skipevery]

if len(plt_dirs) > maxframes:
    plt_dirs = plt_dirs[:maxframes]

# Create a polar mesh with logarithmic radial spacing
r_max = outR
r_bins = np.logspace(np.log10(r_min), np.log10(r_max), num_r)
theta_bins = np.linspace(0, 2*np.pi, num_theta, endpoint=False)

# Create meshgrid for polar coordinates
theta_mesh, r_mesh = np.meshgrid(theta_bins, r_bins)

# Calculate dr and dtheta for area elements
dr_vals = np.diff(r_bins, append=r_bins[-1] + (r_bins[-1] - r_bins[-2]))
dtheta = theta_bins[1] - theta_bins[0]

# Expand to match mesh dimensions
dr_mesh = np.tile(dr_vals, (num_theta, 1)).T
dtheta_mesh = np.full_like(r_mesh, dtheta)

# Area elements for integration (r*dr*dtheta)
dA_mesh = r_mesh * dr_mesh * dtheta_mesh

# Convert to Cartesian coordinates for sampling
x_mesh = r_mesh * np.cos(theta_mesh)
y_mesh = r_mesh * np.sin(theta_mesh)

if debug:
    print(f"Created polar mesh with {num_r} logarithmic radial bins from r={r_min} to r={r_max}")
    print(f"Angular resolution: {num_theta} bins ({360/num_theta} degrees per bin)")
    print(f"dr range: {dr_mesh.min()} to {dr_mesh.max()}")
    print(f"dtheta: {dtheta} radians ({dtheta*180/np.pi} degrees)")

results = []

for frameidx, plt_dir in enumerate(plt_dirs):
    print(f"Processing frame {frameidx+1}/{len(plt_dirs)}: {plt_dir}")
    ds = yt.load(os.path.join(rundir, plt_dir))
    
    # Get level information
    level_left_edges = np.array([ds.index.grid_left_edge[np.where(ds.index.grid_levels==[i])[0]].min(axis=0) for i in range(ds.max_level+1)])
    level_right_edges = np.array([ds.index.grid_right_edge[np.where(ds.index.grid_levels==[i])[0]].max(axis=0) for i in range(ds.max_level+1)])
    level_dxs = ds.index.level_dds
    level_dims = ((level_right_edges - level_left_edges + 1e-10) / level_dxs).astype(int)
    
    # Create covering grids for each level
    field_ds_levels = {}
    for level in range(ds.max_level+1):
        field_ds_levels[level] = ds.covering_grid(level=level, 
                                                 left_edge=level_left_edges[level], 
                                                 dims=level_dims[level])

    # Initialize field arrays for the polar mesh
    vol_x_mesh = np.zeros_like(r_mesh)
    vol_y_mesh = np.zeros_like(r_mesh)
    vol_z_mesh = np.zeros_like(r_mesh)
    mom_x_mesh = np.zeros_like(r_mesh)
    mom_y_mesh = np.zeros_like(r_mesh)
    mom_z_mesh = np.zeros_like(r_mesh)
    psi_mesh = np.zeros_like(r_mesh)
    
    # Track which points have been assigned
    assigned = np.zeros_like(r_mesh, dtype=bool)
    
    # Black hole positions at current time
    t = float(ds.current_time)
    bh1_center = np.array([bbh1_x*np.cos(-t*binary_omega), bbh1_x*np.sin(-t*binary_omega), 0])
    bh2_center = np.array([bbh2_x*np.cos(-t*binary_omega), bbh2_x*np.sin(-t*binary_omega), 0])
    
    # Create masks for black hole excision
    bh1_dist = np.sqrt((x_mesh - bh1_center[0])**2 + (y_mesh - bh1_center[1])**2)
    bh2_dist = np.sqrt((x_mesh - bh2_center[0])**2 + (y_mesh - bh2_center[1])**2)
    bh1_mask = bh1_dist < bbh1_r * excise_factor
    bh2_mask = bh2_dist < bbh2_r * excise_factor
    excision_mask = bh1_mask | bh2_mask
    
    # Helper function to sample field value at a specific point (x,y)
    def sample_field_value(x, y, field, level):
        # Convert from physical coordinates to array indices
        xmin, xmax = level_left_edges[level][0], level_right_edges[level][0]
        ymin, ymax = level_left_edges[level][1], level_right_edges[level][1]
        nx, ny = field_ds_levels[level][field][:].shape[:2]
        
        dx = (xmax - xmin) / nx
        dy = (ymax - ymin) / ny
        
        i = int(np.clip(np.floor((x - xmin) / dx), 0, nx-1))
        j = int(np.clip(np.floor((y - ymin) / dy), 0, ny-1))
        
        # Return the value at this position (sum over z)
        return field_ds_levels[level][field][:][i, j, :].sum()
    
    # Loop through the levels from finest to coarsest
    for level in range(ds.max_level, -1, -1):
        if debug:
            print(f"Processing level {level}")
        
        # Create a mask for points inside this level's domain
        level_mask = ((x_mesh >= level_left_edges[level][0]) & 
                     (x_mesh <= level_right_edges[level][0]) & 
                     (y_mesh >= level_left_edges[level][1]) & 
                     (y_mesh <= level_right_edges[level][1]))
        
        # For all levels except the finest, exclude points in finer levels
        if level < ds.max_level:
            next_level_mask = ((x_mesh >= level_left_edges[level+1][0]) & 
                              (x_mesh <= level_right_edges[level+1][0]) & 
                              (y_mesh >= level_left_edges[level+1][1]) & 
                              (y_mesh <= level_right_edges[level+1][1]))
            level_mask = level_mask & ~next_level_mask
        
        # Also exclude points inside black holes
        level_mask = level_mask & ~excision_mask
        
        # Skip level if no points are in this mask
        if not np.any(level_mask):
            if debug:
                print(f"  No points in level {level} mask, skipping")
            continue
            
        # Mark these points as assigned
        assigned = assigned | level_mask
        
        # Compute W and psi for the current level
        W_field = field_ds_levels[level]['W'][:].sum(axis=2)
        psi_values = np.zeros_like(r_mesh)
        
        # Sample all field values for this level
        for i, j in zip(*np.where(level_mask)):
            x, y = x_mesh[i, j], y_mesh[i, j]
            
            # Convert to array indices for direct access (faster than calling sample_field_value repeatedly)
            xmin, xmax = level_left_edges[level][0], level_right_edges[level][0]
            ymin, ymax = level_left_edges[level][1], level_right_edges[level][1]
            nx, ny = W_field.shape
            
            dx = (xmax - xmin) / nx
            dy = (ymax - ymin) / ny
            
            idx_i = int(np.clip(np.floor((x - xmin) / dx), 0, nx-1))
            idx_j = int(np.clip(np.floor((y - ymin) / dy), 0, ny-1))
            
            # Get W value and compute psi
            W_val = W_field[idx_i, idx_j]
            if W_val > 0:
                psi_val = 1.0 / np.sqrt(W_val)
            else:
                psi_val = 1.0
            psi_mesh[i, j] = psi_val
            
            # Compute sqrt(gamma) for volume elements
            sqrt_gamma = np.power(psi_val, 6)
            
            # Sample all fields
            for field in ['VOLUME_X', 'VOLUME_Y', 'VOLUME_Z', 'SMOMENTUM_X', 'SMOMENTUM_Y', 'SMOMENTUM_Z']:
                val = field_ds_levels[level][field][:][idx_i, idx_j, :].sum()
                
                # Apply metric corrections
                if field.startswith('VOLUME'):
                    if fix_metric_error:
                        # Divide by sqrt(gamma) to fix the metric error
                        val = val / sqrt_gamma
                else:  # SMOMENTUM fields
                    # Multiply by sqrt(gamma) for proper volume integration
                    val = val * sqrt_gamma
                
                # Store in appropriate mesh
                if field == 'VOLUME_X':
                    vol_x_mesh[i, j] = val
                elif field == 'VOLUME_Y':
                    vol_y_mesh[i, j] = val
                elif field == 'VOLUME_Z':
                    vol_z_mesh[i, j] = val
                elif field == 'SMOMENTUM_X':
                    mom_x_mesh[i, j] = val
                elif field == 'SMOMENTUM_Y':
                    mom_y_mesh[i, j] = val
                elif field == 'SMOMENTUM_Z':
                    mom_z_mesh[i, j] = val
    
    # Check if we have any unassigned points that should be assigned
    if debug and np.any(~assigned & ~excision_mask & (r_mesh <= outR)):
        n_unassigned = np.sum(~assigned & ~excision_mask & (r_mesh <= outR))
        print(f"Warning: {n_unassigned} points within r={outR} not assigned to any level")
    
    # Calculate the volume integrals
    # For torque, we need x * vol_y - y * vol_x
    torque_mesh = (x_mesh * vol_y_mesh - y_mesh * vol_x_mesh) * np.power(psi_mesh, psipow)
    # For angular momentum, we need x * mom_y - y * mom_x
    L_z_mesh = (x_mesh * mom_y_mesh - y_mesh * mom_x_mesh) * np.power(psi_mesh, psipow)
    
    # Integrate using the area elements
    vol_x = np.sum(vol_x_mesh * dA_mesh)
    vol_y = np.sum(vol_y_mesh * dA_mesh)
    vol_z = np.sum(vol_z_mesh * dA_mesh)
    mom_x = np.sum(mom_x_mesh * dA_mesh)
    mom_y = np.sum(mom_y_mesh * dA_mesh)
    mom_z = np.sum(mom_z_mesh * dA_mesh)
    torque = np.sum(torque_mesh * dA_mesh)
    L_z = np.sum(L_z_mesh * dA_mesh)
    
    # Output some plots for debugging/verification
    if outplot:
        # Plot the volume integrands
        for field_name, field_mesh in [
            ('VOLUME_X', vol_x_mesh),
            ('VOLUME_Y', vol_y_mesh),
            ('VOLUME_Z', vol_z_mesh),
            ('SMOMENTUM_X', mom_x_mesh),
            ('SMOMENTUM_Y', mom_y_mesh),
            ('SMOMENTUM_Z', mom_z_mesh),
            ('TORQUE', torque_mesh),
            ('L_Z', L_z_mesh),
            ('PSI', psi_mesh)
        ]:
            plt.figure(figsize=(10, 10))
            # Use pcolormesh for a polar plot
            vmax = np.abs(field_mesh).max()
            if vmax == 0:
                vmax = 1.0
            plt.pcolormesh(x_mesh, y_mesh, field_mesh, cmap='RdBu', vmin=-vmax, vmax=vmax)
            plt.colorbar(label=field_name)
            
            # Draw the black hole excision regions
            bh1_circle = plt.Circle((bh1_center[0], bh1_center[1]), bbh1_r * excise_factor, 
                                   fill=False, color='k', linestyle='--')
            bh2_circle = plt.Circle((bh2_center[0], bh2_center[1]), bbh2_r * excise_factor, 
                                   fill=False, color='k', linestyle='--')
            plt.gca().add_patch(bh1_circle)
            plt.gca().add_patch(bh2_circle)
            
            # Draw the integration boundary
            outer_circle = plt.Circle((0, 0), outR, fill=False, color='k')
            plt.gca().add_patch(outer_circle)
            
            plt.title(f'{field_name} at time {float(ds.current_time):.2f}')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.axis('equal')
            plt.grid(True)
            plt.savefig(f"{plotdir}/tmp/{field_name}_{frameidx}_polar.png")
            plt.close()
    
    # Store the results
    results.append([ds.current_time, vol_x, vol_y, vol_z, mom_x, mom_y, mom_z, torque, L_z])

# Convert results to numpy array and save
results = np.array(results)
np.save(f"{simname}_2d_integrals_polar_outR{outR}_excise{excise_factor}_psipow{psipow}.npy", results)

# Plot volume integrals
plt.figure()
plt.plot(results[:,0], results[:,1], label='VOLUME_X')
plt.plot(results[:,0], results[:,2], label='VOLUME_Y')
plt.plot(results[:,0], results[:,3], label='VOLUME_Z')
plt.plot(results[:,0], results[:,7], label='TORQUE')
plt.xlabel('Time')
plt.ylabel('2D Integral')
plt.legend()
plt.savefig(f"{simname}_2d_volume_integrals_polar_outR{outR}_excise{excise_factor}_psipow{psipow}.png")
plt.close()

# Plot momentum integrals
plt.figure()
plt.plot(results[:,0], results[:,4], label='SMOMENTUM_X')
plt.plot(results[:,0], results[:,5], label='SMOMENTUM_Y')
plt.plot(results[:,0], results[:,6], label='SMOMENTUM_Z')
plt.plot(results[:,0], results[:,8], label='ANGULAR MOMENTUM')
plt.xlabel('Time')
plt.ylabel('2D Integral')
plt.legend()
plt.savefig(f"{simname}_2d_momentum_integrals_polar_outR{outR}_excise{excise_factor}_psipow{psipow}.png")
plt.close() 