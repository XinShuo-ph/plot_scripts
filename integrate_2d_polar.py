import yt
import numpy as np
import matplotlib.pyplot as plt
import os, argparse

# Parse arguments similar to test_integrate_2d.py:contentReference[oaicite:3]{index=3}:contentReference[oaicite:4]{index=4}
parser = argparse.ArgumentParser(description="2D volume integration with polar sampling.")
parser.add_argument('--simname', type=str, required=True, help="Simulation directory name")
parser.add_argument('--skipevery', type=int, default=1, help="Skip every n iterations")
parser.add_argument('--maxframes', type=int, default=400, help="Max number of frames to process")
parser.add_argument('--outR', type=float, default=650.0, help="Outer radius of the integration disk")
parser.add_argument('--innerR', type=float, default=-1.0, help="Inner excision radius (if positive)")
parser.add_argument('--bbh1_x', type=float, default=0.0, help="x-coordinate of first BH (if applicable)")
parser.add_argument('--bbh2_x', type=float, default=0.0, help="x-coordinate of second BH")
parser.add_argument('--bbh1_r', type=float, default=0.0, help="Excision radius of first BH")
parser.add_argument('--bbh2_r', type=float, default=0.0, help="Excision radius of second BH")
parser.add_argument('--binary_omega', type=float, default=0.0, help="Orbital angular frequency of binary (for BH positions)")
parser.add_argument('--excise_factor', type=float, default=1.5, help="Factor to scale BH radii for excision")
parser.add_argument('--integratefield', type=str, default='VOLUME_X', help="Field to integrate (e.g., VOLUME_X or SMOMENTUM_X)")
parser.add_argument('--allfields', action='store_true', help="Integrate a preset list of fields (e.g., all components)")
parser.add_argument('--n_theta', type=int, default=360, help="Number of angular samples (θ divisions)")
parser.add_argument('--n_radius', type=int, default=1000, help="Number of radial samples for integration")
parser.add_argument('--outplot', action='store_true', help="Whether to output a sample polar distribution plot")
args = parser.parse_args()

# Unpack arguments
simname       = args.simname
skipevery     = args.skipevery
maxframes     = args.maxframes
outR          = args.outR
innerR        = args.innerR
bbh1_x        = args.bbh1_x
bbh2_x        = args.bbh2_x
bbh1_r        = args.bbh1_r
bbh2_r        = args.bbh2_r
binary_omega  = args.binary_omega
excise_factor = args.excise_factor
integratefield = args.integratefield
allfields     = args.allfields
n_theta       = args.n_theta
n_radius      = args.n_radius
outplot       = args.outplot

# Setup directory paths (consistent with test_integrate_2d.py):contentReference[oaicite:5]{index=5}
basedir = "/pscratch/sd/x/xinshuo/runGReX/"
rundir  = os.path.join(basedir, simname)
plotdir = "/pscratch/sd/x/xinshuo/plotGReX/"

# Collect plot directories and limit frames:contentReference[oaicite:6]{index=6}
plt_dirs = sorted([d for d in os.listdir(rundir) if d.startswith('plt') and d[3:].isdigit()], 
                  key=lambda x: int(x[3:]))
plt_dirs = plt_dirs[::skipevery]
if len(plt_dirs) > maxframes:
    plt_dirs = plt_dirs[:maxframes]

# Determine which fields to integrate
if allfields:
    # Integrate a standard set of 2D volume fields (X, Y, Z components, etc.)
    fields_to_integrate = ['VOLUME_X', 'VOLUME_Y', 'VOLUME_Z', 'SMOMENTUM_X', 'SMOMENTUM_Y', 'SMOMENTUM_Z']
else:
    fields_to_integrate = [integratefield]

# Prepare results container (will be 3D if multiple fields: time x theta x field_index)
results_vs_theta = []

# Precompute angular sampling points and weights
theta_vals = np.linspace(0.0, 2.0*np.pi, n_theta, endpoint=False)
dtheta = 2.0 * np.pi / n_theta  # uniform angle width

# Loop over data frames
for frame_idx, plt_dir in enumerate(plt_dirs):
    ds = yt.load(os.path.join(rundir, plt_dir))
    current_time = float(ds.current_time)  # current simulation time
    
    # Prepare storage for this time step: shape (n_theta, n_fields)
    theta_integrals = np.zeros((n_theta, len(fields_to_integrate)))
    
    # For each angular sample, perform radial integration
    for i, theta in enumerate(theta_vals):
        # Determine radial start (account for inner excision)
        r_start = innerR if innerR > 0 else 0.0
        
        # If no innerR, we might have excision around BHs: compute intersections if needed
        # (Find if radial ray at this θ intersects BH1 or BH2 excision regions)
        # Compute BH positions at this time (if binary_omega provided)
        BH1_center = None
        BH2_center = None
        if innerR <= 0 and bbh1_r > 0 and bbh2_r > 0:
            x1 = bbh1_x * np.cos(-current_time * binary_omega)
            y1 = bbh1_x * np.sin(-current_time * binary_omega)
            x2 = bbh2_x * np.cos(-current_time * binary_omega)
            y2 = bbh2_x * np.sin(-current_time * binary_omega)
            BH1_center = (x1, y1)
            BH2_center = (x2, y2)
        # Now perform radial sampling from r_start to outR
        r_vals = np.linspace(r_start, outR, n_radius)
        dr = r_vals[1] - r_vals[0] if n_radius > 1 else outR  # radial step size
        
        # Coordinates along this radial ray (z = 0 plane, since we integrate through z later)
        x_coords = r_vals * np.cos(theta)
        y_coords = r_vals * np.sin(theta)
        z_coords = np.zeros_like(x_coords)  # we will sum over z via data later
        
        # Use YT to sample field values along this ray. We can use ds.sample or ds.point for each coordinate.
        # Here we use ds.sample to interpolate values from the dataset at given points:
        points = np.vstack([x_coords, y_coords, z_coords]).T  # shape (n_radius, 3)
        
        # For each field, interpolate values at these points:
        for j, field in enumerate(fields_to_integrate):
            # Note: we use YT's interpolate to get field values at specified points (in code units).
            # ds.sample() can be used to sample multiple fields at once. We'll sample each field separately for clarity.
            field_values = ds.sample(points, field)[field]
            field_values = np.array(field_values)  # convert YT data to numpy array
            
            # Apply inner excision for BH masks if defined (set values to 0 inside BH excision)
            if BH1_center is not None:
                # BH1 excision: mask points where (x - x1)^2 + (y - y1)^2 < (bbh1_r * excise_factor)^2
                dx1 = x_coords - BH1_center[0]
                dy1 = y_coords - BH1_center[1]
                inside_bh1 = (dx1**2 + dy1**2) < (bbh1_r * excise_factor)**2
                field_values[inside_bh1] = 0.0
            if BH2_center is not None:
                dx2 = x_coords - BH2_center[0]
                dy2 = y_coords - BH2_center[1]
                inside_bh2 = (dx2**2 + dy2**2) < (bbh2_r * excise_factor)**2
                field_values[inside_bh2] = 0.0
            
            # Sum over z-direction: In the dataset, fields like 'VOLUME_X' might already represent integrated or density values.
            # If needed, integrate through the full z extent. For simplicity, assume fields are volume densities integrated along z by YT sample.
            # (Alternatively, one could use ds.ray or ds.region with height equal to domain to integrate, but ds.sample on z=0 suffices if data symmetric.)
            
            # Perform radial integration using the polar area element:
            # Integral ≈ Σ [field_value(r) * r * dr]  (with uniform dθ accounted separately)
            radial_integral = np.sum(field_values * r_vals) * dr
            theta_integrals[i, j] = radial_integral
    # Multiply by dθ to account for angular width (completing the area element r dr dθ)
    theta_integrals *= dtheta
    
    # Append results [time, distribution...] for each field
    # If multiple fields, we will store separate distributions concatenated or keep 3D array.
    results_vs_theta.append([current_time, theta_integrals])
    print(f"Frame {frame_idx}: time={current_time:.3f} integrated.")
    
# Convert results list to numpy array
# Shape will be (N_frames, 2) where results_vs_theta[k][0] is time and [1] is array (n_theta, n_fields)
results_array = np.array(results_vs_theta, dtype=object)  # using object dtype to hold arrays
# To make usage easier, we can separate time and data:
times = np.array([entry[0] for entry in results_vs_theta])
data = np.stack([entry[1] for entry in results_vs_theta], axis=0)  # shape (N_frames, n_theta, n_fields)

# Save results to .npy file (include outR and possibly note of polar sampling in filename)
outfile = f"{simname}_2d_integrals_polar_outR{int(outR)}.npy"
np.save(outfile, data)
print(f"Saved polar integration results to {outfile} (shape {data.shape}).")

# Optionally, plot a sample distribution (e.g., first field at final time) for verification
if outplot:
    plt.figure(figsize=(8,5))
    # Plot distribution of first field at last time step
    label_field = fields_to_integrate[0] if len(fields_to_integrate)==1 else fields_to_integrate[0]
    plt.plot(theta_vals, data[-1,:,0], label=f"{label_field} (t={times[-1]:.1f})")
    plt.xlabel("Angle θ (rad)")
    plt.ylabel(f"{label_field} integrated along r")
    plt.title(f"{label_field} vs θ at time {times[-1]:.1f}, outR={outR}")
    plt.legend()
    plt.savefig(f"{simname}_{label_field}_vs_theta_outR{int(outR)}.png")
    plt.close()
