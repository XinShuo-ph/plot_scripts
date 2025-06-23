import yt
import numpy as np
import matplotlib.pyplot as plt
import os, argparse

parser = argparse.ArgumentParser(description="2D circular surface sampling.")
parser.add_argument('--simname', type=str, required=True, help="Simulation directory name")
parser.add_argument('--skipevery', type=int, default=1, help="Skip every n iterations")
parser.add_argument('--maxframes', type=int, default=400, help="Max number of frames to process")
parser.add_argument('--outR', type=float, default=650.0, help="Radius of the circular surface")
parser.add_argument('--integratefield', type=str, default='SURFACE_X', help="Field to sample on the surface (e.g., SURFACE_X)")
parser.add_argument('--allfields', action='store_true', help="Sample a set of surface fields (X, Y, Z)")
parser.add_argument('--n_theta', type=int, default=360, help="Number of angular samples on the circle")
parser.add_argument('--outplot', action='store_true', help="Whether to output a sample angular distribution plot")
args = parser.parse_args()

simname       = args.simname
skipevery     = args.skipevery
maxframes     = args.maxframes
outR          = args.outR
integratefield = args.integratefield
allfields     = args.allfields
n_theta       = args.n_theta
outplot       = args.outplot

basedir = "/pscratch/sd/x/xinshuo/runGReX/"
rundir  = os.path.join(basedir, simname)
plotdir = "/pscratch/sd/x/xinshuo/plotGReX/"

# Gather output directories
plt_dirs = sorted([d for d in os.listdir(rundir) if d.startswith('plt') and d[3:].isdigit()],
                  key=lambda x: int(x[3:]))
plt_dirs = plt_dirs[::skipevery]
if len(plt_dirs) > maxframes:
    plt_dirs = plt_dirs[:maxframes]

# Determine fields to sample
if allfields:
    fields_to_sample = ['SURFACE_X', 'SURFACE_Y', 'SURFACE_Z']
else:
    fields_to_sample = [integratefield]

# Prepare results list
results_vs_theta = []

# Precompute angular positions on the circle
theta_vals = np.linspace(0.0, 2.0*np.pi, n_theta, endpoint=False)
# Coordinates of the circle relative to domain center (assume center at (0,0) in domain coordinates)
# We will find the actual domain center from ds.domain_center for generality.
domain_center = None  # to be set after loading first dataset

for frame_idx, plt_dir in enumerate(plt_dirs):
    ds = yt.load(os.path.join(rundir, plt_dir))
    current_time = float(ds.current_time)
    if domain_center is None:
        # get domain center from dataset (assuming center of domain)
        domain_center = ds.domain_center.in_units('code_length').value  # get numpy array of center
    
    # Coordinates of sampling points on the circle (z = 0 plane)
    xc, yc = domain_center[0], domain_center[1]
    x_phys = xc + outR * np.cos(theta_vals)
    y_phys = yc + outR * np.sin(theta_vals)
    z_phys = np.ones_like(x_phys) * domain_center[2]  # use mid-plane (assuming 0 or center in z)
    points = np.vstack([x_phys, y_phys, z_phys]).T  # shape (n_theta, 3)
    
    # Sample each field at these circle points
    theta_values = np.zeros((n_theta, len(fields_to_sample)))
    for j, field in enumerate(fields_to_sample):
        # Sum over z-direction: for surface fields, the data might already represent flux through surface.
        # In case we need to integrate through z (if "surface" fields are 2D slices already), we ensure we sum across any minor z thickness.
        # Here we assume surface fields are defined similarly to volume ones but for surfaces, and have been computed as in original scripts.
        data_vals = ds.sample(points, field)[field]
        theta_values[:, j] = np.array(data_vals)
    results_vs_theta.append([current_time, theta_values])

# Convert to numpy arrays (time series)
times = np.array([entry[0] for entry in results_vs_theta])
data = np.stack([entry[1] for entry in results_vs_theta], axis=0)  # shape (N_frames, n_theta, n_fields)
outfile = f"{simname}_2d_surface_values_outR{int(outR)}.npy"
np.save(outfile, data)
print(f"Saved surface sampling results to {outfile} (shape {data.shape}).")

# Optional: plot an example angular distribution (for the first field at last time)
if outplot:
    plt.figure(figsize=(8,5))
    field_label = fields_to_sample[0]
    plt.plot(theta_vals, data[-1,:,0], label=f"{field_label} (t={times[-1]:.1f})")
    plt.xlabel("Angle θ (rad)")
    plt.ylabel(f"{field_label} value")
    plt.title(f"{field_label} vs θ on surface r={outR} at t={times[-1]:.1f}")
    plt.legend()
    plt.savefig(f"{simname}_{field_label}_surface_vs_theta_outR{int(outR)}.png")
    plt.close()
