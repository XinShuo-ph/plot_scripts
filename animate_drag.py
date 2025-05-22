import yt
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import imageio
from matplotlib.patches import Circle


plt.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": 'cm',  # Computer Modern font - looks like LaTeX
    "font.family": 'STIXGeneral'
})

parser = argparse.ArgumentParser(description='Animate drag forces (VOLUME_X, VOLUME_Y) in simulation data.')
parser.add_argument('--simname', type=str, required=True,
                    help='Name of the simulation directory (under /pscratch/sd/x/xinshuo/runGReX/)')
parser.add_argument('--skipevery', type=int, default=1,
                    help='Skip every n iterations (1 = no skip)')
parser.add_argument('--maxframes', type=int, default=200,
                    help='Maximum number of frames to process')
parser.add_argument('--extent_x', type=float, default=40,
                    help='Half-width of the plot region in x (code units)')
parser.add_argument('--extent_y', type=float, default=40,
                    help='Half-width of the plot region in y (code units)')
parser.add_argument('--cmap', type=str, default='RdBu_r',
                    help='Colormap for the torque field (e.g., RdBu, RdBu_r)')
parser.add_argument('--fontsize', type=int, default=16,
                    help='Font size for labels and annotations')
parser.add_argument('--arrow_step', type=int, default=8,
                    help='Plot every Nth vector in the quiver (to reduce clutter)')
parser.add_argument('--Torbit', type=float,
                    default=2*np.pi/(0.0077395162481920582*2.71811),
                    help='Orbital period of the binary (for time normalization)')
parser.add_argument('--labelT', action='store_true',
                    help='Label time in units of Torbit if set (requires --Torbit)')
parser.add_argument('--bbh1_x',      type=float, default=-(70.49764373525885/2-2.926860031395978)/2.71811)
parser.add_argument('--bbh2_x',      type=float, default= (70.49764373525885/2+2.926860031395978)/2.71811)
parser.add_argument('--bbh1_r',      type=float, default=3.98070/2.71811)   # BH horizon radius
parser.add_argument('--bbh2_r',      type=float, default=3.98070/2.71811)
parser.add_argument('--binary_omega',type=float, default=-0.002657634562418009*2.71811)
parser.add_argument('--excise_factor',type=float, default=1.5,
                    help='Radius multiplier for the excision region')
parser.add_argument('--auto_torque_max', action='store_true',
                    help='Automatically determine the maximum torque magnitude')
parser.add_argument('--torque_max', type=float, default=9e-5,
                    help='Maximum torque magnitude (used if --auto_torque_max is not set)')
args = parser.parse_args()

# Extract arguments
simname   = args.simname
skipevery = args.skipevery
maxframes = args.maxframes
extent_x  = args.extent_x
extent_y  = args.extent_y
cmap_name = args.cmap
fontsize  = args.fontsize
arrow_step = args.arrow_step
Torbit    = args.Torbit
labelT    = args.labelT
auto_torque_max = args.auto_torque_max
torque_max = args.torque_max

# Define base directories
basedir = "/pscratch/sd/x/xinshuo/runGReX/"
plotdir = "/pscratch/sd/x/xinshuo/plotGReX/"
rundir  = os.path.join(basedir, simname)

# Ensure output frames directory exists
frames_dir = os.path.join(plotdir, simname, "frames")
os.makedirs(frames_dir, exist_ok=True)

# Get list of output directories
plt_dirs = sorted([d for d in os.listdir(rundir) if d.startswith('plt') and d[3:].isdigit()],
                  key=lambda x: int(x[3:]))
plt_dirs = plt_dirs[::skipevery]               # apply skipping
if len(plt_dirs) > maxframes:                  # apply max frame limit
    plt_dirs = plt_dirs[:maxframes]

allfiles = []  # to store frame file paths

for frame_idx, plt_dir in enumerate(plt_dirs):
    ds_path = os.path.join(rundir, plt_dir)
    ds = yt.load(ds_path)  # load dataset
    # Slice the dataset at z = 0 plane
    slc = ds.slice('z', 0.0)
    # Create a fixed-resolution buffer (FRB) for the slice over the specified region
    N = 400  # resolution of the FRB (e.g., 400x400 pixels)
    frb = yt.FixedResolutionBuffer(slc,
                                   bounds=(-extent_x, extent_x, -extent_y, extent_y),
                                   buff_size=(N, N))
    # Fetch the VOLUME_X and VOLUME_Y fields on the slice as numpy arrays
    vol_x = np.array(frb["VOLUME_X"])
    vol_y = np.array(frb["VOLUME_Y"])
    # Coordinate arrays for the grid, to compute torque
    x_vals = np.linspace(-extent_x, extent_x, vol_x.shape[1])
    y_vals = np.linspace(-extent_y, extent_y, vol_y.shape[0])
    X, Y = np.meshgrid(x_vals, y_vals)

    # ── excise interior of BHs by zeroing data ─────────────────────────────
    t = float(ds.current_time)           # current simulation time
    # centres rotate with the binary
    c1 = np.array([args.bbh1_x*np.cos(-t*args.binary_omega),
                args.bbh1_x*np.sin(-t*args.binary_omega)])
    c2 = np.array([args.bbh2_x*np.cos(-t*args.binary_omega),
                args.bbh2_x*np.sin(-t*args.binary_omega)])

    R_exc1 = args.bbh1_r * args.excise_factor
    R_exc2 = args.bbh2_r * args.excise_factor

    # distance fields
    dist1 = np.sqrt((X - c1[0])**2 + (Y - c1[1])**2)
    dist2 = np.sqrt((X - c2[0])**2 + (Y - c2[1])**2)

    mask = (dist1 < R_exc1) | (dist2 < R_exc2)
    vol_x[mask] = 0.0
    vol_y[mask] = 0.0


    # Compute torque field: T = x * VY - y * VX
    torque = X * vol_y - Y * vol_x

    # Determine symmetric colormap limits based on this frame's torque extremes
    # Use 80th percentile instead of maximum for better visualization
    # torque_abs = np.abs(torque)
    # max_torque = np.nanpercentile(torque_abs, 95)
    # if max_torque == 0:
    #     max_torque = 1e-10  # avoid zero range
    if auto_torque_max:
        max_torque = np.nanpercentile(np.abs(torque), 95)
        if max_torque == 0:
            max_torque = 1e-10  # avoid zero range
    else:
        max_torque = torque_max

    vmin = -max_torque
    vmax =  max_torque

    

    # Plot the torque field and quiver
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(torque, origin='lower', extent=(-extent_x, extent_x, -extent_y, extent_y),
                   cmap=cmap_name, vmin=vmin, vmax=vmax)
    # Overlay vector field (quiver), skipping points to reduce clutter
    s = arrow_step
    ax.quiver(X[::s, ::s], Y[::s, ::s], vol_x[::s, ::s], vol_y[::s, ::s],
              color='black', pivot='mid', scale_units='xy', angles='xy', alpha = 0.5, scale = 9e-6)


    # dashed excision circles
    for c, R_exc in [(c1, R_exc1), (c2, R_exc2)]:
        ax.add_patch(Circle(c, R_exc, fill=False, ls='--', lw=1.5, color='green'))
    
    # filled BH horizon circles
    for c, R_h in [(c1, args.bbh1_r), (c2, args.bbh2_r)]:
        ax.add_patch(Circle(c, R_h,  fc='black', ec='none'))

    # Axis labels, using M (mass units) and scaling label if large extent
    if extent_x > 100:
        ax.set_xlabel('$x/100M$', fontsize=fontsize)
        ax.set_ylabel('$y/100M$', fontsize=fontsize)
    else:
        ax.set_xlabel('$x/M$', fontsize=fontsize)
        ax.set_ylabel('$y/M$', fontsize=fontsize)
    ax.set_aspect('equal', adjustable='box')  # equal scale for x and y

        
    # leave 12 % of the width for the color-bar
    fig.subplots_adjust(right=0.88)   

    # Colorbar for torque magnitude
    cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label('$\\tau_z$', fontsize=fontsize*1.5, rotation = 0)
    # Tick label font sizes
    ax.tick_params(labelsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)
    # Time annotation
    t_code = float(ds.current_time)
    if labelT:
        ax.text(0.05, 0.95, f"$t = {t_code / Torbit:.2f} \\; T$", transform=ax.transAxes,
                fontsize=fontsize, color='black', verticalalignment='top')
    else:
        ax.text(0.05, 0.95, f"$t = {t_code:.0f} \\; M$", transform=ax.transAxes,
                fontsize=fontsize, color='black', verticalalignment='top')
    # Save the figure as a PNG frame
    frame_name = f"drag_ext{extent_x:.0f}_it{frame_idx:05d}.png"
    frame_path = os.path.join(frames_dir, frame_name)
    fig.savefig(frame_path)
    plt.close(fig)
    allfiles.append(frame_path)

# Create an animated GIF from the saved frames
gif_path = os.path.join(plotdir, simname, f"drag_ext{extent_x:.0f}.gif")
images = [imageio.imread(fn) for fn in allfiles]
imageio.mimsave(gif_path, images)
