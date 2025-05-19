# from mpi4py import MPI
import yt
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

# yt.enable_parallelism()

binary_mass = +2.71811e+00
max_excise_factor = 2.0

parser = argparse.ArgumentParser(description='Plot scalar field rho from simulation data.')
parser.add_argument('--simname', type=str, default="240930_BBH_3D_zboost_v5", help='Name of the simulation directory')
parser.add_argument('--skipevery', type=int, default=1, help='Skip every n iterations')
parser.add_argument('--maxframes', type=int, default=200, help='max num of frames (to avoid OOM)')
parser.add_argument('--outR', type=float, default=650, help='outer radius of the integration sphere')
parser.add_argument('--innerR', type=float, default=-1, help='inner radius of the integration sphere')
parser.add_argument('--bbh1_x', type=float, default=-(70.49764373525885/2-2.926860031395978)/binary_mass, help='x coordinate of the first black hole') 
parser.add_argument('--bbh2_x', type=float, default= (70.49764373525885/2+2.926860031395978)/binary_mass, help='x coordinate of the second black hole')
parser.add_argument('--bbh1_r', type=float, default=3.98070/binary_mass, help='radius of the first black hole')
parser.add_argument('--bbh2_r', type=float, default=3.98070/binary_mass, help='radius of the second black hole')
parser.add_argument('--binary_omega', type=float, default=- 0.002657634562418009 * 2.71811, help='orbital frequency of the binary')
parser.add_argument('--excise_factor', type=float, default=1.5, help='factor to excise the black hole')
parser.add_argument('--outplot', action='store_true', help='Output the plot')
parser.add_argument('--plotsum', action='store_true', help='plot the sum instead of a slice')

args = parser.parse_args()
simname = args.simname
skipevery = args.skipevery
maxframes = args.maxframes
outR = args.outR
innerR = args.innerR
bbh1_x = args.bbh1_x
bbh2_x = args.bbh2_x
bbh1_r = args.bbh1_r
bbh2_r = args.bbh2_r
binary_omega = args.binary_omega
excise_factor = args.excise_factor
outplot = args.outplot
plotsum = args.plotsum

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

results = []

for frameidx, plt_dir in enumerate(plt_dirs):
    ds = yt.load(os.path.join(rundir, plt_dir))
    
    level_left_edges = np.array([ds.index.grid_left_edge[np.where(ds.index.grid_levels==[i])[0]].min(axis=0)  for i in range(ds.max_level+1)])
    level_right_edges = np.array([ds.index.grid_right_edge[np.where(ds.index.grid_levels==[i])[0]].max(axis=0)  for i in range(ds.max_level+1)])
    level_dxs = ds.index.level_dds
    level_dims = (level_right_edges - level_left_edges + 1e-10) / level_dxs
    level_dims = level_dims.astype(int)
    sph = ds.sphere(ds.domain_center, (outR, "code_length"))
    
    field_ds_levels = {}
    sur_x, sur_y, sur_z = 0.0, 0.0, 0.0
    sur_bh1_x, sur_bh1_y, sur_bh1_z = 0.0, 0.0, 0.0
    sur_bh2_x, sur_bh2_y, sur_bh2_z = 0.0, 0.0, 0.0

    # a simple routine to find the level that contains the outer boundary, please make sure the circle does not cross the level boundary
    out_boundary_level = -1
    for curlevel in range(ds.max_level):
        if level_right_edges[curlevel][0] > outR and level_right_edges[curlevel+1][0] < outR:
            out_boundary_level = curlevel
            break
    # print("out_boundary_level", out_boundary_level)
    if out_boundary_level == -1:
        print("outer boundary level not found for dataset", rundir, "/", plt_dir)
        exit()

    # integrate the outer surface
    for curlevel in [out_boundary_level]: # try only the outer surface
        intdomain = ds.box(level_left_edges[curlevel], level_right_edges[curlevel])
        if outplot:
            intdomain = ds.sphere(ds.domain_center, (outR+30, "code_length"))
        if curlevel < ds.max_level:
            intdomain = intdomain - ds.box(level_left_edges[curlevel+1], level_right_edges[curlevel+1])
        
        field_ds_levels[curlevel] = ds.covering_grid(level=curlevel, left_edge=level_left_edges[curlevel], dims=level_dims[curlevel], data_source=intdomain)
        
        for field in ['SURFACE_X', 'SURFACE_Y', 'SURFACE_Z']:
            data = field_ds_levels[curlevel][field][:]
            sum_z = data.sum(axis=2)
            extent = [level_left_edges[curlevel][0], level_right_edges[curlevel][0], 
                        level_left_edges[curlevel][1], level_right_edges[curlevel][1]]
            if outplot:
                plt.figure()
                # Create a custom RdBu colormap with white at center (0)
                cmap = plt.cm.RdBu.copy()
                # Get the absolute maximum value for symmetric color scaling
                vmax = np.abs(sum_z).max()
                # Define the extent using the level boundaries for correct axis scaling
                extent = [level_left_edges[curlevel][0], level_right_edges[curlevel][0], 
                        level_left_edges[curlevel][1], level_right_edges[curlevel][1]]
                plt.imshow(sum_z.T, origin='lower', aspect='auto', cmap=cmap, 
                        vmin=-vmax, vmax=vmax, extent=extent)
                plt.colorbar(label=field)
                plt.xlabel('x')
                plt.ylabel('y')
                plt.title(f'{field} at time {int(ds.current_time)} - level {curlevel}')
                plt.savefig(f"{plotdir}/tmp/{field}_{frameidx}_{curlevel}.png")

            # interpolate and do a line integral
            n_theta = 1000
            
            x_min, x_max, y_min, y_max = extent
            Nx, Ny = sum_z.shape         # array dimensions

            dx = (x_max - x_min) / Nx    # grid spacing in x
            dy = (y_max - y_min) / Ny    # grid spacing in y

            xc = 0.5 * (x_min + x_max)
            yc = 0.5 * (y_min + y_max)
            

            theta = np.linspace(0.0, 2*np.pi, n_theta, endpoint=False)

            # ---------- physical → array indices ----------
            #   x_i = x_min + (i + 0.5)*dx  ->  i = floor((x - x_min)/dx)
            x_phys = xc + outR * np.cos(theta)
            y_phys = yc + outR * np.sin(theta)

            i_idx = np.floor((x_phys - x_min) / dx).astype(int)
            j_idx = np.floor((y_phys - y_min) / dy).astype(int)

            # clamp indices to valid range (needed if the circle grazes the boundary)
            i_idx = np.clip(i_idx, 0, Nx-1)
            j_idx = np.clip(j_idx, 0, Ny-1)

            # ---------- pick values & integrate ----------
            # NB: NumPy arrays are row‑major: first index = y (j), second = x (i)
            vals = sum_z[j_idx, i_idx]

            dl = (2.0 * np.pi * outR) / n_theta        # arc length of each segment
            integral = np.sum(vals) * dl
            
            if outplot:
                print(f"vals: {vals[0:10]}\n theta: {theta[0:10]}\n i_idx: {i_idx[0:10]}\n j_idx: {j_idx[0:10]}\n dl: {dl}\n")
                print(f"integral of {field} at frame {frameidx} level {curlevel}: {integral}")

            if field == 'SURFACE_X':
                sur_x += integral
            elif field == 'SURFACE_Y':
                sur_y += integral
            elif field == 'SURFACE_Z':
                sur_z += integral

    # integrate on the two black holes
    levels_to_use = [ds.max_level-1, ds.max_level]                     # BH excision usually sits in these
    sumz_lv_field = {}                         # (lev, field)  -> 2-D array
    extent_lv     = {}                         # lev          -> [xmin, xmax, ymin, ymax]

    for lev in levels_to_use:
        # build a covering grid on that level only (no subtraction)
        cg = ds.covering_grid(level=lev,
                            left_edge=level_left_edges[lev],
                            dims=level_dims[lev])

        extent_lv[lev] = [level_left_edges[lev][0], level_right_edges[lev][0],
                        level_left_edges[lev][1], level_right_edges[lev][1]]

        for fld in ['SURFACE_X', 'SURFACE_Y', 'SURFACE_Z']:
            # collapse the thin-z direction (sum over z index)
            sumz_lv_field[(lev, fld)] = cg[fld][:].sum(axis=2)

    def sample_surface_value(x, y, fld, sumz_lv_field, extent_lv):
        """
        Return value of `fld` at clipped idx close to physical (x,y),
        choosing level-7 data if the point lies inside the level-7 patch,
        otherwise falling back to level-6.
        """
        # choose level by bounding-box test
        if ( extent_lv[7][0] <= x < extent_lv[7][1] and
            extent_lv[7][2] <= y < extent_lv[7][3] ):
            lev = 7
        else:
            lev = 6

        sum_z  = sumz_lv_field[(lev, fld)]
        xmin, xmax, ymin, ymax = extent_lv[lev]
        nx, ny = sum_z.shape
        dx = (xmax - xmin) / nx
        dy = (ymax - ymin) / ny

        i = int(np.clip(np.floor((x - xmin) / dx), 0, nx-1))
        j = int(np.clip(np.floor((y - ymin) / dy), 0, ny-1))
        return sum_z[j, i]


    # centres (rotate with binary)
    t      = float(ds.current_time)
    c1 = np.array([bbh1_x*np.cos(-t*binary_omega),
                bbh1_x*np.sin(-t*binary_omega)])
    c2 = np.array([bbh2_x*np.cos(-t*binary_omega),
                bbh2_x*np.sin(-t*binary_omega)])
    R_exc1 = bbh1_r * excise_factor      # same for both BHs in this run
    R_exc2 = bbh2_r * excise_factor      # same for both BHs in this run
    n_th  = 100                         # samples around each BH

    theta = np.linspace(0.0, 2*np.pi, n_th, endpoint=False)
    dl_bh1 = 2.0*np.pi*R_exc1 / n_th      # arc-length per segment
    dl_bh2 = 2.0*np.pi*R_exc2 / n_th      # arc-length per segment

    for fld, acc in [('SURFACE_X', (sur_bh1_x, sur_bh2_x)),
                    ('SURFACE_Y', (sur_bh1_y, sur_bh2_y)),
                    ('SURFACE_Z', (sur_bh1_z, sur_bh2_z))]:

        vals_bh1 = np.empty(n_th)
        vals_bh2 = np.empty(n_th)

        for k, ang in enumerate(theta):
            # sample point on each circle
            x1 = c1[0] + R_exc1*np.cos(ang)
            y1 = c1[1] + R_exc1*np.sin(ang)
            x2 = c2[0] + R_exc2*np.cos(ang)
            y2 = c2[1] + R_exc2*np.sin(ang)

            vals_bh1[k] = sample_surface_value(x1, y1, fld,
                                            sumz_lv_field, extent_lv)
            vals_bh2[k] = sample_surface_value(x2, y2, fld,
                                            sumz_lv_field, extent_lv)

        integral_bh1 = np.sum(vals_bh1) * dl_bh1
        integral_bh2 = np.sum(vals_bh2) * dl_bh2

        # accumulate to the correct running totals ---------------
        if fld == 'SURFACE_X':
            sur_bh1_x, sur_bh2_x = integral_bh1, integral_bh2
        elif fld == 'SURFACE_Y':
            sur_bh1_y, sur_bh2_y = integral_bh1, integral_bh2
        elif fld == 'SURFACE_Z':
            sur_bh1_z, sur_bh2_z = integral_bh1, integral_bh2

        print(f"integral of {fld} at frame {frameidx}: {integral_bh1} (BH1), {integral_bh2} (BH2)")
    results.append([ds.current_time,
                sur_x, sur_y, sur_z,          # outer surface
                sur_bh1_x, sur_bh1_y, sur_bh1_z, # BH 1 
                sur_bh2_x, sur_bh2_y, sur_bh2_z]) # BH 2

results = np.array(results)
np.save(f"{simname}_2d_integrals_surface.npy", results)

plt.figure()
plt.plot(results[:,0], results[:,1], label='SURFACE_X')
plt.plot(results[:,0], results[:,2], label='SURFACE_Y')
plt.plot(results[:,0], results[:,3], label='SURFACE_Z')
plt.xlabel('Time')
plt.ylabel('2D Integral')
plt.legend()
plt.savefig(f"{simname}_2d_surface_integrals.png")
plt.close()

plt.figure()
plt.plot(results[:,0], results[:,4], label='SURFACE_X BH1')
plt.plot(results[:,0], results[:,5], label='SURFACE_Y BH1')
plt.plot(results[:,0], results[:,6], label='SURFACE_Z BH1')
plt.plot(results[:,0], results[:,7], label='SURFACE_X BH2')
plt.plot(results[:,0], results[:,8], label='SURFACE_Y BH2')
plt.plot(results[:,0], results[:,9], label='SURFACE_Z BH2')
plt.xlabel('Time')
plt.ylabel('2D Integral')
plt.legend()
plt.savefig(f"{simname}_2d_surface_integrals_bh.png")
plt.close()