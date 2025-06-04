import yt
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse


binary_mass = +2.71811e+00
max_excise_factor = 2.0

parser = argparse.ArgumentParser(description='Plot scalar field rho from simulation data.')
parser.add_argument('--simname', type=str, default="240930_BBH_3D_zboost_v5", help='Name of the simulation directory')
parser.add_argument('--skipevery', type=int, default=1, help='Skip every n iterations')
parser.add_argument('--maxframes', type=int, default=400, help='max num of frames (to avoid OOM)')
parser.add_argument('--outR', type=float, default=650, help='outer radius of the integration sphere')
parser.add_argument('--innerR', type=float, default=-1, help='inner radius of the integration sphere')
parser.add_argument('--bbh1_x', type=float, default=-(70.49764373525885/2-2.926860031395978)/binary_mass, help='x coordinate of the first black hole') 
parser.add_argument('--bbh2_x', type=float, default= (70.49764373525885/2+2.926860031395978)/binary_mass, help='x coordinate of the second black hole')
parser.add_argument('--bbh1_r', type=float, default=3.98070/binary_mass, help='radius of the first black hole')
parser.add_argument('--bbh2_r', type=float, default=3.98070/binary_mass, help='radius of the second black hole')
parser.add_argument('--binary_omega', type=float, default=- 0.002657634562418009 * 2.71811, help='orbital frequency of the binary')
parser.add_argument('--excise_factor', type=float, default=1.5, help='factor to excise the black hole')
parser.add_argument('--integratefield', type=str, default='VOLUME_X', help='Field to integrate')
parser.add_argument('--allfields', action='store_true', help='integrate all fields')
parser.add_argument('--surface', action='store_true', help='integrate the surface quantities')
parser.add_argument('--outplot', action='store_true', help='Output the plot')
parser.add_argument('--plotsum', action='store_true', help='plot the sum instead of a slice')
parser.add_argument('--fix_metric_error', action='store_true', help='Fix the metric error in VOLUME fields (apply 1/sqrt(gamma) correction)')
parser.add_argument('--psipow', type=float, default=2, help='power of psi factor')

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
integratefield = args.integratefield
allfields = args.allfields
surface = args.surface
outplot = args.outplot
plotsum = args.plotsum
fix_metric_error = args.fix_metric_error
psipow = args.psipow

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
# define the mesh and compute them only one for all levels, to save time and memory
all_level_xmeshs = {}
all_level_ymeshs = {}

for frameidx, plt_dir in enumerate(plt_dirs):
    ds = yt.load(os.path.join(rundir, plt_dir))
    
    level_left_edges = np.array([ds.index.grid_left_edge[np.where(ds.index.grid_levels==[i])[0]].min(axis=0)  for i in range(ds.max_level+1)])
    level_right_edges = np.array([ds.index.grid_right_edge[np.where(ds.index.grid_levels==[i])[0]].max(axis=0)  for i in range(ds.max_level+1)])
    level_dxs = ds.index.level_dds
    level_dims = (level_right_edges - level_left_edges + 1e-10) / level_dxs
    level_dims = level_dims.astype(int)
    sph = ds.sphere(ds.domain_center, (outR, "code_length"))
    
    field_ds_levels = {}
    vol_x, vol_y, vol_z = 0.0, 0.0, 0.0
    mom_x, mom_y, mom_z = 0.0, 0.0, 0.0
    torque = 0.0 # torque = x*Fy - y*Fx
    L_z = 0.0 # angular momentum

    for curlevel in range(ds.max_level+1):
        outdomain = ds.box(level_left_edges[curlevel], level_right_edges[curlevel])
        if curlevel < ds.max_level:
            # check if the next level is outside outR (then the current level is not needed)
            if level_right_edges[curlevel+1][0] > outR:
                print("skipping level", curlevel)
                continue
            # check if current level is the level of outer boundary
            if level_right_edges[curlevel][0] > outR:
                outdomain = sph

        # if curlevel <= 2:
        #     outdomain = sph
        # else:
        #     outdomain = ds.box(level_left_edges[curlevel], level_right_edges[curlevel])
        
        if curlevel < ds.max_level:
            # intdomain = sph - ds.box(level_left_edges[curlevel+1], level_right_edges[curlevel+1])
            intdomain = outdomain - ds.box(level_left_edges[curlevel+1], level_right_edges[curlevel+1])
        else:
            if innerR > 0:
                innerdomain = ds.sphere(ds.domain_center, (innerR, "code_length"))
                # intdomain = sph - innerdomain
                intdomain = outdomain - innerdomain
            else:
                bh1center = ds.arr([bbh1_x*np.cos(-ds.current_time * binary_omega), bbh1_x * np.sin(-ds.current_time * binary_omega), 0], "code_length")
                bh1 = ds.sphere(bh1center, (bbh1_r*excise_factor, "code_length"))
                bh2center = ds.arr([bbh2_x*np.cos(-ds.current_time * binary_omega), bbh2_x * np.sin(-ds.current_time * binary_omega), 0], "code_length")
                bh2 = ds.sphere(bh2center, (bbh2_r*excise_factor, "code_length"))
                # intdomain = sph - bh1 - bh2
                intdomain = outdomain - bh1 - bh2
        
        field_ds_levels[curlevel] = ds.covering_grid(level=curlevel, left_edge=level_left_edges[curlevel], dims=level_dims[curlevel], data_source=intdomain)
        
        # Compute sqrt(gamma) for this frame and level since metric is evolved
        myW = field_ds_levels[curlevel]['W'][:]
        myW = myW.sum(axis=2)
        zero_mask = (myW == 0)
        psi = 1.0 / np.sqrt(myW)
        psi[zero_mask] = 1.0
        sqrt_gamma = np.power(psi, 6)
        
        for field in ['VOLUME_X', 'VOLUME_Y', 'VOLUME_Z', 'SMOMENTUM_X', 'SMOMENTUM_Y', 'SMOMENTUM_Z']:
            data = field_ds_levels[curlevel][field][:]
            # print(data.shape)
            sum_z = data.sum(axis=2)  # Extract 2D data by summing over z
            
            # Apply metric corrections based on field type and fix_metric_error flag
            if field.startswith('VOLUME'):
                if fix_metric_error:
                    # VOLUME fields have psi12 in the previous version (before 20250522) but should have psi6, so to correct it, divide by sqrt(gamma) = psi6
                    sum_z_with_metric = sum_z / sqrt_gamma
                else:
                    # VOLUME fields already include psi12 (incorrect but as computed)
                    sum_z_with_metric = sum_z
            else:  # SMOMENTUM fields
                # SMOMENTUM fields need sqrt(gamma) factor for proper volume integration
                sum_z_with_metric = sum_z * sqrt_gamma
            
            if len(results) == 0:# only compute the mesh once to save time and memory
                all_level_xmeshs[curlevel] = field_ds_levels[curlevel]['x'][:].sum(axis=2)
                # print(all_level_xmeshs[curlevel].shape)
                all_level_ymeshs[curlevel] = field_ds_levels[curlevel]['y'][:].sum(axis=2)
                # print(all_level_ymeshs[curlevel].shape)
                if outplot:
                    plt.figure()
                    cmap = plt.cm.RdBu.copy()
                    vmax = np.abs(all_level_xmeshs[curlevel]).max()
                    plt.imshow(all_level_xmeshs[curlevel].T, origin='lower', aspect='auto', cmap=cmap,
                        vmin=-vmax, vmax=vmax, extent=[level_left_edges[curlevel][0], level_right_edges[curlevel][0], 
                        level_left_edges[curlevel][1], level_right_edges[curlevel][1]])
                    plt.colorbar(label='xmesh')
                    plt.xlabel('x')
                    plt.ylabel('y')
                    plt.title(f'xmesh at time {int(ds.current_time)} - level {curlevel}')
                    plt.savefig(f"{plotdir}/tmp/xmesh_{frameidx}_{curlevel}.png")
                    plt.close()
                    plt.figure()
                    plt.imshow(all_level_ymeshs[curlevel].T, origin='lower', aspect='auto', cmap=cmap,
                        vmin=-vmax, vmax=vmax, extent=[level_left_edges[curlevel][0], level_right_edges[curlevel][0], 
                        level_left_edges[curlevel][1], level_right_edges[curlevel][1]])
                    plt.colorbar(label='ymesh')
                    plt.xlabel('x')
                    plt.ylabel('y')
                    plt.title(f'ymesh at time {int(ds.current_time)} - level {curlevel}')
                    plt.savefig(f"{plotdir}/tmp/ymesh_{frameidx}_{curlevel}.png")
                    plt.close()
            if outplot:
                plt.figure()
                # Create a custom RdBu colormap with white at center (0)
                cmap = plt.cm.RdBu.copy()
                # Get the absolute maximum value for symmetric color scaling
                vmax = np.abs(sum_z_with_metric).max()
                # Define the extent using the level boundaries for correct axis scaling
                extent = [level_left_edges[curlevel][0], level_right_edges[curlevel][0], 
                        level_left_edges[curlevel][1], level_right_edges[curlevel][1]]
                plt.imshow(sum_z_with_metric.T, origin='lower', aspect='auto', cmap=cmap, 
                        vmin=-vmax, vmax=vmax, extent=extent)
                plt.colorbar(label=field)
                plt.xlabel('x')
                plt.ylabel('y')
                plt.title(f'{field} at time {int(ds.current_time)} - level {curlevel}')
                plt.savefig(f"{plotdir}/tmp/{field}_{frameidx}_{curlevel}.png")
            dx, dy = level_dxs[curlevel][0], level_dxs[curlevel][1]
            integral = sum_z_with_metric.sum() * dx * dy
            if field == 'VOLUME_X':
                vol_x += integral
                torque += - (all_level_ymeshs[curlevel] * sum_z_with_metric * np.power(psi, psipow)).sum() * dx * dy
            elif field == 'VOLUME_Y':
                vol_y += integral
                torque += (all_level_xmeshs[curlevel] * sum_z_with_metric * np.power(psi, psipow)).sum() * dx * dy
            elif field == 'VOLUME_Z':
                vol_z += integral
            elif field == 'SMOMENTUM_X':
                mom_x += integral
                L_z += - (all_level_ymeshs[curlevel] * sum_z_with_metric * np.power(psi, psipow)).sum() * dx * dy
            elif field == 'SMOMENTUM_Y':
                mom_y += integral
                L_z += (all_level_xmeshs[curlevel] * sum_z_with_metric * np.power(psi, psipow)).sum() * dx * dy
            elif field == 'SMOMENTUM_Z':
                mom_z += integral

    results.append([ds.current_time, vol_x, vol_y, vol_z, mom_x, mom_y, mom_z, torque, L_z])

results = np.array(results)
np.save(f"{simname}_2d_integrals_outR{outR}_excise{excise_factor}_psipow{psipow}.npy", results)

plt.figure()
plt.plot(results[:,0], results[:,1], label='VOLUME_X')
plt.plot(results[:,0], results[:,2], label='VOLUME_Y')
plt.plot(results[:,0], results[:,3], label='VOLUME_Z')
plt.plot(results[:,0], results[:,7], label='TORQUE')
plt.xlabel('Time')
plt.ylabel('2D Integral')
plt.legend()
plt.savefig(f"{simname}_2d_volume_integrals_outR{outR}_excise{excise_factor}_psipow{psipow}.png")
plt.close()

plt.figure()
plt.plot(results[:,0], results[:,4], label='SMOMENTUM_X')
plt.plot(results[:,0], results[:,5], label='SMOMENTUM_Y')
plt.plot(results[:,0], results[:,6], label='SMOMENTUM_Z')
plt.plot(results[:,0], results[:,8], label='ANGULAR MOMENTUM')
plt.xlabel('Time')
plt.ylabel('2D Integral')
plt.legend()
plt.savefig(f"{simname}_2d_momentum_integrals_outR{outR}_excise{excise_factor}_psipow{psipow}.png")
plt.close()