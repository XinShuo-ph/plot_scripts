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
parser.add_argument('--outplot', action='store_true', help='Output the plot')
parser.add_argument('--psipow', type=float, default=-2, help='power of psi factor')
# refer to https://arxiv.org/pdf/2104.13420 Appendix Eq. 43 and 44, we should use \sqrt{\sigma} N_i = \sqrt{\gamma} s_i 
# but in current code (20250603), I normalize N_i to N_i N^i=1, but in reality we should not raise s_i to upper idx
# this leads to an extrafactor of \psi^2, to correct this, we multiply the surface integral by \psi^-2
# (However, we seem to still miss an overall \sqrt{\gamma} factor...   )
parser.add_argument('--psipow_surface_correction', type=float, default=-2, help='power of psi factor for the surface correction ')
parser.add_argument('--withQ', action='store_true', help='process energy flux and noether charge Q flux')
# Add worker arguments for parallelization
parser.add_argument('--worker_id', type=int, default=0, help='Worker ID for parallel processing')
parser.add_argument('--total_workers', type=int, default=1, help='Total number of workers for parallel processing')
parser.add_argument('--temp_output', action='store_true', help='Save intermediate results to temporary files')

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
psipow = args.psipow
worker_id = args.worker_id
total_workers = args.total_workers
temp_output = args.temp_output
psipow_surface_correction = args.psipow_surface_correction
withQ = args.withQ

basedir = "/pscratch/sd/x/xinshuo/runGReX/"
plotdir = "/pscratch/sd/x/xinshuo/plotGReX/"
rundir = basedir + simname +"/"

# make /pscratch/sd/x/xinshuo/plotGReX/tmp if it doesn't exist
if not os.path.exists(plotdir + "tmp"):
    os.makedirs(plotdir + "tmp")

# make the worker tmp directories if they don't exist
if not os.path.exists(plotdir + f"tmp/worker_{worker_id}"):
    os.makedirs(plotdir + f"tmp/worker_{worker_id}")

# Create a tmp directory for worker output files
if temp_output and not os.path.exists(plotdir + f"tmp/worker_{worker_id}"):
    os.makedirs(plotdir + f"tmp/worker_{worker_id}")

plt_dirs = sorted([d for d in os.listdir(rundir) if d.startswith('plt') and d[3:].isdigit()], key=lambda x: int(x[3:]))
plt_dirs = plt_dirs[::skipevery]

if len(plt_dirs) > maxframes:
    plt_dirs = plt_dirs[:maxframes]

# Divide work among workers
total_frames = len(plt_dirs)
frames_per_worker = total_frames // total_workers
remainder = total_frames % total_workers

# Calculate start and end indices for this worker
start_idx = worker_id * frames_per_worker + min(worker_id, remainder)
if worker_id < remainder:
    end_idx = start_idx + frames_per_worker + 1
else:
    end_idx = start_idx + frames_per_worker

# Get the subset of plt_dirs for this worker
plt_dirs = plt_dirs[start_idx:end_idx]

print(f"Worker {worker_id}/{total_workers} processing {len(plt_dirs)} frames from index {start_idx} to {end_idx-1}")

results = []

for frameidx, plt_dir in enumerate(plt_dirs):
    global_frameidx = start_idx + frameidx
    print(f"Worker {worker_id}: Processing frame {global_frameidx} ({plt_dir})")
    
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
    rhoavg = 0.0
    sur_torque = 0.0
    sur_bh1_torque = 0.0
    sur_bh2_torque = 0.0
    # Add variables for energy flux and Q charge flux
    energy_flux = 0.0
    qcharge_flux = 0.0
    bh1_energy_flux = 0.0
    bh2_energy_flux = 0.0
    bh1_qcharge_flux = 0.0
    bh2_qcharge_flux = 0.0

    # a simple routine to find the level that contains the outer boundary, please make sure the circle does not cross the level boundary
    out_boundary_level = -1
    for curlevel in range(ds.max_level):
        if level_right_edges[curlevel][0] > outR and level_right_edges[curlevel+1][0] < outR:
            out_boundary_level = curlevel
            break
    # print("out_boundary_level", out_boundary_level)
    if out_boundary_level == -1:
        print(f"Worker {worker_id}: outer boundary level not found for dataset {rundir}/{plt_dir}")
        continue  # Skip this frame instead of exiting

    # integrate the outer surface
    for curlevel in [out_boundary_level]: # try only the outer surface
        intdomain = ds.box(level_left_edges[curlevel], level_right_edges[curlevel])
        if outplot:
            intdomain = ds.sphere(ds.domain_center, (outR+30, "code_length"))
        if curlevel < ds.max_level:
            # subtract only if the next level is completely inside the outR
            if level_right_edges[curlevel+1][0]*np.sqrt(2) < outR: # assuming the domain is square
                print("subtract next level", curlevel+1)
                intdomain = intdomain - ds.box(level_left_edges[curlevel+1], level_right_edges[curlevel+1])
            else:
                print("the circle crosses the next level", curlevel+1)
        
        field_ds_levels[curlevel] = ds.covering_grid(level=curlevel, left_edge=level_left_edges[curlevel], dims=level_dims[curlevel], data_source=intdomain)
        
        fields_to_integrate = ['SURFACE_X', 'SURFACE_Y', 'SURFACE_Z', 'RHO_ENERGY']
        if withQ:
            fields_to_integrate.extend(['ENERGY_FLUX', 'QCHARGE_FLUX'])
            
        for field in fields_to_integrate:
            data = field_ds_levels[curlevel][field][:]
            sum_z = data.sum(axis=2)
            # 20250603 correction: multiply by psi^(-2) = W for the surface integral, (for generality, psi^(psipow_surface_correction) = W^(-psipow_surface_correction/2) )
            myWdata = field_ds_levels[curlevel]['W'][:]
            myWdata = myWdata.sum(axis=2)
            if field == 'SURFACE_X' or field == 'SURFACE_Y' or field == 'SURFACE_Z' or field == 'ENERGY_FLUX' or field == 'QCHARGE_FLUX':
                sum_z = sum_z * np.power(myWdata, -psipow_surface_correction/2)
                # Define the extent using the level boundaries for correct axis scaling
            extent = [level_left_edges[curlevel][0], level_right_edges[curlevel][0], 
                        level_left_edges[curlevel][1], level_right_edges[curlevel][1]]

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
            # make sure the order of i_idx and j_idx correct (by checking the plots!!)
            vals = sum_z[i_idx, j_idx]

            dl = (2.0 * np.pi * outR) / n_theta        # arc length of each segment
            integral = np.sum(vals) * dl

            
            if outplot:
                # Create figure with three subplots (3 rows, 1 column)
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), 
                                              gridspec_kw={'height_ratios': [3, 1, 1]})
                
                # Top subplot - 2D image
                # Create a custom RdBu colormap with white at center (0)
                cmap = plt.cm.RdBu.copy()
                # Get the absolute maximum value for symmetric color scaling
                vmax = np.abs(sum_z).max()
                extent = [level_left_edges[curlevel][0], level_right_edges[curlevel][0], 
                        level_left_edges[curlevel][1], level_right_edges[curlevel][1]]
                im = ax1.imshow(sum_z.T, origin='lower', aspect='auto', cmap=cmap, 
                        vmin=-vmax, vmax=vmax, extent=extent)
                plt.colorbar(im, ax=ax1, label=field)
                
                # Add scatter points for the sampling locations
                # Create alpha values that increase linearly with theta (0->1)
                ax1.scatter(x_phys, y_phys, s=1, color='black', alpha=theta / (2*np.pi)/2)
                # Draw the circle
                # circle = plt.Circle((xc, yc), outR, fill=False, color='black', linestyle='--')
                # ax1.add_patch(circle)
                
                ax1.set_xlabel('x')
                ax1.set_ylabel('y')
                ax1.set_title(f'{field} at time {int(ds.current_time)} - level {curlevel}')
                
                # Middle subplot - values vs theta
                ax2.plot(theta, vals)
                ax2.set_xlabel('θ (radians)')
                ax2.set_ylabel(f'{field} values')
                ax2.set_xlim(0, 2*np.pi)
                ax2.grid(True)
                
                # Bottom subplot - torque contribution vs theta
                if field == 'SURFACE_X':
                    torque_vals = -(y_phys * vals)
                    torque_label = '-y·Fx (torque contribution)'
                elif field == 'SURFACE_Y':
                    torque_vals = x_phys * vals
                    torque_label = 'x·Fy (torque contribution)'
                else:
                    torque_vals = np.zeros_like(vals)
                    torque_label = 'No torque contribution'
                
                ax3.plot(theta, torque_vals)
                ax3.set_xlabel('θ (radians)')
                ax3.set_ylabel(torque_label)
                ax3.set_xlim(0, 2*np.pi)
                ax3.grid(True)
                
                plt.tight_layout()
                plt.savefig(f"{plotdir}/tmp/worker_{worker_id}/{field}_{global_frameidx}_{curlevel}.png")
                plt.close(fig)
            if outplot:
                print(f"Worker {worker_id}: vals: {vals[0:10]}\n theta: {theta[0:10]}\n i_idx: {i_idx[0:10]}\n j_idx: {j_idx[0:10]}\n dl: {dl}\n")
                print(f"Worker {worker_id}: integral of {field} at frame {global_frameidx} level {curlevel}: {integral}")

            if field == 'SURFACE_X':
                sur_x += integral
                torque_integral = -(y_phys * vals).sum() * dl
                sur_torque += torque_integral
                if outplot:
                    print(f"Worker {worker_id}: torque contributions from -y Fx: {-y_phys[0:10] * vals[0:10]}")
                    print(f"Worker {worker_id}: total contributions from -y Fx: {torque_integral}")
            elif field == 'SURFACE_Y':
                sur_y += integral
                torque_integral = (x_phys * vals).sum() * dl
                sur_torque += torque_integral
                if outplot:
                    print(f"Worker {worker_id}: torque contributions from x Fy: {x_phys[0:10] * vals[0:10]}")
                    print(f"Worker {worker_id}: total contributions from x Fy: {torque_integral}")
            elif field == 'SURFACE_Z':
                sur_z += integral
            elif field == 'RHO_ENERGY':
                rhoavg += integral/(2.0 * np.pi * outR) # compute averaged rho
            elif field == 'ENERGY_FLUX':
                energy_flux += integral
            elif field == 'QCHARGE_FLUX':
                qcharge_flux += integral

    # integrate on the two black holes
    levels_to_use = [ds.max_level-1, ds.max_level]                     # BH excision usually sits in these
    sumz_lv_field = {}                         # (lev, field)  -> 2-D array
    extent_lv     = {}                         # lev          -> [xmin, xmax, ymin, ymax]

    print(f"Worker {worker_id}: levels_to_use for flux across BHs: {levels_to_use}")
    for lev in levels_to_use:
        # build a covering grid on that level only (no subtraction)
        cg = ds.covering_grid(level=lev,
                            left_edge=level_left_edges[lev],
                            dims=level_dims[lev])
        # Compute sqrt(gamma) for this frame and level since metric is evolved
        myW = cg['W'][:]
        myW = myW.sum(axis=2)
        zero_mask = (myW == 0)
        psi = 1.0 / np.sqrt(myW)
        # psi[zero_mask] = 1.0
        # sqrt_gamma = np.power(psi, 6)

        extent_lv[lev] = [level_left_edges[lev][0], level_right_edges[lev][0],
                        level_left_edges[lev][1], level_right_edges[lev][1]]

        bh_fields = ['SURFACE_X', 'SURFACE_Y', 'SURFACE_Z']
        if withQ:
            bh_fields.extend(['ENERGY_FLUX', 'QCHARGE_FLUX'])
            
        for fld in bh_fields:
            # collapse the thin-z direction (sum over z index)
            # 20250603 correction: multiply by psi^(-2) = W for the surface integral, (for generality, psi^(psipow_surface_correction) = W^(-psipow_surface_correction/2) )
            sumz_lv_field[(lev, fld)] = (cg[fld][:].sum(axis=2)) * np.power(psi, psipow_surface_correction)
        fld = 'psifactor'
        sumz_lv_field[(lev, fld)] = np.power(psi, psipow)

    def sample_surface_value(x, y, fld, sumz_lv_field, extent_lv):
        """
        Return value of `fld` at clipped idx close to physical (x,y),
        choosing level-7 data if the point lies inside the level-7 patch,
        otherwise falling back to level-6.
        """
        # choose level by bounding-box test
        highlevel = levels_to_use[1]
        lowlevel = levels_to_use[0]
        if ( extent_lv[highlevel][0] <= x < extent_lv[highlevel][1] and
            extent_lv[highlevel][2] <= y < extent_lv[highlevel][3] ):
            lev = highlevel
        else:
            lev = lowlevel

        sum_z  = sumz_lv_field[(lev, fld)]
        xmin, xmax, ymin, ymax = extent_lv[lev]
        nx, ny = sum_z.shape
        dx = (xmax - xmin) / nx
        dy = (ymax - ymin) / ny

        i = int(np.clip(np.floor((x - xmin) / dx), 0, nx-1))
        j = int(np.clip(np.floor((y - ymin) / dy), 0, ny-1))
        return sum_z[i,j]


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

    bh_surface_fields = [('SURFACE_X', (sur_bh1_x, sur_bh2_x)),
                        ('SURFACE_Y', (sur_bh1_y, sur_bh2_y)),
                        ('SURFACE_Z', (sur_bh1_z, sur_bh2_z))]
    if withQ:
        bh_surface_fields.extend([
            ('ENERGY_FLUX', (bh1_energy_flux, bh2_energy_flux)),
            ('QCHARGE_FLUX', (bh1_qcharge_flux, bh2_qcharge_flux))
        ])
    
    for fld, acc in bh_surface_fields:
        vals_bh1 = np.empty(n_th)
        vals_bh2 = np.empty(n_th)
        vals_bh1_torque = np.empty(n_th)
        vals_bh2_torque = np.empty(n_th)
        psi_bh1 = np.empty(n_th)
        psi_bh2 = np.empty(n_th)
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
            psi_bh1[k] = sample_surface_value(x1, y1, 'psifactor',
                                            sumz_lv_field, extent_lv)
            psi_bh2[k] = sample_surface_value(x2, y2, 'psifactor',
                                            sumz_lv_field, extent_lv)
            if fld == 'SURFACE_X':
                vals_bh1_torque[k] = - (y1 * vals_bh1[k] * psi_bh1[k]) 
                vals_bh2_torque[k] = - (y2 * vals_bh2[k] * psi_bh2[k]) 
            elif fld == 'SURFACE_Y':
                vals_bh1_torque[k] = (x1 * vals_bh1[k] * psi_bh1[k]) 
                vals_bh2_torque[k] = (x2 * vals_bh2[k] * psi_bh2[k]) 

        integral_bh1 = np.sum(vals_bh1) * dl_bh1
        integral_bh2 = np.sum(vals_bh2) * dl_bh2
        integral_bh1_torque = np.sum(vals_bh1_torque) * dl_bh1
        integral_bh2_torque = np.sum(vals_bh2_torque) * dl_bh2

        # accumulate to the correct running totals ---------------
        if fld == 'SURFACE_X':
            sur_bh1_x, sur_bh2_x = integral_bh1, integral_bh2
            sur_bh1_torque += integral_bh1_torque
            sur_bh2_torque += integral_bh2_torque
        elif fld == 'SURFACE_Y':
            sur_bh1_y, sur_bh2_y = integral_bh1, integral_bh2
            sur_bh1_torque += integral_bh1_torque
            sur_bh2_torque += integral_bh2_torque
        elif fld == 'SURFACE_Z':
            sur_bh1_z, sur_bh2_z = integral_bh1, integral_bh2
        elif fld == 'ENERGY_FLUX':
            bh1_energy_flux, bh2_energy_flux = integral_bh1, integral_bh2
        elif fld == 'QCHARGE_FLUX':
            bh1_qcharge_flux, bh2_qcharge_flux = integral_bh1, integral_bh2

        print(f"Worker {worker_id}: integral of {fld} at frame {global_frameidx}: {integral_bh1} (BH1), {integral_bh2} (BH2)")

    result_data = [ds.current_time,
                sur_x, sur_y, sur_z,          # outer surface
                sur_bh1_x, sur_bh1_y, sur_bh1_z, # BH 1 
                sur_bh2_x, sur_bh2_y, sur_bh2_z, # BH 2
                rhoavg, sur_torque, sur_bh1_torque, sur_bh2_torque]
    
    if withQ:
        result_data.extend([energy_flux, qcharge_flux, bh1_energy_flux, bh2_energy_flux, bh1_qcharge_flux, bh2_qcharge_flux])
    
    results.append(result_data)
    
    # Save intermediate results if requested
    if temp_output:
        file_suffix = f"_psipow{psipow:.1f}_psipow_surface_correction{psipow_surface_correction:.1f}"
        if withQ:
            file_suffix += "_withQ"
        np.save(f"{plotdir}/tmp/worker_{worker_id}/{simname}_2d_integrals_surface_outR{outR}_excise{excise_factor}{file_suffix}_worker{worker_id}.npy", 
                np.array(results))

# Save worker results
file_suffix = f"_psipow{psipow:.1f}_psipow_surface_correction{psipow_surface_correction:.1f}"
# if withQ:
#     file_suffix += "_withQ"
output_filename = f"{simname}_2d_integrals_surface_outR{outR}_excise{excise_factor}{file_suffix}_worker{worker_id}.npy"
np.save(output_filename, np.array(results))
print(f"Worker {worker_id}: Saved results to {output_filename}")

# Only create plots for worker 0 or if requested
if worker_id == 0 or total_workers == 1:
    results_array = np.array(results)
    
    plt.figure()
    plt.plot(results_array[:,0], results_array[:,1], label='SURFACE_X')
    plt.plot(results_array[:,0], results_array[:,2], label='SURFACE_Y')
    plt.plot(results_array[:,0], results_array[:,3], label='SURFACE_Z')
    plt.plot(results_array[:,0], results_array[:,10], label='rhoavg')
    plt.plot(results_array[:,0], results_array[:,11], label='sur_torque')
    if withQ:
        plt.plot(results_array[:,0], results_array[:,14], label='ENERGY_FLUX')
        plt.plot(results_array[:,0], results_array[:,15], label='QCHARGE_FLUX')
    plt.xlabel('Time')
    plt.ylabel('2D Integral')
    plt.legend()
    plt.savefig(f"{simname}_2d_surface_integrals_outR{outR}{file_suffix}_worker{worker_id}.png")
    plt.close()

    plt.figure()
    plt.plot(results_array[:,0], results_array[:,4], label='SURFACE_X BH1')
    plt.plot(results_array[:,0], results_array[:,5], label='SURFACE_Y BH1')
    plt.plot(results_array[:,0], results_array[:,6], label='SURFACE_Z BH1')
    plt.plot(results_array[:,0], results_array[:,7], label='SURFACE_X BH2')
    plt.plot(results_array[:,0], results_array[:,8], label='SURFACE_Y BH2')
    plt.plot(results_array[:,0], results_array[:,9], label='SURFACE_Z BH2')
    plt.plot(results_array[:,0], results_array[:,12], label='sur_bh1_torque')
    plt.plot(results_array[:,0], results_array[:,13], label='sur_bh2_torque')
    if withQ:
        plt.plot(results_array[:,0], results_array[:,16], label='ENERGY_FLUX BH1')
        plt.plot(results_array[:,0], results_array[:,17], label='ENERGY_FLUX BH2')
        plt.plot(results_array[:,0], results_array[:,18], label='QCHARGE_FLUX BH1')
        plt.plot(results_array[:,0], results_array[:,19], label='QCHARGE_FLUX BH2')
    plt.xlabel('Time')
    plt.ylabel('2D Integral')
    plt.legend()
    plt.savefig(f"{simname}_2d_surface_integrals_bh_excise{excise_factor}{file_suffix}_worker{worker_id}.png")
    plt.close()

    if withQ:
        # Create a separate plot just for the energy and Q fluxes
        plt.figure()
        plt.plot(results_array[:,0], results_array[:,14], label='ENERGY_FLUX')
        plt.plot(results_array[:,0], results_array[:,15], label='QCHARGE_FLUX')
        plt.plot(results_array[:,0], results_array[:,16], label='ENERGY_FLUX BH1')
        plt.plot(results_array[:,0], results_array[:,17], label='ENERGY_FLUX BH2')
        plt.plot(results_array[:,0], results_array[:,18], label='QCHARGE_FLUX BH1')
        plt.plot(results_array[:,0], results_array[:,19], label='QCHARGE_FLUX BH2')
        plt.xlabel('Time')
        plt.ylabel('Flux')
        plt.legend()
        plt.savefig(f"{simname}_2d_flux_summary{file_suffix}_worker{worker_id}.png")
        plt.close()

print(f"Worker {worker_id}: Finished processing {len(plt_dirs)} frames.") 