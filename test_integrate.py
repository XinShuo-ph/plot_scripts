# from mpi4py import MPI
import yt
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

# yt.enable_parallelism()

binary_mass = +2.71811e+00
max_excise_factor = 2.0 # define the domain where we use to interpolate the surface fields

# parse arguments
parser = argparse.ArgumentParser(description='Plot scalar field rho from simulation data.')
parser.add_argument('--simname', type=str, default="240930_BBH_3D_zboost_v5", help='Name of the simulation directory')
parser.add_argument('--skipevery', type=int, default=1, help='Skip every n iterations')
parser.add_argument('--maxframes', type=int, default=200, help='max num of frames (to avoid OOM)')
parser.add_argument('--frameidx', type=int, default=100, help='frame index')
parser.add_argument('--outR', type=float, default=650, help='outer radius of the integration sphere')
parser.add_argument('--bbh1_x', type=float, default=-(70.49764373525885/2-2.926860031395978)/binary_mass, help='x coordinate of the first black hole') 
parser.add_argument('--bbh2_x', type=float, default= (70.49764373525885/2+2.926860031395978)/binary_mass, help='x coordinate of the second black hole')
parser.add_argument('--bbh1_r', type=float, default=3.98070/binary_mass, help='radius of the first black hole')
parser.add_argument('--bbh2_r', type=float, default=3.98070/binary_mass, help='radius of the second black hole')
parser.add_argument('--binary_omega', type=float, default=- 0.002657634562418009 * 2.71811, help='orbital frequency of the binary, note a minus sign')
parser.add_argument('--excise_factor', type=float, default=1.5, help='factor to excise the black hole')
parser.add_argument('--integratefield', type=str, default='VOLUME_X', help='Field to integrate')
parser.add_argument('--innerR', type=float, default=-1, help='inner radius of the integration sphere')
# store false, i.e. with --outplot, true, without --outplot, false
parser.add_argument('--allfields', action='store_true', help='integrate all fields')
parser.add_argument('--surface', action='store_true', help='integrate the surface quantities')
parser.add_argument('--outplot', action='store_true', help='Output the plot')
parser.add_argument('--plotsum', action='store_true', help='plot the sum instead of a slice')
parser.add_argument('--plotxz', action='store_true', help='plot the xz plane')

args = parser.parse_args()
simname = args.simname
skipevery = args.skipevery
maxframes = args.maxframes
frameidx = args.frameidx
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
plotxz = args.plotxz



basedir = "/pscratch/sd/x/xinshuo/runGReX/"
plotdir = "/pscratch/sd/x/xinshuo/plotGReX/"
rundir = basedir + simname +"/"
# there are a lot of pltXXXXX dirs, corresponding to different iterations, load all of them
# Find all directories in rundir that match the pattern 'pltXXXXX' or more digits
plt_dirs = sorted([d for d in os.listdir(rundir) if d.startswith('plt') and d[3:].isdigit()], key=lambda x: int(x[3:]))
plt_dirs = plt_dirs[::skipevery] # skip some iteration to save time

if len(plt_dirs) > maxframes:
    plt_dirs = plt_dirs[:maxframes]

# # Load all datasets
# datasets = [yt.load(os.path.join(rundir, d)) for d in plt_dirs]

# ds = datasets[-1]

ds = yt.load(os.path.join(rundir, plt_dirs[args.frameidx]))

# the dataset has 7 refinement levels, 0~6
# extract the field 'VOLUME_X' from the dataset
# and do an volume integral of the field

# the field 'VOLUME_X' is a scalar field, so we can directly integrate it
# the integration is done by summing up the field values in all cells
# and multiply by the cell volume
# note to take care of different refinement levels

level_left_edges = np.array([ds.index.grid_left_edge[np.where(ds.index.grid_levels==[i])[0]].min(axis=0)  for i in range(ds.max_level+1)])
level_right_edges = np.array([ds.index.grid_right_edge[np.where(ds.index.grid_levels==[i])[0]].max(axis=0)  for i in range(ds.max_level+1)])
level_dxs = ds.index.level_dds
level_dims = (level_right_edges - level_left_edges + 1e-10) / level_dxs
level_dims = level_dims.astype(int)
sph = ds.sphere(ds.domain_center,(outR,"code_length"))

if outplot:
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

field_3d_levels={}
field_ds_levels={}
allfields_3d_levels={}

for curlevel in [1,2,3,4,5,6,7]:
    if frameidx%64 == 0:
        print(curlevel)
    if curlevel <= 2:
        outdomain = sph
    else:
        outdomain = ds.box(level_left_edges[curlevel], level_right_edges[curlevel])
        # plot the outdomain box on the 2d figure
        if outplot:
            extent = [level_left_edges[curlevel][0],level_right_edges[curlevel][0],level_left_edges[curlevel][1],level_right_edges[curlevel][1]]
            ax1.add_patch(plt.Rectangle((extent[0],extent[2]),extent[1]-extent[0],extent[3]-extent[2],fill=None,edgecolor='r',linewidth=0.005,alpha=0.5))
    # subtract the higher level boxes from sph
    if curlevel < ds.max_level:
        intdomain = sph - ds.box(level_left_edges[curlevel+1], level_right_edges[curlevel+1])
    else:
        if innerR > 0:
            innerdomain = ds.sphere(ds.domain_center,(innerR,"code_length"))
            intdomain = sph - innerdomain
        else:
            bh1center = ds.arr([bbh1_x*np.cos( - ds.current_time * binary_omega ), bbh1_x * np.sin( - ds.current_time * binary_omega ) ,0],"code_length")
            bh1 = ds.sphere(bh1center,(bbh1_r*excise_factor,"code_length"))
            bh2center = ds.arr([bbh2_x*np.cos( - ds.current_time * binary_omega ), bbh2_x * np.sin( - ds.current_time * binary_omega ) ,0],"code_length")
            bh2 = ds.sphere(bh2center,(bbh2_r*excise_factor,"code_length"))
            intdomain = sph - bh1 - bh2
    if frameidx%64 == 0:
        print("covering grid")
    field_ds_levels[curlevel] = ds.covering_grid(level=curlevel, left_edge=level_left_edges[curlevel], dims=level_dims[curlevel],data_source=intdomain)
    if frameidx%64 == 0:
        print("getting grided field")
    field_3d_levels[curlevel] = field_ds_levels[curlevel][integratefield]
    if allfields:
        for field in ['SMOMENTUM_X','SMOMENTUM_Y','SMOMENTUM_Z','VOLUME_X','VOLUME_Y','VOLUME_Z']:
            if frameidx%64 == 0:
                print(field)
            allfields_3d_levels[(field,curlevel)] = field_ds_levels[curlevel][field]
    if surface and curlevel == 2:
        surface_results = np.zeros(4)
        surface_results[0] = float(ds.current_time)
        # integrate the fields (which are precomputed surface integrands) over sphere of radius outR
        domain1 = ds.box(level_left_edges[curlevel], level_right_edges[curlevel])
        domain2 = ds.box(level_left_edges[curlevel+1], level_right_edges[curlevel+1])
        mydomain = domain1 - domain2
        surface = ds.surface(mydomain, 'radius', (outR, 'code_length'))
        for idx,field in enumerate(['SURFACE_X','SURFACE_Y','SURFACE_Z']):
            if frameidx%64 == 0:
                print(field)
            surface_fields_2d = surface[field]
            tris = surface.triangles # the triangles of the surface, of shape (ntriangles, 3, 3)
            # compute the area of each triangle
            # the area of a triangle is 0.5 * |a x b|
            # where a and b are two edges of the triangle
            # and |a x b| is the magnitude of the cross product
            # the area of the triangle is the magnitude of the cross product of two edges
            tris_area = 0.5 * np.linalg.norm(np.cross(tris[:,1] - tris[:,0], tris[:,2] - tris[:,0]), axis=1)
            integral = np.sum(surface_fields_2d* tris_area)
            surface_results[idx+1] = float(integral)
            print("The integral of the field %s is %f"%(field,integral))
            print("area error: ", (float(tris_area.sum()) - 4*np.pi*outR**2)/(4*np.pi*outR**2))
        np.save("/global/homes/x/xinshuo/"+ simname +"/surface_int_frame%d_R%.1f.npy"%(frameidx,outR),surface_results)
        if innerR > 0: # compute level 6 here
            surface_results = np.zeros(4)
            surface_results[0] = float(ds.current_time)
            mydomain = ds.box(level_left_edges[6], level_right_edges[6])
            surface = ds.surface(mydomain, 'radius', (innerR, 'code_length'))
            for idx,field in enumerate(['SURFACE_X','SURFACE_Y','SURFACE_Z']):
                if frameidx%64 == 0:
                    print(field)
                surface_fields_2d = surface[field]
                tris = surface.triangles
                tris_area = 0.5 * np.linalg.norm(np.cross(tris[:,1] - tris[:,0], tris[:,2] - tris[:,0]), axis=1)
                integral = np.sum(surface_fields_2d* tris_area)
                surface_results[idx+1] = float(integral)
                print("The integral of the field %s at innerR is %f"%(field,integral))
                print("area error: ", (float(tris_area.sum()) - 4*np.pi*innerR**2)/(4*np.pi*innerR**2))
            np.save("/global/homes/x/xinshuo/"+ simname +"/surface_int_frame%d_innerR%.1f.npy"%(frameidx,innerR),surface_results)
    if surface and curlevel == 7 and innerR < 0:
        # extract the surface integrals on the two black holes
        surface_results_bh1 = np.zeros(4)
        surface_results_bh1[0] = float(ds.current_time)
        surface_results_bh2 = np.zeros(4)
        surface_results_bh2[0] = float(ds.current_time)
        bh1center = ds.arr([bbh1_x*np.cos( - ds.current_time * binary_omega ), bbh1_x * np.sin( - ds.current_time * binary_omega ) ,0],"code_length")
        bh1_buffer = ds.sphere(bh1center,(bbh1_r*max_excise_factor,"code_length"))
        bh2center = ds.arr([bbh2_x*np.cos( - ds.current_time * binary_omega ), bbh2_x * np.sin( - ds.current_time * binary_omega ) ,0],"code_length")
        bh2_buffer = ds.sphere(bh2center,(bbh2_r*max_excise_factor,"code_length"))
        bh1_surface = ds.surface(bh1_buffer, 'radius', (bbh1_r*excise_factor, 'code_length'))
        bh2_surface = ds.surface(bh2_buffer, 'radius', (bbh2_r*excise_factor, 'code_length'))
        for idx,field in enumerate(['SURFACE_X','SURFACE_Y','SURFACE_Z']):
            if frameidx%64 == 0:
                print(field)
            bh1_surface_fields_2d = bh1_surface[field]
            bh2_surface_fields_2d = bh2_surface[field]
            tris_bh1 = bh1_surface.triangles
            tris_bh2 = bh2_surface.triangles
            tris_bh1_area = 0.5 * np.linalg.norm(np.cross(tris_bh1[:,1] - tris_bh1[:,0], tris_bh1[:,2] - tris_bh1[:,0]), axis=1)
            tris_bh2_area = 0.5 * np.linalg.norm(np.cross(tris_bh2[:,1] - tris_bh2[:,0], tris_bh2[:,2] - tris_bh2[:,0]), axis=1)
            integral_bh1 = np.sum(bh1_surface_fields_2d * tris_bh1_area)
            integral_bh2 = np.sum(bh2_surface_fields_2d * tris_bh2_area)
            surface_results_bh1[idx+1] = float(integral_bh1)
            surface_results_bh2[idx+1] = float(integral_bh2)
            print("The integral of the field %s on black hole 1 is %f"%(field,integral_bh1))
            print("The integral of the field %s on black hole 2 is %f"%(field,integral_bh2))
            print("area error: ", (float(tris_bh1_area.sum()) - 4*np.pi*(bbh1_r*excise_factor)**2)/(4*np.pi*(bbh1_r*excise_factor)**2))
            print("area error: ", (float(tris_bh2_area.sum()) - 4*np.pi*(bbh2_r*excise_factor)**2)/(4*np.pi*(bbh2_r*excise_factor)**2))
        np.save("/global/homes/x/xinshuo/"+ simname +"/surface_int_frame%d_exci%.1f_bh1.npy"%(frameidx,excise_factor),surface_results_bh1)
        np.save("/global/homes/x/xinshuo/"+ simname +"/surface_int_frame%d_exci%.1f_bh2.npy"%(frameidx,excise_factor),surface_results_bh2)
    if outplot:
        extent = [level_left_edges[curlevel][0],level_right_edges[curlevel][0],level_left_edges[curlevel][1],level_right_edges[curlevel][1]]
        # ax1.imshow(np.log10(field3d.sum(axis=0)),extent=extent,origin='lower')  # sum along z axis
        # im = ax1.imshow(np.log10(np.abs(field_3d_levels[curlevel][:,int(field_3d_levels[curlevel].shape[2]/2),:])).T,extent=extent,origin='lower',cmap='jet',vmax = -2, vmin = -9) 
        if plotxz:
            if plotsum:
                im = ax1.imshow(np.log10(np.abs(field_3d_levels[curlevel].sum(axis=1))).T,extent=extent,origin='lower',cmap='jet',vmax = -2, vmin = -9)  
            else:
                im = ax1.imshow(np.log10(np.abs(field_3d_levels[curlevel][:,int(field_3d_levels[curlevel].shape[2]/2),:])).T,extent=extent,origin='lower',cmap='jet',vmax = -2, vmin = -9)
        else:
            if plotsum:
                im = ax1.imshow(np.log10(np.abs(field_3d_levels[curlevel].sum(axis=2))).T,extent=extent,origin='lower',cmap='jet',vmax = -2, vmin = -9)
            else:
                im = ax1.imshow(np.log10(np.abs(field_3d_levels[curlevel][:,:,int(field_3d_levels[curlevel].shape[2]/2)])).T,extent=extent,origin='lower',cmap='jet',vmax = -2, vmin = -9)
if outplot:
    fig.colorbar(im)
    for i in [1,2,3,4,5,6]:
        # change x,y lim to the largest extent
        ax1.set_xlim([level_left_edges[i][0],level_right_edges[i][0]])
        ax1.set_ylim([level_left_edges[i][1],level_right_edges[i][1]])
        # show colorbar and save the plot
        basename = "integrate_" + integratefield + "_level%d"%i
        if plotsum:
            basename += "_sum"
        if plotxz:
            basename += "_xz"
        fig.savefig(basename + ".pdf",dpi = 900)

if allfields:

    # save the result as a dict
    result = {'time':ds.current_time}
    result_array = np.zeros(7)
    result_array[0] = float(ds.current_time)

    print("integrating field")
    # enumerate the field and idx
    for idx,field in enumerate(['SMOMENTUM_X','SMOMENTUM_Y','SMOMENTUM_Z','VOLUME_X','VOLUME_Y','VOLUME_Z']):
    # for field in ['SMOMENTUM_X','SMOMENTUM_Y','SMOMENTUM_Z','VOLUME_X','VOLUME_Y','VOLUME_Z']:
        # result[field] = np.sum([ allfields_3d_levels[(field,curlevel)].sum()*level_dxs[curlevel][0]**3 for curlevel in [1,2,3,4,5,6]])
        if frameidx%64 == 0:
            print(field)
        result_array[idx+1] = np.sum([ allfields_3d_levels[(field,curlevel)].sum()*level_dxs[curlevel][0]**3 for curlevel in [1,2,3,4,5,6]])

    # save the result as a npy file, mark the frameidx, outR and excise_factor
    # np.save("/global/homes/x/xinshuo/"+ simname +"/volume_int_frame%d_R%.1f_excise%.1f.npy"%(args.frameidx,outR,excise_factor),result)
    if innerR > 0:
        np.save("/global/homes/x/xinshuo/"+ simname +"/volume_int_frame%d_R%.1f_innerR%.1f.npy"%(args.frameidx,outR,innerR),result_array)
    else:
        np.save("/global/homes/x/xinshuo/"+ simname +"/volume_int_frame%d_R%.1f_excise%.1f.npy"%(args.frameidx,outR,excise_factor),result_array)


