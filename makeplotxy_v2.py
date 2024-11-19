import yt
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

# make gif animations
# import os
import imageio

# parse arguments
parser = argparse.ArgumentParser(description='Plot scalar fields from simulation data.')
parser.add_argument('--simname', type=str, default="20240719_new", help='Name of the simulation directory')
parser.add_argument('--skipevery', type=int, default=1, help='Skip every n iterations')
parser.add_argument('--plotfield', type=str, default='SPHI', help='Field to plot')
parser.add_argument('--extent_x', type=int, default=10, help='Extent of the plot in x direction')
parser.add_argument('--extent_y', type=int, default=10, help='Extent of the plot in y direction')
parser.add_argument('--plot_log', type=bool, default=False, help='Plot in logarithmic scale')
parser.add_argument('--plot_cmap', type=str, default='RdBu_r', help='Colormap for the plot')
parser.add_argument('--plot_cmap_max', type=float, default=1, help='Maximum value for colormap')
parser.add_argument('--plot_cmap_min', type=float, default=-1, help='Minimum value for colormap')
parser.add_argument('--plot_fontsize', type=int, default=30, help='Font size for the plot')
parser.add_argument('--plotallfield', action='store_true', help='whether to plot all fields')
parser.add_argument('--maxframes', type=int, default=500, help='max num of frames (to avoid OOM)')

args = parser.parse_args()

simname = args.simname
skipevery = args.skipevery
plotfield = args.plotfield
extent_x = args.extent_x
extent_y = args.extent_y
plot_log = args.plot_log
plot_cmap = args.plot_cmap
plot_cmap_max = args.plot_cmap_max
plot_cmap_min = args.plot_cmap_min
plot_fontsize = args.plot_fontsize
plotallfield = args.plotallfield
maxframes = args.maxframes

if plotallfield:
    fields_toplot = ['SPHI', 'SPHI2', 'SPI', 'SPI2']
else:
    fields_toplot = [plotfield]

# depending on the field, set the label, 'SPHI" -> "Re$\\phi$", 'SPHI2' -> "Im$\\phi$", 'SPI' -> "Re$\\Pi$", 'SPI2' -> "Im$\\Pi$" 


basedir = "/pscratch/sd/x/xinshuo/runGReX/"
plotdir = "/pscratch/sd/x/xinshuo/plotGReX/"
rundir = basedir + simname +"/"
# there are a lot of pltXXXXX dirs, corresponding to different iterations, load all of them

# Find all directories in rundir that match the pattern 'pltXXXXX' or more digits
plt_dirs = sorted([d for d in os.listdir(rundir) if d.startswith('plt') and d[3:].isdigit()], key=lambda x: int(x[3:]))
plt_dirs = plt_dirs[::skipevery] # skip some iteration to save time

if len(plt_dirs) > maxframes:
    plt_dirs = plt_dirs[:maxframes]

# Load all datasets
datasets = [yt.load(os.path.join(rundir, d)) for d in plt_dirs]


# Create xgrid and ygrid
ds = datasets[0] # assuming the domain is the same for all datasets and fields
slc = yt.SlicePlot(ds, 'z', plotfield)
frb = slc.frb
data = frb[plotfield].d
x = np.linspace(float(ds.domain_left_edge[0]), float(ds.domain_right_edge[0]), data.shape[0])
y = np.linspace(float(ds.domain_left_edge[1]), float(ds.domain_right_edge[1]), data.shape[1])
xgrid, ygrid = np.meshgrid(x, y)

msk = xgrid<extent_x/2
msk = msk & (xgrid>-extent_x/2)
msk = msk & (ygrid<extent_y/2)
msk = msk & (ygrid>-extent_y/2)

msked_xgrid = xgrid[msk]
msked_ygrid = ygrid[msk]


# plot the scalar field

for plotfield in fields_toplot:

    if plotfield == 'SPHI':
        fieldlabel = "Re$\\phi$"
    elif plotfield == 'SPHI2':
        fieldlabel = "Im$\\phi$"
    elif plotfield == 'SPI':
        fieldlabel = "Re$\\Pi$"
    elif plotfield == 'SPI2':
        fieldlabel = "Im$\\Pi$"
    else:
        fieldlabel = plotfield

    # Check if the directory exists, if not, create it
    outputdir = os.path.join(plotdir, simname, "frames")
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    allfiles = []

    for itidx in range(len(datasets)):

        ds = datasets[itidx]

        # Create a slice plot
        slc = yt.SlicePlot(ds, 'z', plotfield)
        if not plot_log:
            slc.set_log(plotfield, False)

        # Set the width of the plot
        # slc.set_width((extent_x, extent_y))

        # Convert the slice plot to a fixed resolution buffer (FRB)
        # frb = slc.data.to_frb((extent_x, 'unitary'), (extent_y, 'unitary'))
        frb = slc.frb

        # Extract the data as a numpy array
        data = frb[plotfield].d
        msked_data = data[msk]



        # # Now you can perform further calculations with data, xgrid, and ygrid
        # # For example, SPHI * X
        # if plotfield == 'SPHI':
        #     result = data * xgrid

        # # Save the numpy arrays if needed
        # np.save(os.path.join(outputdir, f"{plotfield}_data_it{itidx:05d}.npy"), data)
        # np.save(os.path.join(outputdir, f"{plotfield}_xgrid_it{itidx:05d}.npy"), xgrid)
        # np.save(os.path.join(outputdir, f"{plotfield}_ygrid_it{itidx:05d}.npy"), ygrid)

        # Plot using Matplotlib
        plt.figure(figsize=(10, 8))
        # plt.pcolormesh(msked_xgrid, msked_ygrid, msked_data, cmap=plot_cmap, vmin=plot_cmap_min, vmax=plot_cmap_max)
        # use imshow with edges correctly aligned
        plt.imshow(data.T, origin='lower', extent=[float(ds.domain_left_edge[0]), float(ds.domain_right_edge[0]), float(ds.domain_left_edge[1]), float(ds.domain_right_edge[1])], cmap=plot_cmap, vmin=plot_cmap_min, vmax=plot_cmap_max)
        plt.colorbar(label=fieldlabel)
        plt.xlabel('$x/M$')
        plt.ylabel('$y/M$')
        # plt.title(f"{fieldlabel} at $t = {float(ds.current_time):.2f} M$")
        
        # put a text box in the plot on the top left, text: "t = %f"%(float(ds.current_time))
        slc.annotate_text((0.05, 0.95), "$t = %.2f M$"%(float(ds.current_time)), coord_system='axis', text_args={'color': 'black', 'fontsize': plot_fontsize})
        # using plt to do this
        plt.text(0.05, 0.95, "$t = %.2f M$"%(float(ds.current_time)), transform=plt.gca().transAxes, color='black', fontsize=plot_fontsize)

        plt.xlim(-extent_x, extent_x)
        plt.ylim(-extent_y, extent_y)

        filename_base = f"{plotfield}_ext{extent_x:.1f}_it{itidx:05d}"
        plt.savefig(os.path.join(outputdir, f"{filename_base}.png"))
        plt.savefig(os.path.join(outputdir, f"{filename_base}.pdf"))
        plt.close()
        allfiles.append(os.path.join(outputdir, f"{filename_base}.png"))

    # from the images in allfiles, make a gif
    images = []
    for filename in allfiles:
        images.append(imageio.imread(filename))
    imageio.mimsave(os.path.join(plotdir, simname, f"{plotfield}_ext{extent_x:.1f}.gif"), images, duration=0.1)