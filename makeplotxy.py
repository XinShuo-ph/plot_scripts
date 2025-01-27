import yt
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

# make gif animations
# import os
import imageio


# orbital period of the binary
# Torbit = 2*np.pi/(0.0077395162481920582*2.71811)

# parse arguments
parser = argparse.ArgumentParser(description='Plot scalar field rho from simulation data.')
parser.add_argument('--simname', type=str, default="20240719_new", help='Name of the simulation directory')
parser.add_argument('--skipevery', type=int, default=1, help='Skip every n iterations')
parser.add_argument('--plotfield', type=str, default='SPHI', help='Field to plot')
parser.add_argument('--plotallfield', type=bool, default=False, help='Plot all fields')
parser.add_argument('--extent_x', type=int, default=10, help='Extent of the plot in x direction')
parser.add_argument('--extent_y', type=int, default=10, help='Extent of the plot in y direction')
parser.add_argument('--plot_log', type=bool, default=False, help='Plot in logarithmic scale')
parser.add_argument('--plot_cmap', type=str, default='RdBu_r', help='Colormap for the plot')
parser.add_argument('--plot_cmap_max', type=float, default=1, help='Maximum value for colormap')
parser.add_argument('--plot_cmap_min', type=float, default=-1, help='Minimum value for colormap')
parser.add_argument('--plot_fontsize', type=int, default=30, help='Font size for the plot')
parser.add_argument('--maxframes', type=int, default=500, help='max num of frames (to avoid OOM)')
parser.add_argument('--mu2', type=float, default=0.04, help='scalar field mass')
parser.add_argument('--Torbit', type=float, default=2*np.pi/(0.0077395162481920582*2.71811), help='orbital period of the binary')
parser.add_argument('--labelT', type=bool, default=True, help='label time in units of Torbit')

args = parser.parse_args()

simname = args.simname
skipevery = args.skipevery
plotfield = args.plotfield
plotallfield = args.plotallfield
extent_x = args.extent_x
extent_y = args.extent_y
plot_log = args.plot_log
plot_cmap = args.plot_cmap
plot_cmap_max = args.plot_cmap_max
plot_cmap_min = args.plot_cmap_min
plot_fontsize = args.plot_fontsize
maxframes = args.maxframes
mu2 = args.mu2
Torbit = args.Torbit
labelT = args.labelT

if plotallfield:
    fields_toplot = ['SPHI', 'SPHI2', 'SPI', 'SPI2']
else:
    fields_toplot = [plotfield]

# depending on the field, set the label, 'SPHI" -> "Re$\\phi$", 'SPHI2' -> "Im$\\phi$", 'SPI' -> "Re$\\Pi$", 'SPI2' -> "Im$\\Pi$" 


basedir = "/data/xinshuo/runGReX/"
plotdir = "/data/xinshuo/plot_scripts/"

# make plotdir + simname directory if it does not exist
if not os.path.exists(os.path.join(plotdir, simname)):
    os.makedirs(os.path.join(plotdir, simname))

rundir = basedir + simname +"/"
# there are a lot of pltXXXXX dirs, corresponding to different iterations, load all of them

# Find all directories in rundir that match the pattern 'pltXXXXX' or more digits
plt_dirs = sorted([d for d in os.listdir(rundir) if d.startswith('plt') and d[3:].isdigit()], key=lambda x: int(x[3:]))
plt_dirs = plt_dirs[::skipevery] # skip some iteration to save time

if len(plt_dirs) > maxframes:
    plt_dirs = plt_dirs[:maxframes]

# Load all datasets
datasets = [yt.load(os.path.join(rundir, d)) for d in plt_dirs]

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

        slc = yt.SlicePlot(ds, 'z', plotfield)
        if not plot_log:
            slc.set_log(plotfield, False)
        # slc.set_log(plotfield, False)

        # Set xlim and ylim according to the domain edges
        # slc.set_width((ds.domain_right_edge[0] - ds.domain_left_edge[0],
        #             ds.domain_right_edge[1] - ds.domain_left_edge[1]))
        
        slc.set_width((extent_x,extent_y))
        # Set xlabel and ylabel
        if extent_x>100:
            slc.set_xlabel('$x/100M$')
            slc.set_ylabel('$y/100M$')
        else:
            slc.set_xlabel('$x/M$')
            slc.set_ylabel('$y/M$')

        # Set colorbar label
        slc.set_colorbar_label(plotfield, fieldlabel)
        slc.set_cmap(plotfield, plot_cmap)
        slc.set_zlim(plotfield, plot_cmap_min, plot_cmap_max)

        slc.set_font({'size': plot_fontsize})

        # put a text box in the plot on the top left, text: "t = %f"%(float(ds.current_time))

        if labelT:
            slc.annotate_text((0.05, 0.95), "$t = %.2f T $" % (float(ds.current_time)/Torbit), coord_system='axis',
                        text_args={'color': 'black', 'fontsize': plot_fontsize})
            filename_base = f"{plotfield}_ext{extent_x:.1f}_T_it{itidx:05d}"
        else:
            slc.annotate_text((0.05, 0.95), "$t = %.2f M$" % (float(ds.current_time)), coord_system='axis',
                        text_args={'color': 'black', 'fontsize': plot_fontsize})
            filename_base = f"{plotfield}_ext{extent_x:.1f}_it{itidx:05d}"
        # Save plot files according to plotdir, plotfield, itidx
        
        slc.save(os.path.join(outputdir, f"{filename_base}.png"))
        slc.save(os.path.join(outputdir, f"{filename_base}.pdf"))
        allfiles.append(os.path.join(outputdir, f"{filename_base}.png"))



    # from the images in allfiles, make a gif
    images = []
    for filename in allfiles:
        images.append(imageio.imread(filename))
    imageio.mimsave(os.path.join(plotdir, simname, f"{plotfield}_ext{extent_x:.1f}.gif"), images, duration=0.1)