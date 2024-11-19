import yt
import numpy as np
import os
import argparse
import imageio

# orbital period of the binary
# Torbit = 2*np.pi/(0.0077395162481920582*2.71811)

# parse arguments
parser = argparse.ArgumentParser(description='Plot scalar field rho from simulation data.')
parser.add_argument('--simname', type=str, default="20240719_new", help='Name of the simulation directory')
parser.add_argument('--skipevery', type=int, default=1, help='Skip every n iterations')
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

basedir = "/pscratch/sd/x/xinshuo/runGReX/"
plotdir = "/pscratch/sd/x/xinshuo/plotGReX/"
rundir = basedir + simname + "/"

# Find all directories in rundir that match the pattern 'pltXXXXX' or more digits
plt_dirs = sorted([d for d in os.listdir(rundir) if d.startswith('plt') and d[3:].isdigit()], key=lambda x: int(x[3:]))
plt_dirs = plt_dirs[::skipevery]  # skip some iterations to save time

if len(plt_dirs) > maxframes:
    plt_dirs = plt_dirs[:maxframes]

# Load all datasets
datasets = [yt.load(os.path.join(rundir, d)) for d in plt_dirs]

# Define a derived field for rho
def _rho(field, data):
    return 0.5*(mu2 * data['SPHI']**2 + data['SPI']**2)


# Define a derived field for rho
def _rho2(field, data):
    return 0.5*(mu2 * data['SPHI2']**2 + data['SPI2']**2)

for ds in datasets:
    ds.add_field(('gas', 'scalar_rho'), function=_rho,  sampling_type='cell', units='dimensionless')
    ds.add_field(('gas', 'scalar_rho2'), function=_rho2,  sampling_type='cell', units='dimensionless')

# define a derived field for the beta field, BETAX_NO_OMEGA = BETAX - omega * y, where y is the y coordinate
# def _beta(field, data):
    # y coordinate is not in data
    # need to derive it from using yt


# Check if the directory exists, if not, create it
outputdir = os.path.join(plotdir, simname, "frames")
if not os.path.exists(outputdir):
    os.makedirs(outputdir)

allfiles = []

for itidx in range(len(datasets)):
    ds = datasets[itidx]

    rho = ('gas', 'scalar_rho')

    slc = yt.SlicePlot(ds, 'z', ('gas', 'scalar_rho'))
    if not plot_log:
        slc.set_log(('gas', 'scalar_rho'), False)

    slc.set_width((extent_x, extent_y))
    if extent_x>100:
        slc.set_xlabel('$x/100M$')
        slc.set_ylabel('$y/100M$')
    else:
        slc.set_xlabel('$x/M$')
        slc.set_ylabel('$y/M$')
        
    slc.set_colorbar_label(rho, "$\\rho_1$")
    slc.set_cmap(rho, plot_cmap)
    slc.set_zlim(rho, plot_cmap_min, plot_cmap_max)

    slc.set_font({'size': plot_fontsize})
    if labelT:
        slc.annotate_text((0.05, 0.95), "$t = %.2f T $" % (float(ds.current_time)/Torbit), coord_system='axis',
                      text_args={'color': 'black', 'fontsize': plot_fontsize})
        filename_base = f"rho_ext{extent_x:.1f}_T_it{itidx:05d}"
    else:
        slc.annotate_text((0.05, 0.95), "$t = %.2f M$" % (float(ds.current_time)), coord_system='axis',
                      text_args={'color': 'black', 'fontsize': plot_fontsize})
        filename_base = f"rho_ext{extent_x:.1f}_it{itidx:05d}"
    slc.save(os.path.join(outputdir, f"{filename_base}.png"))
    slc.save(os.path.join(outputdir, f"{filename_base}.pdf"))
    allfiles.append(os.path.join(outputdir, f"{filename_base}.png"))

# Create a GIF animation from the images
images = []
for filename in allfiles:
    images.append(imageio.imread(filename))
if labelT:
    imageio.mimsave(os.path.join(plotdir, simname, f"rho_ext{extent_x:.1f}_T.gif"), images, duration=0.1)
else:
    imageio.mimsave(os.path.join(plotdir, simname, f"rho_ext{extent_x:.1f}.gif"), images, duration=0.1)


allfiles = []

for itidx in range(len(datasets)):
    ds = datasets[itidx]

    rho = ('gas', 'scalar_rho2')

    slc = yt.SlicePlot(ds, 'z', ('gas', 'scalar_rho2'))
    if not plot_log:
        slc.set_log(('gas', 'scalar_rho2'), False)

    slc.set_width((extent_x, extent_y))
    if extent_x>100:
        slc.set_xlabel('$x/100M$')
        slc.set_ylabel('$y/100M$')
    else:
        slc.set_xlabel('$x/M$')
        slc.set_ylabel('$y/M$')

    slc.set_colorbar_label(rho, "$\\rho_2$")
    slc.set_cmap(rho, plot_cmap)
    slc.set_zlim(rho, plot_cmap_min, plot_cmap_max)

    slc.set_font({'size': plot_fontsize})
    if labelT:
        slc.annotate_text((0.05, 0.95), "$t = %.2f T $" % (float(ds.current_time)/Torbit), coord_system='axis',
                      text_args={'color': 'black', 'fontsize': plot_fontsize})
        filename_base = f"rho2_ext{extent_x:.1f}_T_it{itidx:05d}"
    else:
        slc.annotate_text((0.05, 0.95), "$t = %.2f M$" % (float(ds.current_time)), coord_system='axis',
                      text_args={'color': 'black', 'fontsize': plot_fontsize})
        filename_base = f"rho2_ext{extent_x:.1f}_it{itidx:05d}"
    slc.save(os.path.join(outputdir, f"{filename_base}.png"))
    slc.save(os.path.join(outputdir, f"{filename_base}.pdf"))
    allfiles.append(os.path.join(outputdir, f"{filename_base}.png"))

# Create a GIF animation from the images
images = []
for filename in allfiles:
    images.append(imageio.imread(filename))
if labelT:
    imageio.mimsave(os.path.join(plotdir, simname, f"rho2_ext{extent_x:.1f}_T.gif"), images, duration=0.1)
else:
    imageio.mimsave(os.path.join(plotdir, simname, f"rho2_ext{extent_x:.1f}.gif"), images, duration=0.1)