import numpy as np
import matplotlib.pyplot as plt

# Set plot parameters
plt.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": 'cm',  # Computer Modern font - looks like LaTeX
    "font.family": 'STIXGeneral'
})

# Define parameters
outR = 320.0
excise_factor = 1.5
simname_restart = '250528_BBH_r70_moreplots_restart'
simname_early = '250528_BBH_r70_moreplots'

# Define paths
plotdir = "/pscratch/sd/x/xinshuo/plotGReX/"

# Load data from restart simulation (t > 2000)
results_restart = np.load(plotdir+f"{simname_restart}_2d_integrals_outR{outR}_excise{excise_factor}_parallel.npy")
results_sur_restart = np.load(plotdir+f"{simname_restart}_2d_integrals_surface_outR{outR}_excise{excise_factor}_parallel.npy")

# Load data from early simulation (t < 2000)
results_early = np.load(plotdir+f"{simname_early}_2d_integrals_outR{outR}_excise{excise_factor}.npy")
results_sur_early = np.load(plotdir+f"{simname_early}_2d_integrals_surface_outR{outR}_excise{excise_factor}.npy")

# Remove the first data point in each dataset
results_restart = results_restart[1:]
results_sur_restart = results_sur_restart[1:]
results_early = results_early[1:]
results_sur_early = results_sur_early[1:]

# Combine the datasets
# Filter early results to t < 2000
mask_early = results_early[:, 0] < 2000
results_early = results_early[mask_early]
results_sur_early = results_sur_early[mask_early]

# Concatenate datasets
results = np.concatenate((results_early, results_restart), axis=0)
results_sur = np.concatenate((results_sur_early, results_sur_restart), axis=0)

# Calculate time parameters
dt = results[1,0] - results[0,0]
binary_mass = 2.71811e+00
binary_omega = -0.002657634562418009 * binary_mass
T_orbit = np.abs(2*np.pi/binary_omega)
window_size = int(T_orbit/dt)

# Calculate smoothed drag force
smoothed_drag = np.zeros_like(results[:,1])
valid_smoothed = np.convolve(results[:,1], np.ones(window_size)/window_size, mode='valid')
smoothed_drag[window_size-1:window_size-1+len(valid_smoothed)] = valid_smoothed

# Handle left boundary for drag force
for i in range(window_size-1):
    curr_window = i + 1
    smoothed_drag[i] = np.sum(results[:curr_window,1]) / curr_window

# Calculate smoothed torque
smoothed_torque = np.zeros_like(results[:,7])
valid_smoothed_torque = np.convolve(results[:,7], np.ones(window_size)/window_size, mode='valid')
smoothed_torque[window_size-1:window_size-1+len(valid_smoothed_torque)] = valid_smoothed_torque

# Handle left boundary for torque
for i in range(window_size-1):
    curr_window = i + 1
    smoothed_torque[i] = np.sum(results[:curr_window,7]) / curr_window

# Convert time to units of T_orbit
time_in_orbits = results[:,0] / T_orbit

# Create figure with two panels
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

# Plot drag force in upper panel
ax1.plot(time_in_orbits, -results[:,1]/results_sur[:,10], '--', color='grey', alpha=0.5, 
         label='$-F^{\\mathrm{drag}}_x / \\langle \\rho \\rangle$')
ax1.plot(time_in_orbits, -smoothed_drag/results_sur[:,10], 'k', 
         label='$-\\langle F^{\\mathrm{drag}}_x \\rangle_{T_{\\mathrm{orbit}}} / \\langle \\rho \\rangle$')
ax1.set_ylabel('$-F^{\\mathrm{drag}}_x / \\langle \\rho \\rangle$', fontsize=14)
ax1.set_xlim(0, 11)
ax1.legend(fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.7)

# Plot torque in lower panel
ax2.plot(time_in_orbits, -results[:,7]/results_sur[:,10], '--', color='cyan', alpha=0.5, 
         label='$-\\tau^{\\mathrm{drag}}_z / \\langle \\rho \\rangle$')
ax2.plot(time_in_orbits, -smoothed_torque/results_sur[:,10], 'blue', 
         label='$-\\langle \\tau^{\\mathrm{drag}}_z \\rangle_{T_{\\mathrm{orbit}}} / \\langle \\rho \\rangle$')
ax2.set_xlabel('$t/T_{\\mathrm{orbit}}$', fontsize=14)
ax2.set_ylabel('$-\\tau^{\\mathrm{drag}}_z / \\langle \\rho \\rangle$', fontsize=14)
ax2.set_xlim(0, 11)
ax2.legend(fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.7)

# Adjust layout
plt.tight_layout()

# Save figure
plt.savefig(f"{plotdir}/drag_averaging.pdf", bbox_inches='tight')
plt.show() 