import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Set the font and plot parameters to match the notebook
plt.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": 'cm',  # Computer Modern font - looks like LaTeX
    "font.family": 'STIXGeneral'
})

# Use seaborn's colorblind palette for better accessibility
palette = sns.color_palette("colorblind", 4)

# Parameters needed for calculations
outR = 320.0
excise_factor = 1.5
average_rho_after = 5000
surface_factor_bbh = 1.0/np.pi
volume_factor_bbh_torque = 1.2
volume_factor_bbh_macc = 0.5

# Define binary parameters (assuming these are common across simulations)
binary_mass = +2.71811e+00
bbh1_x = -(70.49764373525885/2-2.926860031395978)/binary_mass
bbh2_x = -(70.49764373525885/2+2.926860031395978)/binary_mass
bbh1_r = 3.98070/binary_mass
bbh2_r = 3.98070/binary_mass
binary_omega = - 0.002657634562418009 * 2.71811
T_orbit = np.abs(2*np.pi/binary_omega) 

print("T_orbit = ", T_orbit)

# Define the threshold time for computing averages
t_threshold = 4000

# Path to plot directory (should be adjusted if needed)
plotdir = ""  # Add your plot directory here if needed

# Define the simulation sets with their parameters
simulation_sets = [
    {
        "name": "n256_tuned_damping",
        "label": r"$\mu = 0.2 M^{-1}, d_{\rm BBH}=26M$",
        "simnames": [ 
            (0.3, '20250622_tune_damping_n256_v03'),
            (0.4, '20250622_tune_damping_n256_v04'),
            (0.45, '20250623_tune_damping_n256_v045'),
            (0.5, '20250620_tune_damping_n256_1'),
            (0.55, '20250623_tune_damping_n256_v055'),
            (0.6, '20250621_tune_damping_n256_v06'),
            (0.7, '20250622_tune_damping_n256_v07')
        ],
        "outR": 320.0,
        "excise_factor": 1.5,
        "average_rho_after": 5000,
        "t_threshold": 4000,
        "tend_threshold": None,
        "color": palette[0],
        "filter_v07": True
    },
    {
        "name": "mu02_r35",
        "label": r"$\mu = 0.2 M^{-1}, d_{\rm BBH}=13M$",
        "simnames": [     
            (0.3, '20250702_mu02_v03_r35'),
            (0.4, '20250627_mu02_v04_r35'),
            (0.45, '20250628_mu02_v045_r35'),
            (0.5, '20250627_mu02_v05_r35'),
            (0.55, '20250628_mu02_v055_r35'),
            (0.6, '20250627_mu02_v06_r35'),
            (0.7, '20250627_mu02_v07_r35')
        ],
        "outR": 320.0,
        "excise_factor": 1.5,
        "average_rho_after": 5000,
        "t_threshold": 4000,
        "tend_threshold": None,
        "color": palette[1],
        "filter_v07": False
    },
    {
        "name": "ncell320_mu08",
        "label": r"$\mu = 0.8 M^{-1}, d_{\rm BBH}=26M$",
        "simnames": [     
            (0.4, '20250628_ncell320_v04_mu08'),
            (0.45, '20250629_ncell320_v045_mu08'),
            (0.5, '20250629_ncell320_v05_mu08'),
            (0.55, '20250629_ncell320_v055_mu08'),
            (0.6, '20250628_ncell320_v06_mu08')
        ],
        "outR": 100.0,
        "excise_factor": 1.5,
        "average_rho_after": 5000,
        "t_threshold": 4000,
        "tend_threshold": None,
        "color": palette[2],
        "filter_v07": False
    },
    {
        "name": "ncell416_mu005_r35",
        "label": r"$\mu = 0.05 M^{-1}, d_{\rm BBH}=13M$",
        "simnames": [     
            (0.2, '20250630_ncell416_v02_mu005_r35'),
            (0.3, '20250630_ncell416_v03_mu005_r35'),
            (0.4, '20250630_ncell416_v04_mu005_r35'),
            (0.45, '20250702_ncell416_v045_mu005_r35'),
            (0.5, '20250630_ncell416_v05_mu005_r35'),
            (0.55, '20250702_ncell416_v055_mu005_r35'),
            (0.6, '20250702_ncell416_v06_mu005_r35'),
            (0.7, '20250702_ncell416_v07_mu005_r35')
        ],
        "outR": 100.0,
        "excise_factor": 1.5,
        "average_rho_after": 1000,
        "t_threshold": 1000,
        "tend_threshold": 5000,
        "color": palette[3],
        "filter_v07": False,
        "torque_sign": 1,  # Positive torque normalization as seen in the notebook
        "macc_sign": 1 
    }
]

# Function to process a simulation set
def process_simulation_set(sim_set):
    velocities = []
    fdrag_avg = []
    fdrag_std = []
    torque_avg = []
    torque_std = []
    macc_avg = []
    macc_std = []
    qacc_avg = []
    qacc_std = []
    
    for wave_vel, simname in sim_set["simnames"]:
        print(f"Processing {sim_set['name']} velocity {wave_vel}")
        
        try:
            # Load the data
            results = np.load(plotdir+f"{simname}_2d_integrals_outR{sim_set['outR']}_excise{sim_set['excise_factor']}_psipow2.0_parallel.npy")
            results = results[1:]  # Remove first data point
            
            results_sur = np.load(plotdir+f"{simname}_2d_integrals_surface_outR{sim_set['outR']}_excise{sim_set['excise_factor']}_psipow-2.0_psipow_surface_correction-2.0_parallel.npy")
            results_sur = results_sur[1:]
            
            # Apply tend_threshold if specified
            if sim_set.get("tend_threshold"):
                results = results[results[:,0]<=sim_set["tend_threshold"]]
                results_sur = results_sur[results_sur[:,0]<=sim_set["tend_threshold"]]
            
            dt = results[1,0] - results[0,0]
            window_size = int(T_orbit/dt)
            
            # Average density for normalization
            avgrho = np.mean(results_sur[:,10][results_sur[:,0]>sim_set["average_rho_after"]])
            
            # 1. Drag Force calculation
            smoothed_drag = np.zeros_like(results[:,1])
            valid_smoothed = np.convolve(results[:,1], np.ones(window_size)/window_size, mode='valid')
            smoothed_drag[window_size-1:window_size-1+len(valid_smoothed)] = valid_smoothed
            
            for i in range(window_size-1):
                curr_window = i + 1
                smoothed_drag[i] = np.sum(results[:curr_window,1]) / curr_window
            
            normalized_drag = -smoothed_drag/avgrho
            
            # 2. Torque calculation
            smoothed_torque = np.zeros_like(results[:,0])
            valid_smoothed_torque = np.convolve(
                results[:,7]*volume_factor_bbh_torque - (results_sur[:,12] + results_sur[:,13])*surface_factor_bbh, 
                np.ones(window_size)/window_size, 
                mode='valid'
            )
            smoothed_torque[window_size-1:window_size-1+len(valid_smoothed_torque)] = valid_smoothed_torque
            
            for i in range(window_size-1):
                curr_window = i + 1
                smoothed_torque[i] = np.sum(
                    results[:curr_window,7]*volume_factor_bbh_torque - 
                    (results_sur[:curr_window,12] + results_sur[:curr_window,13])*surface_factor_bbh
                ) / curr_window
            
            # Use the correct sign for torque normalization based on the dataset
            if sim_set.get("torque_sign") == 1:
                normalized_torque = smoothed_torque/avgrho
            else:
                normalized_torque = -smoothed_torque/avgrho
            
            # 3. Mass Accretion calculation
            smoothed_macc = np.zeros_like(results[:,0])
            valid_smoothed_macc = np.convolve(
                results[:,10]*volume_factor_bbh_macc - (results_sur[:,16] + results_sur[:,17])*surface_factor_bbh, 
                np.ones(window_size)/window_size, 
                mode='valid'
            )
            smoothed_macc[window_size-1:window_size-1+len(valid_smoothed_macc)] = valid_smoothed_macc
            
            for i in range(window_size-1):
                curr_window = i + 1
                smoothed_macc[i] = np.sum(
                    results[:curr_window,10]*volume_factor_bbh_macc - 
                    (results_sur[:curr_window,16] + results_sur[:curr_window,17])*surface_factor_bbh
                ) / curr_window

            if sim_set.get("macc_sign") == 1:
                normalized_macc = smoothed_macc/avgrho
            else:
                normalized_macc = -smoothed_macc/avgrho
            
            # 4. Charge Accretion calculation
            smoothed_qacc = np.zeros_like(results[:,0])
            valid_smoothed_qacc = np.convolve(
                (results_sur[:,18] + results_sur[:,19]), 
                np.ones(window_size)/window_size, 
                mode='valid'
            )
            smoothed_qacc[window_size-1:window_size-1+len(valid_smoothed_qacc)] = valid_smoothed_qacc
            
            for i in range(window_size-1):
                curr_window = i + 1
                smoothed_qacc[i] = np.sum(results_sur[:curr_window,18] + results_sur[:curr_window,19]) / curr_window
            
            normalized_qacc = -smoothed_qacc/avgrho
            
            # Calculate statistics for t > t_threshold
            mask = results[:,0] > sim_set["t_threshold"]
            
            # Store velocity and computed statistics
            velocities.append(wave_vel)
            
            # Drag force
            fdrag_avg.append(np.mean(normalized_drag[mask]))
            fdrag_std.append(np.std(normalized_drag[mask]))
            
            # Torque
            torque_avg.append(np.mean(normalized_torque[mask]))
            torque_std.append(np.std(normalized_torque[mask]))
            
            # Mass accretion
            macc_avg.append(np.mean(normalized_macc[mask]))
            macc_std.append(np.std(normalized_macc[mask]))
            
            # Charge accretion
            qacc_avg.append(np.mean(normalized_qacc[mask]))
            qacc_std.append(np.std(normalized_qacc[mask]))
            
        except Exception as e:
            print(f"Error processing velocity {wave_vel}: {e}")
    
    # Convert to numpy arrays for easier processing
    velocities = np.array(velocities)
    fdrag_avg = np.array(fdrag_avg)
    fdrag_std = np.array(fdrag_std)
    torque_avg = np.array(torque_avg)
    torque_std = np.array(torque_std)
    macc_avg = np.array(macc_avg)
    macc_std = np.array(macc_std)
    qacc_avg = np.array(qacc_avg)
    qacc_std = np.array(qacc_std)
    
    # Sort everything by velocity
    sort_indices = np.argsort(velocities)
    velocities = velocities[sort_indices]
    fdrag_avg = fdrag_avg[sort_indices]
    fdrag_std = fdrag_std[sort_indices]
    torque_avg = torque_avg[sort_indices]
    torque_std = torque_std[sort_indices]
    macc_avg = macc_avg[sort_indices]
    macc_std = macc_std[sort_indices]
    qacc_avg = qacc_avg[sort_indices]
    qacc_std = qacc_std[sort_indices]
    
    return {
        "velocities": velocities,
        "fdrag_avg": fdrag_avg,
        "fdrag_std": fdrag_std,
        "torque_avg": torque_avg,
        "torque_std": torque_std,
        "macc_avg": macc_avg,
        "macc_std": macc_std,
        "qacc_avg": qacc_avg,
        "qacc_std": qacc_std
    }

# Process all simulation sets
processed_data = []
for sim_set in simulation_sets:
    data = process_simulation_set(sim_set)
    processed_data.append(data)

# Create 2x2 panel plot
fig = plt.figure(figsize=(10, 8))
gs = GridSpec(2, 2, figure=fig)

# 1. Drag Force vs Velocity
ax1 = fig.add_subplot(gs[0, 0])
for i, sim_set in enumerate(simulation_sets):
    data = processed_data[i]
    
    # Get relevant data
    velocities = data["velocities"]
    fdrag_avg = data["fdrag_avg"]
    fdrag_std = data["fdrag_std"]
    
    # Special filtering for the first simulation set (n256_tuned_damping)
    if sim_set["name"] == "n256_tuned_damping":
        # Filter out v=0.3 and v=0.7 points
        fdrag_plot_mask = (velocities != 0.3) & (velocities != 0.7)
        velocities_plot = velocities[fdrag_plot_mask]
        fdrag_avg_plot = fdrag_avg[fdrag_plot_mask]
        fdrag_std_plot = fdrag_std[fdrag_plot_mask]
    else:
        velocities_plot = velocities
        fdrag_avg_plot = fdrag_avg
        fdrag_std_plot = fdrag_std
    
    # Plot data points with error bars
    ax1.errorbar(
        velocities_plot, 
        fdrag_avg_plot, 
        yerr=fdrag_std_plot, 
        fmt='o', 
        capsize=5, 
        markersize=8,
        color=sim_set["color"],
        label=sim_set["label"]
    )
    
    # Add fit line - linear fit for drag force
    if sim_set["name"] == "n256_tuned_damping":
        # Linear fit only for v=0.4 to v=0.6 range as in the notebook
        fdrag_mask = (velocities >= 0.4) & (velocities <= 0.6)
        fdrag_fit = np.polyfit(velocities[fdrag_mask], fdrag_avg[fdrag_mask], 1)
        x_fit_fdrag = np.linspace(0.4, 0.6, 100)
    else:
        # Linear fit for other datasets
        fdrag_fit = np.polyfit(velocities, fdrag_avg, 1)
        x_fit_fdrag = np.linspace(min(velocities), max(velocities), 100)
        
    fdrag_curve = np.polyval(fdrag_fit, x_fit_fdrag)
    ax1.plot(x_fit_fdrag, fdrag_curve, '--', linewidth=2, color=sim_set["color"])

ax1.set_xlabel('$v/c$', fontsize=16)
ax1.set_ylabel('$\\langle F_x \\rangle/\\langle \\rho \\rangle$', fontsize=16)
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='both', which='major', labelsize=16)

# 2. Torque vs Velocity
ax2 = fig.add_subplot(gs[0, 1])
for i, sim_set in enumerate(simulation_sets):
    data = processed_data[i]
    
    # For the first dataset, filter out v=0.7 data if required
    if sim_set["filter_v07"]:
        velocity_mask = data["velocities"] < 0.7
        velocities = data["velocities"][velocity_mask]
        torque_avg = data["torque_avg"][velocity_mask]
        torque_std = data["torque_std"][velocity_mask]
    else:
        velocities = data["velocities"]
        torque_avg = data["torque_avg"]
        torque_std = data["torque_std"]
    
    # Plot data points with error bars
    ax2.errorbar(
        velocities, 
        torque_avg, 
        yerr=torque_std, 
        fmt='o', 
        capsize=5, 
        markersize=8,
        color=sim_set["color"],
        label=sim_set["label"]
    )
    
    # Add quadratic fit for torque
    torque_fit = np.polyfit(velocities, torque_avg, 2)
    x_fit = np.linspace(min(velocities), max(velocities), 100)
    torque_curve = np.polyval(torque_fit, x_fit)
    ax2.plot(x_fit, torque_curve, '--', linewidth=2, color=sim_set["color"])

ax2.set_xlabel('$v/c$', fontsize=16)
ax2.set_ylabel('$\\langle \\tau_z  \\rangle / \\langle \\rho \\rangle$', fontsize=16)
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='both', which='major', labelsize=16)

# 3. Mass Accretion vs Velocity
ax3 = fig.add_subplot(gs[1, 0])
for i, sim_set in enumerate(simulation_sets):
    data = processed_data[i]
    
    # For the first dataset, filter out v=0.7 data if required
    if sim_set["filter_v07"]:
        velocity_mask = data["velocities"] < 0.7
        velocities = data["velocities"][velocity_mask]
        macc_avg = data["macc_avg"][velocity_mask]
        macc_std = data["macc_std"][velocity_mask]
    else:
        velocities = data["velocities"]
        macc_avg = data["macc_avg"]
        macc_std = data["macc_std"]
    
    # Plot data points with error bars
    ax3.errorbar(
        velocities, 
        macc_avg, 
        yerr=macc_std, 
        fmt='o', 
        capsize=5, 
        markersize=8,
        color=sim_set["color"],
        label=sim_set["label"]
    )
    
    # Add quadratic fit for mass accretion
    macc_fit = np.polyfit(velocities, macc_avg, 2)
    x_fit = np.linspace(min(velocities), max(velocities), 100)
    macc_curve = np.polyval(macc_fit, x_fit)
    ax3.plot(x_fit, macc_curve, '--', linewidth=2, color=sim_set["color"])

ax3.set_xlabel('$v/c$', fontsize=16)
ax3.set_ylabel('$\\langle \\dot{M}^{\\mathrm{acc}} \\rangle$', fontsize=16)
ax3.grid(True, alpha=0.3)
ax3.tick_params(axis='both', which='major', labelsize=16)

# 4. Charge Accretion vs Velocity
ax4 = fig.add_subplot(gs[1, 1])
for i, sim_set in enumerate(simulation_sets):
    data = processed_data[i]
    
    # For the first dataset, filter out v=0.7 data if required
    if sim_set["filter_v07"]:
        velocity_mask = data["velocities"] < 0.7
        velocities = data["velocities"][velocity_mask]
        qacc_avg = data["qacc_avg"][velocity_mask]
        qacc_std = data["qacc_std"][velocity_mask]
    else:
        velocities = data["velocities"]
        qacc_avg = data["qacc_avg"]
        qacc_std = data["qacc_std"]
    
    # Plot data points with error bars
    ax4.errorbar(
        velocities, 
        qacc_avg, 
        yerr=qacc_std, 
        fmt='o', 
        capsize=5, 
        markersize=8,
        color=sim_set["color"],
        label=sim_set["label"]
    )
    
    # Add quadratic fit for charge accretion
    qacc_fit = np.polyfit(velocities, qacc_avg, 2)
    x_fit = np.linspace(min(velocities), max(velocities), 100)
    qacc_curve = np.polyval(qacc_fit, x_fit)
    ax4.plot(x_fit, qacc_curve, '--', linewidth=2, color=sim_set["color"])

ax4.set_xlabel('$v/c$', fontsize=16)
ax4.set_ylabel('$-i\\langle \\dot{Q}^{\\mathrm{acc}} \\rangle$', fontsize=16)
ax4.grid(True, alpha=0.3)
ax4.tick_params(axis='both', which='major', labelsize=16)

# Add a single legend for the entire figure
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), 
           ncol=2, fontsize=16, frameon=True, fancybox=True, shadow=True)

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to accommodate the legend
plt.savefig("combined_quantities_vs_velocity.pdf", bbox_inches='tight')
plt.savefig("combined_quantities_vs_velocity.png", bbox_inches='tight', dpi=300)

print("Combined plot created successfully!") 