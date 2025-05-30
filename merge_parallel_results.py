import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import glob

parser = argparse.ArgumentParser(description='Merge results from parallel workers.')
parser.add_argument('--simname', type=str, default="240930_BBH_3D_zboost_v5", help='Name of the simulation directory')
parser.add_argument('--outR', type=float, default=650, help='outer radius of the integration sphere')
parser.add_argument('--excise_factor', type=float, default=1.5, help='factor to excise the black hole')
parser.add_argument('--type', type=str, default="volume", choices=["volume", "surface"], help='Type of data to merge (volume or surface)')
parser.add_argument('--total_workers', type=int, default=4, help='Total number of workers used')
parser.add_argument('--out_suffix', type=str, default='', help='Suffix to add to output files')

args = parser.parse_args()
simname = args.simname
outR = args.outR
excise_factor = args.excise_factor
data_type = args.type
total_workers = args.total_workers
out_suffix = args.out_suffix

# Add suffix to output filename if provided
suffix = f"_{out_suffix}" if out_suffix else ""

# Determine file pattern based on data type
if data_type == "volume":
    file_pattern = f"{simname}_2d_integrals_outR{outR}_excise{excise_factor}_worker*.npy"
    output_file = f"{simname}_2d_integrals_outR{outR}_excise{excise_factor}{suffix}.npy"
    plot_file1 = f"{simname}_2d_volume_integrals_outR{outR}_excise{excise_factor}{suffix}.png"
    plot_file2 = f"{simname}_2d_momentum_integrals_outR{outR}_excise{excise_factor}{suffix}.png"
else:
    file_pattern = f"{simname}_2d_integrals_surface_outR{outR}_excise{excise_factor}_worker*.npy"
    output_file = f"{simname}_2d_integrals_surface_outR{outR}_excise{excise_factor}{suffix}.npy"
    plot_file1 = f"{simname}_2d_surface_integrals_outR{outR}{suffix}.png"
    plot_file2 = f"{simname}_2d_surface_integrals_bh_excise{excise_factor}{suffix}.png"

# Find all worker result files
worker_files = glob.glob(file_pattern)
print(f"Found {len(worker_files)} worker files matching pattern: {file_pattern}")

if len(worker_files) == 0:
    print("No worker files found. Check the file pattern and directory.")
    exit(1)

# Collect results from all workers
all_results = []
for worker_file in worker_files:
    worker_results = np.load(worker_file)
    all_results.append(worker_results)
    print(f"Loaded {worker_file} with {len(worker_results)} frames")

# Concatenate all results
combined_results = np.concatenate(all_results, axis=0)

# Sort by time
combined_results = combined_results[combined_results[:, 0].argsort()]

# Save the combined results
np.save(output_file, combined_results)
print(f"Saved combined results to {output_file} with {len(combined_results)} total frames")

# Create plots based on data type
if data_type == "volume":
    plt.figure()
    plt.plot(combined_results[:,0], combined_results[:,1], label='VOLUME_X')
    plt.plot(combined_results[:,0], combined_results[:,2], label='VOLUME_Y')
    plt.plot(combined_results[:,0], combined_results[:,3], label='VOLUME_Z')
    plt.plot(combined_results[:,0], combined_results[:,7], label='TORQUE')
    plt.xlabel('Time')
    plt.ylabel('2D Integral')
    plt.legend()
    plt.savefig(plot_file1)
    plt.close()

    plt.figure()
    plt.plot(combined_results[:,0], combined_results[:,4], label='SMOMENTUM_X')
    plt.plot(combined_results[:,0], combined_results[:,5], label='SMOMENTUM_Y')
    plt.plot(combined_results[:,0], combined_results[:,6], label='SMOMENTUM_Z')
    plt.plot(combined_results[:,0], combined_results[:,8], label='ANGULAR MOMENTUM')
    plt.xlabel('Time')
    plt.ylabel('2D Integral')
    plt.legend()
    plt.savefig(plot_file2)
    plt.close()
else:
    plt.figure()
    plt.plot(combined_results[:,0], combined_results[:,1], label='SURFACE_X')
    plt.plot(combined_results[:,0], combined_results[:,2], label='SURFACE_Y')
    plt.plot(combined_results[:,0], combined_results[:,3], label='SURFACE_Z')
    plt.plot(combined_results[:,0], combined_results[:,10], label='rhoavg')
    plt.plot(combined_results[:,0], combined_results[:,11], label='sur_torque')
    plt.xlabel('Time')
    plt.ylabel('2D Integral')
    plt.legend()
    plt.savefig(plot_file1)
    plt.close()

    plt.figure()
    plt.plot(combined_results[:,0], combined_results[:,4], label='SURFACE_X BH1')
    plt.plot(combined_results[:,0], combined_results[:,5], label='SURFACE_Y BH1')
    plt.plot(combined_results[:,0], combined_results[:,6], label='SURFACE_Z BH1')
    plt.plot(combined_results[:,0], combined_results[:,7], label='SURFACE_X BH2')
    plt.plot(combined_results[:,0], combined_results[:,8], label='SURFACE_Y BH2')
    plt.plot(combined_results[:,0], combined_results[:,9], label='SURFACE_Z BH2')
    plt.plot(combined_results[:,0], combined_results[:,12], label='sur_bh1_torque')
    plt.plot(combined_results[:,0], combined_results[:,13], label='sur_bh2_torque')
    plt.xlabel('Time')
    plt.ylabel('2D Integral')
    plt.legend()
    plt.savefig(plot_file2)
    plt.close()

print(f"Created plots: {plot_file1} and {plot_file2}")

# Cross-check with reference file if exists
reference_file = None
if data_type == "volume":
    reference_pattern = f"/pscratch/sd/x/xinshuo/plotGReX/{simname}_2d_integrals_outR{outR}_excise{excise_factor}.npy"
else:
    reference_pattern = f"/pscratch/sd/x/xinshuo/plotGReX/{simname}_2d_integrals_surface_outR{outR}_excise{excise_factor}.npy"

reference_files = glob.glob(reference_pattern)
if reference_files:
    reference_file = reference_files[0]
    print(f"Found reference file: {reference_file}")
    
    try:
        reference_data = np.load(reference_file)
        print(f"Reference data shape: {reference_data.shape}")
        print(f"Combined data shape: {combined_results.shape}")
        
        # Compare number of frames
        if len(reference_data) == len(combined_results):
            print("✓ Number of frames match")
        else:
            print(f"✗ Frame count mismatch: Reference {len(reference_data)} vs Combined {len(combined_results)}")
            # if mismatch, use  only the frames that are common to compare
            common_frames = np.intersect1d(reference_data[:,0], combined_results[:,0])
            print(f"Common frames: {common_frames}")
            reference_data = reference_data[reference_data[:,0].argsort()]
            combined_results = combined_results[combined_results[:,0].argsort()]
            reference_data = reference_data[np.isin(reference_data[:,0], common_frames)]
            combined_results = combined_results[np.isin(combined_results[:,0], common_frames)]
        
        # Compare time values
        time_diff = np.abs(reference_data[:,0] - combined_results[:,0])
        if np.all(time_diff < 1e-10):
            print("✓ Time values match")
        else:
            print(f"✗ Time values differ: Max difference {np.max(time_diff)}")
            
        # Compare data values (all columns except time)
        if reference_data.shape[1] == combined_results.shape[1]:
            for col in range(1, reference_data.shape[1]):
                data_diff = np.abs(reference_data[:,col] - combined_results[:,col])
                max_diff = np.max(data_diff)
                if np.all(data_diff < 1e-10):
                    print(f"✓ Column {col} matches perfectly")
                elif np.all(data_diff < 1e-5):
                    print(f"✓ Column {col} matches within numerical precision (max diff: {max_diff})")
                else:
                    print(f"✗ Column {col} differs: Max difference {max_diff}")
                    
            print("\nSummary of differences:")
            max_rel_diff = np.max(np.abs((reference_data[:,1:] - combined_results[:,1:]) / 
                                 (np.abs(reference_data[:,1:]) + 1e-10)))
            print(f"Maximum relative difference: {max_rel_diff}")
            
            if max_rel_diff < 1e-5:
                print("✓ Results match within expected numerical precision")
            else:
                print("✗ Significant differences found between reference and combined results")
        else:
            print(f"✗ Column count mismatch: Reference {reference_data.shape[1]} vs Combined {combined_results.shape[1]}")
    except Exception as e:
        print(f"Error comparing with reference file: {e}")
else:
    print("No reference file found for comparison.") 