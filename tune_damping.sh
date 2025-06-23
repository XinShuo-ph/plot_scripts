#!/bin/bash

# Check if cell count is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <ncell>"
    echo "  <ncell> - Number of cells (for amr.n_cell parameter)"
    exit 1
fi

NCELL=$1

# Configuration
GREX_DIR="/pscratch/sd/x/xinshuo/GReX"
PROBLEM_DIR="${GREX_DIR}/Problems/FUKA_BBH_ScalarField_boost_r70"
RUNGREX_DIR="/pscratch/sd/x/xinshuo/runGReX"
SOURCE_DIR="/pscratch/sd/x/xinshuo/runGReX/250524_BBH_r70_excise099_lowres5"
TARGET_AVG=8.0e-4
TARGET_TOL=2.0e-5
MIN_DAMPING=0.1
MAX_DAMPING=1.0
TMPFILE="/pscratch/sd/x/xinshuo/plotGReX/tmp/rho_avg_results.txt"
DATE_PREFIX=$(date +%Y%m%d)
SCRIPTDIR="/pscratch/sd/x/xinshuo/plotGReX/tmp"

# Create temporary script directory
mkdir -p "$SCRIPTDIR"

# Create a shared directory for common files
SHARED_DIR="${RUNGREX_DIR}/${DATE_PREFIX}_tune_damping_n${NCELL}_shared"
LATEST_CHK_FILE="" # Will store path to the latest checkpoint file

# Initial damping factor value
damping_factor=0.5

# Function to request allocation in "run" screen
request_allocation() {
    echo "Requesting compute allocation..."
    
    # Check if "run" screen exists, create if not
    screen -list | grep -q "run"
    if [ $? -ne 0 ]; then
        screen -dmS run
    fi
    
    # Create a temporary script file for allocation
    cat > "$SCRIPTDIR/alloc.sh" << EOF
#!/bin/bash
salloc -N 4 -C gpu -q interactive -t 4:00:00 --gres=gpu:4 -A m4575_g
EOF
    chmod +x "$SCRIPTDIR/alloc.sh"
    
    # Run the allocation script in the screen
    screen -S run -X stuff "$SCRIPTDIR/alloc.sh$(printf \\r)"
    
    echo "Allocation request sent to 'run' screen."
    echo "Waiting for allocation to be granted..."
    
    # Wait for allocation to be granted
    allocated=false
    for i in {1..60}; do
        sleep 10
        # Get job status
        job_status=$(squeue -u $(whoami) | grep interact | awk '{print $5}')
        if [ "$job_status" = "R" ]; then
            allocated=true
            echo "Allocation granted and running."
            break
        fi
        echo "Still waiting for allocation... ($i/60)"
    done
    
    if [ "$allocated" = false ]; then
        echo "Failed to get allocation after 10 minutes. Exiting."
        exit 1
    fi
    
    # Get the last node in the allocation for "plot" screen - FIXED to handle node ranges
    node_list=$(squeue -u $(whoami) -o "%N" | grep -v NODELIST | head -1)
    echo "Node list: $node_list"
    
    # Extract a single valid node name from potentially complex formats like nid[001096-001097]
    if [[ $node_list =~ nid\[([0-9]+)-([0-9]+)\] ]]; then
        # Format is nid[start-end]
        last_node="nid${BASH_REMATCH[2]}"
        echo "Extracted last node from range: $last_node"
    elif [[ $node_list =~ nid([0-9,\-]+) ]]; then
        # Format is nidXXX,nidYYY or similar
        last_part=$(echo "$node_list" | tr ',' '\n' | tail -1)
        if [[ $last_part =~ nid([0-9]+) ]]; then
            last_node="$last_part"
        else
            last_node="$node_list"
        fi
        echo "Extracted last node from list: $last_node"
    else
        # Just use as is if we don't recognize the format
        last_node="$node_list"
        echo "Using node as is: $last_node"
    fi
    
    echo "Last node in allocation: $last_node"
    
    # Direct "plot" screen to the last node
    # Check if "plot" screen exists, create if not
    screen -list | grep -q "plot"
    if [ $? -ne 0 ]; then
        screen -dmS plot
    fi
    
    # Create a temporary ssh script
    cat > "$SCRIPTDIR/ssh_node.sh" << EOF
#!/bin/bash
ssh $last_node
EOF
    chmod +x "$SCRIPTDIR/ssh_node.sh"
    
    # Send SSH command to "plot" screen
    screen -S plot -X stuff "$SCRIPTDIR/ssh_node.sh$(printf \\r)"
    echo "Directed 'plot' screen to $last_node"
    
    # Return the last node
    echo "$last_node"
}

# Function to modify source_damping_factor in params.hh
update_params() {
    local damping=$1
    echo "Setting source_damping_factor to $damping"
    sed -i "s/double source_damping_factor = [0-9.]\+e\{0,1\}-\{0,1\}[0-9]*;/double source_damping_factor = ${damping};/" "${GREX_DIR}/Source/scalar_field/params.hh"
}

# Function to compile the code in the "run" screen
compile_code() {
    echo "Compiling code in the 'run' screen..."
    
    # Create a temporary compile script
    cat > "$SCRIPTDIR/compile.sh" << EOF
#!/bin/bash
cd "$PROBLEM_DIR" && make -j128
echo "Compilation finished at $(date)" > /tmp/compile_finished
EOF
    chmod +x "$SCRIPTDIR/compile.sh"
    
    # Clear any existing compile finished marker
    rm -f /tmp/compile_finished
    
    # Send to run screen
    screen -S run -X stuff "^C$(printf \\r)"  # Send Ctrl+C to stop any current process
    sleep 2
    screen -S run -X stuff "$SCRIPTDIR/compile.sh$(printf \\r)"
    
    echo "Compilation command sent. Waiting for compilation to finish..."
    
    # Wait for compilation to finish by watching for the finished marker
    local max_wait=300  # 5 minutes
    local success=false
    
    for ((i=1; i<=max_wait; i++)); do
        if [ -f "/tmp/compile_finished" ]; then
            success=true
            break
        fi
        
        # Wait 1 second between checks
        sleep 1
        
        # Every 10 seconds, print a waiting message
        if [ $((i % 10)) -eq 0 ]; then
            echo "Still waiting for compilation to finish... ($i seconds)"
        fi
    done
    
    if [ "$success" = true ]; then
        echo "Compilation successful."
        sleep 5  # Give a little extra time for file operations to complete
    else
        echo "Compilation may have failed or timed out after $max_wait seconds."
        echo "Check the 'run' screen for details. Continuing with the process..."
    fi
}

# Function to setup shared directory with common input files (only done once)
setup_shared_dir() {
    if [ ! -d "$SHARED_DIR" ]; then
        echo "Setting up shared directory: $SHARED_DIR"
        mkdir -p "$SHARED_DIR"
        
        # Copy input files to shared directory
        cp "${SOURCE_DIR}/BBH_ECC_RED.70.4976.0.3.0.3.2.73552.q0.85.0.0.09.dat" "$SHARED_DIR/"
        cp "${SOURCE_DIR}/BBH_ECC_RED.70.4976.0.3.0.3.2.73552.q0.85.0.0.09.info" "$SHARED_DIR/"
        cp "${SOURCE_DIR}/inputs" "$SHARED_DIR/inputs.original"
        
        # Modify the copied input file to use the specified cell count
        sed "s/amr\.n_cell\s*=\s*[0-9]\+ [0-9]\+ [0-9]\+/amr.n_cell           =  $NCELL $NCELL $NCELL/" \
            "$SHARED_DIR/inputs.original" > "$SHARED_DIR/inputs"
        
        echo "Shared directory setup complete."
    else
        echo "Shared directory already exists: $SHARED_DIR"
    fi
}

# Function to setup a new run directory
setup_run_dir() {
    local run_id=$1
    local run_dir="${RUNGREX_DIR}/${DATE_PREFIX}_tune_damping_n${NCELL}_${run_id}"
    
    echo "Setting up run directory: $run_dir"
    mkdir -p "$run_dir"
    
    # Copy executable from problem directory
    echo "Copying executable to run directory..."
    cp "${PROBLEM_DIR}/main2d.gnu.TPROF.MPI.OMP.CUDA.ex" "$run_dir/"
    
    # Create symbolic links to common input files from the shared directory
    ln -sf "${SHARED_DIR}/BBH_ECC_RED.70.4976.0.3.0.3.2.73552.q0.85.0.0.09.dat" "$run_dir/"
    ln -sf "${SHARED_DIR}/BBH_ECC_RED.70.4976.0.3.0.3.2.73552.q0.85.0.0.09.info" "$run_dir/"
    
    # For first iteration, use original inputs; for subsequent iterations, use checkpoint
    if [ "$run_id" = "1" ] || [ -z "$LATEST_CHK_FILE" ]; then
        echo "First run: Not using checkpoint"
        cp "$SHARED_DIR/inputs" "$run_dir/inputs"
    else
        echo "Using checkpoint from previous run: $LATEST_CHK_FILE"
        # Create inputs file with restart option
        grep -v "amr\.restart" "$SHARED_DIR/inputs" > "$run_dir/inputs"
        echo "amr.restart = $LATEST_CHK_FILE" >> "$run_dir/inputs"
    fi
    
    echo "Run directory setup complete."
    
    # Return the run directory path
    echo "$run_dir"
}

# Function to start a run in the "run" screen
start_run() {
    local run_dir=$1
    
    # Create a temporary run script
    cat > "$SCRIPTDIR/run.sh" << EOF
#!/bin/bash
cd "$run_dir"
export CRAY_ACCEL_TARGET=nvidia80
export MPICH_GPU_SUPPORT_ENABLED=1
export CRAY_ACCEL_TARGET=nvidia80
export AMREX_CUDA_ARCH=8.0
export MPICH_GPU_SUPPORT_ENABLED=0
export SLURM_CPU_BIND="cores"
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
srun -n 16 --gpus-per-task=1 --ntasks-per-node=4 -c 32 --gpu-bind=none ./main2d.gnu.TPROF.MPI.OMP.CUDA.ex inputs > log.txt 2>&1
EOF
    chmod +x "$SCRIPTDIR/run.sh"
    
    # Send to run screen
    screen -S run -X stuff "^C$(printf \\r)"  # Send Ctrl+C to stop any current process
    sleep 2
    screen -S run -X stuff "$SCRIPTDIR/run.sh$(printf \\r)"
    
    echo "Run started in 'run' screen."
}

# Function to monitor results in the "plot" screen
monitor_results() {
    local run_dir=$1
    local simname=$(basename "$run_dir")
    
    # Create a temporary monitor script
    cat > "$SCRIPTDIR/monitor.sh" << EOF
#!/bin/bash
cd /pscratch/sd/x/xinshuo
python plotGReX/profile_rho_line.py --simname $simname --tmpfile $TMPFILE --outplot
watch -n 60 "python plotGReX/profile_rho_line.py --simname $simname --tmpfile $TMPFILE"
EOF
    chmod +x "$SCRIPTDIR/monitor.sh"
    
    # Send to plot screen
    screen -S plot -X stuff "^C$(printf \\r)"  # Send Ctrl+C to stop any current process
    sleep 2
    screen -S plot -X stuff "$SCRIPTDIR/monitor.sh$(printf \\r)"
    
    echo "Monitoring started in 'plot' screen."
}

# Function to check results and adjust damping factor
check_results() {
    # Check if the temporary file exists
    if [ ! -f "$TMPFILE" ]; then
        echo "Results file not found: $TMPFILE"
        return 1
    fi
    
    # Read the average after t=200
    avg_after_t200=$(grep "avg_after_t200:" "$TMPFILE" | awk '{print $2}')
    final_time=$(grep "final_time:" "$TMPFILE" | awk '{print $2}')
    
    echo "Final time: $final_time"
    echo "Average RHO_ENERGY after t=200: $avg_after_t200"
    
    # Check if final time is greater than or equal to 250
    if (( $(echo "$final_time < 250" | bc -l) )); then
        echo "Not enough data yet (t=$final_time < 250)"
        return 1
    fi
    
    # Calculate the difference from the target
    diff=$(echo "$avg_after_t200 - $TARGET_AVG" | bc -l)
    abs_diff=$(echo "if ($diff < 0) -1 * $diff else $diff" | bc -l)
    
    echo "Difference from target: $diff (absolute: $abs_diff)"
    
    # Check if we're within tolerance
    if (( $(echo "$abs_diff <= $TARGET_TOL" | bc -l) )); then
        echo "SUCCESS! Target average achieved within tolerance."
        echo "Final source_damping_factor: $damping_factor"
        return 0
    fi
    
    # Adjust damping factor
    if (( $(echo "$diff > 0" | bc -l) )); then
        # Current value is too high, increase damping
        new_damping=$(echo "$damping_factor + $damping_factor * 0.2" | bc -l)
        if (( $(echo "$new_damping > $MAX_DAMPING" | bc -l) )); then
            new_damping=$MAX_DAMPING
        fi
    else
        # Current value is too low, decrease damping
        new_damping=$(echo "$damping_factor - $damping_factor * 0.2" | bc -l)
        if (( $(echo "$new_damping < $MIN_DAMPING" | bc -l) )); then
            new_damping=$MIN_DAMPING
        fi
    fi
    
    # Store the new damping factor for the next iteration
    damping_factor=$new_damping
    
    echo "Adjusting damping factor to: $damping_factor"
    return 2
}

# Function to find the latest checkpoint file
find_latest_checkpoint() {
    local run_dir=$1
    
    # Find most recent checkpoint file
    local latest_chk_dir=$(find "$run_dir" -name "chk*" -type d -printf '%T+ %p\n' 2>/dev/null | sort -r | head -1 | cut -d' ' -f2-)
    
    if [ -n "$latest_chk_dir" ]; then
        echo "Found latest checkpoint directory: $latest_chk_dir"
        LATEST_CHK_FILE=$(basename "$latest_chk_dir")
        echo "Latest checkpoint file: $LATEST_CHK_FILE"
    else
        echo "No checkpoint files found in $run_dir"
        LATEST_CHK_FILE=""
    fi
}

# Enhanced function to stop the run
stop_run() {
    local run_dir=$1
    
    echo "Stopping run..."
    
    # Create a temporary stop script
    cat > "$SCRIPTDIR/stop.sh" << EOF
#!/bin/bash
# Send Ctrl+C multiple times
echo "Stopping processes..."
killall -15 srun 2>/dev/null
sleep 2
# Force kill if needed
killall -9 srun 2>/dev/null
# Remove executable
rm -f "$run_dir/main2d.gnu.TPROF.MPI.OMP.CUDA.ex"
echo "All processes stopped at $(date)" > /tmp/run_stopped
EOF
    chmod +x "$SCRIPTDIR/stop.sh"
    
    # Clear any existing stop finished marker
    rm -f /tmp/run_stopped
    
    # Send multiple Ctrl+C to the run screen, then run the stop script
    for i in {1..5}; do
        screen -S run -X stuff "^C$(printf \\r)"
        sleep 1
    done
    
    # Run the stop script
    screen -S run -X stuff "$SCRIPTDIR/stop.sh$(printf \\r)"
    
    # Wait for stop to complete
    for i in {1..30}; do
        if [ -f "/tmp/run_stopped" ]; then
            break
        fi
        sleep 1
    done
    
    # Find latest checkpoint before stopping
    find_latest_checkpoint "$run_dir"
    
    echo "Run stopped."
}

# Create tmp directory if it doesn't exist
mkdir -p "$(dirname "$TMPFILE")"

# Start with allocation
echo "==============================================="
echo "SETUP: REQUESTING ALLOCATION"
echo "==============================================="
request_allocation

# Setup shared directory first
setup_shared_dir

# Main tuning loop
iteration=1
max_iterations=20

while [ $iteration -le $max_iterations ]; do
    echo "==============================================="
    echo "ITERATION $iteration"
    echo "==============================================="
    
    # Update parameters with current damping factor
    update_params $damping_factor
    
    # Create a file to mark compilation start time
    touch /tmp/compile_start
    
    # Compile the code in the run screen
    compile_code
    
    # Setup a new run directory
    run_dir=$(setup_run_dir $iteration)
    
    # Start the run
    start_run "$run_dir"
    
    # Start monitoring
    monitor_results "$run_dir"
    
    echo "Waiting for simulation to reach t=250..."
    
    # Wait and periodically check results
    reached_target=false
    
    for i in {1..30}; do  # Wait up to 30 minutes
        echo "Sleeping for 1 minute..."
        sleep 60
        
        echo "Checking results..."
        check_results
        status=$?
        
        if [ $status -eq 0 ]; then
            # Target achieved
            reached_target=true
            break
        elif [ $status -eq 2 ]; then
            # Need to adjust damping factor and retry
            break
        fi
        # Otherwise, status=1 means not enough data yet, continue waiting
    done
    
    # Stop the current run and get latest checkpoint
    stop_run "$run_dir"
    
    if [ "$reached_target" = true ]; then
        echo "==============================================="
        echo "Target achieved! Final damping factor: $damping_factor"
        echo "==============================================="
        break
    fi
    
    iteration=$((iteration+1))
    
    if [ $iteration -gt $max_iterations ]; then
        echo "==============================================="
        echo "Maximum iterations reached without convergence."
        echo "Last damping factor: $damping_factor"
        echo "==============================================="
    fi
done

# Final summary
echo "Tuning process complete."
echo "Final source_damping_factor: $damping_factor"
echo "Check $TMPFILE for final results." 