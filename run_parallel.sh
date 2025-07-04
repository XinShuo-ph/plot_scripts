#!/bin/bash

# Function to format time in seconds to hours, minutes, seconds
format_time() {
    local seconds=$1
    local hours=$((seconds / 3600))
    local minutes=$(( (seconds % 3600) / 60 ))
    local secs=$((seconds % 60))
    printf "%02d:%02d:%02d" $hours $minutes $secs
}

# Record start time for the entire process
total_start_time=$(date +%s)

# Parse command line arguments with defaults
while [[ $# -gt 0 ]]; do
  case $1 in
    --simname)
      simname="$2"
      shift 2
      ;;
    --outR)
      outR="$2"
      shift 2
      ;;
    --total_workers)
      total_workers="$2"
      shift 2
      ;;
    --excise_factor)
      excise_factor="$2"
      shift 2
      ;;
    --maxframes)
      maxframes="$2"
      shift 2
      ;;
    --skipevery)
      skipevery="$2"
      shift 2
      ;;
    --innerR)
      innerR="$2"
      shift 2
      ;;
    --outplot)
      outplot="true"
      shift
      ;;
    --fix_metric_error)
      fix_metric_error="true"
      shift
      ;;
    --psipow_volume)
      psipow_volume="$2"
      shift 2
      ;;
    --psipow_surface)
      psipow_surface="$2"
      shift 2
      ;;
    --temp_output)
      temp_output="true"
      shift
      ;;
    --withQ)
      withQ="true"
      shift
      ;;
    --bbh1_x)
      bbh1_x="$2"
      shift 2
      ;;
    --bbh2_x)
      bbh2_x="$2"
      shift 2
      ;;
    --bbh1_r)
      bbh1_r="$2"
      shift 2
      ;;
    --bbh2_r)
      bbh2_r="$2"
      shift 2
      ;;
    --binary_omega)
      binary_omega="$2"
      shift 2
      ;;
    --parallel_out_suffix)
      parallel_out_suffix="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Set default values for parameters not specified
simname=${simname:-"250514_BBH_r70"}
outR=${outR:-"320.0"}
total_workers=${total_workers:-"4"}
excise_factor=${excise_factor:-"1.5"}
maxframes=${maxframes:-"1000"}
skipevery=${skipevery:-"1"}
innerR=${innerR:-"-1"}
outplot=${outplot:-"false"}
fix_metric_error=${fix_metric_error:-"false"}
psipow_volume=${psipow_volume:-"2"}
psipow_surface=${psipow_surface:-"-2"}
temp_output=${temp_output:-"false"}
withQ=${withQ:-"false"}
parallel_out_suffix=${parallel_out_suffix:-"parallel"}

# Function to convert boolean string to flag
get_flag() {
    if [[ "$1" == "true" ]]; then
        echo "--$2"
    else
        echo ""
    fi
}

# Convert boolean strings to flags
outplot_flag=$(get_flag "$outplot" "outplot")
fix_metric_error_flag=$(get_flag "$fix_metric_error" "fix_metric_error")
temp_output_flag=$(get_flag "$temp_output" "temp_output")
withQ_flag=$(get_flag "$withQ" "withQ")

# Function to add an argument to command if the variable is defined
add_arg() {
    local arg_name="$1"
    local arg_value="$2"
    
    if [[ -n "$arg_value" ]]; then
        echo "--$arg_name $arg_value"
    else
        echo ""
    fi
}

# Function to run a script with multiple workers
run_parallel() {
    local script=$1
    local simname=$2
    local outR=$3
    local total_workers=$4
    local excise_factor=$5
    local maxframes=$6
    local type=$7
    local psipow=$8
    
    # Record start time for this script
    local script_start_time=$(date +%s)
    
    echo "Running $script with $total_workers workers..."
    echo "Parameters: simname=$simname, outR=$outR, excise_factor=$excise_factor, maxframes=$maxframes"
    
    # Create an array of commands for parallel execution
    commands=()
    for ((i=0; i<total_workers; i++)); do
        # Start with required arguments
        cmd="python $script --worker_id $i --total_workers $total_workers --simname $simname --outR $outR --excise_factor $excise_factor --maxframes $maxframes"
        
        # Add optional arguments only if they were provided
        cmd="$cmd $(add_arg "skipevery" "$skipevery")"
        cmd="$cmd $(add_arg "innerR" "$innerR")"
        cmd="$cmd $(add_arg "psipow" "$psipow")"
        
        # Add BH parameters only if they were provided
        cmd="$cmd $(add_arg "bbh1_x" "$bbh1_x")"
        cmd="$cmd $(add_arg "bbh2_x" "$bbh2_x")"
        cmd="$cmd $(add_arg "bbh1_r" "$bbh1_r")"
        cmd="$cmd $(add_arg "bbh2_r" "$bbh2_r")"
        cmd="$cmd $(add_arg "binary_omega" "$binary_omega")"
        
        # Add type-specific arguments
        if [[ "$type" == "volume" ]]; then
            cmd="$cmd $fix_metric_error_flag"
        fi
        
        # Add common boolean flags
        cmd="$cmd $outplot_flag $temp_output_flag $withQ_flag"
        
        # Trim extra spaces
        cmd=$(echo "$cmd" | tr -s ' ')
        
        commands+=("$cmd")
    done
    
    # Run commands in parallel with GNU parallel
    echo "Executing commands:"
    for cmd in "${commands[@]}"; do
        echo "  $cmd"
    done
    
    # Record worker execution start time
    local worker_start_time=$(date +%s)
    parallel --ungroup --jobs 0 '{}' ::: "${commands[@]}"
    local worker_end_time=$(date +%s)
    local worker_elapsed=$((worker_end_time - worker_start_time))
    echo "Workers execution time: $(format_time $worker_elapsed)"
    
    # Merge results when all workers complete
    out_suffix="$parallel_out_suffix"
    if [[ "$out_suffix" != "" ]]; then
        out_suffix="$out_suffix"
    fi
    
    # Record merge start time
    local merge_start_time=$(date +%s)
    merge_cmd="python merge_parallel_results.py --simname $simname --outR $outR --excise_factor $excise_factor --type $type --total_workers $total_workers --out_suffix $out_suffix"

    # add psipow_volume or psipow_surface to the merge command
    if [[ "$type" == "volume" ]]; then
        merge_cmd="$merge_cmd $(add_arg "psipow_volume" "$psipow_volume")"
    else
        merge_cmd="$merge_cmd $(add_arg "psipow_surface" "$psipow_surface")"
        # merge_cmd="$merge_cmd $(add_arg "psipow_surface_correction" "$psipow_surface_correction")"
    fi

    echo "Merging results: $merge_cmd"
    eval $merge_cmd
    local merge_end_time=$(date +%s)
    local merge_elapsed=$((merge_end_time - merge_start_time))
    echo "Merge execution time: $(format_time $merge_elapsed)"
    
    # Calculate total script time
    local script_end_time=$(date +%s)
    local script_elapsed=$((script_end_time - script_start_time))
    echo "Completed $script processing in $(format_time $script_elapsed)"
    echo "----------------------------------------------------------------------"
}

# Create working directory if needed
cd "$(dirname "$0")"
mkdir -p tmp

echo "Starting parallel processing for $simname with $total_workers workers"
echo "----------------------------------------------------------------------"
echo "Volume processing settings:"
echo "  Fix metric error: $fix_metric_error"
echo "  psipow: $psipow_volume"
echo "----------------------------------------------------------------------"
echo "Surface processing settings:"
echo "  psipow: $psipow_surface"
echo "----------------------------------------------------------------------"
echo "Common settings:"
echo "  Skip every: $skipevery"
echo "  Inner radius: $innerR"
echo "  Output plots: $outplot"
echo "  Temporary output: $temp_output"
echo "  Process noether charge Q: $withQ"

# Print BH parameters only if they were provided
if [[ -n "$bbh1_x" ]]; then
    echo "  BH1 x: $bbh1_x"
fi
if [[ -n "$bbh2_x" ]]; then
    echo "  BH2 x: $bbh2_x"
fi
if [[ -n "$bbh1_r" ]]; then
    echo "  BH1 r: $bbh1_r"
fi
if [[ -n "$bbh2_r" ]]; then
    echo "  BH2 r: $bbh2_r"
fi
if [[ -n "$binary_omega" ]]; then
    echo "  Binary omega: $binary_omega"
fi
echo "----------------------------------------------------------------------"

# Process surface integrals
run_parallel "test_integrate_2d_surface_parallel.py" "$simname" "$outR" "$total_workers" "$excise_factor" "$maxframes" "surface" "$psipow_surface"
# Process volume integrals
run_parallel "test_integrate_2d_parallel.py" "$simname" "$outR" "$total_workers" "$excise_factor" "$maxframes" "volume" "$psipow_volume"


# Calculate total execution time
total_end_time=$(date +%s)
total_elapsed=$((total_end_time - total_start_time))
echo "All processing complete!"
echo "----------------------------------------------------------------------"
echo "  Total execution time: $(format_time $total_elapsed)"
echo "----------------------------------------------------------------------"

# Cleanup worker files
# rm -f ${simname}_2d_integrals_outR${outR}_excise${excise_factor}_worker*.npy
# rm -f ${simname}_2d_integrals_surface_outR${outR}_excise${excise_factor}_worker*.npy
rm -f *worker*.npy
echo "Worker files deleted" 