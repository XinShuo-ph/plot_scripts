import os
import numpy as np
from datetime import datetime
# Generate log-spaced outR values from 20.0 to 600.0
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--split_id', type=int, default=0, help='Split ID (0-3)')
# parser.add_argument('--simname', type=str, default="250528_BBH_r70_moreplots_restart", help='Name of simulation directory')
args = parser.parse_args()

# simname = args.simname
# simnames = [
#     '250520_BBH_r70_v04',
#     '250514_BBH_r70',
#     '250520_BBH_r70_v06',
#     '20250520_BBH_r70_v07_correctid',
#     '250125_BBH_r70_mu08_hires_longtime',
#     '250519_BBH_r70_mu005_box3200_lowres'
# ]

simnames = [ 
    '20250702_ncell416_v055_mu005_r35',
    '20250702_ncell416_v045_mu005_r35',
    '20250702_mu02_v03_r35',
    '20250702_ncell416_v06_mu005_r35',
    '20250702_ncell416_v07_mu005_r35',
    # '20250630_ncell416_v02_mu005_r35',
    # '20250630_ncell416_v03_mu005_r35',
    # '20250630_ncell416_v04_mu005_r35',
    # '20250630_ncell416_v05_mu005_r35',
    # '20250630_ncell416_v06_mu005',
    # '20250630_ncell416_v07_mu005',
    # '20250629_ncell320_v055_mu08',
    # '20250629_ncell320_v045_mu08',
    # '20250628_ncell320_v04_mu08',
    # '20250628_ncell320_v07_mu08',
    # '20250628_ncell320_v06_mu08',
    # '20250627_ncell256_v05_mu08'
    # '20250628_mu02_v045_r35',
    # '20250628_mu02_v055_r35',
    # '20250627_mu02_v04_r35',
    # '20250627_mu02_v05_r35',
    # '20250627_mu02_v06_r35',
    # '20250627_mu02_v07_r35'
    # '20250620_tune_damping_n256_1',
    # '20250626_n256_v035',
    # '20250623_tune_damping_n256_v055',
    # '20250623_tune_damping_n256_v045',
    # '20250621_tune_damping_n192',
    # '20250621_tune_damping_n128',
    # '20250622_tune_damping_n256_v03',
    # '20250622_tune_damping_n256_v04',
    # '20250620_tune_damping_n256_1',
    # '20250621_tune_damping_n256_v06',
    # '20250622_tune_damping_n256_v07',
]

# all_outR_values = np.logspace(np.log10(100.0), np.log10(320.0), 8)
all_outR_values = [100.0,320.0, 640.0]
# all_outR_values = [50.0,80.0,100.0,140.0,150.0,160.0,200.0]
# Use interleaved assignment - take every 4th value starting at split_id
# outR_values = all_outR_values[args.split_id::4]
outR_values = all_outR_values
print(outR_values)

for simname in simnames:
    for outR in outR_values:
        # cmd = f"bash run_parallel.sh --simname {simname} --total_workers 128 --maxframes 10000 --outR {outR:.1f} --fix_metric_error > plotlog{args.split_id}.txt 2>&1"
        cmd = f"bash run_parallel.sh --withQ --simname {simname} --total_workers 64 --maxframes 10000 --outR {outR:.1f} > plotlog{args.split_id}.txt 2>&1"
        print(f"Running with outR = {outR:.1f} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("command is", cmd)
        os.system(cmd)