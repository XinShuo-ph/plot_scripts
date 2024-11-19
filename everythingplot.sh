#!/bin/bash

# Get the simulation name from command line argument
if [ -z "$1" ]; then
    echo "Usage: $0 <simulation_name>"
    exit 1
fi
simname=$1

# Open a screen session for each script
screen -dm bash -c "
    cd \$MYPLOTDIR
    bash plot_metric_xz.sh $simname
"

screen -dm bash -c "
    cd \$MYPLOTDIR
    bash plot_metric.sh $simname
"


screen -dm bash -c "
    cd \$MYPLOTDIR
    bash plot_momentum_xz.sh $simname
"

screen -dm bash -c "
    cd \$MYPLOTDIR
    bash plot_momentum.sh $simname
"

screen -dm bash -c "
    cd \$MYPLOTDIR
    bash plot_rho_xz.sh $simname
"

screen -dm bash -c "
    cd \$MYPLOTDIR
    bash plot_rho.sh $simname
"

screen -dm bash -c "
    cd \$MYPLOTDIR
    bash plot_surface_int_xz.sh $simname
"

screen -dm bash -c "
    cd \$MYPLOTDIR
    bash plot_surface_int.sh $simname
"

screen -dm bash -c "
    cd \$MYPLOTDIR
    bash plot_volume_int_xz.sh $simname
"

screen -dm bash -c "
    cd \$MYPLOTDIR
    bash plot_volume_int.sh $simname
"

screen -dm bash -c "
    cd \$MYPLOTDIR
    bash plot_field.sh $simname
"


screen -dm bash -c "
    cd \$MYPLOTDIR
    bash plot_metric_evolved.sh $simname
"

exit 0