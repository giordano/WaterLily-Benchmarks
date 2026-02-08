#!/bin/bash

set -euo pipefail

FEW_FREQUENCIES=(1005 1110 1200 1305 1410)
MANY_FREQUENCIES=(1005 1020 1035 1050 1065 1080 1095 1110 1125 1140 1155 1170 1185 1200 1215 1230 1245 1260 1275 1290 1305 1320 1335 1350 1365 1380 1395 1410)

MIN_POWER_CAP=150
MAX_POWER_CAP=300
POWER_CAPS=(${MIN_POWER_CAP} 175 200 225 250 275 ${MAX_POWER_CAP})

function reset_gpus() {
    sudo nvidia-smi --power-limit=${MAX_POWER_CAP}
    sudo nvidia-smi --reset-gpu-clocks
}

# Reset GPUs on exit
trap reset_gpus INT QUIT TERM EXIT

# Reset GPUs before starting, for good measure
reset_gpus

for frequency in ${FEW_FREQUENCIES[@]}; do
    for power in ${POWER_CAPS[@]}; do
        sudo nvidia-smi --lock-gpu-clocks=${frequency},${frequency}
        sudo nvidia-smi --power-limit=${power}
        sh benchmark.sh -v "1.12" -t "1" -b "CuArray" -c "sphere" -p "6" -s "30" -ft "Float32" -wd "${HOME}/.julia/dev/WaterLily" --data_dir "data-${frequency}-${power}/benchmarks"
    done
done
