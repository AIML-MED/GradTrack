#!/usr/bin/env bash
set -e

for setting in "symmetric 0.6" "asymmetric 0.3" "instance 0.4"; do
  read -r NOISE_TYPE NOISE_RATE <<< "${setting}"
  python train.py --noise_type "${NOISE_TYPE}" --noise_rate "${NOISE_RATE}"
  python gradtrack.py --noise_type "${NOISE_TYPE}" --noise_rate "${NOISE_RATE}"
done
