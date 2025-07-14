#!/usr/bin/bash

## Uncomment whichever line applies to your vehicle.
# export FINGERPRINT="MAZDA 3 2019"
# export FINGERPRINT="MAZDA CX-30"
# export FINGERPRINT="MAZDA CX-50"
# export FINGERPRINT="MAZDA CX-90"
export ATHENA_HOST="wss://athena.konik.ai"
export API_HOST="https://api.konik.ai"

export PASSIVE="0"
exec ./launch_chffrplus.sh

