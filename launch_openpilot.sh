#!/usr/bin/bash
export API_HOST=https://api.konik.ai
export ATHENA_HOST=wss://athena.konik.ai
export MAPS_HOST=https://api.konik.ai/maps
sudo mkdir /cache
sudo mkdir /cache/params
sudo mkdir /cache/params/d
exec ./launch_chffrplus.sh
