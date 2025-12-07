#!/bin/bash
cd ~/DX_aimbot
rm -rf ./build
mkdir build
cd build
cmake ..
make -j8
chmod +777 restart.sh
bash ../restart.sh
