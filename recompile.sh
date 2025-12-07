#!/bin/bash
cd build
cmake ..
make -j8
bash ../restart.sh