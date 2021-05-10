#! /bin/bash

/bin/rm -rf build
mkdir build
cd build
cmake ..
make -j8

/bin/rm -rf ../raytracing/test
cp raytracing ../raytracing/test

