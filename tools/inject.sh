#!/bin/bash
sh ./tools/clean.sh
mkdir -p build
mkdir -p ../lab/software/SharkSVM/src
cp -R * ../lab/software/SharkSVM/src
mkdir -p ../lab/software/SharkSVM/bin
cp bin/* ../lab/software/SharkSVM/bin
cp tools/injectFiles/Makefile ../lab/software/SharkSVM/src
rm -rf ../lab/software/SharkSVM/src/build
rm -rf ../lab/software/SharkSVM/src/build_release

