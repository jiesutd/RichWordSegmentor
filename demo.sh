sh cleanall.sh
cmake .
make STDSeg
./STDSeg -l -train ../data/train.debug -dev ../data/dev.debug -test ../data/test.debug -option ../data/option.STD -model ../data/debug.model -word ../data/ctb.50d.vec
./STDSeg -test ../data/test.debug -model ../data/debug.model -output ../data/test.debug.out
