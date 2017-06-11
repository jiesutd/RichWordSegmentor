sh cleanall.sh
cmake .
make STDSeg
./STDSeg -l -train ../example/train.debug -dev ../example/dev.debug -test ../example/test.debug -option ../example/option.STD -model ../example/debug.model -word ../example/ctb.50d.word.debug
./STDSeg -test ../example/test.debug -model ../example/debug.model -output ../example/test.debug.out
