# TCL commands for batch-mode HLS
open_project -reset proj

# Define the top modules
set_top resnet_top_0
add_files ./resnet_batchnorm.cpp
add_files ./resnet_conv_1x1.cpp
add_files ./resnet_conv_3x3.cpp
add_files ./resnet_conv_7x7.cpp
add_files ./resnet_top0.cpp


# add_files -tb ./bin
add_files -tb ./tb_test_top_resnet0.cpp
add_files -tb ./test_top_resnet0.cpp


open_solution "solution1" -flow_target vivado
# set_part {xc7z020clg400-1}
set_part {xczu9eg-ffvb1156-2-e}
create_clock -period 10 -name default

## C simulation
# Use Makefile instead. This is even slower.
#csim_design -O -clean

## C code synthesis to generate Verilog code
csynth_design

## C and Verilog co-simulation
## This usually takes a long time so it is commented
## You may uncomment it if necessary
#cosim_design

## export synthesized Verilog code
#export_design -format ip_catalog

exit