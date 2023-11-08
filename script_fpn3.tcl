# TCL commands for batch-mode HLS
open_project -reset proj

#Define the top modules
set_top fpn_top
add_files ./fpn_conv_1x1.cpp
add_files ./fpn_conv_3x3.cpp
add_files ./fpn_tiled_conv_fpn_0.cpp
add_files ./fpn_tiled_conv_fpn_1.cpp
add_files ./fpn_tiled_conv_fpn_2.cpp
add_files ./fpn_tiled_conv_fpn_3.cpp
add_files ./fpn_tiled_conv_lateral_0.cpp
add_files ./fpn_tiled_conv_lateral_1.cpp
add_files ./fpn_tiled_conv_lateral_2.cpp
add_files ./fpn_tiled_conv_lateral_3.cpp
add_files ./fpn_top.cpp
add_files ./fpn_utils.cpp


# add_files -tb ./bin
add_files -tb ./tb_test_top_fpn.cpp
add_files -tb ./test_top_fpn.cpp


open_solution "solution1" -flow_target vivado
set_part {xc7z020clg400-1}
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