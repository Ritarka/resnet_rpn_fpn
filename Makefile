AUTOPILOT_ROOT :=/tools/software/xilinx/Vitis_HLS/2021.1/

ASSEMBLE_SRC_ROOT := .
TB_ROOT := .
IFLAG += -I "${AUTOPILOT_ROOT}/include"
IFLAG += -I "${ASSEMBLE_SRC_ROOT}"
IFLAG += -I "${ASSEMBLE_SRC_ROOT}"
IFLAG += -I "/usr/include/x86_64-linux-gnu"
IFLAG += -D__SIM_FPO__ -D__SIM_OPENCV__ -D__SIM_FFT__ -D__SIM_FIR__ -D__SIM_DDS__ -D__DSP48E1__

IFLAG +=  -g 
CFLAG += -fPIC -std=c++11 -O3 -mcmodel=large #-fsanitize=address
CC      = g++ 

ALLOUT+= csim.out

all: $(ALLOUT) 

##TO BE MODIFIED START
########################################################
# RESNET LAYERS
########################################################
resnet_batchnorm.o:./resnet_batchnorm.cpp
	$(CC) $(GCOV)  $(CFLAG)  -o $@ -c $^    -MMD $(IFLAG)
resnet_conv_1x1.o:./resnet_conv_1x1.cpp
	$(CC) $(GCOV)  $(CFLAG)  -o $@ -c $^    -MMD $(IFLAG)
resnet_conv_3x3.o:./resnet_conv_3x3.cpp
	$(CC) $(GCOV)  $(CFLAG)  -o $@ -c $^    -MMD $(IFLAG)
resnet_conv_7x7.o:./resnet_conv_7x7.cpp
	$(CC) $(GCOV)  $(CFLAG)  -o $@ -c $^    -MMD $(IFLAG)
# resnet_top0.o:./resnet_top0.cpp
# 	$(CC) $(GCOV)  $(CFLAG)  -o $@ -c $^    -MMD $(IFLAG)
resnet_top1.o:./resnet_top1.cpp
	$(CC) $(GCOV)  $(CFLAG)  -o $@ -c $^    -MMD $(IFLAG)
# resnet_top2.o:./resnet_top2.cpp
# 	$(CC) $(GCOV)  $(CFLAG)  -o $@ -c $^    -MMD $(IFLAG)
# resnet_top3.o:./resnet_top3.cpp
# 	$(CC) $(GCOV)  $(CFLAG)  -o $@ -c $^    -MMD $(IFLAG)
# resnet_top4.o:./resnet_top4.cpp
# 	$(CC) $(GCOV)  $(CFLAG)  -o $@ -c $^    -MMD $(IFLAG)

########################################################
# FPN LAYERS
########################################################
# fpn_conv_3x3.o:./fpn_conv_3x3.cpp
# 	$(CC) $(GCOV)  $(CFLAG)  -o $@ -c $^    -MMD $(IFLAG)
# fpn_conv_1x1.o:./fpn_conv_1x1.cpp
# 	$(CC) $(GCOV)  $(CFLAG)  -o $@ -c $^    -MMD $(IFLAG)
# fpn_tiled_conv_fpn_3.o:./fpn_tiled_conv_fpn_3.cpp
# 	$(CC) $(GCOV)  $(CFLAG)  -o $@ -c $^    -MMD $(IFLAG)
# fpn_tiled_conv_fpn_2.o:./fpn_tiled_conv_fpn_2.cpp
# 	$(CC) $(GCOV)  $(CFLAG)  -o $@ -c $^    -MMD $(IFLAG)
# fpn_tiled_conv_fpn_1.o:./fpn_tiled_conv_fpn_1.cpp
# 	$(CC) $(GCOV)  $(CFLAG)  -o $@ -c $^    -MMD $(IFLAG)
# fpn_tiled_conv_fpn_0.o:./fpn_tiled_conv_fpn_0.cpp
# 	$(CC) $(GCOV)  $(CFLAG)  -o $@ -c $^    -MMD $(IFLAG)
# fpn_tiled_conv_lateral_3.o:./fpn_tiled_conv_lateral_3.cpp
# 	$(CC) $(GCOV)  $(CFLAG)  -o $@ -c $^    -MMD $(IFLAG)
# fpn_tiled_conv_lateral_2.o:./fpn_tiled_conv_lateral_2.cpp
# 	$(CC) $(GCOV)  $(CFLAG)  -o $@ -c $^    -MMD $(IFLAG)
# fpn_tiled_conv_lateral_1.o:./fpn_tiled_conv_lateral_1.cpp
# 	$(CC) $(GCOV)  $(CFLAG)  -o $@ -c $^    -MMD $(IFLAG)
# fpn_tiled_conv_lateral_0.o:./fpn_tiled_conv_lateral_0.cpp
# 	$(CC) $(GCOV)  $(CFLAG)  -o $@ -c $^    -MMD $(IFLAG)
# fpn_utils.o:./fpn_utils.cpp
# 	$(CC) $(GCOV)  $(CFLAG)  -o $@ -c $^    -MMD $(IFLAG)
# fpn_top.o:./fpn_top0.cpp
# 	$(CC) $(GCOV)  $(CFLAG)  -o $@ -c $^    -MMD $(IFLAG)

########################################################
# RPN LAYERS
########################################################
# rpn_conv_1x1.o:./rpn_conv_1x1.cpp
# 	$(CC) $(GCOV)  $(CFLAG)  -o $@ -c $^    -MMD $(IFLAG)
# rpn_conv_3x3.o:./rpn_conv_3x3.cpp
# 	$(CC) $(GCOV)  $(CFLAG)  -o $@ -c $^    -MMD $(IFLAG)
# rpn_top.o:./rpn_top.cpp
# 	$(CC) $(GCOV)  $(CFLAG)  -o $@ -c $^    -MMD $(IFLAG)
# rpn_top2.o:./rpn_top2.cpp
# 	$(CC) $(GCOV)  $(CFLAG)  -o $@ -c $^    -MMD $(IFLAG)

########################################################
# TESTBENCH
########################################################
# test_top_resnet0.o: ./test_top_resnet0.cpp
# 	$(CC) $(GCOV)  $(CFLAG)  -o $@ -c $^    -MMD $(IFLAG)
test_top_resnet1.o: ./test_top_resnet1.cpp
	$(CC) $(GCOV)  $(CFLAG)  -o $@ -c $^    -MMD $(IFLAG)
# test_top_resnet2.o: ./test_top_resnet2.cpp
# 	$(CC) $(GCOV)  $(CFLAG)  -o $@ -c $^    -MMD $(IFLAG)
# test_top_resnet3.o: ./test_top_resnet3.cpp
# 	$(CC) $(GCOV)  $(CFLAG)  -o $@ -c $^    -MMD $(IFLAG)
# test_top_resnet4.o: ./test_top_resnet4.cpp
# 	$(CC) $(GCOV)  $(CFLAG)  -o $@ -c $^    -MMD $(IFLAG)
# test_top_fpn.o: ./test_top_fpn0.cpp
# 	$(CC) $(GCOV)  $(CFLAG)  -o $@ -c $^    -MMD $(IFLAG)
# test_top_rpn.o: ./test_top_rpn.cpp
# 	$(CC) $(GCOV)  $(CFLAG)  -o $@ -c $^    -MMD $(IFLAG)
# test_top_rpn2.o: ./test_top_rpn2.cpp
# 	$(CC) $(GCOV)  $(CFLAG)  -o $@ -c $^    -MMD $(IFLAG)
# test_top.o: ./test_top.cpp
# 	$(CC) $(GCOV)  $(CFLAG)  -o $@ -c $^    -MMD $(IFLAG)

########################################################
# RESNET LAYERS
########################################################
IP_DEP+=resnet_conv_1x1.o
IP_DEP+=resnet_conv_3x3.o
IP_DEP+=resnet_conv_7x7.o
IP_DEP+=resnet_batchnorm.o
# IP_DEP+=resnet_top0.o
IP_DEP+=resnet_top1.o
# IP_DEP+=resnet_top2.o
# IP_DEP+=resnet_top3.o
# IP_DEP+=resnet_top4.o

########################################################
# FPN LAYERS
########################################################
# IP_DEP+=fpn_conv_3x3.o
# IP_DEP+=fpn_conv_1x1.o
# IP_DEP+=fpn_tiled_conv_fpn_3.o
# IP_DEP+=fpn_tiled_conv_fpn_2.o
# IP_DEP+=fpn_tiled_conv_fpn_1.o
# IP_DEP+=fpn_tiled_conv_fpn_0.o
# IP_DEP+=fpn_tiled_conv_lateral_3.o
# IP_DEP+=fpn_tiled_conv_lateral_2.o
# IP_DEP+=fpn_tiled_conv_lateral_1.o
# IP_DEP+=fpn_tiled_conv_lateral_0.o
# IP_DEP+=fpn_utils.o
# IP_DEP+=fpn_top.o

########################################################
# RPN LAYERS
########################################################
# IP_DEP+=rpn_conv_1x1.o
# IP_DEP+=rpn_conv_3x3.o
# IP_DEP+=rpn_top.o
# IP_DEP+=rpn_top2.o

# IP_DEP+=test_top_resnet0.o
IP_DEP+=test_top_resnet1.o
# IP_DEP+=test_top_resnet2.o
# IP_DEP+=test_top_resnet3.o
# IP_DEP+=test_top_resnet4.o
# IP_DEP+=test_top_fpn.o
# IP_DEP+=test_top_rpn.o
# IP_DEP+=test_top_rpn2.o
# IP_DEP+=test_top.o

main.o:./tb_test_top_resnet1.cpp
	$(CC) $(GCOV)  $(CFLAG)  -I "${ASSEMBLE_SRC_ROOT}" -o $@  -c $^   -MMD $(IFLAG)

csim.out: main.o $(IP_DEP)
	$(CC)  $(GCOV)  $(CFLAG) -MMD $(IFLAG)  -o $@  $^ && ./csim.out
	@$(MAKE) -s clean

synth:
	vitis_hls script.tcl

########################################################
# RESNET LAYERS
########################################################
synth_resnet0:
	vitis_hls script_resnet0.tcl

synth_resnet1:
	vitis_hls script_resnet1.tcl

synth_resnet2:
	vitis_hls script_resnet2.tcl

synth_resnet3:
	vitis_hls script_resnet3.tcl

synth_resnet4:
	vitis_hls script_resnet4.tcl

########################################################
# FPN LAYERS
########################################################
synth_fpn0:
	vitis_hls script_fpn0.tcl

synth_fpn1:
	vitis_hls script_fpn1.tcl

synth_fpn2:
	vitis_hls script_fpn2.tcl

synth_fpn3:
	vitis_hls script_fpn3.tcl

########################################################
# RPN LAYERS
########################################################
synth_rpn:
	vitis_hls script_rpn.tcl

synth_rpn2:
	vitis_hls script_rpn2.tcl

clean:
	rm -f -r csim.d 
	rm -f *.out *.gcno *.gcda *.txt *.o *.d
