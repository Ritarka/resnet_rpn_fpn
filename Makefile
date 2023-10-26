AUTOPILOT_ROOT :=/tools/software/xilinx/Vitis_HLS/2021.1/

ASSEMBLE_SRC_ROOT := .
TB_ROOT := .
IFLAG += -I "${AUTOPILOT_ROOT}/include"
IFLAG += -I "${ASSEMBLE_SRC_ROOT}"
IFLAG += -I "${ASSEMBLE_SRC_ROOT}"
# IFLAG += -I "${TB_ROOT}"
IFLAG += -I "/usr/include/x86_64-linux-gnu"
IFLAG += -D__SIM_FPO__ -D__SIM_OPENCV__ -D__SIM_FFT__ -D__SIM_FIR__ -D__SIM_DDS__ -D__DSP48E1__

IFLAG +=  -g 
CFLAG += -fPIC -std=c++11 -O3 -mcmodel=large #-fsanitize=address
# CFLAG += -fPIC -03 -std=c++11 -mcmodel=large #-fsanitize=address
CC      = g++ 

ALLOUT+= csim.out

all: $(ALLOUT) 
##TO BE MODIFIED START

resnet_batchnorm.o:./resnet_batchnorm.cpp
	$(CC) $(GCOV)  $(CFLAG)  -o $@ -c $^    -MMD $(IFLAG)
resnet_conv_1x1.o:./resnet_conv_1x1.cpp
	$(CC) $(GCOV)  $(CFLAG)  -o $@ -c $^    -MMD $(IFLAG)
resnet_conv_3x3.o:./resnet_conv_3x3.cpp
	$(CC) $(GCOV)  $(CFLAG)  -o $@ -c $^    -MMD $(IFLAG)
resnet_conv_7x7.o:./resnet_conv_7x7.cpp
	$(CC) $(GCOV)  $(CFLAG)  -o $@ -c $^    -MMD $(IFLAG)
resnet_top.o:./resnet_top.cpp
	$(CC) $(GCOV)  $(CFLAG)  -o $@ -c $^    -MMD $(IFLAG)
resnet_top2.o:./resnet_top2.cpp
	$(CC) $(GCOV)  $(CFLAG)  -o $@ -c $^    -MMD $(IFLAG)

fpn_conv_3x3.o:./fpn_conv_3x3.cpp
	$(CC) $(GCOV)  $(CFLAG)  -o $@ -c $^    -MMD $(IFLAG)
fpn_conv_1x1.o:./fpn_conv_1x1.cpp
	$(CC) $(GCOV)  $(CFLAG)  -o $@ -c $^    -MMD $(IFLAG)
fpn_tiled_conv_fpn_3.o:./fpn_tiled_conv_fpn_3.cpp
	$(CC) $(GCOV)  $(CFLAG)  -o $@ -c $^    -MMD $(IFLAG)
fpn_tiled_conv_fpn_2.o:./fpn_tiled_conv_fpn_2.cpp
	$(CC) $(GCOV)  $(CFLAG)  -o $@ -c $^    -MMD $(IFLAG)
fpn_tiled_conv_fpn_1.o:./fpn_tiled_conv_fpn_1.cpp
	$(CC) $(GCOV)  $(CFLAG)  -o $@ -c $^    -MMD $(IFLAG)
fpn_tiled_conv_fpn_0.o:./fpn_tiled_conv_fpn_0.cpp
	$(CC) $(GCOV)  $(CFLAG)  -o $@ -c $^    -MMD $(IFLAG)
fpn_tiled_conv_lateral_3.o:./fpn_tiled_conv_lateral_3.cpp
	$(CC) $(GCOV)  $(CFLAG)  -o $@ -c $^    -MMD $(IFLAG)
fpn_tiled_conv_lateral_2.o:./fpn_tiled_conv_lateral_2.cpp
	$(CC) $(GCOV)  $(CFLAG)  -o $@ -c $^    -MMD $(IFLAG)
fpn_tiled_conv_lateral_1.o:./fpn_tiled_conv_lateral_1.cpp
	$(CC) $(GCOV)  $(CFLAG)  -o $@ -c $^    -MMD $(IFLAG)
fpn_tiled_conv_lateral_0.o:./fpn_tiled_conv_lateral_0.cpp
	$(CC) $(GCOV)  $(CFLAG)  -o $@ -c $^    -MMD $(IFLAG)
fpn_utils.o:./fpn_utils.cpp
	$(CC) $(GCOV)  $(CFLAG)  -o $@ -c $^    -MMD $(IFLAG)
fpn_top.o:./fpn_top.cpp
	$(CC) $(GCOV)  $(CFLAG)  -o $@ -c $^    -MMD $(IFLAG)

rpn_conv_1x1.o:./rpn_conv_1x1.cpp
	$(CC) $(GCOV)  $(CFLAG)  -o $@ -c $^    -MMD $(IFLAG)
rpn_conv_3x3.o:./rpn_conv_3x3.cpp
	$(CC) $(GCOV)  $(CFLAG)  -o $@ -c $^    -MMD $(IFLAG)
rpn_top.o:./rpn_top.cpp
	$(CC) $(GCOV)  $(CFLAG)  -o $@ -c $^    -MMD $(IFLAG)
rpn_top2.o:./rpn_top2.cpp
	$(CC) $(GCOV)  $(CFLAG)  -o $@ -c $^    -MMD $(IFLAG)

test_top.o: ./test_top.cpp
	$(CC) $(GCOV)  $(CFLAG)  -o $@ -c $^    -MMD $(IFLAG)

##TO BE MODIFIED END

IP_DEP+=resnet_conv_1x1.o
IP_DEP+=resnet_conv_3x3.o
IP_DEP+=resnet_conv_7x7.o
IP_DEP+=resnet_batchnorm.o
IP_DEP+=resnet_top.o
IP_DEP+=resnet_top2.o

IP_DEP+=fpn_conv_3x3.o
IP_DEP+=fpn_conv_1x1.o
IP_DEP+=fpn_tiled_conv_fpn_3.o
IP_DEP+=fpn_tiled_conv_fpn_2.o
IP_DEP+=fpn_tiled_conv_fpn_1.o
IP_DEP+=fpn_tiled_conv_fpn_0.o
IP_DEP+=fpn_tiled_conv_lateral_3.o
IP_DEP+=fpn_tiled_conv_lateral_2.o
IP_DEP+=fpn_tiled_conv_lateral_1.o
IP_DEP+=fpn_tiled_conv_lateral_0.o
IP_DEP+=fpn_utils.o
IP_DEP+=fpn_top.o

IP_DEP+=rpn_conv_1x1.o
IP_DEP+=rpn_conv_3x3.o
IP_DEP+=rpn_top.o
IP_DEP+=rpn_top2.o

IP_DEP+=test_top.o

main.o:./tb_test_top.cpp
	$(CC) $(GCOV)  $(CFLAG)  -I "${ASSEMBLE_SRC_ROOT}" -o $@  -c $^   -MMD $(IFLAG)

csim.out: main.o $(IP_DEP)
	$(CC)  $(GCOV)  $(CFLAG) -MMD $(IFLAG)  -o $@  $^ && ./csim.out
	@$(MAKE) -s clean

synth:
	vitis_hls script.tcl

synth_resnet:
	vitis_hls script_resnet.tcl

synth_fpn:
	vitis_hls script_fpn.tcl

synth_rpn:
	vitis_hls script_rpn.tcl

clean:
	rm -f -r csim.d 
	rm -f *.out *.gcno *.gcda *.txt *.o *.d
