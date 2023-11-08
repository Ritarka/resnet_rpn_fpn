#ifndef RESNET_H

#include <iostream>
#include <ap_int.h>
#include <ap_fixed.h>

#pragma once

#define CSIM_DEBUG

#ifdef  CSIM_DEBUG
    typedef float fm_t;
    typedef float wt_t;
#else
    typedef ap_fixed<16, 2> fm_t;
    typedef ap_fixed<16, 2> wt_t;
#endif

#endif

#define FPN_IN_BUF_CH         8
#define FPN_IN_BUF_ROWS      23
#define FPN_IN_BUF_COLS      20

#define FPN_OUT_BUF_CH       16
#define FPN_OUT_BUF_ROWS     23
#define FPN_OUT_BUF_COLS     20

#define FPN_LAST_LAYER_ENABLE   1
#define FPN_LAST_LAYER_DISABLE  0

//FPN CONV Layer 0 param
#define FPN_CONV_0_ID 256
#define FPN_CONV_0_IH 184
#define FPN_CONV_0_IW 320

#define FPN_CONV_0_FILTER_SIZE 256

#define FPN_CONV_0_KERNEL_OD 256
#define FPN_CONV_0_KERNEL_ID 256
#define FPN_CONV_0_KERNEL_KH 3
#define FPN_CONV_0_KERNEL_KW 3

#define FPN_CONV_0_OD 256
#define FPN_CONV_0_OH 184
#define FPN_CONV_0_OW 320

//FPN CONV Layer 1 param
#define FPN_CONV_1_ID 256
#define FPN_CONV_1_IH 92
#define FPN_CONV_1_IW 160

#define FPN_CONV_1_FILTER_SIZE 256

#define FPN_CONV_1_KERNEL_OD 256
#define FPN_CONV_1_KERNEL_ID 256
#define FPN_CONV_1_KERNEL_KH 3
#define FPN_CONV_1_KERNEL_KW 3

#define FPN_CONV_1_OD 256
#define FPN_CONV_1_OH 92
#define FPN_CONV_1_OW 160

//FPN CONV Layer 2 param
#define FPN_CONV_2_ID 256
#define FPN_CONV_2_IH 46
#define FPN_CONV_2_IW 80

#define FPN_CONV_2_FILTER_SIZE 256

#define FPN_CONV_2_KERNEL_OD 256
#define FPN_CONV_2_KERNEL_ID 256
#define FPN_CONV_2_KERNEL_KH 3
#define FPN_CONV_2_KERNEL_KW 3

#define FPN_CONV_2_OD 256
#define FPN_CONV_2_OH 46
#define FPN_CONV_2_OW 80

//FPN CONV Layer 3 param
#define FPN_CONV_3_ID 256
#define FPN_CONV_3_IH 23
#define FPN_CONV_3_IW 40

#define FPN_CONV_3_FILTER_SIZE 256

#define FPN_CONV_3_KERNEL_OD 256
#define FPN_CONV_3_KERNEL_ID 256
#define FPN_CONV_3_KERNEL_KH 3
#define FPN_CONV_3_KERNEL_KW 3

#define FPN_CONV_3_OD 256
#define FPN_CONV_3_OH 23
#define FPN_CONV_3_OW 40

//LATERAL CONV 0 LAYER PARAM
#define LATERAL_CONV_0_ID 	256
#define LATERAL_CONV_0_IH 	184
#define LATERAL_CONV_0_IW 	320
#define LATERAL_CONV_0_OD 	256
#define LATERAL_CONV_0_OH 	184
#define LATERAL_CONV_0_OW 	320

//LATERAL CONV 1 LAYER PARAM
#define LATERAL_CONV_1_ID 	512
#define LATERAL_CONV_1_IH 	92
#define LATERAL_CONV_1_IW 	160
#define LATERAL_CONV_1_OD 	256
#define LATERAL_CONV_1_OH 	92
#define LATERAL_CONV_1_OW 	160

//LATERAL CONV 2 LAYER PARAM
#define LATERAL_CONV_2_ID 	1024
#define LATERAL_CONV_2_IH 	46
#define LATERAL_CONV_2_IW 	80
#define LATERAL_CONV_2_OD 	256
#define LATERAL_CONV_2_OH 	46
#define LATERAL_CONV_2_OW 	80

//LATERAL CONV 3 LAYER PARAM
#define LATERAL_CONV_3_ID 	2048
#define LATERAL_CONV_3_IH 	23
#define LATERAL_CONV_3_IW 	40
#define LATERAL_CONV_3_OD 	256
#define LATERAL_CONV_3_OH 	23
#define LATERAL_CONV_3_OW 	40

void fpn_top(
	fm_t lateral_3_input_feature_map[LATERAL_CONV_3_ID][LATERAL_CONV_3_IH][LATERAL_CONV_3_IW],
    	wt_t lateral_3_layer_weights[LATERAL_CONV_3_OD][LATERAL_CONV_3_ID][1][1],
    	wt_t lateral_3_layer_bias[LATERAL_CONV_3_OD],
	fm_t lateral_2_input_feature_map[LATERAL_CONV_2_ID][LATERAL_CONV_2_IH][LATERAL_CONV_2_IW],
    	wt_t lateral_2_layer_weights[LATERAL_CONV_2_OD][LATERAL_CONV_2_ID][1][1],
    	wt_t lateral_2_layer_bias[LATERAL_CONV_2_OD],
	fm_t lateral_1_input_feature_map[LATERAL_CONV_1_ID][LATERAL_CONV_1_IH][LATERAL_CONV_1_IW],
    	wt_t lateral_1_layer_weights[LATERAL_CONV_1_OD][LATERAL_CONV_1_ID][1][1],
    	wt_t lateral_1_layer_bias[LATERAL_CONV_1_OD],
	fm_t lateral_0_input_feature_map[LATERAL_CONV_0_ID][LATERAL_CONV_0_IH][LATERAL_CONV_0_IW],
    	wt_t lateral_0_layer_weights[LATERAL_CONV_0_OD][LATERAL_CONV_0_ID][1][1],
    	wt_t lateral_0_layer_bias[LATERAL_CONV_0_OD],
    	wt_t fpn_3_layer_weights[FPN_CONV_3_OD][FPN_CONV_3_ID][3][3],
    	wt_t fpn_3_layer_bias[FPN_CONV_3_OD],
    	fm_t fpn_3_output_feature_map[FPN_CONV_3_OD][FPN_CONV_3_IH][FPN_CONV_3_IW],
    	wt_t fpn_2_layer_weights[FPN_CONV_2_OD][FPN_CONV_2_ID][3][3],
    	wt_t fpn_2_layer_bias[FPN_CONV_2_OD],
    	fm_t fpn_2_output_feature_map[FPN_CONV_2_OD][FPN_CONV_2_IH][FPN_CONV_2_IW],
    	wt_t fpn_1_layer_weights[FPN_CONV_1_OD][FPN_CONV_1_ID][3][3],
    	wt_t fpn_1_layer_bias[FPN_CONV_1_OD],
    	fm_t fpn_1_output_feature_map[FPN_CONV_1_OD][FPN_CONV_1_IH][FPN_CONV_1_IW],
    	wt_t fpn_0_layer_weights[FPN_CONV_0_OD][FPN_CONV_0_ID][3][3],
    	wt_t fpn_0_layer_bias[FPN_CONV_0_OD],
    	fm_t fpn_0_output_feature_map[FPN_CONV_0_OD][FPN_CONV_0_IH][FPN_CONV_0_IW]
);

void fpn_conv_3x3 (
        fm_t Y_buf[FPN_OUT_BUF_CH][FPN_OUT_BUF_ROWS][FPN_OUT_BUF_COLS], 
        fm_t X_buf[FPN_IN_BUF_CH][FPN_IN_BUF_ROWS+2][FPN_IN_BUF_COLS+2],
        wt_t W_buf[FPN_OUT_BUF_CH][FPN_IN_BUF_CH][3][3]
);
void fpn_conv_1x1( 
    fm_t Y_buf[FPN_OUT_BUF_CH][FPN_OUT_BUF_ROWS][FPN_OUT_BUF_COLS], 
    fm_t X_buf[FPN_IN_BUF_CH][FPN_IN_BUF_ROWS][FPN_IN_BUF_COLS],
    wt_t W_buf[FPN_OUT_BUF_CH][FPN_IN_BUF_CH][1][1]
);
void fpn_tiled_conv_fpn_3 (
    fm_t input_feature_map[FPN_CONV_3_ID][FPN_CONV_3_IH][FPN_CONV_3_IW],
    wt_t layer_weights[FPN_CONV_3_OD][FPN_CONV_3_ID][3][3],
    wt_t layer_bias[FPN_CONV_3_OD],
    fm_t output_feature_map[FPN_CONV_3_OD][FPN_CONV_3_OH][FPN_CONV_3_OW]
);
void fpn_tiled_conv_fpn_2 (
    fm_t input_feature_map[FPN_CONV_2_ID][FPN_CONV_2_IH][FPN_CONV_2_IW],
    wt_t layer_weights[FPN_CONV_2_OD][FPN_CONV_2_ID][3][3],
    wt_t layer_bias[FPN_CONV_2_OD],
    fm_t output_feature_map[FPN_CONV_2_OD][FPN_CONV_2_OH][FPN_CONV_2_OW]
);
void fpn_tiled_conv_fpn_1 (
    fm_t input_feature_map[FPN_CONV_1_ID][FPN_CONV_1_IH][FPN_CONV_1_IW],
    wt_t layer_weights[FPN_CONV_1_OD][FPN_CONV_1_ID][3][3],
    wt_t layer_bias[FPN_CONV_1_OD],
    fm_t output_feature_map[FPN_CONV_1_OD][FPN_CONV_1_OH][FPN_CONV_1_OW]
);
void fpn_tiled_conv_fpn_0 (
    fm_t input_feature_map[FPN_CONV_0_ID][FPN_CONV_0_IH][FPN_CONV_0_IW],
    wt_t layer_weights[FPN_CONV_0_OD][FPN_CONV_0_ID][3][3],
    wt_t layer_bias[FPN_CONV_0_OD],
    fm_t output_feature_map[FPN_CONV_0_OD][FPN_CONV_0_OH][FPN_CONV_0_OW]
);
void fpn_tiled_conv_lateral_3 (
    fm_t input_feature_map[LATERAL_CONV_3_ID][LATERAL_CONV_3_IH][LATERAL_CONV_3_IW],
    wt_t layer_weights[LATERAL_CONV_3_OD][LATERAL_CONV_3_ID][1][1],
    wt_t layer_bias[LATERAL_CONV_3_OD],
    fm_t output_feature_map[LATERAL_CONV_3_OD][LATERAL_CONV_3_IH][LATERAL_CONV_3_IW]
);
void fpn_tiled_conv_lateral_2 (
    fm_t input_feature_map[LATERAL_CONV_2_ID][LATERAL_CONV_2_IH][LATERAL_CONV_2_IW],
    wt_t layer_weights[LATERAL_CONV_2_OD][LATERAL_CONV_2_ID][1][1],
    wt_t layer_bias[LATERAL_CONV_2_OD],
    fm_t output_feature_map[LATERAL_CONV_2_OD][LATERAL_CONV_2_IH][LATERAL_CONV_2_IW]
);
void fpn_tiled_conv_lateral_1 (
    fm_t input_feature_map[LATERAL_CONV_1_ID][LATERAL_CONV_1_IH][LATERAL_CONV_1_IW],
    wt_t layer_weights[LATERAL_CONV_1_OD][LATERAL_CONV_1_ID][1][1],
    wt_t layer_bias[LATERAL_CONV_1_OD],
    fm_t output_feature_map[LATERAL_CONV_1_OD][LATERAL_CONV_1_IH][LATERAL_CONV_1_IW]
);
void fpn_tiled_conv_lateral_0 (
    fm_t input_feature_map[LATERAL_CONV_0_ID][LATERAL_CONV_0_IH][LATERAL_CONV_0_IW],
    wt_t layer_weights[LATERAL_CONV_0_OD][LATERAL_CONV_0_ID][1][1],
    wt_t layer_bias[LATERAL_CONV_0_OD],
    fm_t output_feature_map[LATERAL_CONV_0_OD][LATERAL_CONV_0_IH][LATERAL_CONV_0_IW]
);

template<const int fm_input_depth,const int fm_input_height,const int fm_input_width,const int N_TILE_ROWS,const int N_TILE_COLS>
void fpn_load_input_tile_block_from_DRAM_3x3 (
    fm_t in_fm_buf[FPN_IN_BUF_CH][FPN_IN_BUF_ROWS+2][FPN_IN_BUF_COLS+2], 
    fm_t in_fm[fm_input_depth][fm_input_height][fm_input_width], 
    int  ti, 
    int  tj, 
    int  d
);

template<const int fm_input_depth,const int fm_input_height,const int fm_input_width,const int N_TILE_ROWS,const int N_TILE_COLS>
void fpn_load_input_tile_block_from_DRAM_1x1 (
    fm_t in_fm_buf[FPN_IN_BUF_CH][FPN_IN_BUF_ROWS][FPN_IN_BUF_COLS], 
    fm_t in_fm[fm_input_depth][fm_input_height][fm_input_width], 
    int  ti, 
    int  tj, 
    int  d
);
template<const int output_depth,const int input_depth>
void fpn_load_layer_params_from_DRAM_1x1 (
    wt_t weight_buf[FPN_OUT_BUF_CH][FPN_IN_BUF_CH][1][1],
    wt_t bias_buf[FPN_OUT_BUF_CH],
    wt_t weights[output_depth][input_depth][1][1],
    wt_t bias[output_depth],
    int b,
    int d
);
template<const int output_depth,const int input_depth>
void fpn_load_layer_params_from_DRAM_3x3 (
    wt_t weight_buf[FPN_OUT_BUF_CH][FPN_IN_BUF_CH][3][3],
    wt_t bias_buf[FPN_OUT_BUF_CH],
    wt_t weights[output_depth][input_depth][3][3],
    wt_t bias[output_depth],
    int b,
    int d
);

//------------------------------------------------------------------------------
// Function to save partial outputs on-chip for each input tile slice processed.
//------------------------------------------------------------------------------
template<const int output_depth>
void fpn_save_partial_output_tile_block (
    fm_t partial_out_buf[FPN_OUT_BUF_CH][FPN_OUT_BUF_ROWS][FPN_OUT_BUF_COLS], 
    fm_t out_fm_buf[FPN_OUT_BUF_CH][FPN_OUT_BUF_ROWS][FPN_OUT_BUF_COLS],
    wt_t bias_buf[FPN_OUT_BUF_CH],
    int  d
);

//--------------------------------------------------------------------------
// Function to store complete output tile block from BRAM to DRAM.
//--------------------------------------------------------------------------
template<const int fm_output_depth,const int fm_output_height,const int fm_output_width>
void fpn_store_output_tile_to_DRAM (
    fm_t out_fm[fm_output_depth][fm_output_height][fm_output_width], 
    fm_t out_fm_buf[FPN_OUT_BUF_CH][FPN_OUT_BUF_ROWS][FPN_OUT_BUF_COLS], 
    int  ti,
    int  tj,
    int  b
);
