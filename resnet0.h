#ifndef RESNET_H
#define RESNET_H

#include "hls_stream.h"
#include <iostream>
#include <ap_int.h>
#include <ap_fixed.h>

#pragma once

//--------------------------------------------------------------------------
// Compiler Defines
//--------------------------------------------------------------------------
#define CSIM_DEBUG

#define TEST_COMPLETE_MODEL

//--------------------------------------------------------------------------
// Type Conversions
//--------------------------------------------------------------------------
#ifdef  CSIM_DEBUG
    typedef float fm_t;
    typedef float wt_t;
#else
    typedef ap_fixed<32, 8, AP_RND, AP_SAT> fm_t;
    typedef ap_fixed<32, 8, AP_RND, AP_SAT> wt_t;
#endif


// Buffer sizes must be even numbers
// Buffer depth must be an integer multiple of 64

#define RESNET_IN_BUF_CH        64
#define RESNET_IN_BUF_ROWS      46
#define RESNET_IN_BUF_COLS      40

#define RESNET_OUT_BUF_CH       64
#define RESNET_OUT_BUF_ROWS     46
#define RESNET_OUT_BUF_COLS     40

#define RESNET_LAST_LAYER_ENABLE   1
#define RESNET_LAST_LAYER_DISABLE  0

//--------------------------------------------------------------------------
// Layer 0: Sizes of botteleck layers' input feature maps
//--------------------------------------------------------------------------
#define RESNET_LAYER0_IN_FM_HEIGHT    736
#define RESNET_LAYER0_IN_FM_WIDTH    1280
#define RESNET_LAYER0_MX_FM_HEIGHT    368
#define RESNET_LAYER0_MX_FM_WIDTH     640
#define RESNET_LAYER0_OUT_FM_HEIGHT   184
#define RESNET_LAYER0_OUT_FM_WIDTH    320

#define RESNET_LAYER0_CONV1_IN_CH     3
#define RESNET_LAYER0_CONV1_OUT_CH   64

//--------------------------------------------------------------------------
// Function Declarations
//--------------------------------------------------------------------------
template<const int RESNET_IN_FM_DEPTH, const int RESNET_IN_FM_HEIGHT, const int RESNET_IN_FM_WIDTH,
         const int RESNET_LAST_LAYER_EN>
void resnet_load_input_fm_tile (
        fm_t in_fm_buf[RESNET_IN_BUF_CH][RESNET_IN_BUF_ROWS + 5][RESNET_IN_BUF_COLS + 5], 
        fm_t in_fm[2048][184][320], 
        int   ti, 
        int   tj, 
        int   P,
        int   d
);

void resnet_load_residual_fm_tile (
        fm_t in_fm_buf[RESNET_IN_BUF_CH][RESNET_IN_BUF_ROWS][RESNET_IN_BUF_COLS], 
        fm_t in_fm[2048][184][320], 
        int   ti, 
        int   tj,
        int   b 
);

template<const int RESNET_OUT_CH, const int RESNET_IN_CH>
void resnet_load_weights_1x1 (
        wt_t weight_buf_1x1[RESNET_IN_BUF_CH],
        wt_t weights[RESNET_OUT_CH][RESNET_IN_CH],
        int f,
        int b,
        int d
);

template<const int RESNET_OUT_CH, const int RESNET_IN_CH>
void resnet_load_weights_3x3 (
        wt_t weight_buf_3x3[RESNET_IN_BUF_CH][3][3],
        wt_t weights[RESNET_OUT_CH][RESNET_IN_CH][3][3],
        int f,
        int b,
        int d
);

void resnet_conv_7x7 (
        fm_t Y_buf[RESNET_OUT_BUF_CH][RESNET_OUT_BUF_ROWS][RESNET_OUT_BUF_COLS], 
        fm_t X_buf[RESNET_IN_BUF_CH][RESNET_IN_BUF_ROWS + 5][RESNET_IN_BUF_COLS + 5],
        wt_t W_buf[RESNET_IN_BUF_CH][7][7],
        int f
);

template<const int RESNET_OUT_CH>
void resnet_load_batchnorm_params (
        fm_t param_buf[3][RESNET_OUT_BUF_CH], 
        wt_t params[3][RESNET_OUT_CH], 
        int b
);

template<const int S>
void resnet_store_out_buf_to_DDR(
        fm_t out_fm[2048][184][320], 
        fm_t out_fm_buf[RESNET_OUT_BUF_CH][RESNET_OUT_BUF_ROWS][RESNET_OUT_BUF_COLS], 
        int ti, 
        int tj,
        int b
);

void resnet_conv_1x1 (
        fm_t Y_buf[RESNET_OUT_BUF_CH][RESNET_OUT_BUF_ROWS][RESNET_OUT_BUF_COLS], 
        fm_t X_buf[RESNET_IN_BUF_CH][RESNET_IN_BUF_ROWS + 5][RESNET_IN_BUF_COLS + 5],
        wt_t W_buf[RESNET_IN_BUF_CH],
        int f,
        int S
);

//TODO: template<const int S>
void resnet_conv_3x3 (
        fm_t Y_buf[RESNET_OUT_BUF_CH][RESNET_OUT_BUF_ROWS][RESNET_OUT_BUF_COLS], 
        fm_t X_buf[RESNET_IN_BUF_CH][RESNET_IN_BUF_ROWS + 5][RESNET_IN_BUF_COLS + 5],
        wt_t W_buf[RESNET_IN_BUF_CH][3][3],
        int f,
        int S
);

void resnet_batchnorm(
        fm_t feature_map[RESNET_OUT_BUF_CH][RESNET_OUT_BUF_ROWS][RESNET_OUT_BUF_COLS], 
        wt_t bn_params[3][RESNET_OUT_BUF_CH],
        bool  enable_relu
);

void resnet_add_residual_fm(
        fm_t feature_map[RESNET_OUT_BUF_CH][RESNET_OUT_BUF_ROWS][RESNET_OUT_BUF_COLS], 
        fm_t residual_map[RESNET_OUT_BUF_CH][RESNET_OUT_BUF_ROWS][RESNET_OUT_BUF_COLS],
        bool  enable_relu
);

template<const int RESNET_IN_FM_DEPTH,  const int RESNET_IN_FM_HEIGHT,  const int RESNET_IN_FM_WIDTH,
         const int RESNET_OUT_FM_DEPTH, const int RESNET_OUT_FM_HEIGHT, const int RESNET_OUT_FM_WIDTH,
         const int STRIDE, const int RESNET_LAYER_EN>
void resnet_bottleneck_conv1_bn1(
        wt_t conv_weights[RESNET_OUT_FM_DEPTH][RESNET_IN_FM_DEPTH],
        wt_t bn_params[3][RESNET_OUT_FM_DEPTH],
        bool  enable_relu
);

template<const int RESNET_IN_FM_DEPTH,  const int RESNET_IN_FM_HEIGHT,  const int RESNET_IN_FM_WIDTH,
         const int RESNET_OUT_FM_DEPTH, const int RESNET_OUT_FM_HEIGHT, const int RESNET_OUT_FM_WIDTH,
         const int STRIDE, const int RESNET_LAYER_EN>
void resnet_bottleneck_conv2_bn2_relu(
        wt_t conv_weights[RESNET_OUT_FM_DEPTH][RESNET_IN_FM_DEPTH][3][3],
        wt_t bn_params[3][RESNET_OUT_FM_DEPTH]
);

template<const int  RESNET_IN_FM_DEPTH, const int  RESNET_IN_FM_HEIGHT, const int  RESNET_IN_FM_WIDTH,
         const int RESNET_OUT_FM_DEPTH, const int RESNET_OUT_FM_HEIGHT, const int RESNET_OUT_FM_WIDTH,
         const int STRIDE, const int RESNET_LAYER_EN>
void resnet_bottleneck_conv3_bn3_add_relu(
        wt_t conv_weights[RESNET_OUT_FM_DEPTH][RESNET_IN_FM_DEPTH],
        wt_t bn_params[3][RESNET_OUT_FM_DEPTH]
);

//--------------------------------------------------------------------------
// Layer 0
//--------------------------------------------------------------------------
void resnet_layer0(
        fm_t   resnet_layer0_input_fm[RESNET_LAYER0_CONV1_IN_CH][RESNET_LAYER0_IN_FM_HEIGHT][RESNET_LAYER0_IN_FM_WIDTH],
        wt_t   resnet_layer0_conv1_weights[RESNET_LAYER0_CONV1_OUT_CH][RESNET_LAYER0_CONV1_IN_CH][7][7],
	    wt_t   resnet_layer0_bn1_params[3][RESNET_LAYER0_CONV1_OUT_CH],
        fm_t   resnet_layer0_output_fm[RESNET_LAYER0_CONV1_OUT_CH][RESNET_LAYER0_OUT_FM_HEIGHT][RESNET_LAYER0_OUT_FM_WIDTH]
);

void test_resnet_top_0 (
    fm_t   resnet_layer0_input_fm[RESNET_LAYER0_CONV1_IN_CH][RESNET_LAYER0_IN_FM_HEIGHT][RESNET_LAYER0_IN_FM_WIDTH],
    wt_t   resnet_layer0_conv1_weights[RESNET_LAYER0_CONV1_OUT_CH][RESNET_LAYER0_CONV1_IN_CH][7][7],
    wt_t   resnet_layer0_bn1_params[3][RESNET_LAYER0_CONV1_OUT_CH],
    fm_t   resnet_layer0_output_fm[RESNET_LAYER0_CONV1_OUT_CH][RESNET_LAYER0_OUT_FM_HEIGHT][RESNET_LAYER0_OUT_FM_WIDTH],

    fm_t   resnet_layer1_input_fm[RESNET_LAYER1_0_CONV1_IN_CH][RESNET_LAYER1_0_FM_HEIGHT][RESNET_LAYER1_0_FM_WIDTH]
);

#endif
