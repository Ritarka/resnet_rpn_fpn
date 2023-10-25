#ifndef QDTRACK_H
#define QDTRACK_H

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
 #define TEST_RPN_CLS
 #define TEST_RPN_CONV
 #define TEST_RPN_REG
//--------------------------------------------------------------------------
// Type Conversions
//--------------------------------------------------------------------------
#ifdef  CSIM_DEBUG
    typedef float fm_t;
    typedef float wt_t;
#else
    typedef ap_fixed<16, 2> fm_t;
    typedef ap_fixed<16, 2> wt_t;
#endif


// // Buffer sizes must be even numbers
// // Buffer depth must be an integer multiple of 64

// #define RESNET_IN_BUF_CH        64
// #define RESNET_IN_BUF_ROWS      46
// #define RESNET_IN_BUF_COLS      40

// #define RESNET_OUT_BUF_CH       64
// #define RESNET_OUT_BUF_ROWS     46
// #define RESNET_OUT_BUF_COLS     40

// #define RESNET_LAST_LAYER_ENABLE   1
// #define RESNET_LAST_LAYER_DISABLE  0

// //--------------------------------------------------------------------------
// // Layer 0: Sizes of botteleck layers' input feature maps
// //--------------------------------------------------------------------------
// #define RESNET_LAYER0_IN_FM_HEIGHT    736
// #define RESNET_LAYER0_IN_FM_WIDTH    1280
// #define RESNET_LAYER0_MX_FM_HEIGHT    368
// #define RESNET_LAYER0_MX_FM_WIDTH     640
// #define RESNET_LAYER0_OUT_FM_HEIGHT   184
// #define RESNET_LAYER0_OUT_FM_WIDTH    320

// #define RESNET_LAYER0_CONV1_IN_CH     3
// #define RESNET_LAYER0_CONV1_OUT_CH   64

// //--------------------------------------------------------------------------
// // Layer 1: Sizes of botteleck layers' input feature maps
// //--------------------------------------------------------------------------
// #define RESNET_LAYER1_0_FM_HEIGHT 184
// #define RESNET_LAYER1_0_FM_WIDTH  320
// #define RESNET_LAYER1_FM_HEIGHT   184
// #define RESNET_LAYER1_FM_WIDTH    320

// #define RESNET_LAYER1_0_DS_IN_CH    64
// #define RESNET_LAYER1_0_DS_OUT_CH  256

// #define RESNET_LAYER1_0_CONV1_IN_CH    64
// #define RESNET_LAYER1_0_CONV1_OUT_CH   64
// #define RESNET_LAYER1_0_CONV2_IN_CH    64
// #define RESNET_LAYER1_0_CONV2_OUT_CH   64
// #define RESNET_LAYER1_0_CONV3_IN_CH    64
// #define RESNET_LAYER1_0_CONV3_OUT_CH  256

// #define RESNET_LAYER1_CONV1_IN_CH   256
// #define RESNET_LAYER1_CONV1_OUT_CH   64
// #define RESNET_LAYER1_CONV2_IN_CH    64
// #define RESNET_LAYER1_CONV2_OUT_CH   64
// #define RESNET_LAYER1_CONV3_IN_CH    64
// #define RESNET_LAYER1_CONV3_OUT_CH  256

// //--------------------------------------------------------------------------
// // Layer 1: Block-wise stride parameters
// //--------------------------------------------------------------------------
// #define RESNET_LAYER1_0_DS_STRIDE     1
// #define RESNET_LAYER1_0_CONV1_STRIDE  1
// #define RESNET_LAYER1_0_CONV2_STRIDE  1
// #define RESNET_LAYER1_0_CONV3_STRIDE  1
// #define RESNET_LAYER1_CONV1_STRIDE    1
// #define RESNET_LAYER1_CONV2_STRIDE    1
// #define RESNET_LAYER1_CONV3_STRIDE    1

// //--------------------------------------------------------------------------
// // Layer 2: Sizes of botteleck layers' input feature maps
// //--------------------------------------------------------------------------
// #define RESNET_LAYER2_0_FM_HEIGHT 184
// #define RESNET_LAYER2_0_FM_WIDTH  320
// #define RESNET_LAYER2_FM_HEIGHT    92
// #define RESNET_LAYER2_FM_WIDTH    160

// #define RESNET_LAYER2_0_DS_IN_CH   256
// #define RESNET_LAYER2_0_DS_OUT_CH  512

// #define RESNET_LAYER2_0_CONV1_IN_CH   256
// #define RESNET_LAYER2_0_CONV1_OUT_CH  128
// #define RESNET_LAYER2_0_CONV2_IN_CH   128
// #define RESNET_LAYER2_0_CONV2_OUT_CH  128
// #define RESNET_LAYER2_0_CONV3_IN_CH   128
// #define RESNET_LAYER2_0_CONV3_OUT_CH  512

// #define RESNET_LAYER2_CONV1_IN_CH   512
// #define RESNET_LAYER2_CONV1_OUT_CH  128
// #define RESNET_LAYER2_CONV2_IN_CH   128
// #define RESNET_LAYER2_CONV2_OUT_CH  128
// #define RESNET_LAYER2_CONV3_IN_CH   128
// #define RESNET_LAYER2_CONV3_OUT_CH  512

// //--------------------------------------------------------------------------
// // Layer 2: Block-wise stride parameters
// //--------------------------------------------------------------------------
// #define RESNET_LAYER2_0_DS_STRIDE     2
// #define RESNET_LAYER2_0_CONV1_STRIDE  1
// #define RESNET_LAYER2_0_CONV2_STRIDE  2
// #define RESNET_LAYER2_0_CONV3_STRIDE  1
// #define RESNET_LAYER2_CONV1_STRIDE    1
// #define RESNET_LAYER2_CONV2_STRIDE    1
// #define RESNET_LAYER2_CONV3_STRIDE    1

// //--------------------------------------------------------------------------
// // Layer 3: Sizes of botteleck layers' input feature maps
// //--------------------------------------------------------------------------
// #define RESNET_LAYER3_0_FM_HEIGHT  92
// #define RESNET_LAYER3_0_FM_WIDTH  160
// #define RESNET_LAYER3_FM_HEIGHT    46
// #define RESNET_LAYER3_FM_WIDTH     80

// #define RESNET_LAYER3_0_DS_IN_CH   512
// #define RESNET_LAYER3_0_DS_OUT_CH 1024

// #define RESNET_LAYER3_0_CONV1_IN_CH   512
// #define RESNET_LAYER3_0_CONV1_OUT_CH  256
// #define RESNET_LAYER3_0_CONV2_IN_CH   256
// #define RESNET_LAYER3_0_CONV2_OUT_CH  256
// #define RESNET_LAYER3_0_CONV3_IN_CH   256
// #define RESNET_LAYER3_0_CONV3_OUT_CH 1024

// #define RESNET_LAYER3_CONV1_IN_CH  1024
// #define RESNET_LAYER3_CONV1_OUT_CH  256
// #define RESNET_LAYER3_CONV2_IN_CH   256
// #define RESNET_LAYER3_CONV2_OUT_CH  256
// #define RESNET_LAYER3_CONV3_IN_CH   256
// #define RESNET_LAYER3_CONV3_OUT_CH 1024

// //--------------------------------------------------------------------------
// // Layer 3: Block-wise stride parameters
// //--------------------------------------------------------------------------
// #define RESNET_LAYER3_0_DS_STRIDE     2
// #define RESNET_LAYER3_0_CONV1_STRIDE  1
// #define RESNET_LAYER3_0_CONV2_STRIDE  2
// #define RESNET_LAYER3_0_CONV3_STRIDE  1
// #define RESNET_LAYER3_CONV1_STRIDE    1
// #define RESNET_LAYER3_CONV2_STRIDE    1
// #define RESNET_LAYER3_CONV3_STRIDE    1

// //--------------------------------------------------------------------------
// // Layer 4: Sizes of botteleck layers' input feature maps
// //--------------------------------------------------------------------------
// #define RESNET_LAYER4_0_FM_HEIGHT 46
// #define RESNET_LAYER4_0_FM_WIDTH  80
// #define RESNET_LAYER4_FM_HEIGHT   23
// #define RESNET_LAYER4_FM_WIDTH    40

// #define RESNET_LAYER4_0_DS_IN_CH  1024
// #define RESNET_LAYER4_0_DS_OUT_CH 2048

// #define RESNET_LAYER4_0_CONV1_IN_CH  1024
// #define RESNET_LAYER4_0_CONV1_OUT_CH  512
// #define RESNET_LAYER4_0_CONV2_IN_CH   512
// #define RESNET_LAYER4_0_CONV2_OUT_CH  512
// #define RESNET_LAYER4_0_CONV3_IN_CH   512
// #define RESNET_LAYER4_0_CONV3_OUT_CH 2048

// #define RESNET_LAYER4_CONV1_IN_CH  2048
// #define RESNET_LAYER4_CONV1_OUT_CH  512
// #define RESNET_LAYER4_CONV2_IN_CH   512
// #define RESNET_LAYER4_CONV2_OUT_CH  512
// #define RESNET_LAYER4_CONV3_IN_CH   512
// #define RESNET_LAYER4_CONV3_OUT_CH 2048

// //--------------------------------------------------------------------------
// // Layer 4: Block-wise stride parameters
// //--------------------------------------------------------------------------
// #define RESNET_LAYER4_0_DS_STRIDE     2
// #define RESNET_LAYER4_0_CONV1_STRIDE  1
// #define RESNET_LAYER4_0_CONV2_STRIDE  2
// #define RESNET_LAYER4_0_CONV3_STRIDE  1
// #define RESNET_LAYER4_CONV1_STRIDE    1
// #define RESNET_LAYER4_CONV2_STRIDE    1
// #define RESNET_LAYER4_CONV3_STRIDE    1

//--------------------------------------------------------------------------
// FPN
//--------------------------------------------------------------------------
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

// //--------------------------------------------------------------------------
// // RPN
// //--------------------------------------------------------------------------
// #define RPN_IN_BUF_CH        64
// #define RPN_IN_BUF_ROWS      46
// #define RPN_IN_BUF_COLS      40

// #define RPN_OUT_BUF_CH       64
// #define RPN_OUT_BUF_ROWS     46
// #define RPN_OUT_BUF_COLS     40

// #define RPN_LAST_LAYER_ENABLE   1
// #define RPN_LAST_LAYER_DISABLE  0

// //--------------------------------------------------------------------------
// // RPN Convolution Stride and Padding Declarations
// //--------------------------------------------------------------------------

// #define RPN_CONV_STRIDE 1
// #define RPN_CONV_PADDING 1
// #define RPN_CLS_STRIDE 1
// #define RPN_CLS_PADDING 0
// #define RPN_REG_STRIDE 1
// #define RPN_REG_PADDING 0


// //--------------------------------------------------------------------------
// // RPN Convolution Size Declarations
// //--------------------------------------------------------------------------

// #define RPN_INPUT0_IN_FM_HEIGHT 184
// #define RPN_INPUT0_IN_FM_WIDTH 320
// #define RPN_INPUT1_IN_FM_HEIGHT 92
// #define RPN_INPUT1_IN_FM_WIDTH 160
// #define RPN_INPUT2_IN_FM_HEIGHT 46
// #define RPN_INPUT2_IN_FM_WIDTH 80
// #define RPN_INPUT3_IN_FM_HEIGHT 23
// #define RPN_INPUT3_IN_FM_WIDTH 40
// #define RPN_INPUT4_IN_FM_HEIGHT 12
// #define RPN_INPUT4_IN_FM_WIDTH 20


// #define RPN_CONV_IN_CH 256
// #define RPN_CONV_OUT_CH 256
// #define RPN_CLS_IN_CH 256
// #define RPN_CLS_OUT_CH 3
// #define RPN_REG_IN_CH 256
// #define RPN_REG_OUT_CH 12



// #define RPN_ANCHORS0_IN_FM 176640
// #define RPN_ANCHORS1_IN_FM 44160
// #define RPN_ANCHORS2_IN_FM 11040
// #define RPN_ANCHORS3_IN_FM 2760
// #define RPN_ANCHORS4_IN_FM 720

// #define RPN_PRE_NMS_SIZE0 1000//((RPN_ANCHORS0_IN_FM) > (1000) ? (RPN_ANCHORS0_IN_FM) : (1000))
// #define RPN_PRE_NMS_SIZE1 1000//((RPN_ANCHORS1_IN_FM) > (1000) ? (RPN_ANCHORS1_IN_FM) : (1000))
// #define RPN_PRE_NMS_SIZE2 1000//((RPN_ANCHORS2_IN_FM) > (1000) ? (RPN_ANCHORS2_IN_FM) : (1000))
// #define RPN_PRE_NMS_SIZE3 1000//((RPN_ANCHORS3_IN_FM) > (1000) ? (RPN_ANCHORS3_IN_FM) : (1000))
// #define RPN_PRE_NMS_SIZE4 720//((RPN_ANCHORS4_IN_FM) > (1000) ? (RPN_ANCHORS4_IN_FM) : (1000))

// #define RPN_PRE_NMS_SIZE 4720
// #define IOU_THRESHOLD 0.7

// //--------------------------------------------------------------------------
// // ResNet-50 Function Declarations
// //--------------------------------------------------------------------------
// template<const int RESNET_IN_FM_DEPTH, const int RESNET_IN_FM_HEIGHT, const int RESNET_IN_FM_WIDTH,
//          const int RESNET_LAST_LAYER_EN>
// void resnet_load_input_fm_tile (
//         fm_t in_fm_buf[RESNET_IN_BUF_CH][RESNET_IN_BUF_ROWS + 5][RESNET_IN_BUF_COLS + 5], 
//         fm_t in_fm[2048][184][320], 
//         int   ti, 
//         int   tj, 
//         int   P,
//         int   d
// );

// template<const int RESNET_OUT_CH, const int RESNET_IN_CH>
// void resnet_load_weights_1x1 (
//         wt_t weight_buf_1x1[RESNET_IN_BUF_CH],
//         wt_t weights[RESNET_OUT_CH][RESNET_IN_CH],
//         int f,
//         int b,
//         int d
// );

// template<const int RESNET_OUT_CH, const int RESNET_IN_CH>
// void resnet_load_weights_3x3 (
//         wt_t weight_buf_3x3[RESNET_IN_BUF_CH][3][3],
//         wt_t weights[RESNET_OUT_CH][RESNET_IN_CH][3][3],
//         int f,
//         int b,
//         int d
// );

// void resnet_conv_7x7 (
//         fm_t Y_buf[RESNET_OUT_BUF_CH][RESNET_OUT_BUF_ROWS][RESNET_OUT_BUF_COLS], 
//         fm_t X_buf[RESNET_IN_BUF_CH][RESNET_IN_BUF_ROWS + 5][RESNET_IN_BUF_COLS + 5],
//         wt_t W_buf[RESNET_IN_BUF_CH][7][7],
//         int f
// );

// template<const int RESNET_OUT_CH>
// void resnet_load_batchnorm_params (
//         fm_t param_buf[3][RESNET_OUT_BUF_CH], 
//         wt_t params[3][RESNET_OUT_CH], 
//         int b
// );

// template<const int S>
// void resnet_store_out_buf_to_DDR(
//         fm_t out_fm[2048][184][320], 
//         fm_t out_fm_buf[RESNET_OUT_BUF_CH][RESNET_OUT_BUF_ROWS][RESNET_OUT_BUF_COLS], 
//         int ti, 
//         int tj,
//         int b
// );

// void resnet_conv_1x1 (
//         fm_t Y_buf[RESNET_OUT_BUF_CH][RESNET_OUT_BUF_ROWS][RESNET_OUT_BUF_COLS], 
//         fm_t X_buf[RESNET_IN_BUF_CH][RESNET_IN_BUF_ROWS + 5][RESNET_IN_BUF_COLS + 5],
//         wt_t W_buf[RESNET_IN_BUF_CH],
//         int f,
//         int S
// );

// //TODO: template<const int S>
// void resnet_conv_3x3 (
//         fm_t Y_buf[RESNET_OUT_BUF_CH][RESNET_OUT_BUF_ROWS][RESNET_OUT_BUF_COLS], 
//         fm_t X_buf[RESNET_IN_BUF_CH][RESNET_IN_BUF_ROWS + 5][RESNET_IN_BUF_COLS + 5],
//         wt_t W_buf[RESNET_IN_BUF_CH][3][3],
//         int f,
//         int S
// );

// void resnet_batchnorm(
//         fm_t feature_map[RESNET_OUT_BUF_CH][RESNET_OUT_BUF_ROWS][RESNET_OUT_BUF_COLS], 
//         wt_t bn_params[3][RESNET_OUT_BUF_CH],
//         bool  enable_relu
// );

// void resnet_add_residual_fm(
//         fm_t feature_map[RESNET_OUT_BUF_CH][RESNET_OUT_BUF_ROWS][RESNET_OUT_BUF_COLS], 
//         fm_t residual_map[RESNET_OUT_BUF_CH][RESNET_OUT_BUF_ROWS][RESNET_OUT_BUF_COLS],
//         bool  enable_relu
// );

// template<const int RESNET_IN_FM_DEPTH,  const int RESNET_IN_FM_HEIGHT,  const int RESNET_IN_FM_WIDTH,
//          const int RESNET_OUT_FM_DEPTH, const int RESNET_OUT_FM_HEIGHT, const int RESNET_OUT_FM_WIDTH,
//          const int STRIDE, const int RESNET_LAYER_EN>
// void resnet_bottleneck_conv1_bn1(
//         wt_t conv_weights[RESNET_OUT_FM_DEPTH][RESNET_IN_FM_DEPTH],
//         wt_t bn_params[3][RESNET_OUT_FM_DEPTH],
//         bool  enable_relu
// );

// template<const int RESNET_IN_FM_DEPTH,  const int RESNET_IN_FM_HEIGHT,  const int RESNET_IN_FM_WIDTH,
//          const int RESNET_OUT_FM_DEPTH, const int RESNET_OUT_FM_HEIGHT, const int RESNET_OUT_FM_WIDTH,
//          const int STRIDE, const int RESNET_LAYER_EN>
// void resnet_bottleneck_conv2_bn2_relu(
//         wt_t conv_weights[RESNET_OUT_FM_DEPTH][RESNET_IN_FM_DEPTH][3][3],
//         wt_t bn_params[3][RESNET_OUT_FM_DEPTH]
// );

// template<const int  RESNET_IN_FM_DEPTH, const int  RESNET_IN_FM_HEIGHT, const int  RESNET_IN_FM_WIDTH,
//          const int RESNET_OUT_FM_DEPTH, const int RESNET_OUT_FM_HEIGHT, const int RESNET_OUT_FM_WIDTH,
//          const int STRIDE, const int RESNET_LAYER_EN>
// void resnet_bottleneck_conv3_bn3_relu(
//         wt_t conv_weights[RESNET_OUT_FM_DEPTH][RESNET_IN_FM_DEPTH],
//         wt_t bn_params[3][RESNET_OUT_FM_DEPTH]
// );

// //--------------------------------------------------------------------------
// // Layer 0
// //--------------------------------------------------------------------------
// void resnet_layer0(
//         fm_t   resnet_layer0_input_fm[RESNET_LAYER0_CONV1_IN_CH][RESNET_LAYER0_IN_FM_HEIGHT][RESNET_LAYER0_IN_FM_WIDTH],
//         wt_t   resnet_layer0_conv1_weights[RESNET_LAYER0_CONV1_OUT_CH][RESNET_LAYER0_CONV1_IN_CH][7][7],
// 	    wt_t   resnet_layer0_bn1_params[3][RESNET_LAYER0_CONV1_OUT_CH],
//         fm_t   resnet_layer0_output_fm[RESNET_LAYER0_CONV1_OUT_CH][RESNET_LAYER0_OUT_FM_HEIGHT][RESNET_LAYER0_OUT_FM_WIDTH]
// );

// //--------------------------------------------------------------------------
// // Layer 1
// //--------------------------------------------------------------------------
// void resnet_layer1(
//         fm_t   resnet_layer1_input_fm[RESNET_LAYER1_0_CONV1_IN_CH][RESNET_LAYER1_0_FM_HEIGHT][RESNET_LAYER1_0_FM_WIDTH],
//         wt_t   resnet_layer1_0_conv1_weights[RESNET_LAYER1_0_CONV1_OUT_CH][RESNET_LAYER1_0_CONV1_IN_CH],
// 	    wt_t   resnet_layer1_0_bn1_params[3][RESNET_LAYER1_0_CONV1_OUT_CH],
//         wt_t   resnet_layer1_0_conv2_weights[RESNET_LAYER1_0_CONV2_OUT_CH][RESNET_LAYER1_0_CONV2_IN_CH][3][3],
// 	    wt_t   resnet_layer1_0_bn2_params[3][RESNET_LAYER1_0_CONV2_OUT_CH],
//         wt_t   resnet_layer1_0_conv3_weights[RESNET_LAYER1_0_CONV3_OUT_CH][RESNET_LAYER1_0_CONV3_IN_CH],
// 	    wt_t   resnet_layer1_0_bn3_params[3][RESNET_LAYER1_0_CONV3_OUT_CH],
//         wt_t   resnet_layer1_0_downsample_0_weights[RESNET_LAYER1_0_DS_OUT_CH][RESNET_LAYER1_0_DS_IN_CH],
// 	    wt_t   resnet_layer1_0_downsample_1_params[3][RESNET_LAYER1_0_DS_OUT_CH],
//         wt_t   resnet_layer1_1_conv1_weights[RESNET_LAYER1_CONV1_OUT_CH][RESNET_LAYER1_CONV1_IN_CH],
// 	    wt_t   resnet_layer1_1_bn1_params[3][RESNET_LAYER1_CONV1_OUT_CH],
//         wt_t   resnet_layer1_1_conv2_weights[RESNET_LAYER1_CONV2_OUT_CH][RESNET_LAYER1_CONV2_IN_CH][3][3],
// 	    wt_t   resnet_layer1_1_bn2_params[3][RESNET_LAYER1_CONV2_OUT_CH],
//         wt_t   resnet_layer1_1_conv3_weights[RESNET_LAYER1_CONV3_OUT_CH][RESNET_LAYER1_CONV3_IN_CH],
// 	    wt_t   resnet_layer1_1_bn3_params[3][RESNET_LAYER1_CONV3_OUT_CH],
//         wt_t   resnet_layer1_2_conv1_weights[RESNET_LAYER1_CONV1_OUT_CH][RESNET_LAYER1_CONV1_IN_CH],
// 	    wt_t   resnet_layer1_2_bn1_params[3][RESNET_LAYER1_CONV1_OUT_CH],
//         wt_t   resnet_layer1_2_conv2_weights[RESNET_LAYER1_CONV2_OUT_CH][RESNET_LAYER1_CONV2_IN_CH][3][3],
// 	    wt_t   resnet_layer1_2_bn2_params[3][RESNET_LAYER1_CONV2_OUT_CH],
//         wt_t   resnet_layer1_2_conv3_weights[RESNET_LAYER1_CONV3_OUT_CH][RESNET_LAYER1_CONV3_IN_CH],
// 	    wt_t   resnet_layer1_2_bn3_params[3][RESNET_LAYER1_CONV3_OUT_CH],
//         fm_t   resnet_layer1_output_fm[RESNET_LAYER1_0_DS_OUT_CH][RESNET_LAYER1_0_FM_HEIGHT][RESNET_LAYER1_0_FM_WIDTH]
// );

// //--------------------------------------------------------------------------
// // Layer 2
// //--------------------------------------------------------------------------
// void resnet_layer2(
//         fm_t   resnet_layer2_input_fm[RESNET_LAYER2_0_CONV1_IN_CH][RESNET_LAYER2_0_FM_HEIGHT][RESNET_LAYER2_0_FM_WIDTH],
//         wt_t   resnet_layer2_0_conv1_weights[RESNET_LAYER2_0_CONV1_OUT_CH][RESNET_LAYER2_0_CONV1_IN_CH],
// 	    wt_t   resnet_layer2_0_bn1_params[3][RESNET_LAYER2_0_CONV1_OUT_CH],
//         wt_t   resnet_layer2_0_conv2_weights[RESNET_LAYER2_0_CONV2_OUT_CH][RESNET_LAYER2_0_CONV2_IN_CH][3][3],
// 	    wt_t   resnet_layer2_0_bn2_params[3][RESNET_LAYER2_0_CONV2_OUT_CH],
//         wt_t   resnet_layer2_0_conv3_weights[RESNET_LAYER2_0_CONV3_OUT_CH][RESNET_LAYER2_0_CONV3_IN_CH],
// 	    wt_t   resnet_layer2_0_bn3_params[3][RESNET_LAYER2_0_CONV3_OUT_CH],
//         wt_t   resnet_layer2_0_downsample_0_weights[RESNET_LAYER2_0_DS_OUT_CH][RESNET_LAYER2_0_DS_IN_CH],
// 	    wt_t   resnet_layer2_0_downsample_1_params[3][RESNET_LAYER2_0_DS_OUT_CH],
//         wt_t   resnet_layer2_1_conv1_weights[RESNET_LAYER2_CONV1_OUT_CH][RESNET_LAYER2_CONV1_IN_CH],
// 	    wt_t   resnet_layer2_1_bn1_params[3][RESNET_LAYER2_CONV1_OUT_CH],
//         wt_t   resnet_layer2_1_conv2_weights[RESNET_LAYER2_CONV2_OUT_CH][RESNET_LAYER2_CONV2_IN_CH][3][3],
// 	    wt_t   resnet_layer2_1_bn2_params[3][RESNET_LAYER2_CONV2_OUT_CH],
//         wt_t   resnet_layer2_1_conv3_weights[RESNET_LAYER2_CONV3_OUT_CH][RESNET_LAYER2_CONV3_IN_CH],
// 	    wt_t   resnet_layer2_1_bn3_params[3][RESNET_LAYER2_CONV3_OUT_CH],
//         wt_t   resnet_layer2_2_conv1_weights[RESNET_LAYER2_CONV1_OUT_CH][RESNET_LAYER2_CONV1_IN_CH],
// 	    wt_t   resnet_layer2_2_bn1_params[3][RESNET_LAYER2_CONV1_OUT_CH],
//         wt_t   resnet_layer2_2_conv2_weights[RESNET_LAYER2_CONV2_OUT_CH][RESNET_LAYER2_CONV2_IN_CH][3][3],
// 	    wt_t   resnet_layer2_2_bn2_params[3][RESNET_LAYER2_CONV2_OUT_CH],
//         wt_t   resnet_layer2_2_conv3_weights[RESNET_LAYER2_CONV3_OUT_CH][RESNET_LAYER2_CONV3_IN_CH],
// 	    wt_t   resnet_layer2_2_bn3_params[3][RESNET_LAYER2_CONV3_OUT_CH],
//         wt_t   resnet_layer2_3_conv1_weights[RESNET_LAYER2_CONV1_OUT_CH][RESNET_LAYER2_CONV1_IN_CH],
// 	    wt_t   resnet_layer2_3_bn1_params[3][RESNET_LAYER2_CONV1_OUT_CH],
//         wt_t   resnet_layer2_3_conv2_weights[RESNET_LAYER2_CONV2_OUT_CH][RESNET_LAYER2_CONV2_IN_CH][3][3],
// 	    wt_t   resnet_layer2_3_bn2_params[3][RESNET_LAYER2_CONV2_OUT_CH],
//         wt_t   resnet_layer2_3_conv3_weights[RESNET_LAYER2_CONV3_OUT_CH][RESNET_LAYER2_CONV3_IN_CH],
// 	    wt_t   resnet_layer2_3_bn3_params[3][RESNET_LAYER2_CONV3_OUT_CH],
//         fm_t   resnet_layer2_output_fm[RESNET_LAYER2_0_DS_OUT_CH][RESNET_LAYER2_0_FM_HEIGHT][RESNET_LAYER2_0_FM_WIDTH]
// );

// //--------------------------------------------------------------------------
// // Layer 3
// //--------------------------------------------------------------------------
// void resnet_layer3(
//         fm_t   resnet_layer3_input_fm[RESNET_LAYER3_0_CONV1_IN_CH][RESNET_LAYER3_0_FM_HEIGHT][RESNET_LAYER3_0_FM_WIDTH],
//         wt_t   resnet_layer3_0_conv1_weights[RESNET_LAYER3_0_CONV1_OUT_CH][RESNET_LAYER3_0_CONV1_IN_CH],
// 	    wt_t   resnet_layer3_0_bn1_params[3][RESNET_LAYER3_0_CONV1_OUT_CH],
//         wt_t   resnet_layer3_0_conv2_weights[RESNET_LAYER3_0_CONV2_OUT_CH][RESNET_LAYER3_0_CONV2_IN_CH][3][3],
// 	    wt_t   resnet_layer3_0_bn2_params[3][RESNET_LAYER3_0_CONV2_OUT_CH],
//         wt_t   resnet_layer3_0_conv3_weights[RESNET_LAYER3_0_CONV3_OUT_CH][RESNET_LAYER3_0_CONV3_IN_CH],
// 	    wt_t   resnet_layer3_0_bn3_params[3][RESNET_LAYER3_0_CONV3_OUT_CH],
//         wt_t   resnet_layer3_0_downsample_0_weights[RESNET_LAYER3_0_DS_OUT_CH][RESNET_LAYER3_0_DS_IN_CH],
// 	    wt_t   resnet_layer3_0_downsample_1_params[3][RESNET_LAYER3_0_DS_OUT_CH],
//         wt_t   resnet_layer3_1_conv1_weights[RESNET_LAYER3_CONV1_OUT_CH][RESNET_LAYER3_CONV1_IN_CH],
// 	    wt_t   resnet_layer3_1_bn1_params[3][RESNET_LAYER3_CONV1_OUT_CH],
//         wt_t   resnet_layer3_1_conv2_weights[RESNET_LAYER3_CONV2_OUT_CH][RESNET_LAYER3_CONV2_IN_CH][3][3],
// 	    wt_t   resnet_layer3_1_bn2_params[3][RESNET_LAYER3_CONV2_OUT_CH],
//         wt_t   resnet_layer3_1_conv3_weights[RESNET_LAYER3_CONV3_OUT_CH][RESNET_LAYER3_CONV3_IN_CH],
// 	    wt_t   resnet_layer3_1_bn3_params[3][RESNET_LAYER3_CONV3_OUT_CH],
//         wt_t   resnet_layer3_2_conv1_weights[RESNET_LAYER3_CONV1_OUT_CH][RESNET_LAYER3_CONV1_IN_CH],
// 	    wt_t   resnet_layer3_2_bn1_params[3][RESNET_LAYER3_CONV1_OUT_CH],
//         wt_t   resnet_layer3_2_conv2_weights[RESNET_LAYER3_CONV2_OUT_CH][RESNET_LAYER3_CONV2_IN_CH][3][3],
// 	    wt_t   resnet_layer3_2_bn2_params[3][RESNET_LAYER3_CONV2_OUT_CH],
//         wt_t   resnet_layer3_2_conv3_weights[RESNET_LAYER3_CONV3_OUT_CH][RESNET_LAYER3_CONV3_IN_CH],
// 	    wt_t   resnet_layer3_2_bn3_params[3][RESNET_LAYER3_CONV3_OUT_CH],
//         wt_t   resnet_layer3_3_conv1_weights[RESNET_LAYER3_CONV1_OUT_CH][RESNET_LAYER3_CONV1_IN_CH],
// 	    wt_t   resnet_layer3_3_bn1_params[3][RESNET_LAYER3_CONV1_OUT_CH],
//         wt_t   resnet_layer3_3_conv2_weights[RESNET_LAYER3_CONV2_OUT_CH][RESNET_LAYER3_CONV2_IN_CH][3][3],
// 	    wt_t   resnet_layer3_3_bn2_params[3][RESNET_LAYER3_CONV2_OUT_CH],
//         wt_t   resnet_layer3_3_conv3_weights[RESNET_LAYER3_CONV3_OUT_CH][RESNET_LAYER3_CONV3_IN_CH],
// 	    wt_t   resnet_layer3_3_bn3_params[3][RESNET_LAYER3_CONV3_OUT_CH],
//         wt_t   resnet_layer3_4_conv1_weights[RESNET_LAYER3_CONV1_OUT_CH][RESNET_LAYER3_CONV1_IN_CH],
// 	    wt_t   resnet_layer3_4_bn1_params[3][RESNET_LAYER3_CONV1_OUT_CH],
//         wt_t   resnet_layer3_4_conv2_weights[RESNET_LAYER3_CONV2_OUT_CH][RESNET_LAYER3_CONV2_IN_CH][3][3],
// 	    wt_t   resnet_layer3_4_bn2_params[3][RESNET_LAYER3_CONV2_OUT_CH],
//         wt_t   resnet_layer3_4_conv3_weights[RESNET_LAYER3_CONV3_OUT_CH][RESNET_LAYER3_CONV3_IN_CH],
// 	    wt_t   resnet_layer3_4_bn3_params[3][RESNET_LAYER3_CONV3_OUT_CH],
//         wt_t   resnet_layer3_5_conv1_weights[RESNET_LAYER3_CONV1_OUT_CH][RESNET_LAYER3_CONV1_IN_CH],
// 	    wt_t   resnet_layer3_5_bn1_params[3][RESNET_LAYER3_CONV1_OUT_CH],
//         wt_t   resnet_layer3_5_conv2_weights[RESNET_LAYER3_CONV2_OUT_CH][RESNET_LAYER3_CONV2_IN_CH][3][3],
// 	    wt_t   resnet_layer3_5_bn2_params[3][RESNET_LAYER3_CONV2_OUT_CH],
//         wt_t   resnet_layer3_5_conv3_weights[RESNET_LAYER3_CONV3_OUT_CH][RESNET_LAYER3_CONV3_IN_CH],
// 	    wt_t   resnet_layer3_5_bn3_params[3][RESNET_LAYER3_CONV3_OUT_CH],
//         fm_t   resnet_layer3_output_fm[RESNET_LAYER3_0_DS_OUT_CH][RESNET_LAYER3_0_FM_HEIGHT][RESNET_LAYER3_0_FM_WIDTH]
// );

// //--------------------------------------------------------------------------
// // Layer 4
// //--------------------------------------------------------------------------
// void resnet_layer4(
//         fm_t   resnet_layer4_input_fm[RESNET_LAYER4_0_CONV1_IN_CH][RESNET_LAYER4_0_FM_HEIGHT][RESNET_LAYER4_0_FM_WIDTH],
//         wt_t   resnet_layer4_0_conv1_weights[RESNET_LAYER4_0_CONV1_OUT_CH][RESNET_LAYER4_0_CONV1_IN_CH],
// 	    wt_t   resnet_layer4_0_bn1_params[3][RESNET_LAYER4_0_CONV1_OUT_CH],
//         wt_t   resnet_layer4_0_conv2_weights[RESNET_LAYER4_0_CONV2_OUT_CH][RESNET_LAYER4_0_CONV2_IN_CH][3][3],
// 	    wt_t   resnet_layer4_0_bn2_params[3][RESNET_LAYER4_0_CONV2_OUT_CH],
//         wt_t   resnet_layer4_0_conv3_weights[RESNET_LAYER4_0_CONV3_OUT_CH][RESNET_LAYER4_0_CONV3_IN_CH],
// 	    wt_t   resnet_layer4_0_bn3_params[3][RESNET_LAYER4_0_CONV3_OUT_CH],
//         wt_t   resnet_layer4_0_downsample_0_weights[RESNET_LAYER4_0_DS_OUT_CH][RESNET_LAYER4_0_DS_IN_CH],
// 	    wt_t   resnet_layer4_0_downsample_1_params[3][RESNET_LAYER4_0_DS_OUT_CH],
//         wt_t   resnet_layer4_1_conv1_weights[RESNET_LAYER4_CONV1_OUT_CH][RESNET_LAYER4_CONV1_IN_CH],
// 	    wt_t   resnet_layer4_1_bn1_params[3][RESNET_LAYER4_CONV1_OUT_CH],
//         wt_t   resnet_layer4_1_conv2_weights[RESNET_LAYER4_CONV2_OUT_CH][RESNET_LAYER4_CONV2_IN_CH][3][3],
// 	    wt_t   resnet_layer4_1_bn2_params[3][RESNET_LAYER4_CONV2_OUT_CH],
//         wt_t   resnet_layer4_1_conv3_weights[RESNET_LAYER4_CONV3_OUT_CH][RESNET_LAYER4_CONV3_IN_CH],
// 	    wt_t   resnet_layer4_1_bn3_params[3][RESNET_LAYER4_CONV3_OUT_CH],
//         wt_t   resnet_layer4_2_conv1_weights[RESNET_LAYER4_CONV1_OUT_CH][RESNET_LAYER4_CONV1_IN_CH],
// 	    wt_t   resnet_layer4_2_bn1_params[3][RESNET_LAYER4_CONV1_OUT_CH],
//         wt_t   resnet_layer4_2_conv2_weights[RESNET_LAYER4_CONV2_OUT_CH][RESNET_LAYER4_CONV2_IN_CH][3][3],
// 	    wt_t   resnet_layer4_2_bn2_params[3][RESNET_LAYER4_CONV2_OUT_CH],
//         wt_t   resnet_layer4_2_conv3_weights[RESNET_LAYER4_CONV3_OUT_CH][RESNET_LAYER4_CONV3_IN_CH],
// 	    wt_t   resnet_layer4_2_bn3_params[3][RESNET_LAYER4_CONV3_OUT_CH],
//         fm_t   resnet_layer4_output_fm[RESNET_LAYER4_0_DS_OUT_CH][RESNET_LAYER4_0_FM_HEIGHT][RESNET_LAYER4_0_FM_WIDTH]
// );

// void resnet_top_1 (
//     fm_t   resnet_layer0_input_fm[RESNET_LAYER0_CONV1_IN_CH][RESNET_LAYER0_IN_FM_HEIGHT][RESNET_LAYER0_IN_FM_WIDTH],
//     wt_t   resnet_layer0_conv1_weights[RESNET_LAYER0_CONV1_OUT_CH][RESNET_LAYER0_CONV1_IN_CH][7][7],
//     wt_t   resnet_layer0_bn1_params[3][RESNET_LAYER0_CONV1_OUT_CH],
//     fm_t   resnet_layer0_output_fm[RESNET_LAYER0_CONV1_OUT_CH][RESNET_LAYER0_OUT_FM_HEIGHT][RESNET_LAYER0_OUT_FM_WIDTH],

//     fm_t   resnet_layer1_input_fm[RESNET_LAYER1_0_CONV1_IN_CH][RESNET_LAYER1_0_FM_HEIGHT][RESNET_LAYER1_0_FM_WIDTH],
//     wt_t   resnet_layer1_0_conv1_weights[RESNET_LAYER1_0_CONV1_OUT_CH][RESNET_LAYER1_0_CONV1_IN_CH],
//     wt_t   resnet_layer1_0_bn1_params[4][RESNET_LAYER1_0_CONV1_OUT_CH],
//     wt_t   resnet_layer1_0_conv2_weights[RESNET_LAYER1_0_CONV2_OUT_CH][RESNET_LAYER1_0_CONV2_IN_CH][3][3],
//     wt_t   resnet_layer1_0_bn2_params[4][RESNET_LAYER1_0_CONV2_OUT_CH],
//     wt_t   resnet_layer1_0_conv3_weights[RESNET_LAYER1_0_CONV3_OUT_CH][RESNET_LAYER1_0_CONV3_IN_CH],
//     wt_t   resnet_layer1_0_bn3_params[4][RESNET_LAYER1_0_CONV3_OUT_CH],
//     wt_t   resnet_layer1_0_downsample_0_weights[RESNET_LAYER1_0_DS_OUT_CH][RESNET_LAYER1_0_DS_IN_CH],
//     wt_t   resnet_layer1_0_downsample_1_params[4][RESNET_LAYER1_0_DS_OUT_CH],
//     wt_t   resnet_layer1_1_conv1_weights[RESNET_LAYER1_CONV1_OUT_CH][RESNET_LAYER1_CONV1_IN_CH],
//     wt_t   resnet_layer1_1_bn1_params[4][RESNET_LAYER1_CONV1_OUT_CH],
//     wt_t   resnet_layer1_1_conv2_weights[RESNET_LAYER1_CONV2_OUT_CH][RESNET_LAYER1_CONV2_IN_CH][3][3],
//     wt_t   resnet_layer1_1_bn2_params[4][RESNET_LAYER1_CONV2_OUT_CH],
//     wt_t   resnet_layer1_1_conv3_weights[RESNET_LAYER1_CONV3_OUT_CH][RESNET_LAYER1_CONV3_IN_CH],
//     wt_t   resnet_layer1_1_bn3_params[4][RESNET_LAYER1_CONV3_OUT_CH],
//     wt_t   resnet_layer1_2_conv1_weights[RESNET_LAYER1_CONV1_OUT_CH][RESNET_LAYER1_CONV1_IN_CH],
//     wt_t   resnet_layer1_2_bn1_params[4][RESNET_LAYER1_CONV1_OUT_CH],
//     wt_t   resnet_layer1_2_conv2_weights[RESNET_LAYER1_CONV2_OUT_CH][RESNET_LAYER1_CONV2_IN_CH][3][3],
//     wt_t   resnet_layer1_2_bn2_params[4][RESNET_LAYER1_CONV2_OUT_CH],
//     wt_t   resnet_layer1_2_conv3_weights[RESNET_LAYER1_CONV3_OUT_CH][RESNET_LAYER1_CONV3_IN_CH],
//     wt_t   resnet_layer1_2_bn3_params[4][RESNET_LAYER1_CONV3_OUT_CH],
//     fm_t   resnet_layer1_output_fm[RESNET_LAYER1_0_DS_OUT_CH][RESNET_LAYER1_0_FM_HEIGHT][RESNET_LAYER1_0_FM_WIDTH],

//     fm_t   resnet_layer2_input_fm[RESNET_LAYER2_0_CONV1_IN_CH][RESNET_LAYER2_0_FM_HEIGHT][RESNET_LAYER2_0_FM_WIDTH],
//     wt_t   resnet_layer2_0_conv1_weights[RESNET_LAYER2_0_CONV1_OUT_CH][RESNET_LAYER2_0_CONV1_IN_CH],
// 	wt_t   resnet_layer2_0_bn1_params[3][RESNET_LAYER2_0_CONV1_OUT_CH],
//     wt_t   resnet_layer2_0_conv2_weights[RESNET_LAYER2_0_CONV2_OUT_CH][RESNET_LAYER2_0_CONV2_IN_CH][3][3],
// 	wt_t   resnet_layer2_0_bn2_params[3][RESNET_LAYER2_0_CONV2_OUT_CH],
//     wt_t   resnet_layer2_0_conv3_weights[RESNET_LAYER2_0_CONV3_OUT_CH][RESNET_LAYER2_0_CONV3_IN_CH],
// 	wt_t   resnet_layer2_0_bn3_params[3][RESNET_LAYER2_0_CONV3_OUT_CH],
//     wt_t   resnet_layer2_0_downsample_0_weights[RESNET_LAYER2_0_DS_OUT_CH][RESNET_LAYER2_0_DS_IN_CH],
// 	wt_t   resnet_layer2_0_downsample_1_params[3][RESNET_LAYER2_0_DS_OUT_CH],
//     wt_t   resnet_layer2_1_conv1_weights[RESNET_LAYER2_CONV1_OUT_CH][RESNET_LAYER2_CONV1_IN_CH],
// 	wt_t   resnet_layer2_1_bn1_params[3][RESNET_LAYER2_CONV1_OUT_CH],
//     wt_t   resnet_layer2_1_conv2_weights[RESNET_LAYER2_CONV2_OUT_CH][RESNET_LAYER2_CONV2_IN_CH][3][3],
// 	wt_t   resnet_layer2_1_bn2_params[3][RESNET_LAYER2_CONV2_OUT_CH],
//     wt_t   resnet_layer2_1_conv3_weights[RESNET_LAYER2_CONV3_OUT_CH][RESNET_LAYER2_CONV3_IN_CH],
// 	wt_t   resnet_layer2_1_bn3_params[3][RESNET_LAYER2_CONV3_OUT_CH],
//     wt_t   resnet_layer2_2_conv1_weights[RESNET_LAYER2_CONV1_OUT_CH][RESNET_LAYER2_CONV1_IN_CH],
// 	wt_t   resnet_layer2_2_bn1_params[3][RESNET_LAYER2_CONV1_OUT_CH],
//     wt_t   resnet_layer2_2_conv2_weights[RESNET_LAYER2_CONV2_OUT_CH][RESNET_LAYER2_CONV2_IN_CH][3][3],
// 	wt_t   resnet_layer2_2_bn2_params[3][RESNET_LAYER2_CONV2_OUT_CH],
//     wt_t   resnet_layer2_2_conv3_weights[RESNET_LAYER2_CONV3_OUT_CH][RESNET_LAYER2_CONV3_IN_CH],
// 	wt_t   resnet_layer2_2_bn3_params[3][RESNET_LAYER2_CONV3_OUT_CH],
//     wt_t   resnet_layer2_3_conv1_weights[RESNET_LAYER2_CONV1_OUT_CH][RESNET_LAYER2_CONV1_IN_CH],
// 	wt_t   resnet_layer2_3_bn1_params[3][RESNET_LAYER2_CONV1_OUT_CH],
//     wt_t   resnet_layer2_3_conv2_weights[RESNET_LAYER2_CONV2_OUT_CH][RESNET_LAYER2_CONV2_IN_CH][3][3],
// 	wt_t   resnet_layer2_3_bn2_params[3][RESNET_LAYER2_CONV2_OUT_CH],
//     wt_t   resnet_layer2_3_conv3_weights[RESNET_LAYER2_CONV3_OUT_CH][RESNET_LAYER2_CONV3_IN_CH],
// 	wt_t   resnet_layer2_3_bn3_params[3][RESNET_LAYER2_CONV3_OUT_CH],
//     fm_t   resnet_layer2_output_fm[RESNET_LAYER2_0_DS_OUT_CH][RESNET_LAYER2_0_FM_HEIGHT][RESNET_LAYER2_0_FM_WIDTH]
// );

// void resnet_top_2(
//     fm_t   resnet_layer2_output_fm[RESNET_LAYER2_0_DS_OUT_CH][RESNET_LAYER2_0_FM_HEIGHT][RESNET_LAYER2_0_FM_WIDTH],
//     fm_t   resnet_layer3_input_fm[RESNET_LAYER3_0_CONV1_IN_CH][RESNET_LAYER3_0_FM_HEIGHT][RESNET_LAYER3_0_FM_WIDTH],
//     wt_t   resnet_layer3_0_conv1_weights[RESNET_LAYER3_0_CONV1_OUT_CH][RESNET_LAYER3_0_CONV1_IN_CH],
// 	wt_t   resnet_layer3_0_bn1_params[3][RESNET_LAYER3_0_CONV1_OUT_CH],
//     wt_t   resnet_layer3_0_conv2_weights[RESNET_LAYER3_0_CONV2_OUT_CH][RESNET_LAYER3_0_CONV2_IN_CH][3][3],
// 	wt_t   resnet_layer3_0_bn2_params[3][RESNET_LAYER3_0_CONV2_OUT_CH],
//     wt_t   resnet_layer3_0_conv3_weights[RESNET_LAYER3_0_CONV3_OUT_CH][RESNET_LAYER3_0_CONV3_IN_CH],
// 	wt_t   resnet_layer3_0_bn3_params[3][RESNET_LAYER3_0_CONV3_OUT_CH],
//     wt_t   resnet_layer3_0_downsample_0_weights[RESNET_LAYER3_0_DS_OUT_CH][RESNET_LAYER3_0_DS_IN_CH],
// 	wt_t   resnet_layer3_0_downsample_1_params[3][RESNET_LAYER3_0_DS_OUT_CH],
//     wt_t   resnet_layer3_1_conv1_weights[RESNET_LAYER3_CONV1_OUT_CH][RESNET_LAYER3_CONV1_IN_CH],
// 	wt_t   resnet_layer3_1_bn1_params[3][RESNET_LAYER3_CONV1_OUT_CH],
//     wt_t   resnet_layer3_1_conv2_weights[RESNET_LAYER3_CONV2_OUT_CH][RESNET_LAYER3_CONV2_IN_CH][3][3],
// 	wt_t   resnet_layer3_1_bn2_params[3][RESNET_LAYER3_CONV2_OUT_CH],
//     wt_t   resnet_layer3_1_conv3_weights[RESNET_LAYER3_CONV3_OUT_CH][RESNET_LAYER3_CONV3_IN_CH],
// 	wt_t   resnet_layer3_1_bn3_params[3][RESNET_LAYER3_CONV3_OUT_CH],
//     wt_t   resnet_layer3_2_conv1_weights[RESNET_LAYER3_CONV1_OUT_CH][RESNET_LAYER3_CONV1_IN_CH],
// 	wt_t   resnet_layer3_2_bn1_params[3][RESNET_LAYER3_CONV1_OUT_CH],
//     wt_t   resnet_layer3_2_conv2_weights[RESNET_LAYER3_CONV2_OUT_CH][RESNET_LAYER3_CONV2_IN_CH][3][3],
// 	wt_t   resnet_layer3_2_bn2_params[3][RESNET_LAYER3_CONV2_OUT_CH],
//     wt_t   resnet_layer3_2_conv3_weights[RESNET_LAYER3_CONV3_OUT_CH][RESNET_LAYER3_CONV3_IN_CH],
// 	wt_t   resnet_layer3_2_bn3_params[3][RESNET_LAYER3_CONV3_OUT_CH],
//     wt_t   resnet_layer3_3_conv1_weights[RESNET_LAYER3_CONV1_OUT_CH][RESNET_LAYER3_CONV1_IN_CH],
// 	wt_t   resnet_layer3_3_bn1_params[3][RESNET_LAYER3_CONV1_OUT_CH],
//     wt_t   resnet_layer3_3_conv2_weights[RESNET_LAYER3_CONV2_OUT_CH][RESNET_LAYER3_CONV2_IN_CH][3][3],
// 	wt_t   resnet_layer3_3_bn2_params[3][RESNET_LAYER3_CONV2_OUT_CH],
//     wt_t   resnet_layer3_3_conv3_weights[RESNET_LAYER3_CONV3_OUT_CH][RESNET_LAYER3_CONV3_IN_CH],
// 	wt_t   resnet_layer3_3_bn3_params[3][RESNET_LAYER3_CONV3_OUT_CH],
//     wt_t   resnet_layer3_4_conv1_weights[RESNET_LAYER3_CONV1_OUT_CH][RESNET_LAYER3_CONV1_IN_CH],
// 	wt_t   resnet_layer3_4_bn1_params[3][RESNET_LAYER3_CONV1_OUT_CH],
//     wt_t   resnet_layer3_4_conv2_weights[RESNET_LAYER3_CONV2_OUT_CH][RESNET_LAYER3_CONV2_IN_CH][3][3],
// 	wt_t   resnet_layer3_4_bn2_params[3][RESNET_LAYER3_CONV2_OUT_CH],
//     wt_t   resnet_layer3_4_conv3_weights[RESNET_LAYER3_CONV3_OUT_CH][RESNET_LAYER3_CONV3_IN_CH],
// 	wt_t   resnet_layer3_4_bn3_params[3][RESNET_LAYER3_CONV3_OUT_CH],
//     wt_t   resnet_layer3_5_conv1_weights[RESNET_LAYER3_CONV1_OUT_CH][RESNET_LAYER3_CONV1_IN_CH],
// 	wt_t   resnet_layer3_5_bn1_params[3][RESNET_LAYER3_CONV1_OUT_CH],
//     wt_t   resnet_layer3_5_conv2_weights[RESNET_LAYER3_CONV2_OUT_CH][RESNET_LAYER3_CONV2_IN_CH][3][3],
// 	wt_t   resnet_layer3_5_bn2_params[3][RESNET_LAYER3_CONV2_OUT_CH],
//     wt_t   resnet_layer3_5_conv3_weights[RESNET_LAYER3_CONV3_OUT_CH][RESNET_LAYER3_CONV3_IN_CH],
// 	wt_t   resnet_layer3_5_bn3_params[3][RESNET_LAYER3_CONV3_OUT_CH],
//     fm_t   resnet_layer3_output_fm[RESNET_LAYER3_0_DS_OUT_CH][RESNET_LAYER3_0_FM_HEIGHT][RESNET_LAYER3_0_FM_WIDTH],

//     fm_t   resnet_layer4_input_fm[RESNET_LAYER4_0_CONV1_IN_CH][RESNET_LAYER4_0_FM_HEIGHT][RESNET_LAYER4_0_FM_WIDTH],
//     wt_t   resnet_layer4_0_conv1_weights[RESNET_LAYER4_0_CONV1_OUT_CH][RESNET_LAYER4_0_CONV1_IN_CH],
// 	wt_t   resnet_layer4_0_bn1_params[3][RESNET_LAYER4_0_CONV1_OUT_CH],
//     wt_t   resnet_layer4_0_conv2_weights[RESNET_LAYER4_0_CONV2_OUT_CH][RESNET_LAYER4_0_CONV2_IN_CH][3][3],
// 	wt_t   resnet_layer4_0_bn2_params[3][RESNET_LAYER4_0_CONV2_OUT_CH],
//     wt_t   resnet_layer4_0_conv3_weights[RESNET_LAYER4_0_CONV3_OUT_CH][RESNET_LAYER4_0_CONV3_IN_CH],
// 	wt_t   resnet_layer4_0_bn3_params[3][RESNET_LAYER4_0_CONV3_OUT_CH],
//     wt_t   resnet_layer4_0_downsample_0_weights[RESNET_LAYER4_0_DS_OUT_CH][RESNET_LAYER4_0_DS_IN_CH],
// 	wt_t   resnet_layer4_0_downsample_1_params[3][RESNET_LAYER4_0_DS_OUT_CH],
//     wt_t   resnet_layer4_1_conv1_weights[RESNET_LAYER4_CONV1_OUT_CH][RESNET_LAYER4_CONV1_IN_CH],
// 	wt_t   resnet_layer4_1_bn1_params[3][RESNET_LAYER4_CONV1_OUT_CH],
//     wt_t   resnet_layer4_1_conv2_weights[RESNET_LAYER4_CONV2_OUT_CH][RESNET_LAYER4_CONV2_IN_CH][3][3],
// 	wt_t   resnet_layer4_1_bn2_params[3][RESNET_LAYER4_CONV2_OUT_CH],
//     wt_t   resnet_layer4_1_conv3_weights[RESNET_LAYER4_CONV3_OUT_CH][RESNET_LAYER4_CONV3_IN_CH],
// 	wt_t   resnet_layer4_1_bn3_params[3][RESNET_LAYER4_CONV3_OUT_CH],
//     wt_t   resnet_layer4_2_conv1_weights[RESNET_LAYER4_CONV1_OUT_CH][RESNET_LAYER4_CONV1_IN_CH],
// 	wt_t   resnet_layer4_2_bn1_params[3][RESNET_LAYER4_CONV1_OUT_CH],
//     wt_t   resnet_layer4_2_conv2_weights[RESNET_LAYER4_CONV2_OUT_CH][RESNET_LAYER4_CONV2_IN_CH][3][3],
// 	wt_t   resnet_layer4_2_bn2_params[3][RESNET_LAYER4_CONV2_OUT_CH],
//     wt_t   resnet_layer4_2_conv3_weights[RESNET_LAYER4_CONV3_OUT_CH][RESNET_LAYER4_CONV3_IN_CH],
// 	wt_t   resnet_layer4_2_bn3_params[3][RESNET_LAYER4_CONV3_OUT_CH],
//     fm_t   resnet_layer4_output_fm[RESNET_LAYER4_0_DS_OUT_CH][RESNET_LAYER4_0_FM_HEIGHT][RESNET_LAYER4_0_FM_WIDTH]
// );

//--------------------------------------------------------------------------
// FPN Function Declarations
//--------------------------------------------------------------------------
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

// //--------------------------------------------------------------------------
// // RPN
// //--------------------------------------------------------------------------
// template<const int N_OUT_CH>
// void rpn_load_bias_params (
//         fm_t param_buf[RPN_OUT_BUF_CH], 
//         wt_t params[N_OUT_CH], 
//         int b
// );


// // void rpn_conv_bias_add(
// //         fm_t feature_map[RPN_OUT_BUF_CH][RPN_OUT_BUF_ROWS][RPN_OUT_BUF_COLS], 
// //         wt_t bias_params[RPN_OUT_BUF_CH],
// //         bool relu
// // );



// template<const int N_IN_FM_DEPTH, const int N_IN_FM_HEIGHT, const int N_IN_FM_WIDTH,
//          const int LAST_LAYER_EN>
// void rpn_load_input_fm_tile (
//         fm_t in_fm_buf[RPN_IN_BUF_CH][RPN_IN_BUF_ROWS + 5][RPN_IN_BUF_COLS + 5], 
//         fm_t in_fm[2048][184][320], 
//         int   ti, 
//         int   tj, 
//         int   P,
//         int   d
// );

// // void rpn_load_residual_fm_tile (
// //         fm_t in_fm_buf[RPN_IN_BUF_CH][RPN_IN_BUF_ROWS][RPN_IN_BUF_COLS], 
// //         fm_t in_fm[2048][184][320], 
// //         int   ti, 
// //         int   tj,
// //         int   b 
// // );


// template<const int N_OUT_CH, const int N_IN_CH>
// void rpn_load_weights_3x3 (
//         wt_t weight_buf_3x3[RPN_IN_BUF_CH][3][3],
//         wt_t weights[N_OUT_CH][N_IN_CH][3][3],
//         int f,
//         int b,
//         int d
// );


// // void rpn_save_partial_out_buf (
// //         fm_t partial_out_fm_buf[RPN_OUT_BUF_CH][RPN_OUT_BUF_ROWS][RPN_OUT_BUF_COLS], 
// //         fm_t out_fm_buf[RPN_OUT_BUF_CH][RPN_OUT_BUF_ROWS][RPN_OUT_BUF_COLS],
// //         int d
// // );

// template<const int S>
// void rpn_store_out_buf_to_DDR(
//         fm_t out_fm[2048][184][320], 
//         fm_t out_fm_buf[RPN_OUT_BUF_CH][RPN_OUT_BUF_ROWS][RPN_OUT_BUF_COLS], 
//         int ti, 
//         int tj,
//         int b
// );

// void rpn_conv_1x1 (
//         fm_t Y_buf[RPN_OUT_BUF_CH][RPN_OUT_BUF_ROWS][RPN_OUT_BUF_COLS], 
//         fm_t X_buf[RPN_IN_BUF_CH][RPN_IN_BUF_ROWS + 5][RPN_IN_BUF_COLS + 5],
//         wt_t W_buf[RPN_IN_BUF_CH][1][1],
//         int f,
//         int S
// );

// //TODO: template<const int S>
// void rpn_conv_3x3 (
//         fm_t Y_buf[RPN_OUT_BUF_CH][RPN_OUT_BUF_ROWS][RPN_OUT_BUF_COLS], 
//         fm_t X_buf[RPN_IN_BUF_CH][RPN_IN_BUF_ROWS + 5][RPN_IN_BUF_COLS + 5],
//         wt_t W_buf[RPN_IN_BUF_CH][3][3],
//         int f,
//         int S
// );

// template<const int N_OUT_CH, const int N_IN_CH>
// void rpn_load_weights_1x1 (
//         wt_t weight_buf_1x1[RPN_IN_BUF_CH][1][1],
//         wt_t weights[N_OUT_CH][N_IN_CH][1][1],
//         int f,
//         int b,
//         int d
// );

// template<const int  N_IN_FM_DEPTH, const int  N_IN_FM_HEIGHT, const int  N_IN_FM_WIDTH,
//          const int N_OUT_FM_DEPTH, const int N_OUT_FM_HEIGHT, const int N_OUT_FM_WIDTH,
//          const int STRIDE, const int LAST_LAYER_EN, const int row_mod, const int col_mod, const int o_depth_mod, const int i_depth_mod>
// void rpn_1x1_conv(
//         wt_t conv_weights[N_OUT_FM_DEPTH][N_IN_FM_DEPTH][1][1],
//         wt_t conv_bias[N_OUT_FM_DEPTH],
//         bool  relu
// );

// template<const int N_IN_FM_DEPTH,  const int N_IN_FM_HEIGHT,  const int N_IN_FM_WIDTH,
//          const int N_OUT_FM_DEPTH, const int N_OUT_FM_HEIGHT, const int N_OUT_FM_WIDTH, 
//          const int STRIDE, const int LAST_LAYER_EN, const int ROW_MOD>
// void rpn_3x3_conv(
//         wt_t conv_weights[N_OUT_FM_DEPTH][N_IN_FM_DEPTH][3][3],
//         wt_t conv_bias[N_OUT_FM_DEPTH],
//         bool relu
// );

// void rpn_top(
    
//     //Inputs to RPN
//     fm_t rpn_input0_fm[RPN_CONV_IN_CH][RPN_INPUT0_IN_FM_HEIGHT][RPN_INPUT0_IN_FM_WIDTH],
//     fm_t rpn_input1_fm[RPN_CONV_IN_CH][RPN_INPUT1_IN_FM_HEIGHT][RPN_INPUT1_IN_FM_WIDTH],
//     fm_t rpn_input2_fm[RPN_CONV_IN_CH][RPN_INPUT2_IN_FM_HEIGHT][RPN_INPUT2_IN_FM_WIDTH],


//     //Weights and Bias for convolutions
//     wt_t rpn_conv_weight[RPN_CONV_OUT_CH][RPN_CONV_IN_CH][3][3],
//     wt_t rpn_conv_bias[RPN_CONV_OUT_CH],
//     wt_t rpn_cls_weight[RPN_CLS_OUT_CH][RPN_CLS_IN_CH][1][1],
//     wt_t rpn_cls_bias[RPN_CLS_OUT_CH],
//     wt_t rpn_reg_weight[RPN_REG_OUT_CH][RPN_REG_IN_CH][1][1],
//     wt_t rpn_reg_bias[RPN_REG_OUT_CH],

//     fm_t rpn_output0_cls_fm[RPN_CLS_OUT_CH][RPN_INPUT0_IN_FM_HEIGHT][RPN_INPUT0_IN_FM_WIDTH],
//     fm_t rpn_output1_cls_fm[RPN_CLS_OUT_CH][RPN_INPUT1_IN_FM_HEIGHT][RPN_INPUT1_IN_FM_WIDTH],
//     fm_t rpn_output2_cls_fm[RPN_CLS_OUT_CH][RPN_INPUT2_IN_FM_HEIGHT][RPN_INPUT2_IN_FM_WIDTH],


//     fm_t rpn_output0_reg_fm[RPN_REG_OUT_CH][RPN_INPUT0_IN_FM_HEIGHT][RPN_INPUT0_IN_FM_WIDTH],
//     fm_t rpn_output1_reg_fm[RPN_REG_OUT_CH][RPN_INPUT1_IN_FM_HEIGHT][RPN_INPUT1_IN_FM_WIDTH],
//     fm_t rpn_output2_reg_fm[RPN_REG_OUT_CH][RPN_INPUT2_IN_FM_HEIGHT][RPN_INPUT2_IN_FM_WIDTH],



//     fm_t rpn_output0_fm[RPN_CONV_IN_CH][RPN_INPUT0_IN_FM_HEIGHT][RPN_INPUT0_IN_FM_WIDTH],
//     fm_t rpn_output1_fm[RPN_CONV_IN_CH][RPN_INPUT1_IN_FM_HEIGHT][RPN_INPUT1_IN_FM_WIDTH],
//     fm_t rpn_output2_fm[RPN_CONV_IN_CH][RPN_INPUT2_IN_FM_HEIGHT][RPN_INPUT2_IN_FM_WIDTH]

// );

// void rpn_top2(
//     int rpn_topk_index0[RPN_PRE_NMS_SIZE0],
//     int rpn_topk_index1[RPN_PRE_NMS_SIZE1],
//     int rpn_topk_index2[RPN_PRE_NMS_SIZE2],

//     fm_t rpn_anchor0_reg_fm [RPN_REG_OUT_CH*RPN_INPUT0_IN_FM_HEIGHT*RPN_INPUT0_IN_FM_WIDTH/4][4],
//     fm_t rpn_anchor1_reg_fm [RPN_REG_OUT_CH*RPN_INPUT0_IN_FM_HEIGHT*RPN_INPUT0_IN_FM_WIDTH/4][4],
//     fm_t rpn_anchor2_reg_fm [RPN_REG_OUT_CH*RPN_INPUT0_IN_FM_HEIGHT*RPN_INPUT0_IN_FM_WIDTH/4][4],

//     fm_t rpn_anchor0_cls_fm [RPN_CLS_OUT_CH*RPN_INPUT0_IN_FM_HEIGHT*RPN_INPUT0_IN_FM_WIDTH],
//     fm_t rpn_anchor1_cls_fm [RPN_CLS_OUT_CH*RPN_INPUT1_IN_FM_HEIGHT*RPN_INPUT1_IN_FM_WIDTH],
//     fm_t rpn_anchor2_cls_fm [RPN_CLS_OUT_CH*RPN_INPUT2_IN_FM_HEIGHT*RPN_INPUT2_IN_FM_WIDTH],

//     //Inputs to RPN
//     fm_t rpn_input3_fm[RPN_CONV_IN_CH][RPN_INPUT3_IN_FM_HEIGHT][RPN_INPUT3_IN_FM_WIDTH],
//     fm_t rpn_input4_fm[RPN_CONV_IN_CH][RPN_INPUT4_IN_FM_HEIGHT][RPN_INPUT4_IN_FM_WIDTH],


//     //Weights and Bias for convolutions
//     wt_t rpn_conv_weight[RPN_CONV_OUT_CH][RPN_CONV_IN_CH][3][3],
//     wt_t rpn_conv_bias[RPN_CONV_OUT_CH],
//     wt_t rpn_cls_weight[RPN_CLS_OUT_CH][RPN_CLS_IN_CH][1][1],
//     wt_t rpn_cls_bias[RPN_CLS_OUT_CH],
//     wt_t rpn_reg_weight[RPN_REG_OUT_CH][RPN_REG_IN_CH][1][1],
//     wt_t rpn_reg_bias[RPN_REG_OUT_CH],

//     fm_t rpn_output3_cls_fm[RPN_CLS_OUT_CH][RPN_INPUT3_IN_FM_HEIGHT][RPN_INPUT3_IN_FM_WIDTH],
//     fm_t rpn_output4_cls_fm[RPN_CLS_OUT_CH][RPN_INPUT4_IN_FM_HEIGHT][RPN_INPUT4_IN_FM_WIDTH],

//     fm_t rpn_output3_reg_fm[RPN_REG_OUT_CH][RPN_INPUT3_IN_FM_HEIGHT][RPN_INPUT3_IN_FM_WIDTH],
//     fm_t rpn_output4_reg_fm[RPN_REG_OUT_CH][RPN_INPUT4_IN_FM_HEIGHT][RPN_INPUT4_IN_FM_WIDTH],

//     fm_t rpn_output3_fm[RPN_CONV_IN_CH][RPN_INPUT3_IN_FM_HEIGHT][RPN_INPUT3_IN_FM_WIDTH],
//     fm_t rpn_output4_fm[RPN_CONV_IN_CH][RPN_INPUT4_IN_FM_HEIGHT][RPN_INPUT4_IN_FM_WIDTH],
    


//     wt_t anchor_box0[RPN_ANCHORS0_IN_FM][4],
//     wt_t anchor_box1[RPN_ANCHORS1_IN_FM][4],
//     wt_t anchor_box2[RPN_ANCHORS2_IN_FM][4],
//     wt_t anchor_box3[RPN_ANCHORS3_IN_FM][4],
//     wt_t anchor_box4[RPN_ANCHORS4_IN_FM][4],
    

//     fm_t bboxes[RPN_PRE_NMS_SIZE][4],
//     fm_t dets[1000][5]
    


// );


void test_top (
    // fm_t   resnet_layer0_input_fm[RESNET_LAYER0_CONV1_IN_CH][RESNET_LAYER0_IN_FM_HEIGHT][RESNET_LAYER0_IN_FM_WIDTH],
    // wt_t   resnet_layer0_conv1_weights[RESNET_LAYER0_CONV1_OUT_CH][RESNET_LAYER0_CONV1_IN_CH][7][7],
    // wt_t   resnet_layer0_bn1_params[3][RESNET_LAYER0_CONV1_OUT_CH],
    // fm_t   resnet_layer0_output_fm[RESNET_LAYER0_CONV1_OUT_CH][RESNET_LAYER0_OUT_FM_HEIGHT][RESNET_LAYER0_OUT_FM_WIDTH],

    // fm_t   resnet_layer1_input_fm[RESNET_LAYER1_0_CONV1_IN_CH][RESNET_LAYER1_0_FM_HEIGHT][RESNET_LAYER1_0_FM_WIDTH],
    // wt_t   resnet_layer1_0_conv1_weights[RESNET_LAYER1_0_CONV1_OUT_CH][RESNET_LAYER1_0_CONV1_IN_CH],
    // wt_t   resnet_layer1_0_bn1_params[4][RESNET_LAYER1_0_CONV1_OUT_CH],
    // wt_t   resnet_layer1_0_conv2_weights[RESNET_LAYER1_0_CONV2_OUT_CH][RESNET_LAYER1_0_CONV2_IN_CH][3][3],
    // wt_t   resnet_layer1_0_bn2_params[4][RESNET_LAYER1_0_CONV2_OUT_CH],
    // wt_t   resnet_layer1_0_conv3_weights[RESNET_LAYER1_0_CONV3_OUT_CH][RESNET_LAYER1_0_CONV3_IN_CH],
    // wt_t   resnet_layer1_0_bn3_params[4][RESNET_LAYER1_0_CONV3_OUT_CH],
    // wt_t   resnet_layer1_0_downsample_0_weights[RESNET_LAYER1_0_DS_OUT_CH][RESNET_LAYER1_0_DS_IN_CH],
    // wt_t   resnet_layer1_0_downsample_1_params[4][RESNET_LAYER1_0_DS_OUT_CH],
    // wt_t   resnet_layer1_1_conv1_weights[RESNET_LAYER1_CONV1_OUT_CH][RESNET_LAYER1_CONV1_IN_CH],
    // wt_t   resnet_layer1_1_bn1_params[4][RESNET_LAYER1_CONV1_OUT_CH],
    // wt_t   resnet_layer1_1_conv2_weights[RESNET_LAYER1_CONV2_OUT_CH][RESNET_LAYER1_CONV2_IN_CH][3][3],
    // wt_t   resnet_layer1_1_bn2_params[4][RESNET_LAYER1_CONV2_OUT_CH],
    // wt_t   resnet_layer1_1_conv3_weights[RESNET_LAYER1_CONV3_OUT_CH][RESNET_LAYER1_CONV3_IN_CH],
    // wt_t   resnet_layer1_1_bn3_params[4][RESNET_LAYER1_CONV3_OUT_CH],
    // wt_t   resnet_layer1_2_conv1_weights[RESNET_LAYER1_CONV1_OUT_CH][RESNET_LAYER1_CONV1_IN_CH],
    // wt_t   resnet_layer1_2_bn1_params[4][RESNET_LAYER1_CONV1_OUT_CH],
    // wt_t   resnet_layer1_2_conv2_weights[RESNET_LAYER1_CONV2_OUT_CH][RESNET_LAYER1_CONV2_IN_CH][3][3],
    // wt_t   resnet_layer1_2_bn2_params[4][RESNET_LAYER1_CONV2_OUT_CH],
    // wt_t   resnet_layer1_2_conv3_weights[RESNET_LAYER1_CONV3_OUT_CH][RESNET_LAYER1_CONV3_IN_CH],
    // wt_t   resnet_layer1_2_bn3_params[4][RESNET_LAYER1_CONV3_OUT_CH],
    // fm_t   resnet_layer1_output_fm[RESNET_LAYER1_0_DS_OUT_CH][RESNET_LAYER1_0_FM_HEIGHT][RESNET_LAYER1_0_FM_WIDTH],

    // fm_t   resnet_layer2_input_fm[RESNET_LAYER2_0_CONV1_IN_CH][RESNET_LAYER2_0_FM_HEIGHT][RESNET_LAYER2_0_FM_WIDTH],
    // wt_t   resnet_layer2_0_conv1_weights[RESNET_LAYER2_0_CONV1_OUT_CH][RESNET_LAYER2_0_CONV1_IN_CH],
	// wt_t   resnet_layer2_0_bn1_params[3][RESNET_LAYER2_0_CONV1_OUT_CH],
    // wt_t   resnet_layer2_0_conv2_weights[RESNET_LAYER2_0_CONV2_OUT_CH][RESNET_LAYER2_0_CONV2_IN_CH][3][3],
	// wt_t   resnet_layer2_0_bn2_params[3][RESNET_LAYER2_0_CONV2_OUT_CH],
    // wt_t   resnet_layer2_0_conv3_weights[RESNET_LAYER2_0_CONV3_OUT_CH][RESNET_LAYER2_0_CONV3_IN_CH],
	// wt_t   resnet_layer2_0_bn3_params[3][RESNET_LAYER2_0_CONV3_OUT_CH],
    // wt_t   resnet_layer2_0_downsample_0_weights[RESNET_LAYER2_0_DS_OUT_CH][RESNET_LAYER2_0_DS_IN_CH],
	// wt_t   resnet_layer2_0_downsample_1_params[3][RESNET_LAYER2_0_DS_OUT_CH],
    // wt_t   resnet_layer2_1_conv1_weights[RESNET_LAYER2_CONV1_OUT_CH][RESNET_LAYER2_CONV1_IN_CH],
	// wt_t   resnet_layer2_1_bn1_params[3][RESNET_LAYER2_CONV1_OUT_CH],
    // wt_t   resnet_layer2_1_conv2_weights[RESNET_LAYER2_CONV2_OUT_CH][RESNET_LAYER2_CONV2_IN_CH][3][3],
	// wt_t   resnet_layer2_1_bn2_params[3][RESNET_LAYER2_CONV2_OUT_CH],
    // wt_t   resnet_layer2_1_conv3_weights[RESNET_LAYER2_CONV3_OUT_CH][RESNET_LAYER2_CONV3_IN_CH],
	// wt_t   resnet_layer2_1_bn3_params[3][RESNET_LAYER2_CONV3_OUT_CH],
    // wt_t   resnet_layer2_2_conv1_weights[RESNET_LAYER2_CONV1_OUT_CH][RESNET_LAYER2_CONV1_IN_CH],
	// wt_t   resnet_layer2_2_bn1_params[3][RESNET_LAYER2_CONV1_OUT_CH],
    // wt_t   resnet_layer2_2_conv2_weights[RESNET_LAYER2_CONV2_OUT_CH][RESNET_LAYER2_CONV2_IN_CH][3][3],
	// wt_t   resnet_layer2_2_bn2_params[3][RESNET_LAYER2_CONV2_OUT_CH],
    // wt_t   resnet_layer2_2_conv3_weights[RESNET_LAYER2_CONV3_OUT_CH][RESNET_LAYER2_CONV3_IN_CH],
	// wt_t   resnet_layer2_2_bn3_params[3][RESNET_LAYER2_CONV3_OUT_CH],
    // wt_t   resnet_layer2_3_conv1_weights[RESNET_LAYER2_CONV1_OUT_CH][RESNET_LAYER2_CONV1_IN_CH],
	// wt_t   resnet_layer2_3_bn1_params[3][RESNET_LAYER2_CONV1_OUT_CH],
    // wt_t   resnet_layer2_3_conv2_weights[RESNET_LAYER2_CONV2_OUT_CH][RESNET_LAYER2_CONV2_IN_CH][3][3],
	// wt_t   resnet_layer2_3_bn2_params[3][RESNET_LAYER2_CONV2_OUT_CH],
    // wt_t   resnet_layer2_3_conv3_weights[RESNET_LAYER2_CONV3_OUT_CH][RESNET_LAYER2_CONV3_IN_CH],
	// wt_t   resnet_layer2_3_bn3_params[3][RESNET_LAYER2_CONV3_OUT_CH],
    // fm_t   resnet_layer2_output_fm[RESNET_LAYER2_0_DS_OUT_CH][RESNET_LAYER2_0_FM_HEIGHT][RESNET_LAYER2_0_FM_WIDTH],

    // fm_t   resnet_layer3_input_fm[RESNET_LAYER3_0_CONV1_IN_CH][RESNET_LAYER3_0_FM_HEIGHT][RESNET_LAYER3_0_FM_WIDTH],
    // wt_t   resnet_layer3_0_conv1_weights[RESNET_LAYER3_0_CONV1_OUT_CH][RESNET_LAYER3_0_CONV1_IN_CH],
	// wt_t   resnet_layer3_0_bn1_params[3][RESNET_LAYER3_0_CONV1_OUT_CH],
    // wt_t   resnet_layer3_0_conv2_weights[RESNET_LAYER3_0_CONV2_OUT_CH][RESNET_LAYER3_0_CONV2_IN_CH][3][3],
	// wt_t   resnet_layer3_0_bn2_params[3][RESNET_LAYER3_0_CONV2_OUT_CH],
    // wt_t   resnet_layer3_0_conv3_weights[RESNET_LAYER3_0_CONV3_OUT_CH][RESNET_LAYER3_0_CONV3_IN_CH],
	// wt_t   resnet_layer3_0_bn3_params[3][RESNET_LAYER3_0_CONV3_OUT_CH],
    // wt_t   resnet_layer3_0_downsample_0_weights[RESNET_LAYER3_0_DS_OUT_CH][RESNET_LAYER3_0_DS_IN_CH],
	// wt_t   resnet_layer3_0_downsample_1_params[3][RESNET_LAYER3_0_DS_OUT_CH],
    // wt_t   resnet_layer3_1_conv1_weights[RESNET_LAYER3_CONV1_OUT_CH][RESNET_LAYER3_CONV1_IN_CH],
	// wt_t   resnet_layer3_1_bn1_params[3][RESNET_LAYER3_CONV1_OUT_CH],
    // wt_t   resnet_layer3_1_conv2_weights[RESNET_LAYER3_CONV2_OUT_CH][RESNET_LAYER3_CONV2_IN_CH][3][3],
	// wt_t   resnet_layer3_1_bn2_params[3][RESNET_LAYER3_CONV2_OUT_CH],
    // wt_t   resnet_layer3_1_conv3_weights[RESNET_LAYER3_CONV3_OUT_CH][RESNET_LAYER3_CONV3_IN_CH],
	// wt_t   resnet_layer3_1_bn3_params[3][RESNET_LAYER3_CONV3_OUT_CH],
    // wt_t   resnet_layer3_2_conv1_weights[RESNET_LAYER3_CONV1_OUT_CH][RESNET_LAYER3_CONV1_IN_CH],
	// wt_t   resnet_layer3_2_bn1_params[3][RESNET_LAYER3_CONV1_OUT_CH],
    // wt_t   resnet_layer3_2_conv2_weights[RESNET_LAYER3_CONV2_OUT_CH][RESNET_LAYER3_CONV2_IN_CH][3][3],
	// wt_t   resnet_layer3_2_bn2_params[3][RESNET_LAYER3_CONV2_OUT_CH],
    // wt_t   resnet_layer3_2_conv3_weights[RESNET_LAYER3_CONV3_OUT_CH][RESNET_LAYER3_CONV3_IN_CH],
	// wt_t   resnet_layer3_2_bn3_params[3][RESNET_LAYER3_CONV3_OUT_CH],
    // wt_t   resnet_layer3_3_conv1_weights[RESNET_LAYER3_CONV1_OUT_CH][RESNET_LAYER3_CONV1_IN_CH],
	// wt_t   resnet_layer3_3_bn1_params[3][RESNET_LAYER3_CONV1_OUT_CH],
    // wt_t   resnet_layer3_3_conv2_weights[RESNET_LAYER3_CONV2_OUT_CH][RESNET_LAYER3_CONV2_IN_CH][3][3],
	// wt_t   resnet_layer3_3_bn2_params[3][RESNET_LAYER3_CONV2_OUT_CH],
    // wt_t   resnet_layer3_3_conv3_weights[RESNET_LAYER3_CONV3_OUT_CH][RESNET_LAYER3_CONV3_IN_CH],
	// wt_t   resnet_layer3_3_bn3_params[3][RESNET_LAYER3_CONV3_OUT_CH],
    // wt_t   resnet_layer3_4_conv1_weights[RESNET_LAYER3_CONV1_OUT_CH][RESNET_LAYER3_CONV1_IN_CH],
	// wt_t   resnet_layer3_4_bn1_params[3][RESNET_LAYER3_CONV1_OUT_CH],
    // wt_t   resnet_layer3_4_conv2_weights[RESNET_LAYER3_CONV2_OUT_CH][RESNET_LAYER3_CONV2_IN_CH][3][3],
	// wt_t   resnet_layer3_4_bn2_params[3][RESNET_LAYER3_CONV2_OUT_CH],
    // wt_t   resnet_layer3_4_conv3_weights[RESNET_LAYER3_CONV3_OUT_CH][RESNET_LAYER3_CONV3_IN_CH],
	// wt_t   resnet_layer3_4_bn3_params[3][RESNET_LAYER3_CONV3_OUT_CH],
    // wt_t   resnet_layer3_5_conv1_weights[RESNET_LAYER3_CONV1_OUT_CH][RESNET_LAYER3_CONV1_IN_CH],
	// wt_t   resnet_layer3_5_bn1_params[3][RESNET_LAYER3_CONV1_OUT_CH],
    // wt_t   resnet_layer3_5_conv2_weights[RESNET_LAYER3_CONV2_OUT_CH][RESNET_LAYER3_CONV2_IN_CH][3][3],
	// wt_t   resnet_layer3_5_bn2_params[3][RESNET_LAYER3_CONV2_OUT_CH],
    // wt_t   resnet_layer3_5_conv3_weights[RESNET_LAYER3_CONV3_OUT_CH][RESNET_LAYER3_CONV3_IN_CH],
	// wt_t   resnet_layer3_5_bn3_params[3][RESNET_LAYER3_CONV3_OUT_CH],
    // fm_t   resnet_layer3_output_fm[RESNET_LAYER3_0_DS_OUT_CH][RESNET_LAYER3_0_FM_HEIGHT][RESNET_LAYER3_0_FM_WIDTH],

    // fm_t   resnet_layer4_input_fm[RESNET_LAYER4_0_CONV1_IN_CH][RESNET_LAYER4_0_FM_HEIGHT][RESNET_LAYER4_0_FM_WIDTH],
    // wt_t   resnet_layer4_0_conv1_weights[RESNET_LAYER4_0_CONV1_OUT_CH][RESNET_LAYER4_0_CONV1_IN_CH],
	// wt_t   resnet_layer4_0_bn1_params[3][RESNET_LAYER4_0_CONV1_OUT_CH],
    // wt_t   resnet_layer4_0_conv2_weights[RESNET_LAYER4_0_CONV2_OUT_CH][RESNET_LAYER4_0_CONV2_IN_CH][3][3],
	// wt_t   resnet_layer4_0_bn2_params[3][RESNET_LAYER4_0_CONV2_OUT_CH],
    // wt_t   resnet_layer4_0_conv3_weights[RESNET_LAYER4_0_CONV3_OUT_CH][RESNET_LAYER4_0_CONV3_IN_CH],
	// wt_t   resnet_layer4_0_bn3_params[3][RESNET_LAYER4_0_CONV3_OUT_CH],
    // wt_t   resnet_layer4_0_downsample_0_weights[RESNET_LAYER4_0_DS_OUT_CH][RESNET_LAYER4_0_DS_IN_CH],
	// wt_t   resnet_layer4_0_downsample_1_params[3][RESNET_LAYER4_0_DS_OUT_CH],
    // wt_t   resnet_layer4_1_conv1_weights[RESNET_LAYER4_CONV1_OUT_CH][RESNET_LAYER4_CONV1_IN_CH],
	// wt_t   resnet_layer4_1_bn1_params[3][RESNET_LAYER4_CONV1_OUT_CH],
    // wt_t   resnet_layer4_1_conv2_weights[RESNET_LAYER4_CONV2_OUT_CH][RESNET_LAYER4_CONV2_IN_CH][3][3],
	// wt_t   resnet_layer4_1_bn2_params[3][RESNET_LAYER4_CONV2_OUT_CH],
    // wt_t   resnet_layer4_1_conv3_weights[RESNET_LAYER4_CONV3_OUT_CH][RESNET_LAYER4_CONV3_IN_CH],
	// wt_t   resnet_layer4_1_bn3_params[3][RESNET_LAYER4_CONV3_OUT_CH],
    // wt_t   resnet_layer4_2_conv1_weights[RESNET_LAYER4_CONV1_OUT_CH][RESNET_LAYER4_CONV1_IN_CH],
	// wt_t   resnet_layer4_2_bn1_params[3][RESNET_LAYER4_CONV1_OUT_CH],
    // wt_t   resnet_layer4_2_conv2_weights[RESNET_LAYER4_CONV2_OUT_CH][RESNET_LAYER4_CONV2_IN_CH][3][3],
	// wt_t   resnet_layer4_2_bn2_params[3][RESNET_LAYER4_CONV2_OUT_CH],
    // wt_t   resnet_layer4_2_conv3_weights[RESNET_LAYER4_CONV3_OUT_CH][RESNET_LAYER4_CONV3_IN_CH],
	// wt_t   resnet_layer4_2_bn3_params[3][RESNET_LAYER4_CONV3_OUT_CH],
    // fm_t   resnet_layer4_output_fm[RESNET_LAYER4_0_DS_OUT_CH][RESNET_LAYER4_0_FM_HEIGHT][RESNET_LAYER4_0_FM_WIDTH],

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

    // int rpn_topk_index0[RPN_PRE_NMS_SIZE0],
    // int rpn_topk_index1[RPN_PRE_NMS_SIZE1],
    // int rpn_topk_index2[RPN_PRE_NMS_SIZE2],

    // fm_t rpn_anchor0_reg_fm [RPN_REG_OUT_CH*RPN_INPUT0_IN_FM_HEIGHT*RPN_INPUT0_IN_FM_WIDTH/4][4],
    // fm_t rpn_anchor1_reg_fm [RPN_REG_OUT_CH*RPN_INPUT0_IN_FM_HEIGHT*RPN_INPUT0_IN_FM_WIDTH/4][4],
    // fm_t rpn_anchor2_reg_fm [RPN_REG_OUT_CH*RPN_INPUT0_IN_FM_HEIGHT*RPN_INPUT0_IN_FM_WIDTH/4][4],

    // fm_t rpn_anchor0_cls_fm[RPN_CLS_OUT_CH*RPN_INPUT0_IN_FM_HEIGHT*RPN_INPUT0_IN_FM_WIDTH],
    // fm_t rpn_anchor1_cls_fm[RPN_CLS_OUT_CH*RPN_INPUT1_IN_FM_HEIGHT*RPN_INPUT1_IN_FM_WIDTH],
    // fm_t rpn_anchor2_cls_fm[RPN_CLS_OUT_CH*RPN_INPUT2_IN_FM_HEIGHT*RPN_INPUT2_IN_FM_WIDTH],

    // //Inputs to RPN
    // fm_t rpn_input0_fm[RPN_CONV_IN_CH][RPN_INPUT0_IN_FM_HEIGHT][RPN_INPUT0_IN_FM_WIDTH],
    // fm_t rpn_input1_fm[RPN_CONV_IN_CH][RPN_INPUT1_IN_FM_HEIGHT][RPN_INPUT1_IN_FM_WIDTH],
    // fm_t rpn_input2_fm[RPN_CONV_IN_CH][RPN_INPUT2_IN_FM_HEIGHT][RPN_INPUT2_IN_FM_WIDTH],
    // fm_t rpn_input3_fm[RPN_CONV_IN_CH][RPN_INPUT3_IN_FM_HEIGHT][RPN_INPUT3_IN_FM_WIDTH],
    // fm_t rpn_input4_fm[RPN_CONV_IN_CH][RPN_INPUT4_IN_FM_HEIGHT][RPN_INPUT4_IN_FM_WIDTH],


    // //Weights and Bias for convolutions
    // wt_t rpn_conv_weight[RPN_CONV_OUT_CH][RPN_CONV_IN_CH][3][3],
    // wt_t rpn_conv_bias[RPN_CONV_OUT_CH],
    // wt_t rpn_cls_weight[RPN_CLS_OUT_CH][RPN_CLS_IN_CH][1][1],
    // wt_t rpn_cls_bias[RPN_CLS_OUT_CH],
    // wt_t rpn_reg_weight[RPN_REG_OUT_CH][RPN_REG_IN_CH][1][1],
    // wt_t rpn_reg_bias[RPN_REG_OUT_CH],

    // fm_t rpn_output0_cls_fm[RPN_CLS_OUT_CH][RPN_INPUT0_IN_FM_HEIGHT][RPN_INPUT0_IN_FM_WIDTH],
    // fm_t rpn_output1_cls_fm[RPN_CLS_OUT_CH][RPN_INPUT1_IN_FM_HEIGHT][RPN_INPUT1_IN_FM_WIDTH],
    // fm_t rpn_output2_cls_fm[RPN_CLS_OUT_CH][RPN_INPUT2_IN_FM_HEIGHT][RPN_INPUT2_IN_FM_WIDTH],
    // fm_t rpn_output3_cls_fm[RPN_CLS_OUT_CH][RPN_INPUT3_IN_FM_HEIGHT][RPN_INPUT3_IN_FM_WIDTH],
    // fm_t rpn_output4_cls_fm[RPN_CLS_OUT_CH][RPN_INPUT4_IN_FM_HEIGHT][RPN_INPUT4_IN_FM_WIDTH],


    // fm_t rpn_output0_reg_fm[RPN_REG_OUT_CH][RPN_INPUT0_IN_FM_HEIGHT][RPN_INPUT0_IN_FM_WIDTH],
    // fm_t rpn_output1_reg_fm[RPN_REG_OUT_CH][RPN_INPUT1_IN_FM_HEIGHT][RPN_INPUT1_IN_FM_WIDTH],
    // fm_t rpn_output2_reg_fm[RPN_REG_OUT_CH][RPN_INPUT2_IN_FM_HEIGHT][RPN_INPUT2_IN_FM_WIDTH],
    // fm_t rpn_output3_reg_fm[RPN_REG_OUT_CH][RPN_INPUT3_IN_FM_HEIGHT][RPN_INPUT3_IN_FM_WIDTH],
    // fm_t rpn_output4_reg_fm[RPN_REG_OUT_CH][RPN_INPUT4_IN_FM_HEIGHT][RPN_INPUT4_IN_FM_WIDTH],



    // fm_t rpn_output0_fm[RPN_CONV_IN_CH][RPN_INPUT0_IN_FM_HEIGHT][RPN_INPUT0_IN_FM_WIDTH],
    // fm_t rpn_output1_fm[RPN_CONV_IN_CH][RPN_INPUT1_IN_FM_HEIGHT][RPN_INPUT1_IN_FM_WIDTH],
    // fm_t rpn_output2_fm[RPN_CONV_IN_CH][RPN_INPUT2_IN_FM_HEIGHT][RPN_INPUT2_IN_FM_WIDTH],
    // fm_t rpn_output3_fm[RPN_CONV_IN_CH][RPN_INPUT3_IN_FM_HEIGHT][RPN_INPUT3_IN_FM_WIDTH],
    // fm_t rpn_output4_fm[RPN_CONV_IN_CH][RPN_INPUT4_IN_FM_HEIGHT][RPN_INPUT4_IN_FM_WIDTH],
    


    // wt_t anchor_box0[RPN_ANCHORS0_IN_FM][4],
    // wt_t anchor_box1[RPN_ANCHORS1_IN_FM][4],
    // wt_t anchor_box2[RPN_ANCHORS2_IN_FM][4],
    // wt_t anchor_box3[RPN_ANCHORS3_IN_FM][4],
    // wt_t anchor_box4[RPN_ANCHORS4_IN_FM][4],
    

    // fm_t bboxes[RPN_PRE_NMS_SIZE][4],
    // fm_t dets[1000][5]
);

#endif
