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

//--------------------------------------------------------------------------
// RPN
//--------------------------------------------------------------------------
#define RPN_IN_BUF_CH        64
#define RPN_IN_BUF_ROWS      46
#define RPN_IN_BUF_COLS      40

#define RPN_OUT_BUF_CH       64
#define RPN_OUT_BUF_ROWS     46
#define RPN_OUT_BUF_COLS     40

#define RPN_LAST_LAYER_ENABLE   1
#define RPN_LAST_LAYER_DISABLE  0

//--------------------------------------------------------------------------
// RPN Convolution Stride and Padding Declarations
//--------------------------------------------------------------------------

#define RPN_CONV_STRIDE 1
#define RPN_CONV_PADDING 1
#define RPN_CLS_STRIDE 1
#define RPN_CLS_PADDING 0
#define RPN_REG_STRIDE 1
#define RPN_REG_PADDING 0


//--------------------------------------------------------------------------
// RPN Convolution Size Declarations
//--------------------------------------------------------------------------

#define RPN_INPUT0_IN_FM_HEIGHT 184
#define RPN_INPUT0_IN_FM_WIDTH 320
#define RPN_INPUT1_IN_FM_HEIGHT 92
#define RPN_INPUT1_IN_FM_WIDTH 160
#define RPN_INPUT2_IN_FM_HEIGHT 46
#define RPN_INPUT2_IN_FM_WIDTH 80
// #define RPN_INPUT3_IN_FM_HEIGHT 23
// #define RPN_INPUT3_IN_FM_WIDTH 40
// #define RPN_INPUT4_IN_FM_HEIGHT 12
// #define RPN_INPUT4_IN_FM_WIDTH 20


#define RPN_CONV_IN_CH 256
#define RPN_CONV_OUT_CH 256
#define RPN_CLS_IN_CH 256
#define RPN_CLS_OUT_CH 3
#define RPN_REG_IN_CH 256
#define RPN_REG_OUT_CH 12



#define RPN_ANCHORS0_IN_FM 176640
#define RPN_ANCHORS1_IN_FM 44160
#define RPN_ANCHORS2_IN_FM 11040
// #define RPN_ANCHORS3_IN_FM 2760
// #define RPN_ANCHORS4_IN_FM 720

#define RPN_PRE_NMS_SIZE0 1000//((RPN_ANCHORS0_IN_FM) > (1000) ? (RPN_ANCHORS0_IN_FM) : (1000))
#define RPN_PRE_NMS_SIZE1 1000//((RPN_ANCHORS1_IN_FM) > (1000) ? (RPN_ANCHORS1_IN_FM) : (1000))
#define RPN_PRE_NMS_SIZE2 1000//((RPN_ANCHORS2_IN_FM) > (1000) ? (RPN_ANCHORS2_IN_FM) : (1000))
// #define RPN_PRE_NMS_SIZE3 1000//((RPN_ANCHORS3_IN_FM) > (1000) ? (RPN_ANCHORS3_IN_FM) : (1000))
// #define RPN_PRE_NMS_SIZE4 720//((RPN_ANCHORS4_IN_FM) > (1000) ? (RPN_ANCHORS4_IN_FM) : (1000))

#define RPN_PRE_NMS_SIZE 4720
#define IOU_THRESHOLD 0.7

//--------------------------------------------------------------------------
// RPN
//--------------------------------------------------------------------------
template<const int N_OUT_CH>
void rpn_load_bias_params (
        fm_t param_buf[RPN_OUT_BUF_CH], 
        wt_t params[N_OUT_CH], 
        int b
);


// void rpn_conv_bias_add(
//         fm_t feature_map[RPN_OUT_BUF_CH][RPN_OUT_BUF_ROWS][RPN_OUT_BUF_COLS], 
//         wt_t bias_params[RPN_OUT_BUF_CH],
//         bool relu
// );



template<const int N_IN_FM_DEPTH, const int N_IN_FM_HEIGHT, const int N_IN_FM_WIDTH,
         const int LAST_LAYER_EN>
void rpn_load_input_fm_tile (
        fm_t in_fm_buf[RPN_IN_BUF_CH][RPN_IN_BUF_ROWS + 5][RPN_IN_BUF_COLS + 5], 
        fm_t in_fm[2048][184][320], 
        int   ti, 
        int   tj, 
        int   P,
        int   d
);

// void rpn_load_residual_fm_tile (
//         fm_t in_fm_buf[RPN_IN_BUF_CH][RPN_IN_BUF_ROWS][RPN_IN_BUF_COLS], 
//         fm_t in_fm[2048][184][320], 
//         int   ti, 
//         int   tj,
//         int   b 
// );


template<const int N_OUT_CH, const int N_IN_CH>
void rpn_load_weights_3x3 (
        wt_t weight_buf_3x3[RPN_IN_BUF_CH][3][3],
        wt_t weights[N_OUT_CH][N_IN_CH][3][3],
        int f,
        int b,
        int d
);


// void rpn_save_partial_out_buf (
//         fm_t partial_out_fm_buf[RPN_OUT_BUF_CH][RPN_OUT_BUF_ROWS][RPN_OUT_BUF_COLS], 
//         fm_t out_fm_buf[RPN_OUT_BUF_CH][RPN_OUT_BUF_ROWS][RPN_OUT_BUF_COLS],
//         int d
// );

template<const int S>
void rpn_store_out_buf_to_DDR(
        fm_t out_fm[2048][184][320], 
        fm_t out_fm_buf[RPN_OUT_BUF_CH][RPN_OUT_BUF_ROWS][RPN_OUT_BUF_COLS], 
        int ti, 
        int tj,
        int b
);

void rpn_conv_1x1 (
        fm_t Y_buf[RPN_OUT_BUF_CH][RPN_OUT_BUF_ROWS][RPN_OUT_BUF_COLS], 
        fm_t X_buf[RPN_IN_BUF_CH][RPN_IN_BUF_ROWS + 5][RPN_IN_BUF_COLS + 5],
        wt_t W_buf[RPN_IN_BUF_CH][1][1],
        int f,
        int S
);

//TODO: template<const int S>
void rpn_conv_3x3 (
        fm_t Y_buf[RPN_OUT_BUF_CH][RPN_OUT_BUF_ROWS][RPN_OUT_BUF_COLS], 
        fm_t X_buf[RPN_IN_BUF_CH][RPN_IN_BUF_ROWS + 5][RPN_IN_BUF_COLS + 5],
        wt_t W_buf[RPN_IN_BUF_CH][3][3],
        int f,
        int S
);

template<const int N_OUT_CH, const int N_IN_CH>
void rpn_load_weights_1x1 (
        wt_t weight_buf_1x1[RPN_IN_BUF_CH][1][1],
        wt_t weights[N_OUT_CH][N_IN_CH][1][1],
        int f,
        int b,
        int d
);

template<const int  N_IN_FM_DEPTH, const int  N_IN_FM_HEIGHT, const int  N_IN_FM_WIDTH,
         const int N_OUT_FM_DEPTH, const int N_OUT_FM_HEIGHT, const int N_OUT_FM_WIDTH,
         const int STRIDE, const int LAST_LAYER_EN, const int row_mod, const int col_mod, const int o_depth_mod, const int i_depth_mod>
void rpn_1x1_conv(
        wt_t conv_weights[N_OUT_FM_DEPTH][N_IN_FM_DEPTH][1][1],
        wt_t conv_bias[N_OUT_FM_DEPTH],
        bool  relu
);

template<const int N_IN_FM_DEPTH,  const int N_IN_FM_HEIGHT,  const int N_IN_FM_WIDTH,
         const int N_OUT_FM_DEPTH, const int N_OUT_FM_HEIGHT, const int N_OUT_FM_WIDTH, 
         const int STRIDE, const int LAST_LAYER_EN, const int ROW_MOD>
void rpn_3x3_conv(
        wt_t conv_weights[N_OUT_FM_DEPTH][N_IN_FM_DEPTH][3][3],
        wt_t conv_bias[N_OUT_FM_DEPTH],
        bool relu
);

void rpn_top(
    
    //Inputs to RPN
    fm_t rpn_input0_fm[RPN_CONV_IN_CH][RPN_INPUT0_IN_FM_HEIGHT][RPN_INPUT0_IN_FM_WIDTH],
    fm_t rpn_input1_fm[RPN_CONV_IN_CH][RPN_INPUT1_IN_FM_HEIGHT][RPN_INPUT1_IN_FM_WIDTH],
    fm_t rpn_input2_fm[RPN_CONV_IN_CH][RPN_INPUT2_IN_FM_HEIGHT][RPN_INPUT2_IN_FM_WIDTH],


    //Weights and Bias for convolutions
    wt_t rpn_conv_weight[RPN_CONV_OUT_CH][RPN_CONV_IN_CH][3][3],
    wt_t rpn_conv_bias[RPN_CONV_OUT_CH],
    wt_t rpn_cls_weight[RPN_CLS_OUT_CH][RPN_CLS_IN_CH][1][1],
    wt_t rpn_cls_bias[RPN_CLS_OUT_CH],
    wt_t rpn_reg_weight[RPN_REG_OUT_CH][RPN_REG_IN_CH][1][1],
    wt_t rpn_reg_bias[RPN_REG_OUT_CH],

    fm_t rpn_output0_cls_fm[RPN_CLS_OUT_CH][RPN_INPUT0_IN_FM_HEIGHT][RPN_INPUT0_IN_FM_WIDTH],
    fm_t rpn_output1_cls_fm[RPN_CLS_OUT_CH][RPN_INPUT1_IN_FM_HEIGHT][RPN_INPUT1_IN_FM_WIDTH],
    fm_t rpn_output2_cls_fm[RPN_CLS_OUT_CH][RPN_INPUT2_IN_FM_HEIGHT][RPN_INPUT2_IN_FM_WIDTH],


    fm_t rpn_output0_reg_fm[RPN_REG_OUT_CH][RPN_INPUT0_IN_FM_HEIGHT][RPN_INPUT0_IN_FM_WIDTH],
    fm_t rpn_output1_reg_fm[RPN_REG_OUT_CH][RPN_INPUT1_IN_FM_HEIGHT][RPN_INPUT1_IN_FM_WIDTH],
    fm_t rpn_output2_reg_fm[RPN_REG_OUT_CH][RPN_INPUT2_IN_FM_HEIGHT][RPN_INPUT2_IN_FM_WIDTH],



    fm_t rpn_output0_fm[RPN_CONV_IN_CH][RPN_INPUT0_IN_FM_HEIGHT][RPN_INPUT0_IN_FM_WIDTH],
    fm_t rpn_output1_fm[RPN_CONV_IN_CH][RPN_INPUT1_IN_FM_HEIGHT][RPN_INPUT1_IN_FM_WIDTH],
    fm_t rpn_output2_fm[RPN_CONV_IN_CH][RPN_INPUT2_IN_FM_HEIGHT][RPN_INPUT2_IN_FM_WIDTH]

);

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
//     int rpn_topk_index0[RPN_PRE_NMS_SIZE0],
//     int rpn_topk_index1[RPN_PRE_NMS_SIZE1],
//     int rpn_topk_index2[RPN_PRE_NMS_SIZE2],

//     fm_t rpn_anchor0_reg_fm [RPN_REG_OUT_CH*RPN_INPUT0_IN_FM_HEIGHT*RPN_INPUT0_IN_FM_WIDTH/4][4],
//     fm_t rpn_anchor1_reg_fm [RPN_REG_OUT_CH*RPN_INPUT0_IN_FM_HEIGHT*RPN_INPUT0_IN_FM_WIDTH/4][4],
//     fm_t rpn_anchor2_reg_fm [RPN_REG_OUT_CH*RPN_INPUT0_IN_FM_HEIGHT*RPN_INPUT0_IN_FM_WIDTH/4][4],

//     fm_t rpn_anchor0_cls_fm[RPN_CLS_OUT_CH*RPN_INPUT0_IN_FM_HEIGHT*RPN_INPUT0_IN_FM_WIDTH],
//     fm_t rpn_anchor1_cls_fm[RPN_CLS_OUT_CH*RPN_INPUT1_IN_FM_HEIGHT*RPN_INPUT1_IN_FM_WIDTH],
//     fm_t rpn_anchor2_cls_fm[RPN_CLS_OUT_CH*RPN_INPUT2_IN_FM_HEIGHT*RPN_INPUT2_IN_FM_WIDTH],

    //Inputs to RPN
    fm_t rpn_input0_fm[RPN_CONV_IN_CH][RPN_INPUT0_IN_FM_HEIGHT][RPN_INPUT0_IN_FM_WIDTH],
    fm_t rpn_input1_fm[RPN_CONV_IN_CH][RPN_INPUT1_IN_FM_HEIGHT][RPN_INPUT1_IN_FM_WIDTH],
    fm_t rpn_input2_fm[RPN_CONV_IN_CH][RPN_INPUT2_IN_FM_HEIGHT][RPN_INPUT2_IN_FM_WIDTH],
//     fm_t rpn_input3_fm[RPN_CONV_IN_CH][RPN_INPUT3_IN_FM_HEIGHT][RPN_INPUT3_IN_FM_WIDTH],
//     fm_t rpn_input4_fm[RPN_CONV_IN_CH][RPN_INPUT4_IN_FM_HEIGHT][RPN_INPUT4_IN_FM_WIDTH],


    //Weights and Bias for convolutions
    wt_t rpn_conv_weight[RPN_CONV_OUT_CH][RPN_CONV_IN_CH][3][3],
    wt_t rpn_conv_bias[RPN_CONV_OUT_CH],
    wt_t rpn_cls_weight[RPN_CLS_OUT_CH][RPN_CLS_IN_CH][1][1],
    wt_t rpn_cls_bias[RPN_CLS_OUT_CH],
    wt_t rpn_reg_weight[RPN_REG_OUT_CH][RPN_REG_IN_CH][1][1],
    wt_t rpn_reg_bias[RPN_REG_OUT_CH],

    fm_t rpn_output0_cls_fm[RPN_CLS_OUT_CH][RPN_INPUT0_IN_FM_HEIGHT][RPN_INPUT0_IN_FM_WIDTH],
    fm_t rpn_output1_cls_fm[RPN_CLS_OUT_CH][RPN_INPUT1_IN_FM_HEIGHT][RPN_INPUT1_IN_FM_WIDTH],
    fm_t rpn_output2_cls_fm[RPN_CLS_OUT_CH][RPN_INPUT2_IN_FM_HEIGHT][RPN_INPUT2_IN_FM_WIDTH],
//     fm_t rpn_output3_cls_fm[RPN_CLS_OUT_CH][RPN_INPUT3_IN_FM_HEIGHT][RPN_INPUT3_IN_FM_WIDTH],
//     fm_t rpn_output4_cls_fm[RPN_CLS_OUT_CH][RPN_INPUT4_IN_FM_HEIGHT][RPN_INPUT4_IN_FM_WIDTH],


    fm_t rpn_output0_reg_fm[RPN_REG_OUT_CH][RPN_INPUT0_IN_FM_HEIGHT][RPN_INPUT0_IN_FM_WIDTH],
    fm_t rpn_output1_reg_fm[RPN_REG_OUT_CH][RPN_INPUT1_IN_FM_HEIGHT][RPN_INPUT1_IN_FM_WIDTH],
    fm_t rpn_output2_reg_fm[RPN_REG_OUT_CH][RPN_INPUT2_IN_FM_HEIGHT][RPN_INPUT2_IN_FM_WIDTH],
//     fm_t rpn_output3_reg_fm[RPN_REG_OUT_CH][RPN_INPUT3_IN_FM_HEIGHT][RPN_INPUT3_IN_FM_WIDTH],
//     fm_t rpn_output4_reg_fm[RPN_REG_OUT_CH][RPN_INPUT4_IN_FM_HEIGHT][RPN_INPUT4_IN_FM_WIDTH],



    fm_t rpn_output0_fm[RPN_CONV_IN_CH][RPN_INPUT0_IN_FM_HEIGHT][RPN_INPUT0_IN_FM_WIDTH],
    fm_t rpn_output1_fm[RPN_CONV_IN_CH][RPN_INPUT1_IN_FM_HEIGHT][RPN_INPUT1_IN_FM_WIDTH],
    fm_t rpn_output2_fm[RPN_CONV_IN_CH][RPN_INPUT2_IN_FM_HEIGHT][RPN_INPUT2_IN_FM_WIDTH]
//     fm_t rpn_output3_fm[RPN_CONV_IN_CH][RPN_INPUT3_IN_FM_HEIGHT][RPN_INPUT3_IN_FM_WIDTH],
//     fm_t rpn_output4_fm[RPN_CONV_IN_CH][RPN_INPUT4_IN_FM_HEIGHT][RPN_INPUT4_IN_FM_WIDTH],
    


//     wt_t anchor_box0[RPN_ANCHORS0_IN_FM][4],
//     wt_t anchor_box1[RPN_ANCHORS1_IN_FM][4],
//     wt_t anchor_box2[RPN_ANCHORS2_IN_FM][4],
//     wt_t anchor_box3[RPN_ANCHORS3_IN_FM][4],
//     wt_t anchor_box4[RPN_ANCHORS4_IN_FM][4],
    

//     fm_t bboxes[RPN_PRE_NMS_SIZE][4],
//     fm_t dets[1000][5]
);

#endif
