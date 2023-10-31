#include "qdtrack_rpn2.h"

void test_top (
    int rpn_topk_index0[RPN_PRE_NMS_SIZE0],
    int rpn_topk_index1[RPN_PRE_NMS_SIZE1],
    int rpn_topk_index2[RPN_PRE_NMS_SIZE2],

    fm_t rpn_anchor0_reg_fm[RPN_REG_OUT_CH*RPN_INPUT0_IN_FM_HEIGHT*RPN_INPUT0_IN_FM_WIDTH/4][4],
    fm_t rpn_anchor1_reg_fm[RPN_REG_OUT_CH*RPN_INPUT0_IN_FM_HEIGHT*RPN_INPUT0_IN_FM_WIDTH/4][4],
    fm_t rpn_anchor2_reg_fm[RPN_REG_OUT_CH*RPN_INPUT0_IN_FM_HEIGHT*RPN_INPUT0_IN_FM_WIDTH/4][4],

    fm_t rpn_anchor0_cls_fm[RPN_CLS_OUT_CH*RPN_INPUT0_IN_FM_HEIGHT*RPN_INPUT0_IN_FM_WIDTH],
    fm_t rpn_anchor1_cls_fm[RPN_CLS_OUT_CH*RPN_INPUT1_IN_FM_HEIGHT*RPN_INPUT1_IN_FM_WIDTH],
    fm_t rpn_anchor2_cls_fm[RPN_CLS_OUT_CH*RPN_INPUT2_IN_FM_HEIGHT*RPN_INPUT2_IN_FM_WIDTH],

    //Inputs to RPN
    // fm_t rpn_input0_fm[RPN_CONV_IN_CH][RPN_INPUT0_IN_FM_HEIGHT][RPN_INPUT0_IN_FM_WIDTH],
    // fm_t rpn_input1_fm[RPN_CONV_IN_CH][RPN_INPUT1_IN_FM_HEIGHT][RPN_INPUT1_IN_FM_WIDTH],
    // fm_t rpn_input2_fm[RPN_CONV_IN_CH][RPN_INPUT2_IN_FM_HEIGHT][RPN_INPUT2_IN_FM_WIDTH],
    fm_t rpn_input3_fm[RPN_CONV_IN_CH][RPN_INPUT3_IN_FM_HEIGHT][RPN_INPUT3_IN_FM_WIDTH],
    fm_t rpn_input4_fm[RPN_CONV_IN_CH][RPN_INPUT4_IN_FM_HEIGHT][RPN_INPUT4_IN_FM_WIDTH],

    //Weights and Bias for convolutions
    wt_t rpn_conv_weight[RPN_CONV_OUT_CH][RPN_CONV_IN_CH][3][3],
    wt_t rpn_conv_bias[RPN_CONV_OUT_CH],
    wt_t rpn_cls_weight[RPN_CLS_OUT_CH][RPN_CLS_IN_CH][1][1],
    wt_t rpn_cls_bias[RPN_CLS_OUT_CH],
    wt_t rpn_reg_weight[RPN_REG_OUT_CH][RPN_REG_IN_CH][1][1],
    wt_t rpn_reg_bias[RPN_REG_OUT_CH],

    // fm_t rpn_output0_cls_fm[RPN_CLS_OUT_CH][RPN_INPUT0_IN_FM_HEIGHT][RPN_INPUT0_IN_FM_WIDTH],
    // fm_t rpn_output1_cls_fm[RPN_CLS_OUT_CH][RPN_INPUT1_IN_FM_HEIGHT][RPN_INPUT1_IN_FM_WIDTH],
    // fm_t rpn_output2_cls_fm[RPN_CLS_OUT_CH][RPN_INPUT2_IN_FM_HEIGHT][RPN_INPUT2_IN_FM_WIDTH],
    fm_t rpn_output3_cls_fm[RPN_CLS_OUT_CH][RPN_INPUT3_IN_FM_HEIGHT][RPN_INPUT3_IN_FM_WIDTH],
    fm_t rpn_output4_cls_fm[RPN_CLS_OUT_CH][RPN_INPUT4_IN_FM_HEIGHT][RPN_INPUT4_IN_FM_WIDTH],


    // fm_t rpn_output0_reg_fm[RPN_REG_OUT_CH][RPN_INPUT0_IN_FM_HEIGHT][RPN_INPUT0_IN_FM_WIDTH],
    // fm_t rpn_output1_reg_fm[RPN_REG_OUT_CH][RPN_INPUT1_IN_FM_HEIGHT][RPN_INPUT1_IN_FM_WIDTH],
    // fm_t rpn_output2_reg_fm[RPN_REG_OUT_CH][RPN_INPUT2_IN_FM_HEIGHT][RPN_INPUT2_IN_FM_WIDTH],
    fm_t rpn_output3_reg_fm[RPN_REG_OUT_CH][RPN_INPUT3_IN_FM_HEIGHT][RPN_INPUT3_IN_FM_WIDTH],
    fm_t rpn_output4_reg_fm[RPN_REG_OUT_CH][RPN_INPUT4_IN_FM_HEIGHT][RPN_INPUT4_IN_FM_WIDTH],



    // fm_t rpn_output0_fm[RPN_CONV_IN_CH][RPN_INPUT0_IN_FM_HEIGHT][RPN_INPUT0_IN_FM_WIDTH],
    // fm_t rpn_output1_fm[RPN_CONV_IN_CH][RPN_INPUT1_IN_FM_HEIGHT][RPN_INPUT1_IN_FM_WIDTH],
    // fm_t rpn_output2_fm[RPN_CONV_IN_CH][RPN_INPUT2_IN_FM_HEIGHT][RPN_INPUT2_IN_FM_WIDTH],
    fm_t rpn_output3_fm[RPN_CONV_IN_CH][RPN_INPUT3_IN_FM_HEIGHT][RPN_INPUT3_IN_FM_WIDTH],
    fm_t rpn_output4_fm[RPN_CONV_IN_CH][RPN_INPUT4_IN_FM_HEIGHT][RPN_INPUT4_IN_FM_WIDTH],
    


    wt_t anchor_box0[RPN_ANCHORS0_IN_FM][4],
    wt_t anchor_box1[RPN_ANCHORS1_IN_FM][4],
    wt_t anchor_box2[RPN_ANCHORS2_IN_FM][4],
    wt_t anchor_box3[RPN_ANCHORS3_IN_FM][4],
    wt_t anchor_box4[RPN_ANCHORS4_IN_FM][4],
    

    fm_t bboxes[RPN_PRE_NMS_SIZE][4],
    fm_t dets[1000][5]
)
{
//   rpn_top (
//     //Inputs to RPN
//     rpn_input0_fm,
//     rpn_input1_fm,
//     rpn_input2_fm,

//     //Weights and Bias for convolutions
//     rpn_conv_weight,
//     rpn_conv_bias,
//     rpn_cls_weight,
//     rpn_cls_bias,
//     rpn_reg_weight,
//     rpn_reg_bias,

//     rpn_output0_cls_fm,
//     rpn_output1_cls_fm,
//     rpn_output2_cls_fm,

//     rpn_output0_reg_fm,
//     rpn_output1_reg_fm,
//     rpn_output2_reg_fm,

//     rpn_output0_fm,
//     rpn_output1_fm,
//     rpn_output2_fm
// );

 rpn_top2 (
    rpn_topk_index0,
    rpn_topk_index1,
    rpn_topk_index2,

    rpn_anchor0_reg_fm,
    rpn_anchor1_reg_fm,
    rpn_anchor2_reg_fm,

    rpn_anchor0_cls_fm,
    rpn_anchor1_cls_fm,
    rpn_anchor2_cls_fm,

    //Inputs to RPN
    rpn_input3_fm,
    rpn_input4_fm,

    //Weights and Bias for convolutions
    rpn_conv_weight,
    rpn_conv_bias,
    rpn_cls_weight,
    rpn_cls_bias,
    rpn_reg_weight,
    rpn_reg_bias,

    rpn_output3_cls_fm,
    rpn_output4_cls_fm,

    rpn_output3_reg_fm,
    rpn_output4_reg_fm,

    rpn_output3_fm,
    rpn_output4_fm,
    
    anchor_box0,
    anchor_box1,
    anchor_box2,
    anchor_box3,
    anchor_box4,

    bboxes,
    dets
);

}
