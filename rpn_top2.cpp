#include "hls_stream.h"
#include "qdtrack.h"
#include "rpn_util.h"
#include "math.h"


// fm_t rpn_anchor0_cls_fm [RPN_CLS_OUT_CH*RPN_INPUT0_IN_FM_HEIGHT*RPN_INPUT0_IN_FM_WIDTH];
// fm_t rpn_anchor1_cls_fm [RPN_CLS_OUT_CH*RPN_INPUT1_IN_FM_HEIGHT*RPN_INPUT1_IN_FM_WIDTH];
// fm_t rpn_anchor2_cls_fm [RPN_CLS_OUT_CH*RPN_INPUT2_IN_FM_HEIGHT*RPN_INPUT2_IN_FM_WIDTH];
fm_t rpn_anchor3_cls_fm [RPN_CLS_OUT_CH*RPN_INPUT3_IN_FM_HEIGHT*RPN_INPUT3_IN_FM_WIDTH];
fm_t rpn_anchor4_cls_fm [RPN_CLS_OUT_CH*RPN_INPUT4_IN_FM_HEIGHT*RPN_INPUT4_IN_FM_WIDTH];

// fm_t rpn_anchor0_reg_fm [RPN_REG_OUT_CH*RPN_INPUT0_IN_FM_HEIGHT*RPN_INPUT0_IN_FM_WIDTH/4][4];
// fm_t rpn_anchor1_reg_fm [RPN_REG_OUT_CH*RPN_INPUT1_IN_FM_HEIGHT*RPN_INPUT1_IN_FM_WIDTH/4][4];
// fm_t rpn_anchor2_reg_fm [RPN_REG_OUT_CH*RPN_INPUT2_IN_FM_HEIGHT*RPN_INPUT2_IN_FM_WIDTH/4][4];
fm_t rpn_anchor3_reg_fm [RPN_REG_OUT_CH*RPN_INPUT3_IN_FM_HEIGHT*RPN_INPUT3_IN_FM_WIDTH/4][4];
fm_t rpn_anchor4_reg_fm [RPN_REG_OUT_CH*RPN_INPUT4_IN_FM_HEIGHT*RPN_INPUT4_IN_FM_WIDTH/4][4];

// int rpn_topk_index0[RPN_PRE_NMS_SIZE0];
// int rpn_topk_index1[RPN_PRE_NMS_SIZE1];
// int rpn_topk_index2[RPN_PRE_NMS_SIZE2];
int rpn_topk_index3[RPN_PRE_NMS_SIZE3];


void rpn_top2   (
    int rpn_topk_index0[RPN_PRE_NMS_SIZE0],
    int rpn_topk_index1[RPN_PRE_NMS_SIZE1],
    int rpn_topk_index2[RPN_PRE_NMS_SIZE2],

    fm_t rpn_anchor0_reg_fm [RPN_REG_OUT_CH*RPN_INPUT0_IN_FM_HEIGHT*RPN_INPUT0_IN_FM_WIDTH/4][4],
    fm_t rpn_anchor1_reg_fm [RPN_REG_OUT_CH*RPN_INPUT0_IN_FM_HEIGHT*RPN_INPUT0_IN_FM_WIDTH/4][4],
    fm_t rpn_anchor2_reg_fm [RPN_REG_OUT_CH*RPN_INPUT0_IN_FM_HEIGHT*RPN_INPUT0_IN_FM_WIDTH/4][4],

    fm_t rpn_anchor0_cls_fm [RPN_CLS_OUT_CH*RPN_INPUT0_IN_FM_HEIGHT*RPN_INPUT0_IN_FM_WIDTH],
    fm_t rpn_anchor1_cls_fm [RPN_CLS_OUT_CH*RPN_INPUT1_IN_FM_HEIGHT*RPN_INPUT1_IN_FM_WIDTH],
    fm_t rpn_anchor2_cls_fm [RPN_CLS_OUT_CH*RPN_INPUT2_IN_FM_HEIGHT*RPN_INPUT2_IN_FM_WIDTH],

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
    // std::cout << "Begin processing RPN CONV 0..." << std::endl;
    

    // for(int c = 0; c < RPN_CONV_IN_CH; c++)
    // {
    //     for(int h = 0; h < RPN_INPUT0_IN_FM_HEIGHT; h++)
    //     {
    //         for(int w = 0; w < RPN_INPUT0_IN_FM_WIDTH; w++)
    //         {
    //             rpn_layer_in_fm[c][h][w] = rpn_input0_fm[c][h][w];
    //             // std::cout<<rpn_layer_in_fm[c][h][w]<<" ";
    //         }
    //         // std::cout<<std::endl;
    //     }
    // }


    // rpn_3x3_conv <RPN_CONV_IN_CH,  RPN_INPUT0_IN_FM_HEIGHT, RPN_INPUT0_IN_FM_WIDTH,
    //                 RPN_CONV_OUT_CH, RPN_INPUT0_IN_FM_HEIGHT, RPN_INPUT0_IN_FM_WIDTH,
    //                 RPN_CONV_STRIDE, RPN_LAST_LAYER_DISABLE, 0>
    //                 (rpn_conv_weight, rpn_conv_bias, true);
    

    // for(int c = 0; c < RPN_CONV_IN_CH; c++)
    // {
    //     for(int h = 0; h < RPN_INPUT0_IN_FM_HEIGHT; h++)
    //     {
    //         for(int w = 0; w < RPN_INPUT0_IN_FM_WIDTH; w++)
    //         {
    //             rpn_output0_fm[c][h][w] = rpn_layer_out_fm[c][h][w];
    //             rpn_layer_in_fm[c][h][w]=0;
    //             rpn_layer_out_fm[c][h][w]=0;
    //         }
    //     }
    // }


    // std::cout << "Begin processing RPN CONV 1..." << std::endl;

    // for(int c = 0; c < RPN_CONV_IN_CH; c++)
    // {
    //     for(int h = 0; h < RPN_INPUT1_IN_FM_HEIGHT; h++)
    //     {
    //         for(int w = 0; w < RPN_INPUT1_IN_FM_WIDTH; w++)
    //         {
    //             rpn_layer_in_fm[c][h][w] = rpn_input1_fm[c][h][w];
    //         }
    //     }
    // }


    // rpn_3x3_conv <RPN_CONV_IN_CH,  RPN_INPUT1_IN_FM_HEIGHT, RPN_INPUT1_IN_FM_WIDTH,
    //                 RPN_CONV_OUT_CH, RPN_INPUT1_IN_FM_HEIGHT, RPN_INPUT1_IN_FM_WIDTH,
    //                 RPN_CONV_STRIDE, RPN_LAST_LAYER_DISABLE,0>
    //                 (rpn_conv_weight, rpn_conv_bias, true);
    

    // for(int c = 0; c < RPN_CONV_IN_CH; c++)
    // {
    //     for(int h = 0; h < RPN_INPUT1_IN_FM_HEIGHT; h++)
    //     {
    //         for(int w = 0; w < RPN_INPUT1_IN_FM_WIDTH; w++)
    //         {
    //             rpn_output1_fm[c][h][w] = rpn_layer_out_fm[c][h][w];
    //             rpn_layer_in_fm[c][h][w]=0;
    //             rpn_layer_out_fm[c][h][w]=0;
    //         }
    //     }
    // }



    // std::cout << "Begin processing RPN CONV 2..." << std::endl;


    // for(int c = 0; c < RPN_CONV_IN_CH; c++)
    // {
    //     for(int h = 0; h < RPN_INPUT2_IN_FM_HEIGHT; h++)
    //     {
    //         for(int w = 0; w < RPN_INPUT2_IN_FM_WIDTH; w++)
    //         {
    //             rpn_layer_in_fm[c][h][w] = rpn_input2_fm[c][h][w];
    //         }
    //     }
    // }


    // rpn_3x3_conv <RPN_CONV_IN_CH,  RPN_INPUT2_IN_FM_HEIGHT, RPN_INPUT2_IN_FM_WIDTH,
    //                 RPN_CONV_OUT_CH, RPN_INPUT2_IN_FM_HEIGHT, RPN_INPUT2_IN_FM_WIDTH,
    //                 RPN_CONV_STRIDE, RPN_LAST_LAYER_DISABLE,0>
    //                 (rpn_conv_weight, rpn_conv_bias, true);
    

    // for(int c = 0; c < RPN_CONV_IN_CH; c++)
    // {
    //     for(int h = 0; h < RPN_INPUT2_IN_FM_HEIGHT; h++)
    //     {
    //         for(int w = 0; w < RPN_INPUT2_IN_FM_WIDTH; w++)
    //         {
    //             rpn_output2_fm[c][h][w] = rpn_layer_out_fm[c][h][w];
    //             rpn_layer_in_fm[c][h][w]=0;
    //             rpn_layer_out_fm[c][h][w]=0;
    //         }
    //     }
    // }

    std::cout << "Begin processing RPN CONV 3..." << std::endl;



    for(int c = 0; c < RPN_CONV_IN_CH; c++)
    {
        for(int h = 0; h < RPN_INPUT3_IN_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RPN_INPUT3_IN_FM_WIDTH; w++)
            {
                rpn_layer_in_fm[c][h][w] = rpn_input3_fm[c][h][w];
            }
        }
    }


    rpn_3x3_conv <RPN_CONV_IN_CH,  RPN_INPUT3_IN_FM_HEIGHT, RPN_INPUT3_IN_FM_WIDTH,
                    RPN_CONV_OUT_CH, RPN_INPUT3_IN_FM_HEIGHT, RPN_INPUT3_IN_FM_WIDTH,
                    RPN_CONV_STRIDE, RPN_LAST_LAYER_DISABLE,1>
                    (rpn_conv_weight, rpn_conv_bias, true);
    

    for(int c = 0; c < RPN_CONV_IN_CH; c++)
    {
        for(int h = 0; h < RPN_INPUT3_IN_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RPN_INPUT3_IN_FM_WIDTH; w++)
            {
                rpn_output3_fm[c][h][w] = rpn_layer_out_fm[c][h][w];
                rpn_layer_in_fm[c][h][w]=0;
                rpn_layer_out_fm[c][h][w]=0;
            }
        }
    }

    std::cout << "Begin processing RPN CONV 4..." << std::endl;



    for(int c = 0; c < RPN_CONV_IN_CH; c++)
    {
        for(int h = 0; h < RPN_INPUT4_IN_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RPN_INPUT4_IN_FM_WIDTH; w++)
            {
                rpn_layer_in_fm[c][h][w] = rpn_input4_fm[c][h][w];
                
            }
            
        }
    }


    rpn_3x3_conv <RPN_CONV_IN_CH,  RPN_INPUT4_IN_FM_HEIGHT, RPN_INPUT4_IN_FM_WIDTH,
                    RPN_CONV_OUT_CH, RPN_INPUT4_IN_FM_HEIGHT, RPN_INPUT4_IN_FM_WIDTH,
                    RPN_CONV_STRIDE, RPN_LAST_LAYER_DISABLE,1>
                    (rpn_conv_weight, rpn_conv_bias, true);
    

    for(int c = 0; c < RPN_CONV_IN_CH; c++)
    {
        for(int h = 0; h < RPN_INPUT4_IN_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RPN_INPUT4_IN_FM_WIDTH; w++)
            {
                rpn_output4_fm[c][h][w] = rpn_layer_out_fm[c][h][w];
                rpn_layer_in_fm[c][h][w]=0;
                rpn_layer_out_fm[c][h][w]=0;
            }
        }
    }


    //----------------------------------------------------------------------
    // RPN CLS Conv
    //----------------------------------------------------------------------



    // std::cout << "Begin processing RPN CLS 0..." << std::endl;



    // for(int c = 0; c < RPN_CONV_IN_CH; c++)
    // {
    //     for(int h = 0; h < RPN_INPUT0_IN_FM_HEIGHT; h++)
    //     {
    //         for(int w = 0; w < RPN_INPUT0_IN_FM_WIDTH; w++)
    //         {
    //             rpn_layer_in_fm[c][h][w] = rpn_output0_fm[c][h][w];
    //             // cout<<rpn_layer_in_fm[c][h][w]<<" ";
    //         }
    //         // cout<<endl;
    //     }
    // }


    // rpn_1x1_conv <RPN_CLS_IN_CH,  RPN_INPUT0_IN_FM_HEIGHT, RPN_INPUT0_IN_FM_WIDTH,
    //                 RPN_CLS_OUT_CH, RPN_INPUT0_IN_FM_HEIGHT, RPN_INPUT0_IN_FM_WIDTH,
    //                 RPN_CLS_STRIDE, RPN_LAST_LAYER_DISABLE,0,0,1,0>
    //                 (rpn_cls_weight, rpn_cls_bias, false);
    

    // for(int h = 0; h < RPN_INPUT0_IN_FM_HEIGHT; h++)
    // {
    //     for(int w = 0; w < RPN_INPUT0_IN_FM_WIDTH; w++)
    //     {
    //         for(int c = 0; c < RPN_CLS_OUT_CH; c++)
    //         {
    //             rpn_output0_cls_fm[c][h][w] = rpn_layer_out_fm[c][h][w];
    //             rpn_anchor0_cls_fm[h*(RPN_INPUT0_IN_FM_WIDTH*RPN_CLS_OUT_CH) + w*RPN_CLS_OUT_CH+ c] = 1/(1+(fm_t)exp((float)(-1*rpn_layer_out_fm[c][h][w])));
    //             rpn_layer_in_fm[c][h][w]=0;
    //             rpn_layer_out_fm[c][h][w]=0;
    //         }
    //         // cout<<endl;
    //     }
    // }


    // std::cout << "Begin processing RPN CLS 1..." << std::endl;



    // for(int c = 0; c < RPN_CONV_IN_CH; c++)
    // {
    //     for(int h = 0; h < RPN_INPUT1_IN_FM_HEIGHT; h++)
    //     {
    //         for(int w = 0; w < RPN_INPUT1_IN_FM_WIDTH; w++)
    //         {
    //             rpn_layer_in_fm[c][h][w] = rpn_output1_fm[c][h][w];
    //         }
    //     }
    // }

    // rpn_1x1_conv <RPN_CLS_IN_CH,  RPN_INPUT1_IN_FM_HEIGHT, RPN_INPUT1_IN_FM_WIDTH,
    //                 RPN_CLS_OUT_CH, RPN_INPUT1_IN_FM_HEIGHT, RPN_INPUT1_IN_FM_WIDTH,
    //                 RPN_CLS_STRIDE, RPN_LAST_LAYER_DISABLE,0,0,1,0>
    //                 (rpn_cls_weight, rpn_cls_bias, false);
    

    // for(int h = 0; h < RPN_INPUT1_IN_FM_HEIGHT; h++)
    // {
    //     for(int w = 0; w < RPN_INPUT1_IN_FM_WIDTH; w++)
    //     {
    //         for(int c = 0; c < RPN_CLS_OUT_CH; c++)
    //         {
    //             rpn_output1_cls_fm[c][h][w] = rpn_layer_out_fm[c][h][w];
    //             rpn_anchor1_cls_fm[h*(RPN_INPUT1_IN_FM_WIDTH*RPN_CLS_OUT_CH) + w*RPN_CLS_OUT_CH+ c] = 1/(1+(fm_t)exp((float)(-1*rpn_layer_out_fm[c][h][w])));                
    //             rpn_layer_in_fm[c][h][w]=0;
    //             rpn_layer_out_fm[c][h][w]=0;
    //         }
    //     }
    // }

    // std::cout << "Begin processing RPN CLS 2..." << std::endl;

    // for(int c = 0; c < RPN_CONV_IN_CH; c++)
    // {
    //     for(int h = 0; h < RPN_INPUT2_IN_FM_HEIGHT; h++)
    //     {
    //         for(int w = 0; w < RPN_INPUT2_IN_FM_WIDTH; w++)
    //         {
    //             rpn_layer_in_fm[c][h][w] = rpn_output2_fm[c][h][w];
    //         }
    //     }
    // }

    // rpn_1x1_conv <RPN_CLS_IN_CH,  RPN_INPUT2_IN_FM_HEIGHT, RPN_INPUT2_IN_FM_WIDTH,
    //                 RPN_CLS_OUT_CH, RPN_INPUT2_IN_FM_HEIGHT, RPN_INPUT2_IN_FM_WIDTH,
    //                 RPN_CLS_STRIDE, RPN_LAST_LAYER_DISABLE,0,0,1,0>
    //                 (rpn_cls_weight, rpn_cls_bias, false);
    

    // for(int h = 0; h < RPN_INPUT2_IN_FM_HEIGHT; h++)
    // {
    //     for(int w = 0; w < RPN_INPUT2_IN_FM_WIDTH; w++)
    //     {
    //         for(int c = 0; c < RPN_CLS_OUT_CH; c++)
    //         {
    //             rpn_output2_cls_fm[c][h][w] = rpn_layer_out_fm[c][h][w];
    //             rpn_anchor2_cls_fm[h*(RPN_INPUT2_IN_FM_WIDTH*RPN_CLS_OUT_CH) + w*RPN_CLS_OUT_CH+ c] = 1/(1+(fm_t)exp((float)(-1*rpn_layer_out_fm[c][h][w])));
    //             rpn_layer_in_fm[c][h][w]=0;
    //             rpn_layer_out_fm[c][h][w]=0;
    //         }
    //     }
    // }

    std::cout << "Begin processing RPN CLS 3..." << std::endl;

    for(int c = 0; c < RPN_CONV_IN_CH; c++)
    {
        for(int h = 0; h < RPN_INPUT3_IN_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RPN_INPUT3_IN_FM_WIDTH; w++)
            {
                rpn_layer_in_fm[c][h][w] = rpn_output3_fm[c][h][w];
            }
        }
    }


    rpn_1x1_conv <RPN_CLS_IN_CH,  RPN_INPUT3_IN_FM_HEIGHT, RPN_INPUT3_IN_FM_WIDTH,
                    RPN_CLS_OUT_CH, RPN_INPUT3_IN_FM_HEIGHT, RPN_INPUT3_IN_FM_WIDTH,
                    RPN_CLS_STRIDE, RPN_LAST_LAYER_DISABLE,1,0,1,0>
                    (rpn_cls_weight, rpn_cls_bias, false);
    

    for(int h = 0; h < RPN_INPUT3_IN_FM_HEIGHT; h++)
    {
        for(int w = 0; w < RPN_INPUT3_IN_FM_WIDTH; w++)
        {
            for(int c = 0; c < RPN_CLS_OUT_CH; c++)
            {
                rpn_output3_cls_fm[c][h][w] = rpn_layer_out_fm[c][h][w];
                rpn_anchor3_cls_fm[h*(RPN_INPUT3_IN_FM_WIDTH*RPN_CLS_OUT_CH) + w*RPN_CLS_OUT_CH+ c] = 1/(1+(fm_t)exp((float)(-1*rpn_layer_out_fm[c][h][w])));
                rpn_layer_in_fm[c][h][w]=0;
                rpn_layer_out_fm[c][h][w]=0;
            }
            // cout<<endl;
        }
    }




   std::cout << "Begin processing RPN CLS 4..." << std::endl;



    for(int c = 0; c < RPN_CONV_IN_CH; c++)
    {
        for(int h = 0; h < RPN_INPUT4_IN_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RPN_INPUT4_IN_FM_WIDTH; w++)
            {
                rpn_layer_in_fm[c][h][w] = rpn_output4_fm[c][h][w];
            }
        }
    }


    rpn_1x1_conv <RPN_CLS_IN_CH,  RPN_INPUT4_IN_FM_HEIGHT, RPN_INPUT4_IN_FM_WIDTH,
                    RPN_CLS_OUT_CH, RPN_INPUT4_IN_FM_HEIGHT, RPN_INPUT4_IN_FM_WIDTH,
                    RPN_CLS_STRIDE, RPN_LAST_LAYER_DISABLE,1,1,1,0>
                    (rpn_cls_weight, rpn_cls_bias, false);
    

    for(int h = 0; h < RPN_INPUT4_IN_FM_HEIGHT; h++)
    {
        for(int w = 0; w < RPN_INPUT4_IN_FM_WIDTH; w++)
        {
            for(int c = 0; c < RPN_CLS_OUT_CH; c++)
            {
                rpn_output4_cls_fm[c][h][w] = rpn_layer_out_fm[c][h][w];
                rpn_anchor4_cls_fm[h*(RPN_INPUT4_IN_FM_WIDTH*RPN_CLS_OUT_CH) + w*RPN_CLS_OUT_CH+ c] = 1/(1+exp((float)(-1*rpn_layer_out_fm[c][h][w])));
                rpn_layer_in_fm[c][h][w]=0;
                rpn_layer_out_fm[c][h][w]=0;
            }
        }
    }




    //----------------------------------------------------------------------
    // RPN REG
    //----------------------------------------------------------------------



    // std::cout << "Begin processing RPN REG 0..." << std::endl;



    // for(int c = 0; c < RPN_CONV_IN_CH; c++)
    // {
    //     for(int h = 0; h < RPN_INPUT0_IN_FM_HEIGHT; h++)
    //     {
    //         for(int w = 0; w < RPN_INPUT0_IN_FM_WIDTH; w++)
    //         {
    //             rpn_layer_in_fm[c][h][w] = rpn_output0_fm[c][h][w];
    //         }
    //     }
    // }


    // rpn_1x1_conv <RPN_REG_IN_CH,  RPN_INPUT0_IN_FM_HEIGHT, RPN_INPUT0_IN_FM_WIDTH,
    //                 RPN_REG_OUT_CH, RPN_INPUT0_IN_FM_HEIGHT, RPN_INPUT0_IN_FM_WIDTH,
    //                 RPN_REG_STRIDE, RPN_LAST_LAYER_DISABLE,0,0,1,0>
    //                 (rpn_reg_weight, rpn_reg_bias, false);
    

    // for(int h = 0; h < RPN_INPUT0_IN_FM_HEIGHT; h++)
    // {
    //     for(int w = 0; w < RPN_INPUT0_IN_FM_WIDTH; w++)
    //     {
    //         for(int c = 0; c < RPN_REG_OUT_CH; c++)
    //         {
    //             rpn_output0_reg_fm[c][h][w] = rpn_layer_out_fm[c][h][w];
    //             rpn_anchor0_reg_fm[(h*(RPN_INPUT0_IN_FM_WIDTH*RPN_REG_OUT_CH)+w*RPN_REG_OUT_CH+c)/4][(h*(RPN_INPUT0_IN_FM_WIDTH*RPN_REG_OUT_CH)+w*RPN_REG_OUT_CH+c)%4]= rpn_layer_out_fm[c][h][w];
    //             rpn_layer_in_fm[c][h][w]=0;
    //             rpn_layer_out_fm[c][h][w]=0;
    //         }
    //     }
    // }


    // std::cout << "Begin processing RPN REG 1..." << std::endl;



    // for(int c = 0; c < RPN_CONV_IN_CH; c++)
    // {
    //     for(int h = 0; h < RPN_INPUT1_IN_FM_HEIGHT; h++)
    //     {
    //         for(int w = 0; w < RPN_INPUT1_IN_FM_WIDTH; w++)
    //         {
    //             rpn_layer_in_fm[c][h][w] = rpn_output1_fm[c][h][w];
    //         }
    //     }
    // }


    // rpn_1x1_conv <RPN_REG_IN_CH,  RPN_INPUT1_IN_FM_HEIGHT, RPN_INPUT1_IN_FM_WIDTH,
    //                 RPN_REG_OUT_CH, RPN_INPUT1_IN_FM_HEIGHT, RPN_INPUT1_IN_FM_WIDTH,
    //                 RPN_REG_STRIDE, RPN_LAST_LAYER_DISABLE,0,0,1,0>
    //                 (rpn_reg_weight, rpn_reg_bias, false);
    

    // for(int h = 0; h < RPN_INPUT1_IN_FM_HEIGHT; h++)
    // {
    //     for(int w = 0; w < RPN_INPUT1_IN_FM_WIDTH; w++)
    //     {
    //         for(int c = 0; c < RPN_REG_OUT_CH; c++)
    //         {
    //             rpn_output1_reg_fm[c][h][w] = rpn_layer_out_fm[c][h][w];
    //             rpn_anchor1_reg_fm[(h*(RPN_INPUT1_IN_FM_WIDTH*RPN_REG_OUT_CH)+w*RPN_REG_OUT_CH+c)/4][(h*(RPN_INPUT1_IN_FM_WIDTH*RPN_REG_OUT_CH)+w*RPN_REG_OUT_CH+c)%4]= rpn_layer_out_fm[c][h][w];
    //             rpn_layer_in_fm[c][h][w]=0;
    //             rpn_layer_out_fm[c][h][w]=0;
    //         }
    //     }
    // }


    // std::cout << "Begin processing RPN REG 2..." << std::endl;



    // for(int c = 0; c < RPN_CONV_IN_CH; c++)
    // {
    //     for(int h = 0; h < RPN_INPUT2_IN_FM_HEIGHT; h++)
    //     {
    //         for(int w = 0; w < RPN_INPUT2_IN_FM_WIDTH; w++)
    //         {
    //             rpn_layer_in_fm[c][h][w] = rpn_output2_fm[c][h][w];
    //         }
    //     }
    // }


    // rpn_1x1_conv <RPN_REG_IN_CH,  RPN_INPUT2_IN_FM_HEIGHT, RPN_INPUT2_IN_FM_WIDTH,
    //                 RPN_REG_OUT_CH, RPN_INPUT2_IN_FM_HEIGHT, RPN_INPUT2_IN_FM_WIDTH,
    //                 RPN_REG_STRIDE, RPN_LAST_LAYER_DISABLE,0,0,1,0>
    //                 (rpn_reg_weight, rpn_reg_bias, false);
    

    // for(int h = 0; h < RPN_INPUT2_IN_FM_HEIGHT; h++)
    // {
    //     for(int w = 0; w < RPN_INPUT2_IN_FM_WIDTH; w++)
    //     {
    //         for(int c = 0; c < RPN_REG_OUT_CH; c++)
    //         {
    //             rpn_output2_reg_fm[c][h][w] = rpn_layer_out_fm[c][h][w];
    //             rpn_anchor2_reg_fm[(h*(RPN_INPUT2_IN_FM_WIDTH*RPN_REG_OUT_CH)+w*RPN_REG_OUT_CH+c)/4][(h*(RPN_INPUT2_IN_FM_WIDTH*RPN_REG_OUT_CH)+w*RPN_REG_OUT_CH+c)%4]= rpn_layer_out_fm[c][h][w];
    //             rpn_layer_in_fm[c][h][w]=0;
    //             rpn_layer_out_fm[c][h][w]=0;
    //         }
    //     }
    // }










    std::cout << "Begin processing RPN REG 3..." << std::endl;



    for(int c = 0; c < RPN_CONV_IN_CH; c++)
    {
        for(int h = 0; h < RPN_INPUT3_IN_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RPN_INPUT3_IN_FM_WIDTH; w++)
            {
                rpn_layer_in_fm[c][h][w] = rpn_output3_fm[c][h][w];
            }
        }
    }


    rpn_1x1_conv <RPN_REG_IN_CH,  RPN_INPUT3_IN_FM_HEIGHT, RPN_INPUT3_IN_FM_WIDTH,
                    RPN_REG_OUT_CH, RPN_INPUT3_IN_FM_HEIGHT, RPN_INPUT3_IN_FM_WIDTH,
                    RPN_REG_STRIDE, RPN_LAST_LAYER_DISABLE,1,0,1,0>
                    (rpn_reg_weight, rpn_reg_bias, false);
    

    for(int h = 0; h < RPN_INPUT3_IN_FM_HEIGHT; h++)
    {
        for(int w = 0; w < RPN_INPUT3_IN_FM_WIDTH; w++)
        {
            for(int c = 0; c < RPN_REG_OUT_CH; c++)
            {
                rpn_output3_reg_fm[c][h][w] = rpn_layer_out_fm[c][h][w];
                rpn_anchor3_reg_fm[(h*(RPN_INPUT3_IN_FM_WIDTH*RPN_REG_OUT_CH)+w*RPN_REG_OUT_CH+c)/4][(h*(RPN_INPUT3_IN_FM_WIDTH*RPN_REG_OUT_CH)+w*RPN_REG_OUT_CH+c)%4]= rpn_layer_out_fm[c][h][w];
                rpn_layer_in_fm[c][h][w]=0;
                rpn_layer_out_fm[c][h][w]=0;
            }
        }
    }




   std::cout << "Begin processing RPN REG 4..." << std::endl;



    for(int c = 0; c < RPN_CONV_IN_CH; c++)
    {
        for(int h = 0; h < RPN_INPUT4_IN_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RPN_INPUT4_IN_FM_WIDTH; w++)
            {
                rpn_layer_in_fm[c][h][w] = rpn_output4_fm[c][h][w];
            }
        }
    }


    rpn_1x1_conv <RPN_REG_IN_CH,  RPN_INPUT4_IN_FM_HEIGHT, RPN_INPUT4_IN_FM_WIDTH,
                    RPN_REG_OUT_CH, RPN_INPUT4_IN_FM_HEIGHT, RPN_INPUT4_IN_FM_WIDTH,
                    RPN_REG_STRIDE, RPN_LAST_LAYER_DISABLE,1,1,1,0>
                    (rpn_reg_weight, rpn_reg_bias, false);
    

    for(int h = 0; h < RPN_INPUT4_IN_FM_HEIGHT; h++)
    {
        for(int w = 0; w < RPN_INPUT4_IN_FM_WIDTH; w++)
        {
            for(int c = 0; c < RPN_REG_OUT_CH; c++)
            {
                rpn_output4_reg_fm[c][h][w] = rpn_layer_out_fm[c][h][w];
                rpn_anchor4_reg_fm[(h*(RPN_INPUT4_IN_FM_WIDTH*RPN_REG_OUT_CH)+w*RPN_REG_OUT_CH+c)/4][(h*(RPN_INPUT4_IN_FM_WIDTH*RPN_REG_OUT_CH)+w*RPN_REG_OUT_CH+c)%4]= rpn_layer_out_fm[c][h][w];
                rpn_layer_in_fm[c][h][w]=0;
                rpn_layer_out_fm[c][h][w]=0;
            }
        }
    }





    //----------------------------------------------------------------------
    // ANCHOR BOX GENERATION AND WORKING
    //----------------------------------------------------------------------

    

    //Sorting topk indices for anchor0
    // rpn_anchor0_cls_fm=>scores

//    std::cout << "Begin processing RPN Anchor Gen 0..." << std::endl;


//     bool flag0[RPN_CLS_OUT_CH*RPN_INPUT0_IN_FM_HEIGHT*RPN_INPUT0_IN_FM_WIDTH]={false};
//     for(int i = 0; i<RPN_PRE_NMS_SIZE0; i++)
//     {
//         fm_t maximum_score = 0.0;
//         int index = -1;
//         for(int j = 0; j<RPN_CLS_OUT_CH*RPN_INPUT0_IN_FM_HEIGHT*RPN_INPUT0_IN_FM_WIDTH; j++)
//         {
//             if(flag0[j] == true)    continue;
//             if(rpn_anchor0_cls_fm[j]>maximum_score)
//             {
//                 maximum_score = rpn_anchor0_cls_fm[j];
//                 index = j;
//             }
            
//         }
//             rpn_topk_index0[i]=index;
//             flag0[index]=true;
//     }


//    std::cout << "Begin processing RPN Anchor Gen 1..." << std::endl;


//     bool flag1[RPN_CLS_OUT_CH*RPN_INPUT1_IN_FM_HEIGHT*RPN_INPUT1_IN_FM_WIDTH]={false};
//     for(int i = 0; i<RPN_PRE_NMS_SIZE1; i++)
//     {
//         fm_t maximum_score = 0.0;
//         int index = -1;
//         for(int j = 0; j<RPN_CLS_OUT_CH*RPN_INPUT1_IN_FM_HEIGHT*RPN_INPUT1_IN_FM_WIDTH; j++)
//         {
//             if(flag1[j] == true)    continue;
//             if(rpn_anchor1_cls_fm[j]>maximum_score)
//             {
//                 maximum_score = rpn_anchor1_cls_fm[j];
//                 index = j;
//             }
//         }
//             rpn_topk_index1[i]=index;
//             flag1[index]=true;
//     }


//    std::cout << "Begin processing RPN Anchor Gen 2..." << std::endl;



//     bool flag2[RPN_CLS_OUT_CH*RPN_INPUT2_IN_FM_HEIGHT*RPN_INPUT2_IN_FM_WIDTH]={false};
//     for(int i = 0; i<RPN_PRE_NMS_SIZE2; i++)
//     {
//         fm_t maximum_score = 0.0;
//         int index = -1;
//         for(int j = 0; j<RPN_CLS_OUT_CH*RPN_INPUT2_IN_FM_HEIGHT*RPN_INPUT2_IN_FM_WIDTH; j++)
//         {
//             if(flag2[j] == true)    continue;
//             if(rpn_anchor2_cls_fm[j]>maximum_score)
//             {
//                 maximum_score = rpn_anchor2_cls_fm[j];
//                 index = j;
//             }
//         }

//             rpn_topk_index2[i]=index;
//             flag2[index]=true;
//     }


   std::cout << "Begin processing RPN Anchor Gen 3..." << std::endl;



    bool flag3[RPN_CLS_OUT_CH*RPN_INPUT3_IN_FM_HEIGHT*RPN_INPUT3_IN_FM_WIDTH]={false};
    for(int i = 0; i<RPN_PRE_NMS_SIZE3; i++)
    {
        fm_t maximum_score = 0.0;
        int index = -1;
        for(int j = 0; j<RPN_CLS_OUT_CH*RPN_INPUT3_IN_FM_HEIGHT*RPN_INPUT3_IN_FM_WIDTH; j++)
        {
            if(flag3[j] == true)    continue;
            if(rpn_anchor3_cls_fm[j]>maximum_score)
            {
                maximum_score = rpn_anchor3_cls_fm[j];
                index = j;
            }
        }
            rpn_topk_index3[i]=index;
            flag3[index]=true;
    }


   std::cout << "Begin processing RPN Anchor Gen 4..." << std::endl;

    fm_t rois[RPN_PRE_NMS_SIZE][4];
    fm_t deltas[RPN_PRE_NMS_SIZE][4];
    fm_t scores[RPN_PRE_NMS_SIZE];


    for(int i = 0; i<1000; i++)
    {
        for(int j = 0; j<4 ; j++)
        {
            rois[0*1000+i][j]=anchor_box0[rpn_topk_index0[i]][j];
            rois[1*1000+i][j]=anchor_box1[rpn_topk_index1[i]][j];
            rois[2*1000+i][j]=anchor_box2[rpn_topk_index2[i]][j];
            rois[3*1000+i][j]=anchor_box3[rpn_topk_index3[i]][j];
            deltas[0*1000+i][j]=rpn_anchor0_reg_fm[rpn_topk_index0[i]][j];
            deltas[1*1000+i][j]=rpn_anchor1_reg_fm[rpn_topk_index1[i]][j];
            deltas[2*1000+i][j]=rpn_anchor2_reg_fm[rpn_topk_index2[i]][j];
            deltas[3*1000+i][j]=rpn_anchor3_reg_fm[rpn_topk_index3[i]][j];
        }
        scores[0*1000+i]=rpn_anchor0_cls_fm[rpn_topk_index0[i]];
        scores[1*1000+i]=rpn_anchor1_cls_fm[rpn_topk_index1[i]];
        scores[2*1000+i]=rpn_anchor2_cls_fm[rpn_topk_index2[i]];
        scores[3*1000+i]=rpn_anchor3_cls_fm[rpn_topk_index3[i]];
    }
    for(int i = 0; i<720; i++)
    {
        for(int j = 0; j<4; j++)
        {
            rois[4*1000+i][j]=anchor_box4[i][j];
            deltas[4*1000+i][j]=rpn_anchor4_reg_fm[i][j];
        }
        scores[4*1000+i]=rpn_anchor4_cls_fm[i];
    }

    //----------------------------------------------------------------------
    // PRE NMS WORK: Converting BBoxesPred to Original Size Bbox
    //----------------------------------------------------------------------

    
    fm_t max_ratio = 4.135166556742355;
    fm_t dxy[RPN_PRE_NMS_SIZE][2];
    fm_t dwh[RPN_PRE_NMS_SIZE][2];
    fm_t pxy[RPN_PRE_NMS_SIZE][2];
    fm_t pwh[RPN_PRE_NMS_SIZE][2];
    fm_t dxy_wh[RPN_PRE_NMS_SIZE][2];
    fm_t gxy[RPN_PRE_NMS_SIZE][2];
    fm_t gwh[RPN_PRE_NMS_SIZE][2];
    fm_t area[RPN_PRE_NMS_SIZE];
    fm_t half = 0.5;
    for(int i = 0 ; i< RPN_PRE_NMS_SIZE; i++)
    {
        dxy[i][0]= deltas[i][0];
        dwh[i][0]= deltas[i][2];
        pxy[i][0]= (rois[i][0]+rois[i][2])*half;
        pwh[i][0]= rois[i][2]-rois[i][0];
        dxy[i][1]= deltas[i][1];
        dwh[i][1]= deltas[i][3];
        pxy[i][1]= (rois[i][1]+rois[i][3])*half;
        pwh[i][1]= rois[i][3]-rois[i][1];
        dxy_wh[i][0] = pwh[i][0] * dxy[i][0];
        dxy_wh[i][1] = pwh[i][1] * dxy[i][1];
        if(dwh[i][0]>max_ratio) dwh[i][0]= max_ratio;
        if(dwh[i][0]<-1*max_ratio) dwh[i][0]= -1*max_ratio;
        if(dwh[i][1]>max_ratio) dwh[i][1]= max_ratio;
        if(dwh[i][1]<-1*max_ratio) dwh[i][1]= -1*max_ratio;
        gxy[i][0] = pxy[i][0] + dxy_wh[i][0];
        gxy[i][1] = pxy[i][1] + dxy_wh[i][1];
        gwh[i][0] = pwh[i][0] * (fm_t)exp((float)dwh[i][0]);
        gwh[i][1] = pwh[i][1] * (fm_t)exp((float)dwh[i][1]);
        
        bboxes[i][0] = gxy[i][0]  - (gwh[i][0]*half);
        bboxes[i][1] = gxy[i][1]  - (gwh[i][1]*half);
        bboxes[i][2] = gxy[i][0]  + (gwh[i][0]*half);
        bboxes[i][3] = gxy[i][1]  + (gwh[i][1]*half);
        if(bboxes[i][0]>1280) bboxes[i][0]=1280;
        if(bboxes[i][2]>1280) bboxes[i][2]=1280;
        if(bboxes[i][1]>720) bboxes[i][1]=720;
        if(bboxes[i][3]>720) bboxes[i][3]=720;
        
        if(bboxes[i][0]<0) bboxes[i][0]=0;
        if(bboxes[i][1]<0) bboxes[i][1]=0;
        if(bboxes[i][2]<0) bboxes[i][2]=0;
        if(bboxes[i][3]<0) bboxes[i][3]=0;
        bboxes[i][0] += ((int)(i/1000))*1281;
        bboxes[i][1] += ((int)(i/1000))*1281;
        bboxes[i][2] += ((int)(i/1000))*1281;
        bboxes[i][3] += ((int)(i/1000))*1281;

        area[i]=  (bboxes[i][2]-bboxes[i][0])*(bboxes[i][3]-bboxes[i][1]);
    }



    //----------------------------------------------------------------------
    // NMS 
    //----------------------------------------------------------------------

    std::cout << "Begin processing NMS..." << std::endl;

    int nms_index[RPN_PRE_NMS_SIZE];
    bool nms_flag[RPN_PRE_NMS_SIZE]={false};

    for(int i = 0; i<RPN_PRE_NMS_SIZE; i++)
    {
        fm_t tm = -100;
        int index = -1;
        for(int j = 0; j < RPN_PRE_NMS_SIZE; j++)
        {
            if(nms_flag[j]==true) continue;
            if(tm<scores[j])
            {
                tm = scores[j];
                index = j;
            }
        }
        nms_flag[index]=true;
        nms_index[i]=index;
    }

    for(int i = 0; i<RPN_PRE_NMS_SIZE; i++) nms_flag[i]=true;


    for (int _i = 0; _i < RPN_PRE_NMS_SIZE; _i++) {
        if (nms_flag[_i] == false) continue;
        int i = nms_index[_i];
        fm_t ix1 = bboxes[i][0];
        fm_t iy1 = bboxes[i][1];
        fm_t ix2 = bboxes[i][2];
        fm_t iy2 = bboxes[i][3];
        fm_t iarea = area[i];

        for (int64_t _j = _i + 1; _j < RPN_PRE_NMS_SIZE; _j++) {
            if (nms_flag[_j] == false) continue;
            int j = nms_index[_j];
            
            fm_t xx1 = (ix1> bboxes[j][0])?ix1:bboxes[j][0];
            fm_t yy1 = (iy1> bboxes[j][1])?iy1:bboxes[j][1];
            fm_t xx2 = (ix2< bboxes[j][2])?ix2:bboxes[j][2];
            fm_t yy2 = (iy2< bboxes[j][3])?iy2:bboxes[j][3]; 
            fm_t w = (xx2 - xx1 );//(((fm_t)0)> (xx2 - xx1 ))?(fm_t)0:(xx2 - xx1);
            fm_t h = (yy2 - yy1);//(((fm_t)0)> (yy2 - yy1) )?(fm_t)0:(yy2 - yy1);
            fm_t inter = w * h;
            fm_t ovr = inter / (iarea + area[j] - inter);
            if (ovr > IOU_THRESHOLD) nms_flag[_j] = false;
        }
    }

    int ind = 0;
    for(int i = 0; i<4720; i++){
        if(nms_flag[i]==false)    continue;
        nms_index[ind]=nms_index[i];
        ind++;
    }
    int numSize = ind;

    for(int i =0; i<1000; i++){
        for(int j = 0; j<5; j++){
            switch(j){
                case 0:
                    dets[i][j]=bboxes[nms_index[i]][0]-(((int)(nms_index[i]/1000))*1281);
                    break;
                case 1:
                    dets[i][j]=bboxes[nms_index[i]][1]-(((int)(nms_index[i]/1000))*1281);
                    break;
                case 2:
                    dets[i][j]=bboxes[nms_index[i]][2]-(((int)(nms_index[i]/1000))*1281);
                    break;
                case 3:
                    dets[i][j]=bboxes[nms_index[i]][3]-(((int)(nms_index[i]/1000))*1281);
                    break;
                case 4:
                    dets[i][j]=scores[nms_index[i]];

            }
        }
    }
}
