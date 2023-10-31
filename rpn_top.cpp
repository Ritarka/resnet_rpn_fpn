#include "hls_stream.h"
#include "qdtrack.h"
#include "rpn_util.h"
#include "math.h"

fm_t rpn_anchor0_cls_fm[RPN_CLS_OUT_CH*RPN_INPUT0_IN_FM_HEIGHT*RPN_INPUT0_IN_FM_WIDTH];
fm_t rpn_anchor1_cls_fm[RPN_CLS_OUT_CH*RPN_INPUT1_IN_FM_HEIGHT*RPN_INPUT1_IN_FM_WIDTH];
fm_t rpn_anchor2_cls_fm[RPN_CLS_OUT_CH*RPN_INPUT2_IN_FM_HEIGHT*RPN_INPUT2_IN_FM_WIDTH];

fm_t rpn_anchor0_reg_fm[RPN_REG_OUT_CH*RPN_INPUT0_IN_FM_HEIGHT*RPN_INPUT0_IN_FM_WIDTH/4][4];
fm_t rpn_anchor1_reg_fm[RPN_REG_OUT_CH*RPN_INPUT1_IN_FM_HEIGHT*RPN_INPUT1_IN_FM_WIDTH/4][4];
fm_t rpn_anchor2_reg_fm[RPN_REG_OUT_CH*RPN_INPUT2_IN_FM_HEIGHT*RPN_INPUT2_IN_FM_WIDTH/4][4];

int rpn_topk_index0[RPN_PRE_NMS_SIZE0];
int rpn_topk_index1[RPN_PRE_NMS_SIZE1];
int rpn_topk_index2[RPN_PRE_NMS_SIZE2];

// Feature Map Buffers 
fm_t rpn_in_fm_buf[RPN_IN_BUF_CH][RPN_IN_BUF_ROWS + 5][RPN_IN_BUF_COLS + 5];
fm_t rpn_out_fm_buf[RPN_OUT_BUF_CH][RPN_OUT_BUF_ROWS][RPN_OUT_BUF_COLS];
fm_t rpn_partial_out_fm_buf[RPN_OUT_BUF_CH][RPN_OUT_BUF_ROWS][RPN_OUT_BUF_COLS];
fm_t rpn_ds_fm_buf[RPN_OUT_BUF_CH][RPN_OUT_BUF_ROWS][RPN_OUT_BUF_COLS];

// TODO: Replace these by pointer-based access to DRAM
fm_t rpn_layer_out_fm[2048][184][320];
fm_t rpn_layer_in_fm[2048][184][320];

// Convolution Weight Buffers
wt_t rpn_weight_buf_1x1[RPN_IN_BUF_CH][1][1];
wt_t rpn_weight_buf_3x3[RPN_IN_BUF_CH][3][3];
wt_t rpn_weight_buf_7x7[RPN_IN_BUF_CH][7][7];

// Conv bias
wt_t rpn_param_buf[RPN_OUT_BUF_CH];


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
)
{
    // std::cout << "Begin processing RPN CONV 0..." << std::endl;
    

    for(int c = 0; c < RPN_CONV_IN_CH; c++)
    {
        for(int h = 0; h < RPN_INPUT0_IN_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RPN_INPUT0_IN_FM_WIDTH; w++)
            {
                rpn_layer_in_fm[c][h][w] = rpn_input0_fm[c][h][w];
                // std::cout<<rpn_layer_in_fm[c][h][w]<<" ";
            }
            // std::cout<<std::endl;
        }
    }


    rpn_3x3_conv <RPN_CONV_IN_CH,  RPN_INPUT0_IN_FM_HEIGHT, RPN_INPUT0_IN_FM_WIDTH,
                    RPN_CONV_OUT_CH, RPN_INPUT0_IN_FM_HEIGHT, RPN_INPUT0_IN_FM_WIDTH,
                    RPN_CONV_STRIDE, RPN_LAST_LAYER_DISABLE, 0>
                    (rpn_conv_weight, rpn_conv_bias, true);
    

    for(int c = 0; c < RPN_CONV_IN_CH; c++)
    {
        for(int h = 0; h < RPN_INPUT0_IN_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RPN_INPUT0_IN_FM_WIDTH; w++)
            {
                rpn_output0_fm[c][h][w] = rpn_layer_out_fm[c][h][w];
                rpn_layer_in_fm[c][h][w]=0;
                rpn_layer_out_fm[c][h][w]=0;
            }
        }
    }


    // std::cout << "Begin processing RPN CONV 1..." << std::endl;

    for(int c = 0; c < RPN_CONV_IN_CH; c++)
    {
        for(int h = 0; h < RPN_INPUT1_IN_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RPN_INPUT1_IN_FM_WIDTH; w++)
            {
                rpn_layer_in_fm[c][h][w] = rpn_input1_fm[c][h][w];
            }
        }
    }


    rpn_3x3_conv <RPN_CONV_IN_CH,  RPN_INPUT1_IN_FM_HEIGHT, RPN_INPUT1_IN_FM_WIDTH,
                    RPN_CONV_OUT_CH, RPN_INPUT1_IN_FM_HEIGHT, RPN_INPUT1_IN_FM_WIDTH,
                    RPN_CONV_STRIDE, RPN_LAST_LAYER_DISABLE,0>
                    (rpn_conv_weight, rpn_conv_bias, true);
    

    for(int c = 0; c < RPN_CONV_IN_CH; c++)
    {
        for(int h = 0; h < RPN_INPUT1_IN_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RPN_INPUT1_IN_FM_WIDTH; w++)
            {
                rpn_output1_fm[c][h][w] = rpn_layer_out_fm[c][h][w];
                rpn_layer_in_fm[c][h][w]=0;
                rpn_layer_out_fm[c][h][w]=0;
            }
        }
    }



    // std::cout << "Begin processing RPN CONV 2..." << std::endl;


    for(int c = 0; c < RPN_CONV_IN_CH; c++)
    {
        for(int h = 0; h < RPN_INPUT2_IN_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RPN_INPUT2_IN_FM_WIDTH; w++)
            {
                rpn_layer_in_fm[c][h][w] = rpn_input2_fm[c][h][w];
            }
        }
    }


    rpn_3x3_conv <RPN_CONV_IN_CH,  RPN_INPUT2_IN_FM_HEIGHT, RPN_INPUT2_IN_FM_WIDTH,
                    RPN_CONV_OUT_CH, RPN_INPUT2_IN_FM_HEIGHT, RPN_INPUT2_IN_FM_WIDTH,
                    RPN_CONV_STRIDE, RPN_LAST_LAYER_DISABLE,0>
                    (rpn_conv_weight, rpn_conv_bias, true);
    

    for(int c = 0; c < RPN_CONV_IN_CH; c++)
    {
        for(int h = 0; h < RPN_INPUT2_IN_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RPN_INPUT2_IN_FM_WIDTH; w++)
            {
                rpn_output2_fm[c][h][w] = rpn_layer_out_fm[c][h][w];
                rpn_layer_in_fm[c][h][w]=0;
                rpn_layer_out_fm[c][h][w]=0;
            }
        }
    }


    //----------------------------------------------------------------------
    // RPN CLS Conv
    //----------------------------------------------------------------------



    // std::cout << "Begin processing RPN CLS 0..." << std::endl;



    for(int c = 0; c < RPN_CONV_IN_CH; c++)
    {
        for(int h = 0; h < RPN_INPUT0_IN_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RPN_INPUT0_IN_FM_WIDTH; w++)
            {
                rpn_layer_in_fm[c][h][w] = rpn_output0_fm[c][h][w];
                // cout<<rpn_layer_in_fm[c][h][w]<<" ";
            }
            // cout<<endl;
        }
    }


    rpn_1x1_conv <RPN_CLS_IN_CH,  RPN_INPUT0_IN_FM_HEIGHT, RPN_INPUT0_IN_FM_WIDTH,
                    RPN_CLS_OUT_CH, RPN_INPUT0_IN_FM_HEIGHT, RPN_INPUT0_IN_FM_WIDTH,
                    RPN_CLS_STRIDE, RPN_LAST_LAYER_DISABLE,0,0,1,0>
                    (rpn_cls_weight, rpn_cls_bias, false);
    

    for(int h = 0; h < RPN_INPUT0_IN_FM_HEIGHT; h++)
    {
        for(int w = 0; w < RPN_INPUT0_IN_FM_WIDTH; w++)
        {
            for(int c = 0; c < RPN_CLS_OUT_CH; c++)
            {
                rpn_output0_cls_fm[c][h][w] = rpn_layer_out_fm[c][h][w];
                rpn_anchor0_cls_fm[h*(RPN_INPUT0_IN_FM_WIDTH*RPN_CLS_OUT_CH) + w*RPN_CLS_OUT_CH+ c] = 1/(1+(fm_t)exp((float)(-1*rpn_layer_out_fm[c][h][w])));
                rpn_layer_in_fm[c][h][w]=0;
                rpn_layer_out_fm[c][h][w]=0;
            }
            // cout<<endl;
        }
    }


    // std::cout << "Begin processing RPN CLS 1..." << std::endl;



    for(int c = 0; c < RPN_CONV_IN_CH; c++)
    {
        for(int h = 0; h < RPN_INPUT1_IN_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RPN_INPUT1_IN_FM_WIDTH; w++)
            {
                rpn_layer_in_fm[c][h][w] = rpn_output1_fm[c][h][w];
            }
        }
    }

    rpn_1x1_conv <RPN_CLS_IN_CH,  RPN_INPUT1_IN_FM_HEIGHT, RPN_INPUT1_IN_FM_WIDTH,
                    RPN_CLS_OUT_CH, RPN_INPUT1_IN_FM_HEIGHT, RPN_INPUT1_IN_FM_WIDTH,
                    RPN_CLS_STRIDE, RPN_LAST_LAYER_DISABLE,0,0,1,0>
                    (rpn_cls_weight, rpn_cls_bias, false);
    

    for(int h = 0; h < RPN_INPUT1_IN_FM_HEIGHT; h++)
    {
        for(int w = 0; w < RPN_INPUT1_IN_FM_WIDTH; w++)
        {
            for(int c = 0; c < RPN_CLS_OUT_CH; c++)
            {
                rpn_output1_cls_fm[c][h][w] = rpn_layer_out_fm[c][h][w];
                rpn_anchor1_cls_fm[h*(RPN_INPUT1_IN_FM_WIDTH*RPN_CLS_OUT_CH) + w*RPN_CLS_OUT_CH+ c] = 1/(1+(fm_t)exp((float)(-1*rpn_layer_out_fm[c][h][w])));                
                rpn_layer_in_fm[c][h][w]=0;
                rpn_layer_out_fm[c][h][w]=0;
            }
        }
    }

    // std::cout << "Begin processing RPN CLS 2..." << std::endl;

    for(int c = 0; c < RPN_CONV_IN_CH; c++)
    {
        for(int h = 0; h < RPN_INPUT2_IN_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RPN_INPUT2_IN_FM_WIDTH; w++)
            {
                rpn_layer_in_fm[c][h][w] = rpn_output2_fm[c][h][w];
            }
        }
    }

    rpn_1x1_conv <RPN_CLS_IN_CH,  RPN_INPUT2_IN_FM_HEIGHT, RPN_INPUT2_IN_FM_WIDTH,
                    RPN_CLS_OUT_CH, RPN_INPUT2_IN_FM_HEIGHT, RPN_INPUT2_IN_FM_WIDTH,
                    RPN_CLS_STRIDE, RPN_LAST_LAYER_DISABLE,0,0,1,0>
                    (rpn_cls_weight, rpn_cls_bias, false);
    

    for(int h = 0; h < RPN_INPUT2_IN_FM_HEIGHT; h++)
    {
        for(int w = 0; w < RPN_INPUT2_IN_FM_WIDTH; w++)
        {
            for(int c = 0; c < RPN_CLS_OUT_CH; c++)
            {
                rpn_output2_cls_fm[c][h][w] = rpn_layer_out_fm[c][h][w];
                rpn_anchor2_cls_fm[h*(RPN_INPUT2_IN_FM_WIDTH*RPN_CLS_OUT_CH) + w*RPN_CLS_OUT_CH+ c] = 1/(1+(fm_t)exp((float)(-1*rpn_layer_out_fm[c][h][w])));
                rpn_layer_in_fm[c][h][w]=0;
                rpn_layer_out_fm[c][h][w]=0;
            }
        }
    }


    //----------------------------------------------------------------------
    // RPN REG
    //----------------------------------------------------------------------



    // std::cout << "Begin processing RPN REG 0..." << std::endl;



    for(int c = 0; c < RPN_CONV_IN_CH; c++)
    {
        for(int h = 0; h < RPN_INPUT0_IN_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RPN_INPUT0_IN_FM_WIDTH; w++)
            {
                rpn_layer_in_fm[c][h][w] = rpn_output0_fm[c][h][w];
            }
        }
    }


    rpn_1x1_conv <RPN_REG_IN_CH,  RPN_INPUT0_IN_FM_HEIGHT, RPN_INPUT0_IN_FM_WIDTH,
                    RPN_REG_OUT_CH, RPN_INPUT0_IN_FM_HEIGHT, RPN_INPUT0_IN_FM_WIDTH,
                    RPN_REG_STRIDE, RPN_LAST_LAYER_DISABLE,0,0,1,0>
                    (rpn_reg_weight, rpn_reg_bias, false);
    

    for(int h = 0; h < RPN_INPUT0_IN_FM_HEIGHT; h++)
    {
        for(int w = 0; w < RPN_INPUT0_IN_FM_WIDTH; w++)
        {
            for(int c = 0; c < RPN_REG_OUT_CH; c++)
            {
                rpn_output0_reg_fm[c][h][w] = rpn_layer_out_fm[c][h][w];
                rpn_anchor0_reg_fm[(h*(RPN_INPUT0_IN_FM_WIDTH*RPN_REG_OUT_CH)+w*RPN_REG_OUT_CH+c)/4][(h*(RPN_INPUT0_IN_FM_WIDTH*RPN_REG_OUT_CH)+w*RPN_REG_OUT_CH+c)%4]= rpn_layer_out_fm[c][h][w];
                rpn_layer_in_fm[c][h][w]=0;
                rpn_layer_out_fm[c][h][w]=0;
            }
        }
    }


    // std::cout << "Begin processing RPN REG 1..." << std::endl;



    for(int c = 0; c < RPN_CONV_IN_CH; c++)
    {
        for(int h = 0; h < RPN_INPUT1_IN_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RPN_INPUT1_IN_FM_WIDTH; w++)
            {
                rpn_layer_in_fm[c][h][w] = rpn_output1_fm[c][h][w];
            }
        }
    }


    rpn_1x1_conv <RPN_REG_IN_CH,  RPN_INPUT1_IN_FM_HEIGHT, RPN_INPUT1_IN_FM_WIDTH,
                    RPN_REG_OUT_CH, RPN_INPUT1_IN_FM_HEIGHT, RPN_INPUT1_IN_FM_WIDTH,
                    RPN_REG_STRIDE, RPN_LAST_LAYER_DISABLE,0,0,1,0>
                    (rpn_reg_weight, rpn_reg_bias, false);
    

    for(int h = 0; h < RPN_INPUT1_IN_FM_HEIGHT; h++)
    {
        for(int w = 0; w < RPN_INPUT1_IN_FM_WIDTH; w++)
        {
            for(int c = 0; c < RPN_REG_OUT_CH; c++)
            {
                rpn_output1_reg_fm[c][h][w] = rpn_layer_out_fm[c][h][w];
                rpn_anchor1_reg_fm[(h*(RPN_INPUT1_IN_FM_WIDTH*RPN_REG_OUT_CH)+w*RPN_REG_OUT_CH+c)/4][(h*(RPN_INPUT1_IN_FM_WIDTH*RPN_REG_OUT_CH)+w*RPN_REG_OUT_CH+c)%4]= rpn_layer_out_fm[c][h][w];
                rpn_layer_in_fm[c][h][w]=0;
                rpn_layer_out_fm[c][h][w]=0;
            }
        }
    }


    // std::cout << "Begin processing RPN REG 2..." << std::endl;



    for(int c = 0; c < RPN_CONV_IN_CH; c++)
    {
        for(int h = 0; h < RPN_INPUT2_IN_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RPN_INPUT2_IN_FM_WIDTH; w++)
            {
                rpn_layer_in_fm[c][h][w] = rpn_output2_fm[c][h][w];
            }
        }
    }


    rpn_1x1_conv <RPN_REG_IN_CH,  RPN_INPUT2_IN_FM_HEIGHT, RPN_INPUT2_IN_FM_WIDTH,
                    RPN_REG_OUT_CH, RPN_INPUT2_IN_FM_HEIGHT, RPN_INPUT2_IN_FM_WIDTH,
                    RPN_REG_STRIDE, RPN_LAST_LAYER_DISABLE,0,0,1,0>
                    (rpn_reg_weight, rpn_reg_bias, false);
    

    for(int h = 0; h < RPN_INPUT2_IN_FM_HEIGHT; h++)
    {
        for(int w = 0; w < RPN_INPUT2_IN_FM_WIDTH; w++)
        {
            for(int c = 0; c < RPN_REG_OUT_CH; c++)
            {
                rpn_output2_reg_fm[c][h][w] = rpn_layer_out_fm[c][h][w];
                rpn_anchor2_reg_fm[(h*(RPN_INPUT2_IN_FM_WIDTH*RPN_REG_OUT_CH)+w*RPN_REG_OUT_CH+c)/4][(h*(RPN_INPUT2_IN_FM_WIDTH*RPN_REG_OUT_CH)+w*RPN_REG_OUT_CH+c)%4]= rpn_layer_out_fm[c][h][w];
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

   // std::cout << "Begin processing RPN Anchor Gen 0..." << std::endl;


    bool flag0[RPN_CLS_OUT_CH*RPN_INPUT0_IN_FM_HEIGHT*RPN_INPUT0_IN_FM_WIDTH]={false};
    for(int i = 0; i<RPN_PRE_NMS_SIZE0; i++)
    {
        fm_t maximum_score = 0.0;
        int index = -1;
        for(int j = 0; j<RPN_CLS_OUT_CH*RPN_INPUT0_IN_FM_HEIGHT*RPN_INPUT0_IN_FM_WIDTH; j++)
        {
            if(flag0[j] == true)    continue;
            if(rpn_anchor0_cls_fm[j]>maximum_score)
            {
                maximum_score = rpn_anchor0_cls_fm[j];
                index = j;
            }
            
        }
            rpn_topk_index0[i]=index;
            flag0[index]=true;
    }


   // std::cout << "Begin processing RPN Anchor Gen 1..." << std::endl;


    bool flag1[RPN_CLS_OUT_CH*RPN_INPUT1_IN_FM_HEIGHT*RPN_INPUT1_IN_FM_WIDTH]={false};
    for(int i = 0; i<RPN_PRE_NMS_SIZE1; i++)
    {
        fm_t maximum_score = 0.0;
        int index = -1;
        for(int j = 0; j<RPN_CLS_OUT_CH*RPN_INPUT1_IN_FM_HEIGHT*RPN_INPUT1_IN_FM_WIDTH; j++)
        {
            if(flag1[j] == true)    continue;
            if(rpn_anchor1_cls_fm[j]>maximum_score)
            {
                maximum_score = rpn_anchor1_cls_fm[j];
                index = j;
            }
        }
            rpn_topk_index1[i]=index;
            flag1[index]=true;
    }


   // std::cout << "Begin processing RPN Anchor Gen 2..." << std::endl;



    bool flag2[RPN_CLS_OUT_CH*RPN_INPUT2_IN_FM_HEIGHT*RPN_INPUT2_IN_FM_WIDTH]={false};
    for(int i = 0; i<RPN_PRE_NMS_SIZE2; i++)
    {
        fm_t maximum_score = 0.0;
        int index = -1;
        for(int j = 0; j<RPN_CLS_OUT_CH*RPN_INPUT2_IN_FM_HEIGHT*RPN_INPUT2_IN_FM_WIDTH; j++)
        {
            if(flag2[j] == true)    continue;
            if(rpn_anchor2_cls_fm[j]>maximum_score)
            {
                maximum_score = rpn_anchor2_cls_fm[j];
                index = j;
            }
        }

            rpn_topk_index2[i]=index;
            flag2[index]=true;
    }
}
