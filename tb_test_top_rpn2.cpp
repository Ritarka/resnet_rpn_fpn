#include <iostream>
#include <fstream>
#include <cmath>
#include "hls_stream.h"
#include "qdtrack_rpn2.h"

using namespace std;

// float  input_image[RESNET_LAYER0_CONV1_IN_CH][RESNET_LAYER0_IN_FM_HEIGHT][RESNET_LAYER0_IN_FM_WIDTH];

//--------------------------------------------------------------------------
// RPN
//--------------------------------------------------------------------------
int rpn_topk_index0_2[RPN_PRE_NMS_SIZE0];
int rpn_topk_index1_2[RPN_PRE_NMS_SIZE1];
int rpn_topk_index2_2[RPN_PRE_NMS_SIZE2];

fm_t rpn_anchor0_reg_fm_2[RPN_REG_OUT_CH*RPN_INPUT0_IN_FM_HEIGHT*RPN_INPUT0_IN_FM_WIDTH/4][4];
fm_t rpn_anchor1_reg_fm_2[RPN_REG_OUT_CH*RPN_INPUT0_IN_FM_HEIGHT*RPN_INPUT0_IN_FM_WIDTH/4][4];
fm_t rpn_anchor2_reg_fm_2[RPN_REG_OUT_CH*RPN_INPUT0_IN_FM_HEIGHT*RPN_INPUT0_IN_FM_WIDTH/4][4];

fm_t rpn_anchor0_cls_fm_2[RPN_CLS_OUT_CH*RPN_INPUT0_IN_FM_HEIGHT*RPN_INPUT0_IN_FM_WIDTH];
fm_t rpn_anchor1_cls_fm_2[RPN_CLS_OUT_CH*RPN_INPUT1_IN_FM_HEIGHT*RPN_INPUT1_IN_FM_WIDTH];
fm_t rpn_anchor2_cls_fm_2[RPN_CLS_OUT_CH*RPN_INPUT2_IN_FM_HEIGHT*RPN_INPUT2_IN_FM_WIDTH];

// fm_t rpn_input0_fm[RPN_CONV_IN_CH][RPN_INPUT0_IN_FM_HEIGHT][RPN_INPUT0_IN_FM_WIDTH];
// fm_t rpn_input1_fm[RPN_CONV_IN_CH][RPN_INPUT1_IN_FM_HEIGHT][RPN_INPUT1_IN_FM_WIDTH];
// fm_t rpn_input2_fm[RPN_CONV_IN_CH][RPN_INPUT2_IN_FM_HEIGHT][RPN_INPUT2_IN_FM_WIDTH];
fm_t rpn_input3_fm[RPN_CONV_IN_CH][RPN_INPUT3_IN_FM_HEIGHT][RPN_INPUT3_IN_FM_WIDTH];
fm_t rpn_input4_fm[RPN_CONV_IN_CH][RPN_INPUT4_IN_FM_HEIGHT][RPN_INPUT4_IN_FM_WIDTH];


// float rpn_input0[RPN_CONV_IN_CH][RPN_INPUT0_IN_FM_HEIGHT][RPN_INPUT0_IN_FM_WIDTH];
// float rpn_input1[RPN_CONV_IN_CH][RPN_INPUT1_IN_FM_HEIGHT][RPN_INPUT1_IN_FM_WIDTH];
// float rpn_input2[RPN_CONV_IN_CH][RPN_INPUT2_IN_FM_HEIGHT][RPN_INPUT2_IN_FM_WIDTH];
float rpn_input3[RPN_CONV_IN_CH][RPN_INPUT3_IN_FM_HEIGHT][RPN_INPUT3_IN_FM_WIDTH];
float rpn_input4[RPN_CONV_IN_CH][RPN_INPUT4_IN_FM_HEIGHT][RPN_INPUT4_IN_FM_WIDTH];

//Weights and Bias for convolutions
wt_t rpn_conv_weight[RPN_CONV_OUT_CH][RPN_CONV_IN_CH][3][3];
wt_t rpn_conv_bias[RPN_CONV_OUT_CH];
wt_t rpn_cls_weight[RPN_CLS_OUT_CH][RPN_CLS_IN_CH][1][1];
wt_t rpn_cls_bias[RPN_CLS_OUT_CH];
wt_t rpn_reg_weight[RPN_REG_OUT_CH][RPN_REG_IN_CH][1][1];
wt_t rpn_reg_bias[RPN_REG_OUT_CH];

float fl_rpn_conv_weight[RPN_CONV_OUT_CH][RPN_CONV_IN_CH][3][3];
float fl_rpn_conv_bias[RPN_CONV_OUT_CH];
float fl_rpn_cls_weight[RPN_CLS_OUT_CH][RPN_CLS_IN_CH][1][1];
float fl_rpn_cls_bias[RPN_CLS_OUT_CH];
float fl_rpn_reg_weight[RPN_REG_OUT_CH][RPN_REG_IN_CH][1][1];
float fl_rpn_reg_bias[RPN_REG_OUT_CH];


// fm_t rpn_output0_cls_fm[RPN_CLS_OUT_CH][RPN_INPUT0_IN_FM_HEIGHT][RPN_INPUT0_IN_FM_WIDTH];
// fm_t rpn_output1_cls_fm[RPN_CLS_OUT_CH][RPN_INPUT1_IN_FM_HEIGHT][RPN_INPUT1_IN_FM_WIDTH];
// fm_t rpn_output2_cls_fm[RPN_CLS_OUT_CH][RPN_INPUT2_IN_FM_HEIGHT][RPN_INPUT2_IN_FM_WIDTH];
fm_t rpn_output3_cls_fm[RPN_CLS_OUT_CH][RPN_INPUT3_IN_FM_HEIGHT][RPN_INPUT3_IN_FM_WIDTH];
fm_t rpn_output4_cls_fm[RPN_CLS_OUT_CH][RPN_INPUT4_IN_FM_HEIGHT][RPN_INPUT4_IN_FM_WIDTH];

// float fl_rpn_output0_cls_fm[RPN_CLS_OUT_CH][RPN_INPUT0_IN_FM_HEIGHT][RPN_INPUT0_IN_FM_WIDTH];
// float fl_rpn_output1_cls_fm[RPN_CLS_OUT_CH][RPN_INPUT1_IN_FM_HEIGHT][RPN_INPUT1_IN_FM_WIDTH];
// float fl_rpn_output2_cls_fm[RPN_CLS_OUT_CH][RPN_INPUT2_IN_FM_HEIGHT][RPN_INPUT2_IN_FM_WIDTH];
float fl_rpn_output3_cls_fm[RPN_CLS_OUT_CH][RPN_INPUT3_IN_FM_HEIGHT][RPN_INPUT3_IN_FM_WIDTH];
float fl_rpn_output4_cls_fm[RPN_CLS_OUT_CH][RPN_INPUT4_IN_FM_HEIGHT][RPN_INPUT4_IN_FM_WIDTH];



// fm_t rpn_output0_reg_fm[RPN_REG_OUT_CH][RPN_INPUT0_IN_FM_HEIGHT][RPN_INPUT0_IN_FM_WIDTH];
// fm_t rpn_output1_reg_fm[RPN_REG_OUT_CH][RPN_INPUT1_IN_FM_HEIGHT][RPN_INPUT1_IN_FM_WIDTH];
// fm_t rpn_output2_reg_fm[RPN_REG_OUT_CH][RPN_INPUT2_IN_FM_HEIGHT][RPN_INPUT2_IN_FM_WIDTH];
fm_t rpn_output3_reg_fm[RPN_REG_OUT_CH][RPN_INPUT3_IN_FM_HEIGHT][RPN_INPUT3_IN_FM_WIDTH];
fm_t rpn_output4_reg_fm[RPN_REG_OUT_CH][RPN_INPUT4_IN_FM_HEIGHT][RPN_INPUT4_IN_FM_WIDTH];

// float fl_rpn_output0_reg_fm[RPN_REG_OUT_CH][RPN_INPUT0_IN_FM_HEIGHT][RPN_INPUT0_IN_FM_WIDTH];
// float fl_rpn_output1_reg_fm[RPN_REG_OUT_CH][RPN_INPUT1_IN_FM_HEIGHT][RPN_INPUT1_IN_FM_WIDTH];
// float fl_rpn_output2_reg_fm[RPN_REG_OUT_CH][RPN_INPUT2_IN_FM_HEIGHT][RPN_INPUT2_IN_FM_WIDTH];
float fl_rpn_output3_reg_fm[RPN_REG_OUT_CH][RPN_INPUT3_IN_FM_HEIGHT][RPN_INPUT3_IN_FM_WIDTH];
float fl_rpn_output4_reg_fm[RPN_REG_OUT_CH][RPN_INPUT4_IN_FM_HEIGHT][RPN_INPUT4_IN_FM_WIDTH];


// float fl_rpn_output0_fm[RPN_CONV_IN_CH][RPN_INPUT0_IN_FM_HEIGHT][RPN_INPUT0_IN_FM_WIDTH];
// float fl_rpn_output1_fm[RPN_CONV_IN_CH][RPN_INPUT1_IN_FM_HEIGHT][RPN_INPUT1_IN_FM_WIDTH];
// float fl_rpn_output2_fm[RPN_CONV_IN_CH][RPN_INPUT2_IN_FM_HEIGHT][RPN_INPUT2_IN_FM_WIDTH];
float fl_rpn_output3_fm[RPN_CONV_IN_CH][RPN_INPUT3_IN_FM_HEIGHT][RPN_INPUT3_IN_FM_WIDTH];
float fl_rpn_output4_fm[RPN_CONV_IN_CH][RPN_INPUT4_IN_FM_HEIGHT][RPN_INPUT4_IN_FM_WIDTH];



// fm_t rpn_output0_fm[RPN_CONV_IN_CH][RPN_INPUT0_IN_FM_HEIGHT][RPN_INPUT0_IN_FM_WIDTH];
// fm_t rpn_output1_fm[RPN_CONV_IN_CH][RPN_INPUT1_IN_FM_HEIGHT][RPN_INPUT1_IN_FM_WIDTH];
// fm_t rpn_output2_fm[RPN_CONV_IN_CH][RPN_INPUT2_IN_FM_HEIGHT][RPN_INPUT2_IN_FM_WIDTH];
fm_t rpn_output3_fm[RPN_CONV_IN_CH][RPN_INPUT3_IN_FM_HEIGHT][RPN_INPUT3_IN_FM_WIDTH];
fm_t rpn_output4_fm[RPN_CONV_IN_CH][RPN_INPUT4_IN_FM_HEIGHT][RPN_INPUT4_IN_FM_WIDTH];


float fl_anchor_box0[RPN_ANCHORS0_IN_FM][4];
float fl_anchor_box1[RPN_ANCHORS1_IN_FM][4];
float fl_anchor_box2[RPN_ANCHORS2_IN_FM][4];
float fl_anchor_box3[RPN_ANCHORS3_IN_FM][4];
float fl_anchor_box4[RPN_ANCHORS4_IN_FM][4];



wt_t anchor_box0[RPN_ANCHORS0_IN_FM][4];
wt_t anchor_box1[RPN_ANCHORS1_IN_FM][4];
wt_t anchor_box2[RPN_ANCHORS2_IN_FM][4];
wt_t anchor_box3[RPN_ANCHORS3_IN_FM][4];
wt_t anchor_box4[RPN_ANCHORS4_IN_FM][4];


float fl_bboxes[RPN_PRE_NMS_SIZE][4];
fm_t bboxes[RPN_PRE_NMS_SIZE][4];

fm_t dets[1000][5];
float fl_dets[1000][5];

void rpn_load_weights ()
{
    ifstream ifs_rpn_conv_weight_param("/usr/scratch/rsamanta9/weights/module.rpn_head.rpn_conv.weight.bin", ios::in | ios::binary);
    ifs_rpn_conv_weight_param.read((char*)(***fl_rpn_conv_weight), RPN_CONV_OUT_CH*RPN_CONV_IN_CH*3*3*sizeof(float));
    ifs_rpn_conv_weight_param.close();

    ifstream ifs_rpn_conv_bias_param("/usr/scratch/rsamanta9/weights/module.rpn_head.rpn_conv.bias.bin", ios::in | ios::binary);
    ifs_rpn_conv_bias_param.read((char*)(fl_rpn_conv_bias), RPN_CONV_OUT_CH*sizeof(float));
    ifs_rpn_conv_bias_param.close();

    ifstream ifs_rpn_cls_weight_param("/usr/scratch/rsamanta9/weights/module.rpn_head.rpn_cls.weight.bin", ios::in | ios::binary);
    ifs_rpn_cls_weight_param.read((char*)(***fl_rpn_cls_weight), RPN_CLS_OUT_CH*RPN_CLS_IN_CH*1*1*sizeof(float));
    ifs_rpn_cls_weight_param.close();

    ifstream ifs_rpn_cls_bias_param("/usr/scratch/rsamanta9/weights/module.rpn_head.rpn_cls.bias.bin", ios::in | ios::binary);
    ifs_rpn_cls_bias_param.read((char*)(fl_rpn_cls_bias), RPN_CLS_OUT_CH*sizeof(float));
    ifs_rpn_cls_bias_param.close();

    ifstream ifs_rpn_reg_weight_param("/usr/scratch/rsamanta9/weights/module.rpn_head.rpn_reg.weight.bin", ios::in | ios::binary);
    ifs_rpn_reg_weight_param.read((char*)(***fl_rpn_reg_weight), RPN_REG_OUT_CH*RPN_REG_IN_CH*1*1*sizeof(float));
    ifs_rpn_reg_weight_param.close();

    ifstream ifs_rpn_reg_bias_param("/usr/scratch/rsamanta9/weights/module.rpn_head.rpn_reg.bias.bin", ios::in | ios::binary);
    ifs_rpn_reg_bias_param.read((char*)(fl_rpn_reg_bias), RPN_REG_OUT_CH*sizeof(float));
    ifs_rpn_reg_bias_param.close();

    ifstream ifs_rpn_anchor0_param("/usr/scratch/rsamanta9/weights/anchor0.bin", ios::in | ios::binary);
    ifs_rpn_anchor0_param.read((char*)(fl_anchor_box0), RPN_ANCHORS0_IN_FM*4*sizeof(float));
    ifs_rpn_anchor0_param.close();

    ifstream ifs_rpn_anchor1_param("/usr/scratch/rsamanta9/weights/anchor1.bin", ios::in | ios::binary);
    ifs_rpn_anchor1_param.read((char*)(fl_anchor_box1), RPN_ANCHORS1_IN_FM*4*sizeof(float));
    ifs_rpn_anchor1_param.close();
    
    ifstream ifs_rpn_anchor2_param("/usr/scratch/rsamanta9/weights/anchor2.bin", ios::in | ios::binary);
    ifs_rpn_anchor2_param.read((char*)(fl_anchor_box2), RPN_ANCHORS2_IN_FM*4*sizeof(float));
    ifs_rpn_anchor2_param.close();

    ifstream ifs_rpn_anchor3_param("/usr/scratch/rsamanta9/weights/anchor3.bin", ios::in | ios::binary);
    ifs_rpn_anchor3_param.read((char*)(fl_anchor_box3), RPN_ANCHORS3_IN_FM*4*sizeof(float));
    ifs_rpn_anchor3_param.close();

    ifstream ifs_rpn_anchor4_param("/usr/scratch/rsamanta9/weights/anchor4.bin", ios::in | ios::binary);
    ifs_rpn_anchor4_param.read((char*)(fl_anchor_box4), RPN_ANCHORS4_IN_FM*4*sizeof(float));
    ifs_rpn_anchor4_param.close();

}

template<const int KERNEL_DEPTH, const int FILTER_SIZE>
void convert_1x1(float  in_weights[FILTER_SIZE][KERNEL_DEPTH], 
                  wt_t  out_weights[FILTER_SIZE][KERNEL_DEPTH])
{
    for(int f = 0; f < FILTER_SIZE; f++)
        for(int k = 0; k < KERNEL_DEPTH; k++)
            out_weights[f][k] = (wt_t) in_weights[f][k];
}

template<const int KERNEL_DEPTH, const int FILTER_SIZE>
void convert_3x3(float  in_weights[FILTER_SIZE][KERNEL_DEPTH][3][3], 
                  wt_t  out_weights[FILTER_SIZE][KERNEL_DEPTH][3][3])
{
    for(int f = 0; f < FILTER_SIZE; f++)
        for(int k = 0; k < KERNEL_DEPTH; k++)
            for(int m = 0; m < 3; m++)
                for(int n =0; n < 3; n++)
                    out_weights[f][k][m][n] = (wt_t) in_weights[f][k][m][n];
}

template<const int FILTER_SIZE>
void convert_bn(float in_params[3][FILTER_SIZE], wt_t out_params[3][FILTER_SIZE])
{
    for(int i = 0; i < 3; i++)
        for(int j = 0; j< FILTER_SIZE; j++)
            out_params[i][j] = (wt_t) in_params[i][j];
}


template<const int KERNEL_DEPTH, const int FILTER_SIZE>
void rpn_convert_1x1(float  in_weights[FILTER_SIZE][KERNEL_DEPTH][1][1], 
                  wt_t  out_weights[FILTER_SIZE][KERNEL_DEPTH][1][1])
{
    for(int f = 0; f < FILTER_SIZE; f++)
        for(int k = 0; k < KERNEL_DEPTH; k++)
            out_weights[f][k][0][0] = (wt_t) in_weights[f][k][0][0];
}

template<const int KERNEL_DEPTH, const int FILTER_SIZE>
void rpn_convert_3x3(float  in_weights[FILTER_SIZE][KERNEL_DEPTH][3][3], 
                  wt_t  out_weights[FILTER_SIZE][KERNEL_DEPTH][3][3])
{
    for(int f = 0; f < FILTER_SIZE; f++)
        for(int k = 0; k < KERNEL_DEPTH; k++)
            for(int m = 0; m < 3; m++)
                for(int n =0; n < 3; n++)
                    out_weights[f][k][m][n] = (wt_t) in_weights[f][k][m][n];
}

template<const int FILTER_SIZE>
void rpn_convert_bias(float in_params[FILTER_SIZE], wt_t out_params[FILTER_SIZE])
{   
    for(int j = 0; j< FILTER_SIZE; j++)
        out_params[j] = (wt_t) in_params[j];
}

void rpn_convert_weights_type()
{
    rpn_convert_3x3<RPN_CONV_IN_CH, RPN_CONV_OUT_CH>
               (fl_rpn_conv_weight, rpn_conv_weight);
    rpn_convert_1x1< RPN_CLS_IN_CH,RPN_CLS_OUT_CH>
               (fl_rpn_cls_weight, rpn_cls_weight);
    rpn_convert_1x1<RPN_REG_IN_CH, RPN_REG_OUT_CH>
               (fl_rpn_reg_weight, rpn_reg_weight);

    rpn_convert_bias<RPN_CONV_OUT_CH>(fl_rpn_conv_bias, rpn_conv_bias);
    rpn_convert_bias<RPN_CLS_OUT_CH>(fl_rpn_cls_bias, rpn_cls_bias);
    rpn_convert_bias<RPN_REG_OUT_CH>(fl_rpn_reg_bias, rpn_reg_bias);

    for(int a = 0; a<RPN_ANCHORS0_IN_FM; a++)
        for(int c = 0; c<4; c++)
            anchor_box0[a][c] = (wt_t) fl_anchor_box0[a][c];
    

    for(int a = 0; a<RPN_ANCHORS1_IN_FM; a++)
        for(int c = 0; c<4; c++)
            anchor_box1[a][c] = (wt_t) fl_anchor_box1[a][c];
    

    for(int a = 0; a<RPN_ANCHORS2_IN_FM; a++)
        for(int c = 0; c<4; c++)
            anchor_box2[a][c] = (wt_t) fl_anchor_box2[a][c];
    

    for(int a = 0; a<RPN_ANCHORS3_IN_FM; a++)
        for(int c = 0; c<4; c++)
            anchor_box3[a][c] = (wt_t) fl_anchor_box3[a][c];
    

    for(int a = 0; a<RPN_ANCHORS4_IN_FM; a++)
        for(int c = 0; c<4; c++)
            anchor_box4[a][c] = (wt_t) fl_anchor_box4[a][c];
    
}

void rpn_load_inputs()
{
    // //----------------------------------------------------------------------
    // // Read and convert RPN Input 0
    // //----------------------------------------------------------------------
    // ifstream ifs_input_img0("/usr/scratch/pchhatrapati3/hls/inputs/rpninput0.bin", ios::in | ios::binary);
    // ifs_input_img0.read((char*)(**rpn_input0), (RPN_CONV_IN_CH)*(RPN_INPUT0_IN_FM_HEIGHT)*(RPN_INPUT0_IN_FM_WIDTH)*sizeof(float));
    // ifs_input_img0.close();

    // for(int c = 0; c < RPN_CONV_IN_CH; c++)
    // {
    //     for(int h = 0; h < RPN_INPUT0_IN_FM_HEIGHT; h++)
    //     {
    //         for(int w = 0; w < RPN_INPUT0_IN_FM_WIDTH; w++)
    //         {
    //             rpn_input0_fm[c][h][w] = (fm_t) rpn_input0[c][h][w];
    //             // cout<<rpn_input0_fm[c][h][w]<<" ";
    //         }
    //         // cout<<endl;
    //     }

    // }

    // //----------------------------------------------------------------------
    // // Read and convert RPN Input 1
    // //----------------------------------------------------------------------
    // ifstream ifs_input_img1("/usr/scratch/pchhatrapati3/hls/inputs/rpninput1.bin", ios::in | ios::binary);
    // ifs_input_img1.read((char*)(**rpn_input1), (RPN_CONV_IN_CH)*(RPN_INPUT1_IN_FM_HEIGHT)*(RPN_INPUT1_IN_FM_WIDTH)*sizeof(float));
    // ifs_input_img1.close();

    // for(int c = 0; c < RPN_CONV_IN_CH; c++)
    // {
    //     for(int h = 0; h < RPN_INPUT1_IN_FM_HEIGHT; h++)
    //     {
    //         for(int w = 0; w < RPN_INPUT1_IN_FM_WIDTH; w++)
    //         {
    //             rpn_input1_fm[c][h][w] = (fm_t) rpn_input1[c][h][w];
                
    //         }
    //     }

    // }

    // //----------------------------------------------------------------------
    // // Read and convert RPN Input 2
    // //----------------------------------------------------------------------
    // ifstream ifs_input_img2("/usr/scratch/pchhatrapati3/hls/inputs/rpninput2.bin", ios::in | ios::binary);
    // ifs_input_img2.read((char*)(**rpn_input2), (RPN_CONV_IN_CH)*(RPN_INPUT2_IN_FM_HEIGHT)*(RPN_INPUT2_IN_FM_WIDTH)*sizeof(float));
    // ifs_input_img2.close();

    // for(int c = 0; c < RPN_CONV_IN_CH; c++)
    // {
    //     for(int h = 0; h < RPN_INPUT2_IN_FM_HEIGHT; h++)
    //     {
    //         for(int w = 0; w < RPN_INPUT2_IN_FM_WIDTH; w++)
    //         {
    //             rpn_input2_fm[c][h][w] = (fm_t) rpn_input2[c][h][w];
    //         }
    //     }
    // }


    //----------------------------------------------------------------------
    // Read and convert RPN Input 3
    //----------------------------------------------------------------------

    ifstream ifs_input_img3("/usr/scratch/pchhatrapati3/hls/inputs/rpninput3.bin", ios::in | ios::binary);
    ifs_input_img3.read((char*)(**rpn_input3), (RPN_CONV_IN_CH)*(RPN_INPUT3_IN_FM_HEIGHT)*(RPN_INPUT3_IN_FM_WIDTH)*sizeof(float));
    ifs_input_img3.close();

    for(int c = 0; c < RPN_CONV_IN_CH; c++)
    {
        for(int h = 0; h < RPN_INPUT3_IN_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RPN_INPUT3_IN_FM_WIDTH; w++)
            {
                rpn_input3_fm[c][h][w] = (fm_t) rpn_input3[c][h][w];
            }
        }
    }



    //----------------------------------------------------------------------
    // Read and convert RPN Input 4
    //----------------------------------------------------------------------

    ifstream ifs_input_img4("/usr/scratch/pchhatrapati3/hls/inputs/rpninput4.bin", ios::in | ios::binary);
    ifs_input_img4.read((char*)(**rpn_input4), (RPN_CONV_IN_CH)*(RPN_INPUT4_IN_FM_HEIGHT)*(RPN_INPUT4_IN_FM_WIDTH)*sizeof(float));
    ifs_input_img4.close();

    for(int c = 0; c < RPN_CONV_IN_CH; c++)
    {
        for(int h = 0; h < RPN_INPUT4_IN_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RPN_INPUT4_IN_FM_WIDTH; w++)
            {
                rpn_input4_fm[c][h][w] = (fm_t) rpn_input4[c][h][w];
            }
        }
    }

}

int main ()
{
    long double mse = 0.0;
    
    // RPN
    rpn_load_weights();
    rpn_convert_weights_type();
    rpn_load_inputs();
    
#ifdef TEST_COMPLETE_MODEL // {
    //----------------------------------------------------------------------
    // ResNet50 Top-level wrapper 
    //----------------------------------------------------------------------
    test_top(

    rpn_topk_index0_2,
    rpn_topk_index1_2,
    rpn_topk_index2_2,

    rpn_anchor0_reg_fm_2,
    rpn_anchor1_reg_fm_2,
    rpn_anchor2_reg_fm_2,

    rpn_anchor0_cls_fm_2,
    rpn_anchor1_cls_fm_2,
    rpn_anchor2_cls_fm_2,

    // rpn_input0_fm,
    // rpn_input1_fm,
    // rpn_input2_fm,
    rpn_input3_fm,
    rpn_input4_fm,

    //Weights and Bias for convolutions
    rpn_conv_weight,
    rpn_conv_bias,
    rpn_cls_weight,
    rpn_cls_bias,
    rpn_reg_weight,
    rpn_reg_bias,

    // rpn_output0_cls_fm,
    // rpn_output1_cls_fm,
    // rpn_output2_cls_fm,
    rpn_output3_cls_fm,
    rpn_output4_cls_fm,

    // rpn_output0_reg_fm,
    // rpn_output1_reg_fm,
    // rpn_output2_reg_fm,
    rpn_output3_reg_fm,
    rpn_output4_reg_fm,

    // rpn_output0_fm,
    // rpn_output1_fm,
    // rpn_output2_fm,
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
    
    // // RPN
    // //----------------------------------------------------------------------
    // // Check mse for RPN conv 0
    // //----------------------------------------------------------------------
    
    // ifstream ifs_l1_output_golden0("/usr/scratch/rsamanta9/hls/inputs/rpnoutput0.bin", ios::in | ios::binary);
    // ifs_l1_output_golden0.read((char*)(**fl_rpn_output0_fm), (RPN_CONV_IN_CH)*(RPN_INPUT0_IN_FM_HEIGHT)*(RPN_INPUT0_IN_FM_WIDTH)*sizeof(float));    
    // ifs_l1_output_golden0.close();
    
    // for(int f = 0; f < RPN_CONV_IN_CH; f++)
    // {
    //     for(int h = 0; h < RPN_INPUT0_IN_FM_HEIGHT; h++)
    //     {
    //         for(int w = 0; w < RPN_INPUT0_IN_FM_WIDTH; w++)
    //         {
    //             mse += std::pow((fl_rpn_output0_fm[f][h][w] - (float) rpn_output0_fm[f][h][w]), 2);
    //             // cout<<rpn_output0_fm[f][h][w]<<" ";
    //         }
    //         // cout<<endl;
    //     }
    //     // cout << "Golden Output: " << fl_rpn_output0_fm[f][0][0] << std::endl;
    //     // cout << "Actual Output: " << rpn_output0_fm[f][0][0] << std::endl;
    //     // cout << std::endl;
    // }
    
    // mse = mse / ((RPN_CONV_IN_CH)*(RPN_INPUT0_IN_FM_HEIGHT)*(RPN_INPUT0_IN_FM_WIDTH));
    // // std::cout << "RPN CONV 0 MSE:  " << mse << std::endl;
    

    // //----------------------------------------------------------------------
    // // Check mse for RPN conv 1
    // //----------------------------------------------------------------------
    // mse = 0;

    // ifstream ifs_l1_output_golden1("/usr/scratch/rsamanta9/hls/inputs/rpnoutput1.bin", ios::in | ios::binary);
    // ifs_l1_output_golden1.read((char*)(**fl_rpn_output1_fm), (RPN_CONV_IN_CH)*(RPN_INPUT1_IN_FM_HEIGHT)*(RPN_INPUT1_IN_FM_WIDTH)*sizeof(float));    
    // ifs_l1_output_golden1.close();
    
    // for(int f = 0; f < RPN_CONV_IN_CH; f++)
    // {
    //     for(int h = 0; h < RPN_INPUT1_IN_FM_HEIGHT; h++)
    //     {
    //         for(int w = 0; w < RPN_INPUT1_IN_FM_WIDTH; w++)
    //         {
    //             mse += std::pow((fl_rpn_output1_fm[f][h][w] - (float) rpn_output1_fm[f][h][w]), 2);
    //             // cout<<rpn_output0_fm[f][h][w]<<" ";
    //         }
    //         // cout<<endl;
    //     }
    //     // cout << "Golden Output: " << fl_rpn_output0_fm[f][0][0] << std::endl;
    //     // cout << "Actual Output: " << rpn_output0_fm[f][0][0] << std::endl;
    //     // cout << std::endl;
    // }
    
    // mse = mse / ((RPN_CONV_IN_CH)*(RPN_INPUT1_IN_FM_HEIGHT)*(RPN_INPUT1_IN_FM_WIDTH));
    // // std::cout << "RPN CONV 1 MSE:  " << mse << std::endl;
    

    // //----------------------------------------------------------------------
    // // Check mse for RPN conv 2
    // //----------------------------------------------------------------------
    // mse = 0;

    // ifstream ifs_l2_output_golden2("/usr/scratch/rsamanta9/hls/inputs/rpnoutput2.bin", ios::in | ios::binary);
    // ifs_l2_output_golden2.read((char*)(**fl_rpn_output2_fm), (RPN_CONV_IN_CH)*(RPN_INPUT2_IN_FM_HEIGHT)*(RPN_INPUT2_IN_FM_WIDTH)*sizeof(float));    
    // ifs_l2_output_golden2.close();
    
    // for(int f = 0; f < RPN_CONV_IN_CH; f++)
    // {
    //     for(int h = 0; h < RPN_INPUT2_IN_FM_HEIGHT; h++)
    //     {
    //         for(int w = 0; w < RPN_INPUT2_IN_FM_WIDTH; w++)
    //         {
    //             mse += std::pow((fl_rpn_output2_fm[f][h][w] - (float) rpn_output2_fm[f][h][w]), 2);
    //             // cout<<rpn_output0_fm[f][h][w]<<" ";
    //         }
    //         // cout<<endl;
    //     }
    //     // cout << "Golden Output: " << fl_rpn_output0_fm[f][0][0] << std::endl;
    //     // cout << "Actual Output: " << rpn_output0_fm[f][0][0] << std::endl;
    //     // cout << std::endl;
    // }
    
    // mse = mse / ((RPN_CONV_IN_CH)*(RPN_INPUT2_IN_FM_HEIGHT)*(RPN_INPUT2_IN_FM_WIDTH));
    // // std::cout << "RPN CONV 2 MSE:  " << mse << std::endl;
    

    //----------------------------------------------------------------------
    // Check mse for RPN conv 3
    //----------------------------------------------------------------------
    mse = 0;

    ifstream ifs_l3_output_golden3("/usr/scratch/rsamanta9/hls/inputs/rpnoutput3.bin", ios::in | ios::binary);
    ifs_l3_output_golden3.read((char*)(**fl_rpn_output3_fm), (RPN_CONV_IN_CH)*(RPN_INPUT3_IN_FM_HEIGHT)*(RPN_INPUT3_IN_FM_WIDTH)*sizeof(float));    
    ifs_l3_output_golden3.close();
    
    for(int f = 0; f < RPN_CONV_IN_CH; f++)
    {
        for(int h = 0; h < RPN_INPUT3_IN_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RPN_INPUT3_IN_FM_WIDTH; w++)
            {
                mse += std::pow((fl_rpn_output3_fm[f][h][w] - (float) rpn_output3_fm[f][h][w]), 2);
                // cout<<rpn_output3_fm[f][h][w]<<" ";
                // if((fl_rpn_output3_fm[f][h][w] - (float) rpn_output3_fm[f][h][w]>0.00001) || (fl_rpn_output3_fm[f][h][w] - (float) rpn_output3_fm[f][h][w]<-0.001)) cout<<f<<" "<<h<<" "<<w<<" "<<rpn_output3_fm[f][h][w]<<" "<<fl_rpn_output3_fm[f][h][w]<<endl;
            }
            // cout<<endl;
        }
        // cout << "Golden Output: " << fl_rpn_output0_fm[f][0][0] << std::endl;
        // cout << "Actual Output: " << rpn_output0_fm[f][0][0] << std::endl;
        // cout << std::endl;
    }
    
    mse = mse / ((RPN_CONV_IN_CH)*(RPN_INPUT3_IN_FM_HEIGHT)*(RPN_INPUT3_IN_FM_WIDTH));
    // std::cout << "RPN CONV 3 MSE:  " << mse << std::endl;

    //----------------------------------------------------------------------
    // Check mse for RPN conv 4
    //----------------------------------------------------------------------
    mse = 0;

    ifstream ifs_l4_output_golden4("/usr/scratch/rsamanta9/hls/inputs/rpnoutput4.bin", ios::in | ios::binary);
    ifs_l4_output_golden4.read((char*)(**fl_rpn_output4_fm), (RPN_CONV_IN_CH)*(RPN_INPUT4_IN_FM_HEIGHT)*(RPN_INPUT4_IN_FM_WIDTH)*sizeof(float));    
    ifs_l4_output_golden4.close();
    
    for(int f = 0; f < RPN_CONV_IN_CH; f++)
    {
        for(int h = 0; h < RPN_INPUT4_IN_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RPN_INPUT4_IN_FM_WIDTH; w++)
            {
                mse += std::pow((fl_rpn_output4_fm[f][h][w] - (float) rpn_output4_fm[f][h][w]), 2);
                // if((fl_rpn_output4_fm[f][h][w] - (float) rpn_output4_fm[f][h][w]>0.00001) || (fl_rpn_output4_fm[f][h][w] - (float) rpn_output4_fm[f][h][w]<-0.001)) cout<<f<<" "<<h<<" "<<w<<" "<<rpn_output4_fm[f][h][w]<<" "<<fl_rpn_output4_fm[f][h][w]<<endl;
            }
            // cout<<endl;
        }
        // cout << "Golden Output: " << fl_rpn_output0_fm[f][0][0] << std::endl;
        // cout << "Actual Output: " << rpn_output0_fm[f][0][0] << std::endl;
        // cout << std::endl;
    }
    
    mse = mse / ((RPN_CONV_IN_CH)*(RPN_INPUT4_IN_FM_HEIGHT)*(RPN_INPUT4_IN_FM_WIDTH));
    // std::cout << "RPN CONV 4 MSE:  " << mse << std::endl;

    // //----------------------------------------------------------------------
    // // Check mse for RPN cls 0
    // //----------------------------------------------------------------------
    
    // ifstream ifs_l1_output_golden_cls0("/usr/scratch/rsamanta9/hls/inputs/rpnclsoutput0.bin", ios::in | ios::binary);
    // ifs_l1_output_golden_cls0.read((char*)(**fl_rpn_output0_cls_fm), (RPN_CLS_OUT_CH)*(RPN_INPUT0_IN_FM_HEIGHT)*(RPN_INPUT0_IN_FM_WIDTH)*sizeof(float));    
    // ifs_l1_output_golden_cls0.close();
    
    // for(int f = 0; f < RPN_CLS_OUT_CH; f++)
    // {
    //     for(int h = 0; h < RPN_INPUT0_IN_FM_HEIGHT; h++)
    //     {
    //         for(int w = 0; w < RPN_INPUT0_IN_FM_WIDTH; w++)
    //         {
    //             mse += std::pow((fl_rpn_output0_cls_fm[f][h][w] - (float) rpn_output0_cls_fm[f][h][w]), 2);
    //             // cout<<rpn_output0_fm[f][h][w]<<" ";
    //         }
    //         // cout<<endl;
    //     }
    //     // cout << "Golden Output: " << fl_rpn_output0_fm[f][0][0] << std::endl;
    //     // cout << "Actual Output: " << rpn_output0_fm[f][0][0] << std::endl;
    //     // cout << std::endl;
    // }
    
    // mse = mse / ((RPN_CLS_OUT_CH)*(RPN_INPUT0_IN_FM_HEIGHT)*(RPN_INPUT0_IN_FM_WIDTH));
    // // std::cout << "RPN CLS 0 MSE:  " << mse << std::endl;
    

    // //----------------------------------------------------------------------
    // // Check mse for RPN CLS 1
    // //----------------------------------------------------------------------
    // mse = 0;

    // ifstream ifs_l1_output_golden_cls1("/usr/scratch/rsamanta9/hls/inputs/rpnclsoutput1.bin", ios::in | ios::binary);
    // ifs_l1_output_golden_cls1.read((char*)(**fl_rpn_output1_cls_fm), (RPN_CLS_OUT_CH)*(RPN_INPUT1_IN_FM_HEIGHT)*(RPN_INPUT1_IN_FM_WIDTH)*sizeof(float));    
    // ifs_l1_output_golden_cls1.close();
    
    // for(int f = 0; f < RPN_CLS_OUT_CH; f++)
    // {
    //     for(int h = 0; h < RPN_INPUT1_IN_FM_HEIGHT; h++)
    //     {
    //         for(int w = 0; w < RPN_INPUT1_IN_FM_WIDTH; w++)
    //         {
    //             mse += std::pow((fl_rpn_output1_cls_fm[f][h][w] - (float) rpn_output1_cls_fm[f][h][w]), 2);
    //             // cout<<rpn_output0_fm[f][h][w]<<" ";
    //         }
    //         // cout<<endl;
    //     }
    //     // cout << "Golden Output: " << fl_rpn_output0_fm[f][0][0] << std::endl;
    //     // cout << "Actual Output: " << rpn_output0_fm[f][0][0] << std::endl;
    //     // cout << std::endl;
    // }
    
    // mse = mse / ((RPN_CLS_OUT_CH)*(RPN_INPUT1_IN_FM_HEIGHT)*(RPN_INPUT1_IN_FM_WIDTH));
    // // std::cout << "RPN CLS 1 MSE:  " << mse << std::endl;
    

    // //----------------------------------------------------------------------
    // // Check mse for RPN CLS 2
    // //----------------------------------------------------------------------
    // mse = 0;

    // ifstream ifs_l2_output_golden_cls2("/usr/scratch/rsamanta9/hls/inputs/rpnclsoutput2.bin", ios::in | ios::binary);
    // ifs_l2_output_golden_cls2.read((char*)(**fl_rpn_output2_cls_fm), (RPN_CLS_OUT_CH)*(RPN_INPUT2_IN_FM_HEIGHT)*(RPN_INPUT2_IN_FM_WIDTH)*sizeof(float));    
    // ifs_l2_output_golden_cls2.close();
    
    // for(int f = 0; f < RPN_CLS_OUT_CH; f++)
    // {
    //     for(int h = 0; h < RPN_INPUT2_IN_FM_HEIGHT; h++)
    //     {
    //         for(int w = 0; w < RPN_INPUT2_IN_FM_WIDTH; w++)
    //         {
    //             mse += std::pow((fl_rpn_output2_cls_fm[f][h][w] - (float) rpn_output2_cls_fm[f][h][w]), 2);
    //             // cout<<rpn_output0_fm[f][h][w]<<" ";
    //         }
    //         // cout<<endl;
    //     }
    //     // cout << "Golden Output: " << fl_rpn_output0_fm[f][0][0] << std::endl;
    //     // cout << "Actual Output: " << rpn_output0_fm[f][0][0] << std::endl;
    //     // cout << std::endl;
    // }
    
    // mse = mse / ((RPN_CLS_OUT_CH)*(RPN_INPUT2_IN_FM_HEIGHT)*(RPN_INPUT2_IN_FM_WIDTH));
    // // std::cout << "RPN CLS 2 MSE:  " << mse << std::endl;
    

    //----------------------------------------------------------------------
    // Check mse for RPN cls 3
    //----------------------------------------------------------------------
    mse = 0;

    ifstream ifs_l3_output_golden_cls3("/usr/scratch/rsamanta9/hls/inputs/rpnclsoutput3.bin", ios::in | ios::binary);
    ifs_l3_output_golden_cls3.read((char*)(**fl_rpn_output3_cls_fm), (RPN_CLS_OUT_CH)*(RPN_INPUT3_IN_FM_HEIGHT)*(RPN_INPUT3_IN_FM_WIDTH)*sizeof(float));    
    ifs_l3_output_golden_cls3.close();
    
    for(int f = 0; f < RPN_CLS_OUT_CH; f++)
    {
        for(int h = 0; h < RPN_INPUT3_IN_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RPN_INPUT3_IN_FM_WIDTH; w++)
            {
                mse += std::pow((fl_rpn_output3_cls_fm[f][h][w] - (float) rpn_output3_cls_fm[f][h][w]), 2);
            }
        }
    }
    
    mse = mse / ((RPN_CLS_OUT_CH)*(RPN_INPUT3_IN_FM_HEIGHT)*(RPN_INPUT3_IN_FM_WIDTH));
    // std::cout << "RPN CLS 3 MSE:  " << mse << std::endl;

    //----------------------------------------------------------------------
    // Check mse for RPN cls 4
    //----------------------------------------------------------------------
    mse = 0;

    ifstream ifs_l4_output_golden_cls4("/usr/scratch/rsamanta9/hls/inputs/rpnclsoutput4.bin", ios::in | ios::binary);
    ifs_l4_output_golden_cls4.read((char*)(**fl_rpn_output4_cls_fm), (RPN_CLS_OUT_CH)*(RPN_INPUT4_IN_FM_HEIGHT)*(RPN_INPUT4_IN_FM_WIDTH)*sizeof(float));    
    ifs_l4_output_golden_cls4.close();
    
    for(int f = 0; f < RPN_CLS_OUT_CH; f++)
    {
        for(int h = 0; h < RPN_INPUT4_IN_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RPN_INPUT4_IN_FM_WIDTH; w++)
            {
                mse += std::pow((fl_rpn_output4_cls_fm[f][h][w] - (float) rpn_output4_cls_fm[f][h][w]), 2);
            }
        }
    }
    
    mse = mse / ((RPN_CLS_OUT_CH)*(RPN_INPUT4_IN_FM_HEIGHT)*(RPN_INPUT4_IN_FM_WIDTH));
    // std::cout << "RPN CLS 4 MSE:  " << mse << std::endl;

    //     //----------------------------------------------------------------------
    // // Check mse for RPN reg 0
    // //----------------------------------------------------------------------
    
    // ifstream ifs_l1_output_golden_reg0("/usr/scratch/rsamanta9/hls/inputs/rpnregoutput0.bin", ios::in | ios::binary);
    // ifs_l1_output_golden_reg0.read((char*)(**fl_rpn_output0_reg_fm), (RPN_REG_OUT_CH)*(RPN_INPUT0_IN_FM_HEIGHT)*(RPN_INPUT0_IN_FM_WIDTH)*sizeof(float));    
    // ifs_l1_output_golden_reg0.close();
    
    // for(int f = 0; f < RPN_REG_OUT_CH; f++)
    // {
    //     for(int h = 0; h < RPN_INPUT0_IN_FM_HEIGHT; h++)
    //     {
    //         for(int w = 0; w < RPN_INPUT0_IN_FM_WIDTH; w++)
    //         {
    //             mse += std::pow((fl_rpn_output0_reg_fm[f][h][w] - (float) rpn_output0_reg_fm[f][h][w]), 2);
    //             // cout<<rpn_output0_fm[f][h][w]<<" ";
    //         }
    //         // cout<<endl;
    //     }
    //     // cout << "Golden Output: " << fl_rpn_output0_fm[f][0][0] << std::endl;
    //     // cout << "Actual Output: " << rpn_output0_fm[f][0][0] << std::endl;
    //     // cout << std::endl;
    // }
    
    // mse = mse / ((RPN_REG_OUT_CH)*(RPN_INPUT0_IN_FM_HEIGHT)*(RPN_INPUT0_IN_FM_WIDTH));
    // // std::cout << "RPN REG 0 MSE:  " << mse << std::endl;
    

    // //----------------------------------------------------------------------
    // // Check mse for RPN REG 1
    // //----------------------------------------------------------------------
    // mse = 0;

    // ifstream ifs_l1_output_golden_reg1("/usr/scratch/rsamanta9/hls/inputs/rpnregoutput1.bin", ios::in | ios::binary);
    // ifs_l1_output_golden_reg1.read((char*)(**fl_rpn_output1_reg_fm), (RPN_REG_OUT_CH)*(RPN_INPUT1_IN_FM_HEIGHT)*(RPN_INPUT1_IN_FM_WIDTH)*sizeof(float));    
    // ifs_l1_output_golden_reg1.close();
    
    // for(int f = 0; f < RPN_REG_OUT_CH; f++)
    // {
    //     for(int h = 0; h < RPN_INPUT1_IN_FM_HEIGHT; h++)
    //     {
    //         for(int w = 0; w < RPN_INPUT1_IN_FM_WIDTH; w++)
    //         {
    //             mse += std::pow((fl_rpn_output1_reg_fm[f][h][w] - (float) rpn_output1_reg_fm[f][h][w]), 2);
    //             // cout<<rpn_output0_fm[f][h][w]<<" ";
    //         }
    //         // cout<<endl;
    //     }
    //     // cout << "Golden Output: " << fl_rpn_output0_fm[f][0][0] << std::endl;
    //     // cout << "Actual Output: " << rpn_output0_fm[f][0][0] << std::endl;
    //     // cout << std::endl;
    // }
    
    // mse = mse / ((RPN_REG_OUT_CH)*(RPN_INPUT1_IN_FM_HEIGHT)*(RPN_INPUT1_IN_FM_WIDTH));
    // // std::cout << "RPN REG 1 MSE:  " << mse << std::endl;
    

    // //----------------------------------------------------------------------
    // // Check mse for RPN REG 2
    // //----------------------------------------------------------------------
    // mse = 0;

    // ifstream ifs_l2_output_golden_reg2("/usr/scratch/rsamanta9/usr/scratch/rsamanta9/hls/inputs/rpnregoutput2.bin", ios::in | ios::binary);
    // ifs_l2_output_golden_reg2.read((char*)(**fl_rpn_output2_reg_fm), (RPN_REG_OUT_CH)*(RPN_INPUT2_IN_FM_HEIGHT)*(RPN_INPUT2_IN_FM_WIDTH)*sizeof(float));    
    // ifs_l2_output_golden_reg2.close();
    
    // for(int f = 0; f < RPN_REG_OUT_CH; f++)
    // {
    //     for(int h = 0; h < RPN_INPUT2_IN_FM_HEIGHT; h++)
    //     {
    //         for(int w = 0; w < RPN_INPUT2_IN_FM_WIDTH; w++)
    //         {
    //             mse += std::pow((fl_rpn_output2_reg_fm[f][h][w] - (float) rpn_output2_reg_fm[f][h][w]), 2);
    //             // cout<<rpn_output0_fm[f][h][w]<<" ";
    //         }
    //         // cout<<endl;
    //     }
    //     // cout << "Golden Output: " << fl_rpn_output0_fm[f][0][0] << std::endl;
    //     // cout << "Actual Output: " << rpn_output0_fm[f][0][0] << std::endl;
    //     // cout << std::endl;
    // }
    
    // mse = mse / ((RPN_REG_OUT_CH)*(RPN_INPUT2_IN_FM_HEIGHT)*(RPN_INPUT2_IN_FM_WIDTH));
    // // std::cout << "RPN REG 2 MSE:  " << mse << std::endl;
    

    //----------------------------------------------------------------------
    // Check mse for RPN reg 3
    //----------------------------------------------------------------------
    mse = 0;

    ifstream ifs_l3_output_golden_reg3("/usr/scratch/rsamanta9/hls/inputs/rpnregoutput3.bin", ios::in | ios::binary);
    ifs_l3_output_golden_reg3.read((char*)(**fl_rpn_output3_reg_fm), (RPN_REG_OUT_CH)*(RPN_INPUT3_IN_FM_HEIGHT)*(RPN_INPUT3_IN_FM_WIDTH)*sizeof(float));    
    ifs_l3_output_golden_reg3.close();
    
    for(int f = 0; f < RPN_REG_OUT_CH; f++)
    {
        for(int h = 0; h < RPN_INPUT3_IN_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RPN_INPUT3_IN_FM_WIDTH; w++)
            {
                mse += std::pow((fl_rpn_output3_reg_fm[f][h][w] - (float) rpn_output3_reg_fm[f][h][w]), 2);
            }
        }
    }
    
    mse = mse / ((RPN_REG_OUT_CH)*(RPN_INPUT3_IN_FM_HEIGHT)*(RPN_INPUT3_IN_FM_WIDTH));
    // std::cout << "RPN REG 3 MSE:  " << mse << std::endl;

    //----------------------------------------------------------------------
    // Check mse for RPN reg 4
    //----------------------------------------------------------------------
    mse = 0;

    ifstream ifs_l4_output_golden_reg4("/usr/scratch/rsamanta9/hls/inputs/rpnregoutput4.bin", ios::in | ios::binary);
    ifs_l4_output_golden_reg4.read((char*)(**fl_rpn_output4_reg_fm), (RPN_REG_OUT_CH)*(RPN_INPUT4_IN_FM_HEIGHT)*(RPN_INPUT4_IN_FM_WIDTH)*sizeof(float));    
    ifs_l4_output_golden_reg4.close();
    
    for(int f = 0; f < RPN_REG_OUT_CH; f++)
    {
        for(int h = 0; h < RPN_INPUT4_IN_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RPN_INPUT4_IN_FM_WIDTH; w++)
            {
                mse += std::pow((fl_rpn_output4_reg_fm[f][h][w] - (float) rpn_output4_reg_fm[f][h][w]), 2);
            }
        }
    }
    
    mse = mse / ((RPN_REG_OUT_CH)*(RPN_INPUT4_IN_FM_HEIGHT)*(RPN_INPUT4_IN_FM_WIDTH));
    // std::cout << "RPN REG 4 MSE:  " << mse << std::endl;
    
    // TODO
    mse = 0;

    ifstream ifs_l4_output_golden_bbox("/usr/scratch/rsamanta9/hls/inputs/rpndets.bin", ios::in | ios::binary);
    ifs_l4_output_golden_bbox.read((char*)(*fl_dets), (1000)*5*sizeof(float));    
    ifs_l4_output_golden_bbox.close();
    
    for(int f = 0; f < 1000; f++)
    {
        for(int h = 0; h < 5; h++)
        {
            mse += std::pow((fl_dets[f][h] - (float) dets[f][h]), 2);   
            // if(abs(fl_dets[f][h] - (float) dets[f][h])>0.1) cout<<f<<" "<<h<<" "<<dets[f][h]<<" "<<fl_dets[f][h]<<endl;
            
        }
        
    }
    
    mse = mse / (1000*4);
    // std::cout << "RPN DETS MSE:  " << mse << std::endl;


#endif // }

    return 0;
}
