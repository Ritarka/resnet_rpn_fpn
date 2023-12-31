#include <iostream>
#include <fstream>
#include <cmath>
#include "hls_stream.h"
#include "qdtrack_resnet4_2.h"

using namespace std;

fm_t   resnet_layer4_output_fm[RESNET_LAYER4_0_DS_OUT_CH][RESNET_LAYER4_0_FM_HEIGHT][RESNET_LAYER4_0_FM_WIDTH] = {0};

//--------------------------------------------------------------------------
// Layer 4.2
//--------------------------------------------------------------------------
float   resnet_layer4_2_conv1_weights[RESNET_LAYER4_CONV1_OUT_CH][RESNET_LAYER4_CONV1_IN_CH];
float   resnet_layer4_2_bn1_params[3][RESNET_LAYER4_CONV1_OUT_CH];
float   resnet_layer4_2_conv2_weights[RESNET_LAYER4_CONV2_OUT_CH][RESNET_LAYER4_CONV2_IN_CH][3][3];
float   resnet_layer4_2_bn2_params[3][RESNET_LAYER4_CONV2_OUT_CH];
float   resnet_layer4_2_conv3_weights[RESNET_LAYER4_CONV3_OUT_CH][RESNET_LAYER4_CONV3_IN_CH];
float   resnet_layer4_2_bn3_params[3][RESNET_LAYER4_CONV3_OUT_CH];

wt_t    fixp_layer4_2_conv1_weights[RESNET_LAYER4_CONV1_OUT_CH][RESNET_LAYER4_CONV1_IN_CH];
wt_t    fixp_layer4_2_bn1_params[3][RESNET_LAYER4_CONV1_OUT_CH];
wt_t    fixp_layer4_2_conv2_weights[RESNET_LAYER4_CONV2_OUT_CH][RESNET_LAYER4_CONV2_IN_CH][3][3];
wt_t    fixp_layer4_2_bn2_params[3][RESNET_LAYER4_CONV2_OUT_CH];
wt_t    fixp_layer4_2_conv3_weights[RESNET_LAYER4_CONV3_OUT_CH][RESNET_LAYER4_CONV3_IN_CH];
wt_t    fixp_layer4_2_bn3_params[3][RESNET_LAYER4_CONV3_OUT_CH];

//--------------------------------------------------------------------------
// PyTorch reference outputs
//--------------------------------------------------------------------------
float   golden_layer4_2_bn3_relu_out[RESNET_LAYER4_CONV3_OUT_CH][RESNET_LAYER4_FM_HEIGHT][RESNET_LAYER4_FM_WIDTH];


void resnet_load_weights()
{

    //--------------------------------------------------------------------------
    // Layer 4.2
    //--------------------------------------------------------------------------
    ifstream ifs_l4_2_conv1_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer4_2_conv1_weights.bin", ios::in | ios::binary);
    ifstream ifs_l4_2_conv2_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer4_2_conv2_weights.bin", ios::in | ios::binary);
    ifstream ifs_l4_2_conv3_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer4_2_conv3_weights.bin", ios::in | ios::binary);

    ifs_l4_2_conv1_param.read((char*)(  *resnet_layer4_2_conv1_weights), RESNET_LAYER4_CONV1_OUT_CH*RESNET_LAYER4_CONV1_IN_CH*sizeof(float));
    ifs_l4_2_conv2_param.read((char*)(***resnet_layer4_2_conv2_weights), RESNET_LAYER4_CONV2_OUT_CH*RESNET_LAYER4_CONV2_IN_CH*3*3*sizeof(float));
    ifs_l4_2_conv3_param.read((char*)(  *resnet_layer4_2_conv3_weights), RESNET_LAYER4_CONV3_OUT_CH*RESNET_LAYER4_CONV3_IN_CH*sizeof(float));

    ifs_l4_2_conv1_param.close();
    ifs_l4_2_conv2_param.close();
    ifs_l4_2_conv3_param.close();

    ifstream ifs_l4_2_bn1_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer4_2_bn1_params.bin", ios::in | ios::binary);
    ifstream ifs_l4_2_bn2_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer4_2_bn2_params.bin", ios::in | ios::binary);
    ifstream ifs_l4_2_bn3_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer4_2_bn3_params.bin", ios::in | ios::binary);

    ifs_l4_2_bn1_param.read((char*)(*resnet_layer4_2_bn1_params), 3*RESNET_LAYER4_CONV1_OUT_CH*sizeof(float));
    ifs_l4_2_bn2_param.read((char*)(*resnet_layer4_2_bn2_params), 3*RESNET_LAYER4_CONV2_OUT_CH*sizeof(float));
    ifs_l4_2_bn3_param.read((char*)(*resnet_layer4_2_bn3_params), 3*RESNET_LAYER4_CONV3_OUT_CH*sizeof(float));

    ifs_l4_2_bn1_param.close();
    ifs_l4_2_bn2_param.close();
    ifs_l4_2_bn3_param.close();
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

void resnet_convert_weights_type()
{

    //--------------------------------------------------------------------------
    // Layer 4.2
    //--------------------------------------------------------------------------
    convert_1x1<RESNET_LAYER4_CONV1_IN_CH, RESNET_LAYER4_CONV1_OUT_CH>
               (resnet_layer4_2_conv1_weights, fixp_layer4_2_conv1_weights);
    convert_3x3<RESNET_LAYER4_CONV2_IN_CH, RESNET_LAYER4_CONV2_OUT_CH>
               (resnet_layer4_2_conv2_weights, fixp_layer4_2_conv2_weights);
    convert_1x1<RESNET_LAYER4_CONV3_IN_CH, RESNET_LAYER4_CONV3_OUT_CH>
               (resnet_layer4_2_conv3_weights, fixp_layer4_2_conv3_weights);
    
    convert_bn<RESNET_LAYER4_CONV1_OUT_CH>(resnet_layer4_2_bn1_params, fixp_layer4_2_bn1_params);
    convert_bn<RESNET_LAYER4_CONV2_OUT_CH>(resnet_layer4_2_bn2_params, fixp_layer4_2_bn2_params);
    convert_bn<RESNET_LAYER4_CONV3_OUT_CH>(resnet_layer4_2_bn3_params, fixp_layer4_2_bn3_params);
}


int main ()
{
    long double mse = 0.0;
    
    // Load ResNet-50 convolution weights and batchnorm parameters
    cout << "Reading ResNet-50 params ..." << endl;
    resnet_load_weights();

    // Convert floating point weights to fixed-point
    cout << "Converting ResNet-50 params to fixed-point type ..." << endl;
    resnet_convert_weights_type();
    
#ifdef TEST_COMPLETE_MODEL // {
    //----------------------------------------------------------------------
    // ResNet50 Top-level wrapper 
    //----------------------------------------------------------------------
    test_resnet_top_4_2 ( 
              fixp_layer4_2_conv1_weights,         fixp_layer4_2_bn1_params,
              fixp_layer4_2_conv2_weights,         fixp_layer4_2_bn2_params,
              fixp_layer4_2_conv3_weights,         fixp_layer4_2_bn3_params,
              resnet_layer4_output_fm

    );
    
    ifstream ifs_l4_output_golden("/usr/scratch/rsamanta9/bin/resnet_backbone/layer4_2_relu.bin", ios::in | ios::binary);
    ifs_l4_output_golden.read((char*)(**golden_layer4_2_bn3_relu_out), RESNET_LAYER4_CONV3_OUT_CH*RESNET_LAYER4_FM_HEIGHT*RESNET_LAYER4_FM_WIDTH*sizeof(float));    
    ifs_l4_output_golden.close();
    
    for(int f = 0; f < RESNET_LAYER4_CONV3_OUT_CH; f++)
    {
        for(int h = 0; h < RESNET_LAYER4_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RESNET_LAYER4_FM_WIDTH; w++)
            {
                mse += std::pow((golden_layer4_2_bn3_relu_out[f][h][w] - (float) resnet_layer4_output_fm[f][h][w]), 2);
            }
        }
        cout << "Golden Output: " << golden_layer4_2_bn3_relu_out[f][0][0] << std::endl;
        cout << "Actual Output: " << resnet_layer4_output_fm[f][0][0] << std::endl;
        cout << std::endl;
    }
    
    mse = mse / (RESNET_LAYER4_CONV3_OUT_CH * RESNET_LAYER4_FM_HEIGHT * RESNET_LAYER4_FM_WIDTH);


#endif // }

    return 0;
}
