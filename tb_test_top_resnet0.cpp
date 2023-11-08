#include <iostream>
#include <fstream>
#include <cmath>
#include "hls_stream.h"
#include "qdtrack_resnet0.h"

using namespace std;

float  input_image[RESNET_LAYER0_CONV1_IN_CH][RESNET_LAYER0_IN_FM_HEIGHT][RESNET_LAYER0_IN_FM_WIDTH];

fm_t   resnet_layer0_input_fm [RESNET_LAYER0_CONV1_IN_CH][RESNET_LAYER0_IN_FM_HEIGHT][RESNET_LAYER0_IN_FM_WIDTH]    = {0};
fm_t   resnet_layer0_output_fm[RESNET_LAYER0_CONV1_OUT_CH][RESNET_LAYER0_OUT_FM_HEIGHT][RESNET_LAYER0_OUT_FM_WIDTH] = {0};

fm_t   resnet_layer1_input_fm [RESNET_LAYER1_0_DS_OUT_CH][RESNET_LAYER1_0_FM_HEIGHT][RESNET_LAYER1_0_FM_WIDTH] = {0};

//--------------------------------------------------------------------------
// Layer 0
//--------------------------------------------------------------------------
float   resnet_layer0_conv1_weights[RESNET_LAYER0_CONV1_OUT_CH][RESNET_LAYER0_CONV1_IN_CH][7][7];
float   resnet_layer0_bn1_params[3][RESNET_LAYER0_CONV1_OUT_CH];

wt_t   fixp_layer0_conv1_weights[RESNET_LAYER0_CONV1_OUT_CH][RESNET_LAYER0_CONV1_IN_CH][7][7];
wt_t   fixp_layer0_bn1_params[3][RESNET_LAYER0_CONV1_OUT_CH];

//--------------------------------------------------------------------------
// PyTorch reference outputs
//--------------------------------------------------------------------------
float   golden_maxpool_out[RESNET_LAYER0_CONV1_OUT_CH][RESNET_LAYER0_OUT_FM_HEIGHT][RESNET_LAYER0_OUT_FM_WIDTH];


void resnet_load_weights()
{
    //--------------------------------------------------------------------------
    // Layer 0
    //--------------------------------------------------------------------------
    ifstream ifs_l0_conv1_param("/usr/scratch/rsamanta9/bin/resnet_backbone/conv1_weights.bin", ios::in | ios::binary);
    ifs_l0_conv1_param.read((char*)(***resnet_layer0_conv1_weights), RESNET_LAYER0_CONV1_OUT_CH*RESNET_LAYER0_CONV1_IN_CH*7*7*sizeof(float));
    ifs_l0_conv1_param.close();

    ifstream ifs_l0_bn1_param("/usr/scratch/rsamanta9/bin/resnet_backbone/bn1_params.bin", ios::in | ios::binary);
    ifs_l0_bn1_param.read((char*)(*resnet_layer0_bn1_params), 3*RESNET_LAYER0_CONV1_OUT_CH*sizeof(float));
    ifs_l0_bn1_param.close();
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
    // Layer 0
    //--------------------------------------------------------------------------
    for(int i = 0; i < RESNET_LAYER0_CONV1_OUT_CH; i++)
        for(int j = 0; j< RESNET_LAYER0_CONV1_IN_CH; j++)
            for(int m = 0; m < 7; m++)
                for(int n =0; n < 7; n++)
                    fixp_layer0_conv1_weights[i][j][m][n] = (wt_t) resnet_layer0_conv1_weights[i][j][m][n];

    convert_bn<RESNET_LAYER0_CONV1_OUT_CH>(resnet_layer0_bn1_params, fixp_layer0_bn1_params);
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
    
    // Read input image
    ifstream ifs_input_img("/usr/scratch/rsamanta9/bin/resnet_backbone/qdtrack_image0.bin", ios::in | ios::binary);
    ifs_input_img.read((char*)(**input_image), 3*736*1280*sizeof(float));
    ifs_input_img.close();

    // Convert input image data to fixed point
    for(int c = 0; c < RESNET_LAYER0_CONV1_IN_CH; c++)
    {
        for(int h = 0; h < RESNET_LAYER0_IN_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RESNET_LAYER0_IN_FM_WIDTH; w++)
            {
                resnet_layer0_input_fm[c][h][w] = (fm_t) input_image[c][h][w];
            }
        }
    }
    
#ifdef TEST_COMPLETE_MODEL // {
    //----------------------------------------------------------------------
    // ResNet50 Top-level wrapper 
    //----------------------------------------------------------------------
    test_resnet_top_0( 
            resnet_layer0_input_fm,
              fixp_layer0_conv1_weights,           fixp_layer0_bn1_params, 
              resnet_layer0_output_fm,
              
              resnet_layer1_input_fm
    );


#endif // }

    return 0;
}
