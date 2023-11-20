#include <iostream>
#include <fstream>
#include <cmath>
#include "hls_stream.h"
#include "qdtrack_resnet3_1.h"

using namespace std;

//--------------------------------------------------------------------------
// Layer 3.1
//--------------------------------------------------------------------------
float   resnet_layer3_1_conv1_weights[RESNET_LAYER3_CONV1_OUT_CH][RESNET_LAYER3_CONV1_IN_CH];
float   resnet_layer3_1_bn1_params[3][RESNET_LAYER3_CONV1_OUT_CH];
float   resnet_layer3_1_conv2_weights[RESNET_LAYER3_CONV2_OUT_CH][RESNET_LAYER3_CONV2_IN_CH][3][3];
float   resnet_layer3_1_bn2_params[3][RESNET_LAYER3_CONV2_OUT_CH];
float   resnet_layer3_1_conv3_weights[RESNET_LAYER3_CONV3_OUT_CH][RESNET_LAYER3_CONV3_IN_CH];
float   resnet_layer3_1_bn3_params[3][RESNET_LAYER3_CONV3_OUT_CH];

wt_t    fixp_layer3_1_conv1_weights[RESNET_LAYER3_CONV1_OUT_CH][RESNET_LAYER3_CONV1_IN_CH];
wt_t    fixp_layer3_1_bn1_params[3][RESNET_LAYER3_CONV1_OUT_CH];
wt_t    fixp_layer3_1_conv2_weights[RESNET_LAYER3_CONV2_OUT_CH][RESNET_LAYER3_CONV2_IN_CH][3][3];
wt_t    fixp_layer3_1_bn2_params[3][RESNET_LAYER3_CONV2_OUT_CH];
wt_t    fixp_layer3_1_conv3_weights[RESNET_LAYER3_CONV3_OUT_CH][RESNET_LAYER3_CONV3_IN_CH];
wt_t    fixp_layer3_1_bn3_params[3][RESNET_LAYER3_CONV3_OUT_CH];

void resnet_load_weights()
{
    
    //--------------------------------------------------------------------------
    // Layer 3.1
    //--------------------------------------------------------------------------
    ifstream ifs_l3_1_conv1_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer3_1_conv1_weights.bin", ios::in | ios::binary);
    ifstream ifs_l3_1_conv2_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer3_1_conv2_weights.bin", ios::in | ios::binary);
    ifstream ifs_l3_1_conv3_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer3_1_conv3_weights.bin", ios::in | ios::binary);

    ifs_l3_1_conv1_param.read((char*)(  *resnet_layer3_1_conv1_weights), RESNET_LAYER3_CONV1_OUT_CH*RESNET_LAYER3_CONV1_IN_CH*sizeof(float));
    ifs_l3_1_conv2_param.read((char*)(***resnet_layer3_1_conv2_weights), RESNET_LAYER3_CONV2_OUT_CH*RESNET_LAYER3_CONV2_IN_CH*3*3*sizeof(float));
    ifs_l3_1_conv3_param.read((char*)(  *resnet_layer3_1_conv3_weights), RESNET_LAYER3_CONV3_OUT_CH*RESNET_LAYER3_CONV3_IN_CH*sizeof(float));
    
    ifs_l3_1_conv1_param.close();
    ifs_l3_1_conv2_param.close();
    ifs_l3_1_conv3_param.close();

    ifstream ifs_l3_1_bn1_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer3_1_bn1_params.bin", ios::in | ios::binary);
    ifstream ifs_l3_1_bn2_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer3_1_bn2_params.bin", ios::in | ios::binary);
    ifstream ifs_l3_1_bn3_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer3_1_bn3_params.bin", ios::in | ios::binary);

    ifs_l3_1_bn1_param.read((char*)(*resnet_layer3_1_bn1_params), 3*RESNET_LAYER3_CONV1_OUT_CH*sizeof(float));
    ifs_l3_1_bn2_param.read((char*)(*resnet_layer3_1_bn2_params), 3*RESNET_LAYER3_CONV2_OUT_CH*sizeof(float));
    ifs_l3_1_bn3_param.read((char*)(*resnet_layer3_1_bn3_params), 3*RESNET_LAYER3_CONV3_OUT_CH*sizeof(float));
    
    ifs_l3_1_bn1_param.close();
    ifs_l3_1_bn2_param.close();
    ifs_l3_1_bn3_param.close();
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
    // Layer 3.1
    //--------------------------------------------------------------------------
    convert_1x1<RESNET_LAYER3_CONV1_IN_CH, RESNET_LAYER3_CONV1_OUT_CH>
               (resnet_layer3_1_conv1_weights, fixp_layer3_1_conv1_weights);
    convert_3x3<RESNET_LAYER3_CONV2_IN_CH, RESNET_LAYER3_CONV2_OUT_CH>
               (resnet_layer3_1_conv2_weights, fixp_layer3_1_conv2_weights);
    convert_1x1<RESNET_LAYER3_CONV3_IN_CH, RESNET_LAYER3_CONV3_OUT_CH>
               (resnet_layer3_1_conv3_weights, fixp_layer3_1_conv3_weights);
    
    convert_bn<RESNET_LAYER3_CONV1_OUT_CH>(resnet_layer3_1_bn1_params, fixp_layer3_1_bn1_params);
    convert_bn<RESNET_LAYER3_CONV2_OUT_CH>(resnet_layer3_1_bn2_params, fixp_layer3_1_bn2_params);
    convert_bn<RESNET_LAYER3_CONV3_OUT_CH>(resnet_layer3_1_bn3_params, fixp_layer3_1_bn3_params);
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
    test_resnet_top_3_1 ( 
              fixp_layer3_1_conv1_weights,         fixp_layer3_1_bn1_params,
              fixp_layer3_1_conv2_weights,         fixp_layer3_1_bn2_params,
              fixp_layer3_1_conv3_weights,         fixp_layer3_1_bn3_params
    );


#endif // }

    return 0;
}
