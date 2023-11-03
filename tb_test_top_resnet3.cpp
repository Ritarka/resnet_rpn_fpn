#include <iostream>
#include <fstream>
#include <cmath>
#include "hls_stream.h"
#include "qdtrack_resnet3.h"

using namespace std;

fm_t   resnet_layer2_output_fm[RESNET_LAYER2_0_DS_OUT_CH][RESNET_LAYER2_0_FM_HEIGHT][RESNET_LAYER2_0_FM_WIDTH] = {0};

fm_t   resnet_layer3_input_fm [RESNET_LAYER3_0_DS_OUT_CH][RESNET_LAYER3_0_FM_HEIGHT][RESNET_LAYER3_0_FM_WIDTH] = {0};
fm_t   resnet_layer3_output_fm[RESNET_LAYER3_0_DS_OUT_CH][RESNET_LAYER3_0_FM_HEIGHT][RESNET_LAYER3_0_FM_WIDTH] = {0};

fm_t   resnet_layer4_input_fm [RESNET_LAYER4_0_DS_OUT_CH][RESNET_LAYER4_0_FM_HEIGHT][RESNET_LAYER4_0_FM_WIDTH] = {0};

//--------------------------------------------------------------------------
// Layer 3.0
//--------------------------------------------------------------------------
float   resnet_layer3_0_downsample_0_weights[RESNET_LAYER3_0_DS_OUT_CH][RESNET_LAYER3_0_DS_IN_CH];
float   resnet_layer3_0_downsample_1_params[3][RESNET_LAYER3_0_DS_OUT_CH];
float   resnet_layer3_0_conv1_weights[RESNET_LAYER3_0_CONV1_OUT_CH][RESNET_LAYER3_0_CONV1_IN_CH];
float   resnet_layer3_0_bn1_params[3][RESNET_LAYER3_0_CONV1_OUT_CH];
float   resnet_layer3_0_conv2_weights[RESNET_LAYER3_0_CONV2_OUT_CH][RESNET_LAYER3_0_CONV2_IN_CH][3][3];
float   resnet_layer3_0_bn2_params[3][RESNET_LAYER3_0_CONV2_OUT_CH];
float   resnet_layer3_0_conv3_weights[RESNET_LAYER3_0_CONV3_OUT_CH][RESNET_LAYER3_0_CONV3_IN_CH];
float   resnet_layer3_0_bn3_params[3][RESNET_LAYER3_0_CONV3_OUT_CH];

wt_t    fixp_layer3_0_downsample_0_weights[RESNET_LAYER3_0_DS_OUT_CH][RESNET_LAYER3_0_DS_IN_CH];
wt_t    fixp_layer3_0_downsample_1_params[3][RESNET_LAYER3_0_DS_OUT_CH];
wt_t    fixp_layer3_0_conv1_weights[RESNET_LAYER3_0_CONV1_OUT_CH][RESNET_LAYER3_0_CONV1_IN_CH];
wt_t    fixp_layer3_0_bn1_params[3][RESNET_LAYER3_0_CONV1_OUT_CH];
wt_t    fixp_layer3_0_conv2_weights[RESNET_LAYER3_0_CONV2_OUT_CH][RESNET_LAYER3_0_CONV2_IN_CH][3][3];
wt_t    fixp_layer3_0_bn2_params[3][RESNET_LAYER3_0_CONV2_OUT_CH];
wt_t    fixp_layer3_0_conv3_weights[RESNET_LAYER3_0_CONV3_OUT_CH][RESNET_LAYER3_0_CONV3_IN_CH];
wt_t    fixp_layer3_0_bn3_params[3][RESNET_LAYER3_0_CONV3_OUT_CH];

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

//--------------------------------------------------------------------------
// Layer 3.2
//--------------------------------------------------------------------------
float   resnet_layer3_2_conv1_weights[RESNET_LAYER3_CONV1_OUT_CH][RESNET_LAYER3_CONV1_IN_CH];
float   resnet_layer3_2_bn1_params[3][RESNET_LAYER3_CONV1_OUT_CH];
float   resnet_layer3_2_conv2_weights[RESNET_LAYER3_CONV2_OUT_CH][RESNET_LAYER3_CONV2_IN_CH][3][3];
float   resnet_layer3_2_bn2_params[3][RESNET_LAYER3_CONV2_OUT_CH];
float   resnet_layer3_2_conv3_weights[RESNET_LAYER3_CONV3_OUT_CH][RESNET_LAYER3_CONV3_IN_CH];
float   resnet_layer3_2_bn3_params[3][RESNET_LAYER3_CONV3_OUT_CH];

wt_t    fixp_layer3_2_conv1_weights[RESNET_LAYER3_CONV1_OUT_CH][RESNET_LAYER3_CONV1_IN_CH];
wt_t    fixp_layer3_2_bn1_params[3][RESNET_LAYER3_CONV1_OUT_CH];
wt_t    fixp_layer3_2_conv2_weights[RESNET_LAYER3_CONV2_OUT_CH][RESNET_LAYER3_CONV2_IN_CH][3][3];
wt_t    fixp_layer3_2_bn2_params[3][RESNET_LAYER3_CONV2_OUT_CH];
wt_t    fixp_layer3_2_conv3_weights[RESNET_LAYER3_CONV3_OUT_CH][RESNET_LAYER3_CONV3_IN_CH];
wt_t    fixp_layer3_2_bn3_params[3][RESNET_LAYER3_CONV3_OUT_CH];

//--------------------------------------------------------------------------
// Layer 3.3
//--------------------------------------------------------------------------
float   resnet_layer3_3_conv1_weights[RESNET_LAYER3_CONV1_OUT_CH][RESNET_LAYER3_CONV1_IN_CH];
float   resnet_layer3_3_bn1_params[3][RESNET_LAYER3_CONV1_OUT_CH];
float   resnet_layer3_3_conv2_weights[RESNET_LAYER3_CONV2_OUT_CH][RESNET_LAYER3_CONV2_IN_CH][3][3];
float   resnet_layer3_3_bn2_params[3][RESNET_LAYER3_CONV2_OUT_CH];
float   resnet_layer3_3_conv3_weights[RESNET_LAYER3_CONV3_OUT_CH][RESNET_LAYER3_CONV3_IN_CH];
float   resnet_layer3_3_bn3_params[3][RESNET_LAYER3_CONV3_OUT_CH];

wt_t    fixp_layer3_3_conv1_weights[RESNET_LAYER3_CONV1_OUT_CH][RESNET_LAYER3_CONV1_IN_CH];
wt_t    fixp_layer3_3_bn1_params[3][RESNET_LAYER3_CONV1_OUT_CH];
wt_t    fixp_layer3_3_conv2_weights[RESNET_LAYER3_CONV2_OUT_CH][RESNET_LAYER3_CONV2_IN_CH][3][3];
wt_t    fixp_layer3_3_bn2_params[3][RESNET_LAYER3_CONV2_OUT_CH];
wt_t    fixp_layer3_3_conv3_weights[RESNET_LAYER3_CONV3_OUT_CH][RESNET_LAYER3_CONV3_IN_CH];
wt_t    fixp_layer3_3_bn3_params[3][RESNET_LAYER3_CONV3_OUT_CH];

//--------------------------------------------------------------------------
// Layer 3.4
//--------------------------------------------------------------------------
float   resnet_layer3_4_conv1_weights[RESNET_LAYER3_CONV1_OUT_CH][RESNET_LAYER3_CONV1_IN_CH];
float   resnet_layer3_4_bn1_params[3][RESNET_LAYER3_CONV1_OUT_CH];
float   resnet_layer3_4_conv2_weights[RESNET_LAYER3_CONV2_OUT_CH][RESNET_LAYER3_CONV2_IN_CH][3][3];
float   resnet_layer3_4_bn2_params[3][RESNET_LAYER3_CONV2_OUT_CH];
float   resnet_layer3_4_conv3_weights[RESNET_LAYER3_CONV3_OUT_CH][RESNET_LAYER3_CONV3_IN_CH];
float   resnet_layer3_4_bn3_params[3][RESNET_LAYER3_CONV3_OUT_CH];

wt_t    fixp_layer3_4_conv1_weights[RESNET_LAYER3_CONV1_OUT_CH][RESNET_LAYER3_CONV1_IN_CH];
wt_t    fixp_layer3_4_bn1_params[3][RESNET_LAYER3_CONV1_OUT_CH];
wt_t    fixp_layer3_4_conv2_weights[RESNET_LAYER3_CONV2_OUT_CH][RESNET_LAYER3_CONV2_IN_CH][3][3];
wt_t    fixp_layer3_4_bn2_params[3][RESNET_LAYER3_CONV2_OUT_CH];
wt_t    fixp_layer3_4_conv3_weights[RESNET_LAYER3_CONV3_OUT_CH][RESNET_LAYER3_CONV3_IN_CH];
wt_t    fixp_layer3_4_bn3_params[3][RESNET_LAYER3_CONV3_OUT_CH];

//--------------------------------------------------------------------------
// Layer 3.5
//--------------------------------------------------------------------------
float   resnet_layer3_5_conv1_weights[RESNET_LAYER3_CONV1_OUT_CH][RESNET_LAYER3_CONV1_IN_CH];
float   resnet_layer3_5_bn1_params[3][RESNET_LAYER3_CONV1_OUT_CH];
float   resnet_layer3_5_conv2_weights[RESNET_LAYER3_CONV2_OUT_CH][RESNET_LAYER3_CONV2_IN_CH][3][3];
float   resnet_layer3_5_bn2_params[3][RESNET_LAYER3_CONV2_OUT_CH];
float   resnet_layer3_5_conv3_weights[RESNET_LAYER3_CONV3_OUT_CH][RESNET_LAYER3_CONV3_IN_CH];
float   resnet_layer3_5_bn3_params[3][RESNET_LAYER3_CONV3_OUT_CH];

wt_t    fixp_layer3_5_conv1_weights[RESNET_LAYER3_CONV1_OUT_CH][RESNET_LAYER3_CONV1_IN_CH];
wt_t    fixp_layer3_5_bn1_params[3][RESNET_LAYER3_CONV1_OUT_CH];
wt_t    fixp_layer3_5_conv2_weights[RESNET_LAYER3_CONV2_OUT_CH][RESNET_LAYER3_CONV2_IN_CH][3][3];
wt_t    fixp_layer3_5_bn2_params[3][RESNET_LAYER3_CONV2_OUT_CH];
wt_t    fixp_layer3_5_conv3_weights[RESNET_LAYER3_CONV3_OUT_CH][RESNET_LAYER3_CONV3_IN_CH];
wt_t    fixp_layer3_5_bn3_params[3][RESNET_LAYER3_CONV3_OUT_CH];

//--------------------------------------------------------------------------
// PyTorch reference outputs
//--------------------------------------------------------------------------
// float   golden_maxpool_out[RESNET_LAYER0_CONV1_OUT_CH][RESNET_LAYER0_OUT_FM_HEIGHT][RESNET_LAYER0_OUT_FM_WIDTH];
// float   golden_layer1_2_bn3_relu_out[RESNET_LAYER1_CONV3_OUT_CH][RESNET_LAYER1_FM_HEIGHT][RESNET_LAYER1_FM_WIDTH];
// float   golden_layer2_3_bn3_relu_out[RESNET_LAYER2_CONV3_OUT_CH][RESNET_LAYER2_FM_HEIGHT][RESNET_LAYER2_FM_WIDTH];
float   golden_layer3_5_bn3_relu_out[RESNET_LAYER3_CONV3_OUT_CH][RESNET_LAYER3_FM_HEIGHT][RESNET_LAYER3_FM_WIDTH];
// float   golden_layer4_2_bn3_relu_out[RESNET_LAYER4_CONV3_OUT_CH][RESNET_LAYER4_FM_HEIGHT][RESNET_LAYER4_FM_WIDTH];


void resnet_load_weights()
{
    //--------------------------------------------------------------------------
    // Layer 3.0
    //--------------------------------------------------------------------------
    ifstream ifs_l3_0_conv1_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer3_0_conv1_weights.bin", ios::in | ios::binary);
    ifstream ifs_l3_0_conv2_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer3_0_conv2_weights.bin", ios::in | ios::binary);
    ifstream ifs_l3_0_conv3_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer3_0_conv3_weights.bin", ios::in | ios::binary);
    
    ifs_l3_0_conv1_param.read((char*)(  *resnet_layer3_0_conv1_weights), RESNET_LAYER3_0_CONV1_OUT_CH*RESNET_LAYER3_0_CONV1_IN_CH*sizeof(float));
    ifs_l3_0_conv2_param.read((char*)(***resnet_layer3_0_conv2_weights), RESNET_LAYER3_0_CONV2_OUT_CH*RESNET_LAYER3_0_CONV2_IN_CH*3*3*sizeof(float));
    ifs_l3_0_conv3_param.read((char*)(  *resnet_layer3_0_conv3_weights), RESNET_LAYER3_0_CONV3_OUT_CH*RESNET_LAYER3_0_CONV3_IN_CH*sizeof(float));
    
    ifs_l3_0_conv1_param.close();
    ifs_l3_0_conv2_param.close();
    ifs_l3_0_conv3_param.close();
    
    ifstream ifs_l3_0_bn1_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer3_0_bn1_params.bin", ios::in | ios::binary);
    ifstream ifs_l3_0_bn2_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer3_0_bn2_params.bin", ios::in | ios::binary);
    ifstream ifs_l3_0_bn3_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer3_0_bn3_params.bin", ios::in | ios::binary);
    
    ifs_l3_0_bn1_param.read((char*)(*resnet_layer3_0_bn1_params), 3*RESNET_LAYER3_0_CONV1_OUT_CH*sizeof(float));
    ifs_l3_0_bn2_param.read((char*)(*resnet_layer3_0_bn2_params), 3*RESNET_LAYER3_0_CONV2_OUT_CH*sizeof(float));
    ifs_l3_0_bn3_param.read((char*)(*resnet_layer3_0_bn3_params), 3*RESNET_LAYER3_0_CONV3_OUT_CH*sizeof(float));
    
    ifs_l3_0_bn1_param.close();
    ifs_l3_0_bn2_param.close();
    ifs_l3_0_bn3_param.close();

    ifstream ifs_l3_0_downsample_0_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer3_0_downsample_0_weights.bin", ios::in | ios::binary);
    ifstream ifs_l3_0_downsample_1_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer3_0_downsample_1_params.bin", ios::in | ios::binary);

    ifs_l3_0_downsample_0_param.read((char*)(*resnet_layer3_0_downsample_0_weights), RESNET_LAYER3_0_DS_OUT_CH*RESNET_LAYER3_0_DS_IN_CH*sizeof(float));
    ifs_l3_0_downsample_1_param.read((char*)(*resnet_layer3_0_downsample_1_params), 3*RESNET_LAYER3_0_DS_OUT_CH*sizeof(float));

    ifs_l3_0_downsample_0_param.close();
    ifs_l3_0_downsample_1_param.close();
    
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

    //--------------------------------------------------------------------------
    // Layer 3.2
    //--------------------------------------------------------------------------
    ifstream ifs_l3_2_conv1_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer3_2_conv1_weights.bin", ios::in | ios::binary);
    ifstream ifs_l3_2_conv2_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer3_2_conv2_weights.bin", ios::in | ios::binary);
    ifstream ifs_l3_2_conv3_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer3_2_conv3_weights.bin", ios::in | ios::binary);
    
    ifs_l3_2_conv1_param.read((char*)(  *resnet_layer3_2_conv1_weights), RESNET_LAYER3_CONV1_OUT_CH*RESNET_LAYER3_CONV1_IN_CH*sizeof(float));
    ifs_l3_2_conv2_param.read((char*)(***resnet_layer3_2_conv2_weights), RESNET_LAYER3_CONV2_OUT_CH*RESNET_LAYER3_CONV2_IN_CH*3*3*sizeof(float));
    ifs_l3_2_conv3_param.read((char*)(  *resnet_layer3_2_conv3_weights), RESNET_LAYER3_CONV3_OUT_CH*RESNET_LAYER3_CONV3_IN_CH*sizeof(float));

    ifs_l3_2_conv1_param.close();
    ifs_l3_2_conv2_param.close();
    ifs_l3_2_conv3_param.close();
    
    ifstream ifs_l3_2_bn1_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer3_2_bn1_params.bin", ios::in | ios::binary);
    ifstream ifs_l3_2_bn2_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer3_2_bn2_params.bin", ios::in | ios::binary);
    ifstream ifs_l3_2_bn3_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer3_2_bn3_params.bin", ios::in | ios::binary);

    ifs_l3_2_bn1_param.read((char*)(*resnet_layer3_2_bn1_params), 3*RESNET_LAYER3_CONV1_OUT_CH*sizeof(float));
    ifs_l3_2_bn2_param.read((char*)(*resnet_layer3_2_bn2_params), 3*RESNET_LAYER3_CONV2_OUT_CH*sizeof(float));
    ifs_l3_2_bn3_param.read((char*)(*resnet_layer3_2_bn3_params), 3*RESNET_LAYER3_CONV3_OUT_CH*sizeof(float));
    
    ifs_l3_2_bn1_param.close();
    ifs_l3_2_bn2_param.close();
    ifs_l3_2_bn3_param.close();

    //--------------------------------------------------------------------------
    // Layer 3.3
    //--------------------------------------------------------------------------
    ifstream ifs_l3_3_conv1_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer3_3_conv1_weights.bin", ios::in | ios::binary);
    ifstream ifs_l3_3_conv2_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer3_3_conv2_weights.bin", ios::in | ios::binary);
    ifstream ifs_l3_3_conv3_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer3_3_conv3_weights.bin", ios::in | ios::binary);

    ifs_l3_3_conv1_param.read((char*)(  *resnet_layer3_3_conv1_weights), RESNET_LAYER3_CONV1_OUT_CH*RESNET_LAYER3_CONV1_IN_CH*sizeof(float));
    ifs_l3_3_conv2_param.read((char*)(***resnet_layer3_3_conv2_weights), RESNET_LAYER3_CONV2_OUT_CH*RESNET_LAYER3_CONV2_IN_CH*3*3*sizeof(float));
    ifs_l3_3_conv3_param.read((char*)(  *resnet_layer3_3_conv3_weights), RESNET_LAYER3_CONV3_OUT_CH*RESNET_LAYER3_CONV3_IN_CH*sizeof(float));

    ifs_l3_3_conv1_param.close();
    ifs_l3_3_conv2_param.close();
    ifs_l3_3_conv3_param.close();
    
    ifstream ifs_l3_3_bn1_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer3_3_bn1_params.bin", ios::in | ios::binary);
    ifstream ifs_l3_3_bn2_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer3_3_bn2_params.bin", ios::in | ios::binary);
    ifstream ifs_l3_3_bn3_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer3_3_bn3_params.bin", ios::in | ios::binary);
    
    ifs_l3_3_bn1_param.read((char*)(*resnet_layer3_3_bn1_params), 3*RESNET_LAYER3_CONV1_OUT_CH*sizeof(float));
    ifs_l3_3_bn2_param.read((char*)(*resnet_layer3_3_bn2_params), 3*RESNET_LAYER3_CONV2_OUT_CH*sizeof(float));
    ifs_l3_3_bn3_param.read((char*)(*resnet_layer3_3_bn3_params), 3*RESNET_LAYER3_CONV3_OUT_CH*sizeof(float));
    
    ifs_l3_3_bn1_param.close();
    ifs_l3_3_bn2_param.close();
    ifs_l3_3_bn3_param.close();

    //--------------------------------------------------------------------------
    // Layer 3.4
    //--------------------------------------------------------------------------
    ifstream ifs_l3_4_conv1_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer3_4_conv1_weights.bin", ios::in | ios::binary);
    ifstream ifs_l3_4_conv2_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer3_4_conv2_weights.bin", ios::in | ios::binary);
    ifstream ifs_l3_4_conv3_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer3_4_conv3_weights.bin", ios::in | ios::binary);

    ifs_l3_4_conv1_param.read((char*)(  *resnet_layer3_4_conv1_weights), RESNET_LAYER3_CONV1_OUT_CH*RESNET_LAYER3_CONV1_IN_CH*sizeof(float));
    ifs_l3_4_conv2_param.read((char*)(***resnet_layer3_4_conv2_weights), RESNET_LAYER3_CONV2_OUT_CH*RESNET_LAYER3_CONV2_IN_CH*3*3*sizeof(float));
    ifs_l3_4_conv3_param.read((char*)(  *resnet_layer3_4_conv3_weights), RESNET_LAYER3_CONV3_OUT_CH*RESNET_LAYER3_CONV3_IN_CH*sizeof(float));
    
    ifs_l3_4_conv1_param.close();
    ifs_l3_4_conv2_param.close();
    ifs_l3_4_conv3_param.close();

    ifstream ifs_l3_4_bn1_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer3_4_bn1_params.bin", ios::in | ios::binary);
    ifstream ifs_l3_4_bn2_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer3_4_bn2_params.bin", ios::in | ios::binary);
    ifstream ifs_l3_4_bn3_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer3_4_bn3_params.bin", ios::in | ios::binary);
    
    ifs_l3_4_bn1_param.read((char*)(*resnet_layer3_4_bn1_params), 3*RESNET_LAYER3_CONV1_OUT_CH*sizeof(float));
    ifs_l3_4_bn2_param.read((char*)(*resnet_layer3_4_bn2_params), 3*RESNET_LAYER3_CONV2_OUT_CH*sizeof(float));
    ifs_l3_4_bn3_param.read((char*)(*resnet_layer3_4_bn3_params), 3*RESNET_LAYER3_CONV3_OUT_CH*sizeof(float));
    
    ifs_l3_4_bn1_param.close();
    ifs_l3_4_bn2_param.close();
    ifs_l3_4_bn3_param.close();

    //--------------------------------------------------------------------------
    // Layer 3.5
    //--------------------------------------------------------------------------
    ifstream ifs_l3_5_conv1_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer3_5_conv1_weights.bin", ios::in | ios::binary);
    ifstream ifs_l3_5_conv2_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer3_5_conv2_weights.bin", ios::in | ios::binary);
    ifstream ifs_l3_5_conv3_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer3_5_conv3_weights.bin", ios::in | ios::binary);
    
    ifs_l3_5_conv1_param.read((char*)(  *resnet_layer3_5_conv1_weights), RESNET_LAYER3_CONV1_OUT_CH*RESNET_LAYER3_CONV1_IN_CH*sizeof(float));
    ifs_l3_5_conv2_param.read((char*)(***resnet_layer3_5_conv2_weights), RESNET_LAYER3_CONV2_OUT_CH*RESNET_LAYER3_CONV2_IN_CH*3*3*sizeof(float));
    ifs_l3_5_conv3_param.read((char*)(  *resnet_layer3_5_conv3_weights), RESNET_LAYER3_CONV3_OUT_CH*RESNET_LAYER3_CONV3_IN_CH*sizeof(float));
    
    ifs_l3_5_conv1_param.close();
    ifs_l3_5_conv2_param.close();
    ifs_l3_5_conv3_param.close();

    ifstream ifs_l3_5_bn1_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer3_5_bn1_params.bin", ios::in | ios::binary);
    ifstream ifs_l3_5_bn2_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer3_5_bn2_params.bin", ios::in | ios::binary);
    ifstream ifs_l3_5_bn3_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer3_5_bn3_params.bin", ios::in | ios::binary);
    
    ifs_l3_5_bn1_param.read((char*)(*resnet_layer3_5_bn1_params), 3*RESNET_LAYER3_CONV1_OUT_CH*sizeof(float));
    ifs_l3_5_bn2_param.read((char*)(*resnet_layer3_5_bn2_params), 3*RESNET_LAYER3_CONV2_OUT_CH*sizeof(float));
    ifs_l3_5_bn3_param.read((char*)(*resnet_layer3_5_bn3_params), 3*RESNET_LAYER3_CONV3_OUT_CH*sizeof(float));
    
    ifs_l3_5_bn1_param.close();
    ifs_l3_5_bn2_param.close();
    ifs_l3_5_bn3_param.close();
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
    // Layer 3.0
    //--------------------------------------------------------------------------
    convert_1x1<RESNET_LAYER3_0_CONV1_IN_CH, RESNET_LAYER3_0_CONV1_OUT_CH>
               (resnet_layer3_0_conv1_weights, fixp_layer3_0_conv1_weights);
    convert_3x3<RESNET_LAYER3_0_CONV2_IN_CH, RESNET_LAYER3_0_CONV2_OUT_CH>
               (resnet_layer3_0_conv2_weights, fixp_layer3_0_conv2_weights);
    convert_1x1<RESNET_LAYER3_0_CONV3_IN_CH, RESNET_LAYER3_0_CONV3_OUT_CH>
               (resnet_layer3_0_conv3_weights, fixp_layer3_0_conv3_weights);
    convert_1x1<RESNET_LAYER3_0_DS_IN_CH, RESNET_LAYER3_0_DS_OUT_CH>
               (resnet_layer3_0_downsample_0_weights, fixp_layer3_0_downsample_0_weights);

    convert_bn<RESNET_LAYER3_0_CONV1_OUT_CH>(resnet_layer3_0_bn1_params, fixp_layer3_0_bn1_params);
    convert_bn<RESNET_LAYER3_0_CONV2_OUT_CH>(resnet_layer3_0_bn2_params, fixp_layer3_0_bn2_params);
    convert_bn<RESNET_LAYER3_0_CONV3_OUT_CH>(resnet_layer3_0_bn3_params, fixp_layer3_0_bn3_params);
    convert_bn<RESNET_LAYER3_0_DS_OUT_CH>(resnet_layer3_0_downsample_1_params, fixp_layer3_0_downsample_1_params);

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

    //--------------------------------------------------------------------------
    // Layer 3.2
    //--------------------------------------------------------------------------
    convert_1x1<RESNET_LAYER3_CONV1_IN_CH, RESNET_LAYER3_CONV1_OUT_CH>
               (resnet_layer3_2_conv1_weights, fixp_layer3_2_conv1_weights);
    convert_3x3<RESNET_LAYER3_CONV2_IN_CH, RESNET_LAYER3_CONV2_OUT_CH>
               (resnet_layer3_2_conv2_weights, fixp_layer3_2_conv2_weights);
    convert_1x1<RESNET_LAYER3_CONV3_IN_CH, RESNET_LAYER3_CONV3_OUT_CH>
               (resnet_layer3_2_conv3_weights, fixp_layer3_2_conv3_weights);
    
    convert_bn<RESNET_LAYER3_CONV1_OUT_CH>(resnet_layer3_2_bn1_params, fixp_layer3_2_bn1_params);
    convert_bn<RESNET_LAYER3_CONV2_OUT_CH>(resnet_layer3_2_bn2_params, fixp_layer3_2_bn2_params);
    convert_bn<RESNET_LAYER3_CONV3_OUT_CH>(resnet_layer3_2_bn3_params, fixp_layer3_2_bn3_params);

    //--------------------------------------------------------------------------
    // Layer 3.3
    //--------------------------------------------------------------------------
    convert_1x1<RESNET_LAYER3_CONV1_IN_CH, RESNET_LAYER3_CONV1_OUT_CH>
               (resnet_layer3_3_conv1_weights, fixp_layer3_3_conv1_weights);
    convert_3x3<RESNET_LAYER3_CONV2_IN_CH, RESNET_LAYER3_CONV2_OUT_CH>
               (resnet_layer3_3_conv2_weights, fixp_layer3_3_conv2_weights);
    convert_1x1<RESNET_LAYER3_CONV3_IN_CH, RESNET_LAYER3_CONV3_OUT_CH>
               (resnet_layer3_3_conv3_weights, fixp_layer3_3_conv3_weights);
    
    convert_bn<RESNET_LAYER3_CONV1_OUT_CH>(resnet_layer3_3_bn1_params, fixp_layer3_3_bn1_params);
    convert_bn<RESNET_LAYER3_CONV2_OUT_CH>(resnet_layer3_3_bn2_params, fixp_layer3_3_bn2_params);
    convert_bn<RESNET_LAYER3_CONV3_OUT_CH>(resnet_layer3_3_bn3_params, fixp_layer3_3_bn3_params);
    
    //--------------------------------------------------------------------------
    // Layer 3.4
    //--------------------------------------------------------------------------
    convert_1x1<RESNET_LAYER3_CONV1_IN_CH, RESNET_LAYER3_CONV1_OUT_CH>
               (resnet_layer3_4_conv1_weights, fixp_layer3_4_conv1_weights);
    convert_3x3<RESNET_LAYER3_CONV2_IN_CH, RESNET_LAYER3_CONV2_OUT_CH>
               (resnet_layer3_4_conv2_weights, fixp_layer3_4_conv2_weights);
    convert_1x1<RESNET_LAYER3_CONV3_IN_CH, RESNET_LAYER3_CONV3_OUT_CH>
               (resnet_layer3_4_conv3_weights, fixp_layer3_4_conv3_weights);
    
    convert_bn<RESNET_LAYER3_CONV1_OUT_CH>(resnet_layer3_4_bn1_params, fixp_layer3_4_bn1_params);
    convert_bn<RESNET_LAYER3_CONV2_OUT_CH>(resnet_layer3_4_bn2_params, fixp_layer3_4_bn2_params);
    convert_bn<RESNET_LAYER3_CONV3_OUT_CH>(resnet_layer3_4_bn3_params, fixp_layer3_4_bn3_params);
    
    //--------------------------------------------------------------------------
    // Layer 3.5
    //--------------------------------------------------------------------------
    convert_1x1<RESNET_LAYER3_CONV1_IN_CH, RESNET_LAYER3_CONV1_OUT_CH>
               (resnet_layer3_5_conv1_weights, fixp_layer3_5_conv1_weights);
    convert_3x3<RESNET_LAYER3_CONV2_IN_CH, RESNET_LAYER3_CONV2_OUT_CH>
               (resnet_layer3_5_conv2_weights, fixp_layer3_5_conv2_weights);
    convert_1x1<RESNET_LAYER3_CONV3_IN_CH, RESNET_LAYER3_CONV3_OUT_CH>
               (resnet_layer3_5_conv3_weights, fixp_layer3_5_conv3_weights);
    
    convert_bn<RESNET_LAYER3_CONV1_OUT_CH>(resnet_layer3_5_bn1_params, fixp_layer3_5_bn1_params);
    convert_bn<RESNET_LAYER3_CONV2_OUT_CH>(resnet_layer3_5_bn2_params, fixp_layer3_5_bn2_params);
    convert_bn<RESNET_LAYER3_CONV3_OUT_CH>(resnet_layer3_5_bn3_params, fixp_layer3_5_bn3_params);
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
    
    // // Read input image
    // ifstream ifs_input_img("/usr/scratch/rsamanta9/bin/resnet_backbone/qdtrack_image0.bin", ios::in | ios::binary);
    // ifs_input_img.read((char*)(**input_image), 3*736*1280*sizeof(float));
    // ifs_input_img.close();

    // // Convert input image data to fixed point
    // for(int c = 0; c < RESNET_LAYER0_CONV1_IN_CH; c++)
    // {
    //     for(int h = 0; h < RESNET_LAYER0_IN_FM_HEIGHT; h++)
    //     {
    //         for(int w = 0; w < RESNET_LAYER0_IN_FM_WIDTH; w++)
    //         {
    //             resnet_layer0_input_fm[c][h][w] = (fm_t) input_image[c][h][w];
    //         }
    //     }
    // }
    
#ifdef TEST_COMPLETE_MODEL // {
    //----------------------------------------------------------------------
    // ResNet50 Top-level wrapper 
    //----------------------------------------------------------------------
    test_resnet_top_3 ( 
              resnet_layer2_output_fm,

              resnet_layer3_input_fm,
              fixp_layer3_0_conv1_weights,         fixp_layer3_0_bn1_params,
              fixp_layer3_0_conv2_weights,         fixp_layer3_0_bn2_params,
              fixp_layer3_0_conv3_weights,         fixp_layer3_0_bn3_params,
              fixp_layer3_0_downsample_0_weights,  fixp_layer3_0_downsample_1_params,
              fixp_layer3_1_conv1_weights,         fixp_layer3_1_bn1_params,
              fixp_layer3_1_conv2_weights,         fixp_layer3_1_bn2_params,
              fixp_layer3_1_conv3_weights,         fixp_layer3_1_bn3_params,
              fixp_layer3_2_conv1_weights,         fixp_layer3_2_bn1_params,
              fixp_layer3_2_conv2_weights,         fixp_layer3_2_bn2_params,
              fixp_layer3_2_conv3_weights,         fixp_layer3_2_bn3_params,
              fixp_layer3_3_conv1_weights,         fixp_layer3_3_bn1_params,
              fixp_layer3_3_conv2_weights,         fixp_layer3_3_bn2_params,
              fixp_layer3_3_conv3_weights,         fixp_layer3_3_bn3_params,
              fixp_layer3_4_conv1_weights,         fixp_layer3_4_bn1_params,
              fixp_layer3_4_conv2_weights,         fixp_layer3_4_bn2_params,
              fixp_layer3_4_conv3_weights,         fixp_layer3_4_bn3_params,
              fixp_layer3_5_conv1_weights,         fixp_layer3_5_bn1_params,
              fixp_layer3_5_conv2_weights,         fixp_layer3_5_bn2_params,
              fixp_layer3_5_conv3_weights,         fixp_layer3_5_bn3_params,
              resnet_layer3_output_fm,
              
              resnet_layer4_input_fm
    );


#endif // }

    return 0;
}
