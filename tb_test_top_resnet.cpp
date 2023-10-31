#include <iostream>
#include <fstream>
#include <cmath>
#include "hls_stream.h"
#include "qdtrack_resnet.h"

using namespace std;

float  input_image[RESNET_LAYER0_CONV1_IN_CH][RESNET_LAYER0_IN_FM_HEIGHT][RESNET_LAYER0_IN_FM_WIDTH];

fm_t   resnet_layer0_input_fm [RESNET_LAYER0_CONV1_IN_CH][RESNET_LAYER0_IN_FM_HEIGHT][RESNET_LAYER0_IN_FM_WIDTH]    = {0};
fm_t   resnet_layer0_output_fm[RESNET_LAYER0_CONV1_OUT_CH][RESNET_LAYER0_OUT_FM_HEIGHT][RESNET_LAYER0_OUT_FM_WIDTH] = {0};

fm_t   resnet_layer1_input_fm [RESNET_LAYER1_0_DS_OUT_CH][RESNET_LAYER1_0_FM_HEIGHT][RESNET_LAYER1_0_FM_WIDTH] = {0};
fm_t   resnet_layer1_output_fm[RESNET_LAYER1_0_DS_OUT_CH][RESNET_LAYER1_0_FM_HEIGHT][RESNET_LAYER1_0_FM_WIDTH] = {0};

fm_t   resnet_layer2_input_fm [RESNET_LAYER2_0_DS_OUT_CH][RESNET_LAYER2_0_FM_HEIGHT][RESNET_LAYER2_0_FM_WIDTH] = {0};
fm_t   resnet_layer2_output_fm[RESNET_LAYER2_0_DS_OUT_CH][RESNET_LAYER2_0_FM_HEIGHT][RESNET_LAYER2_0_FM_WIDTH] = {0};

fm_t   resnet_layer3_input_fm [RESNET_LAYER3_0_DS_OUT_CH][RESNET_LAYER3_0_FM_HEIGHT][RESNET_LAYER3_0_FM_WIDTH] = {0};
fm_t   resnet_layer3_output_fm[RESNET_LAYER3_0_DS_OUT_CH][RESNET_LAYER3_0_FM_HEIGHT][RESNET_LAYER3_0_FM_WIDTH] = {0};

fm_t   resnet_layer4_input_fm [RESNET_LAYER4_0_DS_OUT_CH][RESNET_LAYER4_0_FM_HEIGHT][RESNET_LAYER4_0_FM_WIDTH] = {0};
fm_t   resnet_layer4_output_fm[RESNET_LAYER4_0_DS_OUT_CH][RESNET_LAYER4_0_FM_HEIGHT][RESNET_LAYER4_0_FM_WIDTH] = {0};

//--------------------------------------------------------------------------
// Layer 0
//--------------------------------------------------------------------------
float   resnet_layer0_conv1_weights[RESNET_LAYER0_CONV1_OUT_CH][RESNET_LAYER0_CONV1_IN_CH][7][7];
float   resnet_layer0_bn1_params[3][RESNET_LAYER0_CONV1_OUT_CH];

wt_t   fixp_layer0_conv1_weights[RESNET_LAYER0_CONV1_OUT_CH][RESNET_LAYER0_CONV1_IN_CH][7][7];
wt_t   fixp_layer0_bn1_params[3][RESNET_LAYER0_CONV1_OUT_CH];

//--------------------------------------------------------------------------
// Layer 1.0
//--------------------------------------------------------------------------
float   resnet_layer1_0_downsample_0_weights[RESNET_LAYER1_0_DS_OUT_CH][RESNET_LAYER1_0_DS_IN_CH];
float   resnet_layer1_0_downsample_1_params[3][RESNET_LAYER1_0_DS_OUT_CH];
float   resnet_layer1_0_conv1_weights[RESNET_LAYER1_0_CONV1_OUT_CH][RESNET_LAYER1_0_CONV1_IN_CH];
float   resnet_layer1_0_bn1_params[3][RESNET_LAYER1_0_CONV1_OUT_CH];
float   resnet_layer1_0_conv2_weights[RESNET_LAYER1_0_CONV2_OUT_CH][RESNET_LAYER1_0_CONV2_IN_CH][3][3];
float   resnet_layer1_0_bn2_params[3][RESNET_LAYER1_0_CONV2_OUT_CH];
float   resnet_layer1_0_conv3_weights[RESNET_LAYER1_0_CONV3_OUT_CH][RESNET_LAYER1_0_CONV3_IN_CH];
float   resnet_layer1_0_bn3_params[3][RESNET_LAYER1_0_CONV3_OUT_CH];

wt_t    fixp_layer1_0_downsample_0_weights[RESNET_LAYER1_0_DS_OUT_CH][RESNET_LAYER1_0_DS_IN_CH];
wt_t    fixp_layer1_0_downsample_1_params[3][RESNET_LAYER1_0_DS_OUT_CH];
wt_t    fixp_layer1_0_conv1_weights[RESNET_LAYER1_0_CONV1_OUT_CH][RESNET_LAYER1_0_CONV1_IN_CH];
wt_t    fixp_layer1_0_bn1_params[3][RESNET_LAYER1_0_CONV1_OUT_CH];
wt_t    fixp_layer1_0_conv2_weights[RESNET_LAYER1_0_CONV2_OUT_CH][RESNET_LAYER1_0_CONV2_IN_CH][3][3];
wt_t    fixp_layer1_0_bn2_params[3][RESNET_LAYER1_0_CONV2_OUT_CH];
wt_t    fixp_layer1_0_conv3_weights[RESNET_LAYER1_0_CONV3_OUT_CH][RESNET_LAYER1_0_CONV3_IN_CH];
wt_t    fixp_layer1_0_bn3_params[3][RESNET_LAYER1_0_CONV3_OUT_CH];

//--------------------------------------------------------------------------
// Layer 1.1
//--------------------------------------------------------------------------
float   resnet_layer1_1_conv1_weights[RESNET_LAYER1_CONV1_OUT_CH][RESNET_LAYER1_CONV1_IN_CH];
float   resnet_layer1_1_bn1_params[3][RESNET_LAYER1_CONV1_OUT_CH];
float   resnet_layer1_1_conv2_weights[RESNET_LAYER1_CONV2_OUT_CH][RESNET_LAYER1_CONV2_IN_CH][3][3];
float   resnet_layer1_1_bn2_params[3][RESNET_LAYER1_CONV2_OUT_CH];
float   resnet_layer1_1_conv3_weights[RESNET_LAYER1_CONV3_OUT_CH][RESNET_LAYER1_CONV3_IN_CH];
float   resnet_layer1_1_bn3_params[3][RESNET_LAYER1_CONV3_OUT_CH];

wt_t    fixp_layer1_1_conv1_weights[RESNET_LAYER1_CONV1_OUT_CH][RESNET_LAYER1_CONV1_IN_CH];
wt_t    fixp_layer1_1_bn1_params[3][RESNET_LAYER1_CONV1_OUT_CH];
wt_t    fixp_layer1_1_conv2_weights[RESNET_LAYER1_CONV2_OUT_CH][RESNET_LAYER1_CONV2_IN_CH][3][3];
wt_t    fixp_layer1_1_bn2_params[3][RESNET_LAYER1_CONV2_OUT_CH];
wt_t    fixp_layer1_1_conv3_weights[RESNET_LAYER1_CONV3_OUT_CH][RESNET_LAYER1_CONV3_IN_CH];
wt_t    fixp_layer1_1_bn3_params[3][RESNET_LAYER1_CONV3_OUT_CH];

//--------------------------------------------------------------------------
// Layer 1.2
//--------------------------------------------------------------------------
float   resnet_layer1_2_conv1_weights[RESNET_LAYER1_CONV1_OUT_CH][RESNET_LAYER1_CONV1_IN_CH];
float   resnet_layer1_2_bn1_params[3][RESNET_LAYER1_CONV1_OUT_CH];
float   resnet_layer1_2_conv2_weights[RESNET_LAYER1_CONV2_OUT_CH][RESNET_LAYER1_CONV2_IN_CH][3][3];
float   resnet_layer1_2_bn2_params[3][RESNET_LAYER1_CONV2_OUT_CH];
float   resnet_layer1_2_conv3_weights[RESNET_LAYER1_CONV3_OUT_CH][RESNET_LAYER1_CONV3_IN_CH];
float   resnet_layer1_2_bn3_params[3][RESNET_LAYER1_CONV3_OUT_CH];

wt_t    fixp_layer1_2_conv1_weights[RESNET_LAYER1_CONV1_OUT_CH][RESNET_LAYER1_CONV1_IN_CH];
wt_t    fixp_layer1_2_bn1_params[3][RESNET_LAYER1_CONV1_OUT_CH];
wt_t    fixp_layer1_2_conv2_weights[RESNET_LAYER1_CONV2_OUT_CH][RESNET_LAYER1_CONV2_IN_CH][3][3];
wt_t    fixp_layer1_2_bn2_params[3][RESNET_LAYER1_CONV2_OUT_CH];
wt_t    fixp_layer1_2_conv3_weights[RESNET_LAYER1_CONV3_OUT_CH][RESNET_LAYER1_CONV3_IN_CH];
wt_t    fixp_layer1_2_bn3_params[3][RESNET_LAYER1_CONV3_OUT_CH];

//--------------------------------------------------------------------------
// Layer 2.0
//--------------------------------------------------------------------------
float   resnet_layer2_0_downsample_0_weights[RESNET_LAYER2_0_DS_OUT_CH][RESNET_LAYER2_0_DS_IN_CH];
float   resnet_layer2_0_downsample_1_params[3][RESNET_LAYER2_0_DS_OUT_CH];
float   resnet_layer2_0_conv1_weights[RESNET_LAYER2_0_CONV1_OUT_CH][RESNET_LAYER2_0_CONV1_IN_CH];
float   resnet_layer2_0_bn1_params[3][RESNET_LAYER2_0_CONV1_OUT_CH];
float   resnet_layer2_0_conv2_weights[RESNET_LAYER2_0_CONV2_OUT_CH][RESNET_LAYER2_0_CONV2_IN_CH][3][3];
float   resnet_layer2_0_bn2_params[3][RESNET_LAYER2_0_CONV2_OUT_CH];
float   resnet_layer2_0_conv3_weights[RESNET_LAYER2_0_CONV3_OUT_CH][RESNET_LAYER2_0_CONV3_IN_CH];
float   resnet_layer2_0_bn3_params[3][RESNET_LAYER2_0_CONV3_OUT_CH];

wt_t    fixp_layer2_0_downsample_0_weights[RESNET_LAYER2_0_DS_OUT_CH][RESNET_LAYER2_0_DS_IN_CH];
wt_t    fixp_layer2_0_downsample_1_params[3][RESNET_LAYER2_0_DS_OUT_CH];
wt_t    fixp_layer2_0_conv1_weights[RESNET_LAYER2_0_CONV1_OUT_CH][RESNET_LAYER2_0_CONV1_IN_CH];
wt_t    fixp_layer2_0_bn1_params[3][RESNET_LAYER2_0_CONV1_OUT_CH];
wt_t    fixp_layer2_0_conv2_weights[RESNET_LAYER2_0_CONV2_OUT_CH][RESNET_LAYER2_0_CONV2_IN_CH][3][3];
wt_t    fixp_layer2_0_bn2_params[3][RESNET_LAYER2_0_CONV2_OUT_CH];
wt_t    fixp_layer2_0_conv3_weights[RESNET_LAYER2_0_CONV3_OUT_CH][RESNET_LAYER2_0_CONV3_IN_CH];
wt_t    fixp_layer2_0_bn3_params[3][RESNET_LAYER2_0_CONV3_OUT_CH];

//--------------------------------------------------------------------------
// Layer 2.1
//--------------------------------------------------------------------------
float   resnet_layer2_1_conv1_weights[RESNET_LAYER2_CONV1_OUT_CH][RESNET_LAYER2_CONV1_IN_CH];
float   resnet_layer2_1_bn1_params[3][RESNET_LAYER2_CONV1_OUT_CH];
float   resnet_layer2_1_conv2_weights[RESNET_LAYER2_CONV2_OUT_CH][RESNET_LAYER2_CONV2_IN_CH][3][3];
float   resnet_layer2_1_bn2_params[3][RESNET_LAYER2_CONV2_OUT_CH];
float   resnet_layer2_1_conv3_weights[RESNET_LAYER2_CONV3_OUT_CH][RESNET_LAYER2_CONV3_IN_CH];
float   resnet_layer2_1_bn3_params[3][RESNET_LAYER2_CONV3_OUT_CH];

wt_t    fixp_layer2_1_conv1_weights[RESNET_LAYER2_CONV1_OUT_CH][RESNET_LAYER2_CONV1_IN_CH];
wt_t    fixp_layer2_1_bn1_params[3][RESNET_LAYER2_CONV1_OUT_CH];
wt_t    fixp_layer2_1_conv2_weights[RESNET_LAYER2_CONV2_OUT_CH][RESNET_LAYER2_CONV2_IN_CH][3][3];
wt_t    fixp_layer2_1_bn2_params[3][RESNET_LAYER2_CONV2_OUT_CH];
wt_t    fixp_layer2_1_conv3_weights[RESNET_LAYER2_CONV3_OUT_CH][RESNET_LAYER2_CONV3_IN_CH];
wt_t    fixp_layer2_1_bn3_params[3][RESNET_LAYER2_CONV3_OUT_CH];

//--------------------------------------------------------------------------
// Layer 2.2
//--------------------------------------------------------------------------
float   resnet_layer2_2_conv1_weights[RESNET_LAYER2_CONV1_OUT_CH][RESNET_LAYER2_CONV1_IN_CH];
float   resnet_layer2_2_bn1_params[3][RESNET_LAYER2_CONV1_OUT_CH];
float   resnet_layer2_2_conv2_weights[RESNET_LAYER2_CONV2_OUT_CH][RESNET_LAYER2_CONV2_IN_CH][3][3];
float   resnet_layer2_2_bn2_params[3][RESNET_LAYER2_CONV2_OUT_CH];
float   resnet_layer2_2_conv3_weights[RESNET_LAYER2_CONV3_OUT_CH][RESNET_LAYER2_CONV3_IN_CH];
float   resnet_layer2_2_bn3_params[3][RESNET_LAYER2_CONV3_OUT_CH];

wt_t    fixp_layer2_2_conv1_weights[RESNET_LAYER2_CONV1_OUT_CH][RESNET_LAYER2_CONV1_IN_CH];
wt_t    fixp_layer2_2_bn1_params[3][RESNET_LAYER2_CONV1_OUT_CH];
wt_t    fixp_layer2_2_conv2_weights[RESNET_LAYER2_CONV2_OUT_CH][RESNET_LAYER2_CONV2_IN_CH][3][3];
wt_t    fixp_layer2_2_bn2_params[3][RESNET_LAYER2_CONV2_OUT_CH];
wt_t    fixp_layer2_2_conv3_weights[RESNET_LAYER2_CONV3_OUT_CH][RESNET_LAYER2_CONV3_IN_CH];
wt_t    fixp_layer2_2_bn3_params[3][RESNET_LAYER2_CONV3_OUT_CH];

//--------------------------------------------------------------------------
// Layer 2.3
//--------------------------------------------------------------------------
float   resnet_layer2_3_conv1_weights[RESNET_LAYER2_CONV1_OUT_CH][RESNET_LAYER2_CONV1_IN_CH];
float   resnet_layer2_3_bn1_params[3][RESNET_LAYER2_CONV1_OUT_CH];
float   resnet_layer2_3_conv2_weights[RESNET_LAYER2_CONV2_OUT_CH][RESNET_LAYER2_CONV2_IN_CH][3][3];
float   resnet_layer2_3_bn2_params[3][RESNET_LAYER2_CONV2_OUT_CH];
float   resnet_layer2_3_conv3_weights[RESNET_LAYER2_CONV3_OUT_CH][RESNET_LAYER2_CONV3_IN_CH];
float   resnet_layer2_3_bn3_params[3][RESNET_LAYER2_CONV3_OUT_CH];

wt_t    fixp_layer2_3_conv1_weights[RESNET_LAYER2_CONV1_OUT_CH][RESNET_LAYER2_CONV1_IN_CH];
wt_t    fixp_layer2_3_bn1_params[3][RESNET_LAYER2_CONV1_OUT_CH];
wt_t    fixp_layer2_3_conv2_weights[RESNET_LAYER2_CONV2_OUT_CH][RESNET_LAYER2_CONV2_IN_CH][3][3];
wt_t    fixp_layer2_3_bn2_params[3][RESNET_LAYER2_CONV2_OUT_CH];
wt_t    fixp_layer2_3_conv3_weights[RESNET_LAYER2_CONV3_OUT_CH][RESNET_LAYER2_CONV3_IN_CH];
wt_t    fixp_layer2_3_bn3_params[3][RESNET_LAYER2_CONV3_OUT_CH];

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
// Layer 4.0
//--------------------------------------------------------------------------
float   resnet_layer4_0_downsample_0_weights[RESNET_LAYER4_0_DS_OUT_CH][RESNET_LAYER4_0_DS_IN_CH];
float   resnet_layer4_0_downsample_1_params[3][RESNET_LAYER4_0_DS_OUT_CH];
float   resnet_layer4_0_conv1_weights[RESNET_LAYER4_0_CONV1_OUT_CH][RESNET_LAYER4_0_CONV1_IN_CH];
float   resnet_layer4_0_bn1_params[3][RESNET_LAYER4_0_CONV1_OUT_CH];
float   resnet_layer4_0_conv2_weights[RESNET_LAYER4_0_CONV2_OUT_CH][RESNET_LAYER4_0_CONV2_IN_CH][3][3];
float   resnet_layer4_0_bn2_params[3][RESNET_LAYER4_0_CONV2_OUT_CH];
float   resnet_layer4_0_conv3_weights[RESNET_LAYER4_0_CONV3_OUT_CH][RESNET_LAYER4_0_CONV3_IN_CH];
float   resnet_layer4_0_bn3_params[3][RESNET_LAYER4_0_CONV3_OUT_CH];

wt_t    fixp_layer4_0_downsample_0_weights[RESNET_LAYER4_0_DS_OUT_CH][RESNET_LAYER4_0_DS_IN_CH];
wt_t    fixp_layer4_0_downsample_1_params[3][RESNET_LAYER4_0_DS_OUT_CH];
wt_t    fixp_layer4_0_conv1_weights[RESNET_LAYER4_0_CONV1_OUT_CH][RESNET_LAYER4_0_CONV1_IN_CH];
wt_t    fixp_layer4_0_bn1_params[3][RESNET_LAYER4_0_CONV1_OUT_CH];
wt_t    fixp_layer4_0_conv2_weights[RESNET_LAYER4_0_CONV2_OUT_CH][RESNET_LAYER4_0_CONV2_IN_CH][3][3];
wt_t    fixp_layer4_0_bn2_params[3][RESNET_LAYER4_0_CONV2_OUT_CH];
wt_t    fixp_layer4_0_conv3_weights[RESNET_LAYER4_0_CONV3_OUT_CH][RESNET_LAYER4_0_CONV3_IN_CH];
wt_t    fixp_layer4_0_bn3_params[3][RESNET_LAYER4_0_CONV3_OUT_CH];

//--------------------------------------------------------------------------
// Layer 4.1
//--------------------------------------------------------------------------
float   resnet_layer4_1_conv1_weights[RESNET_LAYER4_CONV1_OUT_CH][RESNET_LAYER4_CONV1_IN_CH];
float   resnet_layer4_1_bn1_params[3][RESNET_LAYER4_CONV1_OUT_CH];
float   resnet_layer4_1_conv2_weights[RESNET_LAYER4_CONV2_OUT_CH][RESNET_LAYER4_CONV2_IN_CH][3][3];
float   resnet_layer4_1_bn2_params[3][RESNET_LAYER4_CONV2_OUT_CH];
float   resnet_layer4_1_conv3_weights[RESNET_LAYER4_CONV3_OUT_CH][RESNET_LAYER4_CONV3_IN_CH];
float   resnet_layer4_1_bn3_params[3][RESNET_LAYER4_CONV3_OUT_CH];

wt_t    fixp_layer4_1_conv1_weights[RESNET_LAYER4_CONV1_OUT_CH][RESNET_LAYER4_CONV1_IN_CH];
wt_t    fixp_layer4_1_bn1_params[3][RESNET_LAYER4_CONV1_OUT_CH];
wt_t    fixp_layer4_1_conv2_weights[RESNET_LAYER4_CONV2_OUT_CH][RESNET_LAYER4_CONV2_IN_CH][3][3];
wt_t    fixp_layer4_1_bn2_params[3][RESNET_LAYER4_CONV2_OUT_CH];
wt_t    fixp_layer4_1_conv3_weights[RESNET_LAYER4_CONV3_OUT_CH][RESNET_LAYER4_CONV3_IN_CH];
wt_t    fixp_layer4_1_bn3_params[3][RESNET_LAYER4_CONV3_OUT_CH];

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
float   golden_maxpool_out[RESNET_LAYER0_CONV1_OUT_CH][RESNET_LAYER0_OUT_FM_HEIGHT][RESNET_LAYER0_OUT_FM_WIDTH];
float   golden_layer1_2_bn3_relu_out[RESNET_LAYER1_CONV3_OUT_CH][RESNET_LAYER1_FM_HEIGHT][RESNET_LAYER1_FM_WIDTH];
float   golden_layer2_3_bn3_relu_out[RESNET_LAYER2_CONV3_OUT_CH][RESNET_LAYER2_FM_HEIGHT][RESNET_LAYER2_FM_WIDTH];
float   golden_layer3_5_bn3_relu_out[RESNET_LAYER3_CONV3_OUT_CH][RESNET_LAYER3_FM_HEIGHT][RESNET_LAYER3_FM_WIDTH];
float   golden_layer4_2_bn3_relu_out[RESNET_LAYER4_CONV3_OUT_CH][RESNET_LAYER4_FM_HEIGHT][RESNET_LAYER4_FM_WIDTH];


void resnet_load_weights()
{
    //--------------------------------------------------------------------------
    // Layer 0
    //--------------------------------------------------------------------------
    ifstream ifs_l0_conv1_param("/usr/scratch/rsamanta9/bin/resnet_backbone/conv1_weights.bin", ios::in | ios::binary);
    ifs_l0_conv1_param.read((char*)(***resnet_layer0_conv1_weights), RESNET_LAYER0_CONV1_OUT_CH*RESNET_LAYER0_CONV1_IN_CH*7*7*sizeof(float));
    ifs_l0_conv1_param.close();

    //cout << resnet_layer0_conv1_weights[0][0][1][1] << std::endl;

    ifstream ifs_l0_bn1_param("/usr/scratch/rsamanta9/bin/resnet_backbone/bn1_params.bin", ios::in | ios::binary);
    ifs_l0_bn1_param.read((char*)(*resnet_layer0_bn1_params), 3*RESNET_LAYER0_CONV1_OUT_CH*sizeof(float));
    ifs_l0_bn1_param.close();

    //--------------------------------------------------------------------------
    // Layer 1.0
    //--------------------------------------------------------------------------
    ifstream ifs_l1_0_conv1_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer1_0_conv1_weights.bin", ios::in | ios::binary);
    ifstream ifs_l1_0_conv2_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer1_0_conv2_weights.bin", ios::in | ios::binary);
    ifstream ifs_l1_0_conv3_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer1_0_conv3_weights.bin", ios::in | ios::binary);
    
    ifs_l1_0_conv1_param.read((char*)(  *resnet_layer1_0_conv1_weights), RESNET_LAYER1_0_CONV1_OUT_CH*RESNET_LAYER1_0_CONV1_IN_CH*sizeof(float));
    ifs_l1_0_conv2_param.read((char*)(***resnet_layer1_0_conv2_weights), RESNET_LAYER1_0_CONV2_OUT_CH*RESNET_LAYER1_0_CONV2_IN_CH*3*3*sizeof(float));
    ifs_l1_0_conv3_param.read((char*)(  *resnet_layer1_0_conv3_weights), RESNET_LAYER1_0_CONV3_OUT_CH*RESNET_LAYER1_0_CONV3_IN_CH*sizeof(float));

    ifs_l1_0_conv1_param.close();
    ifs_l1_0_conv2_param.close();
    ifs_l1_0_conv3_param.close();

    ifstream ifs_l1_0_bn1_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer1_0_bn1_params.bin", ios::in | ios::binary);
    ifstream ifs_l1_0_bn2_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer1_0_bn2_params.bin", ios::in | ios::binary);
    ifstream ifs_l1_0_bn3_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer1_0_bn3_params.bin", ios::in | ios::binary);

    ifs_l1_0_bn1_param.read((char*)(*resnet_layer1_0_bn1_params), 3*RESNET_LAYER1_0_CONV1_OUT_CH*sizeof(float));
    ifs_l1_0_bn2_param.read((char*)(*resnet_layer1_0_bn2_params), 3*RESNET_LAYER1_0_CONV2_OUT_CH*sizeof(float));
    ifs_l1_0_bn3_param.read((char*)(*resnet_layer1_0_bn3_params), 3*RESNET_LAYER1_0_CONV3_OUT_CH*sizeof(float));

    ifs_l1_0_bn1_param.close();
    ifs_l1_0_bn2_param.close();
    ifs_l1_0_bn3_param.close();
    
    ifstream ifs_l1_0_downsample_0_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer1_0_downsample_0_weights.bin", ios::in | ios::binary);
    ifstream ifs_l1_0_downsample_1_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer1_0_downsample_1_params.bin", ios::in | ios::binary);
    
    ifs_l1_0_downsample_0_param.read((char*)(*resnet_layer1_0_downsample_0_weights), RESNET_LAYER1_0_DS_OUT_CH*RESNET_LAYER1_0_DS_IN_CH*sizeof(float));
    ifs_l1_0_downsample_1_param.read((char*)(*resnet_layer1_0_downsample_1_params), 3*RESNET_LAYER1_0_DS_OUT_CH*sizeof(float));
    
    ifs_l1_0_downsample_0_param.close();
    ifs_l1_0_downsample_1_param.close();

    //--------------------------------------------------------------------------
    // Layer 1.1
    //--------------------------------------------------------------------------
    ifstream ifs_l1_1_conv1_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer1_1_conv1_weights.bin", ios::in | ios::binary);
    ifstream ifs_l1_1_conv2_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer1_1_conv2_weights.bin", ios::in | ios::binary);
    ifstream ifs_l1_1_conv3_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer1_1_conv3_weights.bin", ios::in | ios::binary);
    
    ifs_l1_1_conv1_param.read((char*)(  *resnet_layer1_1_conv1_weights), RESNET_LAYER1_CONV1_OUT_CH*RESNET_LAYER1_CONV1_IN_CH*sizeof(float));
    ifs_l1_1_conv2_param.read((char*)(***resnet_layer1_1_conv2_weights), RESNET_LAYER1_CONV2_OUT_CH*RESNET_LAYER1_CONV2_IN_CH*3*3*sizeof(float));
    ifs_l1_1_conv3_param.read((char*)(  *resnet_layer1_1_conv3_weights), RESNET_LAYER1_CONV3_OUT_CH*RESNET_LAYER1_CONV3_IN_CH*sizeof(float));
    
    ifs_l1_1_conv1_param.close();
    ifs_l1_1_conv2_param.close();
    ifs_l1_1_conv3_param.close();
    
    ifstream ifs_l1_1_bn1_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer1_1_bn1_params.bin", ios::in | ios::binary);
    ifstream ifs_l1_1_bn2_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer1_1_bn2_params.bin", ios::in | ios::binary);
    ifstream ifs_l1_1_bn3_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer1_1_bn3_params.bin", ios::in | ios::binary);

    ifs_l1_1_bn1_param.read((char*)(*resnet_layer1_1_bn1_params), 3*RESNET_LAYER1_CONV1_OUT_CH*sizeof(float));
    ifs_l1_1_bn2_param.read((char*)(*resnet_layer1_1_bn2_params), 3*RESNET_LAYER1_CONV2_OUT_CH*sizeof(float));
    ifs_l1_1_bn3_param.read((char*)(*resnet_layer1_1_bn3_params), 3*RESNET_LAYER1_CONV3_OUT_CH*sizeof(float));

    ifs_l1_1_bn1_param.close();
    ifs_l1_1_bn2_param.close();
    ifs_l1_1_bn3_param.close();

    //--------------------------------------------------------------------------
    // Layer 1.2
    //--------------------------------------------------------------------------
    ifstream ifs_l1_2_conv1_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer1_2_conv1_weights.bin", ios::in | ios::binary);
    ifstream ifs_l1_2_conv2_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer1_2_conv2_weights.bin", ios::in | ios::binary);
    ifstream ifs_l1_2_conv3_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer1_2_conv3_weights.bin", ios::in | ios::binary);
    
    ifs_l1_2_conv1_param.read((char*)(  *resnet_layer1_2_conv1_weights), RESNET_LAYER1_CONV1_OUT_CH*RESNET_LAYER1_CONV1_IN_CH*sizeof(float));
    ifs_l1_2_conv2_param.read((char*)(***resnet_layer1_2_conv2_weights), RESNET_LAYER1_CONV2_OUT_CH*RESNET_LAYER1_CONV2_IN_CH*3*3*sizeof(float));
    ifs_l1_2_conv3_param.read((char*)(  *resnet_layer1_2_conv3_weights), RESNET_LAYER1_CONV3_OUT_CH*RESNET_LAYER1_CONV3_IN_CH*sizeof(float));

    ifs_l1_2_conv1_param.close();
    ifs_l1_2_conv2_param.close();
    ifs_l1_2_conv3_param.close();

    ifstream ifs_l1_2_bn1_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer1_2_bn1_params.bin", ios::in | ios::binary);
    ifstream ifs_l1_2_bn2_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer1_2_bn2_params.bin", ios::in | ios::binary);
    ifstream ifs_l1_2_bn3_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer1_2_bn3_params.bin", ios::in | ios::binary);
    
    ifs_l1_2_bn1_param.read((char*)(*resnet_layer1_2_bn1_params), 3*RESNET_LAYER1_CONV1_OUT_CH*sizeof(float));
    ifs_l1_2_bn2_param.read((char*)(*resnet_layer1_2_bn2_params), 3*RESNET_LAYER1_CONV2_OUT_CH*sizeof(float));
    ifs_l1_2_bn3_param.read((char*)(*resnet_layer1_2_bn3_params), 3*RESNET_LAYER1_CONV3_OUT_CH*sizeof(float));
    
    ifs_l1_2_bn1_param.close();
    ifs_l1_2_bn2_param.close();
    ifs_l1_2_bn3_param.close();
    
    //--------------------------------------------------------------------------
    // Layer 2.0
    //--------------------------------------------------------------------------
    ifstream ifs_l2_0_conv1_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer2_0_conv1_weights.bin", ios::in | ios::binary);
    ifstream ifs_l2_0_conv2_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer2_0_conv2_weights.bin", ios::in | ios::binary);
    ifstream ifs_l2_0_conv3_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer2_0_conv3_weights.bin", ios::in | ios::binary);
    
    ifs_l2_0_conv1_param.read((char*)(  *resnet_layer2_0_conv1_weights), RESNET_LAYER2_0_CONV1_OUT_CH*RESNET_LAYER2_0_CONV1_IN_CH*sizeof(float));
    ifs_l2_0_conv2_param.read((char*)(***resnet_layer2_0_conv2_weights), RESNET_LAYER2_0_CONV2_OUT_CH*RESNET_LAYER2_0_CONV2_IN_CH*3*3*sizeof(float));
    ifs_l2_0_conv3_param.read((char*)(  *resnet_layer2_0_conv3_weights), RESNET_LAYER2_0_CONV3_OUT_CH*RESNET_LAYER2_0_CONV3_IN_CH*sizeof(float));
    
    ifs_l2_0_conv1_param.close();
    ifs_l2_0_conv2_param.close();
    ifs_l2_0_conv3_param.close();
    
    ifstream ifs_l2_0_bn1_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer2_0_bn1_params.bin", ios::in | ios::binary);
    ifstream ifs_l2_0_bn2_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer2_0_bn2_params.bin", ios::in | ios::binary);
    ifstream ifs_l2_0_bn3_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer2_0_bn3_params.bin", ios::in | ios::binary);
    
    ifs_l2_0_bn1_param.read((char*)(*resnet_layer2_0_bn1_params), 3*RESNET_LAYER2_0_CONV1_OUT_CH*sizeof(float));
    ifs_l2_0_bn2_param.read((char*)(*resnet_layer2_0_bn2_params), 3*RESNET_LAYER2_0_CONV2_OUT_CH*sizeof(float));
    ifs_l2_0_bn3_param.read((char*)(*resnet_layer2_0_bn3_params), 3*RESNET_LAYER2_0_CONV3_OUT_CH*sizeof(float));
    
    ifs_l2_0_bn1_param.close();
    ifs_l2_0_bn2_param.close();
    ifs_l2_0_bn3_param.close();
    
    ifstream ifs_l2_0_downsample_0_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer2_0_downsample_0_weights.bin", ios::in | ios::binary);
    ifstream ifs_l2_0_downsample_1_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer2_0_downsample_1_params.bin", ios::in | ios::binary);
    
    ifs_l2_0_downsample_0_param.read((char*)(*resnet_layer2_0_downsample_0_weights), RESNET_LAYER2_0_DS_OUT_CH*RESNET_LAYER2_0_DS_IN_CH*sizeof(float));
    ifs_l2_0_downsample_1_param.read((char*)(*resnet_layer2_0_downsample_1_params), 3*RESNET_LAYER2_0_DS_OUT_CH*sizeof(float));
    
    ifs_l2_0_downsample_0_param.close();
    ifs_l2_0_downsample_1_param.close();

    //--------------------------------------------------------------------------
    // Layer 2.1
    //--------------------------------------------------------------------------
    ifstream ifs_l2_1_conv1_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer2_1_conv1_weights.bin", ios::in | ios::binary);
    ifstream ifs_l2_1_conv2_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer2_1_conv2_weights.bin", ios::in | ios::binary);
    ifstream ifs_l2_1_conv3_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer2_1_conv3_weights.bin", ios::in | ios::binary);
    
    ifs_l2_1_conv1_param.read((char*)(  *resnet_layer2_1_conv1_weights), RESNET_LAYER2_CONV1_OUT_CH*RESNET_LAYER2_CONV1_IN_CH*sizeof(float));
    ifs_l2_1_conv2_param.read((char*)(***resnet_layer2_1_conv2_weights), RESNET_LAYER2_CONV2_OUT_CH*RESNET_LAYER2_CONV2_IN_CH*3*3*sizeof(float));
    ifs_l2_1_conv3_param.read((char*)(  *resnet_layer2_1_conv3_weights), RESNET_LAYER2_CONV3_OUT_CH*RESNET_LAYER2_CONV3_IN_CH*sizeof(float));
    
    ifs_l2_1_conv1_param.close();
    ifs_l2_1_conv2_param.close();
    ifs_l2_1_conv3_param.close();
    
    ifstream ifs_l2_1_bn1_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer2_1_bn1_params.bin", ios::in | ios::binary);
    ifstream ifs_l2_1_bn2_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer2_1_bn2_params.bin", ios::in | ios::binary);
    ifstream ifs_l2_1_bn3_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer2_1_bn3_params.bin", ios::in | ios::binary);
    
    ifs_l2_1_bn1_param.read((char*)(*resnet_layer2_1_bn1_params), 3*RESNET_LAYER2_CONV1_OUT_CH*sizeof(float));
    ifs_l2_1_bn2_param.read((char*)(*resnet_layer2_1_bn2_params), 3*RESNET_LAYER2_CONV2_OUT_CH*sizeof(float));
    ifs_l2_1_bn3_param.read((char*)(*resnet_layer2_1_bn3_params), 3*RESNET_LAYER2_CONV3_OUT_CH*sizeof(float));
    
    ifs_l2_1_bn1_param.close();
    ifs_l2_1_bn2_param.close();
    ifs_l2_1_bn3_param.close();

    //--------------------------------------------------------------------------
    // Layer 2.2
    //--------------------------------------------------------------------------
    ifstream ifs_l2_2_conv1_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer2_2_conv1_weights.bin", ios::in | ios::binary);
    ifstream ifs_l2_2_conv2_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer2_2_conv2_weights.bin", ios::in | ios::binary);
    ifstream ifs_l2_2_conv3_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer2_2_conv3_weights.bin", ios::in | ios::binary);
    
    ifs_l2_2_conv1_param.read((char*)(  *resnet_layer2_2_conv1_weights), RESNET_LAYER2_CONV1_OUT_CH*RESNET_LAYER2_CONV1_IN_CH*sizeof(float));
    ifs_l2_2_conv2_param.read((char*)(***resnet_layer2_2_conv2_weights), RESNET_LAYER2_CONV2_OUT_CH*RESNET_LAYER2_CONV2_IN_CH*3*3*sizeof(float));
    ifs_l2_2_conv3_param.read((char*)(  *resnet_layer2_2_conv3_weights), RESNET_LAYER2_CONV3_OUT_CH*RESNET_LAYER2_CONV3_IN_CH*sizeof(float));
    
    ifs_l2_2_conv1_param.close();
    ifs_l2_2_conv2_param.close();
    ifs_l2_2_conv3_param.close();
    
    ifstream ifs_l2_2_bn1_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer2_2_bn1_params.bin", ios::in | ios::binary);
    ifstream ifs_l2_2_bn2_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer2_2_bn2_params.bin", ios::in | ios::binary);
    ifstream ifs_l2_2_bn3_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer2_2_bn3_params.bin", ios::in | ios::binary);
    
    ifs_l2_2_bn1_param.read((char*)(*resnet_layer2_2_bn1_params), 3*RESNET_LAYER2_CONV1_OUT_CH*sizeof(float));
    ifs_l2_2_bn2_param.read((char*)(*resnet_layer2_2_bn2_params), 3*RESNET_LAYER2_CONV2_OUT_CH*sizeof(float));
    ifs_l2_2_bn3_param.read((char*)(*resnet_layer2_2_bn3_params), 3*RESNET_LAYER2_CONV3_OUT_CH*sizeof(float));
    
    ifs_l2_2_bn1_param.close();
    ifs_l2_2_bn2_param.close();
    ifs_l2_2_bn3_param.close();

    //--------------------------------------------------------------------------
    // Layer 2.3
    //--------------------------------------------------------------------------
    ifstream ifs_l2_3_conv1_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer2_3_conv1_weights.bin", ios::in | ios::binary);
    ifstream ifs_l2_3_conv2_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer2_3_conv2_weights.bin", ios::in | ios::binary);
    ifstream ifs_l2_3_conv3_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer2_3_conv3_weights.bin", ios::in | ios::binary);

    ifs_l2_3_conv1_param.read((char*)(  *resnet_layer2_3_conv1_weights), RESNET_LAYER2_CONV1_OUT_CH*RESNET_LAYER2_CONV1_IN_CH*sizeof(float));
    ifs_l2_3_conv2_param.read((char*)(***resnet_layer2_3_conv2_weights), RESNET_LAYER2_CONV2_OUT_CH*RESNET_LAYER2_CONV2_IN_CH*3*3*sizeof(float));
    ifs_l2_3_conv3_param.read((char*)(  *resnet_layer2_3_conv3_weights), RESNET_LAYER2_CONV3_OUT_CH*RESNET_LAYER2_CONV3_IN_CH*sizeof(float));
    
    ifs_l2_3_conv1_param.close();
    ifs_l2_3_conv2_param.close();
    ifs_l2_3_conv3_param.close();

    ifstream ifs_l2_3_bn1_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer2_3_bn1_params.bin", ios::in | ios::binary);
    ifstream ifs_l2_3_bn2_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer2_3_bn2_params.bin", ios::in | ios::binary);
    ifstream ifs_l2_3_bn3_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer2_3_bn3_params.bin", ios::in | ios::binary);
    
    ifs_l2_3_bn1_param.read((char*)(*resnet_layer2_3_bn1_params), 3*RESNET_LAYER2_CONV1_OUT_CH*sizeof(float));
    ifs_l2_3_bn2_param.read((char*)(*resnet_layer2_3_bn2_params), 3*RESNET_LAYER2_CONV2_OUT_CH*sizeof(float));
    ifs_l2_3_bn3_param.read((char*)(*resnet_layer2_3_bn3_params), 3*RESNET_LAYER2_CONV3_OUT_CH*sizeof(float));
    
    ifs_l2_3_bn1_param.close();
    ifs_l2_3_bn2_param.close();
    ifs_l2_3_bn3_param.close();    

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
    
    //--------------------------------------------------------------------------
    // Layer 4.0
    //--------------------------------------------------------------------------
    ifstream ifs_l4_0_conv1_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer4_0_conv1_weights.bin", ios::in | ios::binary);
    ifstream ifs_l4_0_conv2_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer4_0_conv2_weights.bin", ios::in | ios::binary);
    ifstream ifs_l4_0_conv3_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer4_0_conv3_weights.bin", ios::in | ios::binary);

    ifs_l4_0_conv1_param.read((char*)(  *resnet_layer4_0_conv1_weights), RESNET_LAYER4_0_CONV1_OUT_CH*RESNET_LAYER4_0_CONV1_IN_CH*sizeof(float));
    ifs_l4_0_conv2_param.read((char*)(***resnet_layer4_0_conv2_weights), RESNET_LAYER4_0_CONV2_OUT_CH*RESNET_LAYER4_0_CONV2_IN_CH*3*3*sizeof(float));
    ifs_l4_0_conv3_param.read((char*)(  *resnet_layer4_0_conv3_weights), RESNET_LAYER4_0_CONV3_OUT_CH*RESNET_LAYER4_0_CONV3_IN_CH*sizeof(float));

    ifs_l4_0_conv1_param.close();
    ifs_l4_0_conv2_param.close();
    ifs_l4_0_conv3_param.close();

    ifstream ifs_l4_0_bn1_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer4_0_bn1_params.bin", ios::in | ios::binary);
    ifstream ifs_l4_0_bn2_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer4_0_bn2_params.bin", ios::in | ios::binary);
    ifstream ifs_l4_0_bn3_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer4_0_bn3_params.bin", ios::in | ios::binary);

    ifs_l4_0_bn1_param.read((char*)(*resnet_layer4_0_bn1_params), 3*RESNET_LAYER4_0_CONV1_OUT_CH*sizeof(float));
    ifs_l4_0_bn2_param.read((char*)(*resnet_layer4_0_bn2_params), 3*RESNET_LAYER4_0_CONV2_OUT_CH*sizeof(float));
    ifs_l4_0_bn3_param.read((char*)(*resnet_layer4_0_bn3_params), 3*RESNET_LAYER4_0_CONV3_OUT_CH*sizeof(float));

    ifs_l4_0_bn1_param.close();
    ifs_l4_0_bn2_param.close();
    ifs_l4_0_bn3_param.close();

    ifstream ifs_l4_0_downsample_0_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer4_0_downsample_0_weights.bin", ios::in | ios::binary);
    ifstream ifs_l4_0_downsample_1_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer4_0_downsample_1_params.bin", ios::in | ios::binary);

    ifs_l4_0_downsample_0_param.read((char*)(*resnet_layer4_0_downsample_0_weights), RESNET_LAYER4_0_DS_OUT_CH*RESNET_LAYER4_0_DS_IN_CH*sizeof(float));
    ifs_l4_0_downsample_1_param.read((char*)(*resnet_layer4_0_downsample_1_params), 3*RESNET_LAYER4_0_DS_OUT_CH*sizeof(float));

    ifs_l4_0_downsample_0_param.close();
    ifs_l4_0_downsample_1_param.close();
    
    //--------------------------------------------------------------------------
    // Layer 4.1
    //--------------------------------------------------------------------------
    ifstream ifs_l4_1_conv1_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer4_1_conv1_weights.bin", ios::in | ios::binary);
    ifstream ifs_l4_1_conv2_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer4_1_conv2_weights.bin", ios::in | ios::binary);
    ifstream ifs_l4_1_conv3_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer4_1_conv3_weights.bin", ios::in | ios::binary);

    ifs_l4_1_conv1_param.read((char*)(  *resnet_layer4_1_conv1_weights), RESNET_LAYER4_CONV1_OUT_CH*RESNET_LAYER4_CONV1_IN_CH*sizeof(float));
    ifs_l4_1_conv2_param.read((char*)(***resnet_layer4_1_conv2_weights), RESNET_LAYER4_CONV2_OUT_CH*RESNET_LAYER4_CONV2_IN_CH*3*3*sizeof(float));
    ifs_l4_1_conv3_param.read((char*)(  *resnet_layer4_1_conv3_weights), RESNET_LAYER4_CONV3_OUT_CH*RESNET_LAYER4_CONV3_IN_CH*sizeof(float));

    ifs_l4_1_conv1_param.close();
    ifs_l4_1_conv2_param.close();
    ifs_l4_1_conv3_param.close();

    ifstream ifs_l4_1_bn1_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer4_1_bn1_params.bin", ios::in | ios::binary);
    ifstream ifs_l4_1_bn2_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer4_1_bn2_params.bin", ios::in | ios::binary);
    ifstream ifs_l4_1_bn3_param("/usr/scratch/rsamanta9/bin/resnet_backbone/layer4_1_bn3_params.bin", ios::in | ios::binary);

    ifs_l4_1_bn1_param.read((char*)(*resnet_layer4_1_bn1_params), 3*RESNET_LAYER4_CONV1_OUT_CH*sizeof(float));
    ifs_l4_1_bn2_param.read((char*)(*resnet_layer4_1_bn2_params), 3*RESNET_LAYER4_CONV2_OUT_CH*sizeof(float));
    ifs_l4_1_bn3_param.read((char*)(*resnet_layer4_1_bn3_params), 3*RESNET_LAYER4_CONV3_OUT_CH*sizeof(float));

    ifs_l4_1_bn1_param.close();
    ifs_l4_1_bn2_param.close();
    ifs_l4_1_bn3_param.close();

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
    // Layer 0
    //--------------------------------------------------------------------------
    for(int i = 0; i < RESNET_LAYER0_CONV1_OUT_CH; i++)
        for(int j = 0; j< RESNET_LAYER0_CONV1_IN_CH; j++)
            for(int m = 0; m < 7; m++)
                for(int n =0; n < 7; n++)
                    fixp_layer0_conv1_weights[i][j][m][n] = (wt_t) resnet_layer0_conv1_weights[i][j][m][n];

    convert_bn<RESNET_LAYER0_CONV1_OUT_CH>(resnet_layer0_bn1_params, fixp_layer0_bn1_params);

    //--------------------------------------------------------------------------
    // Layer 1.0
    //--------------------------------------------------------------------------
    convert_1x1<RESNET_LAYER1_0_CONV1_IN_CH, RESNET_LAYER1_0_CONV1_OUT_CH>
               (resnet_layer1_0_conv1_weights, fixp_layer1_0_conv1_weights);
    convert_3x3<RESNET_LAYER1_0_CONV2_IN_CH, RESNET_LAYER1_0_CONV2_OUT_CH>
               (resnet_layer1_0_conv2_weights, fixp_layer1_0_conv2_weights);
    convert_1x1<RESNET_LAYER1_0_CONV3_IN_CH, RESNET_LAYER1_0_CONV3_OUT_CH>
               (resnet_layer1_0_conv3_weights, fixp_layer1_0_conv3_weights);
    convert_1x1<RESNET_LAYER1_0_DS_IN_CH, RESNET_LAYER1_0_DS_OUT_CH>
               (resnet_layer1_0_downsample_0_weights, fixp_layer1_0_downsample_0_weights);

    convert_bn<RESNET_LAYER1_0_CONV1_OUT_CH>(resnet_layer1_0_bn1_params, fixp_layer1_0_bn1_params);
    convert_bn<RESNET_LAYER1_0_CONV2_OUT_CH>(resnet_layer1_0_bn2_params, fixp_layer1_0_bn2_params);
    convert_bn<RESNET_LAYER1_0_CONV3_OUT_CH>(resnet_layer1_0_bn3_params, fixp_layer1_0_bn3_params);
    convert_bn<RESNET_LAYER1_0_DS_OUT_CH>(resnet_layer1_0_downsample_1_params, fixp_layer1_0_downsample_1_params);

    //--------------------------------------------------------------------------
    // Layer 1.1
    //--------------------------------------------------------------------------
    convert_1x1<RESNET_LAYER1_CONV1_IN_CH, RESNET_LAYER1_CONV1_OUT_CH>
               (resnet_layer1_1_conv1_weights, fixp_layer1_1_conv1_weights);
    convert_3x3<RESNET_LAYER1_CONV2_IN_CH, RESNET_LAYER1_CONV2_OUT_CH>
               (resnet_layer1_1_conv2_weights, fixp_layer1_1_conv2_weights);
    convert_1x1<RESNET_LAYER1_CONV3_IN_CH, RESNET_LAYER1_CONV3_OUT_CH>
               (resnet_layer1_1_conv3_weights, fixp_layer1_1_conv3_weights);
    
    convert_bn<RESNET_LAYER1_CONV1_OUT_CH>(resnet_layer1_1_bn1_params, fixp_layer1_1_bn1_params);
    convert_bn<RESNET_LAYER1_CONV2_OUT_CH>(resnet_layer1_1_bn2_params, fixp_layer1_1_bn2_params);
    convert_bn<RESNET_LAYER1_CONV3_OUT_CH>(resnet_layer1_1_bn3_params, fixp_layer1_1_bn3_params);

    //--------------------------------------------------------------------------
    // Layer 1.2
    //--------------------------------------------------------------------------
    convert_1x1<RESNET_LAYER1_CONV1_IN_CH, RESNET_LAYER1_CONV1_OUT_CH>
               (resnet_layer1_2_conv1_weights, fixp_layer1_2_conv1_weights);
    convert_3x3<RESNET_LAYER1_CONV2_IN_CH, RESNET_LAYER1_CONV2_OUT_CH>
               (resnet_layer1_2_conv2_weights, fixp_layer1_2_conv2_weights);
    convert_1x1<RESNET_LAYER1_CONV3_IN_CH, RESNET_LAYER1_CONV3_OUT_CH>
               (resnet_layer1_2_conv3_weights, fixp_layer1_2_conv3_weights);
    
    convert_bn<RESNET_LAYER1_CONV1_OUT_CH>(resnet_layer1_2_bn1_params, fixp_layer1_2_bn1_params);
    convert_bn<RESNET_LAYER1_CONV2_OUT_CH>(resnet_layer1_2_bn2_params, fixp_layer1_2_bn2_params);
    convert_bn<RESNET_LAYER1_CONV3_OUT_CH>(resnet_layer1_2_bn3_params, fixp_layer1_2_bn3_params);

    //--------------------------------------------------------------------------
    // Layer 2.0
    //--------------------------------------------------------------------------
    convert_1x1<RESNET_LAYER2_0_CONV1_IN_CH, RESNET_LAYER2_0_CONV1_OUT_CH>
               (resnet_layer2_0_conv1_weights, fixp_layer2_0_conv1_weights);
    convert_3x3<RESNET_LAYER2_0_CONV2_IN_CH, RESNET_LAYER2_0_CONV2_OUT_CH>
               (resnet_layer2_0_conv2_weights, fixp_layer2_0_conv2_weights);
    convert_1x1<RESNET_LAYER2_0_CONV3_IN_CH, RESNET_LAYER2_0_CONV3_OUT_CH>
               (resnet_layer2_0_conv3_weights, fixp_layer2_0_conv3_weights);
    convert_1x1<RESNET_LAYER2_0_DS_IN_CH, RESNET_LAYER2_0_DS_OUT_CH>
               (resnet_layer2_0_downsample_0_weights, fixp_layer2_0_downsample_0_weights);

    convert_bn<RESNET_LAYER2_0_CONV1_OUT_CH>(resnet_layer2_0_bn1_params, fixp_layer2_0_bn1_params);
    convert_bn<RESNET_LAYER2_0_CONV2_OUT_CH>(resnet_layer2_0_bn2_params, fixp_layer2_0_bn2_params);
    convert_bn<RESNET_LAYER2_0_CONV3_OUT_CH>(resnet_layer2_0_bn3_params, fixp_layer2_0_bn3_params);
    convert_bn<RESNET_LAYER2_0_DS_OUT_CH>(resnet_layer2_0_downsample_1_params, fixp_layer2_0_downsample_1_params);

    //--------------------------------------------------------------------------
    // Layer 2.1
    //--------------------------------------------------------------------------
    convert_1x1<RESNET_LAYER2_CONV1_IN_CH, RESNET_LAYER2_CONV1_OUT_CH>
               (resnet_layer2_1_conv1_weights, fixp_layer2_1_conv1_weights);
    convert_3x3<RESNET_LAYER2_CONV2_IN_CH, RESNET_LAYER2_CONV2_OUT_CH>
               (resnet_layer2_1_conv2_weights, fixp_layer2_1_conv2_weights);
    convert_1x1<RESNET_LAYER2_CONV3_IN_CH, RESNET_LAYER2_CONV3_OUT_CH>
               (resnet_layer2_1_conv3_weights, fixp_layer2_1_conv3_weights);
    
    convert_bn<RESNET_LAYER2_CONV1_OUT_CH>(resnet_layer2_1_bn1_params, fixp_layer2_1_bn1_params);
    convert_bn<RESNET_LAYER2_CONV2_OUT_CH>(resnet_layer2_1_bn2_params, fixp_layer2_1_bn2_params);
    convert_bn<RESNET_LAYER2_CONV3_OUT_CH>(resnet_layer2_1_bn3_params, fixp_layer2_1_bn3_params);

    //--------------------------------------------------------------------------
    // Layer 2.2
    //--------------------------------------------------------------------------
    convert_1x1<RESNET_LAYER2_CONV1_IN_CH, RESNET_LAYER2_CONV1_OUT_CH>
               (resnet_layer2_2_conv1_weights, fixp_layer2_2_conv1_weights);
    convert_3x3<RESNET_LAYER2_CONV2_IN_CH, RESNET_LAYER2_CONV2_OUT_CH>
               (resnet_layer2_2_conv2_weights, fixp_layer2_2_conv2_weights);
    convert_1x1<RESNET_LAYER2_CONV3_IN_CH, RESNET_LAYER2_CONV3_OUT_CH>
               (resnet_layer2_2_conv3_weights, fixp_layer2_2_conv3_weights);
    
    convert_bn<RESNET_LAYER2_CONV1_OUT_CH>(resnet_layer2_2_bn1_params, fixp_layer2_2_bn1_params);
    convert_bn<RESNET_LAYER2_CONV2_OUT_CH>(resnet_layer2_2_bn2_params, fixp_layer2_2_bn2_params);
    convert_bn<RESNET_LAYER2_CONV3_OUT_CH>(resnet_layer2_2_bn3_params, fixp_layer2_2_bn3_params);

    //--------------------------------------------------------------------------
    // Layer 2.3
    //--------------------------------------------------------------------------
    convert_1x1<RESNET_LAYER2_CONV1_IN_CH, RESNET_LAYER2_CONV1_OUT_CH>
               (resnet_layer2_3_conv1_weights, fixp_layer2_3_conv1_weights);
    convert_3x3<RESNET_LAYER2_CONV2_IN_CH, RESNET_LAYER2_CONV2_OUT_CH>
               (resnet_layer2_3_conv2_weights, fixp_layer2_3_conv2_weights);
    convert_1x1<RESNET_LAYER2_CONV3_IN_CH, RESNET_LAYER2_CONV3_OUT_CH>
               (resnet_layer2_3_conv3_weights, fixp_layer2_3_conv3_weights);
    
    convert_bn<RESNET_LAYER2_CONV1_OUT_CH>(resnet_layer2_3_bn1_params, fixp_layer2_3_bn1_params);
    convert_bn<RESNET_LAYER2_CONV2_OUT_CH>(resnet_layer2_3_bn2_params, fixp_layer2_3_bn2_params);
    convert_bn<RESNET_LAYER2_CONV3_OUT_CH>(resnet_layer2_3_bn3_params, fixp_layer2_3_bn3_params);

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
    
    //--------------------------------------------------------------------------
    // Layer 4.0
    //--------------------------------------------------------------------------
    convert_1x1<RESNET_LAYER4_0_CONV1_IN_CH, RESNET_LAYER4_0_CONV1_OUT_CH>
               (resnet_layer4_0_conv1_weights, fixp_layer4_0_conv1_weights);
    convert_3x3<RESNET_LAYER4_0_CONV2_IN_CH, RESNET_LAYER4_0_CONV2_OUT_CH>
               (resnet_layer4_0_conv2_weights, fixp_layer4_0_conv2_weights);
    convert_1x1<RESNET_LAYER4_0_CONV3_IN_CH, RESNET_LAYER4_0_CONV3_OUT_CH>
               (resnet_layer4_0_conv3_weights, fixp_layer4_0_conv3_weights);
    convert_1x1<RESNET_LAYER4_0_DS_IN_CH, RESNET_LAYER4_0_DS_OUT_CH>
               (resnet_layer4_0_downsample_0_weights, fixp_layer4_0_downsample_0_weights);

    convert_bn<RESNET_LAYER4_0_CONV1_OUT_CH>(resnet_layer4_0_bn1_params, fixp_layer4_0_bn1_params);
    convert_bn<RESNET_LAYER4_0_CONV2_OUT_CH>(resnet_layer4_0_bn2_params, fixp_layer4_0_bn2_params);
    convert_bn<RESNET_LAYER4_0_CONV3_OUT_CH>(resnet_layer4_0_bn3_params, fixp_layer4_0_bn3_params);
    convert_bn<RESNET_LAYER4_0_DS_OUT_CH>(resnet_layer4_0_downsample_1_params, fixp_layer4_0_downsample_1_params);

    //--------------------------------------------------------------------------
    // Layer 4.1
    //--------------------------------------------------------------------------
    convert_1x1<RESNET_LAYER4_CONV1_IN_CH, RESNET_LAYER4_CONV1_OUT_CH>
               (resnet_layer4_1_conv1_weights, fixp_layer4_1_conv1_weights);
    convert_3x3<RESNET_LAYER4_CONV2_IN_CH, RESNET_LAYER4_CONV2_OUT_CH>
               (resnet_layer4_1_conv2_weights, fixp_layer4_1_conv2_weights);
    convert_1x1<RESNET_LAYER4_CONV3_IN_CH, RESNET_LAYER4_CONV3_OUT_CH>
               (resnet_layer4_1_conv3_weights, fixp_layer4_1_conv3_weights);
    
    convert_bn<RESNET_LAYER4_CONV1_OUT_CH>(resnet_layer4_1_bn1_params, fixp_layer4_1_bn1_params);
    convert_bn<RESNET_LAYER4_CONV2_OUT_CH>(resnet_layer4_1_bn2_params, fixp_layer4_1_bn2_params);
    convert_bn<RESNET_LAYER4_CONV3_OUT_CH>(resnet_layer4_1_bn3_params, fixp_layer4_1_bn3_params);

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
    test_top( resnet_layer0_input_fm,
              fixp_layer0_conv1_weights,           fixp_layer0_bn1_params, 
              resnet_layer0_output_fm,
              
              resnet_layer1_input_fm,
              fixp_layer1_0_conv1_weights,         fixp_layer1_0_bn1_params,
              fixp_layer1_0_conv2_weights,         fixp_layer1_0_bn2_params,
              fixp_layer1_0_conv3_weights,         fixp_layer1_0_bn3_params,
              fixp_layer1_0_downsample_0_weights,  fixp_layer1_0_downsample_1_params,
              fixp_layer1_1_conv1_weights,         fixp_layer1_1_bn1_params,
              fixp_layer1_1_conv2_weights,         fixp_layer1_1_bn2_params,
              fixp_layer1_1_conv3_weights,         fixp_layer1_1_bn3_params,
              fixp_layer1_2_conv1_weights,         fixp_layer1_2_bn1_params,
              fixp_layer1_2_conv2_weights,         fixp_layer1_2_bn2_params,
              fixp_layer1_2_conv3_weights,         fixp_layer1_2_bn3_params,
              resnet_layer1_output_fm,

              resnet_layer2_input_fm,
              fixp_layer2_0_conv1_weights,         fixp_layer2_0_bn1_params,
              fixp_layer2_0_conv2_weights,         fixp_layer2_0_bn2_params,
              fixp_layer2_0_conv3_weights,         fixp_layer2_0_bn3_params,
              fixp_layer2_0_downsample_0_weights,  fixp_layer2_0_downsample_1_params,
              fixp_layer2_1_conv1_weights,         fixp_layer2_1_bn1_params,
              fixp_layer2_1_conv2_weights,         fixp_layer2_1_bn2_params,
              fixp_layer2_1_conv3_weights,         fixp_layer2_1_bn3_params,
              fixp_layer2_2_conv1_weights,         fixp_layer2_2_bn1_params,
              fixp_layer2_2_conv2_weights,         fixp_layer2_2_bn2_params,
              fixp_layer2_2_conv3_weights,         fixp_layer2_2_bn3_params,
              fixp_layer2_3_conv1_weights,         fixp_layer2_3_bn1_params,
              fixp_layer2_3_conv2_weights,         fixp_layer2_3_bn2_params,
              fixp_layer2_3_conv3_weights,         fixp_layer2_3_bn3_params,
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
              
              resnet_layer4_input_fm,
              fixp_layer4_0_conv1_weights,         fixp_layer4_0_bn1_params,
              fixp_layer4_0_conv2_weights,         fixp_layer4_0_bn2_params,
              fixp_layer4_0_conv3_weights,         fixp_layer4_0_bn3_params,
              fixp_layer4_0_downsample_0_weights,  fixp_layer4_0_downsample_1_params,
              fixp_layer4_1_conv1_weights,         fixp_layer4_1_bn1_params,
              fixp_layer4_1_conv2_weights,         fixp_layer4_1_bn2_params,
              fixp_layer4_1_conv3_weights,         fixp_layer4_1_bn3_params,
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
    std::cout << "Layer 4 Output MSE:  " << mse << std::endl;    
    
    if(mse < 0.0000001)
       std::cout << "ResNet-50 Model Verification SUCCESSFUL!!!" << std::endl << std::endl; 
    else 
       std::cout << "ResNet-50 Model Verification FAILED :(" << std::endl << std::endl; 


#endif // }

    return 0;
}
