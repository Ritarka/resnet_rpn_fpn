#include <iostream>
#include <fstream>
#include <cmath>
#include "hls_stream.h"
#include "qdtrack.h"

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


//--------------------------------------------------------------------------
// FPN
//--------------------------------------------------------------------------
//CONV_3x3
float   conv_0_input_feature_map[FPN_CONV_0_ID][FPN_CONV_0_IH][FPN_CONV_0_IW];
float   conv_1_input_feature_map[FPN_CONV_1_ID][FPN_CONV_1_IH][FPN_CONV_1_IW];
float   conv_2_input_feature_map[FPN_CONV_2_ID][FPN_CONV_2_IH][FPN_CONV_2_IW];
float   conv_3_input_feature_map[FPN_CONV_3_ID][FPN_CONV_3_IH][FPN_CONV_3_IW];
float   conv_0_weights[FPN_CONV_0_OD][FPN_CONV_0_ID][3][3];
float   conv_1_weights[FPN_CONV_1_OD][FPN_CONV_1_ID][3][3];
float   conv_2_weights[FPN_CONV_2_OD][FPN_CONV_2_ID][3][3];
float   conv_3_weights[FPN_CONV_3_OD][FPN_CONV_3_ID][3][3];
float   conv_0_bias[FPN_CONV_0_OD];
float   conv_1_bias[FPN_CONV_1_OD];
float   conv_2_bias[FPN_CONV_2_OD];
float   conv_3_bias[FPN_CONV_3_OD];
float   golden_conv_0_golden_output[FPN_CONV_0_OD][FPN_CONV_0_OH][FPN_CONV_0_OW];
float   golden_conv_1_golden_output[FPN_CONV_1_OD][FPN_CONV_1_OH][FPN_CONV_1_OW];
float   golden_conv_2_golden_output[FPN_CONV_2_OD][FPN_CONV_2_OH][FPN_CONV_2_OW];
float   golden_conv_3_golden_output[FPN_CONV_3_OD][FPN_CONV_3_OH][FPN_CONV_3_OW];

fm_t	fixp_conv_0_input_feature_map[FPN_CONV_0_ID][FPN_CONV_0_IH][FPN_CONV_0_IW];
fm_t	fixp_conv_1_input_feature_map[FPN_CONV_1_ID][FPN_CONV_1_IH][FPN_CONV_1_IW];
fm_t	fixp_conv_2_input_feature_map[FPN_CONV_2_ID][FPN_CONV_2_IH][FPN_CONV_2_IW];
fm_t	fixp_conv_3_input_feature_map[FPN_CONV_3_ID][FPN_CONV_3_IH][FPN_CONV_3_IW];
wt_t	fixp_conv_0_weights[FPN_CONV_0_OD][FPN_CONV_0_ID][3][3];
wt_t	fixp_conv_1_weights[FPN_CONV_1_OD][FPN_CONV_1_ID][3][3];
wt_t	fixp_conv_2_weights[FPN_CONV_2_OD][FPN_CONV_2_ID][3][3];
wt_t	fixp_conv_3_weights[FPN_CONV_3_OD][FPN_CONV_3_ID][3][3];
wt_t	fixp_conv_0_bias[FPN_CONV_0_OD];
wt_t	fixp_conv_1_bias[FPN_CONV_1_OD];
wt_t	fixp_conv_2_bias[FPN_CONV_2_OD];
wt_t	fixp_conv_3_bias[FPN_CONV_3_OD];

//CONV_1x1
float   lateral_conv_0_input_feature_map[LATERAL_CONV_0_ID][LATERAL_CONV_0_IH][LATERAL_CONV_0_IW];
float   lateral_conv_1_input_feature_map[LATERAL_CONV_1_ID][LATERAL_CONV_1_IH][LATERAL_CONV_1_IW];
float   lateral_conv_2_input_feature_map[LATERAL_CONV_2_ID][LATERAL_CONV_2_IH][LATERAL_CONV_2_IW];
float   lateral_conv_3_input_feature_map[LATERAL_CONV_3_ID][LATERAL_CONV_3_IH][LATERAL_CONV_3_IW];
float   lateral_conv_0_weights[LATERAL_CONV_0_OD][LATERAL_CONV_0_ID][1][1];
float   lateral_conv_1_weights[LATERAL_CONV_1_OD][LATERAL_CONV_1_ID][1][1];
float   lateral_conv_2_weights[LATERAL_CONV_2_OD][LATERAL_CONV_2_ID][1][1];
float   lateral_conv_3_weights[LATERAL_CONV_3_OD][LATERAL_CONV_3_ID][1][1];
float   lateral_conv_0_bias[LATERAL_CONV_0_OD];
float   lateral_conv_1_bias[LATERAL_CONV_1_OD];
float   lateral_conv_2_bias[LATERAL_CONV_2_OD];
float   lateral_conv_3_bias[LATERAL_CONV_3_OD];

fm_t	fixp_lateral_conv_0_input_feature_map[LATERAL_CONV_0_ID][LATERAL_CONV_0_IH][LATERAL_CONV_0_IW];
fm_t	fixp_lateral_conv_1_input_feature_map[LATERAL_CONV_1_ID][LATERAL_CONV_1_IH][LATERAL_CONV_1_IW];
fm_t	fixp_lateral_conv_2_input_feature_map[LATERAL_CONV_2_ID][LATERAL_CONV_2_IH][LATERAL_CONV_2_IW];
fm_t	fixp_lateral_conv_3_input_feature_map[LATERAL_CONV_3_ID][LATERAL_CONV_3_IH][LATERAL_CONV_3_IW];
wt_t	fixp_lateral_conv_0_weights[LATERAL_CONV_0_OD][LATERAL_CONV_0_ID][1][1];
wt_t	fixp_lateral_conv_1_weights[LATERAL_CONV_1_OD][LATERAL_CONV_1_ID][1][1];
wt_t	fixp_lateral_conv_2_weights[LATERAL_CONV_2_OD][LATERAL_CONV_2_ID][1][1];
wt_t	fixp_lateral_conv_3_weights[LATERAL_CONV_3_OD][LATERAL_CONV_3_ID][1][1];
wt_t	fixp_lateral_conv_0_bias[LATERAL_CONV_0_OD];
wt_t	fixp_lateral_conv_1_bias[LATERAL_CONV_1_OD];
wt_t	fixp_lateral_conv_2_bias[LATERAL_CONV_2_OD];
wt_t	fixp_lateral_conv_3_bias[LATERAL_CONV_3_OD];

//--------------------------------------------------------------------------
// Computed outputs
//--------------------------------------------------------------------------
//CONV_3x3
fm_t    fixp_conv_0_output_feature_map[FPN_CONV_0_OD][FPN_CONV_0_OH][FPN_CONV_0_OW] = {0};
fm_t    fixp_conv_1_output_feature_map[FPN_CONV_1_OD][FPN_CONV_1_OH][FPN_CONV_1_OW] = {0};
fm_t    fixp_conv_2_output_feature_map[FPN_CONV_2_OD][FPN_CONV_2_OH][FPN_CONV_2_OW] = {0};
fm_t    fixp_conv_3_output_feature_map[FPN_CONV_3_OD][FPN_CONV_3_OH][FPN_CONV_3_OW] = {0};


//--------------------------------------------------------------------------
// RPN
//--------------------------------------------------------------------------
int rpn_topk_index0_2[RPN_PRE_NMS_SIZE0];
int rpn_topk_index1_2[RPN_PRE_NMS_SIZE1];
int rpn_topk_index2_2[RPN_PRE_NMS_SIZE2];

fm_t rpn_anchor0_reg_fm_2[RPN_REG_OUT_CH*RPN_INPUT0_IN_FM_HEIGHT*RPN_INPUT0_IN_FM_WIDTH/4][4];
fm_t rpn_anchor1_reg_fm_2[RPN_REG_OUT_CH*RPN_INPUT0_IN_FM_HEIGHT*RPN_INPUT0_IN_FM_WIDTH/4][4];
fm_t rpn_anchor2_reg_fm_2[RPN_REG_OUT_CH*RPN_INPUT0_IN_FM_HEIGHT*RPN_INPUT0_IN_FM_WIDTH/4][4];

// float rpn_anchor0_reg[RPN_REG_OUT_CH*RPN_INPUT0_IN_FM_HEIGHT*RPN_INPUT0_IN_FM_WIDTH/4][4];
// float rpn_anchor1_reg[RPN_REG_OUT_CH*RPN_INPUT0_IN_FM_HEIGHT*RPN_INPUT0_IN_FM_WIDTH/4][4];
// float rpn_anchor2_reg[RPN_REG_OUT_CH*RPN_INPUT0_IN_FM_HEIGHT*RPN_INPUT0_IN_FM_WIDTH/4][4];

fm_t rpn_anchor0_cls_fm_2[RPN_CLS_OUT_CH*RPN_INPUT0_IN_FM_HEIGHT*RPN_INPUT0_IN_FM_WIDTH];
fm_t rpn_anchor1_cls_fm_2[RPN_CLS_OUT_CH*RPN_INPUT1_IN_FM_HEIGHT*RPN_INPUT1_IN_FM_WIDTH];
fm_t rpn_anchor2_cls_fm_2[RPN_CLS_OUT_CH*RPN_INPUT2_IN_FM_HEIGHT*RPN_INPUT2_IN_FM_WIDTH];

// float rpn_anchor0_cls[RPN_CLS_OUT_CH*RPN_INPUT0_IN_FM_HEIGHT*RPN_INPUT0_IN_FM_WIDTH];
// float rpn_anchor1_cls[RPN_CLS_OUT_CH*RPN_INPUT1_IN_FM_HEIGHT*RPN_INPUT1_IN_FM_WIDTH];
// float rpn_anchor2_cls[RPN_CLS_OUT_CH*RPN_INPUT2_IN_FM_HEIGHT*RPN_INPUT2_IN_FM_WIDTH];

fm_t rpn_input0_fm[RPN_CONV_IN_CH][RPN_INPUT0_IN_FM_HEIGHT][RPN_INPUT0_IN_FM_WIDTH];
fm_t rpn_input1_fm[RPN_CONV_IN_CH][RPN_INPUT1_IN_FM_HEIGHT][RPN_INPUT1_IN_FM_WIDTH];
fm_t rpn_input2_fm[RPN_CONV_IN_CH][RPN_INPUT2_IN_FM_HEIGHT][RPN_INPUT2_IN_FM_WIDTH];
fm_t rpn_input3_fm[RPN_CONV_IN_CH][RPN_INPUT3_IN_FM_HEIGHT][RPN_INPUT3_IN_FM_WIDTH];
fm_t rpn_input4_fm[RPN_CONV_IN_CH][RPN_INPUT4_IN_FM_HEIGHT][RPN_INPUT4_IN_FM_WIDTH];


float rpn_input0[RPN_CONV_IN_CH][RPN_INPUT0_IN_FM_HEIGHT][RPN_INPUT0_IN_FM_WIDTH];
float rpn_input1[RPN_CONV_IN_CH][RPN_INPUT1_IN_FM_HEIGHT][RPN_INPUT1_IN_FM_WIDTH];
float rpn_input2[RPN_CONV_IN_CH][RPN_INPUT2_IN_FM_HEIGHT][RPN_INPUT2_IN_FM_WIDTH];
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


fm_t rpn_output0_cls_fm[RPN_CLS_OUT_CH][RPN_INPUT0_IN_FM_HEIGHT][RPN_INPUT0_IN_FM_WIDTH];
fm_t rpn_output1_cls_fm[RPN_CLS_OUT_CH][RPN_INPUT1_IN_FM_HEIGHT][RPN_INPUT1_IN_FM_WIDTH];
fm_t rpn_output2_cls_fm[RPN_CLS_OUT_CH][RPN_INPUT2_IN_FM_HEIGHT][RPN_INPUT2_IN_FM_WIDTH];
fm_t rpn_output3_cls_fm[RPN_CLS_OUT_CH][RPN_INPUT3_IN_FM_HEIGHT][RPN_INPUT3_IN_FM_WIDTH];
fm_t rpn_output4_cls_fm[RPN_CLS_OUT_CH][RPN_INPUT4_IN_FM_HEIGHT][RPN_INPUT4_IN_FM_WIDTH];

float fl_rpn_output0_cls_fm[RPN_CLS_OUT_CH][RPN_INPUT0_IN_FM_HEIGHT][RPN_INPUT0_IN_FM_WIDTH];
float fl_rpn_output1_cls_fm[RPN_CLS_OUT_CH][RPN_INPUT1_IN_FM_HEIGHT][RPN_INPUT1_IN_FM_WIDTH];
float fl_rpn_output2_cls_fm[RPN_CLS_OUT_CH][RPN_INPUT2_IN_FM_HEIGHT][RPN_INPUT2_IN_FM_WIDTH];
float fl_rpn_output3_cls_fm[RPN_CLS_OUT_CH][RPN_INPUT3_IN_FM_HEIGHT][RPN_INPUT3_IN_FM_WIDTH];
float fl_rpn_output4_cls_fm[RPN_CLS_OUT_CH][RPN_INPUT4_IN_FM_HEIGHT][RPN_INPUT4_IN_FM_WIDTH];



fm_t rpn_output0_reg_fm[RPN_REG_OUT_CH][RPN_INPUT0_IN_FM_HEIGHT][RPN_INPUT0_IN_FM_WIDTH];
fm_t rpn_output1_reg_fm[RPN_REG_OUT_CH][RPN_INPUT1_IN_FM_HEIGHT][RPN_INPUT1_IN_FM_WIDTH];
fm_t rpn_output2_reg_fm[RPN_REG_OUT_CH][RPN_INPUT2_IN_FM_HEIGHT][RPN_INPUT2_IN_FM_WIDTH];
fm_t rpn_output3_reg_fm[RPN_REG_OUT_CH][RPN_INPUT3_IN_FM_HEIGHT][RPN_INPUT3_IN_FM_WIDTH];
fm_t rpn_output4_reg_fm[RPN_REG_OUT_CH][RPN_INPUT4_IN_FM_HEIGHT][RPN_INPUT4_IN_FM_WIDTH];

float fl_rpn_output0_reg_fm[RPN_REG_OUT_CH][RPN_INPUT0_IN_FM_HEIGHT][RPN_INPUT0_IN_FM_WIDTH];
float fl_rpn_output1_reg_fm[RPN_REG_OUT_CH][RPN_INPUT1_IN_FM_HEIGHT][RPN_INPUT1_IN_FM_WIDTH];
float fl_rpn_output2_reg_fm[RPN_REG_OUT_CH][RPN_INPUT2_IN_FM_HEIGHT][RPN_INPUT2_IN_FM_WIDTH];
float fl_rpn_output3_reg_fm[RPN_REG_OUT_CH][RPN_INPUT3_IN_FM_HEIGHT][RPN_INPUT3_IN_FM_WIDTH];
float fl_rpn_output4_reg_fm[RPN_REG_OUT_CH][RPN_INPUT4_IN_FM_HEIGHT][RPN_INPUT4_IN_FM_WIDTH];


float fl_rpn_output0_fm[RPN_CONV_IN_CH][RPN_INPUT0_IN_FM_HEIGHT][RPN_INPUT0_IN_FM_WIDTH];
float fl_rpn_output1_fm[RPN_CONV_IN_CH][RPN_INPUT1_IN_FM_HEIGHT][RPN_INPUT1_IN_FM_WIDTH];
float fl_rpn_output2_fm[RPN_CONV_IN_CH][RPN_INPUT2_IN_FM_HEIGHT][RPN_INPUT2_IN_FM_WIDTH];
float fl_rpn_output3_fm[RPN_CONV_IN_CH][RPN_INPUT3_IN_FM_HEIGHT][RPN_INPUT3_IN_FM_WIDTH];
float fl_rpn_output4_fm[RPN_CONV_IN_CH][RPN_INPUT4_IN_FM_HEIGHT][RPN_INPUT4_IN_FM_WIDTH];



fm_t rpn_output0_fm[RPN_CONV_IN_CH][RPN_INPUT0_IN_FM_HEIGHT][RPN_INPUT0_IN_FM_WIDTH];
fm_t rpn_output1_fm[RPN_CONV_IN_CH][RPN_INPUT1_IN_FM_HEIGHT][RPN_INPUT1_IN_FM_WIDTH];
fm_t rpn_output2_fm[RPN_CONV_IN_CH][RPN_INPUT2_IN_FM_HEIGHT][RPN_INPUT2_IN_FM_WIDTH];
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

void resnet_load_weights()
{
    //--------------------------------------------------------------------------
    // Layer 0
    //--------------------------------------------------------------------------
    ifstream ifs_l0_conv1_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/conv1_weights.bin", ios::in | ios::binary);
    ifs_l0_conv1_param.read((char*)(***resnet_layer0_conv1_weights), RESNET_LAYER0_CONV1_OUT_CH*RESNET_LAYER0_CONV1_IN_CH*7*7*sizeof(float));
    ifs_l0_conv1_param.close();

    //cout << resnet_layer0_conv1_weights[0][0][1][1] << std::endl;

    ifstream ifs_l0_bn1_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/bn1_params.bin", ios::in | ios::binary);
    ifs_l0_bn1_param.read((char*)(*resnet_layer0_bn1_params), 3*RESNET_LAYER0_CONV1_OUT_CH*sizeof(float));
    ifs_l0_bn1_param.close();

    //--------------------------------------------------------------------------
    // Layer 1.0
    //--------------------------------------------------------------------------
    ifstream ifs_l1_0_conv1_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer1_0_conv1_weights.bin", ios::in | ios::binary);
    ifstream ifs_l1_0_conv2_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer1_0_conv2_weights.bin", ios::in | ios::binary);
    ifstream ifs_l1_0_conv3_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer1_0_conv3_weights.bin", ios::in | ios::binary);
    
    ifs_l1_0_conv1_param.read((char*)(  *resnet_layer1_0_conv1_weights), RESNET_LAYER1_0_CONV1_OUT_CH*RESNET_LAYER1_0_CONV1_IN_CH*sizeof(float));
    ifs_l1_0_conv2_param.read((char*)(***resnet_layer1_0_conv2_weights), RESNET_LAYER1_0_CONV2_OUT_CH*RESNET_LAYER1_0_CONV2_IN_CH*3*3*sizeof(float));
    ifs_l1_0_conv3_param.read((char*)(  *resnet_layer1_0_conv3_weights), RESNET_LAYER1_0_CONV3_OUT_CH*RESNET_LAYER1_0_CONV3_IN_CH*sizeof(float));

    ifs_l1_0_conv1_param.close();
    ifs_l1_0_conv2_param.close();
    ifs_l1_0_conv3_param.close();

    ifstream ifs_l1_0_bn1_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer1_0_bn1_params.bin", ios::in | ios::binary);
    ifstream ifs_l1_0_bn2_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer1_0_bn2_params.bin", ios::in | ios::binary);
    ifstream ifs_l1_0_bn3_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer1_0_bn3_params.bin", ios::in | ios::binary);

    ifs_l1_0_bn1_param.read((char*)(*resnet_layer1_0_bn1_params), 3*RESNET_LAYER1_0_CONV1_OUT_CH*sizeof(float));
    ifs_l1_0_bn2_param.read((char*)(*resnet_layer1_0_bn2_params), 3*RESNET_LAYER1_0_CONV2_OUT_CH*sizeof(float));
    ifs_l1_0_bn3_param.read((char*)(*resnet_layer1_0_bn3_params), 3*RESNET_LAYER1_0_CONV3_OUT_CH*sizeof(float));

    ifs_l1_0_bn1_param.close();
    ifs_l1_0_bn2_param.close();
    ifs_l1_0_bn3_param.close();
    
    ifstream ifs_l1_0_downsample_0_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer1_0_downsample_0_weights.bin", ios::in | ios::binary);
    ifstream ifs_l1_0_downsample_1_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer1_0_downsample_1_params.bin", ios::in | ios::binary);
    
    ifs_l1_0_downsample_0_param.read((char*)(*resnet_layer1_0_downsample_0_weights), RESNET_LAYER1_0_DS_OUT_CH*RESNET_LAYER1_0_DS_IN_CH*sizeof(float));
    ifs_l1_0_downsample_1_param.read((char*)(*resnet_layer1_0_downsample_1_params), 3*RESNET_LAYER1_0_DS_OUT_CH*sizeof(float));
    
    ifs_l1_0_downsample_0_param.close();
    ifs_l1_0_downsample_1_param.close();

    //--------------------------------------------------------------------------
    // Layer 1.1
    //--------------------------------------------------------------------------
    ifstream ifs_l1_1_conv1_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer1_1_conv1_weights.bin", ios::in | ios::binary);
    ifstream ifs_l1_1_conv2_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer1_1_conv2_weights.bin", ios::in | ios::binary);
    ifstream ifs_l1_1_conv3_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer1_1_conv3_weights.bin", ios::in | ios::binary);
    
    ifs_l1_1_conv1_param.read((char*)(  *resnet_layer1_1_conv1_weights), RESNET_LAYER1_CONV1_OUT_CH*RESNET_LAYER1_CONV1_IN_CH*sizeof(float));
    ifs_l1_1_conv2_param.read((char*)(***resnet_layer1_1_conv2_weights), RESNET_LAYER1_CONV2_OUT_CH*RESNET_LAYER1_CONV2_IN_CH*3*3*sizeof(float));
    ifs_l1_1_conv3_param.read((char*)(  *resnet_layer1_1_conv3_weights), RESNET_LAYER1_CONV3_OUT_CH*RESNET_LAYER1_CONV3_IN_CH*sizeof(float));
    
    ifs_l1_1_conv1_param.close();
    ifs_l1_1_conv2_param.close();
    ifs_l1_1_conv3_param.close();
    
    ifstream ifs_l1_1_bn1_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer1_1_bn1_params.bin", ios::in | ios::binary);
    ifstream ifs_l1_1_bn2_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer1_1_bn2_params.bin", ios::in | ios::binary);
    ifstream ifs_l1_1_bn3_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer1_1_bn3_params.bin", ios::in | ios::binary);

    ifs_l1_1_bn1_param.read((char*)(*resnet_layer1_1_bn1_params), 3*RESNET_LAYER1_CONV1_OUT_CH*sizeof(float));
    ifs_l1_1_bn2_param.read((char*)(*resnet_layer1_1_bn2_params), 3*RESNET_LAYER1_CONV2_OUT_CH*sizeof(float));
    ifs_l1_1_bn3_param.read((char*)(*resnet_layer1_1_bn3_params), 3*RESNET_LAYER1_CONV3_OUT_CH*sizeof(float));

    ifs_l1_1_bn1_param.close();
    ifs_l1_1_bn2_param.close();
    ifs_l1_1_bn3_param.close();

    //--------------------------------------------------------------------------
    // Layer 1.2
    //--------------------------------------------------------------------------
    ifstream ifs_l1_2_conv1_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer1_2_conv1_weights.bin", ios::in | ios::binary);
    ifstream ifs_l1_2_conv2_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer1_2_conv2_weights.bin", ios::in | ios::binary);
    ifstream ifs_l1_2_conv3_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer1_2_conv3_weights.bin", ios::in | ios::binary);
    
    ifs_l1_2_conv1_param.read((char*)(  *resnet_layer1_2_conv1_weights), RESNET_LAYER1_CONV1_OUT_CH*RESNET_LAYER1_CONV1_IN_CH*sizeof(float));
    ifs_l1_2_conv2_param.read((char*)(***resnet_layer1_2_conv2_weights), RESNET_LAYER1_CONV2_OUT_CH*RESNET_LAYER1_CONV2_IN_CH*3*3*sizeof(float));
    ifs_l1_2_conv3_param.read((char*)(  *resnet_layer1_2_conv3_weights), RESNET_LAYER1_CONV3_OUT_CH*RESNET_LAYER1_CONV3_IN_CH*sizeof(float));

    ifs_l1_2_conv1_param.close();
    ifs_l1_2_conv2_param.close();
    ifs_l1_2_conv3_param.close();

    ifstream ifs_l1_2_bn1_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer1_2_bn1_params.bin", ios::in | ios::binary);
    ifstream ifs_l1_2_bn2_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer1_2_bn2_params.bin", ios::in | ios::binary);
    ifstream ifs_l1_2_bn3_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer1_2_bn3_params.bin", ios::in | ios::binary);
    
    ifs_l1_2_bn1_param.read((char*)(*resnet_layer1_2_bn1_params), 3*RESNET_LAYER1_CONV1_OUT_CH*sizeof(float));
    ifs_l1_2_bn2_param.read((char*)(*resnet_layer1_2_bn2_params), 3*RESNET_LAYER1_CONV2_OUT_CH*sizeof(float));
    ifs_l1_2_bn3_param.read((char*)(*resnet_layer1_2_bn3_params), 3*RESNET_LAYER1_CONV3_OUT_CH*sizeof(float));
    
    ifs_l1_2_bn1_param.close();
    ifs_l1_2_bn2_param.close();
    ifs_l1_2_bn3_param.close();
    
    //--------------------------------------------------------------------------
    // Layer 2.0
    //--------------------------------------------------------------------------
    ifstream ifs_l2_0_conv1_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer2_0_conv1_weights.bin", ios::in | ios::binary);
    ifstream ifs_l2_0_conv2_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer2_0_conv2_weights.bin", ios::in | ios::binary);
    ifstream ifs_l2_0_conv3_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer2_0_conv3_weights.bin", ios::in | ios::binary);
    
    ifs_l2_0_conv1_param.read((char*)(  *resnet_layer2_0_conv1_weights), RESNET_LAYER2_0_CONV1_OUT_CH*RESNET_LAYER2_0_CONV1_IN_CH*sizeof(float));
    ifs_l2_0_conv2_param.read((char*)(***resnet_layer2_0_conv2_weights), RESNET_LAYER2_0_CONV2_OUT_CH*RESNET_LAYER2_0_CONV2_IN_CH*3*3*sizeof(float));
    ifs_l2_0_conv3_param.read((char*)(  *resnet_layer2_0_conv3_weights), RESNET_LAYER2_0_CONV3_OUT_CH*RESNET_LAYER2_0_CONV3_IN_CH*sizeof(float));
    
    ifs_l2_0_conv1_param.close();
    ifs_l2_0_conv2_param.close();
    ifs_l2_0_conv3_param.close();
    
    ifstream ifs_l2_0_bn1_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer2_0_bn1_params.bin", ios::in | ios::binary);
    ifstream ifs_l2_0_bn2_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer2_0_bn2_params.bin", ios::in | ios::binary);
    ifstream ifs_l2_0_bn3_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer2_0_bn3_params.bin", ios::in | ios::binary);
    
    ifs_l2_0_bn1_param.read((char*)(*resnet_layer2_0_bn1_params), 3*RESNET_LAYER2_0_CONV1_OUT_CH*sizeof(float));
    ifs_l2_0_bn2_param.read((char*)(*resnet_layer2_0_bn2_params), 3*RESNET_LAYER2_0_CONV2_OUT_CH*sizeof(float));
    ifs_l2_0_bn3_param.read((char*)(*resnet_layer2_0_bn3_params), 3*RESNET_LAYER2_0_CONV3_OUT_CH*sizeof(float));
    
    ifs_l2_0_bn1_param.close();
    ifs_l2_0_bn2_param.close();
    ifs_l2_0_bn3_param.close();
    
    ifstream ifs_l2_0_downsample_0_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer2_0_downsample_0_weights.bin", ios::in | ios::binary);
    ifstream ifs_l2_0_downsample_1_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer2_0_downsample_1_params.bin", ios::in | ios::binary);
    
    ifs_l2_0_downsample_0_param.read((char*)(*resnet_layer2_0_downsample_0_weights), RESNET_LAYER2_0_DS_OUT_CH*RESNET_LAYER2_0_DS_IN_CH*sizeof(float));
    ifs_l2_0_downsample_1_param.read((char*)(*resnet_layer2_0_downsample_1_params), 3*RESNET_LAYER2_0_DS_OUT_CH*sizeof(float));
    
    ifs_l2_0_downsample_0_param.close();
    ifs_l2_0_downsample_1_param.close();

    //--------------------------------------------------------------------------
    // Layer 2.1
    //--------------------------------------------------------------------------
    ifstream ifs_l2_1_conv1_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer2_1_conv1_weights.bin", ios::in | ios::binary);
    ifstream ifs_l2_1_conv2_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer2_1_conv2_weights.bin", ios::in | ios::binary);
    ifstream ifs_l2_1_conv3_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer2_1_conv3_weights.bin", ios::in | ios::binary);
    
    ifs_l2_1_conv1_param.read((char*)(  *resnet_layer2_1_conv1_weights), RESNET_LAYER2_CONV1_OUT_CH*RESNET_LAYER2_CONV1_IN_CH*sizeof(float));
    ifs_l2_1_conv2_param.read((char*)(***resnet_layer2_1_conv2_weights), RESNET_LAYER2_CONV2_OUT_CH*RESNET_LAYER2_CONV2_IN_CH*3*3*sizeof(float));
    ifs_l2_1_conv3_param.read((char*)(  *resnet_layer2_1_conv3_weights), RESNET_LAYER2_CONV3_OUT_CH*RESNET_LAYER2_CONV3_IN_CH*sizeof(float));
    
    ifs_l2_1_conv1_param.close();
    ifs_l2_1_conv2_param.close();
    ifs_l2_1_conv3_param.close();
    
    ifstream ifs_l2_1_bn1_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer2_1_bn1_params.bin", ios::in | ios::binary);
    ifstream ifs_l2_1_bn2_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer2_1_bn2_params.bin", ios::in | ios::binary);
    ifstream ifs_l2_1_bn3_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer2_1_bn3_params.bin", ios::in | ios::binary);
    
    ifs_l2_1_bn1_param.read((char*)(*resnet_layer2_1_bn1_params), 3*RESNET_LAYER2_CONV1_OUT_CH*sizeof(float));
    ifs_l2_1_bn2_param.read((char*)(*resnet_layer2_1_bn2_params), 3*RESNET_LAYER2_CONV2_OUT_CH*sizeof(float));
    ifs_l2_1_bn3_param.read((char*)(*resnet_layer2_1_bn3_params), 3*RESNET_LAYER2_CONV3_OUT_CH*sizeof(float));
    
    ifs_l2_1_bn1_param.close();
    ifs_l2_1_bn2_param.close();
    ifs_l2_1_bn3_param.close();

    //--------------------------------------------------------------------------
    // Layer 2.2
    //--------------------------------------------------------------------------
    ifstream ifs_l2_2_conv1_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer2_2_conv1_weights.bin", ios::in | ios::binary);
    ifstream ifs_l2_2_conv2_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer2_2_conv2_weights.bin", ios::in | ios::binary);
    ifstream ifs_l2_2_conv3_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer2_2_conv3_weights.bin", ios::in | ios::binary);
    
    ifs_l2_2_conv1_param.read((char*)(  *resnet_layer2_2_conv1_weights), RESNET_LAYER2_CONV1_OUT_CH*RESNET_LAYER2_CONV1_IN_CH*sizeof(float));
    ifs_l2_2_conv2_param.read((char*)(***resnet_layer2_2_conv2_weights), RESNET_LAYER2_CONV2_OUT_CH*RESNET_LAYER2_CONV2_IN_CH*3*3*sizeof(float));
    ifs_l2_2_conv3_param.read((char*)(  *resnet_layer2_2_conv3_weights), RESNET_LAYER2_CONV3_OUT_CH*RESNET_LAYER2_CONV3_IN_CH*sizeof(float));
    
    ifs_l2_2_conv1_param.close();
    ifs_l2_2_conv2_param.close();
    ifs_l2_2_conv3_param.close();
    
    ifstream ifs_l2_2_bn1_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer2_2_bn1_params.bin", ios::in | ios::binary);
    ifstream ifs_l2_2_bn2_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer2_2_bn2_params.bin", ios::in | ios::binary);
    ifstream ifs_l2_2_bn3_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer2_2_bn3_params.bin", ios::in | ios::binary);
    
    ifs_l2_2_bn1_param.read((char*)(*resnet_layer2_2_bn1_params), 3*RESNET_LAYER2_CONV1_OUT_CH*sizeof(float));
    ifs_l2_2_bn2_param.read((char*)(*resnet_layer2_2_bn2_params), 3*RESNET_LAYER2_CONV2_OUT_CH*sizeof(float));
    ifs_l2_2_bn3_param.read((char*)(*resnet_layer2_2_bn3_params), 3*RESNET_LAYER2_CONV3_OUT_CH*sizeof(float));
    
    ifs_l2_2_bn1_param.close();
    ifs_l2_2_bn2_param.close();
    ifs_l2_2_bn3_param.close();

    //--------------------------------------------------------------------------
    // Layer 2.3
    //--------------------------------------------------------------------------
    ifstream ifs_l2_3_conv1_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer2_3_conv1_weights.bin", ios::in | ios::binary);
    ifstream ifs_l2_3_conv2_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer2_3_conv2_weights.bin", ios::in | ios::binary);
    ifstream ifs_l2_3_conv3_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer2_3_conv3_weights.bin", ios::in | ios::binary);

    ifs_l2_3_conv1_param.read((char*)(  *resnet_layer2_3_conv1_weights), RESNET_LAYER2_CONV1_OUT_CH*RESNET_LAYER2_CONV1_IN_CH*sizeof(float));
    ifs_l2_3_conv2_param.read((char*)(***resnet_layer2_3_conv2_weights), RESNET_LAYER2_CONV2_OUT_CH*RESNET_LAYER2_CONV2_IN_CH*3*3*sizeof(float));
    ifs_l2_3_conv3_param.read((char*)(  *resnet_layer2_3_conv3_weights), RESNET_LAYER2_CONV3_OUT_CH*RESNET_LAYER2_CONV3_IN_CH*sizeof(float));
    
    ifs_l2_3_conv1_param.close();
    ifs_l2_3_conv2_param.close();
    ifs_l2_3_conv3_param.close();

    ifstream ifs_l2_3_bn1_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer2_3_bn1_params.bin", ios::in | ios::binary);
    ifstream ifs_l2_3_bn2_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer2_3_bn2_params.bin", ios::in | ios::binary);
    ifstream ifs_l2_3_bn3_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer2_3_bn3_params.bin", ios::in | ios::binary);
    
    ifs_l2_3_bn1_param.read((char*)(*resnet_layer2_3_bn1_params), 3*RESNET_LAYER2_CONV1_OUT_CH*sizeof(float));
    ifs_l2_3_bn2_param.read((char*)(*resnet_layer2_3_bn2_params), 3*RESNET_LAYER2_CONV2_OUT_CH*sizeof(float));
    ifs_l2_3_bn3_param.read((char*)(*resnet_layer2_3_bn3_params), 3*RESNET_LAYER2_CONV3_OUT_CH*sizeof(float));
    
    ifs_l2_3_bn1_param.close();
    ifs_l2_3_bn2_param.close();
    ifs_l2_3_bn3_param.close();    

    //--------------------------------------------------------------------------
    // Layer 3.0
    //--------------------------------------------------------------------------
    ifstream ifs_l3_0_conv1_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer3_0_conv1_weights.bin", ios::in | ios::binary);
    ifstream ifs_l3_0_conv2_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer3_0_conv2_weights.bin", ios::in | ios::binary);
    ifstream ifs_l3_0_conv3_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer3_0_conv3_weights.bin", ios::in | ios::binary);
    
    ifs_l3_0_conv1_param.read((char*)(  *resnet_layer3_0_conv1_weights), RESNET_LAYER3_0_CONV1_OUT_CH*RESNET_LAYER3_0_CONV1_IN_CH*sizeof(float));
    ifs_l3_0_conv2_param.read((char*)(***resnet_layer3_0_conv2_weights), RESNET_LAYER3_0_CONV2_OUT_CH*RESNET_LAYER3_0_CONV2_IN_CH*3*3*sizeof(float));
    ifs_l3_0_conv3_param.read((char*)(  *resnet_layer3_0_conv3_weights), RESNET_LAYER3_0_CONV3_OUT_CH*RESNET_LAYER3_0_CONV3_IN_CH*sizeof(float));
    
    ifs_l3_0_conv1_param.close();
    ifs_l3_0_conv2_param.close();
    ifs_l3_0_conv3_param.close();
    
    ifstream ifs_l3_0_bn1_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer3_0_bn1_params.bin", ios::in | ios::binary);
    ifstream ifs_l3_0_bn2_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer3_0_bn2_params.bin", ios::in | ios::binary);
    ifstream ifs_l3_0_bn3_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer3_0_bn3_params.bin", ios::in | ios::binary);
    
    ifs_l3_0_bn1_param.read((char*)(*resnet_layer3_0_bn1_params), 3*RESNET_LAYER3_0_CONV1_OUT_CH*sizeof(float));
    ifs_l3_0_bn2_param.read((char*)(*resnet_layer3_0_bn2_params), 3*RESNET_LAYER3_0_CONV2_OUT_CH*sizeof(float));
    ifs_l3_0_bn3_param.read((char*)(*resnet_layer3_0_bn3_params), 3*RESNET_LAYER3_0_CONV3_OUT_CH*sizeof(float));
    
    ifs_l3_0_bn1_param.close();
    ifs_l3_0_bn2_param.close();
    ifs_l3_0_bn3_param.close();

    ifstream ifs_l3_0_downsample_0_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer3_0_downsample_0_weights.bin", ios::in | ios::binary);
    ifstream ifs_l3_0_downsample_1_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer3_0_downsample_1_params.bin", ios::in | ios::binary);

    ifs_l3_0_downsample_0_param.read((char*)(*resnet_layer3_0_downsample_0_weights), RESNET_LAYER3_0_DS_OUT_CH*RESNET_LAYER3_0_DS_IN_CH*sizeof(float));
    ifs_l3_0_downsample_1_param.read((char*)(*resnet_layer3_0_downsample_1_params), 3*RESNET_LAYER3_0_DS_OUT_CH*sizeof(float));

    ifs_l3_0_downsample_0_param.close();
    ifs_l3_0_downsample_1_param.close();
    
    //--------------------------------------------------------------------------
    // Layer 3.1
    //--------------------------------------------------------------------------
    ifstream ifs_l3_1_conv1_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer3_1_conv1_weights.bin", ios::in | ios::binary);
    ifstream ifs_l3_1_conv2_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer3_1_conv2_weights.bin", ios::in | ios::binary);
    ifstream ifs_l3_1_conv3_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer3_1_conv3_weights.bin", ios::in | ios::binary);

    ifs_l3_1_conv1_param.read((char*)(  *resnet_layer3_1_conv1_weights), RESNET_LAYER3_CONV1_OUT_CH*RESNET_LAYER3_CONV1_IN_CH*sizeof(float));
    ifs_l3_1_conv2_param.read((char*)(***resnet_layer3_1_conv2_weights), RESNET_LAYER3_CONV2_OUT_CH*RESNET_LAYER3_CONV2_IN_CH*3*3*sizeof(float));
    ifs_l3_1_conv3_param.read((char*)(  *resnet_layer3_1_conv3_weights), RESNET_LAYER3_CONV3_OUT_CH*RESNET_LAYER3_CONV3_IN_CH*sizeof(float));
    
    ifs_l3_1_conv1_param.close();
    ifs_l3_1_conv2_param.close();
    ifs_l3_1_conv3_param.close();

    ifstream ifs_l3_1_bn1_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer3_1_bn1_params.bin", ios::in | ios::binary);
    ifstream ifs_l3_1_bn2_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer3_1_bn2_params.bin", ios::in | ios::binary);
    ifstream ifs_l3_1_bn3_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer3_1_bn3_params.bin", ios::in | ios::binary);

    ifs_l3_1_bn1_param.read((char*)(*resnet_layer3_1_bn1_params), 3*RESNET_LAYER3_CONV1_OUT_CH*sizeof(float));
    ifs_l3_1_bn2_param.read((char*)(*resnet_layer3_1_bn2_params), 3*RESNET_LAYER3_CONV2_OUT_CH*sizeof(float));
    ifs_l3_1_bn3_param.read((char*)(*resnet_layer3_1_bn3_params), 3*RESNET_LAYER3_CONV3_OUT_CH*sizeof(float));
    
    ifs_l3_1_bn1_param.close();
    ifs_l3_1_bn2_param.close();
    ifs_l3_1_bn3_param.close();

    //--------------------------------------------------------------------------
    // Layer 3.2
    //--------------------------------------------------------------------------
    ifstream ifs_l3_2_conv1_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer3_2_conv1_weights.bin", ios::in | ios::binary);
    ifstream ifs_l3_2_conv2_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer3_2_conv2_weights.bin", ios::in | ios::binary);
    ifstream ifs_l3_2_conv3_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer3_2_conv3_weights.bin", ios::in | ios::binary);
    
    ifs_l3_2_conv1_param.read((char*)(  *resnet_layer3_2_conv1_weights), RESNET_LAYER3_CONV1_OUT_CH*RESNET_LAYER3_CONV1_IN_CH*sizeof(float));
    ifs_l3_2_conv2_param.read((char*)(***resnet_layer3_2_conv2_weights), RESNET_LAYER3_CONV2_OUT_CH*RESNET_LAYER3_CONV2_IN_CH*3*3*sizeof(float));
    ifs_l3_2_conv3_param.read((char*)(  *resnet_layer3_2_conv3_weights), RESNET_LAYER3_CONV3_OUT_CH*RESNET_LAYER3_CONV3_IN_CH*sizeof(float));

    ifs_l3_2_conv1_param.close();
    ifs_l3_2_conv2_param.close();
    ifs_l3_2_conv3_param.close();
    
    ifstream ifs_l3_2_bn1_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer3_2_bn1_params.bin", ios::in | ios::binary);
    ifstream ifs_l3_2_bn2_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer3_2_bn2_params.bin", ios::in | ios::binary);
    ifstream ifs_l3_2_bn3_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer3_2_bn3_params.bin", ios::in | ios::binary);

    ifs_l3_2_bn1_param.read((char*)(*resnet_layer3_2_bn1_params), 3*RESNET_LAYER3_CONV1_OUT_CH*sizeof(float));
    ifs_l3_2_bn2_param.read((char*)(*resnet_layer3_2_bn2_params), 3*RESNET_LAYER3_CONV2_OUT_CH*sizeof(float));
    ifs_l3_2_bn3_param.read((char*)(*resnet_layer3_2_bn3_params), 3*RESNET_LAYER3_CONV3_OUT_CH*sizeof(float));
    
    ifs_l3_2_bn1_param.close();
    ifs_l3_2_bn2_param.close();
    ifs_l3_2_bn3_param.close();

    //--------------------------------------------------------------------------
    // Layer 3.3
    //--------------------------------------------------------------------------
    ifstream ifs_l3_3_conv1_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer3_3_conv1_weights.bin", ios::in | ios::binary);
    ifstream ifs_l3_3_conv2_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer3_3_conv2_weights.bin", ios::in | ios::binary);
    ifstream ifs_l3_3_conv3_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer3_3_conv3_weights.bin", ios::in | ios::binary);

    ifs_l3_3_conv1_param.read((char*)(  *resnet_layer3_3_conv1_weights), RESNET_LAYER3_CONV1_OUT_CH*RESNET_LAYER3_CONV1_IN_CH*sizeof(float));
    ifs_l3_3_conv2_param.read((char*)(***resnet_layer3_3_conv2_weights), RESNET_LAYER3_CONV2_OUT_CH*RESNET_LAYER3_CONV2_IN_CH*3*3*sizeof(float));
    ifs_l3_3_conv3_param.read((char*)(  *resnet_layer3_3_conv3_weights), RESNET_LAYER3_CONV3_OUT_CH*RESNET_LAYER3_CONV3_IN_CH*sizeof(float));

    ifs_l3_3_conv1_param.close();
    ifs_l3_3_conv2_param.close();
    ifs_l3_3_conv3_param.close();
    
    ifstream ifs_l3_3_bn1_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer3_3_bn1_params.bin", ios::in | ios::binary);
    ifstream ifs_l3_3_bn2_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer3_3_bn2_params.bin", ios::in | ios::binary);
    ifstream ifs_l3_3_bn3_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer3_3_bn3_params.bin", ios::in | ios::binary);
    
    ifs_l3_3_bn1_param.read((char*)(*resnet_layer3_3_bn1_params), 3*RESNET_LAYER3_CONV1_OUT_CH*sizeof(float));
    ifs_l3_3_bn2_param.read((char*)(*resnet_layer3_3_bn2_params), 3*RESNET_LAYER3_CONV2_OUT_CH*sizeof(float));
    ifs_l3_3_bn3_param.read((char*)(*resnet_layer3_3_bn3_params), 3*RESNET_LAYER3_CONV3_OUT_CH*sizeof(float));
    
    ifs_l3_3_bn1_param.close();
    ifs_l3_3_bn2_param.close();
    ifs_l3_3_bn3_param.close();

    //--------------------------------------------------------------------------
    // Layer 3.4
    //--------------------------------------------------------------------------
    ifstream ifs_l3_4_conv1_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer3_4_conv1_weights.bin", ios::in | ios::binary);
    ifstream ifs_l3_4_conv2_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer3_4_conv2_weights.bin", ios::in | ios::binary);
    ifstream ifs_l3_4_conv3_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer3_4_conv3_weights.bin", ios::in | ios::binary);

    ifs_l3_4_conv1_param.read((char*)(  *resnet_layer3_4_conv1_weights), RESNET_LAYER3_CONV1_OUT_CH*RESNET_LAYER3_CONV1_IN_CH*sizeof(float));
    ifs_l3_4_conv2_param.read((char*)(***resnet_layer3_4_conv2_weights), RESNET_LAYER3_CONV2_OUT_CH*RESNET_LAYER3_CONV2_IN_CH*3*3*sizeof(float));
    ifs_l3_4_conv3_param.read((char*)(  *resnet_layer3_4_conv3_weights), RESNET_LAYER3_CONV3_OUT_CH*RESNET_LAYER3_CONV3_IN_CH*sizeof(float));
    
    ifs_l3_4_conv1_param.close();
    ifs_l3_4_conv2_param.close();
    ifs_l3_4_conv3_param.close();

    ifstream ifs_l3_4_bn1_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer3_4_bn1_params.bin", ios::in | ios::binary);
    ifstream ifs_l3_4_bn2_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer3_4_bn2_params.bin", ios::in | ios::binary);
    ifstream ifs_l3_4_bn3_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer3_4_bn3_params.bin", ios::in | ios::binary);
    
    ifs_l3_4_bn1_param.read((char*)(*resnet_layer3_4_bn1_params), 3*RESNET_LAYER3_CONV1_OUT_CH*sizeof(float));
    ifs_l3_4_bn2_param.read((char*)(*resnet_layer3_4_bn2_params), 3*RESNET_LAYER3_CONV2_OUT_CH*sizeof(float));
    ifs_l3_4_bn3_param.read((char*)(*resnet_layer3_4_bn3_params), 3*RESNET_LAYER3_CONV3_OUT_CH*sizeof(float));
    
    ifs_l3_4_bn1_param.close();
    ifs_l3_4_bn2_param.close();
    ifs_l3_4_bn3_param.close();

    //--------------------------------------------------------------------------
    // Layer 3.5
    //--------------------------------------------------------------------------
    ifstream ifs_l3_5_conv1_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer3_5_conv1_weights.bin", ios::in | ios::binary);
    ifstream ifs_l3_5_conv2_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer3_5_conv2_weights.bin", ios::in | ios::binary);
    ifstream ifs_l3_5_conv3_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer3_5_conv3_weights.bin", ios::in | ios::binary);
    
    ifs_l3_5_conv1_param.read((char*)(  *resnet_layer3_5_conv1_weights), RESNET_LAYER3_CONV1_OUT_CH*RESNET_LAYER3_CONV1_IN_CH*sizeof(float));
    ifs_l3_5_conv2_param.read((char*)(***resnet_layer3_5_conv2_weights), RESNET_LAYER3_CONV2_OUT_CH*RESNET_LAYER3_CONV2_IN_CH*3*3*sizeof(float));
    ifs_l3_5_conv3_param.read((char*)(  *resnet_layer3_5_conv3_weights), RESNET_LAYER3_CONV3_OUT_CH*RESNET_LAYER3_CONV3_IN_CH*sizeof(float));
    
    ifs_l3_5_conv1_param.close();
    ifs_l3_5_conv2_param.close();
    ifs_l3_5_conv3_param.close();

    ifstream ifs_l3_5_bn1_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer3_5_bn1_params.bin", ios::in | ios::binary);
    ifstream ifs_l3_5_bn2_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer3_5_bn2_params.bin", ios::in | ios::binary);
    ifstream ifs_l3_5_bn3_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer3_5_bn3_params.bin", ios::in | ios::binary);
    
    ifs_l3_5_bn1_param.read((char*)(*resnet_layer3_5_bn1_params), 3*RESNET_LAYER3_CONV1_OUT_CH*sizeof(float));
    ifs_l3_5_bn2_param.read((char*)(*resnet_layer3_5_bn2_params), 3*RESNET_LAYER3_CONV2_OUT_CH*sizeof(float));
    ifs_l3_5_bn3_param.read((char*)(*resnet_layer3_5_bn3_params), 3*RESNET_LAYER3_CONV3_OUT_CH*sizeof(float));
    
    ifs_l3_5_bn1_param.close();
    ifs_l3_5_bn2_param.close();
    ifs_l3_5_bn3_param.close();
    
    //--------------------------------------------------------------------------
    // Layer 4.0
    //--------------------------------------------------------------------------
    ifstream ifs_l4_0_conv1_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer4_0_conv1_weights.bin", ios::in | ios::binary);
    ifstream ifs_l4_0_conv2_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer4_0_conv2_weights.bin", ios::in | ios::binary);
    ifstream ifs_l4_0_conv3_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer4_0_conv3_weights.bin", ios::in | ios::binary);

    ifs_l4_0_conv1_param.read((char*)(  *resnet_layer4_0_conv1_weights), RESNET_LAYER4_0_CONV1_OUT_CH*RESNET_LAYER4_0_CONV1_IN_CH*sizeof(float));
    ifs_l4_0_conv2_param.read((char*)(***resnet_layer4_0_conv2_weights), RESNET_LAYER4_0_CONV2_OUT_CH*RESNET_LAYER4_0_CONV2_IN_CH*3*3*sizeof(float));
    ifs_l4_0_conv3_param.read((char*)(  *resnet_layer4_0_conv3_weights), RESNET_LAYER4_0_CONV3_OUT_CH*RESNET_LAYER4_0_CONV3_IN_CH*sizeof(float));

    ifs_l4_0_conv1_param.close();
    ifs_l4_0_conv2_param.close();
    ifs_l4_0_conv3_param.close();

    ifstream ifs_l4_0_bn1_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer4_0_bn1_params.bin", ios::in | ios::binary);
    ifstream ifs_l4_0_bn2_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer4_0_bn2_params.bin", ios::in | ios::binary);
    ifstream ifs_l4_0_bn3_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer4_0_bn3_params.bin", ios::in | ios::binary);

    ifs_l4_0_bn1_param.read((char*)(*resnet_layer4_0_bn1_params), 3*RESNET_LAYER4_0_CONV1_OUT_CH*sizeof(float));
    ifs_l4_0_bn2_param.read((char*)(*resnet_layer4_0_bn2_params), 3*RESNET_LAYER4_0_CONV2_OUT_CH*sizeof(float));
    ifs_l4_0_bn3_param.read((char*)(*resnet_layer4_0_bn3_params), 3*RESNET_LAYER4_0_CONV3_OUT_CH*sizeof(float));

    ifs_l4_0_bn1_param.close();
    ifs_l4_0_bn2_param.close();
    ifs_l4_0_bn3_param.close();

    ifstream ifs_l4_0_downsample_0_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer4_0_downsample_0_weights.bin", ios::in | ios::binary);
    ifstream ifs_l4_0_downsample_1_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer4_0_downsample_1_params.bin", ios::in | ios::binary);

    ifs_l4_0_downsample_0_param.read((char*)(*resnet_layer4_0_downsample_0_weights), RESNET_LAYER4_0_DS_OUT_CH*RESNET_LAYER4_0_DS_IN_CH*sizeof(float));
    ifs_l4_0_downsample_1_param.read((char*)(*resnet_layer4_0_downsample_1_params), 3*RESNET_LAYER4_0_DS_OUT_CH*sizeof(float));

    ifs_l4_0_downsample_0_param.close();
    ifs_l4_0_downsample_1_param.close();
    
    //--------------------------------------------------------------------------
    // Layer 4.1
    //--------------------------------------------------------------------------
    ifstream ifs_l4_1_conv1_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer4_1_conv1_weights.bin", ios::in | ios::binary);
    ifstream ifs_l4_1_conv2_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer4_1_conv2_weights.bin", ios::in | ios::binary);
    ifstream ifs_l4_1_conv3_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer4_1_conv3_weights.bin", ios::in | ios::binary);

    ifs_l4_1_conv1_param.read((char*)(  *resnet_layer4_1_conv1_weights), RESNET_LAYER4_CONV1_OUT_CH*RESNET_LAYER4_CONV1_IN_CH*sizeof(float));
    ifs_l4_1_conv2_param.read((char*)(***resnet_layer4_1_conv2_weights), RESNET_LAYER4_CONV2_OUT_CH*RESNET_LAYER4_CONV2_IN_CH*3*3*sizeof(float));
    ifs_l4_1_conv3_param.read((char*)(  *resnet_layer4_1_conv3_weights), RESNET_LAYER4_CONV3_OUT_CH*RESNET_LAYER4_CONV3_IN_CH*sizeof(float));

    ifs_l4_1_conv1_param.close();
    ifs_l4_1_conv2_param.close();
    ifs_l4_1_conv3_param.close();

    ifstream ifs_l4_1_bn1_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer4_1_bn1_params.bin", ios::in | ios::binary);
    ifstream ifs_l4_1_bn2_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer4_1_bn2_params.bin", ios::in | ios::binary);
    ifstream ifs_l4_1_bn3_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer4_1_bn3_params.bin", ios::in | ios::binary);

    ifs_l4_1_bn1_param.read((char*)(*resnet_layer4_1_bn1_params), 3*RESNET_LAYER4_CONV1_OUT_CH*sizeof(float));
    ifs_l4_1_bn2_param.read((char*)(*resnet_layer4_1_bn2_params), 3*RESNET_LAYER4_CONV2_OUT_CH*sizeof(float));
    ifs_l4_1_bn3_param.read((char*)(*resnet_layer4_1_bn3_params), 3*RESNET_LAYER4_CONV3_OUT_CH*sizeof(float));

    ifs_l4_1_bn1_param.close();
    ifs_l4_1_bn2_param.close();
    ifs_l4_1_bn3_param.close();

    //--------------------------------------------------------------------------
    // Layer 4.2
    //--------------------------------------------------------------------------
    ifstream ifs_l4_2_conv1_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer4_2_conv1_weights.bin", ios::in | ios::binary);
    ifstream ifs_l4_2_conv2_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer4_2_conv2_weights.bin", ios::in | ios::binary);
    ifstream ifs_l4_2_conv3_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer4_2_conv3_weights.bin", ios::in | ios::binary);

    ifs_l4_2_conv1_param.read((char*)(  *resnet_layer4_2_conv1_weights), RESNET_LAYER4_CONV1_OUT_CH*RESNET_LAYER4_CONV1_IN_CH*sizeof(float));
    ifs_l4_2_conv2_param.read((char*)(***resnet_layer4_2_conv2_weights), RESNET_LAYER4_CONV2_OUT_CH*RESNET_LAYER4_CONV2_IN_CH*3*3*sizeof(float));
    ifs_l4_2_conv3_param.read((char*)(  *resnet_layer4_2_conv3_weights), RESNET_LAYER4_CONV3_OUT_CH*RESNET_LAYER4_CONV3_IN_CH*sizeof(float));

    ifs_l4_2_conv1_param.close();
    ifs_l4_2_conv2_param.close();
    ifs_l4_2_conv3_param.close();

    ifstream ifs_l4_2_bn1_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer4_2_bn1_params.bin", ios::in | ios::binary);
    ifstream ifs_l4_2_bn2_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer4_2_bn2_params.bin", ios::in | ios::binary);
    ifstream ifs_l4_2_bn3_param("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer4_2_bn3_params.bin", ios::in | ios::binary);

    ifs_l4_2_bn1_param.read((char*)(*resnet_layer4_2_bn1_params), 3*RESNET_LAYER4_CONV1_OUT_CH*sizeof(float));
    ifs_l4_2_bn2_param.read((char*)(*resnet_layer4_2_bn2_params), 3*RESNET_LAYER4_CONV2_OUT_CH*sizeof(float));
    ifs_l4_2_bn3_param.read((char*)(*resnet_layer4_2_bn3_params), 3*RESNET_LAYER4_CONV3_OUT_CH*sizeof(float));

    ifs_l4_2_bn1_param.close();
    ifs_l4_2_bn2_param.close();
    ifs_l4_2_bn3_param.close();
}
//--------------------------------------------------------------------------
// Read the reference files into test bench arrays
//--------------------------------------------------------------------------
void read_fpn_params()
{
    //CONV_3x3
    //--------------------FPN 0------------------------------//
    // Weights
    std::ifstream ifs_conv_0_wt("/usr/scratch/akamath47/IP2/bin/fpn_neck/module_neck_fpn_convs_0_conv_weight.bin", ios::in | ios::binary);
    ifs_conv_0_wt.read((char*)(***conv_0_weights), (FPN_CONV_0_OD)*(FPN_CONV_0_ID)*3*3*sizeof(float));
    ifs_conv_0_wt.close();

    // Bias
    std::ifstream ifs_conv_0_bias("/usr/scratch/akamath47/IP2/bin/fpn_neck/module_neck_fpn_convs_0_conv_bias.bin", ios::in | ios::binary);
    ifs_conv_0_bias.read((char*)(conv_0_bias), (FPN_CONV_0_OD)*sizeof(float));
    ifs_conv_0_bias.close();

    // Golden Output
    std::ifstream ifs_conv_0_output_golden("/usr/scratch/akamath47/IP2/bin/fpn_neck/module_neck_fpn_convs_0_conv.bin", ios::in | ios::binary);
    ifs_conv_0_output_golden.read((char*)(**golden_conv_0_golden_output), (FPN_CONV_0_OD)*(FPN_CONV_0_OW)*(FPN_CONV_0_OH)*sizeof(float));    
    ifs_conv_0_output_golden.close();

    //--------------------FPN 1------------------------------//
    // Weights
    std::ifstream ifs_conv_1_wt("/usr/scratch/akamath47/IP2/bin/fpn_neck/module_neck_fpn_convs_1_conv_weight.bin", ios::in | ios::binary);
    ifs_conv_1_wt.read((char*)(***conv_1_weights), (FPN_CONV_1_OD)*(FPN_CONV_1_ID)*3*3*sizeof(float));
    ifs_conv_1_wt.close();

    // Bias
    std::ifstream ifs_conv_1_bias("/usr/scratch/akamath47/IP2/bin/fpn_neck/module_neck_fpn_convs_1_conv_bias.bin", ios::in | ios::binary);
    ifs_conv_1_bias.read((char*)(conv_1_bias), (FPN_CONV_1_OD)*sizeof(float));
    ifs_conv_1_bias.close();

    // Golden Output
    std::ifstream ifs_conv_1_output_golden("/usr/scratch/akamath47/IP2/bin/fpn_neck/module_neck_fpn_convs_1_conv.bin", ios::in | ios::binary);
    ifs_conv_1_output_golden.read((char*)(**golden_conv_1_golden_output), (FPN_CONV_1_OD)*(FPN_CONV_1_OW)*(FPN_CONV_1_OH)*sizeof(float));    
    ifs_conv_1_output_golden.close();

    //--------------------FPN 2------------------------------//
    // Weights
    std::ifstream ifs_conv_2_wt("/usr/scratch/akamath47/IP2/bin/fpn_neck/module_neck_fpn_convs_2_conv_weight.bin", ios::in | ios::binary);
    ifs_conv_2_wt.read((char*)(***conv_2_weights), (FPN_CONV_2_OD)*(FPN_CONV_2_ID)*3*3*sizeof(float));
    ifs_conv_2_wt.close();

    // Bias
    std::ifstream ifs_conv_2_bias("/usr/scratch/akamath47/IP2/bin/fpn_neck/module_neck_fpn_convs_2_conv_bias.bin", ios::in | ios::binary);
    ifs_conv_2_bias.read((char*)(conv_2_bias), (FPN_CONV_2_OD)*sizeof(float));
    ifs_conv_2_bias.close();

    // Golden Output
    std::ifstream ifs_conv_2_output_golden("/usr/scratch/akamath47/IP2/bin/fpn_neck/module_neck_fpn_convs_2_conv.bin", ios::in | ios::binary);
    ifs_conv_2_output_golden.read((char*)(**golden_conv_2_golden_output), (FPN_CONV_2_OD)*(FPN_CONV_2_OW)*(FPN_CONV_2_OH)*sizeof(float));    
    ifs_conv_2_output_golden.close();

    //--------------------FPN 3------------------------------//
    // Weights
    std::ifstream ifs_conv_3_wt("/usr/scratch/akamath47/IP2/bin/fpn_neck/module_neck_fpn_convs_3_conv_weight.bin", ios::in | ios::binary);
    ifs_conv_3_wt.read((char*)(***conv_3_weights), (FPN_CONV_3_OD)*(FPN_CONV_3_ID)*3*3*sizeof(float));
    ifs_conv_3_wt.close();

    // Bias
    std::ifstream ifs_conv_3_bias("/usr/scratch/akamath47/IP2/bin/fpn_neck/module_neck_fpn_convs_3_conv_bias.bin", ios::in | ios::binary);
    ifs_conv_3_bias.read((char*)(conv_3_bias), (FPN_CONV_3_OD)*sizeof(float));
    ifs_conv_3_bias.close();

    // Golden Output
    std::ifstream ifs_conv_3_output_golden("/usr/scratch/akamath47/IP2/bin/fpn_neck/module_neck_fpn_convs_3_conv.bin", ios::in | ios::binary);
    ifs_conv_3_output_golden.read((char*)(**golden_conv_3_golden_output), (FPN_CONV_3_OD)*(FPN_CONV_3_OW)*(FPN_CONV_3_OH)*sizeof(float));    
    ifs_conv_3_output_golden.close();

    //CONV_1x1
    //--------------Lateral 0-----------------------//
    // Input Feature Map
    std::ifstream ifs_lateral_conv_0_input_fm("/usr/scratch/akamath47/IP2/bin/fpn_neck/module_neck_lateral_convs_0_conv_input.bin", ios::in | ios::binary);
    ifs_lateral_conv_0_input_fm.read((char*)(**lateral_conv_0_input_feature_map), (LATERAL_CONV_0_ID)*(LATERAL_CONV_0_IH)*(LATERAL_CONV_0_IW)*sizeof(float));
    ifs_lateral_conv_0_input_fm.close();

    // Weights
    std::ifstream ifs_lateral_conv_0_wt("/usr/scratch/akamath47/IP2/bin/fpn_neck/module_neck_lateral_convs_0_conv_weight.bin", ios::in | ios::binary);
    ifs_lateral_conv_0_wt.read((char*)(***lateral_conv_0_weights), LATERAL_CONV_0_OD*LATERAL_CONV_0_ID*sizeof(float));
    ifs_lateral_conv_0_wt.close();

    // Bias
    std::ifstream ifs_lateral_conv_0_bias("/usr/scratch/akamath47/IP2/bin/fpn_neck/module_neck_lateral_convs_0_conv_bias.bin", ios::in | ios::binary);
    ifs_lateral_conv_0_bias.read((char*)(lateral_conv_0_bias), LATERAL_CONV_0_OD*sizeof(float));
    ifs_lateral_conv_0_bias.close();

    //--------------Lateral 1-----------------------//
    // Input Feature Map
    std::ifstream ifs_lateral_conv_1_input_fm("/usr/scratch/akamath47/IP2/bin/fpn_neck/module_neck_lateral_convs_1_conv_input.bin", ios::in | ios::binary);
    ifs_lateral_conv_1_input_fm.read((char*)(**lateral_conv_1_input_feature_map), (LATERAL_CONV_1_ID)*(LATERAL_CONV_1_IH)*(LATERAL_CONV_1_IW)*sizeof(float));
    ifs_lateral_conv_1_input_fm.close();

    // Weights
    std::ifstream ifs_lateral_conv_1_wt("/usr/scratch/akamath47/IP2/bin/fpn_neck/module_neck_lateral_convs_1_conv_weight.bin", ios::in | ios::binary);
    ifs_lateral_conv_1_wt.read((char*)(***lateral_conv_1_weights), LATERAL_CONV_1_OD*LATERAL_CONV_1_ID*sizeof(float));
    ifs_lateral_conv_1_wt.close();

    // Bias
    std::ifstream ifs_lateral_conv_1_bias("/usr/scratch/akamath47/IP2/bin/fpn_neck/module_neck_lateral_convs_1_conv_bias.bin", ios::in | ios::binary);
    ifs_lateral_conv_1_bias.read((char*)(lateral_conv_1_bias), LATERAL_CONV_1_OD*sizeof(float));
    ifs_lateral_conv_1_bias.close();

    //--------------Lateral 2-----------------------//
    // Input Feature Map
    std::ifstream ifs_lateral_conv_2_input_fm("/usr/scratch/akamath47/IP2/bin/fpn_neck/module_neck_lateral_convs_2_conv_input.bin", ios::in | ios::binary);
    ifs_lateral_conv_2_input_fm.read((char*)(**lateral_conv_2_input_feature_map), (LATERAL_CONV_2_ID)*(LATERAL_CONV_2_IH)*(LATERAL_CONV_2_IW)*sizeof(float));
    ifs_lateral_conv_2_input_fm.close();

    // Weights
    std::ifstream ifs_lateral_conv_2_wt("/usr/scratch/akamath47/IP2/bin/fpn_neck/module_neck_lateral_convs_2_conv_weight.bin", ios::in | ios::binary);
    ifs_lateral_conv_2_wt.read((char*)(***lateral_conv_2_weights), LATERAL_CONV_2_OD*LATERAL_CONV_2_ID*sizeof(float));
    ifs_lateral_conv_2_wt.close();

    // Bias
    std::ifstream ifs_lateral_conv_2_bias("/usr/scratch/akamath47/IP2/bin/fpn_neck/module_neck_lateral_convs_2_conv_bias.bin", ios::in | ios::binary);
    ifs_lateral_conv_2_bias.read((char*)(lateral_conv_2_bias), LATERAL_CONV_2_OD*sizeof(float));
    ifs_lateral_conv_2_bias.close();

    //--------------Lateral 3-----------------------//
    // Input Feature Map
    std::ifstream ifs_lateral_conv_3_input_fm("/usr/scratch/akamath47/IP2/bin/fpn_neck/module_neck_lateral_convs_3_conv_input.bin", ios::in | ios::binary);
    ifs_lateral_conv_3_input_fm.read((char*)(**lateral_conv_3_input_feature_map), (LATERAL_CONV_3_ID)*(LATERAL_CONV_3_IH)*(LATERAL_CONV_3_IW)*sizeof(float));
    ifs_lateral_conv_3_input_fm.close();

    // Weights
    std::ifstream ifs_lateral_conv_3_wt("/usr/scratch/akamath47/IP2/bin/fpn_neck/module_neck_lateral_convs_3_conv_weight.bin", ios::in | ios::binary);
    ifs_lateral_conv_3_wt.read((char*)(***lateral_conv_3_weights), LATERAL_CONV_3_OD*LATERAL_CONV_3_ID*sizeof(float));
    ifs_lateral_conv_3_wt.close();

    // Bias
    std::ifstream ifs_lateral_conv_3_bias("/usr/scratch/akamath47/IP2/bin/fpn_neck/module_neck_lateral_convs_3_conv_bias.bin", ios::in | ios::binary);
    ifs_lateral_conv_3_bias.read((char*)(lateral_conv_3_bias), LATERAL_CONV_3_OD*sizeof(float));
    ifs_lateral_conv_3_bias.close();
}

//--------------------------------------------------------------------------
// Convert the data types of every array element for specified 
// configuration.
//--------------------------------------------------------------------------
void fpn_convert_type()
{
    // Input Feature Map
    cout << "Convert Input Feature Map ... " << endl;
    for(int c = 0; c < LATERAL_CONV_0_ID; c++)
        for(int i = 0; i < LATERAL_CONV_0_IH; i++)
            for(int j = 0; j < LATERAL_CONV_0_IW; j++)
		fixp_lateral_conv_0_input_feature_map[c][i][j] = (fm_t) lateral_conv_0_input_feature_map[c][i][j];
    for(int c = 0; c < LATERAL_CONV_1_ID; c++)
        for(int i = 0; i < LATERAL_CONV_1_IH; i++)
            for(int j = 0; j < LATERAL_CONV_1_IW; j++)
		fixp_lateral_conv_1_input_feature_map[c][i][j] = (fm_t) lateral_conv_1_input_feature_map[c][i][j];
    for(int c = 0; c < LATERAL_CONV_2_ID; c++)
        for(int i = 0; i < LATERAL_CONV_2_IH; i++)
            for(int j = 0; j < LATERAL_CONV_2_IW; j++)
		fixp_lateral_conv_2_input_feature_map[c][i][j] = (fm_t) lateral_conv_2_input_feature_map[c][i][j];
    for(int c = 0; c < LATERAL_CONV_3_ID; c++)
        for(int i = 0; i < LATERAL_CONV_3_IH; i++)
            for(int j = 0; j < LATERAL_CONV_3_IW; j++)
		fixp_lateral_conv_3_input_feature_map[c][i][j] = (fm_t) lateral_conv_3_input_feature_map[c][i][j];
    for(int c = 0; c < FPN_CONV_0_ID; c++)
        for(int i = 0; i < FPN_CONV_0_IH; i++)
            for(int j = 0; j < FPN_CONV_0_IW; j++)
		fixp_conv_0_input_feature_map[c][i][j] = (fm_t) conv_0_input_feature_map[c][i][j];
    for(int c = 0; c < FPN_CONV_1_ID; c++)
        for(int i = 0; i < FPN_CONV_1_IH; i++)
            for(int j = 0; j < FPN_CONV_1_IW; j++)
		fixp_conv_1_input_feature_map[c][i][j] = (fm_t) conv_1_input_feature_map[c][i][j];
    for(int c = 0; c < FPN_CONV_2_ID; c++)
        for(int i = 0; i < FPN_CONV_2_IH; i++)
            for(int j = 0; j < FPN_CONV_2_IW; j++)
		fixp_conv_2_input_feature_map[c][i][j] = (fm_t) conv_2_input_feature_map[c][i][j];
    for(int c = 0; c < FPN_CONV_3_ID; c++)
        for(int i = 0; i < FPN_CONV_3_IH; i++)
            for(int j = 0; j < FPN_CONV_3_IW; j++)
		fixp_conv_3_input_feature_map[c][i][j] = (fm_t) conv_3_input_feature_map[c][i][j];

    // Weights
    cout << "Convert Weights ... " << endl;
    for(int f = 0; f < LATERAL_CONV_0_OD; f++)
        for(int c = 0; c < LATERAL_CONV_0_ID; c++)
		fixp_lateral_conv_0_weights[f][c][0][0] = (wt_t) lateral_conv_0_weights[f][c][0][0];
    for(int f = 0; f < LATERAL_CONV_1_OD; f++)
        for(int c = 0; c < LATERAL_CONV_1_ID; c++)
		fixp_lateral_conv_1_weights[f][c][0][0] = (wt_t) lateral_conv_1_weights[f][c][0][0];
    for(int f = 0; f < LATERAL_CONV_2_OD; f++)
        for(int c = 0; c < LATERAL_CONV_2_ID; c++)
		fixp_lateral_conv_2_weights[f][c][0][0] = (wt_t) lateral_conv_2_weights[f][c][0][0];
    for(int f = 0; f < LATERAL_CONV_3_OD; f++)
        for(int c = 0; c < LATERAL_CONV_3_ID; c++)
		fixp_lateral_conv_3_weights[f][c][0][0] = (wt_t) lateral_conv_3_weights[f][c][0][0];
    for(int f = 0; f < FPN_CONV_0_OD; f++)
        for(int c = 0; c < FPN_CONV_0_ID; c++)
            for(int m = 0; m < 3; m++)
                for(int n =0; n < 3; n++)
			fixp_conv_0_weights[f][c][m][n] = (wt_t) conv_0_weights[f][c][m][n];
    for(int f = 0; f < FPN_CONV_1_OD; f++)
        for(int c = 0; c < FPN_CONV_1_ID; c++)
            for(int m = 0; m < 3; m++)
                for(int n =0; n < 3; n++)
			fixp_conv_1_weights[f][c][m][n] = (wt_t) conv_1_weights[f][c][m][n];
    for(int f = 0; f < FPN_CONV_2_OD; f++)
        for(int c = 0; c < FPN_CONV_2_ID; c++)
            for(int m = 0; m < 3; m++)
                for(int n =0; n < 3; n++)
			fixp_conv_2_weights[f][c][m][n] = (wt_t) conv_2_weights[f][c][m][n];
    for(int f = 0; f < FPN_CONV_3_OD; f++)
        for(int c = 0; c < FPN_CONV_3_ID; c++)
            for(int m = 0; m < 3; m++)
                for(int n =0; n < 3; n++)
			fixp_conv_3_weights[f][c][m][n] = (wt_t) conv_3_weights[f][c][m][n];

    // Bias
    cout << "Convert Biases ... " << endl;
    for(int f = 0; f < LATERAL_CONV_0_OD; f++)
	fixp_lateral_conv_0_bias[f] = (wt_t) lateral_conv_0_bias[f];
    for(int f = 0; f < LATERAL_CONV_1_OD; f++)
	fixp_lateral_conv_1_bias[f] = (wt_t) lateral_conv_1_bias[f];
    for(int f = 0; f < LATERAL_CONV_2_OD; f++)
	fixp_lateral_conv_2_bias[f] = (wt_t) lateral_conv_2_bias[f];
    for(int f = 0; f < LATERAL_CONV_3_OD; f++)
	fixp_lateral_conv_3_bias[f] = (wt_t) lateral_conv_3_bias[f];
    for(int f = 0; f < FPN_CONV_0_OD; f++)
	fixp_conv_0_bias[f] = (wt_t) conv_0_bias[f];
    for(int f = 0; f < FPN_CONV_1_OD; f++)
	fixp_conv_1_bias[f] = (wt_t) conv_1_bias[f];
    for(int f = 0; f < FPN_CONV_2_OD; f++)
	fixp_conv_2_bias[f] = (wt_t) conv_2_bias[f];
    for(int f = 0; f < FPN_CONV_3_OD; f++)
	fixp_conv_3_bias[f] = (wt_t) conv_3_bias[f];
}

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
    //----------------------------------------------------------------------
    // Read and convert RPN Input 0
    //----------------------------------------------------------------------
    ifstream ifs_input_img0("/usr/scratch/pchhatrapati3/hls/inputs/rpninput0.bin", ios::in | ios::binary);
    ifs_input_img0.read((char*)(**rpn_input0), (RPN_CONV_IN_CH)*(RPN_INPUT0_IN_FM_HEIGHT)*(RPN_INPUT0_IN_FM_WIDTH)*sizeof(float));
    ifs_input_img0.close();

    for(int c = 0; c < RPN_CONV_IN_CH; c++)
    {
        for(int h = 0; h < RPN_INPUT0_IN_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RPN_INPUT0_IN_FM_WIDTH; w++)
            {
                rpn_input0_fm[c][h][w] = (fm_t) rpn_input0[c][h][w];
                // cout<<rpn_input0_fm[c][h][w]<<" ";
            }
            // cout<<endl;
        }

    }

    //----------------------------------------------------------------------
    // Read and convert RPN Input 1
    //----------------------------------------------------------------------
    ifstream ifs_input_img1("/usr/scratch/pchhatrapati3/hls/inputs/rpninput1.bin", ios::in | ios::binary);
    ifs_input_img1.read((char*)(**rpn_input1), (RPN_CONV_IN_CH)*(RPN_INPUT1_IN_FM_HEIGHT)*(RPN_INPUT1_IN_FM_WIDTH)*sizeof(float));
    ifs_input_img1.close();

    for(int c = 0; c < RPN_CONV_IN_CH; c++)
    {
        for(int h = 0; h < RPN_INPUT1_IN_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RPN_INPUT1_IN_FM_WIDTH; w++)
            {
                rpn_input1_fm[c][h][w] = (fm_t) rpn_input1[c][h][w];
                
            }
        }

    }

    //----------------------------------------------------------------------
    // Read and convert RPN Input 2
    //----------------------------------------------------------------------
    ifstream ifs_input_img2("/usr/scratch/pchhatrapati3/hls/inputs/rpninput2.bin", ios::in | ios::binary);
    ifs_input_img2.read((char*)(**rpn_input2), (RPN_CONV_IN_CH)*(RPN_INPUT2_IN_FM_HEIGHT)*(RPN_INPUT2_IN_FM_WIDTH)*sizeof(float));
    ifs_input_img2.close();

    for(int c = 0; c < RPN_CONV_IN_CH; c++)
    {
        for(int h = 0; h < RPN_INPUT2_IN_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RPN_INPUT2_IN_FM_WIDTH; w++)
            {
                rpn_input2_fm[c][h][w] = (fm_t) rpn_input2[c][h][w];
            }
        }
    }


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
    
    // Load ResNet-50 convolution weights and batchnorm parameters
    cout << "Reading ResNet-50 params ..." << endl;
    resnet_load_weights();

    // Convert floating point weights to fixed-point
    cout << "Converting ResNet-50 params to fixed-point type ..." << endl;
    resnet_convert_weights_type();
    
    // Read FPN reference files and activations
    cout << "Reading FPN params ..." << endl;
    read_fpn_params();
    
    // Convert to fixed-point types 
    cout << "Converting FPN params to fixed-point type ..." << endl;
    fpn_convert_type();
    
    // RPN
    rpn_load_weights();
    rpn_convert_weights_type();
    rpn_load_inputs();
    
    // Read input image
    ifstream ifs_input_img("/usr/scratch/akamath47/IP2/bin/resnet_backbone/qdtrack_image0.bin", ios::in | ios::binary);
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
              resnet_layer4_output_fm,

              fixp_lateral_conv_3_input_feature_map,
                fixp_lateral_conv_3_weights,
                fixp_lateral_conv_3_bias,
    		fixp_lateral_conv_2_input_feature_map,
                fixp_lateral_conv_2_weights,
                fixp_lateral_conv_2_bias,
    		fixp_lateral_conv_1_input_feature_map,
                fixp_lateral_conv_1_weights,
                fixp_lateral_conv_1_bias,
    		fixp_lateral_conv_0_input_feature_map,
                fixp_lateral_conv_0_weights,
                fixp_lateral_conv_0_bias,
                fixp_conv_3_weights,
                fixp_conv_3_bias,
                fixp_conv_3_output_feature_map,
                fixp_conv_2_weights,
                fixp_conv_2_bias,
                fixp_conv_2_output_feature_map,
                fixp_conv_1_weights,
                fixp_conv_1_bias,
                fixp_conv_1_output_feature_map,
                fixp_conv_0_weights,
                fixp_conv_0_bias,
                fixp_conv_0_output_feature_map,

    rpn_topk_index0_2,
    rpn_topk_index1_2,
    rpn_topk_index2_2,

    rpn_anchor0_reg_fm_2,
    rpn_anchor1_reg_fm_2,
    rpn_anchor2_reg_fm_2,

    rpn_anchor0_cls_fm_2,
    rpn_anchor1_cls_fm_2,
    rpn_anchor2_cls_fm_2,

    rpn_input0_fm,
    rpn_input1_fm,
    rpn_input2_fm,
    rpn_input3_fm,
    rpn_input4_fm,

    //Weights and Bias for convolutions
    rpn_conv_weight,
    rpn_conv_bias,
    rpn_cls_weight,
    rpn_cls_bias,
    rpn_reg_weight,
    rpn_reg_bias,

    rpn_output0_cls_fm,
    rpn_output1_cls_fm,
    rpn_output2_cls_fm,
    rpn_output3_cls_fm,
    rpn_output4_cls_fm,

    rpn_output0_reg_fm,
    rpn_output1_reg_fm,
    rpn_output2_reg_fm,
    rpn_output3_reg_fm,
    rpn_output4_reg_fm,

    rpn_output0_fm,
    rpn_output1_fm,
    rpn_output2_fm,
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
    
    ifstream ifs_l4_output_golden("/usr/scratch/akamath47/IP2/bin/resnet_backbone/layer4_2_relu.bin", ios::in | ios::binary);
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

        //-----------------VERIFICATION--------------------//
    std::cout << "Compute MSE LAYER_0 .... " << std::endl;
    for(int f = 0; f < FPN_CONV_0_OD; f++)
    {
        for(int h = 0; h < FPN_CONV_0_OH; h++)
        {
            for(int w = 0; w < FPN_CONV_0_OW; w++)
            {
                mse += std::pow((golden_conv_0_golden_output[f][h][w] - (float) fixp_conv_0_output_feature_map[f][h][w]), 2);
            }
        }
        //std::cout << "Golden Output: " << golden_conv_0_golden_output[f][0][0] << std::endl;
        //std::cout << "Actual Output: " << (float) fixp_conv_0_output_feature_map[f][0][0] << std::endl;
        //std::cout << std::endl;
    }
    
    mse = mse / (FPN_CONV_0_OD * FPN_CONV_0_OH * FPN_CONV_0_OW);
    std::cout << "FPN_CONVS_0 Output MSE:  " << mse << std::endl;
    std::cout << "FPN_CONVS_0 Processing Complete!" << std::endl << std::endl;

    mse = 0.0;    
    std::cout << "Compute MSE LAYER_1 .... " << std::endl;
    for(int f = 0; f < FPN_CONV_1_OD; f++)
    {
        for(int h = 0; h < FPN_CONV_1_OH; h++)
        {
            for(int w = 0; w < FPN_CONV_1_OW; w++)
            {
                mse += std::pow((golden_conv_1_golden_output[f][h][w] - (float) fixp_conv_1_output_feature_map[f][h][w]), 2);
            }
        }
        //std::cout << "Golden Output: " << golden_conv_1_golden_output[f][0][0] << std::endl;
        //std::cout << "Actual Output: " << (float) fixp_conv_1_output_feature_map[f][0][0] << std::endl;
        //std::cout << std::endl;
    }
    
    mse = mse / (FPN_CONV_1_OD * FPN_CONV_1_OH * FPN_CONV_1_OW);
    std::cout << "FPN_CONVS_1 Output MSE:  " << mse << std::endl;
    std::cout << "FPN_CONVS_1 Processing Complete!" << std::endl << std::endl;
    
    mse = 0.0;    
    std::cout << "Compute MSE LAYER_2 .... " << std::endl;
    for(int f = 0; f < FPN_CONV_2_OD; f++)
    {
        for(int h = 0; h < FPN_CONV_2_OH; h++)
        {
            for(int w = 0; w < FPN_CONV_2_OW; w++)
            {
                mse += std::pow((golden_conv_2_golden_output[f][h][w] - (float) fixp_conv_2_output_feature_map[f][h][w]), 2);
            }
        }
        //std::cout << "Golden Output: " << golden_conv_2_golden_output[f][0][0] << std::endl;
        //std::cout << "Actual Output: " << (float) fixp_conv_2_output_feature_map[f][0][0] << std::endl;
        //std::cout << std::endl;
    }
    
    mse = mse / (FPN_CONV_2_OD * FPN_CONV_2_OH * FPN_CONV_2_OW);
    std::cout << "FPN_CONVS_2 Output MSE:  " << mse << std::endl;
    std::cout << "FPN_CONVS_2 Processing Complete!" << std::endl << std::endl;
    
    mse = 0.0;    
    std::cout << "Compute MSE LAYER_3 .... " << std::endl;
    for(int f = 0; f < FPN_CONV_3_OD; f++)
    {
        for(int h = 0; h < FPN_CONV_3_OH; h++)
        {
            for(int w = 0; w < FPN_CONV_3_OW; w++)
            {
                mse += std::pow((golden_conv_3_golden_output[f][h][w] - (float) fixp_conv_3_output_feature_map[f][h][w]), 2);
            }
        }
        //std::cout << "Golden Output: " << golden_conv_3_golden_output[f][0][0] << std::endl;
        //std::cout << "Actual Output: " << (float) fixp_conv_3_output_feature_map[f][0][0] << std::endl;
        //std::cout << std::endl;
    }
    
    mse = mse / (FPN_CONV_3_OD * FPN_CONV_3_OH * FPN_CONV_3_OW);
    std::cout << "FPN_CONVS_3 Output MSE:  " << mse << std::endl;
    std::cout << "FPN_CONVS_3 Processing Complete!" << std::endl << std::endl;

    // RPN
    //----------------------------------------------------------------------
    // Check mse for RPN conv 0
    //----------------------------------------------------------------------
    
    ifstream ifs_l1_output_golden0("/usr/scratch/pchhatrapati3/hls/inputs/rpnoutput0.bin", ios::in | ios::binary);
    ifs_l1_output_golden0.read((char*)(**fl_rpn_output0_fm), (RPN_CONV_IN_CH)*(RPN_INPUT0_IN_FM_HEIGHT)*(RPN_INPUT0_IN_FM_WIDTH)*sizeof(float));    
    ifs_l1_output_golden0.close();
    
    for(int f = 0; f < RPN_CONV_IN_CH; f++)
    {
        for(int h = 0; h < RPN_INPUT0_IN_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RPN_INPUT0_IN_FM_WIDTH; w++)
            {
                mse += std::pow((fl_rpn_output0_fm[f][h][w] - (float) rpn_output0_fm[f][h][w]), 2);
                // cout<<rpn_output0_fm[f][h][w]<<" ";
            }
            // cout<<endl;
        }
        // cout << "Golden Output: " << fl_rpn_output0_fm[f][0][0] << std::endl;
        // cout << "Actual Output: " << rpn_output0_fm[f][0][0] << std::endl;
        // cout << std::endl;
    }
    
    mse = mse / ((RPN_CONV_IN_CH)*(RPN_INPUT0_IN_FM_HEIGHT)*(RPN_INPUT0_IN_FM_WIDTH));
    std::cout << "RPN CONV 0 MSE:  " << mse << std::endl;
    

    //----------------------------------------------------------------------
    // Check mse for RPN conv 1
    //----------------------------------------------------------------------
    mse = 0;

    ifstream ifs_l1_output_golden1("/usr/scratch/pchhatrapati3/hls/inputs/rpnoutput1.bin", ios::in | ios::binary);
    ifs_l1_output_golden1.read((char*)(**fl_rpn_output1_fm), (RPN_CONV_IN_CH)*(RPN_INPUT1_IN_FM_HEIGHT)*(RPN_INPUT1_IN_FM_WIDTH)*sizeof(float));    
    ifs_l1_output_golden1.close();
    
    for(int f = 0; f < RPN_CONV_IN_CH; f++)
    {
        for(int h = 0; h < RPN_INPUT1_IN_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RPN_INPUT1_IN_FM_WIDTH; w++)
            {
                mse += std::pow((fl_rpn_output1_fm[f][h][w] - (float) rpn_output1_fm[f][h][w]), 2);
                // cout<<rpn_output0_fm[f][h][w]<<" ";
            }
            // cout<<endl;
        }
        // cout << "Golden Output: " << fl_rpn_output0_fm[f][0][0] << std::endl;
        // cout << "Actual Output: " << rpn_output0_fm[f][0][0] << std::endl;
        // cout << std::endl;
    }
    
    mse = mse / ((RPN_CONV_IN_CH)*(RPN_INPUT1_IN_FM_HEIGHT)*(RPN_INPUT1_IN_FM_WIDTH));
    std::cout << "RPN CONV 1 MSE:  " << mse << std::endl;
    

    //----------------------------------------------------------------------
    // Check mse for RPN conv 2
    //----------------------------------------------------------------------
    mse = 0;

    ifstream ifs_l2_output_golden2("/usr/scratch/pchhatrapati3/hls/inputs/rpnoutput2.bin", ios::in | ios::binary);
    ifs_l2_output_golden2.read((char*)(**fl_rpn_output2_fm), (RPN_CONV_IN_CH)*(RPN_INPUT2_IN_FM_HEIGHT)*(RPN_INPUT2_IN_FM_WIDTH)*sizeof(float));    
    ifs_l2_output_golden2.close();
    
    for(int f = 0; f < RPN_CONV_IN_CH; f++)
    {
        for(int h = 0; h < RPN_INPUT2_IN_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RPN_INPUT2_IN_FM_WIDTH; w++)
            {
                mse += std::pow((fl_rpn_output2_fm[f][h][w] - (float) rpn_output2_fm[f][h][w]), 2);
                // cout<<rpn_output0_fm[f][h][w]<<" ";
            }
            // cout<<endl;
        }
        // cout << "Golden Output: " << fl_rpn_output0_fm[f][0][0] << std::endl;
        // cout << "Actual Output: " << rpn_output0_fm[f][0][0] << std::endl;
        // cout << std::endl;
    }
    
    mse = mse / ((RPN_CONV_IN_CH)*(RPN_INPUT2_IN_FM_HEIGHT)*(RPN_INPUT2_IN_FM_WIDTH));
    std::cout << "RPN CONV 2 MSE:  " << mse << std::endl;
    

    //----------------------------------------------------------------------
    // Check mse for RPN conv 3
    //----------------------------------------------------------------------
    mse = 0;

    ifstream ifs_l3_output_golden3("/usr/scratch/pchhatrapati3/hls/inputs/rpnoutput3.bin", ios::in | ios::binary);
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
    std::cout << "RPN CONV 3 MSE:  " << mse << std::endl;

    //----------------------------------------------------------------------
    // Check mse for RPN conv 4
    //----------------------------------------------------------------------
    mse = 0;

    ifstream ifs_l4_output_golden4("/usr/scratch/pchhatrapati3/hls/inputs/rpnoutput4.bin", ios::in | ios::binary);
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
    std::cout << "RPN CONV 4 MSE:  " << mse << std::endl;

    //----------------------------------------------------------------------
    // Check mse for RPN cls 0
    //----------------------------------------------------------------------
    
    ifstream ifs_l1_output_golden_cls0("/usr/scratch/pchhatrapati3/hls/inputs/rpnclsoutput0.bin", ios::in | ios::binary);
    ifs_l1_output_golden_cls0.read((char*)(**fl_rpn_output0_cls_fm), (RPN_CLS_OUT_CH)*(RPN_INPUT0_IN_FM_HEIGHT)*(RPN_INPUT0_IN_FM_WIDTH)*sizeof(float));    
    ifs_l1_output_golden_cls0.close();
    
    for(int f = 0; f < RPN_CLS_OUT_CH; f++)
    {
        for(int h = 0; h < RPN_INPUT0_IN_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RPN_INPUT0_IN_FM_WIDTH; w++)
            {
                mse += std::pow((fl_rpn_output0_cls_fm[f][h][w] - (float) rpn_output0_cls_fm[f][h][w]), 2);
                // cout<<rpn_output0_fm[f][h][w]<<" ";
            }
            // cout<<endl;
        }
        // cout << "Golden Output: " << fl_rpn_output0_fm[f][0][0] << std::endl;
        // cout << "Actual Output: " << rpn_output0_fm[f][0][0] << std::endl;
        // cout << std::endl;
    }
    
    mse = mse / ((RPN_CLS_OUT_CH)*(RPN_INPUT0_IN_FM_HEIGHT)*(RPN_INPUT0_IN_FM_WIDTH));
    std::cout << "RPN CLS 0 MSE:  " << mse << std::endl;
    

    //----------------------------------------------------------------------
    // Check mse for RPN CLS 1
    //----------------------------------------------------------------------
    mse = 0;

    ifstream ifs_l1_output_golden_cls1("/usr/scratch/pchhatrapati3/hls/inputs/rpnclsoutput1.bin", ios::in | ios::binary);
    ifs_l1_output_golden_cls1.read((char*)(**fl_rpn_output1_cls_fm), (RPN_CLS_OUT_CH)*(RPN_INPUT1_IN_FM_HEIGHT)*(RPN_INPUT1_IN_FM_WIDTH)*sizeof(float));    
    ifs_l1_output_golden_cls1.close();
    
    for(int f = 0; f < RPN_CLS_OUT_CH; f++)
    {
        for(int h = 0; h < RPN_INPUT1_IN_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RPN_INPUT1_IN_FM_WIDTH; w++)
            {
                mse += std::pow((fl_rpn_output1_cls_fm[f][h][w] - (float) rpn_output1_cls_fm[f][h][w]), 2);
                // cout<<rpn_output0_fm[f][h][w]<<" ";
            }
            // cout<<endl;
        }
        // cout << "Golden Output: " << fl_rpn_output0_fm[f][0][0] << std::endl;
        // cout << "Actual Output: " << rpn_output0_fm[f][0][0] << std::endl;
        // cout << std::endl;
    }
    
    mse = mse / ((RPN_CLS_OUT_CH)*(RPN_INPUT1_IN_FM_HEIGHT)*(RPN_INPUT1_IN_FM_WIDTH));
    std::cout << "RPN CLS 1 MSE:  " << mse << std::endl;
    

    //----------------------------------------------------------------------
    // Check mse for RPN CLS 2
    //----------------------------------------------------------------------
    mse = 0;

    ifstream ifs_l2_output_golden_cls2("/usr/scratch/pchhatrapati3/hls/inputs/rpnclsoutput2.bin", ios::in | ios::binary);
    ifs_l2_output_golden_cls2.read((char*)(**fl_rpn_output2_cls_fm), (RPN_CLS_OUT_CH)*(RPN_INPUT2_IN_FM_HEIGHT)*(RPN_INPUT2_IN_FM_WIDTH)*sizeof(float));    
    ifs_l2_output_golden_cls2.close();
    
    for(int f = 0; f < RPN_CLS_OUT_CH; f++)
    {
        for(int h = 0; h < RPN_INPUT2_IN_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RPN_INPUT2_IN_FM_WIDTH; w++)
            {
                mse += std::pow((fl_rpn_output2_cls_fm[f][h][w] - (float) rpn_output2_cls_fm[f][h][w]), 2);
                // cout<<rpn_output0_fm[f][h][w]<<" ";
            }
            // cout<<endl;
        }
        // cout << "Golden Output: " << fl_rpn_output0_fm[f][0][0] << std::endl;
        // cout << "Actual Output: " << rpn_output0_fm[f][0][0] << std::endl;
        // cout << std::endl;
    }
    
    mse = mse / ((RPN_CLS_OUT_CH)*(RPN_INPUT2_IN_FM_HEIGHT)*(RPN_INPUT2_IN_FM_WIDTH));
    std::cout << "RPN CLS 2 MSE:  " << mse << std::endl;
    

    //----------------------------------------------------------------------
    // Check mse for RPN cls 3
    //----------------------------------------------------------------------
    mse = 0;

    ifstream ifs_l3_output_golden_cls3("/usr/scratch/pchhatrapati3/hls/inputs/rpnclsoutput3.bin", ios::in | ios::binary);
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
    std::cout << "RPN CLS 3 MSE:  " << mse << std::endl;

    //----------------------------------------------------------------------
    // Check mse for RPN cls 4
    //----------------------------------------------------------------------
    mse = 0;

    ifstream ifs_l4_output_golden_cls4("/usr/scratch/pchhatrapati3/hls/inputs/rpnclsoutput4.bin", ios::in | ios::binary);
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
    std::cout << "RPN CLS 4 MSE:  " << mse << std::endl;

        //----------------------------------------------------------------------
    // Check mse for RPN reg 0
    //----------------------------------------------------------------------
    
    ifstream ifs_l1_output_golden_reg0("/usr/scratch/pchhatrapati3/hls/inputs/rpnregoutput0.bin", ios::in | ios::binary);
    ifs_l1_output_golden_reg0.read((char*)(**fl_rpn_output0_reg_fm), (RPN_REG_OUT_CH)*(RPN_INPUT0_IN_FM_HEIGHT)*(RPN_INPUT0_IN_FM_WIDTH)*sizeof(float));    
    ifs_l1_output_golden_reg0.close();
    
    for(int f = 0; f < RPN_REG_OUT_CH; f++)
    {
        for(int h = 0; h < RPN_INPUT0_IN_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RPN_INPUT0_IN_FM_WIDTH; w++)
            {
                mse += std::pow((fl_rpn_output0_reg_fm[f][h][w] - (float) rpn_output0_reg_fm[f][h][w]), 2);
                // cout<<rpn_output0_fm[f][h][w]<<" ";
            }
            // cout<<endl;
        }
        // cout << "Golden Output: " << fl_rpn_output0_fm[f][0][0] << std::endl;
        // cout << "Actual Output: " << rpn_output0_fm[f][0][0] << std::endl;
        // cout << std::endl;
    }
    
    mse = mse / ((RPN_REG_OUT_CH)*(RPN_INPUT0_IN_FM_HEIGHT)*(RPN_INPUT0_IN_FM_WIDTH));
    std::cout << "RPN REG 0 MSE:  " << mse << std::endl;
    

    //----------------------------------------------------------------------
    // Check mse for RPN REG 1
    //----------------------------------------------------------------------
    mse = 0;

    ifstream ifs_l1_output_golden_reg1("/usr/scratch/pchhatrapati3/hls/inputs/rpnregoutput1.bin", ios::in | ios::binary);
    ifs_l1_output_golden_reg1.read((char*)(**fl_rpn_output1_reg_fm), (RPN_REG_OUT_CH)*(RPN_INPUT1_IN_FM_HEIGHT)*(RPN_INPUT1_IN_FM_WIDTH)*sizeof(float));    
    ifs_l1_output_golden_reg1.close();
    
    for(int f = 0; f < RPN_REG_OUT_CH; f++)
    {
        for(int h = 0; h < RPN_INPUT1_IN_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RPN_INPUT1_IN_FM_WIDTH; w++)
            {
                mse += std::pow((fl_rpn_output1_reg_fm[f][h][w] - (float) rpn_output1_reg_fm[f][h][w]), 2);
                // cout<<rpn_output0_fm[f][h][w]<<" ";
            }
            // cout<<endl;
        }
        // cout << "Golden Output: " << fl_rpn_output0_fm[f][0][0] << std::endl;
        // cout << "Actual Output: " << rpn_output0_fm[f][0][0] << std::endl;
        // cout << std::endl;
    }
    
    mse = mse / ((RPN_REG_OUT_CH)*(RPN_INPUT1_IN_FM_HEIGHT)*(RPN_INPUT1_IN_FM_WIDTH));
    std::cout << "RPN REG 1 MSE:  " << mse << std::endl;
    

    //----------------------------------------------------------------------
    // Check mse for RPN REG 2
    //----------------------------------------------------------------------
    mse = 0;

    ifstream ifs_l2_output_golden_reg2("/usr/scratch/pchhatrapati3/hls/inputs/rpnregoutput2.bin", ios::in | ios::binary);
    ifs_l2_output_golden_reg2.read((char*)(**fl_rpn_output2_reg_fm), (RPN_REG_OUT_CH)*(RPN_INPUT2_IN_FM_HEIGHT)*(RPN_INPUT2_IN_FM_WIDTH)*sizeof(float));    
    ifs_l2_output_golden_reg2.close();
    
    for(int f = 0; f < RPN_REG_OUT_CH; f++)
    {
        for(int h = 0; h < RPN_INPUT2_IN_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RPN_INPUT2_IN_FM_WIDTH; w++)
            {
                mse += std::pow((fl_rpn_output2_reg_fm[f][h][w] - (float) rpn_output2_reg_fm[f][h][w]), 2);
                // cout<<rpn_output0_fm[f][h][w]<<" ";
            }
            // cout<<endl;
        }
        // cout << "Golden Output: " << fl_rpn_output0_fm[f][0][0] << std::endl;
        // cout << "Actual Output: " << rpn_output0_fm[f][0][0] << std::endl;
        // cout << std::endl;
    }
    
    mse = mse / ((RPN_REG_OUT_CH)*(RPN_INPUT2_IN_FM_HEIGHT)*(RPN_INPUT2_IN_FM_WIDTH));
    std::cout << "RPN REG 2 MSE:  " << mse << std::endl;
    

    //----------------------------------------------------------------------
    // Check mse for RPN reg 3
    //----------------------------------------------------------------------
    mse = 0;

    ifstream ifs_l3_output_golden_reg3("/usr/scratch/pchhatrapati3/hls/inputs/rpnregoutput3.bin", ios::in | ios::binary);
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
    std::cout << "RPN REG 3 MSE:  " << mse << std::endl;

    //----------------------------------------------------------------------
    // Check mse for RPN reg 4
    //----------------------------------------------------------------------
    mse = 0;

    ifstream ifs_l4_output_golden_reg4("/usr/scratch/pchhatrapati3/hls/inputs/rpnregoutput4.bin", ios::in | ios::binary);
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
    std::cout << "RPN REG 4 MSE:  " << mse << std::endl;
    
    // TODO
    mse = 0;

    ifstream ifs_l4_output_golden_bbox("/usr/scratch/pchhatrapati3/hls/inputs/rpndets.bin", ios::in | ios::binary);
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
    std::cout << "RPN DETS MSE:  " << mse << std::endl;


#endif // }

    return 0;
}
