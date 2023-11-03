#include "qdtrack_resnet.h"

void test_resnet_top_1 (
    fm_t   resnet_layer0_input_fm[RESNET_LAYER0_CONV1_IN_CH][RESNET_LAYER0_IN_FM_HEIGHT][RESNET_LAYER0_IN_FM_WIDTH],
    wt_t   resnet_layer0_conv1_weights[RESNET_LAYER0_CONV1_OUT_CH][RESNET_LAYER0_CONV1_IN_CH][7][7],
    wt_t   resnet_layer0_bn1_params[3][RESNET_LAYER0_CONV1_OUT_CH],
    fm_t   resnet_layer0_output_fm[RESNET_LAYER0_CONV1_OUT_CH][RESNET_LAYER0_OUT_FM_HEIGHT][RESNET_LAYER0_OUT_FM_WIDTH],

    fm_t   resnet_layer1_input_fm[RESNET_LAYER1_0_CONV1_IN_CH][RESNET_LAYER1_0_FM_HEIGHT][RESNET_LAYER1_0_FM_WIDTH],
    wt_t   resnet_layer1_0_conv1_weights[RESNET_LAYER1_0_CONV1_OUT_CH][RESNET_LAYER1_0_CONV1_IN_CH],
    wt_t   resnet_layer1_0_bn1_params[4][RESNET_LAYER1_0_CONV1_OUT_CH],
    wt_t   resnet_layer1_0_conv2_weights[RESNET_LAYER1_0_CONV2_OUT_CH][RESNET_LAYER1_0_CONV2_IN_CH][3][3],
    wt_t   resnet_layer1_0_bn2_params[4][RESNET_LAYER1_0_CONV2_OUT_CH],
    wt_t   resnet_layer1_0_conv3_weights[RESNET_LAYER1_0_CONV3_OUT_CH][RESNET_LAYER1_0_CONV3_IN_CH],
    wt_t   resnet_layer1_0_bn3_params[4][RESNET_LAYER1_0_CONV3_OUT_CH],
    wt_t   resnet_layer1_0_downsample_0_weights[RESNET_LAYER1_0_DS_OUT_CH][RESNET_LAYER1_0_DS_IN_CH],
    wt_t   resnet_layer1_0_downsample_1_params[4][RESNET_LAYER1_0_DS_OUT_CH],
    wt_t   resnet_layer1_1_conv1_weights[RESNET_LAYER1_CONV1_OUT_CH][RESNET_LAYER1_CONV1_IN_CH],
    wt_t   resnet_layer1_1_bn1_params[4][RESNET_LAYER1_CONV1_OUT_CH],
    wt_t   resnet_layer1_1_conv2_weights[RESNET_LAYER1_CONV2_OUT_CH][RESNET_LAYER1_CONV2_IN_CH][3][3],
    wt_t   resnet_layer1_1_bn2_params[4][RESNET_LAYER1_CONV2_OUT_CH],
    wt_t   resnet_layer1_1_conv3_weights[RESNET_LAYER1_CONV3_OUT_CH][RESNET_LAYER1_CONV3_IN_CH],
    wt_t   resnet_layer1_1_bn3_params[4][RESNET_LAYER1_CONV3_OUT_CH],
    wt_t   resnet_layer1_2_conv1_weights[RESNET_LAYER1_CONV1_OUT_CH][RESNET_LAYER1_CONV1_IN_CH],
    wt_t   resnet_layer1_2_bn1_params[4][RESNET_LAYER1_CONV1_OUT_CH],
    wt_t   resnet_layer1_2_conv2_weights[RESNET_LAYER1_CONV2_OUT_CH][RESNET_LAYER1_CONV2_IN_CH][3][3],
    wt_t   resnet_layer1_2_bn2_params[4][RESNET_LAYER1_CONV2_OUT_CH],
    wt_t   resnet_layer1_2_conv3_weights[RESNET_LAYER1_CONV3_OUT_CH][RESNET_LAYER1_CONV3_IN_CH],
    wt_t   resnet_layer1_2_bn3_params[4][RESNET_LAYER1_CONV3_OUT_CH],
    fm_t   resnet_layer1_output_fm[RESNET_LAYER1_0_DS_OUT_CH][RESNET_LAYER1_0_FM_HEIGHT][RESNET_LAYER1_0_FM_WIDTH],

    fm_t   resnet_layer2_input_fm[RESNET_LAYER2_0_CONV1_IN_CH][RESNET_LAYER2_0_FM_HEIGHT][RESNET_LAYER2_0_FM_WIDTH]
)
{
 
  resnet_top_1 (
    resnet_layer0_input_fm,
    resnet_layer0_conv1_weights,
    resnet_layer0_bn1_params,
    resnet_layer0_output_fm,

    resnet_layer1_input_fm,
    resnet_layer1_0_conv1_weights,
    resnet_layer1_0_bn1_params,
    resnet_layer1_0_conv2_weights,
    resnet_layer1_0_bn2_params,
    resnet_layer1_0_conv3_weights,
    resnet_layer1_0_bn3_params,
    resnet_layer1_0_downsample_0_weights,
    resnet_layer1_0_downsample_1_params,
    resnet_layer1_1_conv1_weights,
    resnet_layer1_1_bn1_params,
    resnet_layer1_1_conv2_weights,
    resnet_layer1_1_bn2_params,
    resnet_layer1_1_conv3_weights,
    resnet_layer1_1_bn3_params,
    resnet_layer1_2_conv1_weights,
    resnet_layer1_2_bn1_params,
    resnet_layer1_2_conv2_weights,
    resnet_layer1_2_bn2_params,
    resnet_layer1_2_conv3_weights,
    resnet_layer1_2_bn3_params,
    resnet_layer1_output_fm,

    resnet_layer2_input_fm
);

}
