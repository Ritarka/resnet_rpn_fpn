#include "qdtrack_resnet4_2.h"

void test_resnet_top_4_2 (
    wt_t   resnet_layer4_2_conv1_weights[RESNET_LAYER4_CONV1_OUT_CH][RESNET_LAYER4_CONV1_IN_CH],
	wt_t   resnet_layer4_2_bn1_params[4][RESNET_LAYER4_CONV1_OUT_CH],
    wt_t   resnet_layer4_2_conv2_weights[RESNET_LAYER4_CONV2_OUT_CH][RESNET_LAYER4_CONV2_IN_CH][3][3],
	wt_t   resnet_layer4_2_bn2_params[4][RESNET_LAYER4_CONV2_OUT_CH],
    wt_t   resnet_layer4_2_conv3_weights[RESNET_LAYER4_CONV3_OUT_CH][RESNET_LAYER4_CONV3_IN_CH],
	wt_t   resnet_layer4_2_bn3_params[4][RESNET_LAYER4_CONV3_OUT_CH],
    fm_t   resnet_layer4_output_fm[RESNET_LAYER4_0_DS_OUT_CH][RESNET_LAYER4_0_FM_HEIGHT][RESNET_LAYER4_0_FM_WIDTH]
)
{
  resnet_top_4_2 (
    resnet_layer4_2_conv1_weights,
	resnet_layer4_2_bn1_params,
    resnet_layer4_2_conv2_weights,
	resnet_layer4_2_bn2_params,
    resnet_layer4_2_conv3_weights,
	resnet_layer4_2_bn3_params,
    resnet_layer4_output_fm
);

}
