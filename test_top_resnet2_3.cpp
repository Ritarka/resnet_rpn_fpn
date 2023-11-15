#include "qdtrack_resnet2_3.h"

void test_resnet_top_2_3 (
    wt_t   resnet_layer2_3_conv1_weights[RESNET_LAYER2_CONV1_OUT_CH][RESNET_LAYER2_CONV1_IN_CH],
	wt_t   resnet_layer2_3_bn1_params[3][RESNET_LAYER2_CONV1_OUT_CH],
    wt_t   resnet_layer2_3_conv2_weights[RESNET_LAYER2_CONV2_OUT_CH][RESNET_LAYER2_CONV2_IN_CH][3][3],
	wt_t   resnet_layer2_3_bn2_params[3][RESNET_LAYER2_CONV2_OUT_CH],
    wt_t   resnet_layer2_3_conv3_weights[RESNET_LAYER2_CONV3_OUT_CH][RESNET_LAYER2_CONV3_IN_CH],
	wt_t   resnet_layer2_3_bn3_params[3][RESNET_LAYER2_CONV3_OUT_CH],
    fm_t   resnet_layer2_output_fm[RESNET_LAYER2_0_DS_OUT_CH][RESNET_LAYER2_0_FM_HEIGHT][RESNET_LAYER2_0_FM_WIDTH]
)
{
  resnet_top_2_3 (
    resnet_layer2_3_conv1_weights,
	resnet_layer2_3_bn1_params,
    resnet_layer2_3_conv2_weights,
	resnet_layer2_3_bn2_params,
    resnet_layer2_3_conv3_weights,
	resnet_layer2_3_bn3_params,
    resnet_layer2_output_fm
);

}
