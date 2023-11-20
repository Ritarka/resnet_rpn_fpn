#include "qdtrack_resnet3_1.h"

void test_resnet_top_3_1 (
    wt_t   resnet_layer3_1_conv1_weights[RESNET_LAYER3_CONV1_OUT_CH][RESNET_LAYER3_CONV1_IN_CH],
	wt_t   resnet_layer3_1_bn1_params[4][RESNET_LAYER3_CONV1_OUT_CH],
    wt_t   resnet_layer3_1_conv2_weights[RESNET_LAYER3_CONV2_OUT_CH][RESNET_LAYER3_CONV2_IN_CH][3][3],
	wt_t   resnet_layer3_1_bn2_params[4][RESNET_LAYER3_CONV2_OUT_CH],
    wt_t   resnet_layer3_1_conv3_weights[RESNET_LAYER3_CONV3_OUT_CH][RESNET_LAYER3_CONV3_IN_CH],
	wt_t   resnet_layer3_1_bn3_params[4][RESNET_LAYER3_CONV3_OUT_CH]
)
{
  resnet_top_3_1 (
    resnet_layer3_1_conv1_weights,
	resnet_layer3_1_bn1_params,
    resnet_layer3_1_conv2_weights,
	resnet_layer3_1_bn2_params,
    resnet_layer3_1_conv3_weights,
	resnet_layer3_1_bn3_params
);

}
