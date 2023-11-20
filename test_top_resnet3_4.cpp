#include "qdtrack_resnet3_4.h"

void test_resnet_top_3_4 (
    wt_t   resnet_layer3_4_conv1_weights[RESNET_LAYER3_CONV1_OUT_CH][RESNET_LAYER3_CONV1_IN_CH],
	wt_t   resnet_layer3_4_bn1_params[4][RESNET_LAYER3_CONV1_OUT_CH],
    wt_t   resnet_layer3_4_conv2_weights[RESNET_LAYER3_CONV2_OUT_CH][RESNET_LAYER3_CONV2_IN_CH][3][3],
	wt_t   resnet_layer3_4_bn2_params[4][RESNET_LAYER3_CONV2_OUT_CH],
    wt_t   resnet_layer3_4_conv3_weights[RESNET_LAYER3_CONV3_OUT_CH][RESNET_LAYER3_CONV3_IN_CH],
	wt_t   resnet_layer3_4_bn3_params[4][RESNET_LAYER3_CONV3_OUT_CH]
)
{
  resnet_top_3_4 (
    resnet_layer3_4_conv1_weights,
	resnet_layer3_4_bn1_params,
    resnet_layer3_4_conv2_weights,
	resnet_layer3_4_bn2_params,
    resnet_layer3_4_conv3_weights,
	resnet_layer3_4_bn3_params
);

}
