#include "qdtrack_resnet4_1.h"

void test_resnet_top_4_1 (
    wt_t   resnet_layer4_1_conv1_weights[RESNET_LAYER4_CONV1_OUT_CH][RESNET_LAYER4_CONV1_IN_CH],
	wt_t   resnet_layer4_1_bn1_params[4][RESNET_LAYER4_CONV1_OUT_CH],
    wt_t   resnet_layer4_1_conv2_weights[RESNET_LAYER4_CONV2_OUT_CH][RESNET_LAYER4_CONV2_IN_CH][3][3],
	wt_t   resnet_layer4_1_bn2_params[4][RESNET_LAYER4_CONV2_OUT_CH],
    wt_t   resnet_layer4_1_conv3_weights[RESNET_LAYER4_CONV3_OUT_CH][RESNET_LAYER4_CONV3_IN_CH],
	wt_t   resnet_layer4_1_bn3_params[4][RESNET_LAYER4_CONV3_OUT_CH]
)
{
  resnet_top_4_1 (
    resnet_layer4_1_conv1_weights,
	resnet_layer4_1_bn1_params,
    resnet_layer4_1_conv2_weights,
	resnet_layer4_1_bn2_params,
    resnet_layer4_1_conv3_weights,
	resnet_layer4_1_bn3_params
);

}
