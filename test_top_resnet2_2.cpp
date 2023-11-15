#include "qdtrack_resnet2_2.h"

void test_resnet_top_2_2 (
    wt_t   resnet_layer2_2_conv1_weights[RESNET_LAYER2_CONV1_OUT_CH][RESNET_LAYER2_CONV1_IN_CH],
	wt_t   resnet_layer2_2_bn1_params[3][RESNET_LAYER2_CONV1_OUT_CH],
    wt_t   resnet_layer2_2_conv2_weights[RESNET_LAYER2_CONV2_OUT_CH][RESNET_LAYER2_CONV2_IN_CH][3][3],
	wt_t   resnet_layer2_2_bn2_params[3][RESNET_LAYER2_CONV2_OUT_CH],
    wt_t   resnet_layer2_2_conv3_weights[RESNET_LAYER2_CONV3_OUT_CH][RESNET_LAYER2_CONV3_IN_CH],
	wt_t   resnet_layer2_2_bn3_params[3][RESNET_LAYER2_CONV3_OUT_CH]
)
{
  resnet_top_2_2 (
    resnet_layer2_2_conv1_weights,
	resnet_layer2_2_bn1_params,
    resnet_layer2_2_conv2_weights,
	resnet_layer2_2_bn2_params,
    resnet_layer2_2_conv3_weights,
	resnet_layer2_2_bn3_params
);

}
