#include "qdtrack_resnet3_5.h"

void test_resnet_top_3_5 (
    wt_t   resnet_layer3_5_conv1_weights[RESNET_LAYER3_CONV1_OUT_CH][RESNET_LAYER3_CONV1_IN_CH],
	wt_t   resnet_layer3_5_bn1_params[4][RESNET_LAYER3_CONV1_OUT_CH],
    wt_t   resnet_layer3_5_conv2_weights[RESNET_LAYER3_CONV2_OUT_CH][RESNET_LAYER3_CONV2_IN_CH][3][3],
	wt_t   resnet_layer3_5_bn2_params[4][RESNET_LAYER3_CONV2_OUT_CH],
    wt_t   resnet_layer3_5_conv3_weights[RESNET_LAYER3_CONV3_OUT_CH][RESNET_LAYER3_CONV3_IN_CH],
	wt_t   resnet_layer3_5_bn3_params[4][RESNET_LAYER3_CONV3_OUT_CH],
    fm_t   resnet_layer3_output_fm[RESNET_LAYER3_0_DS_OUT_CH][RESNET_LAYER3_0_FM_HEIGHT][RESNET_LAYER3_0_FM_WIDTH],
        
    fm_t   resnet_layer4_input_fm[RESNET_LAYER4_0_CONV1_IN_CH][RESNET_LAYER4_0_FM_HEIGHT][RESNET_LAYER4_0_FM_WIDTH]
)
{
  resnet_top_3_5 (
    resnet_layer3_5_conv1_weights,
	resnet_layer3_5_bn1_params,
    resnet_layer3_5_conv2_weights,
	resnet_layer3_5_bn2_params,
    resnet_layer3_5_conv3_weights,
	resnet_layer3_5_bn3_params,
    resnet_layer3_output_fm,
    
    resnet_layer4_input_fm
);

}
