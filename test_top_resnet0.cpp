#include "qdtrack_resnet0.h"

void test_resnet_top_0 (
    fm_t   resnet_layer0_input_fm[RESNET_LAYER0_CONV1_IN_CH][RESNET_LAYER0_IN_FM_HEIGHT][RESNET_LAYER0_IN_FM_WIDTH],
    wt_t   resnet_layer0_conv1_weights[RESNET_LAYER0_CONV1_OUT_CH][RESNET_LAYER0_CONV1_IN_CH][7][7],
    wt_t   resnet_layer0_bn1_params[3][RESNET_LAYER0_CONV1_OUT_CH],
    fm_t   resnet_layer0_output_fm[RESNET_LAYER0_CONV1_OUT_CH][RESNET_LAYER0_OUT_FM_HEIGHT][RESNET_LAYER0_OUT_FM_WIDTH],

    fm_t   resnet_layer1_input_fm[RESNET_LAYER1_0_CONV1_IN_CH][RESNET_LAYER1_0_FM_HEIGHT][RESNET_LAYER1_0_FM_WIDTH]
)
{
 
  resnet_top_0 (
    resnet_layer0_input_fm,
    resnet_layer0_conv1_weights,
    resnet_layer0_bn1_params,
    resnet_layer0_output_fm,

    resnet_layer1_input_fm
);

}
