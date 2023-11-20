#include "hls_stream.h"
#include "qdtrack_resnet4_0.h"

#include "resnet_layers4_0.cpp"

fm_t in_fm_buf[RESNET_IN_BUF_CH][RESNET_IN_BUF_ROWS + 5][RESNET_IN_BUF_COLS + 5];
fm_t out_fm_buf[RESNET_OUT_BUF_CH][RESNET_OUT_BUF_ROWS][RESNET_OUT_BUF_COLS];
fm_t partial_out_fm_buf[RESNET_OUT_BUF_CH][RESNET_OUT_BUF_ROWS][RESNET_OUT_BUF_COLS];
fm_t ds_fm_buf[RESNET_OUT_BUF_CH][RESNET_OUT_BUF_ROWS][RESNET_OUT_BUF_COLS];
fm_t resnet_layer_out_fm[2048][184][320];
fm_t resnet_layer_in_fm[2048][184][320];
fm_t ds_fm[2048][184][320];

wt_t weight_buf_1x1[RESNET_IN_BUF_CH];
wt_t weight_buf_3x3[RESNET_IN_BUF_CH][3][3];
wt_t weight_buf_7x7[RESNET_IN_BUF_CH][7][7];

wt_t param_buf[3][RESNET_OUT_BUF_CH];

void resnet_top_4_0 (
    fm_t   resnet_layer3_output_fm[RESNET_LAYER3_0_DS_OUT_CH][RESNET_LAYER3_0_FM_HEIGHT][RESNET_LAYER3_0_FM_WIDTH],
        
    fm_t   resnet_layer4_input_fm[RESNET_LAYER4_0_CONV1_IN_CH][RESNET_LAYER4_0_FM_HEIGHT][RESNET_LAYER4_0_FM_WIDTH],
    wt_t   resnet_layer4_0_conv1_weights[RESNET_LAYER4_0_CONV1_OUT_CH][RESNET_LAYER4_0_CONV1_IN_CH],
	wt_t   resnet_layer4_0_bn1_params[4][RESNET_LAYER4_0_CONV1_OUT_CH],
    wt_t   resnet_layer4_0_conv2_weights[RESNET_LAYER4_0_CONV2_OUT_CH][RESNET_LAYER4_0_CONV2_IN_CH][3][3],
	wt_t   resnet_layer4_0_bn2_params[4][RESNET_LAYER4_0_CONV2_OUT_CH],
    wt_t   resnet_layer4_0_conv3_weights[RESNET_LAYER4_0_CONV3_OUT_CH][RESNET_LAYER4_0_CONV3_IN_CH],
	wt_t   resnet_layer4_0_bn3_params[4][RESNET_LAYER4_0_CONV3_OUT_CH],
    wt_t   resnet_layer4_0_downsample_0_weights[RESNET_LAYER4_0_DS_OUT_CH][RESNET_LAYER4_0_DS_IN_CH],
	wt_t   resnet_layer4_0_downsample_1_params[4][RESNET_LAYER4_0_DS_OUT_CH]
)
{

    //----------------------------------------------------------------------
    // Layer 4_0
    //----------------------------------------------------------------------
    std::cout << "Begin processing Layer 4_0..." << std::endl;
    
    resnet_layer4_0( 
            resnet_layer4_input_fm,
            resnet_layer4_0_conv1_weights,         resnet_layer4_0_bn1_params,
            resnet_layer4_0_conv2_weights,         resnet_layer4_0_bn2_params,
            resnet_layer4_0_conv3_weights,         resnet_layer4_0_bn3_params,
            resnet_layer4_0_downsample_0_weights,  resnet_layer4_0_downsample_1_params
    );
    
    std::cout << "Layer 4_0 Processing Complete!" << std::endl << std::endl;
 
}
