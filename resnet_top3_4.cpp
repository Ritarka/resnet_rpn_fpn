#include "hls_stream.h"
#include "qdtrack_resnet3_4.h"

#include "resnet_layers3_4.cpp"

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

void resnet_top_3_4 (
    wt_t   resnet_layer3_4_conv1_weights[RESNET_LAYER3_CONV1_OUT_CH][RESNET_LAYER3_CONV1_IN_CH],
	wt_t   resnet_layer3_4_bn1_params[4][RESNET_LAYER3_CONV1_OUT_CH],
    wt_t   resnet_layer3_4_conv2_weights[RESNET_LAYER3_CONV2_OUT_CH][RESNET_LAYER3_CONV2_IN_CH][3][3],
	wt_t   resnet_layer3_4_bn2_params[4][RESNET_LAYER3_CONV2_OUT_CH],
    wt_t   resnet_layer3_4_conv3_weights[RESNET_LAYER3_CONV3_OUT_CH][RESNET_LAYER3_CONV3_IN_CH],
	wt_t   resnet_layer3_4_bn3_params[4][RESNET_LAYER3_CONV3_OUT_CH]
)
{
    //----------------------------------------------------------------------
    // Layer 3
    //----------------------------------------------------------------------
    std::cout << "Begin processing Layer 3_4..." << std::endl;
    
    resnet_layer3_4( 
            resnet_layer3_4_conv1_weights,         resnet_layer3_4_bn1_params,
            resnet_layer3_4_conv2_weights,         resnet_layer3_4_bn2_params,
            resnet_layer3_4_conv3_weights,         resnet_layer3_4_bn3_params
    );   

    std::cout << "Layer 3_4 Processing Complete!" << std::endl << std::endl;
 
}
