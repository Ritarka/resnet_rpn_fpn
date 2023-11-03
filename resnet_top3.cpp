#include "hls_stream.h"
#include "qdtrack_resnet3.h"

#include "resnet_layers3.cpp"

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

void resnet_top_3 (
    fm_t   resnet_layer2_output_fm[RESNET_LAYER2_0_DS_OUT_CH][RESNET_LAYER2_0_FM_HEIGHT][RESNET_LAYER2_0_FM_WIDTH],

    fm_t   resnet_layer3_input_fm[RESNET_LAYER3_0_CONV1_IN_CH][RESNET_LAYER3_0_FM_HEIGHT][RESNET_LAYER3_0_FM_WIDTH],
    wt_t   resnet_layer3_0_conv1_weights[RESNET_LAYER3_0_CONV1_OUT_CH][RESNET_LAYER3_0_CONV1_IN_CH],
	wt_t   resnet_layer3_0_bn1_params[4][RESNET_LAYER3_0_CONV1_OUT_CH],
    wt_t   resnet_layer3_0_conv2_weights[RESNET_LAYER3_0_CONV2_OUT_CH][RESNET_LAYER3_0_CONV2_IN_CH][3][3],
	wt_t   resnet_layer3_0_bn2_params[4][RESNET_LAYER3_0_CONV2_OUT_CH],
    wt_t   resnet_layer3_0_conv3_weights[RESNET_LAYER3_0_CONV3_OUT_CH][RESNET_LAYER3_0_CONV3_IN_CH],
	wt_t   resnet_layer3_0_bn3_params[4][RESNET_LAYER3_0_CONV3_OUT_CH],
    wt_t   resnet_layer3_0_downsample_0_weights[RESNET_LAYER3_0_DS_OUT_CH][RESNET_LAYER3_0_DS_IN_CH],
	wt_t   resnet_layer3_0_downsample_1_params[4][RESNET_LAYER3_0_DS_OUT_CH],
    wt_t   resnet_layer3_1_conv1_weights[RESNET_LAYER3_CONV1_OUT_CH][RESNET_LAYER3_CONV1_IN_CH],
	wt_t   resnet_layer3_1_bn1_params[4][RESNET_LAYER3_CONV1_OUT_CH],
    wt_t   resnet_layer3_1_conv2_weights[RESNET_LAYER3_CONV2_OUT_CH][RESNET_LAYER3_CONV2_IN_CH][3][3],
	wt_t   resnet_layer3_1_bn2_params[4][RESNET_LAYER3_CONV2_OUT_CH],
    wt_t   resnet_layer3_1_conv3_weights[RESNET_LAYER3_CONV3_OUT_CH][RESNET_LAYER3_CONV3_IN_CH],
	wt_t   resnet_layer3_1_bn3_params[4][RESNET_LAYER3_CONV3_OUT_CH],
    wt_t   resnet_layer3_2_conv1_weights[RESNET_LAYER3_CONV1_OUT_CH][RESNET_LAYER3_CONV1_IN_CH],
	wt_t   resnet_layer3_2_bn1_params[4][RESNET_LAYER3_CONV1_OUT_CH],
    wt_t   resnet_layer3_2_conv2_weights[RESNET_LAYER3_CONV2_OUT_CH][RESNET_LAYER3_CONV2_IN_CH][3][3],
	wt_t   resnet_layer3_2_bn2_params[4][RESNET_LAYER3_CONV2_OUT_CH],
    wt_t   resnet_layer3_2_conv3_weights[RESNET_LAYER3_CONV3_OUT_CH][RESNET_LAYER3_CONV3_IN_CH],
	wt_t   resnet_layer3_2_bn3_params[4][RESNET_LAYER3_CONV3_OUT_CH],
    wt_t   resnet_layer3_3_conv1_weights[RESNET_LAYER3_CONV1_OUT_CH][RESNET_LAYER3_CONV1_IN_CH],
	wt_t   resnet_layer3_3_bn1_params[4][RESNET_LAYER3_CONV1_OUT_CH],
    wt_t   resnet_layer3_3_conv2_weights[RESNET_LAYER3_CONV2_OUT_CH][RESNET_LAYER3_CONV2_IN_CH][3][3],
	wt_t   resnet_layer3_3_bn2_params[4][RESNET_LAYER3_CONV2_OUT_CH],
    wt_t   resnet_layer3_3_conv3_weights[RESNET_LAYER3_CONV3_OUT_CH][RESNET_LAYER3_CONV3_IN_CH],
	wt_t   resnet_layer3_3_bn3_params[4][RESNET_LAYER3_CONV3_OUT_CH],
    wt_t   resnet_layer3_4_conv1_weights[RESNET_LAYER3_CONV1_OUT_CH][RESNET_LAYER3_CONV1_IN_CH],
	wt_t   resnet_layer3_4_bn1_params[4][RESNET_LAYER3_CONV1_OUT_CH],
    wt_t   resnet_layer3_4_conv2_weights[RESNET_LAYER3_CONV2_OUT_CH][RESNET_LAYER3_CONV2_IN_CH][3][3],
	wt_t   resnet_layer3_4_bn2_params[4][RESNET_LAYER3_CONV2_OUT_CH],
    wt_t   resnet_layer3_4_conv3_weights[RESNET_LAYER3_CONV3_OUT_CH][RESNET_LAYER3_CONV3_IN_CH],
	wt_t   resnet_layer3_4_bn3_params[4][RESNET_LAYER3_CONV3_OUT_CH],
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
    // TODO: Glue-logic until pointer-based update is complete
    for(int c = 0; c < RESNET_LAYER2_0_DS_OUT_CH; c++)
    {
        for(int h = 0; h < RESNET_LAYER2_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RESNET_LAYER2_FM_WIDTH; w++)
            {
                resnet_layer3_input_fm[c][h][w] = resnet_layer2_output_fm[c][h][w];
            }
        }
    }   

    //----------------------------------------------------------------------
    // Layer 3
    //----------------------------------------------------------------------
    std::cout << "Begin processing Layer 3..." << std::endl;
    
    resnet_layer3( resnet_layer3_input_fm,
            resnet_layer3_0_conv1_weights,         resnet_layer3_0_bn1_params,
            resnet_layer3_0_conv2_weights,         resnet_layer3_0_bn2_params,
            resnet_layer3_0_conv3_weights,         resnet_layer3_0_bn3_params,
            resnet_layer3_0_downsample_0_weights,  resnet_layer3_0_downsample_1_params,
            resnet_layer3_1_conv1_weights,         resnet_layer3_1_bn1_params,
            resnet_layer3_1_conv2_weights,         resnet_layer3_1_bn2_params,
            resnet_layer3_1_conv3_weights,         resnet_layer3_1_bn3_params,
            resnet_layer3_2_conv1_weights,         resnet_layer3_2_bn1_params,
            resnet_layer3_2_conv2_weights,         resnet_layer3_2_bn2_params,
            resnet_layer3_2_conv3_weights,         resnet_layer3_2_bn3_params,
            resnet_layer3_3_conv1_weights,         resnet_layer3_3_bn1_params,
            resnet_layer3_3_conv2_weights,         resnet_layer3_3_bn2_params,
            resnet_layer3_3_conv3_weights,         resnet_layer3_3_bn3_params,
            resnet_layer3_4_conv1_weights,         resnet_layer3_4_bn1_params,
            resnet_layer3_4_conv2_weights,         resnet_layer3_4_bn2_params,
            resnet_layer3_4_conv3_weights,         resnet_layer3_4_bn3_params,
            resnet_layer3_5_conv1_weights,         resnet_layer3_5_bn1_params,
            resnet_layer3_5_conv2_weights,         resnet_layer3_5_bn2_params,
            resnet_layer3_5_conv3_weights,         resnet_layer3_5_bn3_params,
            resnet_layer3_output_fm
    );   

    std::cout << "Layer 3 Processing Complete!" << std::endl << std::endl;
    
    // TODO: Glue-logic until pointer-based update is complete
    for(int c = 0; c < RESNET_LAYER3_0_DS_OUT_CH; c++)
    {
        for(int h = 0; h < RESNET_LAYER3_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RESNET_LAYER3_FM_WIDTH; w++)
            {
                resnet_layer4_input_fm[c][h][w] = resnet_layer3_output_fm[c][h][w];
            }
        }
    }
 
}
