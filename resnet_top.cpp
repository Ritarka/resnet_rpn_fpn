#include "hls_stream.h"
#include "qdtrack.h"
#include "resnet_util.h"

#include "resnet_layer0.cpp"
#include "resnet_layer1.cpp"
#include "resnet_layer2.cpp"
#include "resnet_layer3.cpp"
#include "resnet_layer4.cpp"
// #include "resnet_layers.cpp"

void resnet_top (
    fm_t   resnet_layer0_input_fm[RESNET_LAYER0_CONV1_IN_CH][RESNET_LAYER0_IN_FM_HEIGHT][RESNET_LAYER0_IN_FM_WIDTH],
    wt_t   resnet_layer0_conv1_weights[RESNET_LAYER0_CONV1_OUT_CH][RESNET_LAYER0_CONV1_IN_CH][7][7],
    wt_t   resnet_layer0_bn1_params[3][RESNET_LAYER0_CONV1_OUT_CH],
    fm_t   resnet_layer0_output_fm[RESNET_LAYER0_CONV1_OUT_CH][RESNET_LAYER0_OUT_FM_HEIGHT][RESNET_LAYER0_OUT_FM_WIDTH],

    fm_t   resnet_layer1_input_fm[RESNET_LAYER1_0_CONV1_IN_CH][RESNET_LAYER1_0_FM_HEIGHT][RESNET_LAYER1_0_FM_WIDTH],
    wt_t   resnet_layer1_0_conv1_weights[RESNET_LAYER1_0_CONV1_OUT_CH][RESNET_LAYER1_0_CONV1_IN_CH],
    wt_t   resnet_layer1_0_bn1_params[4][RESNET_LAYER1_0_CONV1_OUT_CH],
    wt_t   resnet_layer1_0_conv2_weights[RESNET_LAYER1_0_CONV2_OUT_CH][RESNET_LAYER1_0_CONV2_IN_CH][3][3],
    wt_t   resnet_layer1_0_bn2_params[4][RESNET_LAYER1_0_CONV2_OUT_CH],
    wt_t   resnet_layer1_0_conv3_weights[RESNET_LAYER1_0_CONV3_OUT_CH][RESNET_LAYER1_0_CONV3_IN_CH],
    wt_t   resnet_layer1_0_bn3_params[4][RESNET_LAYER1_0_CONV3_OUT_CH],
    wt_t   resnet_layer1_0_downsample_0_weights[RESNET_LAYER1_0_DS_OUT_CH][RESNET_LAYER1_0_DS_IN_CH],
    wt_t   resnet_layer1_0_downsample_1_params[4][RESNET_LAYER1_0_DS_OUT_CH],
    wt_t   resnet_layer1_1_conv1_weights[RESNET_LAYER1_CONV1_OUT_CH][RESNET_LAYER1_CONV1_IN_CH],
    wt_t   resnet_layer1_1_bn1_params[4][RESNET_LAYER1_CONV1_OUT_CH],
    wt_t   resnet_layer1_1_conv2_weights[RESNET_LAYER1_CONV2_OUT_CH][RESNET_LAYER1_CONV2_IN_CH][3][3],
    wt_t   resnet_layer1_1_bn2_params[4][RESNET_LAYER1_CONV2_OUT_CH],
    wt_t   resnet_layer1_1_conv3_weights[RESNET_LAYER1_CONV3_OUT_CH][RESNET_LAYER1_CONV3_IN_CH],
    wt_t   resnet_layer1_1_bn3_params[4][RESNET_LAYER1_CONV3_OUT_CH],
    wt_t   resnet_layer1_2_conv1_weights[RESNET_LAYER1_CONV1_OUT_CH][RESNET_LAYER1_CONV1_IN_CH],
    wt_t   resnet_layer1_2_bn1_params[4][RESNET_LAYER1_CONV1_OUT_CH],
    wt_t   resnet_layer1_2_conv2_weights[RESNET_LAYER1_CONV2_OUT_CH][RESNET_LAYER1_CONV2_IN_CH][3][3],
    wt_t   resnet_layer1_2_bn2_params[4][RESNET_LAYER1_CONV2_OUT_CH],
    wt_t   resnet_layer1_2_conv3_weights[RESNET_LAYER1_CONV3_OUT_CH][RESNET_LAYER1_CONV3_IN_CH],
    wt_t   resnet_layer1_2_bn3_params[4][RESNET_LAYER1_CONV3_OUT_CH],
    fm_t   resnet_layer1_output_fm[RESNET_LAYER1_0_DS_OUT_CH][RESNET_LAYER1_0_FM_HEIGHT][RESNET_LAYER1_0_FM_WIDTH],

    fm_t   resnet_layer2_input_fm[RESNET_LAYER2_0_CONV1_IN_CH][RESNET_LAYER2_0_FM_HEIGHT][RESNET_LAYER2_0_FM_WIDTH],
    wt_t   resnet_layer2_0_conv1_weights[RESNET_LAYER2_0_CONV1_OUT_CH][RESNET_LAYER2_0_CONV1_IN_CH],
	wt_t   resnet_layer2_0_bn1_params[3][RESNET_LAYER2_0_CONV1_OUT_CH],
    wt_t   resnet_layer2_0_conv2_weights[RESNET_LAYER2_0_CONV2_OUT_CH][RESNET_LAYER2_0_CONV2_IN_CH][3][3],
	wt_t   resnet_layer2_0_bn2_params[3][RESNET_LAYER2_0_CONV2_OUT_CH],
    wt_t   resnet_layer2_0_conv3_weights[RESNET_LAYER2_0_CONV3_OUT_CH][RESNET_LAYER2_0_CONV3_IN_CH],
	wt_t   resnet_layer2_0_bn3_params[3][RESNET_LAYER2_0_CONV3_OUT_CH],
    wt_t   resnet_layer2_0_downsample_0_weights[RESNET_LAYER2_0_DS_OUT_CH][RESNET_LAYER2_0_DS_IN_CH],
	wt_t   resnet_layer2_0_downsample_1_params[3][RESNET_LAYER2_0_DS_OUT_CH],
    wt_t   resnet_layer2_1_conv1_weights[RESNET_LAYER2_CONV1_OUT_CH][RESNET_LAYER2_CONV1_IN_CH],
	wt_t   resnet_layer2_1_bn1_params[3][RESNET_LAYER2_CONV1_OUT_CH],
    wt_t   resnet_layer2_1_conv2_weights[RESNET_LAYER2_CONV2_OUT_CH][RESNET_LAYER2_CONV2_IN_CH][3][3],
	wt_t   resnet_layer2_1_bn2_params[3][RESNET_LAYER2_CONV2_OUT_CH],
    wt_t   resnet_layer2_1_conv3_weights[RESNET_LAYER2_CONV3_OUT_CH][RESNET_LAYER2_CONV3_IN_CH],
	wt_t   resnet_layer2_1_bn3_params[3][RESNET_LAYER2_CONV3_OUT_CH],
    wt_t   resnet_layer2_2_conv1_weights[RESNET_LAYER2_CONV1_OUT_CH][RESNET_LAYER2_CONV1_IN_CH],
	wt_t   resnet_layer2_2_bn1_params[3][RESNET_LAYER2_CONV1_OUT_CH],
    wt_t   resnet_layer2_2_conv2_weights[RESNET_LAYER2_CONV2_OUT_CH][RESNET_LAYER2_CONV2_IN_CH][3][3],
	wt_t   resnet_layer2_2_bn2_params[3][RESNET_LAYER2_CONV2_OUT_CH],
    wt_t   resnet_layer2_2_conv3_weights[RESNET_LAYER2_CONV3_OUT_CH][RESNET_LAYER2_CONV3_IN_CH],
	wt_t   resnet_layer2_2_bn3_params[3][RESNET_LAYER2_CONV3_OUT_CH],
    wt_t   resnet_layer2_3_conv1_weights[RESNET_LAYER2_CONV1_OUT_CH][RESNET_LAYER2_CONV1_IN_CH],
	wt_t   resnet_layer2_3_bn1_params[3][RESNET_LAYER2_CONV1_OUT_CH],
    wt_t   resnet_layer2_3_conv2_weights[RESNET_LAYER2_CONV2_OUT_CH][RESNET_LAYER2_CONV2_IN_CH][3][3],
	wt_t   resnet_layer2_3_bn2_params[3][RESNET_LAYER2_CONV2_OUT_CH],
    wt_t   resnet_layer2_3_conv3_weights[RESNET_LAYER2_CONV3_OUT_CH][RESNET_LAYER2_CONV3_IN_CH],
	wt_t   resnet_layer2_3_bn3_params[3][RESNET_LAYER2_CONV3_OUT_CH],
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
        
    fm_t   resnet_layer4_input_fm[RESNET_LAYER4_0_CONV1_IN_CH][RESNET_LAYER4_0_FM_HEIGHT][RESNET_LAYER4_0_FM_WIDTH],
    wt_t   resnet_layer4_0_conv1_weights[RESNET_LAYER4_0_CONV1_OUT_CH][RESNET_LAYER4_0_CONV1_IN_CH],
	wt_t   resnet_layer4_0_bn1_params[4][RESNET_LAYER4_0_CONV1_OUT_CH],
    wt_t   resnet_layer4_0_conv2_weights[RESNET_LAYER4_0_CONV2_OUT_CH][RESNET_LAYER4_0_CONV2_IN_CH][3][3],
	wt_t   resnet_layer4_0_bn2_params[4][RESNET_LAYER4_0_CONV2_OUT_CH],
    wt_t   resnet_layer4_0_conv3_weights[RESNET_LAYER4_0_CONV3_OUT_CH][RESNET_LAYER4_0_CONV3_IN_CH],
	wt_t   resnet_layer4_0_bn3_params[4][RESNET_LAYER4_0_CONV3_OUT_CH],
    wt_t   resnet_layer4_0_downsample_0_weights[RESNET_LAYER4_0_DS_OUT_CH][RESNET_LAYER4_0_DS_IN_CH],
	wt_t   resnet_layer4_0_downsample_1_params[4][RESNET_LAYER4_0_DS_OUT_CH],
    wt_t   resnet_layer4_1_conv1_weights[RESNET_LAYER4_CONV1_OUT_CH][RESNET_LAYER4_CONV1_IN_CH],
	wt_t   resnet_layer4_1_bn1_params[4][RESNET_LAYER4_CONV1_OUT_CH],
    wt_t   resnet_layer4_1_conv2_weights[RESNET_LAYER4_CONV2_OUT_CH][RESNET_LAYER4_CONV2_IN_CH][3][3],
	wt_t   resnet_layer4_1_bn2_params[4][RESNET_LAYER4_CONV2_OUT_CH],
    wt_t   resnet_layer4_1_conv3_weights[RESNET_LAYER4_CONV3_OUT_CH][RESNET_LAYER4_CONV3_IN_CH],
	wt_t   resnet_layer4_1_bn3_params[4][RESNET_LAYER4_CONV3_OUT_CH],
    wt_t   resnet_layer4_2_conv1_weights[RESNET_LAYER4_CONV1_OUT_CH][RESNET_LAYER4_CONV1_IN_CH],
	wt_t   resnet_layer4_2_bn1_params[4][RESNET_LAYER4_CONV1_OUT_CH],
    wt_t   resnet_layer4_2_conv2_weights[RESNET_LAYER4_CONV2_OUT_CH][RESNET_LAYER4_CONV2_IN_CH][3][3],
	wt_t   resnet_layer4_2_bn2_params[4][RESNET_LAYER4_CONV2_OUT_CH],
    wt_t   resnet_layer4_2_conv3_weights[RESNET_LAYER4_CONV3_OUT_CH][RESNET_LAYER4_CONV3_IN_CH],
	wt_t   resnet_layer4_2_bn3_params[4][RESNET_LAYER4_CONV3_OUT_CH],
    fm_t   resnet_layer4_output_fm[RESNET_LAYER4_0_DS_OUT_CH][RESNET_LAYER4_0_FM_HEIGHT][RESNET_LAYER4_0_FM_WIDTH]
)
{
    //----------------------------------------------------------------------
    // Layer 0
    //----------------------------------------------------------------------
    std::cout << "Begin processing Layer 0..." << std::endl;
    
    resnet_layer0( resnet_layer0_input_fm,
            resnet_layer0_conv1_weights,         resnet_layer0_bn1_params, 
            resnet_layer0_output_fm
    );
    
    std::cout << "Layer 0 Processing Complete!" << std::endl << std::endl;
    
    //TODO: Glue-logic until pointer-based update is complete
    for(int c = 0; c < RESNET_LAYER0_CONV1_OUT_CH; c++)
    {
        for(int h = 0; h < RESNET_LAYER0_OUT_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RESNET_LAYER0_OUT_FM_WIDTH; w++)
            {
                resnet_layer1_input_fm[c][h][w] = resnet_layer0_output_fm[c][h][w];
            }
        }
    }    
    
    //----------------------------------------------------------------------
    // Layer 1
    //----------------------------------------------------------------------
    std::cout << "Begin processing Layer 1..." << std::endl;
    
    resnet_layer1( resnet_layer1_input_fm,
            resnet_layer1_0_conv1_weights,         resnet_layer1_0_bn1_params,
            resnet_layer1_0_conv2_weights,         resnet_layer1_0_bn2_params,
            resnet_layer1_0_conv3_weights,         resnet_layer1_0_bn3_params,
            resnet_layer1_0_downsample_0_weights,  resnet_layer1_0_downsample_1_params,
            resnet_layer1_1_conv1_weights,         resnet_layer1_1_bn1_params,
            resnet_layer1_1_conv2_weights,         resnet_layer1_1_bn2_params,
            resnet_layer1_1_conv3_weights,         resnet_layer1_1_bn3_params,
            resnet_layer1_2_conv1_weights,         resnet_layer1_2_bn1_params,
            resnet_layer1_2_conv2_weights,         resnet_layer1_2_bn2_params,
            resnet_layer1_2_conv3_weights,         resnet_layer1_2_bn3_params,
            resnet_layer1_output_fm
    );
    
    std::cout << "Layer 1 Processing Complete!" << std::endl << std::endl;
    
    // TODO: Glue-logic until pointer-based update is complete
    for(int c = 0; c < RESNET_LAYER1_0_DS_OUT_CH; c++)
    {
        for(int h = 0; h < RESNET_LAYER1_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RESNET_LAYER1_FM_WIDTH; w++)
            {
                resnet_layer2_input_fm[c][h][w] = resnet_layer1_output_fm[c][h][w];
            }
        }
    }

    //----------------------------------------------------------------------
    // Layer 2
    //----------------------------------------------------------------------
    std::cout << "Begin processing Layer 2..." << std::endl;
    
    resnet_layer2( resnet_layer2_input_fm,
            resnet_layer2_0_conv1_weights,         resnet_layer2_0_bn1_params,
            resnet_layer2_0_conv2_weights,         resnet_layer2_0_bn2_params,
            resnet_layer2_0_conv3_weights,         resnet_layer2_0_bn3_params,
            resnet_layer2_0_downsample_0_weights,  resnet_layer2_0_downsample_1_params,
            resnet_layer2_1_conv1_weights,         resnet_layer2_1_bn1_params,
            resnet_layer2_1_conv2_weights,         resnet_layer2_1_bn2_params,
            resnet_layer2_1_conv3_weights,         resnet_layer2_1_bn3_params,
            resnet_layer2_2_conv1_weights,         resnet_layer2_2_bn1_params,
            resnet_layer2_2_conv2_weights,         resnet_layer2_2_bn2_params,
            resnet_layer2_2_conv3_weights,         resnet_layer2_2_bn3_params,
            resnet_layer2_3_conv1_weights,         resnet_layer2_3_bn1_params,
            resnet_layer2_3_conv2_weights,         resnet_layer2_3_bn2_params,
            resnet_layer2_3_conv3_weights,         resnet_layer2_3_bn3_params,
            resnet_layer2_output_fm
    );
    
    std::cout << "Layer 2 Processing Complete!" << std::endl << std::endl;
    
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

    //----------------------------------------------------------------------
    // Layer 4
    //----------------------------------------------------------------------
    std::cout << "Begin processing Layer 4..." << std::endl;
    
    resnet_layer4( resnet_layer4_input_fm,
            resnet_layer4_0_conv1_weights,         resnet_layer4_0_bn1_params,
            resnet_layer4_0_conv2_weights,         resnet_layer4_0_bn2_params,
            resnet_layer4_0_conv3_weights,         resnet_layer4_0_bn3_params,
            resnet_layer4_0_downsample_0_weights,  resnet_layer4_0_downsample_1_params,
            resnet_layer4_1_conv1_weights,         resnet_layer4_1_bn1_params,
            resnet_layer4_1_conv2_weights,         resnet_layer4_1_bn2_params,
            resnet_layer4_1_conv3_weights,         resnet_layer4_1_bn3_params,
            resnet_layer4_2_conv1_weights,         resnet_layer4_2_bn1_params,
            resnet_layer4_2_conv2_weights,         resnet_layer4_2_bn2_params,
            resnet_layer4_2_conv3_weights,         resnet_layer4_2_bn3_params,
            resnet_layer4_output_fm
    );
    
    std::cout << "Layer 4 Processing Complete!" << std::endl << std::endl;
 
}
