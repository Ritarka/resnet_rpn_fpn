#include "qdtrack_resnet1_1.h"
#include "resnet_util1.h"

void resnet_layer1_1(
        wt_t   resnet_layer1_1_conv1_weights[RESNET_LAYER1_CONV1_OUT_CH][RESNET_LAYER1_CONV1_IN_CH],
	    wt_t   resnet_layer1_1_bn1_params[4][RESNET_LAYER1_CONV1_OUT_CH],
        wt_t   resnet_layer1_1_conv2_weights[RESNET_LAYER1_CONV2_OUT_CH][RESNET_LAYER1_CONV2_IN_CH][3][3],
	    wt_t   resnet_layer1_1_bn2_params[4][RESNET_LAYER1_CONV2_OUT_CH],
        wt_t   resnet_layer1_1_conv3_weights[RESNET_LAYER1_CONV3_OUT_CH][RESNET_LAYER1_CONV3_IN_CH],
	    wt_t   resnet_layer1_1_bn3_params[4][RESNET_LAYER1_CONV3_OUT_CH]
)
{


    #pragma HLS INTERFACE m_axi depth=1 prot=resnet_layer1_1_conv1_weights bundle=wt
    #pragma HLS INTERFACE m_axi depth=1 port=resnet_layer1_1_conv2_weights bundle=wt
    #pragma HLS INTERFACE m_axi depth=1 port=resnet_layer1_1_bn2_params bundle=wt
    #pragma HLS INTERFACE m_axi depth=1 port=resnet_layer1_1_conv3_weights bundle=wt
    #pragma HLS INTERFACE m_axi depth=1 prot=resnet_layer1_1_bn3_params bundle=wt


    
    // Use HLS pragma to control memory access
    #pragma HLS array_partition variable=resnet_layer1_1_conv1_weights complete dim=1
    #pragma HLS array_partition variable=resnet_layer1_1_conv2_weights complete dim=1
    #pragma HLS array_partition variable=resnet_layer1_1_bn2_params complete
    #pragma HLS array_partition variable=resnet_layer1_1_conv3_weights complete dim=1
    #pragma HLS array_partition variable=resnet_layer1_1_bn3_params complete
    
    std::cout << "\nStarting resnet_layer 1.1.1" << std::endl;
    //---------------------------------------------------------------
    // Layer 1.1.1
    //---------------------------------------------------------------
    resnet_bottleneck_conv1_bn1<RESNET_LAYER1_CONV1_IN_CH,  RESNET_LAYER1_FM_HEIGHT, RESNET_LAYER1_FM_WIDTH,
                         RESNET_LAYER1_CONV1_OUT_CH, RESNET_LAYER1_FM_HEIGHT, RESNET_LAYER1_FM_WIDTH,
                         RESNET_LAYER1_CONV1_STRIDE, RESNET_LAST_LAYER_DISABLE>
                        (resnet_layer1_1_conv1_weights, resnet_layer1_1_bn1_params, true);

    // TODO: Update with pointer-based loading of tiles from DDR
    for(int c = 0; c < RESNET_LAYER1_CONV1_OUT_CH; c++)
    {
        for(int h = 0; h < RESNET_LAYER1_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RESNET_LAYER1_FM_WIDTH; w++)
            {
                resnet_layer_in_fm[c][h][w] = resnet_layer_out_fm[c][h][w];
            }
        }
    }
    
    std::cout << "Layer 1.1.1 done" << std::endl;

    std::cout << "\nStarting resnet_layer 1.1.2" << std::endl;
    //---------------------------------------------------------------
    // Layer 1.1.2
    //---------------------------------------------------------------
    resnet_bottleneck_conv2_bn2_relu<RESNET_LAYER1_CONV2_IN_CH,  RESNET_LAYER1_FM_HEIGHT, RESNET_LAYER1_FM_WIDTH,
                              RESNET_LAYER1_CONV2_OUT_CH, RESNET_LAYER1_FM_HEIGHT, RESNET_LAYER1_FM_WIDTH,
                              RESNET_LAYER1_CONV2_STRIDE, RESNET_LAST_LAYER_DISABLE>
                             (resnet_layer1_1_conv2_weights, resnet_layer1_1_bn2_params);
    
    for(int c = 0; c < RESNET_LAYER1_CONV2_OUT_CH; c++)
    {
        for(int h = 0; h < RESNET_LAYER1_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RESNET_LAYER1_FM_WIDTH; w++)
            {
                resnet_layer_in_fm[c][h][w] = resnet_layer_out_fm[c][h][w];
            }
        }
    }
    
    std::cout << "Layer 1.1.2 done" << std::endl;
    
    std::cout << "\nStarting resnet_layer 1.1.3" << std::endl;
    //---------------------------------------------------------------
    // Layer 1.1.3
    //---------------------------------------------------------------
    resnet_bottleneck_conv3_bn3_add_relu<RESNET_LAYER1_CONV3_IN_CH, RESNET_LAYER1_FM_HEIGHT, RESNET_LAYER1_FM_WIDTH,
                                  RESNET_LAYER1_CONV3_OUT_CH, RESNET_LAYER1_FM_HEIGHT, RESNET_LAYER1_FM_WIDTH,
                                  RESNET_LAYER1_CONV3_STRIDE, RESNET_LAST_LAYER_DISABLE>
                                 (resnet_layer1_1_conv3_weights, resnet_layer1_1_bn3_params);
    
    for(int c = 0; c < RESNET_LAYER1_CONV3_OUT_CH; c++)
    {
        for(int h = 0; h < RESNET_LAYER1_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RESNET_LAYER1_FM_WIDTH; w++)
            {
                resnet_layer_in_fm[c][h][w] = resnet_layer_out_fm[c][h][w];
                ds_fm[c][h][w]         = resnet_layer_out_fm[c][h][w];
            }
        }
    }   
    std::cout << "Layer 1.1.3 done" << std::endl;
}
