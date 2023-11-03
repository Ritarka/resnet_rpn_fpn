#include "qdtrack_resnet3.h"
#include "resnet_util3.h"

void resnet_layer3(
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
        fm_t   resnet_layer3_output_fm[RESNET_LAYER3_0_DS_OUT_CH][RESNET_LAYER3_0_FM_HEIGHT][RESNET_LAYER3_0_FM_WIDTH]
)
{
    //TODO: Update with pointer-based loading of input feature map
    for(int c = 0; c < RESNET_LAYER3_0_DS_IN_CH; c++)
    {
        for(int h = 0; h < RESNET_LAYER3_0_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RESNET_LAYER3_0_FM_WIDTH; w++)
            {
                resnet_layer_in_fm[c][h][w] = resnet_layer3_input_fm[c][h][w];
            }
        }
    }
    
    std::cout << "\nStarting resnet_layer 3.0.0" << std::endl;
    //---------------------------------------------------------------
    // Layer 3.0: downsample_0 + downsample_1 TODO
    //---------------------------------------------------------------
    resnet_bottleneck_conv1_bn1<RESNET_LAYER3_0_DS_IN_CH,  RESNET_LAYER3_0_FM_HEIGHT, RESNET_LAYER3_0_FM_WIDTH,
                         RESNET_LAYER3_0_DS_OUT_CH, RESNET_LAYER3_FM_HEIGHT,   RESNET_LAYER3_FM_WIDTH,
                         RESNET_LAYER3_0_DS_STRIDE, RESNET_LAST_LAYER_DISABLE>
                        (resnet_layer3_0_downsample_0_weights, resnet_layer3_0_downsample_1_params, false);
    
    // TODO: Update with pointer-based loading of tiles from DDR
    for(int c = 0; c < RESNET_LAYER3_0_DS_OUT_CH; c++)
    {
        for(int h = 0; h < RESNET_LAYER3_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RESNET_LAYER3_FM_WIDTH; w++)
            {
                ds_fm[c][h][w] = resnet_layer_out_fm[c][h][w];
            }
        }
    }    
    std::cout << "Layer 3.0.0 done" << std::endl;

    std::cout << "\nStarting resnet_layer 3.0.1" << std::endl;
    //---------------------------------------------------------------
    // Layer 3.0.1
    //---------------------------------------------------------------
    resnet_bottleneck_conv1_bn1<RESNET_LAYER3_0_CONV1_IN_CH,  RESNET_LAYER3_0_FM_HEIGHT, RESNET_LAYER3_0_FM_WIDTH,
                         RESNET_LAYER3_0_CONV1_OUT_CH, RESNET_LAYER3_0_FM_HEIGHT, RESNET_LAYER3_0_FM_WIDTH,
                         RESNET_LAYER3_0_CONV1_STRIDE, RESNET_LAST_LAYER_DISABLE>
                        (resnet_layer3_0_conv1_weights, resnet_layer3_0_bn1_params, true);

    // TODO: Update with pointer-based loading of tiles from DDR
    for(int c = 0; c < RESNET_LAYER3_0_CONV1_OUT_CH; c++)
    {
        for(int h = 0; h < RESNET_LAYER3_0_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RESNET_LAYER3_0_FM_WIDTH; w++)
            {
                resnet_layer_in_fm[c][h][w] = resnet_layer_out_fm[c][h][w];
            }
        }
    }
    
    std::cout << "Layer 3.0.1 done" << std::endl;

    std::cout << "\nStarting resnet_layer 3.0.2" << std::endl;
    //---------------------------------------------------------------
    // Layer 3.0.2
    //---------------------------------------------------------------
    resnet_bottleneck_conv2_bn2_relu<RESNET_LAYER3_0_CONV2_IN_CH,  RESNET_LAYER3_0_FM_HEIGHT, RESNET_LAYER3_0_FM_WIDTH,
                              RESNET_LAYER3_0_CONV2_OUT_CH, RESNET_LAYER3_FM_HEIGHT, RESNET_LAYER3_FM_WIDTH,
                              RESNET_LAYER3_0_CONV2_STRIDE, RESNET_LAST_LAYER_DISABLE>
                             (resnet_layer3_0_conv2_weights, resnet_layer3_0_bn2_params);

    // TODO: Update with pointer-based loading of tiles from DDR
    for(int c = 0; c < RESNET_LAYER3_0_CONV2_OUT_CH; c++)
    {
        for(int h = 0; h < RESNET_LAYER3_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RESNET_LAYER3_FM_WIDTH; w++)
            {
                resnet_layer_in_fm[c][h][w] = resnet_layer_out_fm[c][h][w];
            }
        }
    }

    std::cout << "Layer 3.0.2 done" << std::endl;
    
    std::cout << "\nStarting resnet_layer 3.0.3" << std::endl;
    //---------------------------------------------------------------
    // Layer 3.0.3
    //---------------------------------------------------------------
    resnet_bottleneck_conv3_bn3_add_relu<RESNET_LAYER3_0_CONV3_IN_CH, RESNET_LAYER3_FM_HEIGHT, RESNET_LAYER3_FM_WIDTH,
                                  RESNET_LAYER3_0_CONV3_OUT_CH, RESNET_LAYER3_FM_HEIGHT, RESNET_LAYER3_FM_WIDTH,
                                  RESNET_LAYER3_0_CONV3_STRIDE, RESNET_LAST_LAYER_DISABLE>
                                 (resnet_layer3_0_conv3_weights, resnet_layer3_0_bn3_params);
    
    // TODO: Update with pointer-based loading of tiles from DDR
    for(int c = 0; c < RESNET_LAYER3_0_CONV3_OUT_CH; c++)
    {
        for(int h = 0; h < RESNET_LAYER3_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RESNET_LAYER3_FM_WIDTH; w++)
            {
                resnet_layer_in_fm[c][h][w] = resnet_layer_out_fm[c][h][w];
                ds_fm[c][h][w]         = resnet_layer_out_fm[c][h][w];
            }
        }
    }

    std::cout << "Layer 3.0.3 done" << std::endl;
    
    std::cout << "\nStarting resnet_layer 3.1.1" << std::endl;
    //---------------------------------------------------------------
    // Layer 3.1.1
    //---------------------------------------------------------------
    resnet_bottleneck_conv1_bn1<RESNET_LAYER3_CONV1_IN_CH,  RESNET_LAYER3_FM_HEIGHT, RESNET_LAYER3_FM_WIDTH,
                         RESNET_LAYER3_CONV1_OUT_CH, RESNET_LAYER3_FM_HEIGHT, RESNET_LAYER3_FM_WIDTH,
                         RESNET_LAYER3_CONV1_STRIDE, RESNET_LAST_LAYER_DISABLE>
                        (resnet_layer3_1_conv1_weights, resnet_layer3_1_bn1_params, true);

    // TODO: Update with pointer-based loading of tiles from DDR
    for(int c = 0; c < RESNET_LAYER3_CONV1_OUT_CH; c++)
    {
        for(int h = 0; h < RESNET_LAYER3_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RESNET_LAYER3_FM_WIDTH; w++)
            {
                resnet_layer_in_fm[c][h][w] = resnet_layer_out_fm[c][h][w];
            }
        }
    }
    
    std::cout << "Layer 3.1.1 done" << std::endl;

    std::cout << "\nStarting resnet_layer 3.1.2" << std::endl;
    //---------------------------------------------------------------
    // Layer 3.1.2
    //---------------------------------------------------------------
    resnet_bottleneck_conv2_bn2_relu<RESNET_LAYER3_CONV2_IN_CH,  RESNET_LAYER3_FM_HEIGHT, RESNET_LAYER3_FM_WIDTH,
                              RESNET_LAYER3_CONV2_OUT_CH, RESNET_LAYER3_FM_HEIGHT, RESNET_LAYER3_FM_WIDTH,
                              RESNET_LAYER3_CONV2_STRIDE, RESNET_LAST_LAYER_DISABLE>
                             (resnet_layer3_1_conv2_weights, resnet_layer3_1_bn2_params);
    
    for(int c = 0; c < RESNET_LAYER3_CONV2_OUT_CH; c++)
    {
        for(int h = 0; h < RESNET_LAYER3_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RESNET_LAYER3_FM_WIDTH; w++)
            {
                resnet_layer_in_fm[c][h][w] = resnet_layer_out_fm[c][h][w];
            }
        }
    }
    
    std::cout << "Layer 3.1.2 done" << std::endl;
    
    std::cout << "\nStarting resnet_layer 3.1.3" << std::endl;
    //---------------------------------------------------------------
    // Layer 3.1.3
    //---------------------------------------------------------------
    resnet_bottleneck_conv3_bn3_add_relu<RESNET_LAYER3_CONV3_IN_CH, RESNET_LAYER3_FM_HEIGHT, RESNET_LAYER3_FM_WIDTH,
                                  RESNET_LAYER3_CONV3_OUT_CH, RESNET_LAYER3_FM_HEIGHT, RESNET_LAYER3_FM_WIDTH,
                                  RESNET_LAYER3_CONV3_STRIDE, RESNET_LAST_LAYER_DISABLE>
                                 (resnet_layer3_1_conv3_weights, resnet_layer3_1_bn3_params);
    
    for(int c = 0; c < RESNET_LAYER3_CONV3_OUT_CH; c++)
    {
        for(int h = 0; h < RESNET_LAYER3_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RESNET_LAYER3_FM_WIDTH; w++)
            {
                resnet_layer_in_fm[c][h][w] = resnet_layer_out_fm[c][h][w];
                ds_fm[c][h][w]         = resnet_layer_out_fm[c][h][w];
            }
        }
    }   
    std::cout << "Layer 3.1.3 done" << std::endl;
 
    std::cout << "\nStarting resnet_layer 3.2.1" << std::endl;
    //---------------------------------------------------------------
    // Layer 3.2.1
    //---------------------------------------------------------------
    resnet_bottleneck_conv1_bn1<RESNET_LAYER3_CONV1_IN_CH,  RESNET_LAYER3_FM_HEIGHT, RESNET_LAYER3_FM_WIDTH,
                         RESNET_LAYER3_CONV1_OUT_CH, RESNET_LAYER3_FM_HEIGHT, RESNET_LAYER3_FM_WIDTH,
                         RESNET_LAYER3_CONV1_STRIDE, RESNET_LAST_LAYER_DISABLE>
                        (resnet_layer3_2_conv1_weights, resnet_layer3_2_bn1_params, true);

    // TODO: Update with pointer-based loading of tiles from DDR
    for(int c = 0; c < RESNET_LAYER3_CONV1_OUT_CH; c++)
    {
        for(int h = 0; h < RESNET_LAYER3_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RESNET_LAYER3_FM_WIDTH; w++)
            {
                resnet_layer_in_fm[c][h][w] = resnet_layer_out_fm[c][h][w];
            }
        }
    }
    
    std::cout << "Layer 3.2.1 done" << std::endl;

    std::cout << "\nStarting resnet_layer 3.2.2" << std::endl;
    //---------------------------------------------------------------
    // Layer 3.2.2
    //---------------------------------------------------------------
    resnet_bottleneck_conv2_bn2_relu<RESNET_LAYER3_CONV2_IN_CH,  RESNET_LAYER3_FM_HEIGHT, RESNET_LAYER3_FM_WIDTH,
                              RESNET_LAYER3_CONV2_OUT_CH, RESNET_LAYER3_FM_HEIGHT, RESNET_LAYER3_FM_WIDTH,
                              RESNET_LAYER3_CONV2_STRIDE, RESNET_LAST_LAYER_DISABLE>
                             (resnet_layer3_2_conv2_weights, resnet_layer3_2_bn2_params);
    
    for(int c = 0; c < RESNET_LAYER3_CONV2_OUT_CH; c++)
    {
        for(int h = 0; h < RESNET_LAYER3_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RESNET_LAYER3_FM_WIDTH; w++)
            {
                resnet_layer_in_fm[c][h][w] = resnet_layer_out_fm[c][h][w];
            }
        }
    }
    
    std::cout << "Layer 3.2.2 done" << std::endl;
    
    std::cout << "\nStarting resnet_layer 3.2.3" << std::endl;
    //---------------------------------------------------------------
    // Layer 3.2.3
    //---------------------------------------------------------------
    resnet_bottleneck_conv3_bn3_add_relu<RESNET_LAYER3_CONV3_IN_CH, RESNET_LAYER3_FM_HEIGHT, RESNET_LAYER3_FM_WIDTH,
                                  RESNET_LAYER3_CONV3_OUT_CH, RESNET_LAYER3_FM_HEIGHT, RESNET_LAYER3_FM_WIDTH,
                                  RESNET_LAYER3_CONV3_STRIDE, RESNET_LAST_LAYER_DISABLE>
                                 (resnet_layer3_2_conv3_weights, resnet_layer3_2_bn3_params);
    
    for(int c = 0; c < RESNET_LAYER3_CONV3_OUT_CH; c++)
    {
        for(int h = 0; h < RESNET_LAYER3_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RESNET_LAYER3_FM_WIDTH; w++)
            {
                resnet_layer_in_fm[c][h][w] = resnet_layer_out_fm[c][h][w];
                ds_fm[c][h][w]         = resnet_layer_out_fm[c][h][w];
            }
        }
    }   
    std::cout << "Layer 3.2.3 done" << std::endl;    
    
    std::cout << "\nStarting resnet_layer 3.3.1" << std::endl;
    //---------------------------------------------------------------
    // Layer 3.3.1
    //---------------------------------------------------------------
    resnet_bottleneck_conv1_bn1<RESNET_LAYER3_CONV1_IN_CH,  RESNET_LAYER3_FM_HEIGHT, RESNET_LAYER3_FM_WIDTH,
                         RESNET_LAYER3_CONV1_OUT_CH, RESNET_LAYER3_FM_HEIGHT, RESNET_LAYER3_FM_WIDTH,
                         RESNET_LAYER3_CONV1_STRIDE, RESNET_LAST_LAYER_DISABLE>
                        (resnet_layer3_3_conv1_weights, resnet_layer3_3_bn1_params, true);

    // TODO: Update with pointer-based loading of tiles from DDR
    for(int c = 0; c < RESNET_LAYER3_CONV1_OUT_CH; c++)
    {
        for(int h = 0; h < RESNET_LAYER3_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RESNET_LAYER3_FM_WIDTH; w++)
            {
                resnet_layer_in_fm[c][h][w] = resnet_layer_out_fm[c][h][w];
            }
        }
    }
    
    std::cout << "Layer 3.3.1 done" << std::endl;

    std::cout << "\nStarting resnet_layer 3.3.2" << std::endl;
    //---------------------------------------------------------------
    // Layer 3.3.2
    //---------------------------------------------------------------
    resnet_bottleneck_conv2_bn2_relu<RESNET_LAYER3_CONV2_IN_CH,  RESNET_LAYER3_FM_HEIGHT, RESNET_LAYER3_FM_WIDTH,
                              RESNET_LAYER3_CONV2_OUT_CH, RESNET_LAYER3_FM_HEIGHT, RESNET_LAYER3_FM_WIDTH,
                              RESNET_LAYER3_CONV2_STRIDE, RESNET_LAST_LAYER_DISABLE>
                             (resnet_layer3_3_conv2_weights, resnet_layer3_3_bn2_params);
    
    for(int c = 0; c < RESNET_LAYER3_CONV2_OUT_CH; c++)
    {
        for(int h = 0; h < RESNET_LAYER3_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RESNET_LAYER3_FM_WIDTH; w++)
            {
                resnet_layer_in_fm[c][h][w] = resnet_layer_out_fm[c][h][w];
            }
        }
    }
    
    std::cout << "Layer 3.3.2 done" << std::endl;
    
    std::cout << "\nStarting resnet_layer 3.3.3" << std::endl;
    //---------------------------------------------------------------
    // Layer 3.3.3
    //---------------------------------------------------------------
    resnet_bottleneck_conv3_bn3_add_relu<RESNET_LAYER3_CONV3_IN_CH, RESNET_LAYER3_FM_HEIGHT, RESNET_LAYER3_FM_WIDTH,
                                  RESNET_LAYER3_CONV3_OUT_CH, RESNET_LAYER3_FM_HEIGHT, RESNET_LAYER3_FM_WIDTH,
                                  RESNET_LAYER3_CONV3_STRIDE, RESNET_LAST_LAYER_DISABLE>
                                 (resnet_layer3_3_conv3_weights, resnet_layer3_3_bn3_params);
    
    for(int c = 0; c < RESNET_LAYER3_CONV3_OUT_CH; c++)
    {
        for(int h = 0; h < RESNET_LAYER3_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RESNET_LAYER3_FM_WIDTH; w++)
            {
                resnet_layer_in_fm[c][h][w] = resnet_layer_out_fm[c][h][w];
                ds_fm[c][h][w]         = resnet_layer_out_fm[c][h][w];
            }
        }
    }   
    std::cout << "Layer 3.3.3 done" << std::endl;    

    std::cout << "\nStarting resnet_layer 3.4.1" << std::endl;
    //---------------------------------------------------------------
    // Layer 3.4.1
    //---------------------------------------------------------------
    resnet_bottleneck_conv1_bn1<RESNET_LAYER3_CONV1_IN_CH,  RESNET_LAYER3_FM_HEIGHT, RESNET_LAYER3_FM_WIDTH,
                         RESNET_LAYER3_CONV1_OUT_CH, RESNET_LAYER3_FM_HEIGHT, RESNET_LAYER3_FM_WIDTH,
                         RESNET_LAYER3_CONV1_STRIDE, RESNET_LAST_LAYER_DISABLE>
                        (resnet_layer3_4_conv1_weights, resnet_layer3_4_bn1_params, true);

    // TODO: Update with pointer-based loading of tiles from DDR
    for(int c = 0; c < RESNET_LAYER3_CONV1_OUT_CH; c++)
    {
        for(int h = 0; h < RESNET_LAYER3_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RESNET_LAYER3_FM_WIDTH; w++)
            {
                resnet_layer_in_fm[c][h][w] = resnet_layer_out_fm[c][h][w];
            }
        }
    }
    
    std::cout << "Layer 3.4.1 done" << std::endl;

    std::cout << "\nStarting resnet_layer 3.4.2" << std::endl;
    //---------------------------------------------------------------
    // Layer 3.4.2
    //---------------------------------------------------------------
    resnet_bottleneck_conv2_bn2_relu<RESNET_LAYER3_CONV2_IN_CH,  RESNET_LAYER3_FM_HEIGHT, RESNET_LAYER3_FM_WIDTH,
                              RESNET_LAYER3_CONV2_OUT_CH, RESNET_LAYER3_FM_HEIGHT, RESNET_LAYER3_FM_WIDTH,
                              RESNET_LAYER3_CONV2_STRIDE, RESNET_LAST_LAYER_DISABLE>
                             (resnet_layer3_4_conv2_weights, resnet_layer3_4_bn2_params);
    
    for(int c = 0; c < RESNET_LAYER3_CONV2_OUT_CH; c++)
    {
        for(int h = 0; h < RESNET_LAYER3_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RESNET_LAYER3_FM_WIDTH; w++)
            {
                resnet_layer_in_fm[c][h][w] = resnet_layer_out_fm[c][h][w];
            }
        }
    }
    
    std::cout << "Layer 3.4.2 done" << std::endl;
    
    std::cout << "\nStarting resnet_layer 3.4.3" << std::endl;
    //---------------------------------------------------------------
    // Layer 3.4.3
    //---------------------------------------------------------------
    resnet_bottleneck_conv3_bn3_add_relu<RESNET_LAYER3_CONV3_IN_CH, RESNET_LAYER3_FM_HEIGHT, RESNET_LAYER3_FM_WIDTH,
                                  RESNET_LAYER3_CONV3_OUT_CH, RESNET_LAYER3_FM_HEIGHT, RESNET_LAYER3_FM_WIDTH,
                                  RESNET_LAYER3_CONV3_STRIDE, RESNET_LAST_LAYER_DISABLE>
                                 (resnet_layer3_4_conv3_weights, resnet_layer3_4_bn3_params);
    
    for(int c = 0; c < RESNET_LAYER3_CONV3_OUT_CH; c++)
    {
        for(int h = 0; h < RESNET_LAYER3_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RESNET_LAYER3_FM_WIDTH; w++)
            {
                resnet_layer_in_fm[c][h][w] = resnet_layer_out_fm[c][h][w];
                ds_fm[c][h][w]         = resnet_layer_out_fm[c][h][w];
            }
        }
    }   
    std::cout << "Layer 3.4.3 done" << std::endl;
    
    std::cout << "\nStarting resnet_layer 3.5.1" << std::endl;
    //---------------------------------------------------------------
    // Layer 3.5.1
    //---------------------------------------------------------------
    resnet_bottleneck_conv1_bn1<RESNET_LAYER3_CONV1_IN_CH,  RESNET_LAYER3_FM_HEIGHT, RESNET_LAYER3_FM_WIDTH,
                         RESNET_LAYER3_CONV1_OUT_CH, RESNET_LAYER3_FM_HEIGHT, RESNET_LAYER3_FM_WIDTH,
                         RESNET_LAYER3_CONV1_STRIDE, RESNET_LAST_LAYER_DISABLE>
                        (resnet_layer3_5_conv1_weights, resnet_layer3_5_bn1_params, true);

    // TODO: Update with pointer-based loading of tiles from DDR
    for(int c = 0; c < RESNET_LAYER3_CONV1_OUT_CH; c++)
    {
        for(int h = 0; h < RESNET_LAYER3_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RESNET_LAYER3_FM_WIDTH; w++)
            {
                resnet_layer_in_fm[c][h][w] = resnet_layer_out_fm[c][h][w];
            }
        }
    }
    
    std::cout << "Layer 3.5.1 done" << std::endl;

    std::cout << "\nStarting resnet_layer 3.5.2" << std::endl;
    //---------------------------------------------------------------
    // Layer 3.5.2
    //---------------------------------------------------------------
    resnet_bottleneck_conv2_bn2_relu<RESNET_LAYER3_CONV2_IN_CH,  RESNET_LAYER3_FM_HEIGHT, RESNET_LAYER3_FM_WIDTH,
                              RESNET_LAYER3_CONV2_OUT_CH, RESNET_LAYER3_FM_HEIGHT, RESNET_LAYER3_FM_WIDTH,
                              RESNET_LAYER3_CONV2_STRIDE, RESNET_LAST_LAYER_DISABLE>
                             (resnet_layer3_5_conv2_weights, resnet_layer3_5_bn2_params);
    
    for(int c = 0; c < RESNET_LAYER3_CONV2_OUT_CH; c++)
    {
        for(int h = 0; h < RESNET_LAYER3_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RESNET_LAYER3_FM_WIDTH; w++)
            {
                resnet_layer_in_fm[c][h][w] = resnet_layer_out_fm[c][h][w];
            }
        }
    }
    
    std::cout << "Layer 3.5.2 done" << std::endl;
    
    std::cout << "\nStarting resnet_layer 3.5.3" << std::endl;
    //---------------------------------------------------------------
    // Layer 3.5.3
    //---------------------------------------------------------------
    resnet_bottleneck_conv3_bn3_add_relu<RESNET_LAYER3_CONV3_IN_CH, RESNET_LAYER3_FM_HEIGHT, RESNET_LAYER3_FM_WIDTH,
                                  RESNET_LAYER3_CONV3_OUT_CH, RESNET_LAYER3_FM_HEIGHT, RESNET_LAYER3_FM_WIDTH,
                                  RESNET_LAYER3_CONV3_STRIDE, RESNET_LAST_LAYER_DISABLE>
                                 (resnet_layer3_5_conv3_weights, resnet_layer3_5_bn3_params);
    

    for(int c = 0; c < RESNET_LAYER3_CONV3_OUT_CH; c++)
    {
        for(int h = 0; h < RESNET_LAYER3_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RESNET_LAYER3_FM_WIDTH; w++)
            {
                resnet_layer3_output_fm[c][h][w] = resnet_layer_out_fm[c][h][w];
            }
        }
    }
    
    std::cout << "Layer 3.5.3 done" << std::endl;
}
