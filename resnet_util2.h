#include "hls_stream.h"
#include "qdtrack_resnet2_0.h"

#pragma once 

// Feature Map Buffers 
extern fm_t in_fm_buf[RESNET_IN_BUF_CH][RESNET_IN_BUF_ROWS + 5][RESNET_IN_BUF_COLS + 5];
extern fm_t out_fm_buf[RESNET_OUT_BUF_CH][RESNET_OUT_BUF_ROWS][RESNET_OUT_BUF_COLS];
extern fm_t partial_out_fm_buf[RESNET_OUT_BUF_CH][RESNET_OUT_BUF_ROWS][RESNET_OUT_BUF_COLS];
extern fm_t ds_fm_buf[RESNET_OUT_BUF_CH][RESNET_OUT_BUF_ROWS][RESNET_OUT_BUF_COLS];

// TODO: Replace these by pointer-based access to DRAM
extern fm_t resnet_layer_out_fm[2048][184][320];
extern fm_t resnet_layer_in_fm[2048][184][320];
extern fm_t ds_fm[2048][184][320];

// Convolution Weight Buffers
extern wt_t weight_buf_1x1[RESNET_IN_BUF_CH];
extern wt_t weight_buf_3x3[RESNET_IN_BUF_CH][3][3];
extern wt_t weight_buf_7x7[RESNET_IN_BUF_CH][7][7];

// BatchNorm Parameter Buffer
extern wt_t param_buf[3][RESNET_OUT_BUF_CH];

// Load input feature map tile
// TODO: Replace with pointer-based DRAM access
template<const int  RESNET_IN_FM_DEPTH, const int  RESNET_IN_FM_HEIGHT, const int  RESNET_IN_FM_WIDTH,
         const int  RESNET_LAST_LAYER_EN>
void resnet_load_input_fm_tile (
        fm_t in_fm_buf[RESNET_IN_BUF_CH][RESNET_IN_BUF_ROWS + 5][RESNET_IN_BUF_COLS + 5], 
        fm_t in_fm[2048][184][320], 
        int   ti, 
        int   tj, 
        int   P,
        int   d
)
{
    for(int c = 0; c < RESNET_IN_BUF_CH; c++)
    {
        for(int i = 0; i < RESNET_IN_BUF_ROWS + 5; i++)
        {
            for(int j = 0; j < RESNET_IN_BUF_COLS + 5; j++)
            {
                // For the first tile row, pad zero to the left and/or at the top
                if((ti == 0 && i < P) || (tj == 0 && j < P))
		        {
		            in_fm_buf[c][i][j] = 0.0;
		        }
                // For the last tile row, pad zero to the right and/or at the bottom
                else if(((ti == (RESNET_IN_FM_HEIGHT/RESNET_IN_BUF_ROWS) - 1) && (i > RESNET_IN_BUF_ROWS + P - 1)) 
                     || ((tj == (RESNET_IN_FM_WIDTH/RESNET_IN_BUF_COLS) - 1) && (j > RESNET_IN_BUF_COLS + P - 1)))
		        {
		            in_fm_buf[c][i][j] = 0.0;
		        }
                // For an input feature map smaller than buffer size, pad zero
                // at at the bottom of the feature map
                else if(RESNET_LAST_LAYER_EN && (i > (RESNET_IN_BUF_ROWS/2) + P - 1)) // Assumes buffer height is even
		        {
		            in_fm_buf[c][i][j] = 0.0;
		        }
                // Copy the input feature map elements otherwise
		        else
		        {
		            in_fm_buf[c][i][j] = in_fm[d*RESNET_IN_BUF_CH + c][ti*RESNET_IN_BUF_ROWS + i - P][tj*RESNET_IN_BUF_COLS + j - P];
		        }
            }
        }
    }
}

// Load previous resnet_layer's output (shortcut connection)
static void resnet_load_residual_fm_tile (
        fm_t in_fm_buf[RESNET_IN_BUF_CH][RESNET_IN_BUF_ROWS][RESNET_IN_BUF_COLS], 
        fm_t in_fm[2048][184][320], 
        int   ti, 
        int   tj,
        int   b 
)
{
    for(int c = 0; c < RESNET_IN_BUF_CH; c++)
    {
        for(int i = 0; i < RESNET_IN_BUF_ROWS; i++)
        {
            for(int j = 0; j < RESNET_IN_BUF_COLS; j++)
            {
		        in_fm_buf[c][i][j] = in_fm[b*RESNET_IN_BUF_CH + c][ti*RESNET_IN_BUF_ROWS + i][tj*RESNET_IN_BUF_COLS + j];
            }
        }
    }
}

// Load weights for 1x1 convolution
template<const int RESNET_OUT_CH, const int RESNET_IN_CH>
void resnet_load_weights_1x1 (
        wt_t weight_buf_1x1[RESNET_IN_BUF_CH],
        wt_t weights[RESNET_OUT_CH][RESNET_IN_CH],
        int f,
        int b,
        int d
)
{
    for(int i = 0; i < RESNET_IN_BUF_CH; i++)
    {
	    weight_buf_1x1[i] = weights[b * RESNET_OUT_BUF_CH + f][d * RESNET_IN_BUF_CH + i];
    }
}

// Load weights for 3x3 convolution
template<const int RESNET_OUT_CH, const int RESNET_IN_CH>
void resnet_load_weights_3x3 (
        wt_t weight_buf_3x3[RESNET_IN_BUF_CH][3][3],
        wt_t weights[RESNET_OUT_CH][RESNET_IN_CH][3][3],
        int f,
        int b,
        int d
)
{
    for(int c = 0; c < RESNET_IN_BUF_CH; c++)
    {
        for(int kh = 0; kh < 3; kh++)
	    {
	        for(int kw = 0; kw < 3; kw++)
	        {
	            weight_buf_3x3[c][kh][kw] = weights[b * RESNET_OUT_BUF_CH + f][d * RESNET_IN_BUF_CH + c][kh][kw];
            }
        }
    }
}

// Load BatchNorm parameters
// [0] -> Weight
// [1] -> Bias
// [2] -> Running Mean
// [3] -> Running Variance
template<const int RESNET_OUT_CH>
void resnet_load_batchnorm_params (
        fm_t param_buf[3][RESNET_OUT_BUF_CH], 
        wt_t params[3][RESNET_OUT_CH], 
        int b
)
{
    for(int i = 0; i < 4; i++)
    {
        for(int j = 0; j < RESNET_OUT_BUF_CH; j++)
        {
            param_buf[i][j] = params[i][b * RESNET_OUT_BUF_CH + j];
        }
    }
}

// Save partial outputs when depth of output feature map is more than
// the output buffer depth
static void resnet_save_partial_out_buf (
        fm_t partial_out_fm_buf[RESNET_OUT_BUF_CH][RESNET_OUT_BUF_ROWS][RESNET_OUT_BUF_COLS], 
        fm_t out_fm_buf[RESNET_OUT_BUF_CH][RESNET_OUT_BUF_ROWS][RESNET_OUT_BUF_COLS],
        int d
)
{
    for(int f = 0; f < RESNET_OUT_BUF_CH; f++)
    {
        for(int i = 0; i < RESNET_OUT_BUF_ROWS; i++)
        {
            for(int j = 0; j < RESNET_OUT_BUF_COLS; j++)
            {
                if(d==0)
                    partial_out_fm_buf[f][i][j]  = out_fm_buf[f][i][j];
                else
                    partial_out_fm_buf[f][i][j] += out_fm_buf[f][i][j];
            }
        }
    }
}

// Store output buffer elements to DDR
// TODO: Replace buffer sizes with pointers
template<const int S>
void resnet_store_out_buf_to_DDR(
        fm_t out_fm[2048][184][320], 
        fm_t fm_buf[RESNET_OUT_BUF_CH][RESNET_OUT_BUF_ROWS][RESNET_OUT_BUF_COLS], 
        int   ti, 
        int   tj,
        int   b
)
{
    for(int f = 0; f < RESNET_OUT_BUF_CH; f++)
    {
        for(int i = 0; i < RESNET_OUT_BUF_ROWS/S; i++)
        {
            for(int j = 0; j < RESNET_OUT_BUF_COLS/S; j++)
            {
                 out_fm[b*RESNET_OUT_BUF_CH + f][ti*(RESNET_OUT_BUF_ROWS/S) + i][tj*(RESNET_OUT_BUF_COLS/S) + j] = fm_buf[f][i][j];
            }
        }
    }
}

// Bottleneck resnet_layer's first block
// Performs 1x1 convolution with batchnorm and optionally ReLU
template<const int  RESNET_IN_FM_DEPTH, const int  RESNET_IN_FM_HEIGHT, const int  RESNET_IN_FM_WIDTH,
         const int RESNET_OUT_FM_DEPTH, const int RESNET_OUT_FM_HEIGHT, const int RESNET_OUT_FM_WIDTH,
         const int STRIDE, const int RESNET_LAST_LAYER_EN>
void resnet_bottleneck_conv1_bn1(
        wt_t conv_weights[RESNET_OUT_FM_DEPTH][RESNET_IN_FM_DEPTH],
        wt_t bn_params[3][RESNET_OUT_FM_DEPTH],
        bool  enable_relu
)
{
    const int num_of_tiles = ((RESNET_IN_FM_HEIGHT / RESNET_IN_BUF_ROWS) + RESNET_LAST_LAYER_EN) * (RESNET_IN_FM_WIDTH / RESNET_IN_BUF_COLS);
    std::cout << "\nNo. of tiles in input feature map = " << num_of_tiles << std::endl;

    for(int ti = 0; ti < (RESNET_IN_FM_HEIGHT / RESNET_IN_BUF_ROWS) + RESNET_LAST_LAYER_EN; ti++)
    {
        for(int tj = 0; tj < (RESNET_IN_FM_WIDTH / RESNET_IN_BUF_COLS); tj++)
        {
            std::cout << "Processing Tile " << ti*(RESNET_IN_FM_WIDTH / RESNET_IN_BUF_COLS) + tj + 1;
            std::cout << "/" << num_of_tiles << std::endl;

            for(int b = 0; b < (RESNET_OUT_FM_DEPTH / RESNET_OUT_BUF_CH); b++) 
	        {
                for(int d = 0; d < (RESNET_IN_FM_DEPTH / RESNET_IN_BUF_CH); d++) 
                {
                    //cout << ti << " " << tj << " " << b << " " << d << " " << endl;

                    // Load input feature map from AXI input into input buffer
                    // Assumption: Zero padding
                    resnet_load_input_fm_tile<RESNET_IN_FM_DEPTH, RESNET_IN_FM_HEIGHT, RESNET_IN_FM_WIDTH, RESNET_LAST_LAYER_EN>
                                      (in_fm_buf, resnet_layer_in_fm, ti, tj, 0, d);
                                      //(in_fm_buf, in_fm, ti, tj, 0, d);

	                for(int f = 0; f < RESNET_IN_BUF_CH; f++)
	                {
	                    // Load weights from on-chip memory
                        resnet_load_weights_1x1<RESNET_OUT_FM_DEPTH, RESNET_IN_FM_DEPTH>
                                        (weight_buf_1x1, conv_weights, f, b, d);
	                
                        // Perform convolution
                        resnet_conv_1x1(out_fm_buf, in_fm_buf, weight_buf_1x1, f, STRIDE);
	                }
	   
	                // Save partial output feature map in a buffer
                    resnet_save_partial_out_buf(partial_out_fm_buf, out_fm_buf, d);
                }
                
                // Load BatchNorm params 
                resnet_load_batchnorm_params<RESNET_OUT_FM_DEPTH>(param_buf, bn_params, b);

	            // BatchNorm
	            //batchnorm(partial_out_fm_buf, resnet_layer1_1_bn1_params, true);
	            resnet_batchnorm(partial_out_fm_buf, param_buf, enable_relu);
          
	            // Store output feature map to DDR (off-chip)
                resnet_store_out_buf_to_DDR<STRIDE>(resnet_layer_out_fm, partial_out_fm_buf, ti, tj, b);
            }
        }
    }
}

// Bottleneck resnet_layer's second block
// Performs 3x3 convolution with batchnorm and optionally ReLU
template<const int RESNET_IN_FM_DEPTH,  const int RESNET_IN_FM_HEIGHT,  const int RESNET_IN_FM_WIDTH,
         const int RESNET_OUT_FM_DEPTH, const int RESNET_OUT_FM_HEIGHT, const int RESNET_OUT_FM_WIDTH, 
         const int STRIDE, const int RESNET_LAST_LAYER_EN>
void resnet_bottleneck_conv2_bn2_relu(
        wt_t conv_weights[RESNET_OUT_FM_DEPTH][RESNET_IN_FM_DEPTH][3][3],
        wt_t bn_params[3][RESNET_OUT_FM_DEPTH]
)
{
    const int num_of_tiles = ((RESNET_IN_FM_HEIGHT / RESNET_IN_BUF_ROWS) + RESNET_LAST_LAYER_EN) * (RESNET_IN_FM_WIDTH / RESNET_IN_BUF_COLS);
    std::cout << "\nNo. of tiles in input feature map = " << num_of_tiles << std::endl;

    for(int ti = 0; ti < (RESNET_IN_FM_HEIGHT / RESNET_IN_BUF_ROWS) + RESNET_LAST_LAYER_EN; ti++)
    {
        for(int tj = 0; tj < (RESNET_IN_FM_WIDTH / RESNET_IN_BUF_COLS); tj++)
        {
            std::cout << "Processing Tile " << ti*(RESNET_IN_FM_WIDTH / RESNET_IN_BUF_COLS) + tj + 1;
            std::cout << "/" << num_of_tiles << std::endl;

            for(int b = 0; b < (RESNET_OUT_FM_DEPTH / RESNET_OUT_BUF_CH); b++) 
	        {
                for(int d = 0; d < (RESNET_IN_FM_DEPTH / RESNET_IN_BUF_CH); d++) 
                {
                    //cout << ti << " " << tj << " " << b << " " << d << " " << endl;
                    
                    // Load input feature map from AXI input into input buffer
                    // Assumption: Padding = (1,1)
                    resnet_load_input_fm_tile<RESNET_IN_FM_DEPTH, RESNET_IN_FM_HEIGHT, RESNET_IN_FM_WIDTH, RESNET_LAST_LAYER_EN>
                                      (in_fm_buf, resnet_layer_in_fm, ti, tj, 1, d);
                                      //(in_fm_buf, in_fm, ti, tj, 1, d);
                    
	                for(int f = 0; f < RESNET_IN_BUF_CH; f++)
	                {
	                    // Load weights from on-chip memory
                        resnet_load_weights_3x3<RESNET_OUT_FM_DEPTH, RESNET_IN_FM_DEPTH>
                                        (weight_buf_3x3, conv_weights, f, b, d);
	                
                        // Perform convolution TODO
                        //conv_3x3<2>(out_fm_buf, in_fm_buf, weight_buf_3x3, f); // Linking error
                        resnet_conv_3x3(out_fm_buf, in_fm_buf, weight_buf_3x3, f, STRIDE);
	                }
	   
	                // Save partial output feature map in a buffer
                    resnet_save_partial_out_buf(partial_out_fm_buf, out_fm_buf, d);
                }
                
                // Load BatchNorm params
                resnet_load_batchnorm_params<RESNET_OUT_FM_DEPTH>(param_buf, bn_params, b);

	            // BatchNorm
	            resnet_batchnorm(partial_out_fm_buf, param_buf, true);
          
	            // Store output feature map to DDR (off-chip)
                resnet_store_out_buf_to_DDR<STRIDE>(resnet_layer_out_fm, partial_out_fm_buf, ti, tj, b);
            }
        }
    }
}

// Bottleneck resnet_layer's third and final block
// Performs 1x1 convolution with batchnorm, adds the shortcut connection and runs ReLU
template<const int  RESNET_IN_FM_DEPTH, const int  RESNET_IN_FM_HEIGHT, const int  RESNET_IN_FM_WIDTH,
         const int RESNET_OUT_FM_DEPTH, const int RESNET_OUT_FM_HEIGHT, const int RESNET_OUT_FM_WIDTH,
         const int STRIDE, const int RESNET_LAST_LAYER_EN>
void resnet_bottleneck_conv3_bn3_add_relu(
        wt_t conv_weights[RESNET_OUT_FM_DEPTH][RESNET_IN_FM_DEPTH],
        wt_t bn_params[3][RESNET_OUT_FM_DEPTH]
)
{
    const int num_of_tiles = ((RESNET_IN_FM_HEIGHT / RESNET_IN_BUF_ROWS) + RESNET_LAST_LAYER_EN) * (RESNET_IN_FM_WIDTH / RESNET_IN_BUF_COLS);
    std::cout << "\nNo. of tiles in input feature map = " << num_of_tiles << std::endl;

    for(int ti = 0; ti < (RESNET_IN_FM_HEIGHT / RESNET_IN_BUF_ROWS) + RESNET_LAST_LAYER_EN; ti++)
    {
        for(int tj = 0; tj < (RESNET_IN_FM_WIDTH / RESNET_IN_BUF_COLS); tj++)
        {
            std::cout << "Processing Tile " << ti*(RESNET_IN_FM_WIDTH / RESNET_IN_BUF_COLS) + tj + 1;
            std::cout << "/" << num_of_tiles << std::endl;

            for(int b = 0; b < (RESNET_OUT_FM_DEPTH / RESNET_OUT_BUF_CH); b++) 
	        {
                for(int d = 0; d < (RESNET_IN_FM_DEPTH / RESNET_IN_BUF_CH); d++) 
                {
                    // Load input feature map from AXI input into input buffer
                    // Assumption: No Padding
                    resnet_load_input_fm_tile<RESNET_IN_FM_DEPTH, RESNET_IN_FM_HEIGHT, RESNET_IN_FM_WIDTH, RESNET_LAST_LAYER_EN>
                                      (in_fm_buf, resnet_layer_in_fm, ti, tj, 0, d);
                                      //(in_fm_buf, in_fm, ti, tj, 0, d);
	   
	                for(int f = 0; f < RESNET_IN_BUF_CH; f++)
	                {
	                    // Load weights from on-chip memory
                        resnet_load_weights_1x1<RESNET_OUT_FM_DEPTH, RESNET_IN_FM_DEPTH>
                                        (weight_buf_1x1, conv_weights, f, b, d);
	                
                        // Perform convolution
                        resnet_conv_1x1(out_fm_buf, in_fm_buf, weight_buf_1x1, f, STRIDE);
	                }
	                // Save partial output feature map in a buffer
                    resnet_save_partial_out_buf(partial_out_fm_buf, out_fm_buf, d);
                }
	   
                // Load BatchNorm params 
                resnet_load_batchnorm_params<RESNET_OUT_FM_DEPTH>(param_buf, bn_params, b);

	            // BatchNorm
	            resnet_batchnorm(partial_out_fm_buf, param_buf, false);
          
                // Load residual feature map tile TODO
                resnet_load_residual_fm_tile(ds_fm_buf, ds_fm, ti, tj, b);
                
                // Add downsampled batchnorm output with ReLU
                resnet_add_residual_fm(partial_out_fm_buf, ds_fm_buf, true);
          
	            // Store output feature map to DDR (off-chip)
                resnet_store_out_buf_to_DDR<STRIDE>(resnet_layer_out_fm, partial_out_fm_buf, ti, tj, b);
            }
        }
    }
}
