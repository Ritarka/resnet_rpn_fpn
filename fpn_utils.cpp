#include "fpn.h"

template<const int fm_input_depth,const int fm_input_height,const int fm_input_width,const int N_TILE_ROWS,const int N_TILE_COLS>
void fpn_load_input_tile_block_from_DRAM_3x3 (
    fm_t in_fm_buf[FPN_IN_BUF_CH][FPN_IN_BUF_ROWS+2][FPN_IN_BUF_COLS+2], 
    fm_t in_fm[fm_input_depth][fm_input_height][fm_input_width], 
    int  ti, 
    int  tj, 
    int  d
)
{
    const int depth_offset  =  d * FPN_IN_BUF_CH;
    const int height_offset = ti * FPN_OUT_BUF_ROWS; // OUT_BUF is intended, not a typo. 
    const int width_offset  = tj * FPN_OUT_BUF_COLS;
        
    INPUT_BUFFER_DEPTH:
    for(int c = 0; c < FPN_IN_BUF_CH; c++)
    {
        INPUT_BUFFER_HEIGHT:
        for(int i = 0; i < FPN_IN_BUF_ROWS+2; i++)
        {
            INPUT_BUFFER_WIDTH:
            for(int j = 0; j < FPN_IN_BUF_COLS+2; j++)
            {
                // For tile cubes in the first column and/or the first row, 
                // pad zero to the left and/or at the top (similar to Part A)
                if((ti == 0 && i < 1) || (tj == 0 && j < 1))
		        {
		            in_fm_buf[c][i][j] = (fm_t) 0;
		        }
                // For tile cubes in the last column and/or the last row, 
                // pad zero to the right and/or at the bottom (similar to Part A)
                else if(((ti == N_TILE_ROWS - 1) && (i > FPN_OUT_BUF_ROWS)) 
                     || ((tj == N_TILE_COLS - 1) && (j > FPN_OUT_BUF_COLS)))
		        {
		            in_fm_buf[c][i][j] = (fm_t) 0;
		        }
                // For all other tiles, copy each feature as is from input feature map
		        else
		        {
                    in_fm_buf[c][i][j] = in_fm[depth_offset + c][height_offset + i - 1][width_offset + j - 1];
		        }
            }
        }
    }
}
template<const int fm_input_depth,const int fm_input_height,const int fm_input_width,const int N_TILE_ROWS,const int N_TILE_COLS>
void fpn_load_input_tile_block_from_DRAM_1x1 (
    fm_t in_fm_buf[FPN_IN_BUF_CH][FPN_IN_BUF_ROWS][FPN_IN_BUF_COLS], 
    fm_t in_fm[fm_input_depth][fm_input_height][fm_input_width], 
    int  ti, 
    int  tj, 
    int  d
)
{
    const int depth_offset  =  d * FPN_IN_BUF_CH;
    const int height_offset = ti * FPN_OUT_BUF_ROWS;
    const int width_offset  = tj * FPN_OUT_BUF_COLS;
        
    INPUT_BUFFER_DEPTH:
    for(int c = 0; c < FPN_IN_BUF_CH; c++)
    {
        INPUT_BUFFER_HEIGHT:
        for(int i = 0; i < FPN_IN_BUF_ROWS; i++)
        {
            INPUT_BUFFER_WIDTH:
            for(int j = 0; j < FPN_IN_BUF_COLS; j++)
            {
          	in_fm_buf[c][i][j] = in_fm[depth_offset + c][height_offset + i][width_offset + j];
            }
        }
    }
}

//--------------------------------------------------------------------------
// Function to load layer parameters (weights and bias) for convolution.
//--------------------------------------------------------------------------
template<const int output_depth,const int input_depth>
void fpn_load_layer_params_from_DRAM_1x1 (
    wt_t weight_buf[FPN_OUT_BUF_CH][FPN_IN_BUF_CH][1][1],
    wt_t bias_buf[FPN_OUT_BUF_CH],
    wt_t weights[output_depth][input_depth][1][1],
    wt_t bias[output_depth],
    int b,
    int d
)
{
#pragma HLS inline off

    const int kernel_offset  = b * FPN_OUT_BUF_CH;
    const int channel_offset = d * FPN_IN_BUF_CH;

    WEIGHT_KERNEL_NUM:
    for(int f = 0; f < FPN_OUT_BUF_CH; f++)
    {
        WEIGHT_KERNEL_DEPTH:
        for(int c = 0; c < FPN_IN_BUF_CH; c++)
        {
            WEIGHT_KERNEL_HEIGHT:
            for(int kh = 0; kh < 1; kh++)
	    {
                WEIGHT_KERNEL_WIDTH:
	        for(int kw = 0; kw < 1; kw++)
	        {
	         	weight_buf[f][c][kh][kw] = weights[kernel_offset + f][channel_offset + c][kh][kw];
                }
            }
        }
    }
    
    BIAS:
    for(int f = 0; f < FPN_OUT_BUF_CH; f++)
    {
        bias_buf[f] = bias[kernel_offset + f];
    }

}
template<const int output_depth,const int input_depth>
void fpn_load_layer_params_from_DRAM_3x3 (
    wt_t weight_buf[FPN_OUT_BUF_CH][FPN_IN_BUF_CH][3][3],
    wt_t bias_buf[FPN_OUT_BUF_CH],
    wt_t weights[output_depth][input_depth][3][3],
    wt_t bias[output_depth],
    int b,
    int d
)
{
#pragma HLS inline off

    const int kernel_offset  = b * FPN_OUT_BUF_CH;
    const int channel_offset = d * FPN_IN_BUF_CH;

    WEIGHT_KERNEL_NUM:
    for(int f = 0; f < FPN_OUT_BUF_CH; f++)
    {
        WEIGHT_KERNEL_DEPTH:
        for(int c = 0; c < FPN_IN_BUF_CH; c++)
        {
            WEIGHT_KERNEL_HEIGHT:
            for(int kh = 0; kh < 3; kh++)
	        {
                WEIGHT_KERNEL_WIDTH:
	            for(int kw = 0; kw < 3; kw++)
	            {
	                weight_buf[f][c][kh][kw] = weights[kernel_offset + f][channel_offset + c][kh][kw];
                }
            }
        }
    }
    
    BIAS:
    for(int f = 0; f < FPN_OUT_BUF_CH; f++)
    {
        bias_buf[f] = bias[kernel_offset + f];
    }
}

//------------------------------------------------------------------------------
// Function to save partial outputs on-chip for each input tile slice processed.
//------------------------------------------------------------------------------
template<const int output_depth>
void fpn_save_partial_output_tile_block (
    fm_t partial_out_buf[FPN_OUT_BUF_CH][FPN_OUT_BUF_ROWS][FPN_OUT_BUF_COLS], 
    fm_t out_fm_buf[FPN_OUT_BUF_CH][FPN_OUT_BUF_ROWS][FPN_OUT_BUF_COLS],
    wt_t bias_buf[FPN_OUT_BUF_CH],
    int  d
)
{
    PARTIAL_OUTPUT_BUFFER_HEIGHT:
    for(int i = 0; i < FPN_OUT_BUF_ROWS; i++)
    {
        PARTIAL_OUTPUT_BUFFER_WIDTH:
        for(int j = 0; j < FPN_OUT_BUF_COLS; j++)
        {
            PARTIAL_OUTPUT_BUFFER_DEPTH:
            for(int f = 0; f < FPN_OUT_BUF_CH; f++)
            {
#pragma HLS unroll
                if(d == 0) // Initialize buffer for first kernel group and add bias
                {
                    partial_out_buf[f][i][j]   = out_fm_buf[f][i][j] + bias_buf[f];
                }
                else // Accumulate otherwise
                {
                    partial_out_buf[f][i][j]  += out_fm_buf[f][i][j];
                }
            }
        }
    }
}

//--------------------------------------------------------------------------
// Function to store complete output tile block from BRAM to DRAM.
//--------------------------------------------------------------------------
template<const int fm_output_depth,const int fm_output_height,const int fm_output_width>
void fpn_store_output_tile_to_DRAM (
    fm_t out_fm[fm_output_depth][fm_output_height][fm_output_width], 
    fm_t out_fm_buf[FPN_OUT_BUF_CH][FPN_OUT_BUF_ROWS][FPN_OUT_BUF_COLS], 
    int  ti,
    int  tj,
    int  b
)
{
    const int depth_offset  =  b * FPN_OUT_BUF_CH;
    const int height_offset = ti * FPN_OUT_BUF_ROWS;
    const int width_offset  = tj * FPN_OUT_BUF_COLS;

    OUTPUT_BUFFER_DEPTH:
    for(int f = 0; f < FPN_OUT_BUF_CH; f++)
    {
        OUTPUT_BUFFER_HEIGHT:
        for(int i = 0; i < FPN_OUT_BUF_ROWS; i++)
        {
            OUTPUT_BUFFER_WIDTH:
            for(int j = 0; j < FPN_OUT_BUF_COLS; j++)
            {
              	out_fm[depth_offset + f][height_offset + i][width_offset + j] = out_fm_buf[f][i][j];
            }
        }
    }
}

//Function to store outputs to DRAM
//template<const int fm_output_depth,const int fm_output_height,const int fm_output_width>
//void store_output_tile_to_DRAM (
//    fm_t out_fm[fm_output_depth][fm_output_height][fm_output_width], 
//    fm_t out_fm_buf[FPN_OUT_BUF_CH][FPN_OUT_BUF_ROWS][FPN_OUT_BUF_COLS],
//    wt_t bias_buf[FPN_OUT_BUF_CH],
//    int  ti,
//    int  tj,
//    int  b,
//    int  d
//)
//{
//    const int depth_offset  =  b * FPN_OUT_BUF_CH;
//    const int height_offset = ti * FPN_OUT_BUF_ROWS;
//    const int width_offset  = tj * FPN_OUT_BUF_COLS;
//
//    OUTPUT_BUFFER_DEPTH:
//    for(int f = 0; f < FPN_OUT_BUF_CH; f++)
//    {
//        OUTPUT_BUFFER_HEIGHT:
//        for(int i = 0; i < FPN_OUT_BUF_ROWS; i++)
//        {
//            OUTPUT_BUFFER_WIDTH:
//            for(int j = 0; j < FPN_OUT_BUF_COLS; j++)
//            {
//		if(d == 0){ 	// Initialize buffer for first kernel group and add bias
//			out_fm[depth_offset + f][height_offset + i][width_offset + j] = out_fm_buf[f][i][j] + bias_buf[f];
//		}	
//		else{		// Accumulate otherwise
//              		out_fm[depth_offset + f][height_offset + i][width_offset + j] += out_fm_buf[f][i][j];
//		}
//            }
//        }
//    }
//}
