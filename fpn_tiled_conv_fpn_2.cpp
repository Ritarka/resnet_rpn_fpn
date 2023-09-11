//Author : Rohan Banerjee (rbanerjee45@gatech.edu)
//Description : Tiling convolution for 3x3 layer 2

#include "fpn.h"
#include "fpn_utils.cpp"


void fpn_tiled_conv_fpn_2 (
    fm_t input_feature_map[FPN_CONV_2_ID][FPN_CONV_2_IH][FPN_CONV_2_IW],
    wt_t layer_weights[FPN_CONV_2_OD][FPN_CONV_2_ID][3][3],
    wt_t layer_bias[FPN_CONV_2_OD],
    fm_t output_feature_map[FPN_CONV_2_OD][FPN_CONV_2_IH][FPN_CONV_2_IW]
)
{
#define N_TILE_ROWS (int) (FPN_CONV_2_IH/FPN_OUT_BUF_ROWS)
#define N_TILE_COLS (int) (FPN_CONV_2_IW/FPN_OUT_BUF_COLS)
//--------------------------------------------------------------------------
// Defines interface IO ports for HLS. 
//--------------------------------------------------------------------------

#pragma HLS INTERFACE m_axi depth=256*46*80 	port=input_feature_map   bundle=fm
#pragma HLS INTERFACE m_axi depth=256*256*3*3   port=layer_weights       bundle=wt
#pragma HLS INTERFACE m_axi depth=256          	port=layer_bias          bundle=wt
#pragma HLS INTERFACE m_axi depth=256*46*80 	port=output_feature_map  bundle=fm

#pragma HLS INTERFACE s_axilite register	port=return
    
    //--------------------------------------------------------------------------
    // On-chip buffers
    //--------------------------------------------------------------------------
    fm_t conv_in_buf[FPN_IN_BUF_CH][FPN_IN_BUF_ROWS+2][FPN_IN_BUF_COLS+2];
#pragma HLS array_partition variable=conv_in_buf    dim=3  complete
    
    wt_t conv_bias_buf[FPN_OUT_BUF_CH];
#pragma HLS array_partition variable=conv_bias_buf    dim=1  complete

    wt_t conv_wt_buf[FPN_OUT_BUF_CH][FPN_IN_BUF_CH][3][3];
#pragma HLS array_partition variable=conv_wt_buf    dim=1  complete
    
    fm_t conv_out_buf[FPN_OUT_BUF_CH][FPN_OUT_BUF_ROWS][FPN_OUT_BUF_COLS] = {0};
#pragma HLS array_partition variable=conv_out_buf   dim=1  complete
#pragma HLS array_partition variable=conv_out_buf   dim=3  complete
    
    // Partial output storage buffer   
    fm_t partial_out_fm_buf[FPN_OUT_BUF_CH][FPN_OUT_BUF_ROWS][FPN_OUT_BUF_COLS];
#pragma HLS array_partition variable=partial_out_fm_buf dim=1 complete
    
    // Process each tile iteratively
    TILE_ROW:
    for(int ti = 0; ti < N_TILE_ROWS; ti++)
    {
        TILE_COL:
        for(int tj = 0; tj < N_TILE_COLS; tj++)
        {
            std::cout << "Processing Tile " << ti*N_TILE_COLS + tj + 1;
            std::cout << "/" << N_TILE_ROWS * N_TILE_COLS << std::endl;    

            // Split filter into groups of appropriate number of kernels
            FILTER_SIZE:
            for(int b = 0; b < (FPN_CONV_2_OD/FPN_OUT_BUF_CH); b++) 
            {
                // Divide tile cube into appropriate tile slices
                TILE_CUBE:
                for(int d = 0; d < (FPN_CONV_2_ID/FPN_IN_BUF_CH); d++) 
                {
                    // Load input tile slice from DRAM
                    fpn_load_input_tile_block_from_DRAM_3x3 <FPN_CONV_2_ID,FPN_CONV_2_IH,FPN_CONV_2_IW,N_TILE_ROWS,N_TILE_COLS> 
							(conv_in_buf, 
                                                     	input_feature_map, 
                                                     	ti, 
                                                     	tj, 
                                                     	d);
                    
                    // Load weights and bias slices from DRAM
                    fpn_load_layer_params_from_DRAM_3x3 <FPN_CONV_2_OD,FPN_CONV_2_ID>
						(conv_wt_buf, 
                                                 conv_bias_buf, 
                                                 layer_weights, 
                                                 layer_bias, 
                                                 b, 
                                                 d);

                    // Run convolution on tile slice
                    fpn_conv_3x3 (conv_out_buf, 
                              conv_in_buf, 
                              conv_wt_buf
                            );

		    // Store computed partial output in BRAM
                    fpn_save_partial_output_tile_block<FPN_CONV_2_OD>(partial_out_fm_buf, 
                                                   conv_out_buf, 
                                                   conv_bias_buf, 
                                                   d);
                }
		
                // Store final output tile slice to DRAM
                fpn_store_output_tile_to_DRAM<FPN_CONV_2_OD,FPN_CONV_2_OH,FPN_CONV_2_OW>
					(output_feature_map, 
                                          partial_out_fm_buf, 
                                          ti, 
                                          tj, 
                                          b);
            }      
        }
    }
}
