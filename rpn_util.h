#include "hls_stream.h"
#include "qdtrack.h"

// Feature Map Buffers 
fm_t rpn_in_fm_buf[RPN_IN_BUF_CH][RPN_IN_BUF_ROWS + 5][RPN_IN_BUF_COLS + 5];
fm_t rpn_out_fm_buf[RPN_OUT_BUF_CH][RPN_OUT_BUF_ROWS][RPN_OUT_BUF_COLS];
fm_t rpn_partial_out_fm_buf[RPN_OUT_BUF_CH][RPN_OUT_BUF_ROWS][RPN_OUT_BUF_COLS];
fm_t rpn_ds_fm_buf[RPN_OUT_BUF_CH][RPN_OUT_BUF_ROWS][RPN_OUT_BUF_COLS];

// TODO: Replace these by pointer-based access to DRAM
//float rpn_layer_out_fm[N_LAYER4_0_DS_OUT_CH][N_LAYER4_0_FM_HEIGHT][N_LAYER4_0_FM_WIDTH];
//float rpn_layer_in_fm[N_LAYER4_0_DS_OUT_CH][N_LAYER4_0_FM_HEIGHT][N_LAYER4_0_FM_WIDTH];
//float ds_fm[N_LAYER4_0_DS_OUT_CH][N_LAYER4_0_FM_HEIGHT][N_LAYER4_0_FM_WIDTH];
fm_t rpn_layer_out_fm[2048][184][320];
fm_t rpn_layer_in_fm[2048][184][320];

// Convolution Weight Buffers
wt_t rpn_weight_buf_1x1[RPN_IN_BUF_CH][1][1];
wt_t rpn_weight_buf_3x3[RPN_IN_BUF_CH][3][3];
wt_t rpn_weight_buf_7x7[RPN_IN_BUF_CH][7][7];

// Conv bias
wt_t rpn_param_buf[RPN_OUT_BUF_CH];

template<const int N_OUT_CH>
void rpn_load_bias_params (
        fm_t param_buf[RPN_OUT_BUF_CH], 
        wt_t params[N_OUT_CH], 
        int b
)
{
    for(int j = 0; j < RPN_OUT_BUF_CH; j++)
    {
        param_buf[j] = params[b * RPN_OUT_BUF_CH + j];
    }
}


void rpn_conv_bias_add(
        fm_t feature_map[RPN_OUT_BUF_CH][RPN_OUT_BUF_ROWS][RPN_OUT_BUF_COLS], 
        wt_t bias_params[RPN_OUT_BUF_CH], 
        bool relu)
{
   for(int k = 0; k < RPN_OUT_BUF_CH; k++)
   {
       for(int i = 0; i < RPN_OUT_BUF_ROWS; i++)
       {
           for(int j = 0; j < RPN_OUT_BUF_COLS; j++)
	       {
	           feature_map[k][i][j] += bias_params[k];
               if(relu && feature_map[k][i][j]<=0)  feature_map[k][i][j]=0; 
	           
	       }
       }
   }
}


// Load input feature map tile
// TODO: Replace with pointer-based DRAM access
template<const int  N_IN_FM_DEPTH, const int  N_IN_FM_HEIGHT, const int  N_IN_FM_WIDTH,
         const int  LAST_LAYER_EN>
void rpn_load_input_fm_tile (
        fm_t in_fm_buf[RPN_IN_BUF_CH][RPN_IN_BUF_ROWS + 5][RPN_IN_BUF_COLS + 5], 
        fm_t in_fm[2048][184][320], 
        int   ti, 
        int   tj, 
        int   P,
        int   d
)
{
    for(int c = 0; c < RPN_IN_BUF_CH; c++)
    {
        for(int i = 0; i < RPN_IN_BUF_ROWS + 5; i++)
        {
            for(int j = 0; j < RPN_IN_BUF_COLS + 5; j++)
            {
                // For the first tile row, pad zero to the left and/or at the top
                if((ti == 0 && i < P) || (tj == 0 && j < P))
		        {
		            in_fm_buf[c][i][j] = 0.0;
		        }
                // For the last tile row, pad zero to the right and/or at the bottom
                else if(((ti == (N_IN_FM_HEIGHT/RPN_IN_BUF_ROWS) - 1) && (i > RPN_IN_BUF_ROWS + P - 1)) 
                     || ((tj == (N_IN_FM_WIDTH/RPN_IN_BUF_COLS) - 1) && (j > RPN_IN_BUF_COLS + P - 1)))
		        {
		            in_fm_buf[c][i][j] = 0.0;
		        }
                // For an input feature map smaller than buffer size, pad zero
                // at at the bottom of the feature map
                else if(LAST_LAYER_EN && (i > (RPN_IN_BUF_ROWS/2) + P - 1)) // Assumes buffer height is even
		        {
		            in_fm_buf[c][i][j] = 0.0;
		        }
                // Copy the input feature map elements otherwise
		        else
		        {
		            in_fm_buf[c][i][j] = in_fm[d*RPN_IN_BUF_CH + c][ti*RPN_IN_BUF_ROWS + i - P][tj*RPN_IN_BUF_COLS + j - P];
		        }
            }
        }
    }
}

// Load previous layer's output (shortcut connection)
void rpn_load_residual_fm_tile (
        fm_t in_fm_buf[RPN_IN_BUF_CH][RPN_IN_BUF_ROWS][RPN_IN_BUF_COLS], 
        fm_t in_fm[2048][184][320], 
        int   ti, 
        int   tj,
        int   b 
)
{
    for(int c = 0; c < RPN_IN_BUF_CH; c++)
    {
        for(int i = 0; i < RPN_IN_BUF_ROWS; i++)
        {
            for(int j = 0; j < RPN_IN_BUF_COLS; j++)
            {
		        in_fm_buf[c][i][j] = in_fm[b*RPN_IN_BUF_CH + c][ti*RPN_IN_BUF_ROWS + i][tj*RPN_IN_BUF_COLS + j];
            }
        }
    }
}

// Load weights for 3x3 convolution
template<const int N_OUT_CH, const int N_IN_CH>
void rpn_load_weights_3x3 (
        wt_t weight_buf_3x3[RPN_IN_BUF_CH][3][3],
        wt_t weights[N_OUT_CH][N_IN_CH][3][3],
        int f,
        int b,
        int d
)
{
    for(int c = 0; c < RPN_IN_BUF_CH; c++)
    {
        for(int kh = 0; kh < 3; kh++)
	    {
	        for(int kw = 0; kw < 3; kw++)
	        {
	            weight_buf_3x3[c][kh][kw] = weights[b * RPN_OUT_BUF_CH + f][d * RPN_IN_BUF_CH + c][kh][kw];
            }
        }
    }
}

template<const int N_OUT_CH, const int N_IN_CH>
void rpn_load_weights_1x1 (
        wt_t weight_buf_1x1[RPN_IN_BUF_CH][1][1],
        wt_t weights[N_OUT_CH][N_IN_CH][1][1],
        int f,
        int b,
        int d
)
{
    for(int i = 0; i < RPN_IN_BUF_CH; i++)
    {
	    weight_buf_1x1[i][0][0] = weights[b * RPN_OUT_BUF_CH + f][d * RPN_IN_BUF_CH + i][0][0];
    }
}
// Save partial outputs when depth of output feature map is more than
// the output buffer depth
void rpn_save_partial_out_buf (
        fm_t partial_out_fm_buf[RPN_OUT_BUF_CH][RPN_OUT_BUF_ROWS][RPN_OUT_BUF_COLS], 
        fm_t out_fm_buf[RPN_OUT_BUF_CH][RPN_OUT_BUF_ROWS][RPN_OUT_BUF_COLS],
        int d
)
{
    for(int f = 0; f < RPN_OUT_BUF_CH; f++)
    {
        for(int i = 0; i < RPN_OUT_BUF_ROWS; i++)
        {
            for(int j = 0; j < RPN_OUT_BUF_COLS; j++)
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
void rpn_store_out_buf_to_DDR(
        fm_t out_fm[2048][184][320], 
        fm_t fm_buf[RPN_OUT_BUF_CH][RPN_OUT_BUF_ROWS][RPN_OUT_BUF_COLS], 
        int   ti, 
        int   tj,
        int   b
)
{
    for(int f = 0; f < RPN_OUT_BUF_CH; f++)
    {
        for(int i = 0; i < RPN_OUT_BUF_ROWS/S; i++)
        {
            for(int j = 0; j < RPN_OUT_BUF_COLS/S; j++)
            {
                 out_fm[b*RPN_OUT_BUF_CH + f][ti*(RPN_OUT_BUF_ROWS/S) + i][tj*(RPN_OUT_BUF_COLS/S) + j] = fm_buf[f][i][j];
            }
        }
    }
}

template<const int  N_IN_FM_DEPTH, const int  N_IN_FM_HEIGHT, const int  N_IN_FM_WIDTH,
         const int N_OUT_FM_DEPTH, const int N_OUT_FM_HEIGHT, const int N_OUT_FM_WIDTH,
         const int STRIDE, const int LAST_LAYER_EN, const int row_mod, const int col_mod, const int o_depth_mod, const int i_depth_mod>
void rpn_1x1_conv(
        wt_t conv_weights[N_OUT_FM_DEPTH][N_IN_FM_DEPTH][1][1],
        wt_t conv_bias[N_OUT_FM_DEPTH],
        bool  relu
)
{
    // int row_mod = (N_IN_FM_HEIGHT % RPN_IN_BUF_ROWS)!= 0;
    // int col_mod = (N_IN_FM_WIDTH % RPN_IN_BUF_COLS)!=0 ;
    // int o_depth_mod = (N_OUT_FM_DEPTH % RPN_OUT_BUF_CH)!=0 ;
    // int i_depth_mod = (N_IN_FM_DEPTH % RPN_IN_BUF_CH)!=0 ;
    
    // cout<<endl<<row_mod<<" "<<col_mod<<" "<<o_depth_mod<<" "<<i_depth_mod<<endl;
    const int num_of_tiles = ((N_IN_FM_HEIGHT / RPN_IN_BUF_ROWS) +row_mod ) * ((N_IN_FM_WIDTH / RPN_IN_BUF_COLS)+col_mod);
    std::cout << "\nNo. of tiles in input feature map = " << num_of_tiles << std::endl;

    for(int ti = 0; ti < (N_IN_FM_HEIGHT / RPN_IN_BUF_ROWS) + row_mod; ti++)
    {
        for(int tj = 0; tj < (N_IN_FM_WIDTH / RPN_IN_BUF_COLS) +col_mod; tj++)
        {
            std::cout << "Processing Tile " << ti*(N_IN_FM_WIDTH / RPN_IN_BUF_COLS) + tj + 1;
            std::cout << "/" << num_of_tiles << std::endl;

            for(int b = 0; b < (N_OUT_FM_DEPTH / RPN_OUT_BUF_CH)+o_depth_mod; b++) 
	        {
                for(int d = 0; d < (N_IN_FM_DEPTH / RPN_IN_BUF_CH)+i_depth_mod; d++) 
                {
                    //cout << ti << " " << tj << " " << b << " " << d << " " << endl;

                    // Load input feature map from AXI input into input buffer
                    // Assumption: Zero padding
                    rpn_load_input_fm_tile<N_IN_FM_DEPTH, N_IN_FM_HEIGHT, N_IN_FM_WIDTH, LAST_LAYER_EN>
                                      (rpn_in_fm_buf, rpn_layer_in_fm, ti, tj, 0, d);
                                      //(rpn_in_fm_buf, in_fm, ti, tj, 0, d);

	                for(int f = 0; f < RPN_IN_BUF_CH; f++)
	                {
	                    // Load weights from on-chip memory
                        rpn_load_weights_1x1<N_OUT_FM_DEPTH, N_IN_FM_DEPTH>
                                        (rpn_weight_buf_1x1, conv_weights, f, b, d);
	                
                        // Perform convolution
                        rpn_conv_1x1(rpn_out_fm_buf, rpn_in_fm_buf, rpn_weight_buf_1x1, f, STRIDE);
	                }
                    
	                // Save partial output feature map in a buffer
                    rpn_save_partial_out_buf(rpn_partial_out_fm_buf, rpn_out_fm_buf, d);
                }
                
                rpn_load_bias_params<N_OUT_FM_DEPTH> (rpn_param_buf,conv_bias,b);
                rpn_conv_bias_add(rpn_partial_out_fm_buf, rpn_param_buf, relu);
                
	            // Store output feature map to DDR (off-chip)
                rpn_store_out_buf_to_DDR<STRIDE>(rpn_layer_out_fm, rpn_partial_out_fm_buf, ti, tj, b);
            }
        }
    }
}

// Performs 3x3 convolution with optionally ReLU
template<const int N_IN_FM_DEPTH,  const int N_IN_FM_HEIGHT,  const int N_IN_FM_WIDTH,
         const int N_OUT_FM_DEPTH, const int N_OUT_FM_HEIGHT, const int N_OUT_FM_WIDTH, 
         const int STRIDE, const int LAST_LAYER_EN, const int ROW_MOD>
void rpn_3x3_conv(
        wt_t conv_weights[N_OUT_FM_DEPTH][N_IN_FM_DEPTH][3][3],
        wt_t conv_bias[N_OUT_FM_DEPTH],
        bool relu
)
{
    
    const int num_of_tiles = ((N_IN_FM_HEIGHT / RPN_IN_BUF_ROWS) +ROW_MOD ) * ((N_IN_FM_WIDTH / RPN_IN_BUF_COLS)+ROW_MOD);
    std::cout << "\nNo. of tiles in input feature map = " << num_of_tiles << std::endl;

    loop1: for(int ti = 0; ti < (N_IN_FM_HEIGHT / RPN_IN_BUF_ROWS) + ROW_MOD; ti++)
    {
        for(int tj = 0; tj < (N_IN_FM_WIDTH / RPN_IN_BUF_COLS) +ROW_MOD; tj++)
        {
            std::cout << "Processing Tile " << ti*(N_IN_FM_WIDTH / RPN_IN_BUF_COLS) + tj + 1;
            std::cout << "/" << num_of_tiles << std::endl;

            for(int b = 0; b < (N_OUT_FM_DEPTH / RPN_OUT_BUF_CH); b++) 
	        {
                for(int d = 0; d < (N_IN_FM_DEPTH / RPN_IN_BUF_CH); d++) 
                {
                    //cout << ti << " " << tj << " " << b << " " << d << " " << endl;
                    
                    // Load input feature map from AXI input into input buffer
                    // Assumption: Padding = (1,1)
                    rpn_load_input_fm_tile<N_IN_FM_DEPTH, N_IN_FM_HEIGHT, N_IN_FM_WIDTH, LAST_LAYER_EN>
                                      (rpn_in_fm_buf, rpn_layer_in_fm, ti, tj, 1, d);
                                      //(rpn_in_fm_buf, in_fm, ti, tj, 1, d);
                    
	                for(int f = 0; f < RPN_IN_BUF_CH; f++)
	                {
	                    // Load weights from on-chip memory
                        rpn_load_weights_3x3<N_OUT_FM_DEPTH, N_IN_FM_DEPTH>
                                        (rpn_weight_buf_3x3, conv_weights, f, b, d);
	                
                        // Perform convolution TODO
                        //conv_3x3<2>(rpn_out_fm_buf, rpn_in_fm_buf, rpn_weight_buf_3x3, f); // Linking error
                        rpn_conv_3x3(rpn_out_fm_buf, rpn_in_fm_buf, rpn_weight_buf_3x3, f, STRIDE);
	                }
	   
	                // Save partial output feature map in a buffer
                    rpn_save_partial_out_buf(rpn_partial_out_fm_buf, rpn_out_fm_buf, d);
                }
                

                rpn_load_bias_params<N_OUT_FM_DEPTH> (rpn_param_buf,conv_bias,b);
                rpn_conv_bias_add(rpn_partial_out_fm_buf, rpn_param_buf, relu);

	            // Store output feature map to DDR (off-chip)
                rpn_store_out_buf_to_DDR<STRIDE>(rpn_layer_out_fm, rpn_partial_out_fm_buf, ti, tj, b);
            }
        }
    }
}
