#include "qdtrack.h"
// #include "resnet.h"
// #include "resnet_util.h"

void resnet_layer0(
        fm_t   resnet_layer0_input_fm[RESNET_LAYER0_CONV1_IN_CH][RESNET_LAYER0_IN_FM_HEIGHT][RESNET_LAYER0_IN_FM_WIDTH],
        wt_t   resnet_layer0_conv1_weights[RESNET_LAYER0_CONV1_OUT_CH][RESNET_LAYER0_CONV1_IN_CH][7][7],
	    wt_t   resnet_layer0_bn1_params[3][RESNET_LAYER0_CONV1_OUT_CH],
        fm_t   resnet_layer0_output_fm[RESNET_LAYER0_CONV1_OUT_CH][RESNET_LAYER0_OUT_FM_HEIGHT][RESNET_LAYER0_OUT_FM_WIDTH]
)
{
    //TODO: Update with pointer-based loading of input feature map
    //std::cout << RESNET_LAYER0_CONV1_IN_CH << "\t" << RESNET_LAYER0_IN_FM_HEIGHT << "\t" << RESNET_LAYER0_IN_FM_WIDTH << std::endl;

    for(int c = 0; c < RESNET_LAYER0_CONV1_IN_CH; c++)
    {
        for(int h = 0; h < RESNET_LAYER0_IN_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RESNET_LAYER0_IN_FM_WIDTH; w++)
            {
                //resnet_layer0_in_fm[c][h][w] = (float) resnet_layer0_input_fm[c][h][w];
                resnet_layer0_in_fm[c][h][w] = resnet_layer0_input_fm[c][h][w];
            }
        }
    }

    //---------------------------------------------------------------
    // conv1 + bn1 + relu
    //---------------------------------------------------------------
    int P = 3; // Padding size
    int S = 2; // Stride step
    
    const int num_of_tiles = (RESNET_LAYER0_IN_FM_HEIGHT / RESNET_IN_BUF_ROWS) * (RESNET_LAYER0_IN_FM_WIDTH / RESNET_IN_BUF_COLS);
    std::cout << "\nNo. of tiles in input feature map = " << num_of_tiles << std::endl << std::endl;

    for(int ti = 0; ti < (RESNET_LAYER0_IN_FM_HEIGHT / RESNET_IN_BUF_ROWS); ti++)
    {
        for(int tj = 0; tj < (RESNET_LAYER0_IN_FM_WIDTH / RESNET_IN_BUF_COLS); tj++)
        {
            //std::cout << ti << "," << tj << std::endl;
            std::cout << "Processing Tile " << ti*(RESNET_LAYER0_IN_FM_WIDTH / RESNET_IN_BUF_COLS) + tj + 1;
            std::cout << "/" << num_of_tiles << std::endl;
            
            // Load input feature tile
            for(int c = 0; c < RESNET_LAYER0_CONV1_IN_CH; c++)
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
                        else if(((ti == (RESNET_LAYER0_IN_FM_HEIGHT/RESNET_IN_BUF_ROWS) - 1) && (i > RESNET_IN_BUF_ROWS + P - 1)) 
                             || ((tj == (RESNET_LAYER0_IN_FM_WIDTH/RESNET_IN_BUF_COLS) - 1) && (j > RESNET_IN_BUF_COLS + P - 1)))
	        	        {
	        	            in_fm_buf[c][i][j] = 0.0;
	        	        }
                        // For an input feature map smaller than buffer size, pad zero
                        // at at the bottom of the feature map
                        //else if(RESNET_LAST_LAYER_EN && (i > (RESNET_IN_BUF_ROWS/2) + P - 1)) // Assumes buffer height is even
	        	        //{
	        	        //    in_fm_buf[c][i][j] = 0.0;
	        	        //}
                        // Copy the input feature map elements otherwise
	        	        else
	        	        {
	        	            //in_fm_buf[c][i][j] = (float) resnet_layer0_in_fm[c][ti*RESNET_IN_BUF_ROWS + i - P][tj*RESNET_IN_BUF_COLS + j - P];
	        	            in_fm_buf[c][i][j] = (fm_t) resnet_layer0_in_fm[c][ti*RESNET_IN_BUF_ROWS + i - P][tj*RESNET_IN_BUF_COLS + j - P];
	        	        }
                    }
                }
            }

            for(int f = 0; f < RESNET_IN_BUF_CH; f++)
	        {
	            // Load weights from on-chip memory
                for(int c = 0; c < RESNET_LAYER0_CONV1_IN_CH; c++)
                {
                    for(int kh = 0; kh < 7; kh++)
	                {
	                    for(int kw = 0; kw < 7; kw++)
	                    {
	                        weight_buf_7x7[c][kh][kw] = resnet_layer0_conv1_weights[f][c][kh][kw];
                        }
                    }
                }
	        
                // Perform convolution
                resnet_conv_7x7(out_fm_buf, in_fm_buf, weight_buf_7x7, f);
	        }
            
            //std::cout << "Before batchnorm" << std::endl;
            // Load BatchNorm params
            resnet_load_batchnorm_params<RESNET_LAYER0_CONV1_OUT_CH>(param_buf, resnet_layer0_bn1_params, 0);

	        // BatchNorm
	        resnet_batchnorm(out_fm_buf, param_buf, true);
            //std::cout << "After batchnorm" << std::endl;
            
            // Store to DDR
            for(int f = 0; f < RESNET_OUT_BUF_CH; f++)
            {
                for(int i = 0; i < RESNET_OUT_BUF_ROWS/S; i++)
                {
                    for(int j = 0; j < RESNET_OUT_BUF_COLS/S; j++)
                    {
                         resnet_layer0_mx_fm[f][ti*(RESNET_OUT_BUF_ROWS/S) + i][tj*(RESNET_OUT_BUF_COLS/S) + j] = out_fm_buf[f][i][j];
                    }
                }
            }
        }
    }   
    
    std::cout << "Layer 0.0.1 done" << std::endl;    
    
    //---------------------------------------------------------------
    // Maxpool
    // Fixed-parameters, not templated
    // Padding row: Top row
    // Padding col: Leftmost column
    //---------------------------------------------------------------
    P = 1; // Padding size
    S = 2; // Stride step
    
    fm_t max_val = 0.0;

    for(int ti = 0; ti < (RESNET_LAYER0_MX_FM_HEIGHT / RESNET_IN_BUF_ROWS); ti++)
    {
       for(int tj = 0; tj < (RESNET_LAYER0_MX_FM_WIDTH / RESNET_IN_BUF_COLS); tj++)
       {
           // Load input feature tile
           for(int c = 0; c < RESNET_IN_BUF_CH; c++)
           {
               for(int i = 0; i < RESNET_IN_BUF_ROWS + P; i++)
               {
                   for(int j = 0; j < RESNET_IN_BUF_COLS + P; j++)
                   {
                       if((ti == 0 && i < P) || (tj == 0 && j < P))
		               {
		                   in_fm_buf[c][i][j] = 0.0;
		               }
                       else
                       {
                           //in_fm_buf[c][i][j] = (float) resnet_layer0_mx_fm[c][ti*RESNET_IN_BUF_ROWS + i - P][tj*RESNET_IN_BUF_COLS + j - P];
                           in_fm_buf[c][i][j] = (fm_t) resnet_layer0_mx_fm[c][ti*RESNET_IN_BUF_ROWS + i - P][tj*RESNET_IN_BUF_COLS + j - P];
                       }
                   }
               }
           }

           // Perform maxpooling on loaded tile
           for(int c = 0; c < RESNET_IN_BUF_CH; c++)
           {
               for(int i = P; i < RESNET_IN_BUF_ROWS + P; i+=S)
               {
                   for(int j = P; j < RESNET_IN_BUF_COLS + P; j+=S)
                   {
                       // Clear previous max value
                       max_val = 0.0;
                       
                       // Pick max value in a 3x3 window
                       for(int m = i-1; m <= i+1; m++)
                       {
                           for(int n = j-1; n <= j+1; n++)
                           {
                               if(in_fm_buf[c][m][n] > max_val)
                               {
                                   max_val = in_fm_buf[c][m][n];
                               }
                           }
                       }
                       
                       // Save max value in output buffer 
                       out_fm_buf[c][(i-1)/S][(j-1)/S] = max_val;
                   }
               }
           }

           // Store output tile to DDR
           for(int c = 0; c < RESNET_OUT_BUF_CH; c++)
           {
               for(int i = 0; i < RESNET_OUT_BUF_ROWS/S; i++)
               {
                   for(int j = 0; j < RESNET_OUT_BUF_COLS/S; j++)
                   {
                        resnet_layer0_out_fm[c][ti*(RESNET_OUT_BUF_ROWS/S) + i][tj*(RESNET_OUT_BUF_COLS/S) + j] = out_fm_buf[c][i][j];
                   }
               }
           }
       }
    }
    
    for(int c = 0; c < RESNET_LAYER0_CONV1_OUT_CH; c++)
    {
        for(int h = 0; h < RESNET_LAYER0_OUT_FM_HEIGHT; h++)
        {
            for(int w = 0; w < RESNET_LAYER0_OUT_FM_WIDTH; w++)
            {
                resnet_layer0_output_fm[c][h][w] = resnet_layer0_out_fm[c][h][w];
            }
        }
    }
    
    std::cout << "Layer 0.0.2 done" << std::endl;    
}
