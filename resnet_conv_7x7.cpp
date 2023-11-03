#include "hls_stream.h"
#include "qdtrack_resnet2.h"

//template<const int S>
void resnet_conv_7x7 (
        fm_t Y_buf[RESNET_OUT_BUF_CH][RESNET_OUT_BUF_ROWS][RESNET_OUT_BUF_COLS], 
        fm_t X_buf[RESNET_IN_BUF_CH][RESNET_IN_BUF_ROWS + 5][RESNET_IN_BUF_COLS + 5],
        wt_t W_buf[RESNET_IN_BUF_CH][7][7],
        int f
)
{
    // #pragma HLS array_partition variable=out_fm dim=1 complete
    // #pragma HLS array_partition variable=in_fm dim=1 complete
    // #pragma HLS array_partition variable=wt_buf dim=1 complete

    int S = 2; // Stride TODO: Template

    // For each row in stride steps
    for(int i = 0; i < RESNET_OUT_BUF_ROWS; i=i+S) 
    {
        // For each column in stride steps
        for(int j = 0; j < RESNET_OUT_BUF_COLS; j=j+S) 
        {
            //std::cout << i << " " << j << std::endl;
            
            // Clear output buffer
            Y_buf[f][i/S][j/S] = 0.0;

            // For each channel (pipelined)
	        for(int c = 0; c < RESNET_LAYER0_CONV1_IN_CH; c++)
            {
                // Perform convolution operation (i.e. element-wise MAC)
                for(int m = i; m < i + 7; m++) // Kernel height = 7
                {
                    for(int n = j; n < j + 7; n++) // Kernel width = 7
                    {
                        Y_buf[f][i/S][j/S] += X_buf[c][m][n] * W_buf[c][m-i][n-j];
                    }
                }
            }
        }
    }
}
