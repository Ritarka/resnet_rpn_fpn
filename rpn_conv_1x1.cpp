#include "qdtrack.h"

void rpn_conv_1x1 (
        fm_t Y_buf[RPN_OUT_BUF_CH][RPN_OUT_BUF_ROWS][RPN_OUT_BUF_COLS], 
        fm_t X_buf[RPN_IN_BUF_CH][RPN_IN_BUF_ROWS + 5][RPN_IN_BUF_COLS + 5],
        wt_t W_buf[RPN_IN_BUF_CH][1][1],
        int f,
        int S
)
{
    // Unroll the loop over channels for increased parallelism
    #pragma HLS UNROLL factor=4
    for(int c = 0; c < RPN_OUT_BUF_CH; c += 4)
    {
        // For each row in stride steps
        for(int i = 0; i < RPN_OUT_BUF_ROWS; i += S) 
        {
            // For each column in stride steps
            for(int j = 0; j < RPN_OUT_BUF_COLS; j += S) 
            {
                // Clear output buffer
                Y_buf[f][i/S][j/S] = 0.0;
                
                // Unroll the loop for the channel dimension
                #pragma HLS UNROLL factor=4
                for(int k = 0; k < 4; k++)
                {
                    // Perform convolution operation (i.e., element-wise MAC)
                    // Stride = S
                    Y_buf[f][i/S][j/S] += X_buf[c + k][i][j] * W_buf[c + k][0][0];
                }
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    // // #pragma HLS inline off
    // // #pragma HLS array_partition variable=Y_buf dim=1 complete
    // // #pragma HLS array_partition variable=X_buf dim=1 complete
    // // #pragma HLS array_partition variable=W_buf dim=1 complete

    // // For each row in stride steps
    // for(int i = 0; i < RPN_OUT_BUF_ROWS; i=i+S) 
    // {
    //     // For each column in stride steps
    //     for(int j = 0; j < RPN_OUT_BUF_COLS; j=j+S) 
    //     {
    //         // Clear output buffer
    //         Y_buf[f][i/S][j/S] = 0.0;

    //         // For each channel (pipelined)
	//         for(int c = 0; c < RPN_OUT_BUF_CH; c++)
    //         {
    //             // Perform convolution operation (i.e. element-wise MAC)
    //             // Stride = S
    //             Y_buf[f][i/S][j/S] += X_buf[c][i][j] * W_buf[c][0][0]; 
    //         }
    //     }
    // }
}
