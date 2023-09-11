#include "qdtrack.h"

//template<const int S>
void rpn_conv_3x3 (
        fm_t Y_buf[RPN_OUT_BUF_CH][RPN_OUT_BUF_ROWS][RPN_OUT_BUF_COLS], 
        fm_t X_buf[RPN_IN_BUF_CH][RPN_IN_BUF_ROWS + 5][RPN_IN_BUF_COLS + 5],
        wt_t W_buf[RPN_IN_BUF_CH][3][3],
        int f,
        int S
)
{
#pragma HLS inline off
    //#pragma HLS array_partition variable=out_fm dim=1 complete
    //#pragma HLS array_partition variable=in_fm dim=1 complete
    //#pragma HLS array_partition variable=wt_buf dim=1 complete

    // For each row in stride steps
    for(int i = 0; i < RPN_OUT_BUF_ROWS; i=i+S) 
    {
        // For each column in stride steps
        for(int j = 0; j < RPN_OUT_BUF_COLS; j=j+S) 
        {
            //std::cout << i << " " << j << std::endl;
            
            // Clear output buffer
            Y_buf[f][i/S][j/S] = 0.0;

            // For each channel (pipelined)
	        for(int c = 0; c < RPN_OUT_BUF_CH; c++)
            {
                // Perform convolution operation (i.e. element-wise MAC)
                for(int m = i; m < i + 3; m++) // Kernel height = 3
                {
                    for(int n = j; n < j + 3; n++) // Kernel width = 3
                    {
                        Y_buf[f][i/S][j/S] += X_buf[c][m][n] * W_buf[c][m-i][n-j];
                    }
                }
            }
        }
    }
}
