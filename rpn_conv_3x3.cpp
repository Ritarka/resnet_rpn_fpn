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
                // Unroll the loop for the channel dimension
                #pragma HLS UNROLL factor=4
                for(int k = 0; k < 4; k++)
                {
                    // Clear output buffer only once per tile
                    if (k == 0)
                    {
                        Y_buf[f][i/S][j/S] = 0.0;
                    }

                    // Perform convolution operation (i.e., element-wise MAC)
                    for(int m = i; m < i + 3; m++) // Kernel height = 3
                    {
                        for(int n = j; n < j + 3; n++) // Kernel width = 3
                        {
                            Y_buf[f][i/S][j/S] += X_buf[c + k][m][n] * W_buf[c + k][m-i][n-j];
                        }
                    }
                }
            }
        }
    }
    
    //////////////////////////////////////////////////////////////////////////////////
    // // For each row in stride steps
    // for (int i = 0; i < RPN_OUT_BUF_ROWS; i += S)
    // {
    //     // For each column in stride steps
    //     for (int j = 0; j < RPN_OUT_BUF_COLS; j += S)
    //     {
    //         // Clear output buffer
    //         Y_buf[f][i / S][j / S] = 0.0;

    //         // For each channel (pipelined)
    //         CHANNEL:
    //         for (int c = 0; c < RPN_OUT_BUF_CH; c++)
    //         {
    //             // Perform convolution operation (i.e., element-wise MAC)
    //             for (int m = 0; m < 3; m++) // Kernel height = 3
    //             {
    //                 for (int n = 0; n < 3; n++) // Kernel width = 3
    //                 {
    //                     // Accumulate results into the output buffer
    //                     Y_buf[f][i / S][j / S] += X_buf[c][i + m][j + n] * W_buf[c][m][n];
    //                 }
    //             }
    //         }
    //     }
    // }

    ///////////////////////////////////////////////////////////////////////////////////////////
    // #pragma HLS inline off
    // // #pragma HLS array_partition variable=Y_buf dim=1 complete
    // // #pragma HLS array_partition variable=X_buf dim=1 complete
    // // #pragma HLS array_partition variable=W_buf dim=1 complete

    // // Unroll the outer loop by the stride 'S'
    // for (int i = 0; i < RPN_OUT_BUF_ROWS; i += S) 
    // {
    //     for (int j = 0; j < RPN_OUT_BUF_COLS; j += S) 
    //     {
    //         // Clear output buffer
    //         Y_buf[f][i / S][j / S] = 0.0;

    //         // Loop unrolling and loop transformations
    //         // Iterate over channels (pipelined)
    //         CHANNEL:
    //         for (int c = 0; c < RPN_OUT_BUF_CH; c++)
    //         {
    //             // Initialize a temporary variable to accumulate results
    //             fm_t temp_result = 0.0;

    //             // Loop unrolling for the kernel computation
    //             // Unroll by 2 for both height and width
    //             for (int m = i; m < i + 3; m += 2) // Kernel height = 3
    //             {
    //                 for (int n = j; n < j + 3; n += 2) // Kernel width = 3
    //                 {
    //                     // Use partial results to reduce memory access
    //                     temp_result += X_buf[c][m][n] * W_buf[c][m - i][n - j] +
    //                                   X_buf[c][m + 1][n] * W_buf[c][m - i + 1][n - j] +
    //                                   X_buf[c][m][n + 1] * W_buf[c][m - i][n - j + 1] +
    //                                   X_buf[c][m + 1][n + 1] * W_buf[c][m - i + 1][n - j + 1];
    //                 }
    //             }

    //             // Update the result in the output buffer
    //             Y_buf[f][i / S][j / S] += temp_result;
    //         }
    //     }
    // }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    // #pragma HLS inline off
    // // #pragma HLS array_partition variable=Y_buf dim=1 complete
    // // #pragma HLS array_partition variable=X_buf dim=1 complete
    // // #pragma HLS array_partition variable=W_buf dim=1 complete

    // // For each row in stride steps
    // for(int i = 0; i < RPN_OUT_BUF_ROWS; i=i+S) 
    // {
    //     // For each column in stride steps
    //     for(int j = 0; j < RPN_OUT_BUF_COLS; j=j+S) 
    //     {
    //         //std::cout << i << " " << j << std::endl;
            
    //         // Clear output buffer
    //         Y_buf[f][i/S][j/S] = 0.0;

    //         // For each channel (pipelined)
	//         for(int c = 0; c < RPN_OUT_BUF_CH; c++)
    //         {
    //             // Perform convolution operation (i.e. element-wise MAC)
    //             for(int m = i; m < i + 3; m++) // Kernel height = 3
    //             {
    //                 for(int n = j; n < j + 3; n++) // Kernel width = 3
    //                 {
    //                     Y_buf[f][i/S][j/S] += X_buf[c][m][n] * W_buf[c][m-i][n-j];
    //                 }
    //             }
    //         }
    //     }
    // }
}
