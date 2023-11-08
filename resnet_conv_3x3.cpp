#include "qdtrack_resnet0.h"

//template<const int S>
void resnet_conv_3x3 (
        fm_t Y_buf[RESNET_OUT_BUF_CH][RESNET_OUT_BUF_ROWS][RESNET_OUT_BUF_COLS], 
        fm_t X_buf[RESNET_IN_BUF_CH][RESNET_IN_BUF_ROWS + 5][RESNET_IN_BUF_COLS + 5],
        wt_t W_buf[RESNET_IN_BUF_CH][3][3],
        int f,
        int S
)
{
    fm_t local_Y[RESNET_OUT_BUF_CH][RESNET_OUT_BUF_ROWS][RESNET_OUT_BUF_COLS];
#pragma HLS array_partition variable=local_Y dim=1 complete
#pragma HLS array_partition variable=local_Y dim=2 complete
#pragma HLS array_partition variable=local_Y dim=3 complete

    // For each row in stride steps
    for(int i = 0; i < RESNET_OUT_BUF_ROWS; i += S) 
    {
        // For each column in stride steps
        for(int j = 0; j < RESNET_OUT_BUF_COLS; j += S) 
        {
#pragma HLS pipeline
            // Initialize local accumulators
            fm_t local_accum[RESNET_OUT_BUF_CH] = {0.0};

            // For each channel (pipelined)
            CHANNEL:
            for(int c = 0; c < RESNET_OUT_BUF_CH; c++)
            {
                // Unroll kernel height and width loops
                KERNEL_HEIGHT:
                for(int m = i; m < i + 3; m++)
                {
                    KERNEL_WIDTH:
                    for(int n = j; n < j + 3; n++)
                    {
                        // Perform convolution operation (i.e., element-wise MAC)
                        local_accum[c] += X_buf[c][m][n] * W_buf[c][m-i][n-j];
                    }
                }
            }

            // Assign the results to the local buffer
            for(int c = 0; c < RESNET_OUT_BUF_CH; c++)
            {
                local_Y[c][i/S][j/S] = local_accum[c];
            }
        }
    }

    // Copy the local buffer to the global output buffer
    for(int c = 0; c < RESNET_OUT_BUF_CH; c++)
    {
        for(int i = 0; i < RESNET_OUT_BUF_ROWS; i += S)
        {
            for(int j = 0; j < RESNET_OUT_BUF_COLS; j += S)
            {
#pragma HLS pipeline
                Y_buf[c][i/S][j/S] = local_Y[c][i/S][j/S];
            }
        }
    }

//     fm_t local_Y[RESNET_OUT_BUF_ROWS][RESNET_OUT_BUF_COLS];
// #pragma HLS array_partition variable=local_Y dim=1 complete
// #pragma HLS array_partition variable=local_Y dim=2 complete


//     // For each row in stride steps
//     for(int i = 0; i < RESNET_OUT_BUF_ROWS; i=i+S) 
//     {
//         // For each column in stride steps
//         for(int j = 0; j < RESNET_OUT_BUF_COLS; j=j+S) 
//         {
//             // Initialize local accumulator
//             fm_t local_accum = 0.0;

//             // For each channel (pipelined)
//             CHANNEL:
//             for(int c = 0; c < RESNET_OUT_BUF_CH; c++)
//             {
//                 // Unroll kernel height and width loops
//                 KERNEL_HEIGHT:
//                 for(int m = i; m < i + 3; m++)
//                 {
//                     KERNEL_WIDTH:
//                     for(int n = j; n < j + 3; n++)
//                     {
//                         // Perform convolution operation (i.e., element-wise MAC)
//                         local_accum += X_buf[c][m][n] * W_buf[c][m-i][n-j];
//                     }
//                 }
//             }

//             // Assign the result to the local buffer
//             local_Y[i/S][j/S] = local_accum;
//         }
//     }

//     // Copy the local buffer to the global output buffer
//     for(int i = 0; i < RESNET_OUT_BUF_ROWS; i=i+S) 
//     {
//         for(int j = 0; j < RESNET_OUT_BUF_COLS; j=j+S) 
//         {
// #pragma HLS pipeline
//             Y_buf[f][i/S][j/S] = local_Y[i/S][j/S];
//         }
//     }
    
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //     // Loop over rows in stride steps
    // for (int i = 0; i < RESNET_OUT_BUF_ROWS; i += S)
    // {
    //     // Loop over columns in stride steps
    //     for (int j = 0; j < RESNET_OUT_BUF_COLS; j += S)
    //     {
    //         // Clear output buffer
    //         Y_buf[f][i / S][j / S] = 0.0;

    //         // Loop over channels (pipelined)
    //         CHANNEL:
    //         for (int c = 0; c < RESNET_OUT_BUF_CH; c++)
    //         {
    //             // Unroll the kernel computation loops for height and width
    //             // to increase parallelism
    //             for (int m = 0; m < 3; m++)
    //             {
    //                 for (int n = 0; n < 3; n++)
    //                 {
    //                     // Calculate the current indices for input and weight
    //                     int x_row = i + m;
    //                     int x_col = j + n;

    //                     // Calculate the current weight index
    //                     int w_row = m;
    //                     int w_col = n;

    //                     // Compute the convolution result for a single element
    //                     fm_t temp_result = X_buf[c][x_row][x_col] * W_buf[c][w_row][w_col];

    //                     // Accumulate the result into the output buffer
    //                     Y_buf[f][i / S][j / S] += temp_result;
    //                 }
    //             }
    //         }
    //     }
    // }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // // Loop over rows in stride steps
    // for (int i = 0; i < RESNET_OUT_BUF_ROWS; i += S)
    // {
    //     // Loop over columns in stride steps
    //     for (int j = 0; j < RESNET_OUT_BUF_COLS; j += S)
    //     {
    //         // Clear output buffer
    //         Y_buf[f][i / S][j / S] = 0.0;

    //         // Loop over channels (pipelined)
    //         CHANNEL:
    //         for (int c = 0; c < RESNET_OUT_BUF_CH; c++)
    //         {
    //             // Loop unrolling for kernel computation
    //             // Unroll by 2 for both height and width for parallelism
    //             for (int m = i; m < i + 3; m += 2) // Kernel height = 3
    //             {
    //                 for (int n = j; n < j + 3; n += 2) // Kernel width = 3
    //                 {
    //                     // Temporary variables to accumulate results
    //                     fm_t temp_results[2][2] = {0.0};

    //                     // Element-wise MAC operations
    //                     for (int mi = 0; mi < 2; mi++)
    //                     {
    //                         for (int ni = 0; ni < 2; ni++)
    //                         {
    //                             temp_results[mi][ni] = X_buf[c][m + mi][n + ni] * W_buf[c][m - i + mi][n - j + ni];
    //                         }
    //                     }

    //                     // Accumulate results into the output buffer
    //                     for (int mi = 0; mi < 2; mi++)
    //                     {
    //                         for (int ni = 0; ni < 2; ni++)
    //                         {
    //                             Y_buf[f][i / S][j / S] += temp_results[mi][ni];
    //                         }
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }

    //////////////////////////////////////////////////////////////////////////////////////////////////
    // // Loop unroll factors
    // const int UNROLL_C = 3; // Unroll the channel loop
    // const int UNROLL_M = 2; // Unroll the kernel height loop
    // const int UNROLL_N = 2; // Unroll the kernel width loop

    // // Unroll the outer loop by the stride 'S'
    // for (int i = 0; i < RESNET_OUT_BUF_ROWS; i += S) 
    // {
    //     for (int j = 0; j < RESNET_OUT_BUF_COLS; j += S) 
    //     {
    //         // Clear output buffer
    //         Y_buf[f][i / S][j / S] = 0.0;

    //         // Loop unrolling and loop transformations
    //         for (int c = 0; c < UNROLL_C; c++)
    //         {
    //             // Initialize a temporary variable to accumulate results
    //             fm_t temp_result = 0.0;

    //             // Loop unrolling for the kernel computation
    //             for (int m = i; m < i + 3; m += UNROLL_M) // Kernel height = 3
    //             {
    //                 for (int n = j; n < j + 3; n += UNROLL_N) // Kernel width = 3
    //                 {
    //                     // Use partial results to reduce memory access
    //                     for (int mi = 0; mi < UNROLL_M; mi++) {
    //                         for (int ni = 0; ni < UNROLL_N; ni++) {
    //                             temp_result += X_buf[c][m + mi][n + ni] * W_buf[c][m - i + mi][n - j + ni];
    //                         }
    //                     }
    //                 }
    //             }

    //             // Update the result in the output buffer
    //             Y_buf[f][i / S][j / S] += temp_result;
    //         }
    //     }
    // }


    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // // #pragma HLS array_partition variable=out_fm dim=1 complete
    // // #pragma HLS array_partition variable=in_fm dim=1 complete
    // // #pragma HLS array_partition variable=wt_buf dim=1 complete

    // // For each row in stride steps
    // for(int i = 0; i < RESNET_OUT_BUF_ROWS; i=i+S) 
    // {
    //     // For each column in stride steps
    //     for(int j = 0; j < RESNET_OUT_BUF_COLS; j=j+S) 
    //     {
    //         //std::cout << i << " " << j << std::endl;
            
    //         // Clear output buffer
    //         Y_buf[f][i/S][j/S] = 0.0;

    //         // For each channel (pipelined)
	//         for(int c = 0; c < RESNET_OUT_BUF_CH; c++)
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
