// #include "hls_stream.h"
// #include "qdtrack_resnet1_0.h"

// //template<const int S>
// void resnet_conv_7x7 (
//         fm_t Y_buf[RESNET_OUT_BUF_CH][RESNET_OUT_BUF_ROWS][RESNET_OUT_BUF_COLS], 
//         fm_t X_buf[RESNET_IN_BUF_CH][RESNET_IN_BUF_ROWS + 5][RESNET_IN_BUF_COLS + 5],
//         wt_t W_buf[RESNET_IN_BUF_CH][7][7],
//         int f
// )
// {
//     int S = 2; // Stride

//     // Local accumulators for each channel
//     fm_t local_Y[RESNET_OUT_BUF_CH][RESNET_OUT_BUF_ROWS][RESNET_OUT_BUF_COLS];
// #pragma HLS array_partition variable=local_Y dim=1 complete
// #pragma HLS array_partition variable=local_Y dim=2 complete
// #pragma HLS array_partition variable=local_Y dim=3 complete

//     // For each row in stride steps
//     for(int i = 0; i < RESNET_OUT_BUF_ROWS; i += S) 
//     {
//         // For each column in stride steps
//         for(int j = 0; j < RESNET_OUT_BUF_COLS; j += S) 
//         {
// #pragma HLS pipeline
//             // Initialize local accumulators
//             fm_t local_accum[RESNET_OUT_BUF_CH] = {0.0};

//             // For each channel (pipelined)
//             CHANNEL:
//             for(int c = 0; c < RESNET_OUT_BUF_CH; c++)
//             {
//                 // Unroll kernel height and width loops
//                 KERNEL_HEIGHT:
//                 for(int m = i; m < i + 7; m++)
//                 {
//                     KERNEL_WIDTH:
//                     for(int n = j; n < j + 7; n++)
//                     {
//                         // Perform convolution operation (i.e., element-wise MAC)
//                         local_accum[c] += X_buf[c][m][n] * W_buf[c][m-i][n-j];
//                     }
//                 }
//             }

//             // Assign the results to the local buffer
//             for(int c = 0; c < RESNET_OUT_BUF_CH; c++)
//             {
//                 local_Y[c][i/S][j/S] = local_accum[c];
//             }
//         }
//     }

//     // Copy the local buffer to the global output buffer
//     for(int c = 0; c < RESNET_OUT_BUF_CH; c++)
//     {
//         for(int i = 0; i < RESNET_OUT_BUF_ROWS; i += S)
//         {
//             for(int j = 0; j < RESNET_OUT_BUF_COLS; j += S)
//             {
// #pragma HLS pipeline
//                 Y_buf[c][i/S][j/S] = local_Y[c][i/S][j/S];
//             }
//         }
//     }

//     ////////////////////////////////////////////////////////////////////////////////////////////////////
//     // const int S = 2; // Stride

//     // // Loop over rows in stride steps
//     // for (int i = 0; i < RESNET_OUT_BUF_ROWS; i += S)
//     // {
//     //     // Loop over columns in stride steps
//     //     for (int j = 0; j < RESNET_OUT_BUF_COLS; j += S)
//     //     {
//     //         // Clear output buffer
//     //         Y_buf[f][i / S][j / S] = 0.0;

//     //         // Loop over channels (pipelined)
//     //         CHANNEL:
//     //         for (int c = 0; c < 3; c++)
//     //         {
//     //             // Unroll the kernel computation loops for height and width to increase parallelism
//     //             for (int m = i; m < i + 7; m += 1) // Kernel height = 7
//     //             {
//     //                 for (int n = j; n < j + 7; n += 1) // Kernel width = 7
//     //                 {
//     //                     // Temporary variables to accumulate results
//     //                     fm_t temp_result = 0.0;

//     //                     // Element-wise MAC operations
//     //                     for (int mi = 0; mi < 7; mi++)
//     //                     {
//     //                         for (int ni = 0; ni < 7; ni++)
//     //                         {
//     //                             temp_result += X_buf[c][m + mi][n + ni] * W_buf[c][mi][ni];
//     //                         }
//     //                     }

//     //                     // Accumulate the result into the output buffer
//     //                     Y_buf[f][i / S][j / S] += temp_result;
//     //                 }
//     //             }
//     //         }
//     //     }
//     // }

//     ////////////////////////////////////////////////////////////////////////////////////////////////////////
//     // const int S = 2; // Stride

//     // // Loop over rows in stride steps
//     // for (int i = 0; i < RESNET_OUT_BUF_ROWS; i += S)
//     // {
//     //     // Loop over columns in stride steps
//     //     for (int j = 0; j < RESNET_OUT_BUF_COLS; j += S)
//     //     {
//     //         // Clear output buffer
//     //         Y_buf[f][i / S][j / S] = 0.0;

//     //         // Loop over channels (pipelined)
//     //         CHANNEL:
//     //         for (int c = 0; c < 3; c++)
//     //         {
//     //             // Loop unrolling and loop transformations for kernel computation
//     //             // Unroll by 4 for both height and width for parallelism
//     //             for (int m = i; m < i + 7; m += 4) // Kernel height = 7
//     //             {
//     //                 for (int n = j; n < j + 7; n += 4) // Kernel width = 7
//     //                 {
//     //                     // Temporary variables to accumulate results
//     //                     fm_t temp_results[4][4] = {0.0};

//     //                     // Element-wise MAC operations
//     //                     for (int mi = 0; mi < 4; mi++)
//     //                     {
//     //                         for (int ni = 0; ni < 4; ni++)
//     //                         {
//     //                             temp_results[mi][ni] = X_buf[c][m + mi][n + ni] * W_buf[c][m - i + mi][n - j + ni];
//     //                         }
//     //                     }

//     //                     // Accumulate results into the output buffer
//     //                     for (int mi = 0; mi < 4; mi++)
//     //                     {
//     //                         for (int ni = 0; ni < 4; ni++)
//     //                         {
//     //                             Y_buf[f][i / S][j / S] += temp_results[mi][ni];
//     //                         }
//     //                     }
//     //                 }
//     //             }
//     //         }
//     //     }
//     // }

//     ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
//     // int S = 2; // Stride

//     // // Loop unroll factors
//     // const int UNROLL_C = 3; // Unroll the channel loop
//     // const int UNROLL_M = 2; // Unroll the kernel height loop
//     // const int UNROLL_N = 2; // Unroll the kernel width loop

//     // // Unroll the outer loop by the stride 'S'
//     // for (int i = 0; i < RESNET_OUT_BUF_ROWS; i += S) 
//     // {
//     //     for (int j = 0; j < RESNET_OUT_BUF_COLS; j += S) 
//     //     {
//     //         // Clear output buffer
//     //         Y_buf[f][i / S][j / S] = 0.0;

//     //         // Loop unrolling and loop transformations
//     //         // Iterate over channels (pipelined)
//     //         for (int c = 0; c < UNROLL_C; c++)
//     //         {
//     //             // Initialize a temporary variable to accumulate results
//     //             fm_t temp_result = 0.0;

//     //             // Loop unrolling for the kernel computation
//     //             for (int m = i; m < i + 7; m += UNROLL_M) // Kernel height = 7
//     //             {
//     //                 for (int n = j; n < j + 7; n += UNROLL_N) // Kernel width = 7
//     //                 {
//     //                     // Use partial results to reduce memory access
//     //                     for (int mi = 0; mi < UNROLL_M; mi++) {
//     //                         for (int ni = 0; ni < UNROLL_N; ni++) {
//     //                             temp_result += X_buf[c][m + mi][n + ni] * W_buf[c][m - i + mi][n - j + ni];
//     //                         }
//     //                     }
//     //                 }
//     //             }

//     //             // Update the result in the output buffer
//     //             Y_buf[f][i / S][j / S] += temp_result;
//     //         }
//     //     }
//     // }

//     ///////////////////////////////////////////////////////////////////////////////////////
//     // // #pragma HLS array_partition variable=out_fm dim=1 complete
//     // // #pragma HLS array_partition variable=in_fm dim=1 complete
//     // // #pragma HLS array_partition variable=wt_buf dim=1 complete

//     // int S = 2; // Stride TODO: Template

//     // // For each row in stride steps
//     // for(int i = 0; i < RESNET_OUT_BUF_ROWS; i=i+S) 
//     // {
//     //     // For each column in stride steps
//     //     for(int j = 0; j < RESNET_OUT_BUF_COLS; j=j+S) 
//     //     {
//     //         //std::cout << i << " " << j << std::endl;
            
//     //         // Clear output buffer
//     //         Y_buf[f][i/S][j/S] = 0.0;

//     //         // For each channel (pipelined)
// 	//         for(int c = 0; c < 3; c++)
//     //         {
//     //             // Perform convolution operation (i.e. element-wise MAC)
//     //             for(int m = i; m < i + 7; m++) // Kernel height = 7
//     //             {
//     //                 for(int n = j; n < j + 7; n++) // Kernel width = 7
//     //                 {
//     //                     Y_buf[f][i/S][j/S] += X_buf[c][m][n] * W_buf[c][m-i][n-j];
//     //                 }
//     //             }
//     //         }
//     //     }
//     // }
// }

#include "hls_stream.h"
#include "qdtrack_resnet2_0.h"

//template<const int S>
void resnet_conv_7x7 (
        fm_t Y_buf[RESNET_OUT_BUF_CH][RESNET_OUT_BUF_ROWS][RESNET_OUT_BUF_COLS], 
        fm_t X_buf[RESNET_IN_BUF_CH][RESNET_IN_BUF_ROWS + 5][RESNET_IN_BUF_COLS + 5],
        wt_t W_buf[RESNET_IN_BUF_CH][7][7],
        int f
)
{
    //#pragma HLS array_partition variable=out_fm dim=1 complete
    //#pragma HLS array_partition variable=in_fm dim=1 complete
    //#pragma HLS array_partition variable=wt_buf dim=1 complete
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
