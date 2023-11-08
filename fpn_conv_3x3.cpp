//---------------------------------------------------------------------------
// ECE-8893 Parallel Programming for FPGA
// 
// Reference solution to Lab-2 Part-B and Part-C
//
// The following code is optimized to achieve a 320x speedup in convolution
// computation by unrolling across the output width (20x) and 
// output depth (16x) dimensions.
//
// Author: Akshay Kamath
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
// Perform synthesizable tiling-based convolution for a single tile slice.
//---------------------------------------------------------------------------
#define FPN_H_ONLY_HEADERS_
#include "fpn0.h"

void fpn_conv_3x3 (
    fm_t Y_buf[FPN_OUT_BUF_CH][FPN_OUT_BUF_ROWS][FPN_OUT_BUF_COLS], 
    fm_t X_buf[FPN_IN_BUF_CH][FPN_IN_BUF_ROWS+2][FPN_IN_BUF_COLS+2],
    wt_t W_buf[FPN_OUT_BUF_CH][FPN_IN_BUF_CH][3][3]
)
{
    fm_t temp[FPN_OUT_BUF_CH][FPN_OUT_BUF_COLS];
    
    for (int i = 0; i < FPN_OUT_BUF_ROWS; i++)
    {
        for (int j = 0; j < FPN_OUT_BUF_COLS; j++)
        {
            for (int c = 0; c < FPN_IN_BUF_CH; c++)
            {
                temp[c][j] = 0.0;
                
                for (int m = 0; m < 3; m++)
                {
                    for (int n = 0; n < 3; n++)
                    {
                        for (int f = 0; f < FPN_OUT_BUF_CH; f++)
                        {
                            temp[f][j] += X_buf[c][i + m][j + n] * W_buf[f][c][m][n];
                            Y_buf[f][i][j] = temp[f][j];
                        }
                    }
                }
            }
        }
    }


    //////////////////////////////////////////////////////////////////////////////////////////
//     fm_t temp[FPN_OUT_BUF_CH][FPN_OUT_BUF_COLS];
// #pragma HLS array_partition variable=temp dim=0 complete

//     TILE_SLICE_DEPTH: 
//     for(int c = 0; c < FPN_IN_BUF_CH; c++)
//     {
//         KERNEL_HEIGHT: 
//         for(int m = 0; m < 3; m++) 
//         {
//             KERNEL_WIDTH: 
//             for(int n = 0; n < 3; n++) 
//             {
//                 TILE_SLICE_HEIGHT: 
//                 for(int i = 0; i < FPN_OUT_BUF_ROWS; i++) 
//                 {
// #pragma HLS pipeline
//                     // Parallelize computation across all columns of a row
//                     TILE_SLICE_WIDTH: 
//                     for(int j = 0; j < FPN_OUT_BUF_COLS; j++) 
//                     {
// #pragma HLS unroll
//                         // Setup pipeline
//                         temp[0][j] = (c == 0 && m == 0 && n == 0) ? (fm_t) 0 : Y_buf[0][i][j];
                           
//                         // Infer 16 instances of DSPs (with muxes) for each column feature 
//                         KERNEL_GROUP: 
//                         for(int f = 0; f < FPN_OUT_BUF_CH; f++) 
//                         {
// #pragma HLS unroll
//                             // MAC for current feature
//                             temp[f][j] += X_buf[c][i+m][j+n] * W_buf[f][c][m][n];
                            
//                             // Initialize feature for next iteration
//                             if(f != FPN_OUT_BUF_CH - 1) 
//                             {
//                                 temp[f+1][j] = (c == 0 && m == 0 && n == 0) ? (fm_t) 0 : Y_buf[f+1][i][j];
//                             }
                            
//                             // Store feature computed in previous iteration to output
//                             if(f > 0) 
//                             {
//                                 Y_buf[f-1][i][j] = temp[f-1][j];
//                             }
//                         }

//                         // Store last feature in the pipeline
//                         Y_buf[FPN_OUT_BUF_CH-1][i][j] = temp[FPN_OUT_BUF_CH-1][j];
//                     }
//                 }
//             }
//         }
//     } 
}
