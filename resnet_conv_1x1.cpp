#include "qdtrack_resnet0.h"

void resnet_conv_1x1 (
        fm_t Y_buf[RESNET_OUT_BUF_CH][RESNET_OUT_BUF_ROWS][RESNET_OUT_BUF_COLS], 
        fm_t X_buf[RESNET_IN_BUF_CH][RESNET_IN_BUF_ROWS + 5][RESNET_IN_BUF_COLS + 5],
        wt_t W_buf[RESNET_IN_BUF_CH],
        int f,
        int S
)
{
    // For each row in stride steps
    for (int i = 0; i < RESNET_OUT_BUF_ROWS; i += S)
    {
        // For each column in stride steps
        for (int j = 0; j < RESNET_OUT_BUF_COLS; j += S)
        {
            // Clear output buffer
            Y_buf[f][i / S][j / S] = 0.0;

            // For each channel (pipelined)
            CHANNEL:
            for (int c = 0; c < RESNET_OUT_BUF_CH; c++)
            {
                // Perform convolution operation (i.e., element-wise MAC)
                // Stride = S
                Y_buf[f][i / S][j / S] += X_buf[c][i][j] * W_buf[c];
            }
        }
    }

    /////////////////////////////////////////////////////////////////////////////////////////
    // // For each row in stride steps
    // for(int i = 0; i < RESNET_OUT_BUF_ROWS; i=i+S) 
    // {
    //     // For each column in stride steps
    //     for(int j = 0; j < RESNET_OUT_BUF_COLS; j=j+S) 
    //     {
    //         // Clear output buffer
    //         Y_buf[f][i/S][j/S] = 0.0;

    //         // For each channel (pipelined)
	//         for(int c = 0; c < RESNET_OUT_BUF_CH; c++)
    //         {
    //             // Perform convolution operation (i.e. element-wise MAC)
    //             // Stride = S
    //             Y_buf[f][i/S][j/S] += X_buf[c][i][j] * W_buf[c]; 
    //         }
    //     }
    // }
}
