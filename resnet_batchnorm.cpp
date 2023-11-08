#include "qdtrack_resnet0.h"

const wt_t EPSILON = 0.00001;

void resnet_batchnorm(
        fm_t feature_map[RESNET_OUT_BUF_CH][RESNET_OUT_BUF_ROWS][RESNET_OUT_BUF_COLS], 
        wt_t bn_params[3][RESNET_OUT_BUF_CH], 
        bool enable_relu)
{
// #pragma HLS inline
// #pragma HLS array_partition variable=feature_map complete dim=1
// #pragma HLS array_partition variable=bn_params complete dim=1

//     // Precompute constants for each channel
//     wt_t scale[RESNET_OUT_BUF_CH];
//     wt_t bias[RESNET_OUT_BUF_CH];
//     wt_t mean[RESNET_OUT_BUF_CH];

//     LOOP1: for (int k = 0; k < RESNET_OUT_BUF_CH; k++) {
// #pragma HLS pipeline
//         scale[k] = bn_params[0][k];
//         bias[k] = bn_params[1][k];
//         mean[k] = bn_params[2][k];
//     }

//     LOOP2: for (int k = 0; k < RESNET_OUT_BUF_CH; k++) {
//         LOOP3: for (int i = 0; i < RESNET_OUT_BUF_ROWS; i++) {
//             LOOP4: for (int j = 0; j < RESNET_OUT_BUF_COLS; j++) {
// #pragma HLS pipeline
//                 fm_t feature_val = feature_map[k][i][j];
//                 wt_t scale_val = scale[k];
//                 wt_t bias_val = bias[k];
//                 wt_t mean_val = mean[k];
//                 fm_t normalized_value = (feature_val - mean_val) * scale_val + bias_val;
//                 if (enable_relu && normalized_value < 0.0) {
//                     feature_map[k][i][j] = 0.0;
//                 } else {
//                     feature_map[k][i][j] = normalized_value;
//                 }
//             }
//         }
//     }
    ////////////////////////////////////////////////////////////////////////////////////////
    #pragma HLS inline off

    wt_t scale[RESNET_OUT_BUF_CH];
    wt_t bias[RESNET_OUT_BUF_CH];
    wt_t mean[RESNET_OUT_BUF_CH];

    // Precompute constants for each channel
    for (int k = 0; k < RESNET_OUT_BUF_CH; k++)
    {
        scale[k] = bn_params[0][k];
        bias[k] = bn_params[1][k];
        mean[k] = bn_params[2][k];
    }

    for (int k = 0; k < RESNET_OUT_BUF_CH; k++)
    {
        for (int i = 0; i < RESNET_OUT_BUF_ROWS; i++)
        {
            for (int j = 0; j < RESNET_OUT_BUF_COLS; j++)
            {
                fm_t normalized_value = (feature_map[k][i][j] - mean[k]) * scale[k] + bias[k];
                if (enable_relu && normalized_value < 0.0)
                {
                    feature_map[k][i][j] = 0.0;
                }
                else
                {
                    feature_map[k][i][j] = normalized_value;
                }
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // // Loop unroll factors
    // const int UNROLL_K = 4; // Unroll the channel loop
    // const int UNROLL_I = 2; // Unroll the row loop
    // const int UNROLL_J = 2; // Unroll the column loop

    // for (int k = 0; k < RESNET_OUT_BUF_CH; k += UNROLL_K) {
    //     for (int i = 0; i < RESNET_OUT_BUF_ROWS; i += UNROLL_I) {
    //         for (int j = 0; j < RESNET_OUT_BUF_COLS; j += UNROLL_J) {
    //             // Loop over a chunk of channels
    //             for (int chunk_k = 0; chunk_k < UNROLL_K; chunk_k++) {
    //                 for (int chunk_i = 0; chunk_i < UNROLL_I; chunk_i++) {
    //                     for (int chunk_j = 0; chunk_j < UNROLL_J; chunk_j++) {
    //                         int current_k = k + chunk_k;
    //                         int current_i = i + chunk_i;
    //                         int current_j = j + chunk_j;

    //                         fm_t diff = feature_map[current_k][current_i][current_j] - bn_params[2][current_k];
    //                         diff *= bn_params[0][current_k];
    //                         diff += bn_params[1][current_k];

    //                         // Apply ReLU if enabled
    //                         if (enable_relu && (diff < 0.0)) {
    //                             diff = 0.0;
    //                         }

    //                         feature_map[current_k][current_i][current_j] = diff;
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    for(int k = 0; k < RESNET_OUT_BUF_CH; k++)
//    {
//        for(int i = 0; i < RESNET_OUT_BUF_ROWS; i++)
//        {
//            for(int j = 0; j < RESNET_OUT_BUF_COLS; j++)
// 	       {
// 	           feature_map[k][i][j] -= bn_params[2][k]; // Subtract mean
// 	           feature_map[k][i][j] *= bn_params[0][k]; // Multiply (weight/sqrt(var + EPSILON)
// 	           feature_map[k][i][j] += bn_params[1][k]; // Add bias

//                if(enable_relu && (feature_map[k][i][j] < 0.0)) 
//                    feature_map[k][i][j] = 0.0;
// 	       }
//        }
//    }
}

void resnet_add_residual_fm(
        fm_t feature_map[RESNET_OUT_BUF_CH][RESNET_OUT_BUF_ROWS][RESNET_OUT_BUF_COLS], 
        fm_t residual_map[RESNET_OUT_BUF_CH][RESNET_OUT_BUF_ROWS][RESNET_OUT_BUF_COLS], 
        bool enable_relu)
{

#pragma HLS array_partition variable=feature_map complete dim=1
#pragma HLS array_partition variable=feature_map complete dim=2
#pragma HLS array_partition variable=feature_map complete dim=3
#pragma HLS array_partition variable=residual_map complete dim=1
#pragma HLS array_partition variable=residual_map complete dim=2
#pragma HLS array_partition variable=residual_map complete dim=3

    for (int k = 0; k < RESNET_OUT_BUF_CH; k++) {
        for (int i = 0; i < RESNET_OUT_BUF_ROWS; i++) {
            for (int j = 0; j < RESNET_OUT_BUF_COLS; j++) {
#pragma HLS pipeline
                feature_map[k][i][j] += residual_map[k][i][j];

                if (enable_relu) {
                    feature_map[k][i][j] = (feature_map[k][i][j] > 0.0) ? feature_map[k][i][j] : 0.0;
                }
            }
        }
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////////
    // for (int k = 0; k < RESNET_OUT_BUF_CH; k++)
    // {
    //     if (enable_relu)
    //     {
    //         for (int i = 0; i < RESNET_OUT_BUF_ROWS; i++)
    //         {
    //             for (int j = 0; j < RESNET_OUT_BUF_COLS; j++)
    //             {
    //                 feature_map[k][i][j] = feature_map[k][i][j] + residual_map[k][i][j];
    //                 if (feature_map[k][i][j] < 0.0)
    //                 {
    //                     feature_map[k][i][j] = 0.0;
    //                 }
    //             }
    //         }
    //     }
    //     else
    //     {
    //         for (int i = 0; i < RESNET_OUT_BUF_ROWS; i++)
    //         {
    //             for (int j = 0; j < RESNET_OUT_BUF_COLS; j++)
    //             {
    //                 feature_map[k][i][j] = feature_map[k][i][j] + residual_map[k][i][j];
    //             }
    //         }
    //     }
    // }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    // // Loop unroll factors
    // const int UNROLL_K = 4; // Unroll the channel loop
    // const int UNROLL_I = 2; // Unroll the row loop
    // const int UNROLL_J = 2; // Unroll the column loop

    // for (int k = 0; k < RESNET_OUT_BUF_CH; k += UNROLL_K) {
    //     for (int i = 0; i < RESNET_OUT_BUF_ROWS; i += UNROLL_I) {
    //         for (int j = 0; j < RESNET_OUT_BUF_COLS; j += UNROLL_J) {
    //             // Loop over a chunk of channels
    //             for (int chunk_k = 0; chunk_k < UNROLL_K; chunk_k++) {
    //                 for (int chunk_i = 0; chunk_i < UNROLL_I; chunk_i++) {
    //                     for (int chunk_j = 0; chunk_j < UNROLL_J; chunk_j++) {
    //                         int current_k = k + chunk_k;
    //                         int current_i = i + chunk_i;
    //                         int current_j = j + chunk_j;

    //                         feature_map[current_k][current_i][current_j] += residual_map[current_k][current_i][current_j];

    //                         // Apply ReLU if enabled
    //                         if (enable_relu && (feature_map[current_k][current_i][current_j] < 0.0)) {
    //                             feature_map[current_k][current_i][current_j] = 0.0;
    //                         }
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }
}
