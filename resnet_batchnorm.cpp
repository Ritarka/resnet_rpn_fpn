#include "qdtrack_resnet0.h"
//#include <cmath>
// #include "hls_math.h"

const wt_t EPSILON = 0.00001;

void resnet_batchnorm(
        fm_t feature_map[RESNET_OUT_BUF_CH][RESNET_OUT_BUF_ROWS][RESNET_OUT_BUF_COLS], 
        wt_t bn_params[3][RESNET_OUT_BUF_CH], 
        bool enable_relu)
{
    for (int k = 0; k < RESNET_OUT_BUF_CH; k++)
    {
        wt_t scale = bn_params[0][k];
        wt_t bias = bn_params[1][k];
        wt_t mean = bn_params[2][k];

        for (int i = 0; i < RESNET_OUT_BUF_ROWS; i++)
        {
            for (int j = 0; j < RESNET_OUT_BUF_COLS; j++)
            {
                feature_map[k][i][j] = (feature_map[k][i][j] - mean) * scale + bias;

                if (enable_relu && feature_map[k][i][j] < 0.0)
                {
                    feature_map[k][i][j] = 0.0;
                }
            }
        }
    }

    /////////////////////////////////////////////////////////////////////////////////////////////
    // for (int k = 0; k < RESNET_OUT_BUF_CH; k++)
    // {
    //     wt_t scale = bn_params[0][k];
    //     wt_t bias = bn_params[1][k];
    //     wt_t mean = bn_params[2][k];

    //     if (enable_relu)
    //     {
    //         for (int i = 0; i < RESNET_OUT_BUF_ROWS; i++)
    //         {
    //             for (int j = 0; j < RESNET_OUT_BUF_COLS; j++)
    //             {
    //                 feature_map[k][i][j] = (feature_map[k][i][j] - mean) * scale + bias;
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
    //                 feature_map[k][i][j] = (feature_map[k][i][j] - mean) * scale + bias;
    //             }
    //         }
    //     }
    // }

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

    for (int k = 0; k < RESNET_OUT_BUF_CH; k++)
    {
        for (int i = 0; i < RESNET_OUT_BUF_ROWS; i++)
        {
            for (int j = 0; j < RESNET_OUT_BUF_COLS; j++)
            {
                feature_map[k][i][j] += residual_map[k][i][j];

                if (enable_relu && feature_map[k][i][j] < 0.0)
                {
                    feature_map[k][i][j] = 0.0;
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
