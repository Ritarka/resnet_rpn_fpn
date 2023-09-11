#include "qdtrack.h"
//#include <cmath>
//#include "hls_math.h"

const wt_t EPSILON = 0.00001;

void resnet_batchnorm(
        fm_t feature_map[RESNET_OUT_BUF_CH][RESNET_OUT_BUF_ROWS][RESNET_OUT_BUF_COLS], 
        wt_t bn_params[3][RESNET_OUT_BUF_CH], 
        bool enable_relu)
{
   for(int k = 0; k < RESNET_OUT_BUF_CH; k++)
   {
       for(int i = 0; i < RESNET_OUT_BUF_ROWS; i++)
       {
           for(int j = 0; j < RESNET_OUT_BUF_COLS; j++)
	       {
	           feature_map[k][i][j] -= bn_params[2][k]; // Subtract mean
	           //feature_map[k][i][j] *= (bn_params[0][k]/(sqrt(bn_params[3][k] + EPSILON)));
	           feature_map[k][i][j] *= bn_params[0][k]; // Multiply (weight/sqrt(var + EPSILON)
	           feature_map[k][i][j] += bn_params[1][k]; // Add bias

               if(enable_relu && (feature_map[k][i][j] < 0.0)) 
                   feature_map[k][i][j] = 0.0;
	       }
       }
       //printf("c: %d, W: %f, B: %f, U: %f, V: %f\n", k, bn_params[0][k], 
       //               bn_params[1][k], bn_params[2][k], bn_params[3][k]);
   }
}

void resnet_add_residual_fm(
        fm_t feature_map[RESNET_OUT_BUF_CH][RESNET_OUT_BUF_ROWS][RESNET_OUT_BUF_COLS], 
        fm_t residual_map[RESNET_OUT_BUF_CH][RESNET_OUT_BUF_ROWS][RESNET_OUT_BUF_COLS], 
        bool enable_relu)
{
   for(int k = 0; k < RESNET_OUT_BUF_CH; k++)
   {
       for(int i = 0; i < RESNET_OUT_BUF_ROWS; i++)
       {
           for(int j = 0; j < RESNET_OUT_BUF_COLS; j++)
	       {
               feature_map[k][i][j] += residual_map[k][i][j];
               
               if(enable_relu && (feature_map[k][i][j] < 0.0)) 
                   feature_map[k][i][j] = 0.0;
           }
       }
   }
}
