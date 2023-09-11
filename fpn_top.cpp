#include <iostream>
#include <fstream>
#include <cmath>

#include "fpn.h"

using namespace std;

//Lateral conv_1x1 layer outputs
fm_t    lateral_0_output_feature_map[LATERAL_CONV_0_OD][LATERAL_CONV_0_IH][LATERAL_CONV_0_IW];
fm_t    lateral_1_output_feature_map[LATERAL_CONV_1_OD][LATERAL_CONV_1_IH][LATERAL_CONV_1_IW];
fm_t    lateral_2_output_feature_map[LATERAL_CONV_2_OD][LATERAL_CONV_2_IH][LATERAL_CONV_2_IW];
fm_t    lateral_3_output_feature_map[LATERAL_CONV_3_OD][LATERAL_CONV_3_IH][LATERAL_CONV_3_IW];

void fpn_top(
	fm_t lateral_3_input_feature_map[LATERAL_CONV_3_ID][LATERAL_CONV_3_IH][LATERAL_CONV_3_IW],
    	wt_t lateral_3_layer_weights[LATERAL_CONV_3_OD][LATERAL_CONV_3_ID][1][1],
    	wt_t lateral_3_layer_bias[LATERAL_CONV_3_OD],
	fm_t lateral_2_input_feature_map[LATERAL_CONV_2_ID][LATERAL_CONV_2_IH][LATERAL_CONV_2_IW],
    	wt_t lateral_2_layer_weights[LATERAL_CONV_2_OD][LATERAL_CONV_2_ID][1][1],
    	wt_t lateral_2_layer_bias[LATERAL_CONV_2_OD],
	fm_t lateral_1_input_feature_map[LATERAL_CONV_1_ID][LATERAL_CONV_1_IH][LATERAL_CONV_1_IW],
    	wt_t lateral_1_layer_weights[LATERAL_CONV_1_OD][LATERAL_CONV_1_ID][1][1],
    	wt_t lateral_1_layer_bias[LATERAL_CONV_1_OD],
	fm_t lateral_0_input_feature_map[LATERAL_CONV_0_ID][LATERAL_CONV_0_IH][LATERAL_CONV_0_IW],
    	wt_t lateral_0_layer_weights[LATERAL_CONV_0_OD][LATERAL_CONV_0_ID][1][1],
    	wt_t lateral_0_layer_bias[LATERAL_CONV_0_OD],
    	wt_t fpn_3_layer_weights[FPN_CONV_3_OD][FPN_CONV_3_ID][3][3],
    	wt_t fpn_3_layer_bias[FPN_CONV_3_OD],
    	fm_t fpn_3_output_feature_map[FPN_CONV_3_OD][FPN_CONV_3_IH][FPN_CONV_3_IW],
    	wt_t fpn_2_layer_weights[FPN_CONV_2_OD][FPN_CONV_2_ID][3][3],
    	wt_t fpn_2_layer_bias[FPN_CONV_2_OD],
    	fm_t fpn_2_output_feature_map[FPN_CONV_2_OD][FPN_CONV_2_IH][FPN_CONV_2_IW],
    	wt_t fpn_1_layer_weights[FPN_CONV_1_OD][FPN_CONV_1_ID][3][3],
    	wt_t fpn_1_layer_bias[FPN_CONV_1_OD],
    	fm_t fpn_1_output_feature_map[FPN_CONV_1_OD][FPN_CONV_1_IH][FPN_CONV_1_IW],
    	wt_t fpn_0_layer_weights[FPN_CONV_0_OD][FPN_CONV_0_ID][3][3],
    	wt_t fpn_0_layer_bias[FPN_CONV_0_OD],
    	fm_t fpn_0_output_feature_map[FPN_CONV_0_OD][FPN_CONV_0_IH][FPN_CONV_0_IW]
    	//fm_t lateral_0_output_feature_map[LATERAL_CONV_0_OD][LATERAL_CONV_0_IH][LATERAL_CONV_0_IW],
    	//fm_t lateral_1_output_feature_map[LATERAL_CONV_1_OD][LATERAL_CONV_1_IH][LATERAL_CONV_1_IW],
    	//fm_t lateral_2_output_feature_map[LATERAL_CONV_2_OD][LATERAL_CONV_2_IH][LATERAL_CONV_2_IW],
    	//fm_t lateral_3_output_feature_map[LATERAL_CONV_3_OD][LATERAL_CONV_3_IH][LATERAL_CONV_3_IW]
)
{
    //Lateral conv_1x1 layer outputs
    //fm_t    lateral_0_output_feature_map[LATERAL_CONV_0_OD][LATERAL_CONV_0_IH][LATERAL_CONV_0_IW];
    //fm_t    lateral_1_output_feature_map[LATERAL_CONV_1_OD][LATERAL_CONV_1_IH][LATERAL_CONV_1_IW];
    //fm_t    lateral_2_output_feature_map[LATERAL_CONV_2_OD][LATERAL_CONV_2_IH][LATERAL_CONV_2_IW];
    //fm_t    lateral_3_output_feature_map[LATERAL_CONV_3_OD][LATERAL_CONV_3_IH][LATERAL_CONV_3_IW];
    //INTERPOLATOR variable declarations
    static fm_t nn_interpl_out_3_2[LATERAL_CONV_2_OD][LATERAL_CONV_2_IH][LATERAL_CONV_2_IW];
    static fm_t nn_interpl_out_2_1[LATERAL_CONV_1_OD][LATERAL_CONV_1_IH][LATERAL_CONV_1_IW];
    static fm_t nn_interpl_out_1_0[LATERAL_CONV_0_OD][LATERAL_CONV_0_IH][LATERAL_CONV_0_IW];
	
    cout << "Processing Lateral 0 Layer..." << endl;
    fpn_tiled_conv_lateral_0 (lateral_0_input_feature_map,
                lateral_0_layer_weights,
                lateral_0_layer_bias,
                lateral_0_output_feature_map
    );
    cout << "Processing Lateral 1 Layer..." << endl;
    fpn_tiled_conv_lateral_1 (lateral_1_input_feature_map,
                lateral_1_layer_weights,
                lateral_1_layer_bias,
                lateral_1_output_feature_map
    );
    cout << "Processing Lateral 2 Layer..." << endl;
    fpn_tiled_conv_lateral_2 (lateral_2_input_feature_map,
                lateral_2_layer_weights,
                lateral_2_layer_bias,
                lateral_2_output_feature_map
    );
    cout << "Processing Lateral 3 Layer..." << endl;
    fpn_tiled_conv_lateral_3 (lateral_3_input_feature_map,
                lateral_3_layer_weights,
                lateral_3_layer_bias,
                lateral_3_output_feature_map
    );

    //-------TIME FOR INTERPOLATOR TO KICK IN-------//
    //INTERPOLATION FROM LAYER 3 TO LAYER 2
    for(int d=0; d < FPN_CONV_3_ID; d++){
        for(int h=0; h < FPN_CONV_3_IH*2; h++){
        	for(int w=0; w < FPN_CONV_3_IW*2; w++){
        		nn_interpl_out_3_2[d][h][w] = lateral_3_output_feature_map[d][h/2][w/2];
        	}
        }
    }
    for(int i=0;i<FPN_CONV_2_ID;i++){
        for(int j=0;j<FPN_CONV_2_IH;j++){
        	for(int k=0;k<FPN_CONV_2_IW;k++){
        		lateral_2_output_feature_map[i][j][k] += nn_interpl_out_3_2[i][j][k];
        	}
        }
    }

    //INTERPOLATION FROM LAYER 2 TO LAYER 1
    for(int d=0; d < FPN_CONV_2_ID; d++){
        for(int h=0; h < FPN_CONV_2_IH*2; h++){
        	for(int w=0; w < FPN_CONV_2_IW*2; w++){
        		nn_interpl_out_2_1[d][h][w] = lateral_2_output_feature_map[d][h/2][w/2];
        	}
        }
    }
    for(int i=0;i<FPN_CONV_1_ID;i++){
        for(int j=0;j<FPN_CONV_1_IH;j++){
        	for(int k=0;k<FPN_CONV_1_IW;k++){
        		lateral_1_output_feature_map[i][j][k] += nn_interpl_out_2_1[i][j][k];
        	}
        }
    }

    //INTERPOLATION FROM LAYER 1 TO LAYER 0
    for(int d=0; d < FPN_CONV_1_ID; d++){
        for(int h=0; h < FPN_CONV_1_IH*2; h++){
        	for(int w=0; w < FPN_CONV_1_IW*2; w++){
        		nn_interpl_out_1_0[d][h][w] = lateral_1_output_feature_map[d][h/2][w/2];
        	}
        }
    }
    for(int i=0;i<FPN_CONV_0_ID;i++){
        for(int j=0;j<FPN_CONV_0_IH;j++){
        	for(int k=0;k<FPN_CONV_0_IW;k++){
        		lateral_0_output_feature_map[i][j][k] += nn_interpl_out_1_0[i][j][k];
        	}
        }
    }

    cout << "Processing FPN Layer 0 ... " << endl;
    fpn_tiled_conv_fpn_0 (lateral_0_output_feature_map,
                fpn_0_layer_weights,
                fpn_0_layer_bias,
                fpn_0_output_feature_map
    );
    cout << "Processing FPN Layer 1 ... " << endl;
    fpn_tiled_conv_fpn_1 (lateral_1_output_feature_map,
                fpn_1_layer_weights,
                fpn_1_layer_bias,
                fpn_1_output_feature_map
    );
    cout << "Processing FPN Layer 2 ... " << endl;
    fpn_tiled_conv_fpn_2 (lateral_2_output_feature_map,
                fpn_2_layer_weights,
                fpn_2_layer_bias,
                fpn_2_output_feature_map
    );
    cout << "Processing FPN Layer 3 ... " << endl;
    fpn_tiled_conv_fpn_3 (lateral_3_output_feature_map,
                fpn_3_layer_weights,
                fpn_3_layer_bias,
                fpn_3_output_feature_map
    );
}
