#include "qdtrack_fpn0.h"

void test_top (
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
)
{
 

 fpn_top (
    lateral_3_input_feature_map,
    lateral_3_layer_weights,
    lateral_3_layer_bias,
	lateral_2_input_feature_map,
    lateral_2_layer_weights,
    lateral_2_layer_bias,
	lateral_1_input_feature_map,
    lateral_1_layer_weights,
    lateral_1_layer_bias,
	lateral_0_input_feature_map,
    lateral_0_layer_weights,
    lateral_0_layer_bias,
    fpn_3_layer_weights,
    fpn_3_layer_bias,
    fpn_3_output_feature_map,
    fpn_2_layer_weights,
    fpn_2_layer_bias,
    fpn_2_output_feature_map,
    fpn_1_layer_weights,
    fpn_1_layer_bias,
    fpn_1_output_feature_map,
    fpn_0_layer_weights,
    fpn_0_layer_bias,
    fpn_0_output_feature_map
);


}
