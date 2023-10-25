#include "qdtrack_resnet.h"

void test_top (
    fm_t   resnet_layer0_input_fm[RESNET_LAYER0_CONV1_IN_CH][RESNET_LAYER0_IN_FM_HEIGHT][RESNET_LAYER0_IN_FM_WIDTH],
    wt_t   resnet_layer0_conv1_weights[RESNET_LAYER0_CONV1_OUT_CH][RESNET_LAYER0_CONV1_IN_CH][7][7],
    wt_t   resnet_layer0_bn1_params[3][RESNET_LAYER0_CONV1_OUT_CH],
    fm_t   resnet_layer0_output_fm[RESNET_LAYER0_CONV1_OUT_CH][RESNET_LAYER0_OUT_FM_HEIGHT][RESNET_LAYER0_OUT_FM_WIDTH],

    fm_t   resnet_layer1_input_fm[RESNET_LAYER1_0_CONV1_IN_CH][RESNET_LAYER1_0_FM_HEIGHT][RESNET_LAYER1_0_FM_WIDTH],
    wt_t   resnet_layer1_0_conv1_weights[RESNET_LAYER1_0_CONV1_OUT_CH][RESNET_LAYER1_0_CONV1_IN_CH],
    wt_t   resnet_layer1_0_bn1_params[4][RESNET_LAYER1_0_CONV1_OUT_CH],
    wt_t   resnet_layer1_0_conv2_weights[RESNET_LAYER1_0_CONV2_OUT_CH][RESNET_LAYER1_0_CONV2_IN_CH][3][3],
    wt_t   resnet_layer1_0_bn2_params[4][RESNET_LAYER1_0_CONV2_OUT_CH],
    wt_t   resnet_layer1_0_conv3_weights[RESNET_LAYER1_0_CONV3_OUT_CH][RESNET_LAYER1_0_CONV3_IN_CH],
    wt_t   resnet_layer1_0_bn3_params[4][RESNET_LAYER1_0_CONV3_OUT_CH],
    wt_t   resnet_layer1_0_downsample_0_weights[RESNET_LAYER1_0_DS_OUT_CH][RESNET_LAYER1_0_DS_IN_CH],
    wt_t   resnet_layer1_0_downsample_1_params[4][RESNET_LAYER1_0_DS_OUT_CH],
    wt_t   resnet_layer1_1_conv1_weights[RESNET_LAYER1_CONV1_OUT_CH][RESNET_LAYER1_CONV1_IN_CH],
    wt_t   resnet_layer1_1_bn1_params[4][RESNET_LAYER1_CONV1_OUT_CH],
    wt_t   resnet_layer1_1_conv2_weights[RESNET_LAYER1_CONV2_OUT_CH][RESNET_LAYER1_CONV2_IN_CH][3][3],
    wt_t   resnet_layer1_1_bn2_params[4][RESNET_LAYER1_CONV2_OUT_CH],
    wt_t   resnet_layer1_1_conv3_weights[RESNET_LAYER1_CONV3_OUT_CH][RESNET_LAYER1_CONV3_IN_CH],
    wt_t   resnet_layer1_1_bn3_params[4][RESNET_LAYER1_CONV3_OUT_CH],
    wt_t   resnet_layer1_2_conv1_weights[RESNET_LAYER1_CONV1_OUT_CH][RESNET_LAYER1_CONV1_IN_CH],
    wt_t   resnet_layer1_2_bn1_params[4][RESNET_LAYER1_CONV1_OUT_CH],
    wt_t   resnet_layer1_2_conv2_weights[RESNET_LAYER1_CONV2_OUT_CH][RESNET_LAYER1_CONV2_IN_CH][3][3],
    wt_t   resnet_layer1_2_bn2_params[4][RESNET_LAYER1_CONV2_OUT_CH],
    wt_t   resnet_layer1_2_conv3_weights[RESNET_LAYER1_CONV3_OUT_CH][RESNET_LAYER1_CONV3_IN_CH],
    wt_t   resnet_layer1_2_bn3_params[4][RESNET_LAYER1_CONV3_OUT_CH],
    fm_t   resnet_layer1_output_fm[RESNET_LAYER1_0_DS_OUT_CH][RESNET_LAYER1_0_FM_HEIGHT][RESNET_LAYER1_0_FM_WIDTH],

    fm_t   resnet_layer2_input_fm[RESNET_LAYER2_0_CONV1_IN_CH][RESNET_LAYER2_0_FM_HEIGHT][RESNET_LAYER2_0_FM_WIDTH],
    wt_t   resnet_layer2_0_conv1_weights[RESNET_LAYER2_0_CONV1_OUT_CH][RESNET_LAYER2_0_CONV1_IN_CH],
	wt_t   resnet_layer2_0_bn1_params[3][RESNET_LAYER2_0_CONV1_OUT_CH],
    wt_t   resnet_layer2_0_conv2_weights[RESNET_LAYER2_0_CONV2_OUT_CH][RESNET_LAYER2_0_CONV2_IN_CH][3][3],
	wt_t   resnet_layer2_0_bn2_params[3][RESNET_LAYER2_0_CONV2_OUT_CH],
    wt_t   resnet_layer2_0_conv3_weights[RESNET_LAYER2_0_CONV3_OUT_CH][RESNET_LAYER2_0_CONV3_IN_CH],
	wt_t   resnet_layer2_0_bn3_params[3][RESNET_LAYER2_0_CONV3_OUT_CH],
    wt_t   resnet_layer2_0_downsample_0_weights[RESNET_LAYER2_0_DS_OUT_CH][RESNET_LAYER2_0_DS_IN_CH],
	wt_t   resnet_layer2_0_downsample_1_params[3][RESNET_LAYER2_0_DS_OUT_CH],
    wt_t   resnet_layer2_1_conv1_weights[RESNET_LAYER2_CONV1_OUT_CH][RESNET_LAYER2_CONV1_IN_CH],
	wt_t   resnet_layer2_1_bn1_params[3][RESNET_LAYER2_CONV1_OUT_CH],
    wt_t   resnet_layer2_1_conv2_weights[RESNET_LAYER2_CONV2_OUT_CH][RESNET_LAYER2_CONV2_IN_CH][3][3],
	wt_t   resnet_layer2_1_bn2_params[3][RESNET_LAYER2_CONV2_OUT_CH],
    wt_t   resnet_layer2_1_conv3_weights[RESNET_LAYER2_CONV3_OUT_CH][RESNET_LAYER2_CONV3_IN_CH],
	wt_t   resnet_layer2_1_bn3_params[3][RESNET_LAYER2_CONV3_OUT_CH],
    wt_t   resnet_layer2_2_conv1_weights[RESNET_LAYER2_CONV1_OUT_CH][RESNET_LAYER2_CONV1_IN_CH],
	wt_t   resnet_layer2_2_bn1_params[3][RESNET_LAYER2_CONV1_OUT_CH],
    wt_t   resnet_layer2_2_conv2_weights[RESNET_LAYER2_CONV2_OUT_CH][RESNET_LAYER2_CONV2_IN_CH][3][3],
	wt_t   resnet_layer2_2_bn2_params[3][RESNET_LAYER2_CONV2_OUT_CH],
    wt_t   resnet_layer2_2_conv3_weights[RESNET_LAYER2_CONV3_OUT_CH][RESNET_LAYER2_CONV3_IN_CH],
	wt_t   resnet_layer2_2_bn3_params[3][RESNET_LAYER2_CONV3_OUT_CH],
    wt_t   resnet_layer2_3_conv1_weights[RESNET_LAYER2_CONV1_OUT_CH][RESNET_LAYER2_CONV1_IN_CH],
	wt_t   resnet_layer2_3_bn1_params[3][RESNET_LAYER2_CONV1_OUT_CH],
    wt_t   resnet_layer2_3_conv2_weights[RESNET_LAYER2_CONV2_OUT_CH][RESNET_LAYER2_CONV2_IN_CH][3][3],
	wt_t   resnet_layer2_3_bn2_params[3][RESNET_LAYER2_CONV2_OUT_CH],
    wt_t   resnet_layer2_3_conv3_weights[RESNET_LAYER2_CONV3_OUT_CH][RESNET_LAYER2_CONV3_IN_CH],
	wt_t   resnet_layer2_3_bn3_params[3][RESNET_LAYER2_CONV3_OUT_CH],
    fm_t   resnet_layer2_output_fm[RESNET_LAYER2_0_DS_OUT_CH][RESNET_LAYER2_0_FM_HEIGHT][RESNET_LAYER2_0_FM_WIDTH],

    fm_t   resnet_layer3_input_fm[RESNET_LAYER3_0_CONV1_IN_CH][RESNET_LAYER3_0_FM_HEIGHT][RESNET_LAYER3_0_FM_WIDTH],
    wt_t   resnet_layer3_0_conv1_weights[RESNET_LAYER3_0_CONV1_OUT_CH][RESNET_LAYER3_0_CONV1_IN_CH],
	wt_t   resnet_layer3_0_bn1_params[4][RESNET_LAYER3_0_CONV1_OUT_CH],
    wt_t   resnet_layer3_0_conv2_weights[RESNET_LAYER3_0_CONV2_OUT_CH][RESNET_LAYER3_0_CONV2_IN_CH][3][3],
	wt_t   resnet_layer3_0_bn2_params[4][RESNET_LAYER3_0_CONV2_OUT_CH],
    wt_t   resnet_layer3_0_conv3_weights[RESNET_LAYER3_0_CONV3_OUT_CH][RESNET_LAYER3_0_CONV3_IN_CH],
	wt_t   resnet_layer3_0_bn3_params[4][RESNET_LAYER3_0_CONV3_OUT_CH],
    wt_t   resnet_layer3_0_downsample_0_weights[RESNET_LAYER3_0_DS_OUT_CH][RESNET_LAYER3_0_DS_IN_CH],
	wt_t   resnet_layer3_0_downsample_1_params[4][RESNET_LAYER3_0_DS_OUT_CH],
    wt_t   resnet_layer3_1_conv1_weights[RESNET_LAYER3_CONV1_OUT_CH][RESNET_LAYER3_CONV1_IN_CH],
	wt_t   resnet_layer3_1_bn1_params[4][RESNET_LAYER3_CONV1_OUT_CH],
    wt_t   resnet_layer3_1_conv2_weights[RESNET_LAYER3_CONV2_OUT_CH][RESNET_LAYER3_CONV2_IN_CH][3][3],
	wt_t   resnet_layer3_1_bn2_params[4][RESNET_LAYER3_CONV2_OUT_CH],
    wt_t   resnet_layer3_1_conv3_weights[RESNET_LAYER3_CONV3_OUT_CH][RESNET_LAYER3_CONV3_IN_CH],
	wt_t   resnet_layer3_1_bn3_params[4][RESNET_LAYER3_CONV3_OUT_CH],
    wt_t   resnet_layer3_2_conv1_weights[RESNET_LAYER3_CONV1_OUT_CH][RESNET_LAYER3_CONV1_IN_CH],
	wt_t   resnet_layer3_2_bn1_params[4][RESNET_LAYER3_CONV1_OUT_CH],
    wt_t   resnet_layer3_2_conv2_weights[RESNET_LAYER3_CONV2_OUT_CH][RESNET_LAYER3_CONV2_IN_CH][3][3],
	wt_t   resnet_layer3_2_bn2_params[4][RESNET_LAYER3_CONV2_OUT_CH],
    wt_t   resnet_layer3_2_conv3_weights[RESNET_LAYER3_CONV3_OUT_CH][RESNET_LAYER3_CONV3_IN_CH],
	wt_t   resnet_layer3_2_bn3_params[4][RESNET_LAYER3_CONV3_OUT_CH],
    wt_t   resnet_layer3_3_conv1_weights[RESNET_LAYER3_CONV1_OUT_CH][RESNET_LAYER3_CONV1_IN_CH],
	wt_t   resnet_layer3_3_bn1_params[4][RESNET_LAYER3_CONV1_OUT_CH],
    wt_t   resnet_layer3_3_conv2_weights[RESNET_LAYER3_CONV2_OUT_CH][RESNET_LAYER3_CONV2_IN_CH][3][3],
	wt_t   resnet_layer3_3_bn2_params[4][RESNET_LAYER3_CONV2_OUT_CH],
    wt_t   resnet_layer3_3_conv3_weights[RESNET_LAYER3_CONV3_OUT_CH][RESNET_LAYER3_CONV3_IN_CH],
	wt_t   resnet_layer3_3_bn3_params[4][RESNET_LAYER3_CONV3_OUT_CH],
    wt_t   resnet_layer3_4_conv1_weights[RESNET_LAYER3_CONV1_OUT_CH][RESNET_LAYER3_CONV1_IN_CH],
	wt_t   resnet_layer3_4_bn1_params[4][RESNET_LAYER3_CONV1_OUT_CH],
    wt_t   resnet_layer3_4_conv2_weights[RESNET_LAYER3_CONV2_OUT_CH][RESNET_LAYER3_CONV2_IN_CH][3][3],
	wt_t   resnet_layer3_4_bn2_params[4][RESNET_LAYER3_CONV2_OUT_CH],
    wt_t   resnet_layer3_4_conv3_weights[RESNET_LAYER3_CONV3_OUT_CH][RESNET_LAYER3_CONV3_IN_CH],
	wt_t   resnet_layer3_4_bn3_params[4][RESNET_LAYER3_CONV3_OUT_CH],
    wt_t   resnet_layer3_5_conv1_weights[RESNET_LAYER3_CONV1_OUT_CH][RESNET_LAYER3_CONV1_IN_CH],
	wt_t   resnet_layer3_5_bn1_params[4][RESNET_LAYER3_CONV1_OUT_CH],
    wt_t   resnet_layer3_5_conv2_weights[RESNET_LAYER3_CONV2_OUT_CH][RESNET_LAYER3_CONV2_IN_CH][3][3],
	wt_t   resnet_layer3_5_bn2_params[4][RESNET_LAYER3_CONV2_OUT_CH],
    wt_t   resnet_layer3_5_conv3_weights[RESNET_LAYER3_CONV3_OUT_CH][RESNET_LAYER3_CONV3_IN_CH],
	wt_t   resnet_layer3_5_bn3_params[4][RESNET_LAYER3_CONV3_OUT_CH],
    fm_t   resnet_layer3_output_fm[RESNET_LAYER3_0_DS_OUT_CH][RESNET_LAYER3_0_FM_HEIGHT][RESNET_LAYER3_0_FM_WIDTH],
        
    fm_t   resnet_layer4_input_fm[RESNET_LAYER4_0_CONV1_IN_CH][RESNET_LAYER4_0_FM_HEIGHT][RESNET_LAYER4_0_FM_WIDTH],
    wt_t   resnet_layer4_0_conv1_weights[RESNET_LAYER4_0_CONV1_OUT_CH][RESNET_LAYER4_0_CONV1_IN_CH],
	wt_t   resnet_layer4_0_bn1_params[4][RESNET_LAYER4_0_CONV1_OUT_CH],
    wt_t   resnet_layer4_0_conv2_weights[RESNET_LAYER4_0_CONV2_OUT_CH][RESNET_LAYER4_0_CONV2_IN_CH][3][3],
	wt_t   resnet_layer4_0_bn2_params[4][RESNET_LAYER4_0_CONV2_OUT_CH],
    wt_t   resnet_layer4_0_conv3_weights[RESNET_LAYER4_0_CONV3_OUT_CH][RESNET_LAYER4_0_CONV3_IN_CH],
	wt_t   resnet_layer4_0_bn3_params[4][RESNET_LAYER4_0_CONV3_OUT_CH],
    wt_t   resnet_layer4_0_downsample_0_weights[RESNET_LAYER4_0_DS_OUT_CH][RESNET_LAYER4_0_DS_IN_CH],
	wt_t   resnet_layer4_0_downsample_1_params[4][RESNET_LAYER4_0_DS_OUT_CH],
    wt_t   resnet_layer4_1_conv1_weights[RESNET_LAYER4_CONV1_OUT_CH][RESNET_LAYER4_CONV1_IN_CH],
	wt_t   resnet_layer4_1_bn1_params[4][RESNET_LAYER4_CONV1_OUT_CH],
    wt_t   resnet_layer4_1_conv2_weights[RESNET_LAYER4_CONV2_OUT_CH][RESNET_LAYER4_CONV2_IN_CH][3][3],
	wt_t   resnet_layer4_1_bn2_params[4][RESNET_LAYER4_CONV2_OUT_CH],
    wt_t   resnet_layer4_1_conv3_weights[RESNET_LAYER4_CONV3_OUT_CH][RESNET_LAYER4_CONV3_IN_CH],
	wt_t   resnet_layer4_1_bn3_params[4][RESNET_LAYER4_CONV3_OUT_CH],
    wt_t   resnet_layer4_2_conv1_weights[RESNET_LAYER4_CONV1_OUT_CH][RESNET_LAYER4_CONV1_IN_CH],
	wt_t   resnet_layer4_2_bn1_params[4][RESNET_LAYER4_CONV1_OUT_CH],
    wt_t   resnet_layer4_2_conv2_weights[RESNET_LAYER4_CONV2_OUT_CH][RESNET_LAYER4_CONV2_IN_CH][3][3],
	wt_t   resnet_layer4_2_bn2_params[4][RESNET_LAYER4_CONV2_OUT_CH],
    wt_t   resnet_layer4_2_conv3_weights[RESNET_LAYER4_CONV3_OUT_CH][RESNET_LAYER4_CONV3_IN_CH],
	wt_t   resnet_layer4_2_bn3_params[4][RESNET_LAYER4_CONV3_OUT_CH],
    fm_t   resnet_layer4_output_fm[RESNET_LAYER4_0_DS_OUT_CH][RESNET_LAYER4_0_FM_HEIGHT][RESNET_LAYER4_0_FM_WIDTH]

    // fm_t lateral_3_input_feature_map[LATERAL_CONV_3_ID][LATERAL_CONV_3_IH][LATERAL_CONV_3_IW],
    // wt_t lateral_3_layer_weights[LATERAL_CONV_3_OD][LATERAL_CONV_3_ID][1][1],
    // wt_t lateral_3_layer_bias[LATERAL_CONV_3_OD],
	// fm_t lateral_2_input_feature_map[LATERAL_CONV_2_ID][LATERAL_CONV_2_IH][LATERAL_CONV_2_IW],
    // wt_t lateral_2_layer_weights[LATERAL_CONV_2_OD][LATERAL_CONV_2_ID][1][1],
    // wt_t lateral_2_layer_bias[LATERAL_CONV_2_OD],
	// fm_t lateral_1_input_feature_map[LATERAL_CONV_1_ID][LATERAL_CONV_1_IH][LATERAL_CONV_1_IW],
    // wt_t lateral_1_layer_weights[LATERAL_CONV_1_OD][LATERAL_CONV_1_ID][1][1],
    // wt_t lateral_1_layer_bias[LATERAL_CONV_1_OD],
	// fm_t lateral_0_input_feature_map[LATERAL_CONV_0_ID][LATERAL_CONV_0_IH][LATERAL_CONV_0_IW],
    // wt_t lateral_0_layer_weights[LATERAL_CONV_0_OD][LATERAL_CONV_0_ID][1][1],
    // wt_t lateral_0_layer_bias[LATERAL_CONV_0_OD],
    // wt_t fpn_3_layer_weights[FPN_CONV_3_OD][FPN_CONV_3_ID][3][3],
    // wt_t fpn_3_layer_bias[FPN_CONV_3_OD],
    // fm_t fpn_3_output_feature_map[FPN_CONV_3_OD][FPN_CONV_3_IH][FPN_CONV_3_IW],
    // wt_t fpn_2_layer_weights[FPN_CONV_2_OD][FPN_CONV_2_ID][3][3],
    // wt_t fpn_2_layer_bias[FPN_CONV_2_OD],
    // fm_t fpn_2_output_feature_map[FPN_CONV_2_OD][FPN_CONV_2_IH][FPN_CONV_2_IW],
    // wt_t fpn_1_layer_weights[FPN_CONV_1_OD][FPN_CONV_1_ID][3][3],
    // wt_t fpn_1_layer_bias[FPN_CONV_1_OD],
    // fm_t fpn_1_output_feature_map[FPN_CONV_1_OD][FPN_CONV_1_IH][FPN_CONV_1_IW],
    // wt_t fpn_0_layer_weights[FPN_CONV_0_OD][FPN_CONV_0_ID][3][3],
    // wt_t fpn_0_layer_bias[FPN_CONV_0_OD],
    // fm_t fpn_0_output_feature_map[FPN_CONV_0_OD][FPN_CONV_0_IH][FPN_CONV_0_IW],

    // int rpn_topk_index0[RPN_PRE_NMS_SIZE0],
    // int rpn_topk_index1[RPN_PRE_NMS_SIZE1],
    // int rpn_topk_index2[RPN_PRE_NMS_SIZE2],

    // fm_t rpn_anchor0_reg_fm[RPN_REG_OUT_CH*RPN_INPUT0_IN_FM_HEIGHT*RPN_INPUT0_IN_FM_WIDTH/4][4],
    // fm_t rpn_anchor1_reg_fm[RPN_REG_OUT_CH*RPN_INPUT0_IN_FM_HEIGHT*RPN_INPUT0_IN_FM_WIDTH/4][4],
    // fm_t rpn_anchor2_reg_fm[RPN_REG_OUT_CH*RPN_INPUT0_IN_FM_HEIGHT*RPN_INPUT0_IN_FM_WIDTH/4][4],

    // fm_t rpn_anchor0_cls_fm[RPN_CLS_OUT_CH*RPN_INPUT0_IN_FM_HEIGHT*RPN_INPUT0_IN_FM_WIDTH],
    // fm_t rpn_anchor1_cls_fm[RPN_CLS_OUT_CH*RPN_INPUT1_IN_FM_HEIGHT*RPN_INPUT1_IN_FM_WIDTH],
    // fm_t rpn_anchor2_cls_fm[RPN_CLS_OUT_CH*RPN_INPUT2_IN_FM_HEIGHT*RPN_INPUT2_IN_FM_WIDTH],

    // //Inputs to RPN
    // fm_t rpn_input0_fm[RPN_CONV_IN_CH][RPN_INPUT0_IN_FM_HEIGHT][RPN_INPUT0_IN_FM_WIDTH],
    // fm_t rpn_input1_fm[RPN_CONV_IN_CH][RPN_INPUT1_IN_FM_HEIGHT][RPN_INPUT1_IN_FM_WIDTH],
    // fm_t rpn_input2_fm[RPN_CONV_IN_CH][RPN_INPUT2_IN_FM_HEIGHT][RPN_INPUT2_IN_FM_WIDTH],
    // fm_t rpn_input3_fm[RPN_CONV_IN_CH][RPN_INPUT3_IN_FM_HEIGHT][RPN_INPUT3_IN_FM_WIDTH],
    // fm_t rpn_input4_fm[RPN_CONV_IN_CH][RPN_INPUT4_IN_FM_HEIGHT][RPN_INPUT4_IN_FM_WIDTH],

    // //Weights and Bias for convolutions
    // wt_t rpn_conv_weight[RPN_CONV_OUT_CH][RPN_CONV_IN_CH][3][3],
    // wt_t rpn_conv_bias[RPN_CONV_OUT_CH],
    // wt_t rpn_cls_weight[RPN_CLS_OUT_CH][RPN_CLS_IN_CH][1][1],
    // wt_t rpn_cls_bias[RPN_CLS_OUT_CH],
    // wt_t rpn_reg_weight[RPN_REG_OUT_CH][RPN_REG_IN_CH][1][1],
    // wt_t rpn_reg_bias[RPN_REG_OUT_CH],

    // fm_t rpn_output0_cls_fm[RPN_CLS_OUT_CH][RPN_INPUT0_IN_FM_HEIGHT][RPN_INPUT0_IN_FM_WIDTH],
    // fm_t rpn_output1_cls_fm[RPN_CLS_OUT_CH][RPN_INPUT1_IN_FM_HEIGHT][RPN_INPUT1_IN_FM_WIDTH],
    // fm_t rpn_output2_cls_fm[RPN_CLS_OUT_CH][RPN_INPUT2_IN_FM_HEIGHT][RPN_INPUT2_IN_FM_WIDTH],
    // fm_t rpn_output3_cls_fm[RPN_CLS_OUT_CH][RPN_INPUT3_IN_FM_HEIGHT][RPN_INPUT3_IN_FM_WIDTH],
    // fm_t rpn_output4_cls_fm[RPN_CLS_OUT_CH][RPN_INPUT4_IN_FM_HEIGHT][RPN_INPUT4_IN_FM_WIDTH],


    // fm_t rpn_output0_reg_fm[RPN_REG_OUT_CH][RPN_INPUT0_IN_FM_HEIGHT][RPN_INPUT0_IN_FM_WIDTH],
    // fm_t rpn_output1_reg_fm[RPN_REG_OUT_CH][RPN_INPUT1_IN_FM_HEIGHT][RPN_INPUT1_IN_FM_WIDTH],
    // fm_t rpn_output2_reg_fm[RPN_REG_OUT_CH][RPN_INPUT2_IN_FM_HEIGHT][RPN_INPUT2_IN_FM_WIDTH],
    // fm_t rpn_output3_reg_fm[RPN_REG_OUT_CH][RPN_INPUT3_IN_FM_HEIGHT][RPN_INPUT3_IN_FM_WIDTH],
    // fm_t rpn_output4_reg_fm[RPN_REG_OUT_CH][RPN_INPUT4_IN_FM_HEIGHT][RPN_INPUT4_IN_FM_WIDTH],



    // fm_t rpn_output0_fm[RPN_CONV_IN_CH][RPN_INPUT0_IN_FM_HEIGHT][RPN_INPUT0_IN_FM_WIDTH],
    // fm_t rpn_output1_fm[RPN_CONV_IN_CH][RPN_INPUT1_IN_FM_HEIGHT][RPN_INPUT1_IN_FM_WIDTH],
    // fm_t rpn_output2_fm[RPN_CONV_IN_CH][RPN_INPUT2_IN_FM_HEIGHT][RPN_INPUT2_IN_FM_WIDTH],
    // fm_t rpn_output3_fm[RPN_CONV_IN_CH][RPN_INPUT3_IN_FM_HEIGHT][RPN_INPUT3_IN_FM_WIDTH],
    // fm_t rpn_output4_fm[RPN_CONV_IN_CH][RPN_INPUT4_IN_FM_HEIGHT][RPN_INPUT4_IN_FM_WIDTH],
    


    // wt_t anchor_box0[RPN_ANCHORS0_IN_FM][4],
    // wt_t anchor_box1[RPN_ANCHORS1_IN_FM][4],
    // wt_t anchor_box2[RPN_ANCHORS2_IN_FM][4],
    // wt_t anchor_box3[RPN_ANCHORS3_IN_FM][4],
    // wt_t anchor_box4[RPN_ANCHORS4_IN_FM][4],
    

    // fm_t bboxes[RPN_PRE_NMS_SIZE][4],
    // fm_t dets[1000][5]
)
{
 
  resnet_top_1 (
    resnet_layer0_input_fm,
    resnet_layer0_conv1_weights,
    resnet_layer0_bn1_params,
    resnet_layer0_output_fm,

    resnet_layer1_input_fm,
    resnet_layer1_0_conv1_weights,
    resnet_layer1_0_bn1_params,
    resnet_layer1_0_conv2_weights,
    resnet_layer1_0_bn2_params,
    resnet_layer1_0_conv3_weights,
    resnet_layer1_0_bn3_params,
    resnet_layer1_0_downsample_0_weights,
    resnet_layer1_0_downsample_1_params,
    resnet_layer1_1_conv1_weights,
    resnet_layer1_1_bn1_params,
    resnet_layer1_1_conv2_weights,
    resnet_layer1_1_bn2_params,
    resnet_layer1_1_conv3_weights,
    resnet_layer1_1_bn3_params,
    resnet_layer1_2_conv1_weights,
    resnet_layer1_2_bn1_params,
    resnet_layer1_2_conv2_weights,
    resnet_layer1_2_bn2_params,
    resnet_layer1_2_conv3_weights,
    resnet_layer1_2_bn3_params,
    resnet_layer1_output_fm,

    resnet_layer2_input_fm,
    resnet_layer2_0_conv1_weights,
	resnet_layer2_0_bn1_params,
    resnet_layer2_0_conv2_weights,
	resnet_layer2_0_bn2_params,
    resnet_layer2_0_conv3_weights,
	resnet_layer2_0_bn3_params,
    resnet_layer2_0_downsample_0_weights,
	resnet_layer2_0_downsample_1_params,
    resnet_layer2_1_conv1_weights,
	resnet_layer2_1_bn1_params,
    resnet_layer2_1_conv2_weights,
	resnet_layer2_1_bn2_params,
    resnet_layer2_1_conv3_weights,
	resnet_layer2_1_bn3_params,
    resnet_layer2_2_conv1_weights,
	resnet_layer2_2_bn1_params,
    resnet_layer2_2_conv2_weights,
	resnet_layer2_2_bn2_params,
    resnet_layer2_2_conv3_weights,
	resnet_layer2_2_bn3_params,
    resnet_layer2_3_conv1_weights,
	resnet_layer2_3_bn1_params,
    resnet_layer2_3_conv2_weights,
	resnet_layer2_3_bn2_params,
    resnet_layer2_3_conv3_weights,
	resnet_layer2_3_bn3_params,
    resnet_layer2_output_fm
);

  resnet_top_2 (
    resnet_layer2_output_fm,
    resnet_layer3_input_fm,
    resnet_layer3_0_conv1_weights,
	resnet_layer3_0_bn1_params,
    resnet_layer3_0_conv2_weights,
	resnet_layer3_0_bn2_params,
    resnet_layer3_0_conv3_weights,
	resnet_layer3_0_bn3_params,
    resnet_layer3_0_downsample_0_weights,
	resnet_layer3_0_downsample_1_params,
    resnet_layer3_1_conv1_weights,
	resnet_layer3_1_bn1_params,
    resnet_layer3_1_conv2_weights,
	resnet_layer3_1_bn2_params,
    resnet_layer3_1_conv3_weights,
	resnet_layer3_1_bn3_params,
    resnet_layer3_2_conv1_weights,
	resnet_layer3_2_bn1_params,
    resnet_layer3_2_conv2_weights,
	resnet_layer3_2_bn2_params,
    resnet_layer3_2_conv3_weights,
	resnet_layer3_2_bn3_params,
    resnet_layer3_3_conv1_weights,
	resnet_layer3_3_bn1_params,
    resnet_layer3_3_conv2_weights,
	resnet_layer3_3_bn2_params,
    resnet_layer3_3_conv3_weights,
	resnet_layer3_3_bn3_params,
    resnet_layer3_4_conv1_weights,
	resnet_layer3_4_bn1_params,
    resnet_layer3_4_conv2_weights,
	resnet_layer3_4_bn2_params,
    resnet_layer3_4_conv3_weights,
	resnet_layer3_4_bn3_params,
    resnet_layer3_5_conv1_weights,
	resnet_layer3_5_bn1_params,
    resnet_layer3_5_conv2_weights,
	resnet_layer3_5_bn2_params,
    resnet_layer3_5_conv3_weights,
	resnet_layer3_5_bn3_params,
    resnet_layer3_output_fm,
    
    resnet_layer4_input_fm,
    resnet_layer4_0_conv1_weights,
	resnet_layer4_0_bn1_params,
    resnet_layer4_0_conv2_weights,
	resnet_layer4_0_bn2_params,
    resnet_layer4_0_conv3_weights,
	resnet_layer4_0_bn3_params,
    resnet_layer4_0_downsample_0_weights,
	resnet_layer4_0_downsample_1_params,
    resnet_layer4_1_conv1_weights,
	resnet_layer4_1_bn1_params,
    resnet_layer4_1_conv2_weights,
	resnet_layer4_1_bn2_params,
    resnet_layer4_1_conv3_weights,
	resnet_layer4_1_bn3_params,
    resnet_layer4_2_conv1_weights,
	resnet_layer4_2_bn1_params,
    resnet_layer4_2_conv2_weights,
	resnet_layer4_2_bn2_params,
    resnet_layer4_2_conv3_weights,
	resnet_layer4_2_bn3_params,
    resnet_layer4_output_fm
);

//  fpn_top (
//     lateral_3_input_feature_map,
//     lateral_3_layer_weights,
//     lateral_3_layer_bias,
// 	lateral_2_input_feature_map,
//     lateral_2_layer_weights,
//     lateral_2_layer_bias,
// 	lateral_1_input_feature_map,
//     lateral_1_layer_weights,
//     lateral_1_layer_bias,
// 	lateral_0_input_feature_map,
//     lateral_0_layer_weights,
//     lateral_0_layer_bias,
//     fpn_3_layer_weights,
//     fpn_3_layer_bias,
//     fpn_3_output_feature_map,
//     fpn_2_layer_weights,
//     fpn_2_layer_bias,
//     fpn_2_output_feature_map,
//     fpn_1_layer_weights,
//     fpn_1_layer_bias,
//     fpn_1_output_feature_map,
//     fpn_0_layer_weights,
//     fpn_0_layer_bias,
//     fpn_0_output_feature_map
// );

//   rpn_top (
//     //Inputs to RPN
//     rpn_input0_fm,
//     rpn_input1_fm,
//     rpn_input2_fm,

//     //Weights and Bias for convolutions
//     rpn_conv_weight,
//     rpn_conv_bias,
//     rpn_cls_weight,
//     rpn_cls_bias,
//     rpn_reg_weight,
//     rpn_reg_bias,

//     rpn_output0_cls_fm,
//     rpn_output1_cls_fm,
//     rpn_output2_cls_fm,

//     rpn_output0_reg_fm,
//     rpn_output1_reg_fm,
//     rpn_output2_reg_fm,

//     rpn_output0_fm,
//     rpn_output1_fm,
//     rpn_output2_fm
// );

//  rpn_top2 (
//     rpn_topk_index0,
//     rpn_topk_index1,
//     rpn_topk_index2,

//     rpn_anchor0_reg_fm,
//     rpn_anchor1_reg_fm,
//     rpn_anchor2_reg_fm,

//     rpn_anchor0_cls_fm,
//     rpn_anchor1_cls_fm,
//     rpn_anchor2_cls_fm,

//     //Inputs to RPN
//     rpn_input3_fm,
//     rpn_input4_fm,

//     //Weights and Bias for convolutions
//     rpn_conv_weight,
//     rpn_conv_bias,
//     rpn_cls_weight,
//     rpn_cls_bias,
//     rpn_reg_weight,
//     rpn_reg_bias,

//     rpn_output3_cls_fm,
//     rpn_output4_cls_fm,

//     rpn_output3_reg_fm,
//     rpn_output4_reg_fm,

//     rpn_output3_fm,
//     rpn_output4_fm,
    
//     anchor_box0,
//     anchor_box1,
//     anchor_box2,
//     anchor_box3,
//     anchor_box4,

//     bboxes,
//     dets
// );

}
