#include <iostream>
#include <fstream>
#include <cmath>
#include "hls_stream.h"
#include "qdtrack_fpn0.h"

using namespace std;

//--------------------------------------------------------------------------
// FPN
//--------------------------------------------------------------------------
//CONV_3x3
float   conv_0_input_feature_map[FPN_CONV_0_ID][FPN_CONV_0_IH][FPN_CONV_0_IW];
float   conv_1_input_feature_map[FPN_CONV_1_ID][FPN_CONV_1_IH][FPN_CONV_1_IW];
float   conv_2_input_feature_map[FPN_CONV_2_ID][FPN_CONV_2_IH][FPN_CONV_2_IW];
float   conv_3_input_feature_map[FPN_CONV_3_ID][FPN_CONV_3_IH][FPN_CONV_3_IW];
float   conv_0_weights[FPN_CONV_0_OD][FPN_CONV_0_ID][3][3];
float   conv_1_weights[FPN_CONV_1_OD][FPN_CONV_1_ID][3][3];
float   conv_2_weights[FPN_CONV_2_OD][FPN_CONV_2_ID][3][3];
float   conv_3_weights[FPN_CONV_3_OD][FPN_CONV_3_ID][3][3];
float   conv_0_bias[FPN_CONV_0_OD];
float   conv_1_bias[FPN_CONV_1_OD];
float   conv_2_bias[FPN_CONV_2_OD];
float   conv_3_bias[FPN_CONV_3_OD];
float   golden_conv_0_golden_output[FPN_CONV_0_OD][FPN_CONV_0_OH][FPN_CONV_0_OW];
float   golden_conv_1_golden_output[FPN_CONV_1_OD][FPN_CONV_1_OH][FPN_CONV_1_OW];
float   golden_conv_2_golden_output[FPN_CONV_2_OD][FPN_CONV_2_OH][FPN_CONV_2_OW];
float   golden_conv_3_golden_output[FPN_CONV_3_OD][FPN_CONV_3_OH][FPN_CONV_3_OW];

fm_t	fixp_conv_0_input_feature_map[FPN_CONV_0_ID][FPN_CONV_0_IH][FPN_CONV_0_IW];
fm_t	fixp_conv_1_input_feature_map[FPN_CONV_1_ID][FPN_CONV_1_IH][FPN_CONV_1_IW];
fm_t	fixp_conv_2_input_feature_map[FPN_CONV_2_ID][FPN_CONV_2_IH][FPN_CONV_2_IW];
fm_t	fixp_conv_3_input_feature_map[FPN_CONV_3_ID][FPN_CONV_3_IH][FPN_CONV_3_IW];
wt_t	fixp_conv_0_weights[FPN_CONV_0_OD][FPN_CONV_0_ID][3][3];
wt_t	fixp_conv_1_weights[FPN_CONV_1_OD][FPN_CONV_1_ID][3][3];
wt_t	fixp_conv_2_weights[FPN_CONV_2_OD][FPN_CONV_2_ID][3][3];
wt_t	fixp_conv_3_weights[FPN_CONV_3_OD][FPN_CONV_3_ID][3][3];
wt_t	fixp_conv_0_bias[FPN_CONV_0_OD];
wt_t	fixp_conv_1_bias[FPN_CONV_1_OD];
wt_t	fixp_conv_2_bias[FPN_CONV_2_OD];
wt_t	fixp_conv_3_bias[FPN_CONV_3_OD];

//CONV_1x1
float   lateral_conv_0_input_feature_map[LATERAL_CONV_0_ID][LATERAL_CONV_0_IH][LATERAL_CONV_0_IW];
float   lateral_conv_1_input_feature_map[LATERAL_CONV_1_ID][LATERAL_CONV_1_IH][LATERAL_CONV_1_IW];
float   lateral_conv_2_input_feature_map[LATERAL_CONV_2_ID][LATERAL_CONV_2_IH][LATERAL_CONV_2_IW];
float   lateral_conv_3_input_feature_map[LATERAL_CONV_3_ID][LATERAL_CONV_3_IH][LATERAL_CONV_3_IW];
float   lateral_conv_0_weights[LATERAL_CONV_0_OD][LATERAL_CONV_0_ID][1][1];
float   lateral_conv_1_weights[LATERAL_CONV_1_OD][LATERAL_CONV_1_ID][1][1];
float   lateral_conv_2_weights[LATERAL_CONV_2_OD][LATERAL_CONV_2_ID][1][1];
float   lateral_conv_3_weights[LATERAL_CONV_3_OD][LATERAL_CONV_3_ID][1][1];
float   lateral_conv_0_bias[LATERAL_CONV_0_OD];
float   lateral_conv_1_bias[LATERAL_CONV_1_OD];
float   lateral_conv_2_bias[LATERAL_CONV_2_OD];
float   lateral_conv_3_bias[LATERAL_CONV_3_OD];

fm_t	fixp_lateral_conv_0_input_feature_map[LATERAL_CONV_0_ID][LATERAL_CONV_0_IH][LATERAL_CONV_0_IW];
fm_t	fixp_lateral_conv_1_input_feature_map[LATERAL_CONV_1_ID][LATERAL_CONV_1_IH][LATERAL_CONV_1_IW];
fm_t	fixp_lateral_conv_2_input_feature_map[LATERAL_CONV_2_ID][LATERAL_CONV_2_IH][LATERAL_CONV_2_IW];
fm_t	fixp_lateral_conv_3_input_feature_map[LATERAL_CONV_3_ID][LATERAL_CONV_3_IH][LATERAL_CONV_3_IW];
wt_t	fixp_lateral_conv_0_weights[LATERAL_CONV_0_OD][LATERAL_CONV_0_ID][1][1];
wt_t	fixp_lateral_conv_1_weights[LATERAL_CONV_1_OD][LATERAL_CONV_1_ID][1][1];
wt_t	fixp_lateral_conv_2_weights[LATERAL_CONV_2_OD][LATERAL_CONV_2_ID][1][1];
wt_t	fixp_lateral_conv_3_weights[LATERAL_CONV_3_OD][LATERAL_CONV_3_ID][1][1];
wt_t	fixp_lateral_conv_0_bias[LATERAL_CONV_0_OD];
wt_t	fixp_lateral_conv_1_bias[LATERAL_CONV_1_OD];
wt_t	fixp_lateral_conv_2_bias[LATERAL_CONV_2_OD];
wt_t	fixp_lateral_conv_3_bias[LATERAL_CONV_3_OD];

//--------------------------------------------------------------------------
// Computed outputs
//--------------------------------------------------------------------------
//CONV_3x3
fm_t    fixp_conv_0_output_feature_map[FPN_CONV_0_OD][FPN_CONV_0_OH][FPN_CONV_0_OW] = {0};
fm_t    fixp_conv_1_output_feature_map[FPN_CONV_1_OD][FPN_CONV_1_OH][FPN_CONV_1_OW] = {0};
fm_t    fixp_conv_2_output_feature_map[FPN_CONV_2_OD][FPN_CONV_2_OH][FPN_CONV_2_OW] = {0};
fm_t    fixp_conv_3_output_feature_map[FPN_CONV_3_OD][FPN_CONV_3_OH][FPN_CONV_3_OW] = {0};

//--------------------------------------------------------------------------
// Read the reference files into test bench arrays
//--------------------------------------------------------------------------
void read_fpn_params()
{
    //CONV_3x3
    //--------------------FPN 0------------------------------//
    // Weights
    std::ifstream ifs_conv_0_wt("/usr/scratch/rsamanta9/bin/fpn_neck/module_neck_fpn_convs_0_conv_weight.bin", ios::in | ios::binary);
    ifs_conv_0_wt.read((char*)(***conv_0_weights), (FPN_CONV_0_OD)*(FPN_CONV_0_ID)*3*3*sizeof(float));
    ifs_conv_0_wt.close();

    // Bias
    std::ifstream ifs_conv_0_bias("/usr/scratch/rsamanta9/bin/fpn_neck/module_neck_fpn_convs_0_conv_bias.bin", ios::in | ios::binary);
    ifs_conv_0_bias.read((char*)(conv_0_bias), (FPN_CONV_0_OD)*sizeof(float));
    ifs_conv_0_bias.close();

    // Golden Output
    std::ifstream ifs_conv_0_output_golden("/usr/scratch/rsamanta9/bin/fpn_neck/module_neck_fpn_convs_0_conv.bin", ios::in | ios::binary);
    ifs_conv_0_output_golden.read((char*)(**golden_conv_0_golden_output), (FPN_CONV_0_OD)*(FPN_CONV_0_OW)*(FPN_CONV_0_OH)*sizeof(float));    
    ifs_conv_0_output_golden.close();

    //--------------------FPN 1------------------------------//
    // Weights
    std::ifstream ifs_conv_1_wt("/usr/scratch/rsamanta9/bin/fpn_neck/module_neck_fpn_convs_1_conv_weight.bin", ios::in | ios::binary);
    ifs_conv_1_wt.read((char*)(***conv_1_weights), (FPN_CONV_1_OD)*(FPN_CONV_1_ID)*3*3*sizeof(float));
    ifs_conv_1_wt.close();

    // Bias
    std::ifstream ifs_conv_1_bias("/usr/scratch/rsamanta9/bin/fpn_neck/module_neck_fpn_convs_1_conv_bias.bin", ios::in | ios::binary);
    ifs_conv_1_bias.read((char*)(conv_1_bias), (FPN_CONV_1_OD)*sizeof(float));
    ifs_conv_1_bias.close();

    // Golden Output
    std::ifstream ifs_conv_1_output_golden("/usr/scratch/rsamanta9/bin/fpn_neck/module_neck_fpn_convs_1_conv.bin", ios::in | ios::binary);
    ifs_conv_1_output_golden.read((char*)(**golden_conv_1_golden_output), (FPN_CONV_1_OD)*(FPN_CONV_1_OW)*(FPN_CONV_1_OH)*sizeof(float));    
    ifs_conv_1_output_golden.close();

    //--------------------FPN 2------------------------------//
    // Weights
    std::ifstream ifs_conv_2_wt("/usr/scratch/rsamanta9/bin/fpn_neck/module_neck_fpn_convs_2_conv_weight.bin", ios::in | ios::binary);
    ifs_conv_2_wt.read((char*)(***conv_2_weights), (FPN_CONV_2_OD)*(FPN_CONV_2_ID)*3*3*sizeof(float));
    ifs_conv_2_wt.close();

    // Bias
    std::ifstream ifs_conv_2_bias("/usr/scratch/rsamanta9/bin/fpn_neck/module_neck_fpn_convs_2_conv_bias.bin", ios::in | ios::binary);
    ifs_conv_2_bias.read((char*)(conv_2_bias), (FPN_CONV_2_OD)*sizeof(float));
    ifs_conv_2_bias.close();

    // Golden Output
    std::ifstream ifs_conv_2_output_golden("/usr/scratch/rsamanta9/bin/fpn_neck/module_neck_fpn_convs_2_conv.bin", ios::in | ios::binary);
    ifs_conv_2_output_golden.read((char*)(**golden_conv_2_golden_output), (FPN_CONV_2_OD)*(FPN_CONV_2_OW)*(FPN_CONV_2_OH)*sizeof(float));    
    ifs_conv_2_output_golden.close();

    //--------------------FPN 3------------------------------//
    // Weights
    std::ifstream ifs_conv_3_wt("/usr/scratch/rsamanta9/bin/fpn_neck/module_neck_fpn_convs_3_conv_weight.bin", ios::in | ios::binary);
    ifs_conv_3_wt.read((char*)(***conv_3_weights), (FPN_CONV_3_OD)*(FPN_CONV_3_ID)*3*3*sizeof(float));
    ifs_conv_3_wt.close();

    // Bias
    std::ifstream ifs_conv_3_bias("/usr/scratch/rsamanta9/bin/fpn_neck/module_neck_fpn_convs_3_conv_bias.bin", ios::in | ios::binary);
    ifs_conv_3_bias.read((char*)(conv_3_bias), (FPN_CONV_3_OD)*sizeof(float));
    ifs_conv_3_bias.close();

    // Golden Output
    std::ifstream ifs_conv_3_output_golden("/usr/scratch/rsamanta9/bin/fpn_neck/module_neck_fpn_convs_3_conv.bin", ios::in | ios::binary);
    ifs_conv_3_output_golden.read((char*)(**golden_conv_3_golden_output), (FPN_CONV_3_OD)*(FPN_CONV_3_OW)*(FPN_CONV_3_OH)*sizeof(float));    
    ifs_conv_3_output_golden.close();

    //CONV_1x1
    //--------------Lateral 0-----------------------//
    // Input Feature Map
    std::ifstream ifs_lateral_conv_0_input_fm("/usr/scratch/rsamanta9/bin/fpn_neck/module_neck_lateral_convs_0_conv_input.bin", ios::in | ios::binary);
    ifs_lateral_conv_0_input_fm.read((char*)(**lateral_conv_0_input_feature_map), (LATERAL_CONV_0_ID)*(LATERAL_CONV_0_IH)*(LATERAL_CONV_0_IW)*sizeof(float));
    ifs_lateral_conv_0_input_fm.close();

    // Weights
    std::ifstream ifs_lateral_conv_0_wt("/usr/scratch/rsamanta9/bin/fpn_neck/module_neck_lateral_convs_0_conv_weight.bin", ios::in | ios::binary);
    ifs_lateral_conv_0_wt.read((char*)(***lateral_conv_0_weights), LATERAL_CONV_0_OD*LATERAL_CONV_0_ID*sizeof(float));
    ifs_lateral_conv_0_wt.close();

    // Bias
    std::ifstream ifs_lateral_conv_0_bias("/usr/scratch/rsamanta9/bin/fpn_neck/module_neck_lateral_convs_0_conv_bias.bin", ios::in | ios::binary);
    ifs_lateral_conv_0_bias.read((char*)(lateral_conv_0_bias), LATERAL_CONV_0_OD*sizeof(float));
    ifs_lateral_conv_0_bias.close();

    //--------------Lateral 1-----------------------//
    // Input Feature Map
    std::ifstream ifs_lateral_conv_1_input_fm("/usr/scratch/rsamanta9/bin/fpn_neck/module_neck_lateral_convs_1_conv_input.bin", ios::in | ios::binary);
    ifs_lateral_conv_1_input_fm.read((char*)(**lateral_conv_1_input_feature_map), (LATERAL_CONV_1_ID)*(LATERAL_CONV_1_IH)*(LATERAL_CONV_1_IW)*sizeof(float));
    ifs_lateral_conv_1_input_fm.close();

    // Weights
    std::ifstream ifs_lateral_conv_1_wt("/usr/scratch/rsamanta9/bin/fpn_neck/module_neck_lateral_convs_1_conv_weight.bin", ios::in | ios::binary);
    ifs_lateral_conv_1_wt.read((char*)(***lateral_conv_1_weights), LATERAL_CONV_1_OD*LATERAL_CONV_1_ID*sizeof(float));
    ifs_lateral_conv_1_wt.close();

    // Bias
    std::ifstream ifs_lateral_conv_1_bias("/usr/scratch/rsamanta9/bin/fpn_neck/module_neck_lateral_convs_1_conv_bias.bin", ios::in | ios::binary);
    ifs_lateral_conv_1_bias.read((char*)(lateral_conv_1_bias), LATERAL_CONV_1_OD*sizeof(float));
    ifs_lateral_conv_1_bias.close();

    //--------------Lateral 2-----------------------//
    // Input Feature Map
    std::ifstream ifs_lateral_conv_2_input_fm("/usr/scratch/rsamanta9/bin/fpn_neck/module_neck_lateral_convs_2_conv_input.bin", ios::in | ios::binary);
    ifs_lateral_conv_2_input_fm.read((char*)(**lateral_conv_2_input_feature_map), (LATERAL_CONV_2_ID)*(LATERAL_CONV_2_IH)*(LATERAL_CONV_2_IW)*sizeof(float));
    ifs_lateral_conv_2_input_fm.close();

    // Weights
    std::ifstream ifs_lateral_conv_2_wt("/usr/scratch/rsamanta9/bin/fpn_neck/module_neck_lateral_convs_2_conv_weight.bin", ios::in | ios::binary);
    ifs_lateral_conv_2_wt.read((char*)(***lateral_conv_2_weights), LATERAL_CONV_2_OD*LATERAL_CONV_2_ID*sizeof(float));
    ifs_lateral_conv_2_wt.close();

    // Bias
    std::ifstream ifs_lateral_conv_2_bias("/usr/scratch/rsamanta9/bin/fpn_neck/module_neck_lateral_convs_2_conv_bias.bin", ios::in | ios::binary);
    ifs_lateral_conv_2_bias.read((char*)(lateral_conv_2_bias), LATERAL_CONV_2_OD*sizeof(float));
    ifs_lateral_conv_2_bias.close();

    //--------------Lateral 3-----------------------//
    // Input Feature Map
    std::ifstream ifs_lateral_conv_3_input_fm("/usr/scratch/rsamanta9/bin/fpn_neck/module_neck_lateral_convs_3_conv_input.bin", ios::in | ios::binary);
    ifs_lateral_conv_3_input_fm.read((char*)(**lateral_conv_3_input_feature_map), (LATERAL_CONV_3_ID)*(LATERAL_CONV_3_IH)*(LATERAL_CONV_3_IW)*sizeof(float));
    ifs_lateral_conv_3_input_fm.close();

    // Weights
    std::ifstream ifs_lateral_conv_3_wt("/usr/scratch/rsamanta9/bin/fpn_neck/module_neck_lateral_convs_3_conv_weight.bin", ios::in | ios::binary);
    ifs_lateral_conv_3_wt.read((char*)(***lateral_conv_3_weights), LATERAL_CONV_3_OD*LATERAL_CONV_3_ID*sizeof(float));
    ifs_lateral_conv_3_wt.close();

    // Bias
    std::ifstream ifs_lateral_conv_3_bias("/usr/scratch/rsamanta9/bin/fpn_neck/module_neck_lateral_convs_3_conv_bias.bin", ios::in | ios::binary);
    ifs_lateral_conv_3_bias.read((char*)(lateral_conv_3_bias), LATERAL_CONV_3_OD*sizeof(float));
    ifs_lateral_conv_3_bias.close();
}

//--------------------------------------------------------------------------
// Convert the data types of every array element for specified 
// configuration.
//--------------------------------------------------------------------------
void fpn_convert_type()
{
    // Input Feature Map
    cout << "Convert Input Feature Map ... " << endl;
    for(int c = 0; c < LATERAL_CONV_0_ID; c++)
        for(int i = 0; i < LATERAL_CONV_0_IH; i++)
            for(int j = 0; j < LATERAL_CONV_0_IW; j++)
		fixp_lateral_conv_0_input_feature_map[c][i][j] = (fm_t) lateral_conv_0_input_feature_map[c][i][j];
    for(int c = 0; c < LATERAL_CONV_1_ID; c++)
        for(int i = 0; i < LATERAL_CONV_1_IH; i++)
            for(int j = 0; j < LATERAL_CONV_1_IW; j++)
		fixp_lateral_conv_1_input_feature_map[c][i][j] = (fm_t) lateral_conv_1_input_feature_map[c][i][j];
    for(int c = 0; c < LATERAL_CONV_2_ID; c++)
        for(int i = 0; i < LATERAL_CONV_2_IH; i++)
            for(int j = 0; j < LATERAL_CONV_2_IW; j++)
		fixp_lateral_conv_2_input_feature_map[c][i][j] = (fm_t) lateral_conv_2_input_feature_map[c][i][j];
    for(int c = 0; c < LATERAL_CONV_3_ID; c++)
        for(int i = 0; i < LATERAL_CONV_3_IH; i++)
            for(int j = 0; j < LATERAL_CONV_3_IW; j++)
		fixp_lateral_conv_3_input_feature_map[c][i][j] = (fm_t) lateral_conv_3_input_feature_map[c][i][j];
    for(int c = 0; c < FPN_CONV_0_ID; c++)
        for(int i = 0; i < FPN_CONV_0_IH; i++)
            for(int j = 0; j < FPN_CONV_0_IW; j++)
		fixp_conv_0_input_feature_map[c][i][j] = (fm_t) conv_0_input_feature_map[c][i][j];
    for(int c = 0; c < FPN_CONV_1_ID; c++)
        for(int i = 0; i < FPN_CONV_1_IH; i++)
            for(int j = 0; j < FPN_CONV_1_IW; j++)
		fixp_conv_1_input_feature_map[c][i][j] = (fm_t) conv_1_input_feature_map[c][i][j];
    for(int c = 0; c < FPN_CONV_2_ID; c++)
        for(int i = 0; i < FPN_CONV_2_IH; i++)
            for(int j = 0; j < FPN_CONV_2_IW; j++)
		fixp_conv_2_input_feature_map[c][i][j] = (fm_t) conv_2_input_feature_map[c][i][j];
    for(int c = 0; c < FPN_CONV_3_ID; c++)
        for(int i = 0; i < FPN_CONV_3_IH; i++)
            for(int j = 0; j < FPN_CONV_3_IW; j++)
		fixp_conv_3_input_feature_map[c][i][j] = (fm_t) conv_3_input_feature_map[c][i][j];

    // Weights
    cout << "Convert Weights ... " << endl;
    for(int f = 0; f < LATERAL_CONV_0_OD; f++)
        for(int c = 0; c < LATERAL_CONV_0_ID; c++)
		fixp_lateral_conv_0_weights[f][c][0][0] = (wt_t) lateral_conv_0_weights[f][c][0][0];
    for(int f = 0; f < LATERAL_CONV_1_OD; f++)
        for(int c = 0; c < LATERAL_CONV_1_ID; c++)
		fixp_lateral_conv_1_weights[f][c][0][0] = (wt_t) lateral_conv_1_weights[f][c][0][0];
    for(int f = 0; f < LATERAL_CONV_2_OD; f++)
        for(int c = 0; c < LATERAL_CONV_2_ID; c++)
		fixp_lateral_conv_2_weights[f][c][0][0] = (wt_t) lateral_conv_2_weights[f][c][0][0];
    for(int f = 0; f < LATERAL_CONV_3_OD; f++)
        for(int c = 0; c < LATERAL_CONV_3_ID; c++)
		fixp_lateral_conv_3_weights[f][c][0][0] = (wt_t) lateral_conv_3_weights[f][c][0][0];
    for(int f = 0; f < FPN_CONV_0_OD; f++)
        for(int c = 0; c < FPN_CONV_0_ID; c++)
            for(int m = 0; m < 3; m++)
                for(int n =0; n < 3; n++)
			fixp_conv_0_weights[f][c][m][n] = (wt_t) conv_0_weights[f][c][m][n];
    for(int f = 0; f < FPN_CONV_1_OD; f++)
        for(int c = 0; c < FPN_CONV_1_ID; c++)
            for(int m = 0; m < 3; m++)
                for(int n =0; n < 3; n++)
			fixp_conv_1_weights[f][c][m][n] = (wt_t) conv_1_weights[f][c][m][n];
    for(int f = 0; f < FPN_CONV_2_OD; f++)
        for(int c = 0; c < FPN_CONV_2_ID; c++)
            for(int m = 0; m < 3; m++)
                for(int n =0; n < 3; n++)
			fixp_conv_2_weights[f][c][m][n] = (wt_t) conv_2_weights[f][c][m][n];
    for(int f = 0; f < FPN_CONV_3_OD; f++)
        for(int c = 0; c < FPN_CONV_3_ID; c++)
            for(int m = 0; m < 3; m++)
                for(int n =0; n < 3; n++)
			fixp_conv_3_weights[f][c][m][n] = (wt_t) conv_3_weights[f][c][m][n];

    // Bias
    cout << "Convert Biases ... " << endl;
    for(int f = 0; f < LATERAL_CONV_0_OD; f++)
	fixp_lateral_conv_0_bias[f] = (wt_t) lateral_conv_0_bias[f];
    for(int f = 0; f < LATERAL_CONV_1_OD; f++)
	fixp_lateral_conv_1_bias[f] = (wt_t) lateral_conv_1_bias[f];
    for(int f = 0; f < LATERAL_CONV_2_OD; f++)
	fixp_lateral_conv_2_bias[f] = (wt_t) lateral_conv_2_bias[f];
    for(int f = 0; f < LATERAL_CONV_3_OD; f++)
	fixp_lateral_conv_3_bias[f] = (wt_t) lateral_conv_3_bias[f];
    for(int f = 0; f < FPN_CONV_0_OD; f++)
	fixp_conv_0_bias[f] = (wt_t) conv_0_bias[f];
    for(int f = 0; f < FPN_CONV_1_OD; f++)
	fixp_conv_1_bias[f] = (wt_t) conv_1_bias[f];
    for(int f = 0; f < FPN_CONV_2_OD; f++)
	fixp_conv_2_bias[f] = (wt_t) conv_2_bias[f];
    for(int f = 0; f < FPN_CONV_3_OD; f++)
	fixp_conv_3_bias[f] = (wt_t) conv_3_bias[f];
}



int main ()
{
    long double mse = 0.0;
    
#ifdef TEST_COMPLETE_MODEL // {
    //----------------------------------------------------------------------
    // ResNet50 Top-level wrapper 
    //----------------------------------------------------------------------
    test_top( 
              fixp_lateral_conv_3_input_feature_map,
                fixp_lateral_conv_3_weights,
                fixp_lateral_conv_3_bias,
    		fixp_lateral_conv_2_input_feature_map,
                fixp_lateral_conv_2_weights,
                fixp_lateral_conv_2_bias,
    		fixp_lateral_conv_1_input_feature_map,
                fixp_lateral_conv_1_weights,
                fixp_lateral_conv_1_bias,
    		fixp_lateral_conv_0_input_feature_map,
                fixp_lateral_conv_0_weights,
                fixp_lateral_conv_0_bias,
                fixp_conv_3_weights,
                fixp_conv_3_bias,
                fixp_conv_3_output_feature_map,
                fixp_conv_2_weights,
                fixp_conv_2_bias,
                fixp_conv_2_output_feature_map,
                fixp_conv_1_weights,
                fixp_conv_1_bias,
                fixp_conv_1_output_feature_map,
                fixp_conv_0_weights,
                fixp_conv_0_bias,
                fixp_conv_0_output_feature_map

    );

        //-----------------VERIFICATION--------------------//
    std::cout << "Compute MSE LAYER_0 .... " << std::endl;
    for(int f = 0; f < FPN_CONV_0_OD; f++)
    {
        for(int h = 0; h < FPN_CONV_0_OH; h++)
        {
            for(int w = 0; w < FPN_CONV_0_OW; w++)
            {
                mse += std::pow((golden_conv_0_golden_output[f][h][w] - (float) fixp_conv_0_output_feature_map[f][h][w]), 2);
            }
        }
        //std::cout << "Golden Output: " << golden_conv_0_golden_output[f][0][0] << std::endl;
        //std::cout << "Actual Output: " << (float) fixp_conv_0_output_feature_map[f][0][0] << std::endl;
        //std::cout << std::endl;
    }
    
    mse = mse / (FPN_CONV_0_OD * FPN_CONV_0_OH * FPN_CONV_0_OW);
    std::cout << "FPN_CONVS_0 Output MSE:  " << mse << std::endl;
    std::cout << "FPN_CONVS_0 Processing Complete!" << std::endl << std::endl;

    mse = 0.0;    
    std::cout << "Compute MSE LAYER_1 .... " << std::endl;
    for(int f = 0; f < FPN_CONV_1_OD; f++)
    {
        for(int h = 0; h < FPN_CONV_1_OH; h++)
        {
            for(int w = 0; w < FPN_CONV_1_OW; w++)
            {
                mse += std::pow((golden_conv_1_golden_output[f][h][w] - (float) fixp_conv_1_output_feature_map[f][h][w]), 2);
            }
        }
        //std::cout << "Golden Output: " << golden_conv_1_golden_output[f][0][0] << std::endl;
        //std::cout << "Actual Output: " << (float) fixp_conv_1_output_feature_map[f][0][0] << std::endl;
        //std::cout << std::endl;
    }
    
    mse = mse / (FPN_CONV_1_OD * FPN_CONV_1_OH * FPN_CONV_1_OW);
    std::cout << "FPN_CONVS_1 Output MSE:  " << mse << std::endl;
    std::cout << "FPN_CONVS_1 Processing Complete!" << std::endl << std::endl;
    
    mse = 0.0;    
    std::cout << "Compute MSE LAYER_2 .... " << std::endl;
    for(int f = 0; f < FPN_CONV_2_OD; f++)
    {
        for(int h = 0; h < FPN_CONV_2_OH; h++)
        {
            for(int w = 0; w < FPN_CONV_2_OW; w++)
            {
                mse += std::pow((golden_conv_2_golden_output[f][h][w] - (float) fixp_conv_2_output_feature_map[f][h][w]), 2);
            }
        }
        //std::cout << "Golden Output: " << golden_conv_2_golden_output[f][0][0] << std::endl;
        //std::cout << "Actual Output: " << (float) fixp_conv_2_output_feature_map[f][0][0] << std::endl;
        //std::cout << std::endl;
    }
    
    mse = mse / (FPN_CONV_2_OD * FPN_CONV_2_OH * FPN_CONV_2_OW);
    std::cout << "FPN_CONVS_2 Output MSE:  " << mse << std::endl;
    std::cout << "FPN_CONVS_2 Processing Complete!" << std::endl << std::endl;
    
    mse = 0.0;    
    std::cout << "Compute MSE LAYER_3 .... " << std::endl;
    for(int f = 0; f < FPN_CONV_3_OD; f++)
    {
        for(int h = 0; h < FPN_CONV_3_OH; h++)
        {
            for(int w = 0; w < FPN_CONV_3_OW; w++)
            {
                mse += std::pow((golden_conv_3_golden_output[f][h][w] - (float) fixp_conv_3_output_feature_map[f][h][w]), 2);
            }
        }
        //std::cout << "Golden Output: " << golden_conv_3_golden_output[f][0][0] << std::endl;
        //std::cout << "Actual Output: " << (float) fixp_conv_3_output_feature_map[f][0][0] << std::endl;
        //std::cout << std::endl;
    }
    
    mse = mse / (FPN_CONV_3_OD * FPN_CONV_3_OH * FPN_CONV_3_OW);
    std::cout << "FPN_CONVS_3 Output MSE:  " << mse << std::endl;
    std::cout << "FPN_CONVS_3 Processing Complete!" << std::endl << std::endl;

#endif // }

    return 0;
}
