#include "pose.h"

namespace {
#define DIM 18
#define EDIM 18
#define MEDIM 18
typedef void (*Hfun)(double *, double *, double *);
const static double MAHA_THRESH_4 = 7.814727903251177;
const static double MAHA_THRESH_10 = 7.814727903251177;
const static double MAHA_THRESH_13 = 7.814727903251177;
const static double MAHA_THRESH_14 = 7.814727903251177;

/******************************************************************************
 *                      Code generated with SymPy 1.14.0                      *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                         This file is part of 'ekf'                         *
 ******************************************************************************/
void err_fun(double *nom_x, double *delta_x, double *out_1610068831843926615) {
   out_1610068831843926615[0] = delta_x[0] + nom_x[0];
   out_1610068831843926615[1] = delta_x[1] + nom_x[1];
   out_1610068831843926615[2] = delta_x[2] + nom_x[2];
   out_1610068831843926615[3] = delta_x[3] + nom_x[3];
   out_1610068831843926615[4] = delta_x[4] + nom_x[4];
   out_1610068831843926615[5] = delta_x[5] + nom_x[5];
   out_1610068831843926615[6] = delta_x[6] + nom_x[6];
   out_1610068831843926615[7] = delta_x[7] + nom_x[7];
   out_1610068831843926615[8] = delta_x[8] + nom_x[8];
   out_1610068831843926615[9] = delta_x[9] + nom_x[9];
   out_1610068831843926615[10] = delta_x[10] + nom_x[10];
   out_1610068831843926615[11] = delta_x[11] + nom_x[11];
   out_1610068831843926615[12] = delta_x[12] + nom_x[12];
   out_1610068831843926615[13] = delta_x[13] + nom_x[13];
   out_1610068831843926615[14] = delta_x[14] + nom_x[14];
   out_1610068831843926615[15] = delta_x[15] + nom_x[15];
   out_1610068831843926615[16] = delta_x[16] + nom_x[16];
   out_1610068831843926615[17] = delta_x[17] + nom_x[17];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_9009726282604965886) {
   out_9009726282604965886[0] = -nom_x[0] + true_x[0];
   out_9009726282604965886[1] = -nom_x[1] + true_x[1];
   out_9009726282604965886[2] = -nom_x[2] + true_x[2];
   out_9009726282604965886[3] = -nom_x[3] + true_x[3];
   out_9009726282604965886[4] = -nom_x[4] + true_x[4];
   out_9009726282604965886[5] = -nom_x[5] + true_x[5];
   out_9009726282604965886[6] = -nom_x[6] + true_x[6];
   out_9009726282604965886[7] = -nom_x[7] + true_x[7];
   out_9009726282604965886[8] = -nom_x[8] + true_x[8];
   out_9009726282604965886[9] = -nom_x[9] + true_x[9];
   out_9009726282604965886[10] = -nom_x[10] + true_x[10];
   out_9009726282604965886[11] = -nom_x[11] + true_x[11];
   out_9009726282604965886[12] = -nom_x[12] + true_x[12];
   out_9009726282604965886[13] = -nom_x[13] + true_x[13];
   out_9009726282604965886[14] = -nom_x[14] + true_x[14];
   out_9009726282604965886[15] = -nom_x[15] + true_x[15];
   out_9009726282604965886[16] = -nom_x[16] + true_x[16];
   out_9009726282604965886[17] = -nom_x[17] + true_x[17];
}
void H_mod_fun(double *state, double *out_7227948086580598908) {
   out_7227948086580598908[0] = 1.0;
   out_7227948086580598908[1] = 0.0;
   out_7227948086580598908[2] = 0.0;
   out_7227948086580598908[3] = 0.0;
   out_7227948086580598908[4] = 0.0;
   out_7227948086580598908[5] = 0.0;
   out_7227948086580598908[6] = 0.0;
   out_7227948086580598908[7] = 0.0;
   out_7227948086580598908[8] = 0.0;
   out_7227948086580598908[9] = 0.0;
   out_7227948086580598908[10] = 0.0;
   out_7227948086580598908[11] = 0.0;
   out_7227948086580598908[12] = 0.0;
   out_7227948086580598908[13] = 0.0;
   out_7227948086580598908[14] = 0.0;
   out_7227948086580598908[15] = 0.0;
   out_7227948086580598908[16] = 0.0;
   out_7227948086580598908[17] = 0.0;
   out_7227948086580598908[18] = 0.0;
   out_7227948086580598908[19] = 1.0;
   out_7227948086580598908[20] = 0.0;
   out_7227948086580598908[21] = 0.0;
   out_7227948086580598908[22] = 0.0;
   out_7227948086580598908[23] = 0.0;
   out_7227948086580598908[24] = 0.0;
   out_7227948086580598908[25] = 0.0;
   out_7227948086580598908[26] = 0.0;
   out_7227948086580598908[27] = 0.0;
   out_7227948086580598908[28] = 0.0;
   out_7227948086580598908[29] = 0.0;
   out_7227948086580598908[30] = 0.0;
   out_7227948086580598908[31] = 0.0;
   out_7227948086580598908[32] = 0.0;
   out_7227948086580598908[33] = 0.0;
   out_7227948086580598908[34] = 0.0;
   out_7227948086580598908[35] = 0.0;
   out_7227948086580598908[36] = 0.0;
   out_7227948086580598908[37] = 0.0;
   out_7227948086580598908[38] = 1.0;
   out_7227948086580598908[39] = 0.0;
   out_7227948086580598908[40] = 0.0;
   out_7227948086580598908[41] = 0.0;
   out_7227948086580598908[42] = 0.0;
   out_7227948086580598908[43] = 0.0;
   out_7227948086580598908[44] = 0.0;
   out_7227948086580598908[45] = 0.0;
   out_7227948086580598908[46] = 0.0;
   out_7227948086580598908[47] = 0.0;
   out_7227948086580598908[48] = 0.0;
   out_7227948086580598908[49] = 0.0;
   out_7227948086580598908[50] = 0.0;
   out_7227948086580598908[51] = 0.0;
   out_7227948086580598908[52] = 0.0;
   out_7227948086580598908[53] = 0.0;
   out_7227948086580598908[54] = 0.0;
   out_7227948086580598908[55] = 0.0;
   out_7227948086580598908[56] = 0.0;
   out_7227948086580598908[57] = 1.0;
   out_7227948086580598908[58] = 0.0;
   out_7227948086580598908[59] = 0.0;
   out_7227948086580598908[60] = 0.0;
   out_7227948086580598908[61] = 0.0;
   out_7227948086580598908[62] = 0.0;
   out_7227948086580598908[63] = 0.0;
   out_7227948086580598908[64] = 0.0;
   out_7227948086580598908[65] = 0.0;
   out_7227948086580598908[66] = 0.0;
   out_7227948086580598908[67] = 0.0;
   out_7227948086580598908[68] = 0.0;
   out_7227948086580598908[69] = 0.0;
   out_7227948086580598908[70] = 0.0;
   out_7227948086580598908[71] = 0.0;
   out_7227948086580598908[72] = 0.0;
   out_7227948086580598908[73] = 0.0;
   out_7227948086580598908[74] = 0.0;
   out_7227948086580598908[75] = 0.0;
   out_7227948086580598908[76] = 1.0;
   out_7227948086580598908[77] = 0.0;
   out_7227948086580598908[78] = 0.0;
   out_7227948086580598908[79] = 0.0;
   out_7227948086580598908[80] = 0.0;
   out_7227948086580598908[81] = 0.0;
   out_7227948086580598908[82] = 0.0;
   out_7227948086580598908[83] = 0.0;
   out_7227948086580598908[84] = 0.0;
   out_7227948086580598908[85] = 0.0;
   out_7227948086580598908[86] = 0.0;
   out_7227948086580598908[87] = 0.0;
   out_7227948086580598908[88] = 0.0;
   out_7227948086580598908[89] = 0.0;
   out_7227948086580598908[90] = 0.0;
   out_7227948086580598908[91] = 0.0;
   out_7227948086580598908[92] = 0.0;
   out_7227948086580598908[93] = 0.0;
   out_7227948086580598908[94] = 0.0;
   out_7227948086580598908[95] = 1.0;
   out_7227948086580598908[96] = 0.0;
   out_7227948086580598908[97] = 0.0;
   out_7227948086580598908[98] = 0.0;
   out_7227948086580598908[99] = 0.0;
   out_7227948086580598908[100] = 0.0;
   out_7227948086580598908[101] = 0.0;
   out_7227948086580598908[102] = 0.0;
   out_7227948086580598908[103] = 0.0;
   out_7227948086580598908[104] = 0.0;
   out_7227948086580598908[105] = 0.0;
   out_7227948086580598908[106] = 0.0;
   out_7227948086580598908[107] = 0.0;
   out_7227948086580598908[108] = 0.0;
   out_7227948086580598908[109] = 0.0;
   out_7227948086580598908[110] = 0.0;
   out_7227948086580598908[111] = 0.0;
   out_7227948086580598908[112] = 0.0;
   out_7227948086580598908[113] = 0.0;
   out_7227948086580598908[114] = 1.0;
   out_7227948086580598908[115] = 0.0;
   out_7227948086580598908[116] = 0.0;
   out_7227948086580598908[117] = 0.0;
   out_7227948086580598908[118] = 0.0;
   out_7227948086580598908[119] = 0.0;
   out_7227948086580598908[120] = 0.0;
   out_7227948086580598908[121] = 0.0;
   out_7227948086580598908[122] = 0.0;
   out_7227948086580598908[123] = 0.0;
   out_7227948086580598908[124] = 0.0;
   out_7227948086580598908[125] = 0.0;
   out_7227948086580598908[126] = 0.0;
   out_7227948086580598908[127] = 0.0;
   out_7227948086580598908[128] = 0.0;
   out_7227948086580598908[129] = 0.0;
   out_7227948086580598908[130] = 0.0;
   out_7227948086580598908[131] = 0.0;
   out_7227948086580598908[132] = 0.0;
   out_7227948086580598908[133] = 1.0;
   out_7227948086580598908[134] = 0.0;
   out_7227948086580598908[135] = 0.0;
   out_7227948086580598908[136] = 0.0;
   out_7227948086580598908[137] = 0.0;
   out_7227948086580598908[138] = 0.0;
   out_7227948086580598908[139] = 0.0;
   out_7227948086580598908[140] = 0.0;
   out_7227948086580598908[141] = 0.0;
   out_7227948086580598908[142] = 0.0;
   out_7227948086580598908[143] = 0.0;
   out_7227948086580598908[144] = 0.0;
   out_7227948086580598908[145] = 0.0;
   out_7227948086580598908[146] = 0.0;
   out_7227948086580598908[147] = 0.0;
   out_7227948086580598908[148] = 0.0;
   out_7227948086580598908[149] = 0.0;
   out_7227948086580598908[150] = 0.0;
   out_7227948086580598908[151] = 0.0;
   out_7227948086580598908[152] = 1.0;
   out_7227948086580598908[153] = 0.0;
   out_7227948086580598908[154] = 0.0;
   out_7227948086580598908[155] = 0.0;
   out_7227948086580598908[156] = 0.0;
   out_7227948086580598908[157] = 0.0;
   out_7227948086580598908[158] = 0.0;
   out_7227948086580598908[159] = 0.0;
   out_7227948086580598908[160] = 0.0;
   out_7227948086580598908[161] = 0.0;
   out_7227948086580598908[162] = 0.0;
   out_7227948086580598908[163] = 0.0;
   out_7227948086580598908[164] = 0.0;
   out_7227948086580598908[165] = 0.0;
   out_7227948086580598908[166] = 0.0;
   out_7227948086580598908[167] = 0.0;
   out_7227948086580598908[168] = 0.0;
   out_7227948086580598908[169] = 0.0;
   out_7227948086580598908[170] = 0.0;
   out_7227948086580598908[171] = 1.0;
   out_7227948086580598908[172] = 0.0;
   out_7227948086580598908[173] = 0.0;
   out_7227948086580598908[174] = 0.0;
   out_7227948086580598908[175] = 0.0;
   out_7227948086580598908[176] = 0.0;
   out_7227948086580598908[177] = 0.0;
   out_7227948086580598908[178] = 0.0;
   out_7227948086580598908[179] = 0.0;
   out_7227948086580598908[180] = 0.0;
   out_7227948086580598908[181] = 0.0;
   out_7227948086580598908[182] = 0.0;
   out_7227948086580598908[183] = 0.0;
   out_7227948086580598908[184] = 0.0;
   out_7227948086580598908[185] = 0.0;
   out_7227948086580598908[186] = 0.0;
   out_7227948086580598908[187] = 0.0;
   out_7227948086580598908[188] = 0.0;
   out_7227948086580598908[189] = 0.0;
   out_7227948086580598908[190] = 1.0;
   out_7227948086580598908[191] = 0.0;
   out_7227948086580598908[192] = 0.0;
   out_7227948086580598908[193] = 0.0;
   out_7227948086580598908[194] = 0.0;
   out_7227948086580598908[195] = 0.0;
   out_7227948086580598908[196] = 0.0;
   out_7227948086580598908[197] = 0.0;
   out_7227948086580598908[198] = 0.0;
   out_7227948086580598908[199] = 0.0;
   out_7227948086580598908[200] = 0.0;
   out_7227948086580598908[201] = 0.0;
   out_7227948086580598908[202] = 0.0;
   out_7227948086580598908[203] = 0.0;
   out_7227948086580598908[204] = 0.0;
   out_7227948086580598908[205] = 0.0;
   out_7227948086580598908[206] = 0.0;
   out_7227948086580598908[207] = 0.0;
   out_7227948086580598908[208] = 0.0;
   out_7227948086580598908[209] = 1.0;
   out_7227948086580598908[210] = 0.0;
   out_7227948086580598908[211] = 0.0;
   out_7227948086580598908[212] = 0.0;
   out_7227948086580598908[213] = 0.0;
   out_7227948086580598908[214] = 0.0;
   out_7227948086580598908[215] = 0.0;
   out_7227948086580598908[216] = 0.0;
   out_7227948086580598908[217] = 0.0;
   out_7227948086580598908[218] = 0.0;
   out_7227948086580598908[219] = 0.0;
   out_7227948086580598908[220] = 0.0;
   out_7227948086580598908[221] = 0.0;
   out_7227948086580598908[222] = 0.0;
   out_7227948086580598908[223] = 0.0;
   out_7227948086580598908[224] = 0.0;
   out_7227948086580598908[225] = 0.0;
   out_7227948086580598908[226] = 0.0;
   out_7227948086580598908[227] = 0.0;
   out_7227948086580598908[228] = 1.0;
   out_7227948086580598908[229] = 0.0;
   out_7227948086580598908[230] = 0.0;
   out_7227948086580598908[231] = 0.0;
   out_7227948086580598908[232] = 0.0;
   out_7227948086580598908[233] = 0.0;
   out_7227948086580598908[234] = 0.0;
   out_7227948086580598908[235] = 0.0;
   out_7227948086580598908[236] = 0.0;
   out_7227948086580598908[237] = 0.0;
   out_7227948086580598908[238] = 0.0;
   out_7227948086580598908[239] = 0.0;
   out_7227948086580598908[240] = 0.0;
   out_7227948086580598908[241] = 0.0;
   out_7227948086580598908[242] = 0.0;
   out_7227948086580598908[243] = 0.0;
   out_7227948086580598908[244] = 0.0;
   out_7227948086580598908[245] = 0.0;
   out_7227948086580598908[246] = 0.0;
   out_7227948086580598908[247] = 1.0;
   out_7227948086580598908[248] = 0.0;
   out_7227948086580598908[249] = 0.0;
   out_7227948086580598908[250] = 0.0;
   out_7227948086580598908[251] = 0.0;
   out_7227948086580598908[252] = 0.0;
   out_7227948086580598908[253] = 0.0;
   out_7227948086580598908[254] = 0.0;
   out_7227948086580598908[255] = 0.0;
   out_7227948086580598908[256] = 0.0;
   out_7227948086580598908[257] = 0.0;
   out_7227948086580598908[258] = 0.0;
   out_7227948086580598908[259] = 0.0;
   out_7227948086580598908[260] = 0.0;
   out_7227948086580598908[261] = 0.0;
   out_7227948086580598908[262] = 0.0;
   out_7227948086580598908[263] = 0.0;
   out_7227948086580598908[264] = 0.0;
   out_7227948086580598908[265] = 0.0;
   out_7227948086580598908[266] = 1.0;
   out_7227948086580598908[267] = 0.0;
   out_7227948086580598908[268] = 0.0;
   out_7227948086580598908[269] = 0.0;
   out_7227948086580598908[270] = 0.0;
   out_7227948086580598908[271] = 0.0;
   out_7227948086580598908[272] = 0.0;
   out_7227948086580598908[273] = 0.0;
   out_7227948086580598908[274] = 0.0;
   out_7227948086580598908[275] = 0.0;
   out_7227948086580598908[276] = 0.0;
   out_7227948086580598908[277] = 0.0;
   out_7227948086580598908[278] = 0.0;
   out_7227948086580598908[279] = 0.0;
   out_7227948086580598908[280] = 0.0;
   out_7227948086580598908[281] = 0.0;
   out_7227948086580598908[282] = 0.0;
   out_7227948086580598908[283] = 0.0;
   out_7227948086580598908[284] = 0.0;
   out_7227948086580598908[285] = 1.0;
   out_7227948086580598908[286] = 0.0;
   out_7227948086580598908[287] = 0.0;
   out_7227948086580598908[288] = 0.0;
   out_7227948086580598908[289] = 0.0;
   out_7227948086580598908[290] = 0.0;
   out_7227948086580598908[291] = 0.0;
   out_7227948086580598908[292] = 0.0;
   out_7227948086580598908[293] = 0.0;
   out_7227948086580598908[294] = 0.0;
   out_7227948086580598908[295] = 0.0;
   out_7227948086580598908[296] = 0.0;
   out_7227948086580598908[297] = 0.0;
   out_7227948086580598908[298] = 0.0;
   out_7227948086580598908[299] = 0.0;
   out_7227948086580598908[300] = 0.0;
   out_7227948086580598908[301] = 0.0;
   out_7227948086580598908[302] = 0.0;
   out_7227948086580598908[303] = 0.0;
   out_7227948086580598908[304] = 1.0;
   out_7227948086580598908[305] = 0.0;
   out_7227948086580598908[306] = 0.0;
   out_7227948086580598908[307] = 0.0;
   out_7227948086580598908[308] = 0.0;
   out_7227948086580598908[309] = 0.0;
   out_7227948086580598908[310] = 0.0;
   out_7227948086580598908[311] = 0.0;
   out_7227948086580598908[312] = 0.0;
   out_7227948086580598908[313] = 0.0;
   out_7227948086580598908[314] = 0.0;
   out_7227948086580598908[315] = 0.0;
   out_7227948086580598908[316] = 0.0;
   out_7227948086580598908[317] = 0.0;
   out_7227948086580598908[318] = 0.0;
   out_7227948086580598908[319] = 0.0;
   out_7227948086580598908[320] = 0.0;
   out_7227948086580598908[321] = 0.0;
   out_7227948086580598908[322] = 0.0;
   out_7227948086580598908[323] = 1.0;
}
void f_fun(double *state, double dt, double *out_4483597399355948729) {
   out_4483597399355948729[0] = atan2((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), -(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]));
   out_4483597399355948729[1] = asin(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]));
   out_4483597399355948729[2] = atan2(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), -(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]));
   out_4483597399355948729[3] = dt*state[12] + state[3];
   out_4483597399355948729[4] = dt*state[13] + state[4];
   out_4483597399355948729[5] = dt*state[14] + state[5];
   out_4483597399355948729[6] = state[6];
   out_4483597399355948729[7] = state[7];
   out_4483597399355948729[8] = state[8];
   out_4483597399355948729[9] = state[9];
   out_4483597399355948729[10] = state[10];
   out_4483597399355948729[11] = state[11];
   out_4483597399355948729[12] = state[12];
   out_4483597399355948729[13] = state[13];
   out_4483597399355948729[14] = state[14];
   out_4483597399355948729[15] = state[15];
   out_4483597399355948729[16] = state[16];
   out_4483597399355948729[17] = state[17];
}
void F_fun(double *state, double dt, double *out_832782703925887625) {
   out_832782703925887625[0] = ((-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*cos(state[0])*cos(state[1]) - sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*cos(state[0])*cos(state[1]) - sin(dt*state[6])*sin(state[0])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_832782703925887625[1] = ((-sin(dt*state[6])*sin(dt*state[8]) - sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*cos(state[1]) - (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*sin(state[1]) - sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(state[0]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*sin(state[1]) + (-sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) + sin(dt*state[8])*cos(dt*state[6]))*cos(state[1]) - sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(state[0]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_832782703925887625[2] = 0;
   out_832782703925887625[3] = 0;
   out_832782703925887625[4] = 0;
   out_832782703925887625[5] = 0;
   out_832782703925887625[6] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(dt*cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) - dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_832782703925887625[7] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*sin(dt*state[7])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[6])*sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) - dt*sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[7])*cos(dt*state[6])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[8])*sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]) - dt*sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_832782703925887625[8] = ((dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((dt*sin(dt*state[6])*sin(dt*state[8]) + dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_832782703925887625[9] = 0;
   out_832782703925887625[10] = 0;
   out_832782703925887625[11] = 0;
   out_832782703925887625[12] = 0;
   out_832782703925887625[13] = 0;
   out_832782703925887625[14] = 0;
   out_832782703925887625[15] = 0;
   out_832782703925887625[16] = 0;
   out_832782703925887625[17] = 0;
   out_832782703925887625[18] = (-sin(dt*state[7])*sin(state[0])*cos(state[1]) - sin(dt*state[8])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_832782703925887625[19] = (-sin(dt*state[7])*sin(state[1])*cos(state[0]) + sin(dt*state[8])*sin(state[0])*sin(state[1])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_832782703925887625[20] = 0;
   out_832782703925887625[21] = 0;
   out_832782703925887625[22] = 0;
   out_832782703925887625[23] = 0;
   out_832782703925887625[24] = 0;
   out_832782703925887625[25] = (dt*sin(dt*state[7])*sin(dt*state[8])*sin(state[0])*cos(state[1]) - dt*sin(dt*state[7])*sin(state[1])*cos(dt*state[8]) + dt*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_832782703925887625[26] = (-dt*sin(dt*state[8])*sin(state[1])*cos(dt*state[7]) - dt*sin(state[0])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_832782703925887625[27] = 0;
   out_832782703925887625[28] = 0;
   out_832782703925887625[29] = 0;
   out_832782703925887625[30] = 0;
   out_832782703925887625[31] = 0;
   out_832782703925887625[32] = 0;
   out_832782703925887625[33] = 0;
   out_832782703925887625[34] = 0;
   out_832782703925887625[35] = 0;
   out_832782703925887625[36] = ((sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_832782703925887625[37] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-sin(dt*state[7])*sin(state[2])*cos(state[0])*cos(state[1]) + sin(dt*state[8])*sin(state[0])*sin(state[2])*cos(dt*state[7])*cos(state[1]) - sin(state[1])*sin(state[2])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(-sin(dt*state[7])*cos(state[0])*cos(state[1])*cos(state[2]) + sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1])*cos(state[2]) - sin(state[1])*cos(dt*state[7])*cos(dt*state[8])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_832782703925887625[38] = ((-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (-sin(state[0])*sin(state[1])*sin(state[2]) - cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_832782703925887625[39] = 0;
   out_832782703925887625[40] = 0;
   out_832782703925887625[41] = 0;
   out_832782703925887625[42] = 0;
   out_832782703925887625[43] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(dt*(sin(state[0])*cos(state[2]) - sin(state[1])*sin(state[2])*cos(state[0]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*sin(state[2])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(dt*(-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_832782703925887625[44] = (dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*sin(state[2])*cos(dt*state[7])*cos(state[1]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + (dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[7])*cos(state[1])*cos(state[2]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_832782703925887625[45] = 0;
   out_832782703925887625[46] = 0;
   out_832782703925887625[47] = 0;
   out_832782703925887625[48] = 0;
   out_832782703925887625[49] = 0;
   out_832782703925887625[50] = 0;
   out_832782703925887625[51] = 0;
   out_832782703925887625[52] = 0;
   out_832782703925887625[53] = 0;
   out_832782703925887625[54] = 0;
   out_832782703925887625[55] = 0;
   out_832782703925887625[56] = 0;
   out_832782703925887625[57] = 1;
   out_832782703925887625[58] = 0;
   out_832782703925887625[59] = 0;
   out_832782703925887625[60] = 0;
   out_832782703925887625[61] = 0;
   out_832782703925887625[62] = 0;
   out_832782703925887625[63] = 0;
   out_832782703925887625[64] = 0;
   out_832782703925887625[65] = 0;
   out_832782703925887625[66] = dt;
   out_832782703925887625[67] = 0;
   out_832782703925887625[68] = 0;
   out_832782703925887625[69] = 0;
   out_832782703925887625[70] = 0;
   out_832782703925887625[71] = 0;
   out_832782703925887625[72] = 0;
   out_832782703925887625[73] = 0;
   out_832782703925887625[74] = 0;
   out_832782703925887625[75] = 0;
   out_832782703925887625[76] = 1;
   out_832782703925887625[77] = 0;
   out_832782703925887625[78] = 0;
   out_832782703925887625[79] = 0;
   out_832782703925887625[80] = 0;
   out_832782703925887625[81] = 0;
   out_832782703925887625[82] = 0;
   out_832782703925887625[83] = 0;
   out_832782703925887625[84] = 0;
   out_832782703925887625[85] = dt;
   out_832782703925887625[86] = 0;
   out_832782703925887625[87] = 0;
   out_832782703925887625[88] = 0;
   out_832782703925887625[89] = 0;
   out_832782703925887625[90] = 0;
   out_832782703925887625[91] = 0;
   out_832782703925887625[92] = 0;
   out_832782703925887625[93] = 0;
   out_832782703925887625[94] = 0;
   out_832782703925887625[95] = 1;
   out_832782703925887625[96] = 0;
   out_832782703925887625[97] = 0;
   out_832782703925887625[98] = 0;
   out_832782703925887625[99] = 0;
   out_832782703925887625[100] = 0;
   out_832782703925887625[101] = 0;
   out_832782703925887625[102] = 0;
   out_832782703925887625[103] = 0;
   out_832782703925887625[104] = dt;
   out_832782703925887625[105] = 0;
   out_832782703925887625[106] = 0;
   out_832782703925887625[107] = 0;
   out_832782703925887625[108] = 0;
   out_832782703925887625[109] = 0;
   out_832782703925887625[110] = 0;
   out_832782703925887625[111] = 0;
   out_832782703925887625[112] = 0;
   out_832782703925887625[113] = 0;
   out_832782703925887625[114] = 1;
   out_832782703925887625[115] = 0;
   out_832782703925887625[116] = 0;
   out_832782703925887625[117] = 0;
   out_832782703925887625[118] = 0;
   out_832782703925887625[119] = 0;
   out_832782703925887625[120] = 0;
   out_832782703925887625[121] = 0;
   out_832782703925887625[122] = 0;
   out_832782703925887625[123] = 0;
   out_832782703925887625[124] = 0;
   out_832782703925887625[125] = 0;
   out_832782703925887625[126] = 0;
   out_832782703925887625[127] = 0;
   out_832782703925887625[128] = 0;
   out_832782703925887625[129] = 0;
   out_832782703925887625[130] = 0;
   out_832782703925887625[131] = 0;
   out_832782703925887625[132] = 0;
   out_832782703925887625[133] = 1;
   out_832782703925887625[134] = 0;
   out_832782703925887625[135] = 0;
   out_832782703925887625[136] = 0;
   out_832782703925887625[137] = 0;
   out_832782703925887625[138] = 0;
   out_832782703925887625[139] = 0;
   out_832782703925887625[140] = 0;
   out_832782703925887625[141] = 0;
   out_832782703925887625[142] = 0;
   out_832782703925887625[143] = 0;
   out_832782703925887625[144] = 0;
   out_832782703925887625[145] = 0;
   out_832782703925887625[146] = 0;
   out_832782703925887625[147] = 0;
   out_832782703925887625[148] = 0;
   out_832782703925887625[149] = 0;
   out_832782703925887625[150] = 0;
   out_832782703925887625[151] = 0;
   out_832782703925887625[152] = 1;
   out_832782703925887625[153] = 0;
   out_832782703925887625[154] = 0;
   out_832782703925887625[155] = 0;
   out_832782703925887625[156] = 0;
   out_832782703925887625[157] = 0;
   out_832782703925887625[158] = 0;
   out_832782703925887625[159] = 0;
   out_832782703925887625[160] = 0;
   out_832782703925887625[161] = 0;
   out_832782703925887625[162] = 0;
   out_832782703925887625[163] = 0;
   out_832782703925887625[164] = 0;
   out_832782703925887625[165] = 0;
   out_832782703925887625[166] = 0;
   out_832782703925887625[167] = 0;
   out_832782703925887625[168] = 0;
   out_832782703925887625[169] = 0;
   out_832782703925887625[170] = 0;
   out_832782703925887625[171] = 1;
   out_832782703925887625[172] = 0;
   out_832782703925887625[173] = 0;
   out_832782703925887625[174] = 0;
   out_832782703925887625[175] = 0;
   out_832782703925887625[176] = 0;
   out_832782703925887625[177] = 0;
   out_832782703925887625[178] = 0;
   out_832782703925887625[179] = 0;
   out_832782703925887625[180] = 0;
   out_832782703925887625[181] = 0;
   out_832782703925887625[182] = 0;
   out_832782703925887625[183] = 0;
   out_832782703925887625[184] = 0;
   out_832782703925887625[185] = 0;
   out_832782703925887625[186] = 0;
   out_832782703925887625[187] = 0;
   out_832782703925887625[188] = 0;
   out_832782703925887625[189] = 0;
   out_832782703925887625[190] = 1;
   out_832782703925887625[191] = 0;
   out_832782703925887625[192] = 0;
   out_832782703925887625[193] = 0;
   out_832782703925887625[194] = 0;
   out_832782703925887625[195] = 0;
   out_832782703925887625[196] = 0;
   out_832782703925887625[197] = 0;
   out_832782703925887625[198] = 0;
   out_832782703925887625[199] = 0;
   out_832782703925887625[200] = 0;
   out_832782703925887625[201] = 0;
   out_832782703925887625[202] = 0;
   out_832782703925887625[203] = 0;
   out_832782703925887625[204] = 0;
   out_832782703925887625[205] = 0;
   out_832782703925887625[206] = 0;
   out_832782703925887625[207] = 0;
   out_832782703925887625[208] = 0;
   out_832782703925887625[209] = 1;
   out_832782703925887625[210] = 0;
   out_832782703925887625[211] = 0;
   out_832782703925887625[212] = 0;
   out_832782703925887625[213] = 0;
   out_832782703925887625[214] = 0;
   out_832782703925887625[215] = 0;
   out_832782703925887625[216] = 0;
   out_832782703925887625[217] = 0;
   out_832782703925887625[218] = 0;
   out_832782703925887625[219] = 0;
   out_832782703925887625[220] = 0;
   out_832782703925887625[221] = 0;
   out_832782703925887625[222] = 0;
   out_832782703925887625[223] = 0;
   out_832782703925887625[224] = 0;
   out_832782703925887625[225] = 0;
   out_832782703925887625[226] = 0;
   out_832782703925887625[227] = 0;
   out_832782703925887625[228] = 1;
   out_832782703925887625[229] = 0;
   out_832782703925887625[230] = 0;
   out_832782703925887625[231] = 0;
   out_832782703925887625[232] = 0;
   out_832782703925887625[233] = 0;
   out_832782703925887625[234] = 0;
   out_832782703925887625[235] = 0;
   out_832782703925887625[236] = 0;
   out_832782703925887625[237] = 0;
   out_832782703925887625[238] = 0;
   out_832782703925887625[239] = 0;
   out_832782703925887625[240] = 0;
   out_832782703925887625[241] = 0;
   out_832782703925887625[242] = 0;
   out_832782703925887625[243] = 0;
   out_832782703925887625[244] = 0;
   out_832782703925887625[245] = 0;
   out_832782703925887625[246] = 0;
   out_832782703925887625[247] = 1;
   out_832782703925887625[248] = 0;
   out_832782703925887625[249] = 0;
   out_832782703925887625[250] = 0;
   out_832782703925887625[251] = 0;
   out_832782703925887625[252] = 0;
   out_832782703925887625[253] = 0;
   out_832782703925887625[254] = 0;
   out_832782703925887625[255] = 0;
   out_832782703925887625[256] = 0;
   out_832782703925887625[257] = 0;
   out_832782703925887625[258] = 0;
   out_832782703925887625[259] = 0;
   out_832782703925887625[260] = 0;
   out_832782703925887625[261] = 0;
   out_832782703925887625[262] = 0;
   out_832782703925887625[263] = 0;
   out_832782703925887625[264] = 0;
   out_832782703925887625[265] = 0;
   out_832782703925887625[266] = 1;
   out_832782703925887625[267] = 0;
   out_832782703925887625[268] = 0;
   out_832782703925887625[269] = 0;
   out_832782703925887625[270] = 0;
   out_832782703925887625[271] = 0;
   out_832782703925887625[272] = 0;
   out_832782703925887625[273] = 0;
   out_832782703925887625[274] = 0;
   out_832782703925887625[275] = 0;
   out_832782703925887625[276] = 0;
   out_832782703925887625[277] = 0;
   out_832782703925887625[278] = 0;
   out_832782703925887625[279] = 0;
   out_832782703925887625[280] = 0;
   out_832782703925887625[281] = 0;
   out_832782703925887625[282] = 0;
   out_832782703925887625[283] = 0;
   out_832782703925887625[284] = 0;
   out_832782703925887625[285] = 1;
   out_832782703925887625[286] = 0;
   out_832782703925887625[287] = 0;
   out_832782703925887625[288] = 0;
   out_832782703925887625[289] = 0;
   out_832782703925887625[290] = 0;
   out_832782703925887625[291] = 0;
   out_832782703925887625[292] = 0;
   out_832782703925887625[293] = 0;
   out_832782703925887625[294] = 0;
   out_832782703925887625[295] = 0;
   out_832782703925887625[296] = 0;
   out_832782703925887625[297] = 0;
   out_832782703925887625[298] = 0;
   out_832782703925887625[299] = 0;
   out_832782703925887625[300] = 0;
   out_832782703925887625[301] = 0;
   out_832782703925887625[302] = 0;
   out_832782703925887625[303] = 0;
   out_832782703925887625[304] = 1;
   out_832782703925887625[305] = 0;
   out_832782703925887625[306] = 0;
   out_832782703925887625[307] = 0;
   out_832782703925887625[308] = 0;
   out_832782703925887625[309] = 0;
   out_832782703925887625[310] = 0;
   out_832782703925887625[311] = 0;
   out_832782703925887625[312] = 0;
   out_832782703925887625[313] = 0;
   out_832782703925887625[314] = 0;
   out_832782703925887625[315] = 0;
   out_832782703925887625[316] = 0;
   out_832782703925887625[317] = 0;
   out_832782703925887625[318] = 0;
   out_832782703925887625[319] = 0;
   out_832782703925887625[320] = 0;
   out_832782703925887625[321] = 0;
   out_832782703925887625[322] = 0;
   out_832782703925887625[323] = 1;
}
void h_4(double *state, double *unused, double *out_2235244296302641010) {
   out_2235244296302641010[0] = state[6] + state[9];
   out_2235244296302641010[1] = state[7] + state[10];
   out_2235244296302641010[2] = state[8] + state[11];
}
void H_4(double *state, double *unused, double *out_5408249360674302047) {
   out_5408249360674302047[0] = 0;
   out_5408249360674302047[1] = 0;
   out_5408249360674302047[2] = 0;
   out_5408249360674302047[3] = 0;
   out_5408249360674302047[4] = 0;
   out_5408249360674302047[5] = 0;
   out_5408249360674302047[6] = 1;
   out_5408249360674302047[7] = 0;
   out_5408249360674302047[8] = 0;
   out_5408249360674302047[9] = 1;
   out_5408249360674302047[10] = 0;
   out_5408249360674302047[11] = 0;
   out_5408249360674302047[12] = 0;
   out_5408249360674302047[13] = 0;
   out_5408249360674302047[14] = 0;
   out_5408249360674302047[15] = 0;
   out_5408249360674302047[16] = 0;
   out_5408249360674302047[17] = 0;
   out_5408249360674302047[18] = 0;
   out_5408249360674302047[19] = 0;
   out_5408249360674302047[20] = 0;
   out_5408249360674302047[21] = 0;
   out_5408249360674302047[22] = 0;
   out_5408249360674302047[23] = 0;
   out_5408249360674302047[24] = 0;
   out_5408249360674302047[25] = 1;
   out_5408249360674302047[26] = 0;
   out_5408249360674302047[27] = 0;
   out_5408249360674302047[28] = 1;
   out_5408249360674302047[29] = 0;
   out_5408249360674302047[30] = 0;
   out_5408249360674302047[31] = 0;
   out_5408249360674302047[32] = 0;
   out_5408249360674302047[33] = 0;
   out_5408249360674302047[34] = 0;
   out_5408249360674302047[35] = 0;
   out_5408249360674302047[36] = 0;
   out_5408249360674302047[37] = 0;
   out_5408249360674302047[38] = 0;
   out_5408249360674302047[39] = 0;
   out_5408249360674302047[40] = 0;
   out_5408249360674302047[41] = 0;
   out_5408249360674302047[42] = 0;
   out_5408249360674302047[43] = 0;
   out_5408249360674302047[44] = 1;
   out_5408249360674302047[45] = 0;
   out_5408249360674302047[46] = 0;
   out_5408249360674302047[47] = 1;
   out_5408249360674302047[48] = 0;
   out_5408249360674302047[49] = 0;
   out_5408249360674302047[50] = 0;
   out_5408249360674302047[51] = 0;
   out_5408249360674302047[52] = 0;
   out_5408249360674302047[53] = 0;
}
void h_10(double *state, double *unused, double *out_7990426856358959486) {
   out_7990426856358959486[0] = 9.8100000000000005*sin(state[1]) - state[4]*state[8] + state[5]*state[7] + state[12] + state[15];
   out_7990426856358959486[1] = -9.8100000000000005*sin(state[0])*cos(state[1]) + state[3]*state[8] - state[5]*state[6] + state[13] + state[16];
   out_7990426856358959486[2] = -9.8100000000000005*cos(state[0])*cos(state[1]) - state[3]*state[7] + state[4]*state[6] + state[14] + state[17];
}
void H_10(double *state, double *unused, double *out_3094779841330175744) {
   out_3094779841330175744[0] = 0;
   out_3094779841330175744[1] = 9.8100000000000005*cos(state[1]);
   out_3094779841330175744[2] = 0;
   out_3094779841330175744[3] = 0;
   out_3094779841330175744[4] = -state[8];
   out_3094779841330175744[5] = state[7];
   out_3094779841330175744[6] = 0;
   out_3094779841330175744[7] = state[5];
   out_3094779841330175744[8] = -state[4];
   out_3094779841330175744[9] = 0;
   out_3094779841330175744[10] = 0;
   out_3094779841330175744[11] = 0;
   out_3094779841330175744[12] = 1;
   out_3094779841330175744[13] = 0;
   out_3094779841330175744[14] = 0;
   out_3094779841330175744[15] = 1;
   out_3094779841330175744[16] = 0;
   out_3094779841330175744[17] = 0;
   out_3094779841330175744[18] = -9.8100000000000005*cos(state[0])*cos(state[1]);
   out_3094779841330175744[19] = 9.8100000000000005*sin(state[0])*sin(state[1]);
   out_3094779841330175744[20] = 0;
   out_3094779841330175744[21] = state[8];
   out_3094779841330175744[22] = 0;
   out_3094779841330175744[23] = -state[6];
   out_3094779841330175744[24] = -state[5];
   out_3094779841330175744[25] = 0;
   out_3094779841330175744[26] = state[3];
   out_3094779841330175744[27] = 0;
   out_3094779841330175744[28] = 0;
   out_3094779841330175744[29] = 0;
   out_3094779841330175744[30] = 0;
   out_3094779841330175744[31] = 1;
   out_3094779841330175744[32] = 0;
   out_3094779841330175744[33] = 0;
   out_3094779841330175744[34] = 1;
   out_3094779841330175744[35] = 0;
   out_3094779841330175744[36] = 9.8100000000000005*sin(state[0])*cos(state[1]);
   out_3094779841330175744[37] = 9.8100000000000005*sin(state[1])*cos(state[0]);
   out_3094779841330175744[38] = 0;
   out_3094779841330175744[39] = -state[7];
   out_3094779841330175744[40] = state[6];
   out_3094779841330175744[41] = 0;
   out_3094779841330175744[42] = state[4];
   out_3094779841330175744[43] = -state[3];
   out_3094779841330175744[44] = 0;
   out_3094779841330175744[45] = 0;
   out_3094779841330175744[46] = 0;
   out_3094779841330175744[47] = 0;
   out_3094779841330175744[48] = 0;
   out_3094779841330175744[49] = 0;
   out_3094779841330175744[50] = 1;
   out_3094779841330175744[51] = 0;
   out_3094779841330175744[52] = 0;
   out_3094779841330175744[53] = 1;
}
void h_13(double *state, double *unused, double *out_3582796290628829151) {
   out_3582796290628829151[0] = state[3];
   out_3582796290628829151[1] = state[4];
   out_3582796290628829151[2] = state[5];
}
void H_13(double *state, double *unused, double *out_1574493897371778023) {
   out_1574493897371778023[0] = 0;
   out_1574493897371778023[1] = 0;
   out_1574493897371778023[2] = 0;
   out_1574493897371778023[3] = 1;
   out_1574493897371778023[4] = 0;
   out_1574493897371778023[5] = 0;
   out_1574493897371778023[6] = 0;
   out_1574493897371778023[7] = 0;
   out_1574493897371778023[8] = 0;
   out_1574493897371778023[9] = 0;
   out_1574493897371778023[10] = 0;
   out_1574493897371778023[11] = 0;
   out_1574493897371778023[12] = 0;
   out_1574493897371778023[13] = 0;
   out_1574493897371778023[14] = 0;
   out_1574493897371778023[15] = 0;
   out_1574493897371778023[16] = 0;
   out_1574493897371778023[17] = 0;
   out_1574493897371778023[18] = 0;
   out_1574493897371778023[19] = 0;
   out_1574493897371778023[20] = 0;
   out_1574493897371778023[21] = 0;
   out_1574493897371778023[22] = 1;
   out_1574493897371778023[23] = 0;
   out_1574493897371778023[24] = 0;
   out_1574493897371778023[25] = 0;
   out_1574493897371778023[26] = 0;
   out_1574493897371778023[27] = 0;
   out_1574493897371778023[28] = 0;
   out_1574493897371778023[29] = 0;
   out_1574493897371778023[30] = 0;
   out_1574493897371778023[31] = 0;
   out_1574493897371778023[32] = 0;
   out_1574493897371778023[33] = 0;
   out_1574493897371778023[34] = 0;
   out_1574493897371778023[35] = 0;
   out_1574493897371778023[36] = 0;
   out_1574493897371778023[37] = 0;
   out_1574493897371778023[38] = 0;
   out_1574493897371778023[39] = 0;
   out_1574493897371778023[40] = 0;
   out_1574493897371778023[41] = 1;
   out_1574493897371778023[42] = 0;
   out_1574493897371778023[43] = 0;
   out_1574493897371778023[44] = 0;
   out_1574493897371778023[45] = 0;
   out_1574493897371778023[46] = 0;
   out_1574493897371778023[47] = 0;
   out_1574493897371778023[48] = 0;
   out_1574493897371778023[49] = 0;
   out_1574493897371778023[50] = 0;
   out_1574493897371778023[51] = 0;
   out_1574493897371778023[52] = 0;
   out_1574493897371778023[53] = 0;
}
void h_14(double *state, double *unused, double *out_316760564705387105) {
   out_316760564705387105[0] = state[6];
   out_316760564705387105[1] = state[7];
   out_316760564705387105[2] = state[8];
}
void H_14(double *state, double *unused, double *out_2325460928378929751) {
   out_2325460928378929751[0] = 0;
   out_2325460928378929751[1] = 0;
   out_2325460928378929751[2] = 0;
   out_2325460928378929751[3] = 0;
   out_2325460928378929751[4] = 0;
   out_2325460928378929751[5] = 0;
   out_2325460928378929751[6] = 1;
   out_2325460928378929751[7] = 0;
   out_2325460928378929751[8] = 0;
   out_2325460928378929751[9] = 0;
   out_2325460928378929751[10] = 0;
   out_2325460928378929751[11] = 0;
   out_2325460928378929751[12] = 0;
   out_2325460928378929751[13] = 0;
   out_2325460928378929751[14] = 0;
   out_2325460928378929751[15] = 0;
   out_2325460928378929751[16] = 0;
   out_2325460928378929751[17] = 0;
   out_2325460928378929751[18] = 0;
   out_2325460928378929751[19] = 0;
   out_2325460928378929751[20] = 0;
   out_2325460928378929751[21] = 0;
   out_2325460928378929751[22] = 0;
   out_2325460928378929751[23] = 0;
   out_2325460928378929751[24] = 0;
   out_2325460928378929751[25] = 1;
   out_2325460928378929751[26] = 0;
   out_2325460928378929751[27] = 0;
   out_2325460928378929751[28] = 0;
   out_2325460928378929751[29] = 0;
   out_2325460928378929751[30] = 0;
   out_2325460928378929751[31] = 0;
   out_2325460928378929751[32] = 0;
   out_2325460928378929751[33] = 0;
   out_2325460928378929751[34] = 0;
   out_2325460928378929751[35] = 0;
   out_2325460928378929751[36] = 0;
   out_2325460928378929751[37] = 0;
   out_2325460928378929751[38] = 0;
   out_2325460928378929751[39] = 0;
   out_2325460928378929751[40] = 0;
   out_2325460928378929751[41] = 0;
   out_2325460928378929751[42] = 0;
   out_2325460928378929751[43] = 0;
   out_2325460928378929751[44] = 1;
   out_2325460928378929751[45] = 0;
   out_2325460928378929751[46] = 0;
   out_2325460928378929751[47] = 0;
   out_2325460928378929751[48] = 0;
   out_2325460928378929751[49] = 0;
   out_2325460928378929751[50] = 0;
   out_2325460928378929751[51] = 0;
   out_2325460928378929751[52] = 0;
   out_2325460928378929751[53] = 0;
}
#include <eigen3/Eigen/Dense>
#include <iostream>

typedef Eigen::Matrix<double, DIM, DIM, Eigen::RowMajor> DDM;
typedef Eigen::Matrix<double, EDIM, EDIM, Eigen::RowMajor> EEM;
typedef Eigen::Matrix<double, DIM, EDIM, Eigen::RowMajor> DEM;

void predict(double *in_x, double *in_P, double *in_Q, double dt) {
  typedef Eigen::Matrix<double, MEDIM, MEDIM, Eigen::RowMajor> RRM;

  double nx[DIM] = {0};
  double in_F[EDIM*EDIM] = {0};

  // functions from sympy
  f_fun(in_x, dt, nx);
  F_fun(in_x, dt, in_F);


  EEM F(in_F);
  EEM P(in_P);
  EEM Q(in_Q);

  RRM F_main = F.topLeftCorner(MEDIM, MEDIM);
  P.topLeftCorner(MEDIM, MEDIM) = (F_main * P.topLeftCorner(MEDIM, MEDIM)) * F_main.transpose();
  P.topRightCorner(MEDIM, EDIM - MEDIM) = F_main * P.topRightCorner(MEDIM, EDIM - MEDIM);
  P.bottomLeftCorner(EDIM - MEDIM, MEDIM) = P.bottomLeftCorner(EDIM - MEDIM, MEDIM) * F_main.transpose();

  P = P + dt*Q;

  // copy out state
  memcpy(in_x, nx, DIM * sizeof(double));
  memcpy(in_P, P.data(), EDIM * EDIM * sizeof(double));
}

// note: extra_args dim only correct when null space projecting
// otherwise 1
template <int ZDIM, int EADIM, bool MAHA_TEST>
void update(double *in_x, double *in_P, Hfun h_fun, Hfun H_fun, Hfun Hea_fun, double *in_z, double *in_R, double *in_ea, double MAHA_THRESHOLD) {
  typedef Eigen::Matrix<double, ZDIM, ZDIM, Eigen::RowMajor> ZZM;
  typedef Eigen::Matrix<double, ZDIM, DIM, Eigen::RowMajor> ZDM;
  typedef Eigen::Matrix<double, Eigen::Dynamic, EDIM, Eigen::RowMajor> XEM;
  //typedef Eigen::Matrix<double, EDIM, ZDIM, Eigen::RowMajor> EZM;
  typedef Eigen::Matrix<double, Eigen::Dynamic, 1> X1M;
  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> XXM;

  double in_hx[ZDIM] = {0};
  double in_H[ZDIM * DIM] = {0};
  double in_H_mod[EDIM * DIM] = {0};
  double delta_x[EDIM] = {0};
  double x_new[DIM] = {0};


  // state x, P
  Eigen::Matrix<double, ZDIM, 1> z(in_z);
  EEM P(in_P);
  ZZM pre_R(in_R);

  // functions from sympy
  h_fun(in_x, in_ea, in_hx);
  H_fun(in_x, in_ea, in_H);
  ZDM pre_H(in_H);

  // get y (y = z - hx)
  Eigen::Matrix<double, ZDIM, 1> pre_y(in_hx); pre_y = z - pre_y;
  X1M y; XXM H; XXM R;
  if (Hea_fun){
    typedef Eigen::Matrix<double, ZDIM, EADIM, Eigen::RowMajor> ZAM;
    double in_Hea[ZDIM * EADIM] = {0};
    Hea_fun(in_x, in_ea, in_Hea);
    ZAM Hea(in_Hea);
    XXM A = Hea.transpose().fullPivLu().kernel();


    y = A.transpose() * pre_y;
    H = A.transpose() * pre_H;
    R = A.transpose() * pre_R * A;
  } else {
    y = pre_y;
    H = pre_H;
    R = pre_R;
  }
  // get modified H
  H_mod_fun(in_x, in_H_mod);
  DEM H_mod(in_H_mod);
  XEM H_err = H * H_mod;

  // Do mahalobis distance test
  if (MAHA_TEST){
    XXM a = (H_err * P * H_err.transpose() + R).inverse();
    double maha_dist = y.transpose() * a * y;
    if (maha_dist > MAHA_THRESHOLD){
      R = 1.0e16 * R;
    }
  }

  // Outlier resilient weighting
  double weight = 1;//(1.5)/(1 + y.squaredNorm()/R.sum());

  // kalman gains and I_KH
  XXM S = ((H_err * P) * H_err.transpose()) + R/weight;
  XEM KT = S.fullPivLu().solve(H_err * P.transpose());
  //EZM K = KT.transpose(); TODO: WHY DOES THIS NOT COMPILE?
  //EZM K = S.fullPivLu().solve(H_err * P.transpose()).transpose();
  //std::cout << "Here is the matrix rot:\n" << K << std::endl;
  EEM I_KH = Eigen::Matrix<double, EDIM, EDIM>::Identity() - (KT.transpose() * H_err);

  // update state by injecting dx
  Eigen::Matrix<double, EDIM, 1> dx(delta_x);
  dx  = (KT.transpose() * y);
  memcpy(delta_x, dx.data(), EDIM * sizeof(double));
  err_fun(in_x, delta_x, x_new);
  Eigen::Matrix<double, DIM, 1> x(x_new);

  // update cov
  P = ((I_KH * P) * I_KH.transpose()) + ((KT.transpose() * R) * KT);

  // copy out state
  memcpy(in_x, x.data(), DIM * sizeof(double));
  memcpy(in_P, P.data(), EDIM * EDIM * sizeof(double));
  memcpy(in_z, y.data(), y.rows() * sizeof(double));
}




}
extern "C" {

void pose_update_4(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<3, 3, 0>(in_x, in_P, h_4, H_4, NULL, in_z, in_R, in_ea, MAHA_THRESH_4);
}
void pose_update_10(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<3, 3, 0>(in_x, in_P, h_10, H_10, NULL, in_z, in_R, in_ea, MAHA_THRESH_10);
}
void pose_update_13(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<3, 3, 0>(in_x, in_P, h_13, H_13, NULL, in_z, in_R, in_ea, MAHA_THRESH_13);
}
void pose_update_14(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<3, 3, 0>(in_x, in_P, h_14, H_14, NULL, in_z, in_R, in_ea, MAHA_THRESH_14);
}
void pose_err_fun(double *nom_x, double *delta_x, double *out_1610068831843926615) {
  err_fun(nom_x, delta_x, out_1610068831843926615);
}
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_9009726282604965886) {
  inv_err_fun(nom_x, true_x, out_9009726282604965886);
}
void pose_H_mod_fun(double *state, double *out_7227948086580598908) {
  H_mod_fun(state, out_7227948086580598908);
}
void pose_f_fun(double *state, double dt, double *out_4483597399355948729) {
  f_fun(state,  dt, out_4483597399355948729);
}
void pose_F_fun(double *state, double dt, double *out_832782703925887625) {
  F_fun(state,  dt, out_832782703925887625);
}
void pose_h_4(double *state, double *unused, double *out_2235244296302641010) {
  h_4(state, unused, out_2235244296302641010);
}
void pose_H_4(double *state, double *unused, double *out_5408249360674302047) {
  H_4(state, unused, out_5408249360674302047);
}
void pose_h_10(double *state, double *unused, double *out_7990426856358959486) {
  h_10(state, unused, out_7990426856358959486);
}
void pose_H_10(double *state, double *unused, double *out_3094779841330175744) {
  H_10(state, unused, out_3094779841330175744);
}
void pose_h_13(double *state, double *unused, double *out_3582796290628829151) {
  h_13(state, unused, out_3582796290628829151);
}
void pose_H_13(double *state, double *unused, double *out_1574493897371778023) {
  H_13(state, unused, out_1574493897371778023);
}
void pose_h_14(double *state, double *unused, double *out_316760564705387105) {
  h_14(state, unused, out_316760564705387105);
}
void pose_H_14(double *state, double *unused, double *out_2325460928378929751) {
  H_14(state, unused, out_2325460928378929751);
}
void pose_predict(double *in_x, double *in_P, double *in_Q, double dt) {
  predict(in_x, in_P, in_Q, dt);
}
}

const EKF pose = {
  .name = "pose",
  .kinds = { 4, 10, 13, 14 },
  .feature_kinds = {  },
  .f_fun = pose_f_fun,
  .F_fun = pose_F_fun,
  .err_fun = pose_err_fun,
  .inv_err_fun = pose_inv_err_fun,
  .H_mod_fun = pose_H_mod_fun,
  .predict = pose_predict,
  .hs = {
    { 4, pose_h_4 },
    { 10, pose_h_10 },
    { 13, pose_h_13 },
    { 14, pose_h_14 },
  },
  .Hs = {
    { 4, pose_H_4 },
    { 10, pose_H_10 },
    { 13, pose_H_13 },
    { 14, pose_H_14 },
  },
  .updates = {
    { 4, pose_update_4 },
    { 10, pose_update_10 },
    { 13, pose_update_13 },
    { 14, pose_update_14 },
  },
  .Hes = {
  },
  .sets = {
  },
  .extra_routines = {
  },
};

ekf_lib_init(pose)
