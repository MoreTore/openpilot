#pragma once
#include "rednose/helpers/ekf.h"
extern "C" {
void pose_update_4(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_10(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_13(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_14(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_err_fun(double *nom_x, double *delta_x, double *out_1610068831843926615);
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_9009726282604965886);
void pose_H_mod_fun(double *state, double *out_7227948086580598908);
void pose_f_fun(double *state, double dt, double *out_4483597399355948729);
void pose_F_fun(double *state, double dt, double *out_832782703925887625);
void pose_h_4(double *state, double *unused, double *out_2235244296302641010);
void pose_H_4(double *state, double *unused, double *out_5408249360674302047);
void pose_h_10(double *state, double *unused, double *out_7990426856358959486);
void pose_H_10(double *state, double *unused, double *out_3094779841330175744);
void pose_h_13(double *state, double *unused, double *out_3582796290628829151);
void pose_H_13(double *state, double *unused, double *out_1574493897371778023);
void pose_h_14(double *state, double *unused, double *out_316760564705387105);
void pose_H_14(double *state, double *unused, double *out_2325460928378929751);
void pose_predict(double *in_x, double *in_P, double *in_Q, double dt);
}