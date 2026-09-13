#pragma once
#include "rednose/helpers/ekf.h"
extern "C" {
void car_update_25(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_24(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_30(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_26(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_27(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_29(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_28(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_31(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_err_fun(double *nom_x, double *delta_x, double *out_3750715589988540673);
void car_inv_err_fun(double *nom_x, double *true_x, double *out_5753922970186248840);
void car_H_mod_fun(double *state, double *out_8083608636152881178);
void car_f_fun(double *state, double dt, double *out_6073296693713362518);
void car_F_fun(double *state, double dt, double *out_6381757595176208300);
void car_h_25(double *state, double *unused, double *out_1429954134961332910);
void car_H_25(double *state, double *unused, double *out_1986700269546506562);
void car_h_24(double *state, double *unused, double *out_3742540879801299667);
void car_H_24(double *state, double *unused, double *out_4212408053525375124);
void car_h_30(double *state, double *unused, double *out_1154760072676827021);
void car_H_30(double *state, double *unused, double *out_8903390611038123317);
void car_h_26(double *state, double *unused, double *out_211250219775164586);
void car_H_26(double *state, double *unused, double *out_1754803049327549662);
void car_h_27(double *state, double *unused, double *out_4034359455770030561);
void car_H_27(double *state, double *unused, double *out_6728627299237698406);
void car_h_29(double *state, double *unused, double *out_4191546124121159706);
void car_H_29(double *state, double *unused, double *out_9033122118357036115);
void car_h_28(double *state, double *unused, double *out_967313811890815084);
void car_H_28(double *state, double *unused, double *out_67134444701383201);
void car_h_31(double *state, double *unused, double *out_1272767466610203765);
void car_H_31(double *state, double *unused, double *out_2017346231423466990);
void car_predict(double *in_x, double *in_P, double *in_Q, double dt);
void car_set_mass(double x);
void car_set_rotational_inertia(double x);
void car_set_center_to_front(double x);
void car_set_center_to_rear(double x);
void car_set_stiffness_front(double x);
void car_set_stiffness_rear(double x);
}