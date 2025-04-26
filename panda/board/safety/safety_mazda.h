// CAN msgs we care about
/********** GEN1 msgs **********/
#define MAZDA_LKAS          0x243
#define MAZDA_LKAS_HUD      0x440
#define MAZDA_CRZ_CTRL      0x21c
#define MAZDA_CRZ_BTNS      0x09d
#define MAZDA_STEER_TORQUE  0x240
#define MAZDA_ENGINE_DATA   0x202
#define MAZDA_PEDALS        0x165
#define TI_STEER_TORQUE     0x24A

// Radar
#define MAZDA_CRZ_INFO      0x21B
#define MAZDA_CRZ_CTRL      0x21c
#define MAZDA_RADAR_361     0x361
#define MAZDA_RADAR_362     0x362
#define MAZDA_RADAR_363     0x363
#define MAZDA_RADAR_364     0x364
#define MAZDA_RADAR_365     0x365
#define MAZDA_RADAR_366     0x366
#define MAZDA_RADAR_499     0x499

/************* GEN2 msgs *************/
#define MAZDA_2019_BRAKE          0x43F  // main bus
#define MAZDA_2019_GAS            0x202  // camera bus DBC: ENGINE_DATA
#define MAZDA_2019_CRUISE         0x44A  // main bus. DBC: CRUISE_STATE
#define MAZDA_2019_SPEED          0x217  // camera bus. DBC: SPEED
#define MAZDA_2019_STEER_TORQUE   0x24B  // aux bus. DBC: EPS_FEEDBACK
#define MAZDA_2019_CRZ_BTNS       0x9d   // rx on main tx on camera. DBC: CRZ_BTNS
#define MAZDA_2019_ACC            0x220  // main bus. DBC: ACC

#define MAZDA_TI_LKAS       0x249

// CAN bus numbers
#define MAZDA_MAIN 0
#define MAZDA_AUX  1
#define MAZDA_CAM  2

// param flag masks
const int FLAG_GEN1 = 1;
const int FLAG_GEN2 = 2;
const int FLAG_TORQUE_INTERCEPTOR = 4;
const int FLAG_RADAR_INTERCEPTOR = 8;
const int FLAG_NO_FSC = 16;
const int FLAG_NO_MRCC = 32;

bool gen1 = false;
bool gen2 = false;
bool torque_interceptor = false;
bool radar_interceptor = false;
bool no_fsc = false;
bool no_mrcc = false;

const SteeringLimits MAZDA_STEERING_LIMITS = {
  .max_steer = 800,
  .max_rate_up = 10,
  .max_rate_down = 25,
  .max_rt_delta = 300,
  .max_rt_interval = 250000,
  .driver_torque_factor = 1,
  .driver_torque_allowance = 15,
  .type = TorqueDriverLimited,
};

const SteeringLimits MAZDA_2019_STEERING_LIMITS = {
  .max_steer = 8000,
  .max_rate_up = 45,
  .max_rate_down = 80,
  .max_rt_delta = 1688,  // (45*100hz*250000/1000000)*1.5
  .max_rt_interval = 250000,
  .driver_torque_factor = 1,
  .driver_torque_allowance = 1400,
  .type = TorqueDriverLimited,
};

const CanMsg MAZDA_TX_MSGS[] = {{MAZDA_LKAS, 0, 8}, {MAZDA_CRZ_BTNS, 0, 8}, {MAZDA_LKAS_HUD, 0, 8}};
const CanMsg MAZDA_TI_TX_MSGS[] = {{MAZDA_LKAS, 0, 8}, {MAZDA_TI_LKAS, 1, 8}, {MAZDA_CRZ_BTNS, 0, 8}, {MAZDA_LKAS_HUD, 0, 8}};
const CanMsg MAZDA_RI_TX_MSGS[] = {{MAZDA_LKAS, 0, 8}, {MAZDA_CRZ_BTNS, 0, 8}, {MAZDA_LKAS_HUD, 0, 8},
                                  {MAZDA_CRZ_CTRL, 0, 8}, {MAZDA_CRZ_INFO, 0, 8}, {MAZDA_RADAR_361, 0, 8}, {MAZDA_RADAR_362, 0, 8},
                                  {MAZDA_RADAR_363, 0, 8}, {MAZDA_RADAR_364, 0, 8}, {MAZDA_RADAR_365, 0, 8}, {MAZDA_RADAR_366, 0, 8},
                                  {MAZDA_RADAR_499, 0, 8}};
const CanMsg MAZDA_TI_RI_TX_MSGS[] = {{MAZDA_LKAS, 0, 8}, {MAZDA_TI_LKAS, 1, 8}, {MAZDA_CRZ_BTNS, 0, 8}, {MAZDA_LKAS_HUD, 0, 8},
                                {MAZDA_CRZ_CTRL, 0, 8}, {MAZDA_CRZ_INFO, 0, 8}, {MAZDA_RADAR_361, 0, 8}, {MAZDA_RADAR_362, 0, 8},
                                {MAZDA_RADAR_363, 0, 8}, {MAZDA_RADAR_364, 0, 8}, {MAZDA_RADAR_365, 0, 8}, {MAZDA_RADAR_366, 0, 8},
                                {MAZDA_RADAR_499, 0, 8}};

const CanMsg MAZDA_2019_TX_MSGS[] = {{MAZDA_TI_LKAS, 1, 8}, {MAZDA_2019_ACC, 2, 8}};

RxCheck mazda_rx_checks[] = {
  {.msg = {{MAZDA_CRZ_CTRL,     0, 8, .frequency = 50U}, { 0 }, { 0 }}},
  {.msg = {{MAZDA_CRZ_BTNS,     0, 8, .frequency = 10U}, { 0 }, { 0 }}},
  {.msg = {{MAZDA_STEER_TORQUE, 0, 8, .frequency = 83U}, { 0 }, { 0 }}},
  {.msg = {{MAZDA_ENGINE_DATA,  0, 8, .frequency = 100U}, { 0 }, { 0 }}},
  {.msg = {{MAZDA_PEDALS,       0, 8, .frequency = 50U}, { 0 }, { 0 }}},
};

RxCheck mazda_ti_rx_checks[] = {
  {.msg = {{MAZDA_CRZ_BTNS,     0, 8, .frequency = 10U}, { 0 }, { 0 }}},
  {.msg = {{MAZDA_STEER_TORQUE, 0, 8, .frequency = 83U}, { 0 }, { 0 }}},
  {.msg = {{MAZDA_ENGINE_DATA,  0, 8, .frequency = 100U}, { 0 }, { 0 }}},
  {.msg = {{MAZDA_PEDALS,       0, 8, .frequency = 50U}, { 0 }, { 0 }}},
  {.msg = {{TI_STEER_TORQUE,    1, 8, .frequency = 50U}, { 0 }, { 0 }}},
};

RxCheck mazda_2019_rx_checks[] = {
  {.msg = {{MAZDA_2019_BRAKE,         0, 8, .frequency = 20U}, { 0 }, { 0 }}},
  {.msg = {{MAZDA_2019_GAS,           2, 8, .frequency = 100U}, { 0 }, { 0 }}},
  {.msg = {{MAZDA_2019_CRUISE,        0, 8, .frequency = 10U}, { 0 }, { 0 }}},
  {.msg = {{MAZDA_2019_SPEED,         2, 8, .frequency = 30U}, { 0 }, { 0 }}},
  {.msg = {{MAZDA_2019_STEER_TORQUE,  1, 8, .frequency = 50U}, { 0 }, { 0 }}},
};

// track msgs coming from OP so that we know what CAM msgs to drop and what to forward
static void mazda_rx_hook(const CANPacket_t *to_push) {
  int addr = GET_ADDR(to_push);
  if ((int)GET_BUS(to_push) == MAZDA_MAIN) {
    if (gen1){
      if (addr == MAZDA_ENGINE_DATA) {
        // sample speed: scale by 0.01 to get kph
        int speed = (GET_BYTE(to_push, 2) << 8) | GET_BYTE(to_push, 3);
        vehicle_moving = speed > 10; // moving when speed > 0.1 kph
      }

      if (addr == MAZDA_STEER_TORQUE && !torque_interceptor) {
        int torque_driver_new = GET_BYTE(to_push, 0) - 127U;
        // update array of samples
        update_sample(&torque_driver, torque_driver_new);
      }

      // enter controls on rising edge of ACC, exit controls on ACC off
      if (addr == MAZDA_CRZ_CTRL && !no_mrcc && !radar_interceptor) {
        acc_main_on = GET_BIT(to_push, 17U);
        bool cruise_engaged = GET_BYTE(to_push, 0) & 0x8U;
        pcm_cruise_check(cruise_engaged);
      }

      if (addr == MAZDA_ENGINE_DATA) {
        gas_pressed = (GET_BYTE(to_push, 4) || (GET_BYTE(to_push, 5) & 0xF0U));
      }

      if (addr == MAZDA_PEDALS) {
        brake_pressed = (GET_BYTE(to_push, 0) & 0x10U);
        if (radar_interceptor || no_mrcc) {
          // acc_main_on = GET_BIT(to_push, 17U); TODO
          bool cruise_engaged = GET_BYTE(to_push, 0) & 0x8U;
          pcm_cruise_check(cruise_engaged);
        }
      }

      generic_rx_checks((addr == MAZDA_LKAS));
    }
    if (gen2) {
      if (addr == MAZDA_2019_BRAKE) {
        brake_pressed = (GET_BYTE(to_push, 5) & 0x4U);
      }

      if (addr == MAZDA_2019_CRUISE) {
        acc_main_on = true;
        bool cruise_engaged = GET_BYTE(to_push, 0) & 0x20U;
        bool pre_enable = GET_BYTE(to_push, 0) & 0x40U;
        pcm_cruise_check((cruise_engaged || pre_enable));
      }

      generic_rx_checks(addr == MAZDA_2019_SPEED);
    }
  }

  if ((GET_BUS(to_push) == MAZDA_AUX)) {
    if (addr == TI_STEER_TORQUE && gen1 && torque_interceptor) {
      int torque_driver_new = GET_BYTE(to_push, 0) - 127;
      update_sample(&torque_driver, torque_driver_new);
    }

    if (addr == MAZDA_2019_STEER_TORQUE && gen2) {
      update_sample(&torque_driver, (int16_t)(GET_BYTE(to_push, 0) << 8 | GET_BYTE(to_push, 1)));
    }
  }

  if ((GET_BUS(to_push) == MAZDA_CAM)) {
    if (gen2) {
      if (addr == MAZDA_2019_GAS) {
        gas_pressed = (GET_BYTE(to_push, 4) || ((GET_BYTE(to_push, 5) & 0xC0U)));
      }

      if (addr == MAZDA_2019_SPEED) {
        // sample speed: scale by 0.01 to get kph
        int speed = (GET_BYTE(to_push, 4) << 8) | (GET_BYTE(to_push, 5));
        vehicle_moving = (speed > 10);  // moving when speed > 0.1 kph
      }
      generic_rx_checks(addr == MAZDA_2019_CRUISE);
    }
  }
}

static bool mazda_tx_hook(const CANPacket_t *to_send) {
  bool tx = true;
  int bus = GET_BUS(to_send);
  int addr = GET_ADDR(to_send);
  // Check if msg is sent on the main BUS
  if (gen1 && (bus == MAZDA_MAIN)) {

    // steer cmd checks
    if (addr == MAZDA_LKAS) {
      int desired_torque = (((GET_BYTE(to_send, 0) & 0x0FU) << 8) | GET_BYTE(to_send, 1)) - 2048U;

      if (steer_torque_cmd_checks(desired_torque, -1, MAZDA_STEERING_LIMITS)) {
        tx = false;
      }
    }

    // cruise buttons check
    if (addr == MAZDA_CRZ_BTNS) {
      // allow resume spamming while controls allowed, but
      // only allow cancel while contrls not allowed
      bool cancel_cmd = (GET_BYTE(to_send, 0) == 0x1U);
      if (!controls_allowed && !cancel_cmd) {
        tx = false;
      }
    }
  }
  if (gen2 && (bus == MAZDA_AUX) && (addr == MAZDA_TI_LKAS)) {
    int desired_torque = (int16_t)((GET_BYTE(to_send, 0) << 8) | GET_BYTE(to_send, 1));  // signal is signed
    if (steer_torque_cmd_checks(desired_torque, -1, MAZDA_2019_STEERING_LIMITS)) {
      tx = false;
    }
  }

  return tx;
}

static int mazda_fwd_hook(int bus, int addr) {
  int bus_fwd = -1;
  bool block = (addr == MAZDA_TI_LKAS);
  if (bus == MAZDA_MAIN) {
    block |= ((addr == MAZDA_2019_ACC) && gen2);
    if (!block) {
      bus_fwd = MAZDA_CAM;
    }
  } else if (bus == MAZDA_CAM) {
    if (gen1) {
      block |= (addr == MAZDA_LKAS) || (addr == MAZDA_LKAS_HUD);
      if (radar_interceptor) {
        block |= (addr == MAZDA_CRZ_INFO) || (addr == MAZDA_CRZ_CTRL);
        block |= (addr == MAZDA_RADAR_361) || (addr == MAZDA_RADAR_362);
        block |= (addr == MAZDA_RADAR_363) || (addr == MAZDA_RADAR_364);
        block |= (addr == MAZDA_RADAR_365) || (addr == MAZDA_RADAR_366);
      }
    }
    if (!block) {
      bus_fwd = MAZDA_MAIN;
    }
  } else {
    // don't fwd
  }

  return bus_fwd;
}

static safety_config mazda_init(uint16_t param) {
  safety_config ret = BUILD_SAFETY_CFG(mazda_rx_checks, MAZDA_TX_MSGS);
  gen1 = GET_FLAG(param, FLAG_GEN1);
  gen2 = GET_FLAG(param, FLAG_GEN2);
  radar_interceptor = GET_FLAG(param, FLAG_RADAR_INTERCEPTOR);
  torque_interceptor = GET_FLAG(param, FLAG_TORQUE_INTERCEPTOR);
  no_fsc = GET_FLAG(param, FLAG_NO_FSC);
  no_mrcc = GET_FLAG(param, FLAG_NO_MRCC);
  if (gen1) {
    SET_RX_CHECKS(mazda_rx_checks, ret);
    if (radar_interceptor && torque_interceptor) {
      SET_RX_CHECKS(mazda_ti_rx_checks, ret);
      SET_TX_MSGS(MAZDA_TI_RI_TX_MSGS, ret);
    } else if (radar_interceptor) {
      SET_TX_MSGS(MAZDA_RI_TX_MSGS, ret);
    } else if (torque_interceptor) {
      SET_RX_CHECKS(mazda_ti_rx_checks, ret);
      SET_TX_MSGS(MAZDA_TI_TX_MSGS, ret);
    } else {
      SET_TX_MSGS(MAZDA_TX_MSGS, ret);
    }
  }
  if (gen2) {
    ret = BUILD_SAFETY_CFG(mazda_2019_rx_checks, MAZDA_2019_TX_MSGS);
  }

  return ret;
}

const safety_hooks mazda_hooks = {
  .init = mazda_init,
  .rx = mazda_rx_hook,
  .tx = mazda_tx_hook,
  .fwd = mazda_fwd_hook,
};
