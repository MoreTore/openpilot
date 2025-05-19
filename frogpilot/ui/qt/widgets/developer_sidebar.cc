#include "frogpilot/ui/qt/widgets/developer_sidebar.h"

void DeveloperSidebar::drawMetric(QPainter &p, const QPair<QString, QString> &label, QColor c, int y) {
  const QRect rect = {12, y, 275, 126};

  p.setPen(Qt::NoPen);
  p.setBrush(QBrush(c));
  p.setClipRect(rect.x() + 4, rect.y(), 18, rect.height(), Qt::ClipOperation::ReplaceClip);
  p.drawRoundedRect(QRect(rect.x() + 4, rect.y() + 4, 100, 118), 18, 18);
  p.setClipping(false);

  QPen pen = QPen(QColor(0xff, 0xff, 0xff, 0x55));
  pen.setWidth(2);
  p.setPen(pen);
  p.setBrush(Qt::NoBrush);
  p.drawRoundedRect(rect, 20, 20);

  p.setPen(QColor(0xff, 0xff, 0xff));
  p.setFont(InterFont(35, QFont::DemiBold));
  p.drawText(rect.adjusted(22, 0, 0, 0), Qt::AlignCenter, label.first + "\n" + label.second);
}

DeveloperSidebar::DeveloperSidebar(QWidget *parent) : QFrame(parent) {
  setAttribute(Qt::WA_OpaquePaintEvent);
  setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Expanding);
  setFixedWidth(300);

  QObject::connect(uiState(), &UIState::uiUpdate, this, &DeveloperSidebar::updateState);
}

void DeveloperSidebar::showEvent(QShowEvent *event) {
  update_theme(frogpilotUIState());

  FrogPilotUIState &fs = *frogpilotUIState();
  FrogPilotUIScene &frogpilot_scene = fs.frogpilot_scene;
  QJsonObject &frogpilot_toggles = fs.frogpilot_toggles;

  metricAssignments.clear();
  for (int i = 1; i <= 7; ++i) {
    QString key = QString("developer_sidebar_metric%1").arg(i);
    int metricId = frogpilot_toggles.value(key).toInt();
    metricAssignments.push_back(metricId);
  }

  metricColor = frogpilot_scene.use_stock_colors ? QColor(255, 255, 255) : frogpilot_scene.sidebar_color1;
}

void DeveloperSidebar::updateState(const UIState &s, const FrogPilotUIState &fs) {
  if (!isVisible()) {
    return;
  }

  const FrogPilotUIScene &frogpilot_scene = fs.frogpilot_scene;
  const SubMaster &fpsm = *(fs.sm);

  const cereal::CarControl::Reader &carControl = fpsm["carControl"].getCarControl();
  const cereal::CarState::Reader &carState = fpsm["carState"].getCarState();
  const cereal::FrogPilotPlan::Reader &frogpilotPlan = fpsm["frogpilotPlan"].getFrogpilotPlan();
  const cereal::LiveDelayData::Reader &liveDelay = fpsm["liveDelay"].getLiveDelay();
  const cereal::LiveParametersData::Reader &liveParameters = fpsm["liveParameters"].getLiveParameters();
  const cereal::LiveTorqueParametersData::Reader &liveTorqueParameters = fpsm["liveTorqueParameters"].getLiveTorqueParameters();

  const bool is_metric = s.scene.is_metric;
  const bool use_si = fs.frogpilot_toggles.value("use_si_metrics").toBool();

  const QString accelerationUnit = (is_metric || use_si) ? tr(" m/s²") : tr(" ft/s²");
  const float accelerationConversion = (is_metric || use_si) ? 1.0f : METER_TO_FOOT;

  double acceleration = carState.getAEgo() * accelerationConversion;
  static double maxAcceleration = 0.0;
  maxAcceleration = std::max(maxAcceleration, acceleration);

  static double lateralEngagementTime = 0.0;
  lateralEngagementTime += carControl.getLatActive() && !frogpilot_scene.reverse && !frogpilot_scene.standstill ? 1 : 0;

  static double longitudinalEngagementTime = 0.0;
  longitudinalEngagementTime += carControl.getLongActive() && !frogpilot_scene.reverse && !frogpilot_scene.standstill ? 1 : 0;

  static double totalEngagementTime = 0.0;
  totalEngagementTime += !(frogpilot_scene.reverse || frogpilot_scene.standstill) ? 1 : 0;

  accelerationStatus = ItemStatus(QPair<QString, QString>(tr("ACCEL"), QString::number(acceleration, 'f', 2) + accelerationUnit), metricColor);
  accelerationJerkStatus = ItemStatus(QPair<QString, QString>(tr("ACCEL JERK"), QString::number(frogpilotPlan.getAccelerationJerk(), 'f', 2)), metricColor);
  actuatorAccelerationStatus = ItemStatus(QPair<QString, QString>(tr("ACT ACCEL"), QString::number(carControl.getActuators().getAccel() * accelerationConversion, 'f', 2) + accelerationUnit), metricColor);
  dangerJerkStatus = ItemStatus(QPair<QString, QString>(tr("DANGER JERK"), QString::number(frogpilotPlan.getDangerJerk(), 'f', 2)), metricColor);
  delayStatus = ItemStatus(QPair<QString, QString>(tr("STEER DELAY"), QString::number(liveDelay.getLateralDelay(), 'f', 5)), metricColor);
  frictionStatus = ItemStatus(QPair<QString, QString>(tr("FRICTION"), QString::number(liveTorqueParameters.getFrictionCoefficientFiltered(), 'f', 5)), metricColor);
  latAccelStatus = ItemStatus(QPair<QString, QString>(tr("LAT ACCEL"), QString::number(liveTorqueParameters.getLatAccelFactorFiltered(), 'f', 5)), metricColor);
  lateralEngagementStatus = ItemStatus(QPair<QString, QString>(tr("LATERAL %"), QString::number((lateralEngagementTime / totalEngagementTime) * 100.0f, 'f', 2) + "%"), metricColor);
  longitudinalEngagementStatus = ItemStatus(QPair<QString, QString>(tr("LONG %"), QString::number((longitudinalEngagementTime / totalEngagementTime) * 100.0f, 'f', 2) + "%"), metricColor);
  maxAccelerationStatus = ItemStatus(QPair<QString, QString>(tr("MAX ACCEL"), QString::number(maxAcceleration, 'f', 2) + accelerationUnit), metricColor);
  speedJerkStatus = ItemStatus(QPair<QString, QString>(tr("SPEED JERK"), QString::number(frogpilotPlan.getSpeedJerk(), 'f', 2)), metricColor);
  steerAngleStatus = ItemStatus(QPair<QString, QString>(tr("STEER ANGLE"), QString::number(fabs(carState.getSteeringAngleDeg()), 'f', 2)), metricColor);
  steerRatioStatus = ItemStatus(QPair<QString, QString>(tr("STEER RATIO"), QString::number(liveParameters.getSteerRatio(), 'f', 5)), metricColor);
  stiffnessFactorStatus = ItemStatus(QPair<QString, QString>(tr("STEER STIFF"), QString::number(liveParameters.getStiffnessFactor(), 'f', 5)), metricColor);
  torqueStatus = ItemStatus(QPair<QString, QString>(tr("TORQUE %"), QString::number(fabs(carControl.getActuators().getSteer() * 100.0f), 'f', 2)), metricColor);

  update();
}

void DeveloperSidebar::paintEvent(QPaintEvent *event) {
  QPainter p(this);
  p.setPen(Qt::NoPen);
  p.setRenderHint(QPainter::Antialiasing);

  p.fillRect(rect(), QColor(57, 57, 57));

  QMap<int, ItemStatus*> metricMap;
  metricMap.insert(1, &accelerationStatus);
  metricMap.insert(2, &maxAccelerationStatus);
  metricMap.insert(3, &delayStatus);
  metricMap.insert(4, &frictionStatus);
  metricMap.insert(5, &latAccelStatus);
  metricMap.insert(6, &steerRatioStatus);
  metricMap.insert(7, &stiffnessFactorStatus);
  metricMap.insert(8, &lateralEngagementStatus);
  metricMap.insert(9, &longitudinalEngagementStatus);
  metricMap.insert(10, &steerAngleStatus);
  metricMap.insert(11, &torqueStatus);
  metricMap.insert(12, &actuatorAccelerationStatus);
  metricMap.insert(13, &accelerationJerkStatus);
  metricMap.insert(14, &dangerJerkStatus);
  metricMap.insert(15, &speedJerkStatus);

  int count = 0;
  for (size_t i = 0; i < metricAssignments.size(); ++i) {
    if (metricAssignments[i] > 0 && metricMap.contains(metricAssignments[i])) {
      count++;
    }
  }
  if (count == 0) {
    return;
  }

  int metricHeight = 126;
  int spacing = (height() - (count * metricHeight)) / (count + 1);
  int y = spacing;

  for (size_t i = 0; i < metricAssignments.size(); ++i) {
    int metricId = metricAssignments[i];

    if (metricId == 0) {
      continue;
    }

    if (!metricMap.contains(metricId)) {
      continue;
    }

    ItemStatus *status = metricMap[metricId];
    drawMetric(p, status->first, status->second, y);
    y += metricHeight + spacing;
  }
}
