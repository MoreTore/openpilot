#include "frogpilot/ui/qt/offroad/model_settings.h"

bool hasAllTinygradFiles(const QDir &modelDir, const QString &modelKey) {
  QStringList tinygradSuffixes = {
    "_driving_policy_metadata.pkl",
    "_driving_policy_tinygrad.pkl",
    "_driving_vision_metadata.pkl",
    "_driving_vision_tinygrad.pkl"
  };

  for (const QString &suffix : tinygradSuffixes) {
    if (!modelDir.exists(modelKey + suffix)) {
      return false;
    }
  }
  return true;
}

QString normalizeModelKey(QString key) {
  key = key.toLower();
  if (key.endsWith("_default")) {
    key.chop(QString("_default").size());
  }
  return key;
}

FrogPilotModelPanel::FrogPilotModelPanel(FrogPilotSettingsWindow *parent) : FrogPilotListWidget(parent), parent(parent) {
  QStackedLayout *modelLayout = new QStackedLayout();
  addItem(modelLayout);

  FrogPilotListWidget *modelList = new FrogPilotListWidget(this);

  ScrollView *modelPanel = new ScrollView(modelList, this);

  modelLayout->addWidget(modelPanel);

  FrogPilotListWidget *modelLabelsList = new FrogPilotListWidget(this);

  ScrollView *modelLabelsPanel = new ScrollView(modelLabelsList, this);

  modelLayout->addWidget(modelLabelsPanel);

  const std::vector<std::tuple<QString, QString, QString, QString>> modelToggles {
    {"AutomaticallyDownloadModels", tr("Automatically Download New Models"), tr("Automatically download new driving models as they become available."), ""},
    {"DeleteModel", tr("Delete Driving Models"), tr("Delete driving models from the device."), ""},
    {"DownloadModel", tr("Download Driving Models"), tr("Download driving models to the device."), ""},
    {"ModelRandomizer", tr("Model Randomizer"), tr("Driving models are chosen at random each drive and feedback prompts are used to find the model that best suits your needs."), ""},
    {"ManageBlacklistedModels", tr("Manage Model Blacklist"), tr("Add or remove models from the <b>Model Randomizer</b>'s blacklist list."), ""},
    {"ManageScores", tr("Manage Model Ratings"), tr("Reset or view the saved ratings for the driving models."), ""},
    {"SelectModel", tr("Select Driving Model"), tr("Select the active driving model."), ""},
    {"UpdateTinygrad", tr("Update Tinygrad"), tr("Update the Tinygrad model process to support the latest models."), ""}
  };

  for (const auto &[param, title, desc, icon] : modelToggles) {
    AbstractControl *modelToggle;

    if (param == "DeleteModel") {
      deleteModelBtn = new FrogPilotButtonsControl(title, desc, icon, {tr("DELETE"), tr("DELETE ALL")});
      QObject::connect(deleteModelBtn, &FrogPilotButtonsControl::buttonClicked, [this](int id) {
        QStringList deletableModels;
        for (const QString &file : modelDir.entryList(QDir::Files)) {
          QString base = QFileInfo(file).baseName();
          for (const QString &modelKey : modelFileToNameMapProcessed.keys()) {
            if (base.startsWith(modelKey)) {
              QString modelName = modelFileToNameMapProcessed.value(modelKey);
              if (!deletableModels.contains(modelName)) {
                deletableModels.append(modelName);
              }
            }
          }
        }
        deletableModels.removeAll(processModelName(currentModel));
        deletableModels.removeAll(modelFileToNameMapProcessed.value(normalizeModelKey(QString::fromStdString(params_default.get("Model")))));
        noModelsDownloaded = deletableModels.isEmpty();

        if (id == 0) {
          QString modelToDelete = MultiOptionDialog::getSelection(tr("Select a driving model to delete"), deletableModels, "", this);
          if (!modelToDelete.isEmpty() && ConfirmationDialog::confirm(tr("Are you sure you want to delete the \"%1\" model?").arg(modelToDelete), tr("Delete"), this)) {
            QString modelFile = modelFileToNameMapProcessed.key(modelToDelete);
            for (const QString &file : modelDir.entryList(QDir::Files)) {
              QString base = QFileInfo(file).baseName();
              if (base.startsWith(modelFile)) {
                QFile::remove(modelDir.filePath(file));
              }
            }

            allModelsDownloaded = false;
          }
        } else if (id == 1) {
          if (ConfirmationDialog::confirm(tr("Are you sure you want to delete all of your downloaded driving models?"), tr("Delete"), this)) {
            for (const QString &file : modelDir.entryList(QDir::Files)) {
              QString base = QFileInfo(file).baseName();
              for (const QString &modelKey : modelFileToNameMapProcessed.keys()) {
                QString modelName = modelFileToNameMapProcessed.value(modelKey);
                if (deletableModels.contains(modelName) && base.startsWith(modelKey)) {
                  QFile::remove(modelDir.filePath(file));
                  break;
                }
              }
            }

            allModelsDownloaded = false;
            noModelsDownloaded = true;
          }
        }
      });
      modelToggle = deleteModelBtn;
    } else if (param == "DownloadModel") {
      downloadModelBtn = new FrogPilotButtonsControl(title, desc, icon, {tr("DOWNLOAD"), tr("DOWNLOAD ALL")});
      QObject::connect(downloadModelBtn, &FrogPilotButtonsControl::buttonClicked, [this](int id) {
        if (tinygradUpdate) {
          if (FrogPilotConfirmationDialog::yesorno(tr("Your Tinygrad version is out of date and must be updated before downloading new models. Update now?"), this)) {
            if (FrogPilotConfirmationDialog::yesorno(tr("Updating Tinygrad will delete existing Tinygrad-based models and need to be re-downloaded. Proceed?"), this)) {
              params_memory.putBool("UpdateTinygrad", true);
              params_memory.put("ModelDownloadProgress", "Downloading...");

              updateTinygradBtn->setText(0, tr("CANCEL"));
              updateTinygradBtn->setValue(tr("Updating..."));

              updatingTinygrad = true;
            }
          }
        } else if (id == 0) {
          if (modelDownloading) {
            params_memory.putBool("CancelModelDownload", true);

            cancellingDownload = true;
          } else {
            QStringList downloadableModels = availableModelNames;
            for (const QString &modelKey : modelFileToNameMap.keys()) {
              QString modelName = modelFileToNameMap.value(modelKey);
              if (modelDir.exists(modelKey + ".thneed") || hasAllTinygradFiles(modelDir, modelKey)) {
                downloadableModels.removeAll(modelName);
              }
            }
            allModelsDownloaded = downloadableModels.isEmpty();

            QString modelToDownload = MultiOptionDialog::getSelection(tr("Select a driving model to download"), downloadableModels, "", this);
            if (!modelToDownload.isEmpty()) {
              params_memory.put("ModelToDownload", modelFileToNameMap.key(modelToDownload).toStdString());
              params_memory.put("ModelDownloadProgress", "Downloading...");

              downloadModelBtn->setText(0, tr("CANCEL"));

              downloadModelBtn->setValue("Downloading...");

              downloadModelBtn->setVisibleButton(1, false);

              modelDownloading = true;
            }
          }
        } else if (id == 1) {
          if (allModelsDownloading) {
            params_memory.putBool("CancelModelDownload", true);

            cancellingDownload = true;
          } else {
            params_memory.putBool("DownloadAllModels", true);
            params_memory.put("ModelDownloadProgress", "Downloading...");

            downloadModelBtn->setText(1, tr("CANCEL"));

            downloadModelBtn->setValue("Downloading...");

            downloadModelBtn->setVisibleButton(0, false);

            allModelsDownloading = true;
          }
        }
      });
      modelToggle = downloadModelBtn;
    } else if (param == "ManageBlacklistedModels") {
      FrogPilotButtonsControl *blacklistBtn = new FrogPilotButtonsControl(title, desc, icon, {tr("ADD"), tr("REMOVE"), tr("REMOVE ALL")});
      QObject::connect(blacklistBtn, &FrogPilotButtonsControl::buttonClicked, [this](int id) {
        QStringList blacklistedModels = QString::fromStdString(params.get("BlacklistedModels")).split(",");
        blacklistedModels.removeAll("");

        if (id == 0) {
          QStringList blacklistableModels;
          for (const QString &model : modelFileToNameMapProcessed.keys()) {
            if (!blacklistedModels.contains(model)) {
              blacklistableModels.append(modelFileToNameMapProcessed.value(model));
            }
          }

          if (blacklistableModels.size() <= 1) {
            ConfirmationDialog::alert(tr("There are no more models to blacklist! The only available model is \"%1\"!").arg(blacklistableModels.first()), this);
          } else {
            QString modelToBlacklist = MultiOptionDialog::getSelection(tr("Select a model to add to the blacklist"), blacklistableModels, "", this);
            if (!modelToBlacklist.isEmpty()) {
              if (ConfirmationDialog::confirm(tr("Are you sure you want to add the \"%1\" model to the blacklist?").arg(modelToBlacklist), tr("Add"), this)) {
                blacklistedModels.append(modelFileToNameMapProcessed.key(modelToBlacklist));

                params.put("BlacklistedModels", blacklistedModels.join(",").toStdString());
              }
            }
          }
        } else if (id == 1) {
          QStringList whitelistableModels;
          for (const QString &model : blacklistedModels) {
            QString modelName = modelFileToNameMapProcessed.value(model);
            whitelistableModels.append(modelName);
          }
          whitelistableModels.sort();

          QString modelToWhitelist = MultiOptionDialog::getSelection(tr("Select a model to remove from the blacklist"), whitelistableModels, "", this);
          if (!modelToWhitelist.isEmpty()) {
            if (ConfirmationDialog::confirm(tr("Are you sure you want to remove the \"%1\" model from the blacklist?").arg(modelToWhitelist), tr("Remove"), this)) {
              blacklistedModels.removeAll(modelFileToNameMapProcessed.key(modelToWhitelist));

              params.put("BlacklistedModels", blacklistedModels.join(",").toStdString());
            }
          }
        } else if (id == 2) {
          if (FrogPilotConfirmationDialog::yesorno(tr("Are you sure you want to remove all of your blacklisted models?"), this)) {
            params.remove("BlacklistedModels");
            params_cache.remove("BlacklistedModels");
          }
        }
      });
      modelToggle = blacklistBtn;
    } else if (param == "ManageScores") {
      FrogPilotButtonsControl *manageScoresBtn = new FrogPilotButtonsControl(title, desc, icon, {tr("RESET"), tr("VIEW")});
      QObject::connect(manageScoresBtn, &FrogPilotButtonsControl::buttonClicked, [this, modelLayout, modelLabelsList, modelLabelsPanel](int id) {
        if (id == 0) {
          if (FrogPilotConfirmationDialog::yesorno(tr("Are you sure you want to reset all of your model drives and scores?"), this)) {
            params.remove("ModelDrivesAndScores");
            params_cache.remove("ModelDrivesAndScores");
          }
        } else if (id == 1) {
          openSubPanel();

          updateModelLabels(modelLabelsList);

          modelLayout->setCurrentWidget(modelLabelsPanel);
        }
      });
      modelToggle = manageScoresBtn;
    } else if (param == "SelectModel") {
      selectModelBtn = new ButtonControl(title, tr("SELECT"), desc);
      QObject::connect(selectModelBtn, &ButtonControl::clicked, [this]() {
        QStringList selectableModels;
        for (const QString &modelKey : modelFileToNameMap.keys()) {
          QString modelName = modelFileToNameMap.value(modelKey);
          if (modelName.contains("(Default)")) {
            continue;
          }

          if (modelDir.exists(modelKey + ".thneed") || hasAllTinygradFiles(modelDir, modelKey)) {
            selectableModels.append(modelName);
          }
        }
        selectableModels.sort();
        selectableModels.prepend(modelFileToNameMap.value(normalizeModelKey(QString::fromStdString(params_default.get("Model")))));

        QString modelToSelect = MultiOptionDialog::getSelection(tr("Select a model - 🗺️ = Navigation | 📡 = Radar | 👀 = VOACC"), selectableModels, currentModel, this);
        if (!modelToSelect.isEmpty()) {
          currentModel = modelToSelect;

          params.put("Model", modelFileToNameMap.key(modelToSelect).toStdString());

          updateFrogPilotToggles();

          if (started) {
            if (FrogPilotConfirmationDialog::toggleReboot(this)) {
              Hardware::reboot();
            }
          }
          selectModelBtn->setValue(modelToSelect);

          QStringList deletableModels;
          for (const QString &file : modelDir.entryList(QDir::Files)) {
            QString base = QFileInfo(file).baseName();
            for (const QString &modelKey : modelFileToNameMapProcessed.keys()) {
              if (base.startsWith(modelKey)) {
                QString modelName = modelFileToNameMapProcessed.value(modelKey);
                if (!deletableModels.contains(modelName)) {
                  deletableModels.append(modelName);
                }
              }
            }
          }
          deletableModels.removeAll(processModelName(currentModel));
          deletableModels.removeAll(modelFileToNameMapProcessed.value(normalizeModelKey(QString::fromStdString(params_default.get("Model")))));
          noModelsDownloaded = deletableModels.isEmpty();
        }
      });
      modelToggle = selectModelBtn;

    } else if (param == "UpdateTinygrad") {
      updateTinygradBtn = new FrogPilotButtonsControl(title, desc, icon, {tr("UPDATE")});
      QObject::connect(updateTinygradBtn, &FrogPilotButtonsControl::buttonClicked, [this]() {
        if (updatingTinygrad) {
          params_memory.putBool("CancelModelDownload", true);

          updateTinygradBtn->setEnabled(false);
          updateTinygradBtn->setValue(tr("Cancelling..."));

          cancellingDownload = true;
        } else {
          if (FrogPilotConfirmationDialog::yesorno(tr("Updating Tinygrad will delete existing Tinygrad-based models and need to be re-downloaded. Proceed?"), this)) {
            params_memory.putBool("UpdateTinygrad", true);
            params_memory.put("ModelDownloadProgress", "Downloading...");

            updateTinygradBtn->setText(0, tr("CANCEL"));
            updateTinygradBtn->setValue(tr("Updating..."));

            updatingTinygrad = true;
          }
        }
      });
      modelToggle = updateTinygradBtn;

    } else {
      modelToggle = new ParamControl(param, title, desc, icon);
    }

    toggles[param] = modelToggle;

    modelList->addItem(modelToggle);

    QObject::connect(modelToggle, &AbstractControl::showDescriptionEvent, [this]() {
      update();
    });
  }

  QObject::connect(static_cast<ToggleControl*>(toggles["ModelRandomizer"]), &ToggleControl::toggleFlipped, [this](bool state) {
    updateToggles();

    if (state && !allModelsDownloaded) {
      if (FrogPilotConfirmationDialog::yesorno(tr("The \"Model Randomizer\" only works with downloaded models. Do you want to download all the driving models?"), this)) {
        params_memory.putBool("DownloadAllModels", true);
        params_memory.put("ModelDownloadProgress", "Downloading...");

        downloadModelBtn->setValue("Downloading...");

        allModelsDownloading = true;
      }
    }
  });

  QObject::connect(parent, &FrogPilotSettingsWindow::closeSubPanel, [modelLayout, modelPanel] {modelLayout->setCurrentWidget(modelPanel);});
  QObject::connect(uiState(), &UIState::uiUpdate, this, &FrogPilotModelPanel::updateState);
}

void FrogPilotModelPanel::showEvent(QShowEvent *event) {
  FrogPilotUIState &fs = *frogpilotUIState();
  UIState &s = *uiState();

  frogpilotToggleLevels = parent->frogpilotToggleLevels;
  tuningLevel = parent->tuningLevel;

  allModelsDownloading = params_memory.getBool("DownloadAllModels");
  modelDownloading = !params_memory.get("ModelDownloadProgress").empty();
  tinygradUpdate = params.getBool("TinygradUpdateAvailable");
  updatingTinygrad = params_memory.getBool("UpdateTinygrad");

  QStringList availableModels = QString::fromStdString(params.get("AvailableModels")).split(",");
  availableModels.sort();
  availableModelNames = QString::fromStdString(params.get("AvailableModelNames")).split(",");
  availableModelNames.sort();

  modelFileToNameMap.clear();
  modelFileToNameMapProcessed.clear();
  int size = qMin(availableModels.size(), availableModelNames.size());
  for (int i = 0; i < size; ++i) {
    modelFileToNameMap.insert(availableModels[i], availableModelNames[i]);
    modelFileToNameMapProcessed.insert(availableModels[i], processModelName(availableModelNames[i]));
  }

  QStringList downloadableModels = availableModelNames;
  for (const QString &modelKey : modelFileToNameMap.keys()) {
    QString modelName = modelFileToNameMap.value(modelKey);
    if (modelDir.exists(modelKey + ".thneed") || hasAllTinygradFiles(modelDir, modelKey)) {
      downloadableModels.removeAll(modelName);
    }
  }
  allModelsDownloaded = downloadableModels.isEmpty();

  QStringList deletableModels;
  for (const QString &file : modelDir.entryList(QDir::Files)) {
    QString base = QFileInfo(file).baseName();
    for (const QString &modelKey : modelFileToNameMapProcessed.keys()) {
      if (base.startsWith(modelKey)) {
        QString modelName = modelFileToNameMapProcessed.value(modelKey);
        if (!deletableModels.contains(modelName)) {
          deletableModels.append(modelName);
        }
      }
    }
  }
  deletableModels.removeAll(processModelName(currentModel));
  deletableModels.removeAll(modelFileToNameMapProcessed.value(normalizeModelKey(QString::fromStdString(params_default.get("Model")))));
  noModelsDownloaded = deletableModels.isEmpty();

  QString modelKey = normalizeModelKey(QString::fromStdString(params.get("Model")));
  if (!modelDir.exists(modelKey + ".thneed") && !hasAllTinygradFiles(modelDir, modelKey)) {
    modelKey = normalizeModelKey(QString::fromStdString(params_default.get("Model")));
  }
  currentModel = modelFileToNameMap.value(modelKey);
  selectModelBtn->setValue(currentModel);

  bool parked = !s.scene.started || fs.frogpilot_scene.parked || fs.frogpilot_toggles.value("frogs_go_moo").toBool();

  deleteModelBtn->setEnabled(!(allModelsDownloading || modelDownloading || noModelsDownloaded));

  downloadModelBtn->setEnabledButtons(0, !allModelsDownloaded && !allModelsDownloading && !cancellingDownload && !updatingTinygrad && fs.frogpilot_scene.online && parked);
  downloadModelBtn->setEnabledButtons(1, !allModelsDownloaded && !modelDownloading && !cancellingDownload && !updatingTinygrad && fs.frogpilot_scene.online && parked);

  downloadModelBtn->setValue(fs.frogpilot_scene.online ? (parked ? "" : "Not parked") : tr("Offline..."));

  updateTinygradBtn->setEnabled(!modelDownloading && !cancellingDownload && fs.frogpilot_scene.online && parked && tinygradUpdate);
  updateTinygradBtn->setValue(tinygradUpdate ? tr("Update available!") : tr("Up to date!"));

  started = s.scene.started;

  updateToggles();
}

void FrogPilotModelPanel::updateState(const UIState &s, const FrogPilotUIState &fs) {
  if (!isVisible() || finalizingDownload) {
    return;
  }

  bool parked = !started || fs.frogpilot_scene.parked || fs.frogpilot_toggles.value("frogs_go_moo").toBool();

  if (allModelsDownloading || modelDownloading) {
    QString progress = QString::fromStdString(params_memory.get("ModelDownloadProgress"));
    bool downloadFailed = progress.contains(QRegularExpression("cancelled|exists|failed|missing|offline", QRegularExpression::CaseInsensitiveOption));

    if (progress != "Downloading...") {
      downloadModelBtn->setValue(progress);
    }

    if (progress == "All models downloaded!" && allModelsDownloading || progress == "Downloaded!" && modelDownloading || downloadFailed) {
      finalizingDownload = true;

      QTimer::singleShot(2500, [this, progress]() {
        allModelsDownloaded = progress == "All models downloaded!";
        allModelsDownloading = false;
        cancellingDownload = false;
        finalizingDownload = false;
        modelDownloading = false;
        noModelsDownloaded = false;

        params_memory.remove("ModelDownloadProgress");

        downloadModelBtn->setEnabled(true);
        downloadModelBtn->setValue("");
      });
    }
  } else {
    downloadModelBtn->setValue(fs.frogpilot_scene.online ? (parked ? "" : "Not parked") : tr("Offline..."));
  }

  if (updatingTinygrad) {
    QString progress = QString::fromStdString(params_memory.get("ModelDownloadProgress"));
    bool downloadFailed = progress.contains(QRegularExpression("cancelled|exists|failed|missing|offline", QRegularExpression::CaseInsensitiveOption));

    if (progress != "Downloading...") {
      updateTinygradBtn->setValue(progress);
    }

    if (progress == "Updated!" && updatingTinygrad || downloadFailed) {
      finalizingDownload = true;

      QTimer::singleShot(2500, [this, progress]() {
        modelDownloading = !params_memory.get("ModelDownloadProgress").empty();
        if (modelDownloading) {
          downloadModelBtn->setText(1, tr("CANCEL"));

          downloadModelBtn->setValue("Downloading...");

          downloadModelBtn->setVisibleButton(0, false);
        } else {
          cancellingDownload = false;

          params_memory.remove("ModelDownloadProgress");
        }

        tinygradUpdate = params.getBool("TinygradUpdateAvailable");

        finalizingDownload = false;
        updatingTinygrad = false;

        updateTinygradBtn->setEnabled(tinygradUpdate);
        updateTinygradBtn->setText(0, tr("UPDATE"));
        updateTinygradBtn->setValue(tinygradUpdate ? tr("Update available!") : tr("Up to date!"));
      });
    }
  }

  deleteModelBtn->setEnabled(!(allModelsDownloading || modelDownloading || noModelsDownloaded));

  downloadModelBtn->setText(0, modelDownloading ? tr("CANCEL") : tr("DOWNLOAD"));
  downloadModelBtn->setText(1, allModelsDownloading ? tr("CANCEL") : tr("DOWNLOAD ALL"));

  downloadModelBtn->setEnabledButtons(0, !allModelsDownloaded && !allModelsDownloading && !cancellingDownload && !finalizingDownload && !updatingTinygrad && fs.frogpilot_scene.online && parked);
  downloadModelBtn->setEnabledButtons(1, !allModelsDownloaded && !modelDownloading && !cancellingDownload && !finalizingDownload && !updatingTinygrad && fs.frogpilot_scene.online && parked);

  downloadModelBtn->setVisibleButton(0, !allModelsDownloading);
  downloadModelBtn->setVisibleButton(1, !modelDownloading);

  updateTinygradBtn->setEnabled(!modelDownloading && !cancellingDownload && !cancellingDownload && !finalizingDownload && fs.frogpilot_scene.online && parked && tinygradUpdate);

  started = s.scene.started;

  parent->keepScreenOn = allModelsDownloading || modelDownloading || updatingTinygrad;
}

void FrogPilotModelPanel::updateModelLabels(FrogPilotListWidget *labelsList) {
  labelsList->clear();

  QJsonObject modelDrivesAndScores = QJsonDocument::fromJson(QString::fromStdString(params.get("ModelDrivesAndScores")).toUtf8()).object();

  for (const QString &modelName : availableModelNames) {
    QJsonObject modelData = modelDrivesAndScores.value(processModelName(modelName)).toObject();

    int drives = modelData.value("Drives").toInt(0);
    int score = modelData.value("Score").toInt(0);

    QString drivesDisplay = drives == 1 ? QString("%1 Drive").arg(drives) : drives > 0 ? QString("%1 Drives").arg(drives) : "N/A";
    QString scoreDisplay = drives > 0 ? QString("Score: %1%").arg(score) : "N/A";

    QString labelTitle = processModelName(modelName);
    QString labelText = QString("%1 (%2)").arg(scoreDisplay, drivesDisplay);

    LabelControl *labelControl = new LabelControl(labelTitle, labelText, "", this);
    labelsList->addItem(labelControl);
  }
}

void FrogPilotModelPanel::updateToggles() {
  for (auto &[key, toggle] : toggles) {
    bool setVisible = tuningLevel >= frogpilotToggleLevels[key].toDouble();

    if (key == "ManageBlacklistedModels" || key == "ManageScores") {
      setVisible &= params.getBool("ModelRandomizer");
    }

    else if (key == "SelectModel") {
      setVisible &= !params.getBool("ModelRandomizer");
    }

    toggle->setVisible(setVisible);
  }

  update();
}
