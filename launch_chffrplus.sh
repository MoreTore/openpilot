#!/usr/bin/bash

if [ -z "$BASEDIR" ]; then
  BASEDIR="/data/openpilot"
fi

source "$BASEDIR/launch_env.sh"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

 # --- BEGIN: auto-fix persist from squashfs -> ext4 (one-time, preserves identity) ---
function persist_convert_if_needed {
  LOG="/tmp/persist_fix.log"
  {
    echo "[persist-fix] ----- start $(date) -----"

    # Guard: if we've already converted on this device, skip
    if [ -f /data/.persist_converted ] || [ -f /persist/.converted_to_ext4 ]; then
      echo "[persist-fix] Marker exists; skipping."
      return 0
    fi

    # Discover persist device
    DEV="$(blkid -t LABEL=persist -o device 2>/dev/null || true)"
    if [ -z "$DEV" ] && [ -e /dev/disk/by-partlabel/persist ]; then
      DEV="$(realpath /dev/disk/by-partlabel/persist)"
    fi
    if [ -z "$DEV" ]; then
      # Fallback commonly used path
      if lsblk -no PATH,LABEL | grep -E "persist$" >/dev/null 2>&1; then
        DEV="$(lsblk -no PATH,LABEL | awk '$2=="persist"{print $1; exit}')"
      else
        DEV="/dev/sda2"
      fi
    fi
    echo "[persist-fix] Using device: ${DEV}"

    # Detect device filesystem type
    FSTYPE="$(blkid -o value -s TYPE "$DEV" 2>/dev/null || true)"
    echo "[persist-fix] Detected $DEV fstype='$FSTYPE'"
    UNSQS="$DIR/third_party/bin/unsquashfs"

    # Detect current /persist mount status & fstype
    CUR_MNT_TYPE="$(mount | awk '$3==\"/persist\"{print $5}' | head -n1)"
    if mountpoint -q /persist; then
      echo "[persist-fix] /persist currently mounted as type='${CUR_MNT_TYPE}'"
    else
      echo "[persist-fix] /persist is not a mountpoint (empty dir)."
    fi

    # Only proceed on NEW devices: either /persist is not mounted OR is squashfs,
    # AND the underlying persist partition is squashfs (factory RO image).
    if { ! mountpoint -q /persist || [ "${CUR_MNT_TYPE}" = "squashfs" ]; } && [ "${FSTYPE}" = "squashfs" ]; then
      echo "[persist-fix] NEW device detected (squashfs persist). Converting to ext4."

      # Stream identity files using unsquashfs -cat directly
      for f in id_rsa id_rsa.pub color_cal dongle_id; do
        if sudo "$UNSQS" -cat "$DEV" "comma/$f" > "/data/$f" 2>/dev/null; then
          echo "[persist-fix] Preserved $f"
        else
          echo "[persist-fix] $f not found in squashfs (ok, may not exist)."
        fi
      done

      # (Optional) raw backup of the original partition image (once)
      if [ ! -f /data/persist_backup.img ]; then
        echo "[persist-fix] Creating raw backup of ${DEV} to /data/persist_backup.img"
        sudo dd if="$DEV" of=/data/persist_backup.img bs=1M status=none || true
        sync
      fi

      # Reformat persist as ext4 and label it
      echo "[persist-fix] Formatting ${DEV} as ext4..."
      if ! sudo mkfs.ext4 -F "$DEV"; then
        echo "[persist-fix][ERROR] mkfs.ext4 failed on ${DEV}. Aborting."
        return 0
      fi
      sudo e2label "$DEV" persist || true

      # Mount the new ext4 persist RW
      if ! sudo mount -t ext4 -o rw,discard "$DEV" /persist; then
        echo "[persist-fix][ERROR] Failed to mount new ext4 persist. Aborting."
        return 0
      fi

      # Recreate expected structure and restore identity
      sudo mkdir -p /persist/{comma,params,tracking}
      sudo chmod 755 /persist /persist/{comma,params,tracking}
      for f in id_rsa id_rsa.pub color_cal dongle_id; do
        if [ -f "/data/$f" ]; then
          sudo cp -p "/data/$f" "/persist/comma/$f"
        fi
      done
      if [ -f /persist/comma/id_rsa ]; then sudo chmod 600 /persist/comma/id_rsa; fi
      sudo chown -R comma:comma /persist/ /persist/comma /persist/params /persist/tracking || true
      sudo touch /persist/tracking/.lock

      # Seed minimal params (only if not present)
      [ -f /persist/params/HasAcceptedTerms ] || echo -n 1 | sudo tee /persist/params/HasAcceptedTerms >/dev/null
      [ -f /persist/params/AlwaysOnDM ]      || echo -n 1 | sudo tee /persist/params/AlwaysOnDM      >/dev/null

      # Mark complete so we don't run again; reboot once
      echo "ext4" | sudo tee /persist/.converted_to_ext4 >/dev/null
      sudo touch /data/.persist_converted
      sync
      echo "[persist-fix] Conversion complete. Rebooting once to finalize..."
      sudo reboot
    else
      echo "[persist-fix] Device is not NEW/squashfs (persist already ext4 or mounted). Skipping."
    fi

    echo "[persist-fix] ----- end $(date) -----"
  } >>"$LOG" 2>&1
}
# --- END: auto-fix persist from squashfs -> ext4 ---

function agnos_init {
  persist_convert_if_needed
  # TODO: move this to agnos
  sudo rm -f /data/etc/NetworkManager/system-connections/*.nmmeta

  # set success flag for current boot slot
  sudo abctl --set_success

  # TODO: do this without udev in AGNOS
  # udev does this, but sometimes we startup faster
  sudo chgrp gpu /dev/adsprpc-smd /dev/ion /dev/kgsl-3d0
  sudo chmod 660 /dev/adsprpc-smd /dev/ion /dev/kgsl-3d0
  sudo chmod 0777 /cache

  # Check if AGNOS update is required
  if [ $(< /VERSION) != "$AGNOS_VERSION" ]; then
    AGNOS_PY="$DIR/system/hardware/tici/agnos.py"
    MANIFEST="$DIR/system/hardware/tici/agnos.json"
    if $AGNOS_PY --verify $MANIFEST; then
      sudo reboot
    fi
    $DIR/system/hardware/tici/updater $AGNOS_PY $MANIFEST
  fi
}

function launch {
  # Remove orphaned git lock if it exists on boot
  [ -f "$DIR/.git/index.lock" ] && rm -f $DIR/.git/index.lock

  # Check to see if there's a valid overlay-based update available. Conditions
  # are as follows:
  #
  # 1. The BASEDIR init file has to exist, with a newer modtime than anything in
  #    the BASEDIR Git repo. This checks for local development work or the user
  #    switching branches/forks, which should not be overwritten.
  # 2. The FINALIZED consistent file has to exist, indicating there's an update
  #    that completed successfully and synced to disk.

  if [ -f "${BASEDIR}/.overlay_init" ]; then
    find ${BASEDIR}/.git -newer ${BASEDIR}/.overlay_init | grep -q '.' 2> /dev/null
    if [ $? -eq 0 ]; then
      echo "${BASEDIR} has been modified, skipping overlay update installation"
    else
      if [ -f "${STAGING_ROOT}/finalized/.overlay_consistent" ]; then
        if [ ! -d /data/safe_staging/old_openpilot ]; then
          echo "Valid overlay update found, installing"
          LAUNCHER_LOCATION="${BASH_SOURCE[0]}"

          mv $BASEDIR /data/safe_staging/old_openpilot
          mv "${STAGING_ROOT}/finalized" $BASEDIR
          cd $BASEDIR

          echo "Restarting launch script ${LAUNCHER_LOCATION}"
          unset AGNOS_VERSION
          exec "${LAUNCHER_LOCATION}"
        else
          echo "openpilot backup found, not updating"
          # TODO: restore backup? This means the updater didn't start after swapping
        fi
      fi
    fi
  fi

  # handle pythonpath
  ln -sfn $(pwd) /data/pythonpath
  export PYTHONPATH="$PWD"

  # hardware specific init
  if [ -f /AGNOS ]; then
    agnos_init
  fi

  # write tmux scrollback to a file
  tmux capture-pane -pq -S-1000 > /tmp/launch_log

  # start manager
  cd system/manager
  if [ ! -f $DIR/prebuilt ]; then
    ./build.py
  fi
  ./manager.py

  # if broken, keep on screen error
  while true; do sleep 1; done
}

launch
