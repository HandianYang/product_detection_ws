#!/bin/bash

set -euo pipefail
command -v rosbag >/dev/null 2>&1 || { echo "rosbag not found";  }
command -v python3 >/dev/null 2>&1 || { echo "python3 not found"; }

EXP_NAME=$1

if [ -z "${EXP_NAME}" ]; then
    echo "[Error] Please provide an experiment name."
    echo "  Usage: ./record_experiment.sh P<user>_<ctrl><task><trial>[_rX]"
    echo "    <user>: 01, 02, ..., 99"
    echo "    <ctrl>: m (manual control), s (shared control)"
    echo "    <task>: s (single item), m (multiple items), r (replenishment task)"
    echo "    <trial>: 01, 02, ..., 99"
    echo "  Example: "
    echo "    ./record_experiment.sh P01_ms01 (manual-single)"
    echo "    ./record_experiment.sh P02_sm03 (shared-multiple)"
    exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BAGFILE_DIR="${SCRIPT_DIR}/../bags"
mkdir -p "${BAGFILE_DIR}"
BAG_PATH="${BAGFILE_DIR}/${EXP_NAME}.bag"
BAG_NAME="$(basename "$BAG_PATH")"

echo "Start recording bag file ${BAG_NAME}:"
rosbag record \
    -O ${BAG_PATH} \
    /camera/color/image_raw/compressed \
    /gripper/cmd_gripper \
    /marslite_control/gripper_pose \
    /marslite_control/objects_with_belief \
    /marslite_control/record_signal \
    /marslite_control/restart_attempt_signal \
    /marslite_control/robot_state \
    /target_frame \
    /tf \
    /tf_static \
    /vr_controller/joy 
echo "...Bag file stored in ${BAG_PATH}."
echo ""
echo "Start analyzing bag file ${BAG_PATH}:"
python3 "${SCRIPT_DIR}/analyze_bagfile.py" "${BAG_PATH}"