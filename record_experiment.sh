#!/bin/bash

EXP_NAME=$1

if [ -z "$EXP_NAME" ]; then
    echo "[Error] Please provide an experiment name."
    echo "  Usage: ./record_experiment.sh <user>_<ctrl><task><trial>[_rX]"
    echo "    <user>: 01, 02, ..., 99"
    echo "    <ctrl>: m (manual control), s (shared control)"
    echo "    <task>: s (single item), m (multiple items), c (complete task)"
    echo "    <trial>: 01, 02, ..., 99"
    echo "    [_rX]: (optional) X-th repeat of the same trial (e.g., _r1, _r2)"
    echo "  Example: "
    echo "    ./record_experiment.sh 01_ms01"
    echo "    ./record_experiment.sh 02_sm03_r2"
    exit 0
else
    DATE_STR=$(date +%Y-%m-%d)
    SAVE_DIR=bags/$DATE_STR
    mkdir -p $SAVE_DIR

    BASE_NAME=${EXP_NAME}
    BAG_PATH=$SAVE_DIR/${BASE_NAME}.bag

    if [ -f "$BAG_PATH" ]; then
        idx=1
        while true; do
            BAG_PATH=$SAVE_DIR/${BASE_NAME}_r${idx}.bag
            if [ ! -f "$BAG_PATH" ]; then
                break
            fi
            idx=$((idx + 1))
        done
    fi

    echo "Start recording bag file $(basename $BAG_PATH):"
    rosbag record \
        -O $BAG_PATH \
        /camera/color/image_raw/compressed \
        /gripper/cmd_gripper \
        /marslite_control/gripper_pose \
        /marslite_control/objects_with_belief \
        /marslite_control/record_signal \
        /marslite_control/robot_state \
        /target_frame \
        /tf \
        /tf_static \
        /vr_controller/joy 
    echo "...Bag file stored in $BAG_PATH."
    sleep 3
    echo ""
    echo "Start analyzing bag file $(basename $BAG_PATH):"
    python3 analyze_bagfile.py $BAG_PATH
fi

