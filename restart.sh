#!/bin/bash
echo "INFO:Please running this shell in its root path,like ~/infantry_xx,or you will get error in file path initialize"
# sleep 2
source /opt/intel/openvino_2021/bin/setupvars.sh
cd /home/xjturm/DX_aimbot_test/build/
BaudRate=115200
while true
do
    name=`ls /dev/| grep ACM`
    echo "Port Detect:$name"
    > /dev/$name
    if ! `ps aux | grep -v grep | grep /home/xjturm/DX_aimbot_test/build/infantry_new`
    then
            echo 'bfjg' | sudo -S chmod 666 /dev/ttyACM0
            /home/xjturm/DX_aimbot/build/infantry_new /dev/$name $BaudRate
    fi
    echo "Unexpected Dump"
    sleep 0.2
done