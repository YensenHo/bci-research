#!/bin/sh
# run_pwm_test.sh — 板载 PWM 能力测试（含释放/恢复 backlight）
# 用法: adb shell "sh /root/run_pwm_test.sh"
set -x
# 1. 释放 backlight 对 pwm1 的占用
echo backlight > /sys/bus/platform/drivers/pwm-backlight/unbind 2>/dev/null
sleep 0.2
# 2. 导出 pwmchip1 ch0（若已导出会报错，忽略）
echo 0 > /sys/class/pwm/pwmchip1/export 2>/dev/null
sleep 0.2
# 3. 跑测试
/root/pwm_cap_test
# 4. 恢复：注销导出，重新绑定 backlight（恢复出厂状态）
echo 0 > /sys/class/pwm/pwmchip1/unexport 2>/dev/null
sleep 0.2
echo backlight > /sys/bus/platform/drivers/pwm-backlight/bind 2>/dev/null
sleep 0.2
echo "--- backlight 已恢复: $(ls /sys/class/backlight/ 2>/dev/null)"
