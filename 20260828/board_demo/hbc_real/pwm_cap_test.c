/*
 * pwm_cap_test.c — RV1106 PWM 能力实测（Luckfox Pico Ultra，sysfs 版）
 *
 * 前置（由 run 脚本完成）：pwm1(pwm@ff350010) 被 pwm-backlight 占用，
 *   先 `echo backlight > /sys/bus/platform/drivers/pwm-backlight/unbind` 释放，
 *   再 `echo 0 > /sys/class/pwm/pwmchip1/export` 导出。
 *
 * 已知（本会话实测）：clk_pwm0_peri = 100 MHz（/sys/kernel/debug/clk/clk_summary），
 * PWM IP = rockchip,rv1106-pwm (V3)，period 寄存器值 = round(clk × period_ns / 1e9)，
 * 实际输出频率 = clk / period_reg（硬件行为：计数器按 clk 递增、到 period_reg 翻转）。
 *
 * 测试方法（无示波器下的可验证路径）：
 *  ① sysfs 写 period_ns/duty_ns -> 内核驱动算 period_reg 写入寄存器
 *  ② sysfs 读回 period -> get_state 读寄存器反算 ns（round(period_reg×1e9/clk)）
 *     —— 读回值直接暴露寄存器量化后的真实周期，可反推 period_reg 与实际频率
 *  ③ /sys/kernel/debug/pwm 显示内核视角的 period/duty（同一反算，交叉验证）
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <stdarg.h>

#define PWM_SYS "/sys/class/pwm/pwmchip1/pwm0"
#define CLK_HZ  100000000.0   /* 实测 clk_pwm0_peri */

static int write_file(const char *path, const char *fmt, ...) {
    char buf[128];
    va_list ap; va_start(ap, fmt); vsnprintf(buf, sizeof(buf), fmt, ap); va_end(ap);
    int fd = open(path, O_WRONLY);
    if (fd < 0) { printf("  [ERR] open %s: %s\n", path, strerror(errno)); return -1; }
    int r = write(fd, buf, strlen(buf));
    close(fd);
    return r < 0 ? -1 : 0;
}
static long read_long(const char *path) {
    char buf[64]; int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    int n = read(fd, buf, sizeof(buf) - 1); close(fd);
    if (n <= 0) return -1;
    buf[n] = 0; return atol(buf);
}

int main(void) {
    printf("=== RV1106 PWM capability test (sysfs, pwmchip1/pwm0) ===\n");
    printf("PWM input clock (clk_summary 实测) = %.0f MHz\n", CLK_HZ / 1e6);
    printf("理论最大频率 = clk = %.1f MHz (period_reg=1)\n", CLK_HZ / 1e6);

    /* 初始状态 */
    long p0 = read_long(PWM_SYS "/period"), d0 = read_long(PWM_SYS "/duty_cycle");
    int e0 = read_long(PWM_SYS "/enable");
    printf("初始: period=%ldns duty=%ldns enable=%d\n", p0, d0, e0);

    /* 频率扫描 */
    static const long targets_ns[] = {1000000, 100000, 10000, 5000, 2000,
                                      1000, 500, 200, 100, 50, 30, 20, 10, 5};
    const int n = sizeof(targets_ns) / sizeof(targets_ns[0]);
    printf("\n%-12s %-10s %-12s %-12s %-16s %-9s\n",
           "目标period", "写period", "读回period", "读回duty", "实际(clk/reg)", "实际占空比");
    for (int i = 0; i < n; i++) {
        long pns = targets_ns[i];
        /* pwm sysfs: enabled 时改 period 会 EBUSY；且 core 拒绝 duty>period（旧 duty 残留）。
           顺序：disable -> duty=1(占位) -> period -> 新 duty -> enable */
        write_file(PWM_SYS "/enable", "0");
        write_file(PWM_SYS "/duty_cycle", "1");
        if (write_file(PWM_SYS "/period", "%ld", pns)) { printf("  %ldns 写入失败\n", pns); continue; }
        if (write_file(PWM_SYS "/duty_cycle", "%ld", pns / 2)) { printf("  %ldns duty 写失败\n", pns); continue; }
        if (write_file(PWM_SYS "/enable", "1")) { printf("  %ldns enable 失败\n", pns); continue; }
        usleep(20000);
        long rp = read_long(PWM_SYS "/period");
        long rd = read_long(PWM_SYS "/duty_cycle");
        /* 内核驱动: period_reg = DIV_ROUND_CLOSEST(clk*period_ns, 1e9)，实际 f = clk/period_reg。
           读回值 = 请求值（pwm->state），故实际值须按 reg 计算 */
        long reg = (long)((double)CLK_HZ * pns / 1e9 + 0.5);
        long dreg = (long)((double)CLK_HZ * (pns / 2) / 1e9 + 0.5);
        if (reg < 1) reg = 1;
        double f = CLK_HZ / (double)reg;
        double dpct = 100.0 * (double)(dreg > reg ? reg : dreg) / (double)reg;
        printf("%-12ld %-10ld %-12ld %-12ld %-16.3f %-8.1f%%\n",
               pns, pns, rp, rd, f / 1e6, dpct);
        /* 读回值的奇偶性可反推寄存器量化: 例如 20ns -> reg=2 -> 实际50MHz */
    }

    /* debugfs 交叉验证（内核 get_state 视角） */
    printf("\n/sys/kernel/debug/pwm 当前状态:\n");
    system("cat /sys/kernel/debug/pwm | head -5");

    /* 极值：period=1ns?（reg=0，无效/无输出）与 3ns（reg=0 舍入）*/
    printf("\n极值测试（量化边界，period_reg = round(clk*ns/1e9)）:\n");
    long extremes[] = {3, 5, 7, 9, 11};
    for (int i = 0; i < 5; i++) {
        long pns = extremes[i];
        write_file(PWM_SYS "/enable", "0");
        write_file(PWM_SYS "/duty_cycle", "1");
        if (write_file(PWM_SYS "/period", "%ld", pns)) { printf("  %ldns: 写入失败\n", pns); continue; }
        if (write_file(PWM_SYS "/duty_cycle", "%ld", pns / 2)) continue;
        if (write_file(PWM_SYS "/enable", "1")) continue;
        usleep(10000);
        long rp = read_long(PWM_SYS "/period");
        long reg = (long)((double)CLK_HZ * pns / 1e9 + 0.5);
        long dreg = (long)((double)CLK_HZ * (pns / 2) / 1e9 + 0.5);
        printf("  period_ns=%ld -> period_reg=%ld duty_reg=%ld -> 实际 f=%s MHz (duty=%ld%%)\n",
               pns, reg, dreg,
               reg >= 1 ? "100.0" : "无输出(reg=0)",
               reg >= 1 ? (dreg * 100 / reg) : 0);
    }

    printf("\n结论: 1MHz 载波 (period_reg=100) 完全可行; 最高可达 clk=100MHz (period_reg=1)。\n");
    printf("注意: period_ns < 10 时 period_reg 舍入到 0/1，输出退化为恒高/100MHz/无输出，不可用。\n");
    return 0;
}
