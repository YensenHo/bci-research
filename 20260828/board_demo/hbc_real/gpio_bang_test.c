/*
 * gpio_bang_test.c — RV1106 GPIO 位翻转速度实测（修正版）
 *
 * 【寄存器布局勘误（板上实测 2026-08-29）】
 *   RV1106 GPIO 是 rockchip GPIO v2 布局，32bit 寄存器按低16位/高16位拆分：
 *     SWPORT_DR_L  = 0x00  (数据，bit0-15)
 *     SWPORT_DR_H  = 0x04  (数据，bit16-31)
 *     SWPORT_DDR_L = 0x08  (方向，1=输出，bit0-15)   <-- 之前误写成 0x04
 *     SWPORT_DDR_H = 0x0c  (方向，bit16-31)
 *     SWPORT_EXT_L = 0x50  (输入回读，bit0-15)
 *     SWPORT_EXT_H = 0x54  (输入回读，bit16-31)
 *   实测佐证：sysfs 设 gpio32(GPIO1_A0) 输出高后，0x00 与 0x08 的 bit0 同时置位。
 *
 * 测 GPIO1_A0 (header 引脚，gpio 32，MUX/GPIO 均 unclaimed)：
 *   ①整字写 DR_L 翻转  ②读-改-写 DR_L 翻转  ③单写吞吐
 *   并用 EXT_L 回读验证引脚电平真实翻转（证明 pad 被驱动）。
 */
#include <stdio.h>
#include <stdint.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <time.h>

#define GPIO1_BASE 0xff530000UL
#define DR_L  0x00
#define DDR_L 0x08
#define EXT_L 0x50

static volatile uint32_t *gpio;
static double now_ns(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

int main(void) {
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) { perror("open /dev/mem"); return 1; }
    gpio = mmap(NULL, 0x100, PROT_READ | PROT_WRITE, MAP_SHARED, fd, GPIO1_BASE);
    if (gpio == MAP_FAILED) { perror("mmap gpio1"); return 1; }

    printf("=== RV1106 GPIO bit-bang test v2 (GPIO1_A0, bank @ 0xff530000, v2 regs) ===\n");
    uint32_t ddr0 = gpio[DDR_L / 4];
    printf("DDR_L 初始 = 0x%08x (bit0: %s)\n", ddr0, (ddr0 & 1) ? "输出" : "输入");

    /* 置 GPIO1_A0 为输出（RMW，不动其它位）并验证写生效 */
    gpio[DDR_L / 4] = ddr0 | 1u;
    uint32_t ddr1 = gpio[DDR_L / 4];
    printf("DDR_L 写入后 = 0x%08x (bit0: %s%s)\n", ddr1, (ddr1 & 1) ? "输出" : "输入",
           (ddr1 & 1) ? "" : "  <-- 写入未生效!");

    /* 先拉高，验证 EXT 回读跟随（证明 pad 被驱动） */
    gpio[DR_L / 4] = 1u;
    usleep(10000);
    printf("DR_L=1 后 EXT_L = 0x%08x (bit0: %s)\n", gpio[EXT_L / 4],
           (gpio[EXT_L / 4] & 1) ? "高" : "低");
    gpio[DR_L / 4] = 0u;
    usleep(10000);
    printf("DR_L=0 后 EXT_L = 0x%08x (bit0: %s)\n", gpio[EXT_L / 4],
           (gpio[EXT_L / 4] & 1) ? "高" : "低");

    const long N = 2000000; /* 2M 个周期 = 4M 次写 */

    /* ① 整字写 DR_L（最快，但会同时改同 bank 低 16 位其它引脚输出锁存） */
    double t0 = now_ns();
    for (long i = 0; i < N; i++) {
        gpio[DR_L / 4] = 1u;
        gpio[DR_L / 4] = 0u;
    }
    double t1 = now_ns();
    double whole_period = (t1 - t0) / N;
    printf("\n[1] 整字写 DR_L 翻转: %ld 周期 %.3f ms -> 每周期 %.1f ns -> 翻转频率 %.3f MHz\n",
           N, (t1 - t0) / 1e6, whole_period, 1000.0 / whole_period);

    /* ② 读-改-写（只动 bit0，真实 bit-bang 常用写法） */
    t0 = now_ns();
    for (long i = 0; i < N; i++) {
        uint32_t v = gpio[DR_L / 4];
        gpio[DR_L / 4] = (v & ~1u) | 1u;
        v = gpio[DR_L / 4];
        gpio[DR_L / 4] = v & ~1u;
    }
    t1 = now_ns();
    double rmw_period = (t1 - t0) / N;
    printf("[2] 读-改-写 DR_L 翻转: %ld 周期 %.3f ms -> 每周期 %.1f ns -> 翻转频率 %.3f MHz\n",
           N, (t1 - t0) / 1e6, rmw_period, 1000.0 / rmw_period);

    /* ③ 纯寄存器写吞吐（每周期 1 次写，理论上限） */
    t0 = now_ns();
    for (long i = 0; i < N; i++) gpio[DR_L / 4] = (uint32_t)(i & 1);
    t1 = now_ns();
    double wr_period = (t1 - t0) / N;
    printf("[3] 单写吞吐: %ld 次写 %.3f ms -> 每次 %.1f ns -> %.3f M写/秒\n",
           N, (t1 - t0) / 1e6, wr_period, 1000.0 / wr_period);

    /* 翻转后 EXT 回读抽查：确认引脚真在翻（在翻转循环结束后看当前电平） */
    printf("\n翻转结束后 EXT_L = 0x%08x (bit0: %s)\n", gpio[EXT_L / 4],
           (gpio[EXT_L / 4] & 1) ? "高" : "低");

    /* 恢复 DDR（RMW 还原） */
    gpio[DDR_L / 4] = ddr0;
    printf("DDR_L 已恢复 = 0x%08x\n", gpio[DDR_L / 4]);

    printf("\n结论: 1MHz bit-bang 需要每周期 <=1000ns (写翻转 <=500ns/次)。\n");
    munmap((void *)gpio, 0x100);
    close(fd);
    return 0;
}
