/*
 * adc_cap_test.c — RV1106 SARADC 能力实测（Luckfox Pico Ultra）
 *
 * 板上 SARADC: saradc@ff3c0000, 2 通道 x 10bit, vref=1.8V (scale 1.7578125 mV/LSB)
 *  - ch0 (in_voltage0_raw) = GPIO4_C0/SARADC_M0, 被 adc-keys 按键电路占用（实测恒 1023）
 *  - ch1 (in_voltage1_raw) = GPIO4_C1/SARADC_M1, 空闲（浮空，实测噪声 200~450）
 *
 * 测两部分：
 *  A) sysfs IIO 读：N 次连续读 in_voltage1_raw -> 实际采样率(SPS)、噪声统计
 *  B) /dev/mem 直接寄存器：复刻内核驱动转换序列 -> 单次转换耗时、最大 kSPS
 *
 * SARADC2 寄存器（base 0xff3c0000）：
 *  CONV_CON=0x00  T_PD_SOC=0x04  T_DAS_SOC=0x0c  END_INT_EN=0x104
 *  ST_CON=0x108  STATUS=0x10c  END_INT_ST=0x110  DATA_BASE=0x120 (+ch*4)
 *  转换序列（rockchip_saradc_start_v2）：T_DAS=0xc T_PD=0x20 END_INT_EN=0x10001
 *  CONV_CON = (START|SINGLE|chn)<<16 | (START|SINGLE|chn)；START=bit4 SINGLE=bit5
 *  完成标志：END_INT_ST bit0（写 1 清）；数据在 DATA_BASE+chn*4（10bit）
 */
#include <stdio.h>
#include <stdint.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>

#define SARADC_BASE 0xff3c0000UL
#define CONV_CON    0x00
#define T_PD_SOC    0x04
#define T_DAS_SOC   0x0c
#define END_INT_EN  0x104
#define STATUS      0x10c
#define END_INT_ST  0x110
#define DATA_BASE   0x120

#define START_BIT   (1u<<4)
#define SINGLE_BIT  (1u<<5)

static volatile uint32_t *sar;
static double now_us(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

/* 直接寄存器单次转换（读 chn），返回 10bit 值；超时返回 -1 */
static int adc_direct(int chn) {
    sar[T_DAS_SOC / 4] = 0xc;
    sar[T_PD_SOC / 4] = 0x20;
    sar[END_INT_EN / 4] = 0x10001u;
    uint32_t v = START_BIT | SINGLE_BIT | (uint32_t)chn;
    sar[CONV_CON / 4] = (v << 16) | v;
    /* 轮询完成标志，最多 500us（转换本身 ~44 个 1MHz 时钟 = 44us） */
    for (int i = 0; i < 5000; i++) {
        if (sar[END_INT_ST / 4] & 1u) {
            int val = (int)(sar[(DATA_BASE + chn * 4) / 4] & 0x3ffu);
            sar[END_INT_ST / 4] = 1u; /* 清标志 */
            return val;
        }
    }
    return -1;
}

int main(void) {
    printf("=== RV1106 SARADC capability test (saradc @ 0xff3c0000) ===\n");
    printf("sysfs: scale=%.6f mV/LSB, 满量程 = 1023*scale = %.1f mV\n",
           1.757812500, 1023 * 1.7578125);

    /* A) sysfs 连续读 */
    printf("\n[A] sysfs IIO reads of in_voltage1_raw (ch1, 空闲浮空通道):\n");
    const int N = 1000;
    int vals[N];
    double t0 = now_us();
    FILE *f = fopen("/sys/bus/iio/devices/iio:device0/in_voltage1_raw", "r");
    if (!f) { perror("open sysfs adc"); return 1; }
    for (int i = 0; i < N; i++) {
        rewind(f); fflush(f);
        if (fscanf(f, "%d", &vals[i]) != 1) { vals[i] = -1; }
    }
    double t1 = now_us();
    fclose(f);
    double sps = N / ((t1 - t0) / 1e6);
    long long sum = 0; int mn = 99999, mx = -1;
    for (int i = 0; i < N; i++) { sum += vals[i]; if (vals[i] < mn) mn = vals[i]; if (vals[i] > mx) mx = vals[i]; }
    double mean = (double)sum / N, var = 0;
    for (int i = 0; i < N; i++) var += (vals[i] - mean) * (vals[i] - mean);
    var /= N;
    printf("  %d 次连续读耗时 %.1f ms -> 实际采样率 %.1f SPS (%.2f kSPS)\n",
           N, t1 - t0, sps, sps / 1e3);
    printf("  ch1 噪声统计: min=%d max=%d mean=%.1f (%.2f mV) std=%.1f LSB (%.2f mV)\n",
           mn, mx, mean, mean * 1.7578125, sqrt(var), sqrt(var) * 1.7578125);
    printf("  首 20 个原始值: ");
    for (int i = 0; i < 20 && i < N; i++) printf("%d ", vals[i]);
    printf("\n");
    /* ch0 对照（按键通道） */
    f = fopen("/sys/bus/iio/devices/iio:device0/in_voltage0_raw", "r");
    int v0 = -1; if (f) { fscanf(f, "%d", &v0); fclose(f); }
    printf("  ch0 对照(adc-keys): %d (%.2f mV)\n", v0, v0 * 1.7578125);

    /* B) 直接寄存器 */
    printf("\n[B] /dev/mem direct register conversions (ch1):\n");
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) { perror("open /dev/mem"); return 1; }
    sar = mmap(NULL, 0x200, PROT_READ | PROT_WRITE, MAP_SHARED, fd, SARADC_BASE);
    if (sar == MAP_FAILED) { perror("mmap saradc"); return 1; }

    /* 先读一次（初始化） */
    int first = adc_direct(1);
    printf("  首次直接转换 ch1 = %d\n", first);

    const int M = 500;
    int dvals[M];
    double d0 = now_us();
    for (int i = 0; i < M; i++) dvals[i] = adc_direct(1);
    double d1 = now_us();
    double per_us = (d1 - d0) / M;
    printf("  %d 次直接转换耗时 %.1f ms -> 单次 %.2f us -> 最大 %.2f kSPS\n",
           M, d1 - d0, per_us, 1000.0 / per_us);
    int dmn = 99999, dmx = -1; long long dsum = 0;
    for (int i = 0; i < M; i++) { dsum += dvals[i]; if (dvals[i] < dmn) dmn = dvals[i]; if (dvals[i] > dmx) dmx = dvals[i]; }
    double dmean = (double)dsum / M;
    printf("  直接转换 ch1 统计: min=%d max=%d mean=%.1f (%.2f mV)  (与 sysfs 同引脚应相近)\n",
           dmn, dmx, dmean, dmean * 1.7578125);
    printf("  前 20 个: ");
    for (int i = 0; i < 20 && i < M; i++) printf("%d ", dvals[i]);
    printf("\n");

    /* 直接转换 ch0 对照（应接近 1023；注意与 adc-keys 轮询可能冲突） */
    int dv0 = adc_direct(0);
    printf("  直接转换 ch0 = %d (期望 ~1023, adc-keys 占用)\n", dv0);

    munmap((void *)sar, 0x200);
    close(fd);
    printf("\nSARADC capability test done.\n");
    return 0;
}
