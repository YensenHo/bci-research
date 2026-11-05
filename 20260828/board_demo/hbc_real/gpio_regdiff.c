/*
 * gpio_regdiff.c — 确定 RV1106 GPIO1 寄存器布局（v1 整字 vs v2 _L/_H 拆分）
 *
 * 方法：用内核 sysfs 把 gpio32 (GPIO1_A0) 设为输出高，
 *       再读 /dev/mem 0xff530000 偏移 0x00~0x5c，对比设置前后的变化。
 *       变化所在的偏移即真实的数据/方向寄存器（v2 布局下 DR_L=0x00, DDR_L=0x08）。
 */
#include <stdio.h>
#include <stdint.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>

#define GPIO1_BASE 0xff530000UL

static volatile uint32_t *gpio;

static void dump(const char *tag, uint32_t before[24]) {
    printf("--- %s ---\n", tag);
    for (int i = 0; i < 24; i++) {
        uint32_t v = gpio[i];
        uint32_t b = before ? before[i] : 0;
        printf("  +0x%02x (0x%02x): 0x%08x%s\n", i * 4, i * 4, v,
               (before && v != b) ? "   <-- CHANGED" : "");
    }
}

static int wf(const char *path, const char *s) {
    int fd = open(path, O_WRONLY);
    if (fd < 0) { perror(path); return -1; }
    int r = write(fd, s, strlen(s));
    close(fd);
    return r < 0 ? -1 : 0;
}

int main(void) {
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) { perror("open /dev/mem"); return 1; }
    gpio = mmap(NULL, 0x100, PROT_READ | PROT_WRITE, MAP_SHARED, fd, GPIO1_BASE);
    if (gpio == MAP_FAILED) { perror("mmap"); return 1; }

    printf("=== GPIO1 (0xff530000) register layout probe, gpio32 = GPIO1_A0 ===\n");

    uint32_t before[24];
    for (int i = 0; i < 24; i++) before[i] = gpio[i];
    dump("before (all gpio1 input, free)", before);

    /* sysfs: export gpio32 -> out -> high */
    wf("/sys/class/gpio/export", "32");
    usleep(100000);
    wf("/sys/class/gpio/gpio32/direction", "out");
    usleep(50000);
    wf("/sys/class/gpio/gpio32/value", "1");
    usleep(50000);

    printf("\n");
    dump("after  (gpio32 = output high)", before);

    /* 再读 EXT 输入寄存器区域 */
    printf("\n--- EXT (input) region 0x50/0x54 ---\n");
    printf("  +0x50: 0x%08x   +0x54: 0x%08x\n", gpio[0x50 / 4], gpio[0x54 / 4]);

    /* cleanup */
    wf("/sys/class/gpio/gpio32/value", "0");
    wf("/sys/class/gpio/gpio32/unexport", "32");

    munmap((void *)gpio, 0x100);
    close(fd);
    printf("\ndone\n");
    return 0;
}
