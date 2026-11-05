/*
 * devmem_probe.c — 探测 /dev/mem 直接寄存器访问是否可用（RV1106）
 * mmap PWM1(0xff350010) / SARADC(0xff3c0000) / GPIO1(0xff530000)，只读不写。
 */
#include <stdio.h>
#include <stdint.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

static int probe(const char *name, uint32_t phys, uint32_t len) {
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) { perror("open /dev/mem"); return -1; }
    volatile uint8_t *p = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, phys);
    if (p == MAP_FAILED) {
        perror("mmap");
        printf("%-10s mmap(0x%08x) FAILED\n", name, phys);
        close(fd);
        return -1;
    }
    /* 读 4 个字 */
    printf("%-10s mmap(0x%08x) OK  regs: ", name, phys);
    for (int i = 0; i < 4; i++)
        printf("0x%08x ", *(volatile uint32_t *)(p + i * 4));
    printf("\n");
    munmap((void *)p, len);
    close(fd);
    return 0;
}

int main(void) {
    printf("devmem probe (RV1106):\n");
    probe("pwm1",   0xff350010, 0x10);
    probe("saradc", 0xff3c0000, 0x200);
    probe("gpio1",  0xff530000, 0x100);
    printf("done\n");
    return 0;
}
