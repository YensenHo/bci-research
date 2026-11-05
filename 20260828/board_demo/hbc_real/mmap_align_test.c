/* mmap 页对齐测试：PWM 控制器0 基址 0xff350000 页对齐映射，读 ch1 寄存器(offset 0x10) */
#include <stdio.h>
#include <stdint.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
int main(void) {
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) { perror("open"); return 1; }
    /* 尝试 3 种映射方式 */
    uint32_t tries[][2] = {{0xff350000, 0x1000}, {0xff350000, 0x40}, {0xff350010, 0x40}};
    for (int i = 0; i < 3; i++) {
        uint32_t phys = tries[i][0], len = tries[i][1];
        volatile uint8_t *p = mmap(NULL, len, PROT_READ|PROT_WRITE, MAP_SHARED, fd, phys);
        if (p == MAP_FAILED) { printf("mmap(0x%08x,len=0x%x) FAILED\n", phys, len); continue; }
        printf("mmap(0x%08x,len=0x%x) OK: ", phys, len);
        for (int j = 0; j < 8; j++) printf("0x%08x ", *(volatile uint32_t *)(p + j*4));
        printf("\n");
        munmap((void*)p, len);
    }
    close(fd);
    return 0;
}
