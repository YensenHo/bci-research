/*
 * lif_surrogate_infer.c — LIF-EEGNet（surrogate 发放率版）纯 C 推理（CPU 版）
 *
 * 严格复刻 algorithm/lif_eegnet.py 的 LIFActivation.forward（第 100-109 行）语义：
 *   1. decay = exp(-1/tau) = exp(-0.5) = 0.6065（tau=2.0，来自 SNN_DECAY 宏）
 *   2. 每个 timestep：mem = decay*mem + x（x = 卷积+BN 后的恒定电流，每步相同）
 *   3. spk = H(mem - thr)（thr=1.0，严格 > 判定，与 SurrogateStep.forward 的 (x>0) 一致）
 *   4. 硬重置：mem = mem*(1-spk)
 *   5. 输出 = sum(spk)/timesteps（发放率，连续值 0~1）
 *
 * ⚠️ 结构位置（与 ann2snn 版 lif_eegnet_infer.c 不同）：
 *   LIF 在 **pool 之前**：bn2 后 act1 → pool1；bn3 后 act2 → pool2。
 *   pool 作用在**连续发放率**上，不是 0/1 脉冲。decay=0.6065（非 1.0）、thr=1.0（非标定值）。
 *
 * 结构（折叠 BN，与 eegnet_infer.c 一致）：
 *   conv1(1→8, 1×63, pad L31/R31)[fold bn1] → depthwise(8→16, 8×1, groups8)[fold bn2]
 *   → LIF(thr1,decay) → AvgPool(1,4) → sep_depthwise(16→16, 1×15, pad L7/R7)[无BN]
 *   → pointwise(16→16)[fold bn3] → LIF(thr2,decay) → AvgPool(1,8) → FC(240→n_classes)
 *
 * 用法（板载单测 / 端到端精度实测）：
 *   # 交叉编译（RV1106 = ARMv7-A musl，静态）
 *   zig cc -target arm-linux-musleabihf -O2 -static lif_surrogate_infer.c -o lif_surrogate_infer -lm
 *   # 端到端：读取 fusion_test_full.bin（int32 N + 每试次 float32×4000 + int32 label），输出精度
 *   ./lif_surrogate_infer /root/fusion_test_full.bin
 *   # 单测（无参数）：随机输入计时
 *   ./lif_surrogate_infer
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "lif_surrogate_weights.h"

#define NCH      EEGNET_N_CH        /* 8 */
#define T        EEGNET_T           /* 500 */
#define F1       EEGNET_F1          /* 8 */
#define F2       EEGNET_F2          /* 16 */
#define TOUT     EEGNET_T_OUT       /* 15 = 500//32 */
#define NCLASSES SNN_N_CLASSES      /* 2 */
#define TSTEPS   SNN_TIMESTEPS      /* 8 */
#define THR1     SNN_THR1           /* 1.0 */
#define THR2     SNN_THR2           /* 1.0 */
#define DECAY    SNN_DECAY          /* 0.6065 */

#define T4   (T / 4)               /* 125 */
#define T32  (T / 32)              /* 15 */
#define NSAMP (NCH * T)            /* 4000 */

/* 单元素 LIF：恒定电流 I 在 TSTEPS 步内的发放率 ∈ [0,1]（严格复刻 LIFActivation） */
static inline float lif_rate(float I, float thr) {
    float mem = 0.0f;
    int spikes = 0;
    for (int s = 0; s < TSTEPS; s++) {
        mem = DECAY * mem + I;                 /* 泄漏积分（I 恒定） */
        if (mem > thr) {                        /* spk = H(mem-thr)，严格 > */
            spikes++;
            mem = 0.0f;                         /* 硬重置 */
        }
    }
    return (float)spikes / (float)TSTEPS;       /* 发放率 */
}

/*
 * lif_surrogate_infer：输入 x[NCH*T]（通道优先，逐 500 样本），输出 logits[NCLASSES]。
 * 返回预测类别。每试次独立（无跨试次状态）。
 */
int lif_surrogate_infer(const float *x, float *logits) {
    /* ---- Block 1：恒定电流 a1 = depthwise(conv1(x))（折叠 BN），只算一次 ---- */
    static float c1[F1][NCH][T];
    for (int f = 0; f < F1; f++) {
        for (int c = 0; c < NCH; c++) {
            for (int t = 0; t < T; t++) {
                float s = conv1_b[f];
                for (int k = 0; k < 63; k++) {
                    int idx = t + k - 31;               /* 左 pad 31 */
                    if (idx >= 0 && idx < T)
                        s += conv1_w[(f * 1 * 1 + 0) * 63 + k] * x[c * T + idx];
                }
                c1[f][c][t] = s;
            }
        }
    }
    static float a1[F2][T];
    for (int o = 0; o < F2; o++) {
        int g = o / 2;                                  /* depthwise 输入通道 */
        for (int t = 0; t < T; t++) {
            float s = depthwise_b[o];
            for (int c = 0; c < NCH; c++)
                s += depthwise_w[(o * 1 * 1 + 0) * NCH + c] * c1[g][c][t];
            a1[o][t] = s;
        }
    }

    /* ---- LIF(thr1) → 发放率 → AvgPool(1,4) → p1[F2][T4] ---- */
    static float p1[F2][T4];
    for (int o = 0; o < F2; o++) {
        for (int t = 0; t < T4; t++) {
            float acc = 0.0f;
            for (int k = 0; k < 4; k++)
                acc += lif_rate(a1[o][t * 4 + k], THR1);
            p1[o][t] = acc / 4.0f;                      /* 池化作用在连续发放率上 */
        }
    }

    /* ---- Block 2：恒定电流 a2 = pointwise(sep_depthwise(p1))（折叠 BN），只算一次 ---- */
    static float sdw[F2][T4];
    for (int c = 0; c < F2; c++) {
        for (int t = 0; t < T4; t++) {
            float s = 0.0f;
            for (int k = 0; k < 15; k++) {
                int idx = t + k - 7;                    /* 左 pad 7 */
                if (idx >= 0 && idx < T4)
                    s += sep_dw_w[(c * 1 * 1 + 0) * 15 + k] * p1[c][idx];
            }
            sdw[c][t] = s;
        }
    }
    static float a2[F2][T4];
    for (int o = 0; o < F2; o++) {
        for (int t = 0; t < T4; t++) {
            float s = pointwise_b[o];
            for (int c = 0; c < F2; c++)
                s += pointwise_w[(o * F2 + c) * 1 * 1] * sdw[c][t];
            a2[o][t] = s;
        }
    }

    /* ---- LIF(thr2) → 发放率 → AvgPool(1,8) → p2[F2][T32] ---- */
    static float p2[F2][T32];
    for (int o = 0; o < F2; o++) {
        for (int t = 0; t < T32; t++) {
            float acc = 0.0f;
            for (int k = 0; k < 8; k++)
                acc += lif_rate(a2[o][t * 8 + k], THR2);
            p2[o][t] = acc / 8.0f;
        }
    }

    /* ---- FC: F2*T32 → NCLASSES（线性读出）---- */
    int best = 0;
    for (int o = 0; o < NCLASSES; o++) {
        float s = fc_b[o];
        for (int c = 0; c < F2; c++)
            for (int t = 0; t < T32; t++)
                s += fc_w[o * (F2 * T32) + c * T32 + t] * p2[c][t];
        logits[o] = s;
        if (o > 0 && s > logits[best]) best = o;
    }
    return best;
}

#ifdef LIF_SURROGATE_INFER_MAIN
int main(int argc, char **argv) {
    /* ---- 模式 1：读二进制（int32 N + 每试次 float32×4000 + int32 label），端到端精度 ---- */
    if (argc >= 2) {
        FILE *f = fopen(argv[1], "rb");
        if (!f) { fprintf(stderr, "打不开 %s\n", argv[1]); return 1; }
        int N;
        if (fread(&N, 4, 1, f) != 1) { fprintf(stderr, "读取 N 失败\n"); return 1; }

        int correct = 0;
        int per_class[2] = {0, 0}, per_class_correct[2] = {0, 0};
        float *x = (float *)malloc(NSAMP * sizeof(float));
        clock_t t0 = clock();
        for (int n = 0; n < N; n++) {
            int y;
            if (fread(x, 4, NSAMP, f) != (size_t)NSAMP) { fprintf(stderr, "读取试次 %d 失败\n", n); break; }
            if (fread(&y, 4, 1, f) != 1) break;
            float logits[NCLASSES];
            int pred = lif_surrogate_infer(x, logits);
            per_class[y]++;
            if (pred == y) { correct++; per_class_correct[y]++; }
        }
        fclose(f);
        free(x);
        double dt = 1000.0 * (clock() - t0) / CLOCKS_PER_SEC;

        printf("============================================================\n");
        printf("  LIF-EEGNet（surrogate 发放率版）板载端到端精度实测\n");
        printf("  thr=%.2f  decay=%.4f  timesteps=%d  LIF 在 pool 前\n",
               (double)THR1, (double)DECAY, TSTEPS);
        printf("============================================================\n");
        printf("  试次数         : %d\n", N);
        printf("  正确数         : %d\n", correct);
        printf("  端到端精度     : %.4f%% (%d/%d)\n", 100.0 * correct / N, correct, N);
        for (int c = 0; c < 2; c++)
            printf("  类别 %d 精度     : %.2f%% (%d/%d)\n",
                   c, 100.0 * per_class_correct[c] / per_class[c], per_class_correct[c], per_class[c]);
        printf("  总耗时         : %.1f ms\n", dt);
        printf("  单次前向耗时   : %.3f ms\n", dt / N);
        printf("============================================================\n");
        return 0;
    }

    /* ---- 模式 2：随机输入，跑 100 次计时 ---- */
    static float x[NSAMP];
    for (int i = 0; i < NSAMP; i++) x[i] = (float)(i % 7) / 7.0f - 0.5f;
    float logits[NCLASSES];
    clock_t t0 = clock();
    for (int n = 0; n < 100; n++) lif_surrogate_infer(x, logits);
    double ms = 1000.0 * (clock() - t0) / CLOCKS_PER_SEC / 100.0;
    printf("预测类别: %d, logits=[", logits[0] >= logits[1] ? 0 : 1);
    for (int o = 0; o < NCLASSES; o++) printf("%.3f%s", logits[o], o + 1 < NCLASSES ? ", " : "");
    printf("]\n单次前向耗时: %.3f ms（%d timesteps，decay=%.4f）\n", ms, TSTEPS, (double)DECAY);
    return 0;
}
#endif
