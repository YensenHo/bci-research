/*
 * lif_eegnet_infer.c — LIF-EEGNet（SNN）前向推理的纯 C 实现（CPU 版）
 *
 * 把 ELU-EEGNet 的 C 移植（eegnet_infer.c）换成脉冲类脑模型：
 *   两个 ELU 激活位点 → LIF 脉冲神经元（integrate-and-fire，硬重置）
 *   前向 = 恒定电流输入 x 在 SNN_TIMESTEPS 个时间步内反复馈入，
 *         各层膜电位 mem1/mem2 跨时间步持久，输出层线性读出取时间均值。
 *
 * 与 PyTorch 侧（algorithm/ann2snn.py 的 SNNEEGNet.forward, decay=1.0 纯 IF）逐层对应：
 *   lif_step : mem += current; spk = (mem >= thr); mem *= (1-spk)   ← 硬重置
 *   pool     : 对 spk(0/1) 做 AvgPool(1,4) / AvgPool(1,8)
 *   logits   : Σ_t fc(flatten(pool2(spk2_t))) / timesteps
 *
 * 结构（与 eegnet_infer.c 一致，仅激活层不同）：
 *   conv1(1→8, 1×63, pad L31/R31) → depthwise(8→16, 8×1, groups8) [已折叠 BN]
 *   → LIF(thr1) → AvgPool(1,4) → sep_depthwise(16→16, 1×15, pad L7/R7)
 *   → pointwise(16→16) [已折叠 BN] → LIF(thr2) → AvgPool(1,8) → FC(240→n_classes)
 *
 * n_classes 可配置（SNN_N_CLASSES），为 BCIC 2a 四分类升级留接口。
 *
 * 用法（板载单测）：
 *   zig cc -target arm-linux-musleabihf -O2 -static lif_eegnet_infer.c -o lif_eegnet_infer -lm
 */
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "snn_weights.h"

#define NCH      EEGNET_N_CH        /* 8 */
#define T        EEGNET_T           /* 500 */
#define F1       EEGNET_F1          /* 8 */
#define F2       EEGNET_F2          /* 16 */
#define TOUT     EEGNET_T_OUT       /* 15 = 500//32 */
#define NCLASSES SNN_N_CLASSES      /* 2（可配置） */
#define TSTEPS   SNN_TIMESTEPS      /* timesteps（可配置） */
#define THR1     SNN_THR1
#define THR2     SNN_THR2

#define T4  (T / 4)                /* 125 */
#define T32 (T / 32)               /* 15 */

/*
 * lif_eegnet_infer：输入 x[NCH*T]（通道优先，逐 500 样本），输出 logits[NCLASSES]。
 * 返回预测类别。膜电位 mem1/mem2 用 static 持久化，跨 timesteps 累积。
 */
int lif_eegnet_infer(const float *x, float *logits) {
    /* ---- LIF 膜电位（跨 timesteps 持久）---- */
    static float mem1[F2][T];       /* act1 前：depthwise 输出 (D*F1=16, T=500) */
    static float mem2[F2][T4];      /* act2 前：pointwise 输出 (F2=16, T4=125) */
    static float logit_acc[NCLASSES];  /* 输出层时间累积 */

    for (int o = 0; o < NCLASSES; o++) logit_acc[o] = 0.0f;
    for (int o = 0; o < F2; o++)
        for (int t = 0; t < T; t++) mem1[o][t] = 0.0f;
    for (int o = 0; o < F2; o++)
        for (int t = 0; t < T4; t++) mem2[o][t] = 0.0f;

    for (int step = 0; step < TSTEPS; step++) {
        /* ---- Block 1 ---- */
        /* conv1: 1→F1, kernel (1,63), pad 左31 右31（对称，奇数核），已折叠 BN（含 bias） */
        static float c1[F1][NCH][T];
        for (int f = 0; f < F1; f++) {
            for (int c = 0; c < NCH; c++) {
                for (int t = 0; t < T; t++) {
                    float s = conv1_b[f];
                    for (int k = 0; k < 63; k++) {
                        int idx = t + k - 31;           /* 左 pad 31 */
                        if (idx >= 0 && idx < T) {
                            s += conv1_w[(f * 1 * 1 + 0) * 63 + k] * x[c * T + idx];
                        }
                    }
                    c1[f][c][t] = s;
                }
            }
        }

        /* depthwise: F1→2*F1, kernel (NCH,1), groups=F1，已折叠 BN（含 bias） */
        static float a1[F2][T];
        for (int o = 0; o < F2; o++) {
            int g = o / 2;                              /* 输入通道 = 输出//2 */
            for (int t = 0; t < T; t++) {
                float s = depthwise_b[o];
                for (int c = 0; c < NCH; c++) {
                    s += depthwise_w[(o * 1 * 1 + 0) * NCH + c] * c1[g][c][t];
                }
                a1[o][t] = s;
            }
        }

        /* LIF(thr1) + AvgPool(1,4) → (F2, T4) */
        static float s1[F2][T];
        static float p1[F2][T4];
        for (int o = 0; o < F2; o++) {
            for (int t = 0; t < T; t++) {
                mem1[o][t] += a1[o][t];                 /* 积分 */
                float spk = (mem1[o][t] >= THR1) ? 1.0f : 0.0f;  /* 发放 */
                s1[o][t] = spk;
                mem1[o][t] *= (1.0f - spk);             /* 硬重置 */
            }
        }
        for (int o = 0; o < F2; o++) {
            for (int t = 0; t < T4; t++) {
                float acc = 0.0f;
                for (int k = 0; k < 4; k++) acc += s1[o][t * 4 + k];
                p1[o][t] = acc / 4.0f;
            }
        }

        /* ---- Block 2 ---- */
        /* sep_depthwise: F2→F2, kernel (1,15), pad 左7 右7（对称，奇数核），无 BN 无 bias */
        static float sdw[F2][T4];
        for (int c = 0; c < F2; c++) {
            for (int t = 0; t < T4; t++) {
                float s = 0.0f;
                for (int k = 0; k < 15; k++) {
                    int idx = t + k - 7;                /* 左 pad 7 */
                    if (idx >= 0 && idx < T4) {
                        s += sep_dw_w[(c * 1 * 1 + 0) * 15 + k] * p1[c][idx];
                    }
                }
                sdw[c][t] = s;
            }
        }

        /* pointwise: F2→F2, (1,1)，已折叠 BN（含 bias） */
        static float a2[F2][T4];
        for (int o = 0; o < F2; o++) {
            for (int t = 0; t < T4; t++) {
                float s = pointwise_b[o];
                for (int c = 0; c < F2; c++) {
                    s += pointwise_w[(o * F2 + c) * 1 * 1] * sdw[c][t];
                }
                a2[o][t] = s;
            }
        }

        /* LIF(thr2) + AvgPool(1,8) → (F2, T32) */
        static float s2[F2][T4];
        static float p2[F2][T32];
        for (int o = 0; o < F2; o++) {
            for (int t = 0; t < T4; t++) {
                mem2[o][t] += a2[o][t];                 /* 积分 */
                float spk = (mem2[o][t] >= THR2) ? 1.0f : 0.0f;  /* 发放 */
                s2[o][t] = spk;
                mem2[o][t] *= (1.0f - spk);             /* 硬重置 */
            }
        }
        for (int o = 0; o < F2; o++) {
            for (int t = 0; t < T32; t++) {
                float acc = 0.0f;
                for (int k = 0; k < 8; k++) acc += s2[o][t * 8 + k];
                p2[o][t] = acc / 8.0f;
            }
        }

        /* ---- FC: F2*T32 → NCLASSES（线性读出，不脉冲化）---- */
        for (int o = 0; o < NCLASSES; o++) {
            float s = fc_b[o];
            for (int c = 0; c < F2; c++) {
                for (int t = 0; t < T32; t++) {
                    s += fc_w[o * (F2 * T32) + c * T32 + t] * p2[c][t];
                }
            }
            logit_acc[o] += s;
        }
    }

    /* ---- 时间均值 + argmax ---- */
    int best = 0;
    for (int o = 0; o < NCLASSES; o++) {
        logits[o] = logit_acc[o] / (float)TSTEPS;
        if (o > 0 && logits[o] > logits[best]) best = o;
    }
    return best;
}

#ifdef LIF_EEGNET_INFER_MAIN
/* 单测：随机输入，跑 100 次计时 */
int main(void) {
    static float x[NCH * T];
    for (int i = 0; i < NCH * T; i++) x[i] = (float)(i % 7) / 7.0f - 0.5f;
    float logits[NCLASSES];
    clock_t t0 = clock();
    for (int n = 0; n < 100; n++) lif_eegnet_infer(x, logits);
    double ms = 1000.0 * (clock() - t0) / CLOCKS_PER_SEC / 100.0;
    printf("预测类别: %d, logits=[", logits[0] >= logits[1] ? 0 : 1);
    for (int o = 0; o < NCLASSES; o++) printf("%.3f%s", logits[o], o + 1 < NCLASSES ? ", " : "");
    printf("]\n单次前向耗时: %.2f ms（%d timesteps）\n", ms, TSTEPS);
    return 0;
}
#endif
