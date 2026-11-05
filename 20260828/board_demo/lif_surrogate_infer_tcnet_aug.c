/*
 * lif_surrogate_infer_tcnet.c — LIF-EEG-TCNet（surrogate 发放率版）纯 C 推理（22ch 四分类，CPU 版）
 *
 * 严格复刻 algorithm/lif_eegnet.py 的 LIFActivation.forward（第 100-112 行）语义：
 *   1. decay = exp(-1/tau) = exp(-0.5) = 0.6065（tau=2.0，来自 SNN_DECAY 宏）
 *   2. 每个 timestep：mem = decay*mem + x（x = 卷积+BN 后的恒定电流，每步相同）
 *   3. spk = H(mem - thr)（thr=1.0，严格 > 判定，与 SurrogateStep.forward 的 (x>0) 一致）
 *   4. 硬重置：mem = mem*(1-spk)
 *   5. 输出 = sum(spk)/timesteps（发放率，连续值 0~1）
 *
 * 与 lif_surrogate_infer_4class.c（LIF-EEGNet 四分类版）的结构差异（本文件核心新增）：
 *   a. TCN 两个残差块：每块两层**膨胀因果 Conv1d**（KT=12，块0 d=1 / 块1 d=2），
 *      因果性靠手动左补零 (KT-1)*d 实现（eeg_tcnet_lif.py 的 pad_left 语义）；
 *      C 索引：out[t] = Σ_j w[j] * in[t + j*d - pad]，仅需检查下界 idx>=0
 *      （上界自动满足：idx = t + j*d - pad ≤ t ≤ T15-1）。
 *   b. 残差连接：skip 分支 1×1 卷积（块0 16→12 带偏置 / 块1 12→12 无偏置）
 *      与两层卷积输出逐元素相加，再过 act_res（LIF）。
 *   c. 读出头：GAP 全局平均池化（对 TCN 输出的 15 个时间步求均值）→ Linear(12→4)。
 *   d. LIF 位点共 8 个：act1/act2（前端 pool 前）+ 每块 act_a/act_b/act_res。
 *
 * 结构（折叠 BN，brief 配置 KE=63/KT=12/pool1=4/pool2=8/GAP）：
 *   conv1(1→8, 1×63, pad L31/R31)[fold bn1] → depthwise(8→16, 22×1, groups8)[fold bn2]
 *   → LIF(thr1) → AvgPool(1,4) → sep_depthwise(16→16, 1×16, pad L8/R8)[无BN]
 *   → pointwise(16→16)[fold bn3] → LIF(thr2) → AvgPool(1,8) → (16,15)
 *   → TCN块0(d=1): conv_a(16→12,k12)[fold bn] → LIF → conv_b(12→12)[fold bn] → LIF
 *                  → skip(1×1,16→12,bias) 残差 add → LIF
 *   → TCN块1(d=2): conv_a(12→12,k12)[fold bn] → LIF → conv_b(12→12)[fold bn] → LIF
 *                  → skip(1×1,12→12,无bias) 残差 add → LIF
 *   → GAP(时间均值, 15→1) → Linear(12→4)
 *
 * 用法（板载单测 / 端到端精度实测）：
 *   # 宿主机编译（逐位一致验证用 -ffp-contract=off）
 *   cc -O2 -ffp-contract=off -DLIF_TCNET_INFER_MAIN lif_surrogate_infer_tcnet.c -o lif_surrogate_infer_tcnet_host -lm
 *   # 交叉编译（RV1106 = ARMv7-A musl，静态；zig 勿加 -mcpu/-march，默认即 ARMv7 硬浮点）
 *   ~/luckfox/zig-0.13.0/zig cc -target arm-linux-musleabihf -O2 -static \
 *       -DLIF_TCNET_INFER_MAIN lif_surrogate_infer_tcnet.c -o lif_surrogate_infer_tcnet -lm
 *   # 端到端：读取 lif_4class_test_full.bin（int32 N + 每试次 float32×11000 + int32 label）
 *   ./lif_surrogate_infer_tcnet /root/lif_4class_test_full.bin
 *   # 单测（无参数）：随机输入计时
 *   ./lif_surrogate_infer_tcnet
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "lif_tcnet_snn_weights_aug.h"

#define NCH      EEGNET_N_CH        /* 22 */
#define T        EEGNET_T           /* 500 */
#define F1       EEGNET_F1          /* 8 */
#define D        EEGNET_D           /* 2 */
#define F2       EEGNET_F2          /* 16 = D*F1 */
#define FT       EEGNET_FT          /* 12（TCN 滤波器数） */
#define KE       EEGNET_KE          /* 63（conv1 核） */
#define SEP_K    EEGNET_SEP_K       /* 16（separable 核） */
#define KT       EEGNET_KT          /* 12（TCN 膨胀卷积核） */
#define T4       EEGNET_T4          /* 125 = 500//4 */
#define T15      EEGNET_T15         /* 15  = 125//8（TCN 输入时间步） */
#define NCLASSES SNN_N_CLASSES      /* 4 */
#define TSTEPS   SNN_TIMESTEPS      /* 8 */
#define THR      SNN_THR1           /* 1.0（8 个 LIF 位点同阈值） */
#define DECAY    SNN_DECAY          /* 0.6065 */
#define NSAMP    (NCH * T)          /* 11000 */

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
 * lif_surrogate_infer_tcnet：输入 x[NCH*T]（通道优先，逐 500 样本），输出 logits[NCLASSES]。
 * 返回预测类别。每试次独立（无跨试次状态）。
 */
int lif_surrogate_infer_tcnet(const float *x, float *logits) {
    /* ================= Block 1：恒定电流 a1 = depthwise(conv1(x))（折叠 BN），只算一次 ================= */
    static float c1[F1][NCH][T];
    for (int f = 0; f < F1; f++) {
        for (int c = 0; c < NCH; c++) {
            for (int t = 0; t < T; t++) {
                float s = conv1_b[f];
                for (int k = 0; k < KE; k++) {
                    int idx = t + k - (KE / 2);          /* 对称补零 KE/2=31，双侧 */
                    if (idx >= 0 && idx < T)
                        s += conv1_w[f * KE + k] * x[c * T + idx];
                }
                c1[f][c][t] = s;
            }
        }
    }
    static float a1[F2][T];
    for (int o = 0; o < F2; o++) {
        int g = o / D;                                  /* depthwise 输入通道（group） */
        for (int t = 0; t < T; t++) {
            float s = depthwise_b[o];
            for (int c = 0; c < NCH; c++)
                s += depthwise_w[o * NCH + c] * c1[g][c][t];
            a1[o][t] = s;
        }
    }

    /* ---- LIF(act1) → 发放率 → AvgPool(1,4) → p1[F2][T4] ---- */
    static float p1[F2][T4];
    for (int o = 0; o < F2; o++) {
        for (int t = 0; t < T4; t++) {
            float acc = 0.0f;
            for (int k = 0; k < 4; k++)
                acc += lif_rate(a1[o][t * 4 + k], THR);
            p1[o][t] = acc / 4.0f;                      /* 池化作用在连续发放率上 */
        }
    }

    /* ================= Block 2：恒定电流 a2 = pointwise(sep_depthwise(p1))（折叠 BN） ================= */
    static float sdw[F2][T4];
    for (int c = 0; c < F2; c++) {
        for (int t = 0; t < T4; t++) {
            float s = 0.0f;
            for (int k = 0; k < SEP_K; k++) {
                int idx = t + k - (SEP_K / 2);          /* 对称补零 8 */
                if (idx >= 0 && idx < T4)
                    s += sep_dw_w[c * SEP_K + k] * p1[c][idx];
            }
            sdw[c][t] = s;
        }
    }
    static float a2[F2][T4];
    for (int o = 0; o < F2; o++) {
        for (int t = 0; t < T4; t++) {
            float s = pointwise_b[o];
            for (int c = 0; c < F2; c++)
                s += pointwise_w[o * F2 + c] * sdw[c][t];
            a2[o][t] = s;
        }
    }

    /* ---- LIF(act2) → 发放率 → AvgPool(1,8) → p2[F2][T15]（TCN 输入） ---- */
    static float p2[F2][T15];
    for (int o = 0; o < F2; o++) {
        for (int t = 0; t < T15; t++) {
            float acc = 0.0f;
            for (int k = 0; k < 8; k++)
                acc += lif_rate(a2[o][t * 8 + k], THR);
            p2[o][t] = acc / 8.0f;
        }
    }

    /* ================= TCN 块 0（dilation=1）：两层膨胀因果卷积 + 残差 ================= */
    static float t0_ca[FT][T15], t0_ra[FT][T15];        /* conv_a 输出 / act_a 发放率 */
    static float t0_cb[FT][T15], t0_rb[FT][T15];        /* conv_b 输出 / act_b 发放率 */
    static float t0_sk[FT][T15], t0_out[FT][T15];       /* skip 投影 / 残差输出（act_res） */
    const int pad0 = (KT - 1) * 1;                      /* 因果左补零 (KT-1)*d = 11 */
    /* conv_a：16→12，膨胀 1。idx = t + j*d - pad，仅需检查下界 */
    for (int o = 0; o < FT; o++) {
        for (int t = 0; t < T15; t++) {
            float s = tcn0_conv_a_b[o];
            for (int c = 0; c < F2; c++)
                for (int j = 0; j < KT; j++) {
                    int idx = t + j - pad0;
                    if (idx >= 0)
                        s += tcn0_conv_a_w[(o * F2 + c) * KT + j] * p2[c][idx];
                }
            t0_ca[o][t] = s;
        }
    }
    for (int o = 0; o < FT; o++)
        for (int t = 0; t < T15; t++)
            t0_ra[o][t] = lif_rate(t0_ca[o][t], THR);   /* LIF 位点：act_a */
    /* conv_b：12→12，膨胀 1 */
    for (int o = 0; o < FT; o++) {
        for (int t = 0; t < T15; t++) {
            float s = tcn0_conv_b_b[o];
            for (int c = 0; c < FT; c++)
                for (int j = 0; j < KT; j++) {
                    int idx = t + j - pad0;
                    if (idx >= 0)
                        s += tcn0_conv_b_w[(o * FT + c) * KT + j] * t0_ra[c][idx];
                }
            t0_cb[o][t] = s;
        }
    }
    for (int o = 0; o < FT; o++)
        for (int t = 0; t < T15; t++)
            t0_rb[o][t] = lif_rate(t0_cb[o][t], THR);   /* LIF 位点：act_b */
    /* skip 1×1 投影：16→12（带偏置），对输入 p2 逐时间步线性组合 */
    for (int o = 0; o < FT; o++) {
        for (int t = 0; t < T15; t++) {
            float s = tcn0_skip_b[o];
            for (int c = 0; c < F2; c++)
                s += tcn0_skip_w[o * F2 + c] * p2[c][t];
            t0_sk[o][t] = s;
        }
    }
    /* 残差 add：skip + conv_b 输出 → act_res（LIF 位点） */
    for (int o = 0; o < FT; o++)
        for (int t = 0; t < T15; t++)
            t0_out[o][t] = lif_rate(t0_sk[o][t] + t0_rb[o][t], THR);

    /* ================= TCN 块 1（dilation=2）：两层膨胀因果卷积 + 残差 ================= */
    static float t1_ca[FT][T15], t1_ra[FT][T15];
    static float t1_cb[FT][T15], t1_rb[FT][T15];
    static float t1_sk[FT][T15], t1_out[FT][T15];
    const int pad1 = (KT - 1) * 2;                      /* 因果左补零 (KT-1)*d = 22 */
    /* conv_a：12→12，膨胀 2。idx = t + 2j - 22 */
    for (int o = 0; o < FT; o++) {
        for (int t = 0; t < T15; t++) {
            float s = tcn1_conv_a_b[o];
            for (int c = 0; c < FT; c++)
                for (int j = 0; j < KT; j++) {
                    int idx = t + 2 * j - pad1;
                    if (idx >= 0)
                        s += tcn1_conv_a_w[(o * FT + c) * KT + j] * t0_out[c][idx];
                }
            t1_ca[o][t] = s;
        }
    }
    for (int o = 0; o < FT; o++)
        for (int t = 0; t < T15; t++)
            t1_ra[o][t] = lif_rate(t1_ca[o][t], THR);   /* LIF 位点：act_a */
    /* conv_b：12→12，膨胀 2 */
    for (int o = 0; o < FT; o++) {
        for (int t = 0; t < T15; t++) {
            float s = tcn1_conv_b_b[o];
            for (int c = 0; c < FT; c++)
                for (int j = 0; j < KT; j++) {
                    int idx = t + 2 * j - pad1;
                    if (idx >= 0)
                        s += tcn1_conv_b_w[(o * FT + c) * KT + j] * t1_ra[c][idx];
                }
            t1_cb[o][t] = s;
        }
    }
    for (int o = 0; o < FT; o++)
        for (int t = 0; t < T15; t++)
            t1_rb[o][t] = lif_rate(t1_cb[o][t], THR);   /* LIF 位点：act_b */
    /* skip 1×1 投影：12→12（无偏置，对齐原文 4272 反推） */
    for (int o = 0; o < FT; o++) {
        for (int t = 0; t < T15; t++) {
            float s = 0.0f;
            for (int c = 0; c < FT; c++)
                s += tcn1_skip_w[o * FT + c] * t0_out[c][t];
            t1_sk[o][t] = s;
        }
    }
    /* 残差 add → act_res（LIF 位点）→ t1_out[FT][T15] */
    for (int o = 0; o < FT; o++)
        for (int t = 0; t < T15; t++)
            t1_out[o][t] = lif_rate(t1_sk[o][t] + t1_rb[o][t], THR);

    /* ================= GAP：对 TCN 输出的 15 个时间步求均值（读出头） ================= */
    static float g[FT];
    for (int o = 0; o < FT; o++) {
        float acc = 0.0f;
        for (int t = 0; t < T15; t++)
            acc += t1_out[o][t];
        g[o] = acc / (float)T15;                        /* 与 AdaptiveAvgPool1d(1) 等价 */
    }

    /* ================= FC：Linear(12→4) ================= */
    int best = 0;
    for (int o = 0; o < NCLASSES; o++) {
        float s = fc_b[o];
        for (int c = 0; c < FT; c++)
            s += fc_w[o * FT + c] * g[c];
        logits[o] = s;
        if (o > 0 && s > logits[best]) best = o;
    }
    return best;
}

#ifdef LIF_TCNET_INFER_MAIN
static double now_ms(void) {
    /* RV1106 上 clock() 太粗（CLOCKS_PER_SEC 小），用单调时钟（见 rockchip-edge-boards 技能） */
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(int argc, char **argv) {
    /* ---- 模式 1：读二进制（int32 N + 每试次 float32×11000 + int32 label），端到端精度 ---- */
    if (argc >= 2) {
        FILE *f = fopen(argv[1], "rb");
        if (!f) { fprintf(stderr, "打不开 %s\n", argv[1]); return 1; }
        int N;
        if (fread(&N, 4, 1, f) != 1) { fprintf(stderr, "读取 N 失败\n"); return 1; }

        int correct = 0;
        int per_class[NCLASSES], per_class_correct[NCLASSES];
        for (int c = 0; c < NCLASSES; c++) per_class[c] = per_class_correct[c] = 0;
        float *x = (float *)malloc(NSAMP * sizeof(float));
        double t0 = now_ms();
        for (int n = 0; n < N; n++) {
            int y;
            if (fread(x, 4, NSAMP, f) != (size_t)NSAMP) { fprintf(stderr, "读取试次 %d 失败\n", n); break; }
            if (fread(&y, 4, 1, f) != 1) break;
            float logits[NCLASSES];
            int pred = lif_surrogate_infer_tcnet(x, logits);
            per_class[y]++;
            if (pred == y) { correct++; per_class_correct[y]++; }
        }
        fclose(f);
        free(x);
        double dt = now_ms() - t0;

        printf("============================================================\n");
        printf("  LIF-EEG-TCNet（surrogate 发放率版，22ch 四分类）板载端到端精度实测\n");
        printf("  thr=%.2f  decay=%.4f  timesteps=%d  8 个 LIF 位点（含 TCN act_a/act_b/act_res）\n",
               (double)THR, (double)DECAY, TSTEPS);
        printf("  TCN: KT=%d dilations=(1,2) 残差 + GAP 读出头\n", KT);
        printf("============================================================\n");
        printf("  试次数         : %d\n", N);
        printf("  正确数         : %d\n", correct);
        printf("  端到端精度     : %.4f%% (%d/%d)\n", 100.0 * correct / N, correct, N);
        for (int c = 0; c < NCLASSES; c++)
            printf("  类别 %d 精度     : %.2f%% (%d/%d)\n",
                   c, 100.0 * per_class_correct[c] / per_class[c],
                   per_class_correct[c], per_class[c]);
        printf("  总耗时         : %.1f ms\n", dt);
        printf("  单次前向耗时   : %.3f ms\n", dt / N);
        printf("============================================================\n");
        return 0;
    }

    /* ---- 模式 2：随机输入，跑 100 次计时 ---- */
    static float x[NSAMP];
    for (int i = 0; i < NSAMP; i++) x[i] = (float)(i % 7) / 7.0f - 0.5f;
    float logits[NCLASSES];
    double t0 = now_ms();
    for (int n = 0; n < 100; n++) lif_surrogate_infer_tcnet(x, logits);
    double ms = (now_ms() - t0) / 100.0;
    int best = 0;
    for (int o = 1; o < NCLASSES; o++) if (logits[o] > logits[best]) best = o;
    printf("预测类别: %d, logits=[", best);
    for (int o = 0; o < NCLASSES; o++) printf("%.3f%s", logits[o], o + 1 < NCLASSES ? ", " : "");
    printf("]\n单次前向耗时: %.3f ms（%d timesteps，decay=%.4f）\n", ms, TSTEPS, (double)DECAY);
    return 0;
}
#endif
