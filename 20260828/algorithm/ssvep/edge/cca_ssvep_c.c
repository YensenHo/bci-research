/* cca_ssvep_c.c — SSVEP 标准 CCA 端侧(C 语言)参考实现
 * 目标平台: RV1106 (Luckfox Pico, 双核 Cortex-A7 @1.2GHz)
 * 规模: 6 通道 x 40 频率 x 3 次谐波, 窗口 1s @250Hz (NSAMP=250)
 * 算法: 标准 CCA (H = Cxx^{-1/2} Cxy Cyy^{-1/2}, rho = 最大奇异值)
 *       特征分解: 对称 6x6 Jacobi; 奇异值: 对 M=HH^T 幂迭代
 * 编译: gcc -O2 -o cca_ssvep_c cca_ssvep_c.c -lm
 * 运行: ./cca_ssvep_c [ntrials]   (默认 1000 次决策计时 + 合成数据验证)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define NCH   6
#define NFREQ 40
#define NHARM 3
#define NSAMP 250   /* 1s @250Hz; 0.5s=125, 2s=500 按需调整 */

static double freqs[NFREQ];
static double Yt[NFREQ][NCH][NSAMP];   /* sin/cos 模板, 全局(只读) */

/* ---------- 6x6 对称矩阵 Jacobi 特征分解: A = V D V^T (A 原地覆盖) ---------- */
static void jacobi_eig(double A[NCH][NCH], double d[NCH], double V[NCH][NCH]) {
    int i, j, k, p, q, sweep;
    for (i = 0; i < NCH; i++)
        for (j = 0; j < NCH; j++) V[i][j] = (i == j) ? 1.0 : 0.0;
    for (sweep = 0; sweep < 60; sweep++) {
        double off = 0.0;
        for (p = 0; p < NCH; p++)
            for (q = p + 1; q < NCH; q++) off += A[p][q] * A[p][q];
        if (off < 1e-20) break;
        for (p = 0; p < NCH; p++) {
            for (q = p + 1; q < NCH; q++) {
                if (fabs(A[p][q]) < 1e-16) continue;
                double theta = 0.5 * atan2(2.0 * A[p][q], A[q][q] - A[p][p]);
                double c = cos(theta), s = sin(theta);
                for (k = 0; k < NCH; k++) {          /* R^T A (行旋转) */
                    double akp = A[k][p], akq = A[k][q];
                    A[k][p] = c * akp - s * akq;
                    A[k][q] = s * akp + c * akq;
                }
                for (k = 0; k < NCH; k++) {          /* A R (列旋转) */
                    double apk = A[p][k], aqk = A[q][k];
                    A[p][k] = c * apk - s * aqk;
                    A[q][k] = s * apk + c * aqk;
                }
                for (k = 0; k < NCH; k++) {          /* V R */
                    double vkp = V[k][p], vkq = V[k][q];
                    V[k][p] = c * vkp - s * vkq;
                    V[k][q] = s * vkp + c * vkq;
                }
            }
        }
    }
    for (i = 0; i < NCH; i++) d[i] = A[i][i];
}

/* C^{-1/2} = V diag(1/sqrt(d)) V^T (C 对称正定) */
static void mat_sqrt_inv(const double C[NCH][NCH], double out[NCH][NCH]) {
    double Cc[NCH][NCH], V[NCH][NCH], d[NCH], T[NCH][NCH];
    int i, j, k;
    memcpy(Cc, C, sizeof(Cc));
    jacobi_eig(Cc, d, V);
    for (i = 0; i < NCH; i++)
        for (j = 0; j < NCH; j++) T[i][j] = V[i][j] / sqrt(fmax(d[j], 1e-12));
    for (i = 0; i < NCH; i++)
        for (j = 0; j < NCH; j++) {
            double s = 0.0;
            for (k = 0; k < NCH; k++) s += T[i][k] * V[j][k];
            out[i][j] = s;
        }
}

/* 单次决策: X(6xNSAMP) -> 40 个频率的 rho, 返回最大 rho 及对应频率下标 */
static double cca_decision(const double X[NCH][NSAMP],
                           const double Cyy_ih[NFREQ][NCH][NCH],
                           int *pred_freq) {
    double Xc[NCH][NSAMP];
    double Cxx[NCH][NCH], Cxx_ih[NCH][NCH];
    double H[NCH][NCH], M[NCH][NCH], T[NCH][NCH];
    double best = -1.0;
    int bestf = -1, c, i, j, k, f, n, it;

    for (c = 0; c < NCH; c++) {                 /* 去均值 */
        double m = 0.0;
        for (n = 0; n < NSAMP; n++) m += X[c][n];
        m /= NSAMP;
        for (n = 0; n < NSAMP; n++) Xc[c][n] = X[c][n] - m;
    }
    for (i = 0; i < NCH; i++)                   /* Cxx = X X^T / n */
        for (j = 0; j < NCH; j++) {
            double s = 0.0;
            for (n = 0; n < NSAMP; n++) s += Xc[i][n] * Xc[j][n];
            Cxx[i][j] = s / NSAMP;
        }
    mat_sqrt_inv(Cxx, Cxx_ih);

    for (f = 0; f < NFREQ; f++) {
        double Cxy[NCH][NCH];
        for (i = 0; i < NCH; i++)               /* Cxy = X Y^T / n */
            for (j = 0; j < NCH; j++) {
                double s = 0.0;
                for (n = 0; n < NSAMP; n++) s += Xc[i][n] * Yt[f][j][n];
                Cxy[i][j] = s / NSAMP;
            }
        /* H = Cxx_ih Cxy Cyy_ih[f] */
        for (i = 0; i < NCH; i++)
            for (j = 0; j < NCH; j++) {
                double s = 0.0;
                for (k = 0; k < NCH; k++) s += Cxx_ih[i][k] * Cxy[k][j];
                T[i][j] = s;
            }
        for (i = 0; i < NCH; i++)
            for (j = 0; j < NCH; j++) {
                double s = 0.0;
                for (k = 0; k < NCH; k++) s += T[i][k] * Cyy_ih[f][k][j];
                H[i][j] = s;
            }
        for (i = 0; i < NCH; i++)               /* M = H H^T (对称半正定) */
            for (j = 0; j < NCH; j++) {
                double s = 0.0;
                for (k = 0; k < NCH; k++) s += H[i][k] * H[j][k];
                M[i][j] = s;
            }
        /* 幂迭代求 M 最大特征值 (rho = sqrt(lambda)) */
        double v[NCH], w[NCH], lam = 0.0;
        for (i = 0; i < NCH; i++) v[i] = 1.0;
        for (it = 0; it < 40; it++) {
            double nrm = 0.0;
            for (i = 0; i < NCH; i++) {
                double s = 0.0;
                for (k = 0; k < NCH; k++) s += M[i][k] * v[k];
                w[i] = s;
                nrm += s * s;
            }
            nrm = sqrt(nrm);
            if (nrm < 1e-15) break;
            for (i = 0; i < NCH; i++) v[i] = w[i] / nrm;
            lam = 0.0;
            for (i = 0; i < NCH; i++) lam += v[i] * w[i];
        }
        double rho = sqrt(fmax(lam, 0.0));
        if (rho > best) { best = rho; bestf = f; }
    }
    *pred_freq = bestf;
    return best;
}

/* 合成模板 Y[f][6][NSAMP]: sin/cos 1~3 次谐波 (与 Python 端一致) */
static void build_templates(double Cyy_ih[NFREQ][NCH][NCH]) {
    int f, h, n, i, j, k;
    for (f = 0; f < NFREQ; f++)
        for (h = 1; h <= NHARM; h++)
            for (n = 0; n < NSAMP; n++) {
                double t = (double)n / 250.0;
                Yt[f][2 * (h - 1)][n] = sin(2 * M_PI * h * freqs[f] * t);
                Yt[f][2 * (h - 1) + 1][n] = cos(2 * M_PI * h * freqs[f] * t);
            }
    /* 预计算 Cyy^{-1/2}[f] (模板固定, 只需一次) */
    for (f = 0; f < NFREQ; f++) {
        double Cyy[NCH][NCH];
        for (i = 0; i < NCH; i++)
            for (j = 0; j < NCH; j++) {
                double s = 0.0;
                for (n = 0; n < NSAMP; n++) s += Yt[f][i][n] * Yt[f][j][n];
                Cyy[i][j] = s / NSAMP;
            }
        mat_sqrt_inv(Cyy, Cyy_ih[f]);
    }
}

/* 合成 6 通道 SSVEP: 目标频率 f0 的 1~3 次谐波 + 高斯噪声 */
static void synth_trial(double X[NCH][NSAMP], double f0, double noise_sd,
                        unsigned *seed) {
    int c, n, h;
    for (c = 0; c < NCH; c++)
        for (n = 0; n < NSAMP; n++) {
            double t = (double)n / 250.0, s = 0.0;
            for (h = 1; h <= NHARM; h++)
                s += 0.8 * pow(h, -0.5) * sin(2 * M_PI * h * f0 * t
                                              + (double)((c * 7 + h * 3) % 6));
            X[c][n] = s + noise_sd * ((double)rand() / RAND_MAX - 0.5) * 3.46;
        }
}

int main(int argc, char **argv) {
    int ntrials = (argc > 1) ? atoi(argv[1]) : 1000;
    static double Cyy_ih[NFREQ][NCH][NCH];
    static double X[NCH][NSAMP];
    int f, t;
    srand(42);
    for (f = 0; f < NFREQ; f++) freqs[f] = 8.0 + (f % 8) + 0.2 * (f / 8);
    build_templates(Cyy_ih);

    /* 验证: 100 个合成 trial, 噪声 0.4, 期望 100% 正确 */
    int ok = 0, N = 100;
    for (t = 0; t < N; t++) {
        int tg = t % NFREQ, pred;
        synth_trial(X, freqs[tg], 0.4, NULL);
        cca_decision(X, Cyy_ih, &pred);
        if (pred == tg) ok++;
    }
    printf("synthetic top1 acc: %d/%d = %.2f (expect 1.00)\n", ok, N,
           (double)ok / N);

    /* 计时: ntrials 次决策 */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    volatile double sink = 0.0;
    for (t = 0; t < ntrials; t++) {
        int pred;
        synth_trial(X, freqs[t % NFREQ], 0.4, NULL);
        sink += cca_decision(X, Cyy_ih, &pred);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double secs = (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);
    printf("host(x86/ARM64) 单次决策: %.1f us  (%d trials, %.2f ms 总耗时)\n",
           1e6 * secs / ntrials, ntrials, 1e3 * secs);
    printf("(sink=%f 防止优化删除)\n", (double)sink);
    return 0;
}
