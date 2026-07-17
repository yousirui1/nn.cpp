/*
 * 最简单的 8 点 FFT + STFT + ISTFT 实现
 * 没有任何优化，纯教学目的，每一行都有注释
 * 编译：gcc -o demo demo.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 8          /* FFT 长度，必须是 2 的幂 */
#define M_PI 3.14159265358979323846

/*=============================================================
 *  第一部分：复数基础运算
 *=============================================================*/

typedef struct {
    double re;  /* 实部 */
    double im;  /* 虚部 */
} Complex;

/* 复数加法: a + b */
Complex c_add(Complex a, Complex b) {
    Complex r;
    r.re = a.re + b.re;
    r.im = a.im + b.im;
    return r;
}

/* 复数减法: a - b */
Complex c_sub(Complex a, Complex b) {
    Complex r;
    r.re = a.re - b.re;
    r.im = a.im - b.im;
    return r;
}

/* 复数乘法: a × b
 *
 *   (a_re + j·a_im) × (b_re + j·b_im)
 * = (a_re·b_re - a_im·b_im) + j·(a_re·b_im + a_im·b_re)
 *
 * 蝶式运算里旋转因子和信号的相乘就用这个
 */
Complex c_mul(Complex a, Complex b) {
    Complex r;
    r.re = a.re * b.re - a.im * b.im;
    r.im = a.re * b.im + a.im * b.re;
    return r;
}

/* 复数的模（幅度/强度）= |a| = √(re² + im²) */
double c_abs(Complex a) {
    return sqrt(a.re * a.re + a.im * a.im);
}

/* 复数的相位（角度）= atan2(im, re) */
double c_phase(Complex a) {
    return atan2(a.im, a.re);
}

/*=============================================================
 *  第二部分：旋转因子表（只算一次，之后查表）
 *
 *  W_N^k = e^{-j2πk/N} = cos(2πk/N) - j·sin(2πk/N)
 *
 *  这就是"标准波形"，用来跟信号逐点比对
 *  只跟 N 有关，跟采样率无关
 *=============================================================*/

Complex twiddles[N];  /* 旋转因子表 */

void init_twiddles(int inverse) {
    /*
     * inverse = 0: 正变换(FFT)  → e^{-j2πk/N}  顺时针
     * inverse = 1: 逆变换(IFFT) → e^{+j2πk/N}  逆时针
     *
     * FFT 和 IFFT 唯一的区别：符号取反
     */
    for (int k = 0; k < N; k++) {
        double angle;
        if (inverse) {
            angle =  2.0 * M_PI * k / N;   /* IFFT: 正号 */
        } else {
            angle = -2.0 * M_PI * k / N;   /* FFT:  负号 */
        }
        twiddles[k].re = cos(angle);  /* 预算好存表里 */
        twiddles[k].im = sin(angle);
    }
}

/*=============================================================
 *  第三部分：8 点 FFT（最简单的 radix-2 DIT）
 *
 *  N=8, log₂8 = 3 级，每级 N/2 = 4 个蝶式
 *
 *  这就是 KISS FFT 里 kf_bfly2 的教学版
 *=============================================================*/

/*
 * 一个蝶式运算：
 *
 *   输入: a, b, 旋转因子 w
 *   输出: a' = a + w·b
 *         b' = a - w·b
 *
 *   为什么能工作：
 *     偶数项 DFT = E[k], 奇数项 DFT = O[k]
 *     奇数项比偶数项在时域上晚了 1 个采样
 *     → 频域上每个频率多转了一个角度
 *     → 用旋转因子 w 补偿这个角度
 *     → 然后加减合并，一对蝶式算出两个频点
 */
void butterfly(Complex *a, Complex *b, Complex w) {
    Complex t = c_mul(w, *b);  /* t = w × b（旋转） */
    *b = c_sub(*a, t);         /* b' = a - t（减） */
    *a = c_add(*a, t);         /* a' = a + t（加） */
}

/*
 * bit-reverse 排列：
 *   FFT 的分治结构要求输入按 bit-reverse 顺序排列
 *
 *   N=8 时:
 *     正常顺序: 0(000) 1(001) 2(010) 3(011) 4(100) 5(101) 6(110) 7(111)
 *     反转后:   0(000) 4(100) 2(010) 6(110) 1(001) 5(101) 3(011) 7(111)
 *
 *     0→0, 1→4, 2→2, 3→6, 4→1, 5→5, 6→3, 7→7
 */
void bit_reverse(Complex *x) {
    /* N=8 的硬编码 bit-reverse */
    Complex tmp;

    /* 交换 1(001) 和 4(100) */
    tmp = x[1]; x[1] = x[4]; x[4] = tmp;

    /* 交换 3(011) 和 6(110) */
    tmp = x[3]; x[3] = x[6]; x[6] = tmp;

    /* 0,2,5,7 不变 */
}

/*
 * FFT 主函数
 *
 * 3 级蝶式级联：
 *   第 1 级: 间距 1, 每组 1 个蝶式, 共 4 组
 *   第 2 级: 间距 2, 每组 2 个蝶式, 共 2 组
 *   第 3 级: 间距 4, 每组 4 个蝶式, 共 1 组
 */
void fft_8(Complex *x) {
    int m, k, idx1, idx2;

    /* 步骤 1: bit-reverse 排列 */
    bit_reverse(x);

    /* 步骤 2: 3 级蝶式 */
    for (int stage = 1; stage <= 3; stage++) {
        /*
         * stage=1: half=1 (配对间距1)
         * stage=2: half=2 (配对间距2)
         * stage=3: half=4 (配对间距4)
         */
        int half = 1 << (stage - 1);   /* 2^(stage-1) */
        int step = 1 << stage;         /* 2^stage */

        for (k = 0; k < N; k += step) {          /* 每组的起始 */
            for (m = 0; m < half; m++) {           /* 组内每个蝶式 */
                idx1 = k + m;                       /* 上半 */
                idx2 = k + m + half;                /* 下半 */

                /*
                 * 旋转因子索引 = m × (N / step)
                 *
                 * stage=1: tw_idx = m×4 (用 W⁰, W⁴)
                 * stage=2: tw_idx = m×2 (用 W⁰, W², W⁴, W⁶)
                 * stage=3: tw_idx = m×1 (用 W⁰, W¹, W², ..., W⁷)
                 */
                int tw_idx = m * (N / step);

                butterfly(&x[idx1], &x[idx2], twiddles[tw_idx]);
            }
        }
    }
}

/* IFFT：完全相同的代码，只是旋转因子表方向反了 */
void ifft_8(Complex *x) {
    fft_8(x);          /* 同样的蝶式运算 */

    /* IFFT 需要除以 N */
    for (int i = 0; i < N; i++) {
        x[i].re /= N;
        x[i].im /= N;
    }
}

/*=============================================================
 *  第四部分：STFT
 *
 *  输入: 长信号 signal[], 长度 signal_len
 *  输出: 每一帧的 FFT 结果 stft_data[][]
 *
 *  步骤: 分帧 → 加窗 → 逐帧 FFT
 *=============================================================*/

#define FRAME_LEN   N       /* 帧长 = FFT 长度 = 8 */
#define HOP_SIZE    (N/2)   /* 帧移 = 4（50% 重叠） */
#define NUM_FRAMES  4       /* 帧数，根据信号长度算 */

/* Hann 窗函数 */
double window[N];

void init_window() {
    for (int n = 0; n < N; n++) {
        window[n] = 0.5 * (1.0 - cos(2.0 * M_PI * n / (N - 1)));
    }

    printf("=== Hann 窗 ===\n");
    printf("n:   ");
    for (int n = 0; n < N; n++) printf("%8d", n);
    printf("\nw[n]:");
    for (int n = 0; n < N; n++) printf("%8.4f", window[n]);
    printf("\n\n");
}

/*
 * STFT：把长信号切成帧，逐帧做 FFT
 *
 * 信号:  [x0  x1  x2  x3  x4  x5  x6  x7  x8  x9  x10 x11 x12 x13 x14 x15 ...]
 * 帧 0:  [x0  x1  x2  x3  x4  x5  x6  x7]
 * 帧 1:          [x4  x5  x6  x7  x8  x9  x10 x11]
 * 帧 2:                  [x8  x9  x10 x11 x12 x13 x14 x15]
 *
 * 每帧加窗后做 FFT
 */
void do_stft(const double *signal, int signal_len,
             Complex stft_out[][N], int *num_frames)
{
    *num_frames = (signal_len - FRAME_LEN) / HOP_SIZE + 1;

    printf("=== STFT ===\n");
    printf("信号长度: %d, 帧长: %d, 帧移: %d, 帧数: %d\n\n",
           signal_len, FRAME_LEN, HOP_SIZE, *num_frames);

    for (int m = 0; m < *num_frames; m++) {
        int offset = m * HOP_SIZE;

        printf("--- 第 %d 帧 (起始位置 n=%d) ---\n", m, offset);

        /* 步骤 1: 取出一帧，加窗 */
        Complex frame[N];
        printf("  加窗前: ");
        for (int n = 0; n < N; n++) {
            frame[n].re = signal[offset + n] * window[n];  /* 乘窗 */
            frame[n].im = 0.0;                              /* 实数信号，虚部=0 */
            printf("%6.2f ", signal[offset + n]);
        }
        printf("\n");

        printf("  加窗后: ");
        for (int n = 0; n < N; n++) {
            printf("%6.2f ", frame[n].re);
        }
        printf("\n");

        /* 步骤 2: 做 FFT */
        init_twiddles(0);   /* 正变换 */
        fft_8(frame);

        /* 步骤 3: 存储结果 */
        printf("  FFT结果:\n");
        printf("    k  |  X[k] 实部  |  X[k] 虚部  |  幅度(强度) |  相位(角度)\n");
        printf("    ---|-------------|-------------|-------------|------------\n");
        for (int k = 0; k < N; k++) {
            stft_out[m][k] = frame[k];
            printf("    %d  |  %8.4f   |  %8.4f   |   %8.4f  |  %7.2f°\n",
                   k, frame[k].re, frame[k].im,
                   c_abs(frame[k]),          /* 幅度/强度 = |X[k]| */
                   c_phase(frame[k]) * 180.0 / M_PI);  /* 相位/角度 */
        }
        printf("\n");
    }
}

/*=============================================================
 *  第五部分：ISTFT
 *
 *  输入: STFT 结果 stft_data[][]
 *  输出: 重建信号 output[]
 *
 *  步骤: 逐帧 IFFT → 加合成窗 → 重叠相加 → 归一化
 *=============================================================*/

void do_istft(Complex stft_in[][N], int num_frames,
              double *output, double *norm_buf, int signal_len)
{
    printf("=== ISTFT ===\n\n");

    for (int m = 0; m < num_frames; m++) {
        int offset = m * HOP_SIZE;

        printf("--- 第 %d 帧 IFFT ---\n", m);

        /* 步骤 1: 复制一帧频域数据 */
        Complex frame[N];
        for (int n = 0; n < N; n++) {
            frame[n] = stft_in[m][n];
        }

        /* 步骤 2: IFFT（逆变换） */
        init_twiddles(1);       /* 逆变换：旋转因子方向取反 */
        ifft_8(frame);          /* 同样的蝶式运算 + 除以 N */

        printf("  IFFT 结果: ");
        for (int n = 0; n < N; n++) {
            printf("%6.3f ", frame[n].re);
        }
        printf("\n");

        /* 步骤 3: 加合成窗（这里合成窗=分析窗=Hann窗） */
        printf("  加合成窗:  ");
        for (int n = 0; n < N; n++) {
            frame[n].re *= window[n];
            printf("%6.3f ", frame[n].re);
        }
        printf("\n");

        /* 步骤 4: 重叠相加 */
        printf("  重叠相加到位置 %d:\n", offset);
        for (int n = 0; n < N; n++) {
            if (offset + n < signal_len) {
                output[offset + n]   += frame[n].re;           /* 累加信号 */
                norm_buf[offset + n] += window[n] * window[n]; /* 累加窗的平方 */
            }
        }

        printf("\n");
    }

    /* 步骤 5: 归一化（补偿窗重叠造成的幅度变化） */
    printf("--- 归一化 ---\n");
    printf("  位置  |  累加值  |  窗累积  |  最终值\n");
    printf("  ------|----------|----------|--------\n");
    for (int n = 0; n < signal_len; n++) {
        if (norm_buf[n] > 1e-10) {
            output[n] /= norm_buf[n];   /* 除以窗的累积 */
        }
        printf("  %4d  | %7.4f  | %7.4f  | %7.4f\n",
               n, output[n] * norm_buf[n], norm_buf[n], output[n]);
    }
    printf("\n");
}

/*=============================================================
 *  主函数：完整演示
 *=============================================================*/

int main() {
    /*
     * 原始信号：16 个采样点
     * 模拟一个低频 + 高频叠加的信号
     *
     * 这里用简单数值方便手算验证
     */
    double signal[] = {
        1.0,  2.0,  1.5,  0.5,
       -0.5, -1.0, -0.5,  0.5,
        1.0,  2.0,  1.5,  0.5,
       -0.5, -1.0, -0.5,  0.5
    };
    int signal_len = 16;

    /* 也准备一个纯正弦波信号方便理解 */
    double sine_signal[16];
    printf("====================================\n");
    printf("  原始信号（16 个采样点）\n");
    printf("====================================\n");
    printf("n:     ");
    for (int n = 0; n < signal_len; n++) {
        sine_signal[n] = sin(2.0 * M_PI * n / 8);  /* 频率=1 的正弦波 */
        printf("%6d ", n);
    }
    printf("\n");
    printf("x[n]:  ");
    for (int n = 0; n < signal_len; n++) {
        printf("%6.2f ", signal[n]);
    }
    printf("\n\n");

    /* 初始化窗函数 */
    init_window();

    /* =============================================
     *  STFT：时域 → 时频域
     * ============================================= */
    Complex stft_data[NUM_FRAMES][N];
    int num_frames = 0;

    do_stft(signal, signal_len, stft_data, &num_frames);

    /* =============================================
     *  ISTFT：时频域 → 重建时域
     * ============================================= */
    double *output   = (double *)calloc(signal_len, sizeof(double));
    double *norm_buf = (double *)calloc(signal_len, sizeof(double));

    do_istft(stft_data, num_frames, output, norm_buf, signal_len);

    /* =============================================
     *  对比：原始信号 vs 重建信号
     * ============================================= */
    printf("====================================\n");
    printf("  最终对比：原始 vs 重建\n");
    printf("====================================\n");
    printf("n:      原始     重建     误差\n");
    printf("----   ------   ------   ------\n");
    double max_err = 0;
    for (int n = 0; n < signal_len; n++) {
        double err = fabs(signal[n] - output[n]);
        if (err > max_err) max_err = err;
        printf(" %2d:   %6.3f   %6.3f   %6.4f\n", n, signal[n], output[n], err);
    }
    printf("\n最大误差: %e\n", max_err);

    if (max_err < 0.01) {
        printf("重建成功！误差小于 1%%\n");
    }

    free(output);
    free(norm_buf);

    return 0;
}
