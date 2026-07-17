//#include "ggml_fft.h"
//#include "ggml_cpu_flag.h"

#include <math.h>
#include <string.h>
#include <stdalign.h>
#include "fft.h"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
//#define USE_SIMD
#elif defined(__aarch64__) || defined(_M_ARM64)
#define SIMDE_ENABLE_NATIVE_ALIASES
#include <simde/x86/avx2.h>
#include <simde/x86/fma.h>
//#define USE_SIMD
#endif

#ifdef USE_SIMD
void kf_bfly2(
    float* Foutr,
    float* Fouti,
    int          fstride,
    const float* tw_r,
    const float* tw_i,
    int          m)
{
    int i = 0;
    if (fstride == 1)
        for (; i + 7 < m; i += 8)
        {
            __m256 F2r = _mm256_loadu_ps(Foutr + m + i);
            __m256 F2i = _mm256_loadu_ps(Fouti + m + i);
            __m256 Twr = _mm256_loadu_ps(tw_r + i);
            __m256 Twi = _mm256_loadu_ps(tw_i + i);

            __m256 tr = _mm256_fmsub_ps(F2r, Twr, _mm256_mul_ps(F2i, Twi));
            __m256 ti = _mm256_fmadd_ps(F2r, Twi, _mm256_mul_ps(F2i, Twr));

            __m256 Fr = _mm256_loadu_ps(Foutr + i);
            __m256 Fi = _mm256_loadu_ps(Fouti + i);

            _mm256_storeu_ps(Foutr + m + i, _mm256_sub_ps(Fr, tr));
            _mm256_storeu_ps(Fouti + m + i, _mm256_sub_ps(Fi, ti));
            _mm256_storeu_ps(Foutr + i, _mm256_add_ps(Fr, tr));
            _mm256_storeu_ps(Fouti + i, _mm256_add_ps(Fi, ti));
        }
    else for (const __m256i vidx0 = _mm256_setr_epi32(
        0 * fstride, 1 * fstride, 2 * fstride, 3 * fstride,
        4 * fstride, 5 * fstride, 6 * fstride, 7 * fstride); i + 7 < m; i += 8)
    {
        __m256 F2r = _mm256_loadu_ps(Foutr + m + i);
        __m256 F2i = _mm256_loadu_ps(Fouti + m + i);
        __m256 Twr = _mm256_i32gather_ps(tw_r + i * fstride, vidx0, sizeof(float));
        __m256 Twi = _mm256_i32gather_ps(tw_i + i * fstride, vidx0, sizeof(float));

        __m256 tr = _mm256_fmsub_ps(F2r, Twr, _mm256_mul_ps(F2i, Twi));
        __m256 ti = _mm256_fmadd_ps(F2r, Twi, _mm256_mul_ps(F2i, Twr));

        __m256 Fr = _mm256_loadu_ps(Foutr + i);
        __m256 Fi = _mm256_loadu_ps(Fouti + i);

        _mm256_storeu_ps(Foutr + m + i, _mm256_sub_ps(Fr, tr));
        _mm256_storeu_ps(Fouti + m + i, _mm256_sub_ps(Fi, ti));
        _mm256_storeu_ps(Foutr + i, _mm256_add_ps(Fr, tr));
        _mm256_storeu_ps(Fouti + i, _mm256_add_ps(Fi, ti));
    }

    for (; i + 3 < m; i += 4)
    {
        __m128 F2r = _mm_loadu_ps(Foutr + m + i);
        __m128 F2i = _mm_loadu_ps(Fouti + m + i);
        __m128 Twr;
        __m128 Twi;
        if (fstride == 1)
        {
            Twr = _mm_loadu_ps(tw_r + i);
            Twi = _mm_loadu_ps(tw_i + i);
        }
        else
        {
            const __m128i vidx0_128 = _mm_setr_epi32(
                0 * fstride, 1 * fstride, 2 * fstride, 3 * fstride);
            Twr = _mm_i32gather_ps(tw_r + i * fstride, vidx0_128, sizeof(float));
            Twi = _mm_i32gather_ps(tw_i + i * fstride, vidx0_128, sizeof(float));
        }

        __m128 tr = _mm_fmsub_ps(F2r, Twr, _mm_mul_ps(F2i, Twi));
        __m128 ti = _mm_fmadd_ps(F2r, Twi, _mm_mul_ps(F2i, Twr));

        __m128 Fr = _mm_loadu_ps(Foutr + i);
        __m128 Fi = _mm_loadu_ps(Fouti + i);

        _mm_storeu_ps(Foutr + m + i, _mm_sub_ps(Fr, tr));
        _mm_storeu_ps(Fouti + m + i, _mm_sub_ps(Fi, ti));
        _mm_storeu_ps(Foutr + i, _mm_add_ps(Fr, tr));
        _mm_storeu_ps(Fouti + i, _mm_add_ps(Fi, ti));
    }

    for (; i < m; ++i)
    {
        const float twr = tw_r[i * fstride];
        const float twi = tw_i[i * fstride];

        float F2r = Foutr[m + i];
        float F2i = Fouti[m + i];

        float tr = F2r * twr - F2i * twi;
        float ti = F2r * twi + F2i * twr;

        float Fr = Foutr[i];
        float Fi = Fouti[i];

        Foutr[m + i] = Fr - tr;
        Fouti[m + i] = Fi - ti;
        Foutr[i] = Fr + tr;
        Fouti[i] = Fi + ti;
    }
}

void kf_bfly4(
    float* Foutr,
    float* Fouti,
    int fstride,
    float* twr,
    float* twi,
    int m)
{
    const int m2 = 2 * m;
    const int m3 = 3 * m;

    int k = 0;
    for (const __m256i vidx0 = _mm256_setr_epi32(
        0, fstride, 2 * fstride, 3 * fstride,
        4 * fstride, 5 * fstride, 6 * fstride, 7 * fstride),
        vidx1 = _mm256_mullo_epi32(
            vidx0, _mm256_set1_epi32(2)),  /* 2 * fstride */
        vidx2 = _mm256_mullo_epi32(
            vidx0, _mm256_set1_epi32(3)); k + 7 < m; k += 8)
    {
        float* p0r = Foutr + k;
        float* p0i = Fouti + k;

        float* p1r = Foutr + k + m;
        float* p1i = Fouti + k + m;

        float* p2r = Foutr + k + m2;
        float* p2i = Fouti + k + m2;

        float* p3r = Foutr + k + m3;
        float* p3i = Fouti + k + m3;

        __m256 r0 = _mm256_loadu_ps(p0r);
        __m256 i0 = _mm256_loadu_ps(p0i);

        __m256 r1 = _mm256_loadu_ps(p1r);
        __m256 i1 = _mm256_loadu_ps(p1i);

        __m256 r2 = _mm256_loadu_ps(p2r);
        __m256 i2 = _mm256_loadu_ps(p2i);

        __m256 r3 = _mm256_loadu_ps(p3r);
        __m256 i3 = _mm256_loadu_ps(p3i);

        __m256 t1r, t1i, t2r, t2i, t3r, t3i;

        t1r = _mm256_i32gather_ps(twr + k * fstride, vidx0, 4);
        t1i = _mm256_i32gather_ps(twi + k * fstride, vidx0, 4);

        t2r = _mm256_i32gather_ps(twr + 2 * k * fstride,
            vidx1, 4);
        t2i = _mm256_i32gather_ps(twi + 2 * k * fstride,
            vidx1, 4);

        t3r = _mm256_i32gather_ps(twr + 3 * k * fstride,
            vidx2, 4);
        t3i = _mm256_i32gather_ps(twi + 3 * k * fstride,
            vidx2, 4);

        __m256 s0r = _mm256_fmsub_ps(r1, t1r, _mm256_mul_ps(i1, t1i));
        __m256 s0i = _mm256_fmadd_ps(r1, t1i, _mm256_mul_ps(i1, t1r));

        __m256 s1r = _mm256_fmsub_ps(r2, t2r, _mm256_mul_ps(i2, t2i));
        __m256 s1i = _mm256_fmadd_ps(r2, t2i, _mm256_mul_ps(i2, t2r));

        __m256 s2r = _mm256_fmsub_ps(r3, t3r, _mm256_mul_ps(i3, t3i));
        __m256 s2i = _mm256_fmadd_ps(r3, t3i, _mm256_mul_ps(i3, t3r));

        __m256 tmp_r = _mm256_sub_ps(r0, s1r);
        __m256 tmp_i = _mm256_sub_ps(i0, s1i);

        r0 = _mm256_add_ps(r0, s1r);
        i0 = _mm256_add_ps(i0, s1i);

        __m256 s3r = _mm256_add_ps(s0r, s2r);
        __m256 s3i = _mm256_add_ps(s0i, s2i);

        __m256 s4r = _mm256_sub_ps(s0r, s2r);
        __m256 s4i = _mm256_sub_ps(s0i, s2i);

        _mm256_storeu_ps(p2r, _mm256_sub_ps(r0, s3r));
        _mm256_storeu_ps(p2i, _mm256_sub_ps(i0, s3i));

        _mm256_storeu_ps(p0r, _mm256_add_ps(r0, s3r));
        _mm256_storeu_ps(p0i, _mm256_add_ps(i0, s3i));

        _mm256_storeu_ps(p1r, _mm256_add_ps(tmp_r, s4i));
        _mm256_storeu_ps(p1i, _mm256_sub_ps(tmp_i, s4r));

        _mm256_storeu_ps(p3r, _mm256_sub_ps(tmp_r, s4i));
        _mm256_storeu_ps(p3i, _mm256_add_ps(tmp_i, s4r));
    }

    for (const __m128i vidx0 = _mm_setr_epi32(
        0 * fstride, 1 * fstride, 2 * fstride, 3 * fstride),
        vidx1 = _mm_mullo_epi32(
            vidx0, _mm_set1_epi32(2)),  /* 2 * fstride */
        vidx2 = _mm_mullo_epi32(
            vidx0, _mm_set1_epi32(3))  /* 3 * fstride */; k + 3 < m; k += 4)
    {
        float* p0r = Foutr + k;
        float* p0i = Fouti + k;

        float* p1r = Foutr + k + m;
        float* p1i = Fouti + k + m;

        float* p2r = Foutr + k + m2;
        float* p2i = Fouti + k + m2;

        float* p3r = Foutr + k + m3;
        float* p3i = Fouti + k + m3;

        __m128 r0 = _mm_loadu_ps(p0r);
        __m128 i0 = _mm_loadu_ps(p0i);

        __m128 r1 = _mm_loadu_ps(p1r);
        __m128 i1 = _mm_loadu_ps(p1i);

        __m128 r2 = _mm_loadu_ps(p2r);
        __m128 i2 = _mm_loadu_ps(p2i);

        __m128 r3 = _mm_loadu_ps(p3r);
        __m128 i3 = _mm_loadu_ps(p3i);

        __m128 t1r, t1i, t2r, t2i, t3r, t3i;

        t1r = _mm_i32gather_ps(twr + k * fstride, vidx0, 4);
        t1i = _mm_i32gather_ps(twi + k * fstride, vidx0, 4);

        t2r = _mm_i32gather_ps(twr + 2 * k * fstride,
            vidx1, 4);
        t2i = _mm_i32gather_ps(twi + 2 * k * fstride,
            vidx1, 4);

        t3r = _mm_i32gather_ps(twr + 3 * k * fstride,
            vidx2, 4);
        t3i = _mm_i32gather_ps(twi + 3 * k * fstride,
            vidx2, 4);

        __m128 s0r = _mm_fmsub_ps(r1, t1r, _mm_mul_ps(i1, t1i));
        __m128 s0i = _mm_fmadd_ps(r1, t1i, _mm_mul_ps(i1, t1r));

        __m128 s1r = _mm_fmsub_ps(r2, t2r, _mm_mul_ps(i2, t2i));
        __m128 s1i = _mm_fmadd_ps(r2, t2i, _mm_mul_ps(i2, t2r));

        __m128 s2r = _mm_fmsub_ps(r3, t3r, _mm_mul_ps(i3, t3i));
        __m128 s2i = _mm_fmadd_ps(r3, t3i, _mm_mul_ps(i3, t3r));

        __m128 tmp_r = _mm_sub_ps(r0, s1r);
        __m128 tmp_i = _mm_sub_ps(i0, s1i);

        r0 = _mm_add_ps(r0, s1r);
        i0 = _mm_add_ps(i0, s1i);

        __m128 s3r = _mm_add_ps(s0r, s2r);
        __m128 s3i = _mm_add_ps(s0i, s2i);

        __m128 s4r = _mm_sub_ps(s0r, s2r);
        __m128 s4i = _mm_sub_ps(s0i, s2i);

        _mm_storeu_ps(p2r, _mm_sub_ps(r0, s3r));
        _mm_storeu_ps(p2i, _mm_sub_ps(i0, s3i));

        _mm_storeu_ps(p0r, _mm_add_ps(r0, s3r));
        _mm_storeu_ps(p0i, _mm_add_ps(i0, s3i));

        _mm_storeu_ps(p1r, _mm_add_ps(tmp_r, s4i));
        _mm_storeu_ps(p1i, _mm_sub_ps(tmp_i, s4r));

        _mm_storeu_ps(p3r, _mm_sub_ps(tmp_r, s4i));
        _mm_storeu_ps(p3i, _mm_add_ps(tmp_i, s4r));
    }

    for (Foutr += k, Fouti += k; k != m; ++k)
    {
        float scratchr[6];
        float scratchi[6];

        scratchr[0] = Foutr[m] * twr[k * fstride] - Fouti[m] * twi[k * fstride];
        scratchi[0] = Foutr[m] * twi[k * fstride] + Fouti[m] * twr[k * fstride];
        scratchr[1] = Foutr[m2] * twr[k * 2 * fstride] - Fouti[m2] * twi[k * 2 * fstride];
        scratchi[1] = Foutr[m2] * twi[k * 2 * fstride] + Fouti[m2] * twr[k * 2 * fstride];
        scratchr[2] = Foutr[m3] * twr[k * 3 * fstride] - Fouti[m3] * twi[k * 3 * fstride];
        scratchi[2] = Foutr[m3] * twi[k * 3 * fstride] + Fouti[m3] * twr[k * 3 * fstride];

        scratchr[5] = *Foutr - scratchr[1];
        scratchi[5] = *Fouti - scratchi[1];
        *Foutr += scratchr[1];
        *Fouti += scratchi[1];

        scratchr[3] = scratchr[0] + scratchr[2];
        scratchi[3] = scratchi[0] + scratchi[2];
        scratchr[4] = scratchr[0] - scratchr[2];
        scratchi[4] = scratchi[0] - scratchi[2];

        Foutr[m2] = *Foutr - scratchr[3];
        Fouti[m2] = *Fouti - scratchi[3];

        *Foutr += scratchr[3];
        *Fouti += scratchi[3];

        Foutr[m] = scratchr[5] + scratchi[4];
        Fouti[m] = scratchi[5] - scratchr[4];
        Foutr[m3] = scratchr[5] - scratchi[4];
        Fouti[m3] = scratchi[5] + scratchr[4];

        ++Foutr;
        ++Fouti;
    }
}

void kf_bfly3(
    float* Foutr,
    float* Fouti,
    int    fstride,
    float* tw1_r,
    float* tw1_i,
    int    m)
{
    const int m2 = 2 * m;
    const float epi3_i = tw1_i[fstride * m];

    int k = 0;

    for (const __m256 v_half = _mm256_set1_ps(0.5f), v_epi3 = _mm256_set1_ps(epi3_i); k + 7 < m; k += 8)
    {
        __m256 fr0 = _mm256_loadu_ps(Foutr + k);
        __m256 fi0 = _mm256_loadu_ps(Fouti + k);
        __m256 fr1 = _mm256_loadu_ps(Foutr + m + k);
        __m256 fi1 = _mm256_loadu_ps(Fouti + m + k);
        __m256 fr2 = _mm256_loadu_ps(Foutr + m2 + k);
        __m256 fi2 = _mm256_loadu_ps(Fouti + m2 + k);

        __m256i idx1 = _mm256_setr_epi32(
            k * fstride, (k + 1) * fstride, (k + 2) * fstride, (k + 3) * fstride,
            (k + 4) * fstride, (k + 5) * fstride, (k + 6) * fstride, (k + 7) * fstride);
        __m256i idx2 = _mm256_mullo_epi32(idx1, _mm256_set1_epi32(2));

        __m256 tw1r = _mm256_i32gather_ps(tw1_r, idx1, 4);
        __m256 tw1i = _mm256_i32gather_ps(tw1_i, idx1, 4);
        __m256 tw2r = _mm256_i32gather_ps(tw1_r, idx2, 4);
        __m256 tw2i = _mm256_i32gather_ps(tw1_i, idx2, 4);

        __m256 sr1 = _mm256_fmsub_ps(fr1, tw1r, _mm256_mul_ps(fi1, tw1i));
        __m256 si1 = _mm256_fmadd_ps(fr1, tw1i, _mm256_mul_ps(fi1, tw1r));
        __m256 sr2 = _mm256_fmsub_ps(fr2, tw2r, _mm256_mul_ps(fi2, tw2i));
        __m256 si2 = _mm256_fmadd_ps(fr2, tw2i, _mm256_mul_ps(fi2, tw2r));

        __m256 sr3 = _mm256_add_ps(sr1, sr2);
        __m256 si3 = _mm256_add_ps(si1, si2);
        __m256 sr0 = _mm256_sub_ps(sr1, sr2);
        __m256 si0 = _mm256_sub_ps(si1, si2);

        __m256 fr1o = _mm256_sub_ps(fr0, _mm256_mul_ps(sr3, v_half));
        __m256 fi1o = _mm256_sub_ps(fi0, _mm256_mul_ps(si3, v_half));

        sr0 = _mm256_mul_ps(sr0, v_epi3);
        si0 = _mm256_mul_ps(si0, v_epi3);

        __m256 fr0o = _mm256_add_ps(fr0, sr3);
        __m256 fi0o = _mm256_add_ps(fi0, si3);

        __m256 fr2o = _mm256_add_ps(fr1o, si0);
        __m256 fi2o = _mm256_sub_ps(fi1o, sr0);

        fr1o = _mm256_sub_ps(fr1o, si0);
        fi1o = _mm256_add_ps(fi1o, sr0);

        _mm256_storeu_ps(Foutr + k, fr0o);
        _mm256_storeu_ps(Fouti + k, fi0o);
        _mm256_storeu_ps(Foutr + m + k, fr1o);
        _mm256_storeu_ps(Fouti + m + k, fi1o);
        _mm256_storeu_ps(Foutr + m2 + k, fr2o);
        _mm256_storeu_ps(Fouti + m2 + k, fi2o);
    }

    for (const __m128 v_half_128 = _mm_set_ps1(0.5f), v_epi3_128 = _mm_set_ps1(epi3_i);
        k + 3 < m; k += 4)
    {
        __m128 fr0 = _mm_loadu_ps(Foutr + k);
        __m128 fi0 = _mm_loadu_ps(Fouti + k);
        __m128 fr1 = _mm_loadu_ps(Foutr + m + k);
        __m128 fi1 = _mm_loadu_ps(Fouti + m + k);
        __m128 fr2 = _mm_loadu_ps(Foutr + m2 + k);
        __m128 fi2 = _mm_loadu_ps(Fouti + m2 + k);

        __m128 tw1r;
        __m128 tw1i;
        __m128 tw2r;
        __m128 tw2i;
        if (fstride == 1)
        {
            tw1r = _mm_loadu_ps(tw1_r + k);
            tw1i = _mm_loadu_ps(tw1_i + k);

            const int s2 = fstride * 2;
            __m128i idx2 = _mm_setr_epi32(
                k * s2, (k + 1) * s2, (k + 2) * s2, (k + 3) * s2);
            tw2r = _mm_i32gather_ps(tw1_r, idx2, 4);
            tw2i = _mm_i32gather_ps(tw1_i, idx2, 4);
        }
        else
        {
            __m128i idx1 = _mm_setr_epi32(
                k * fstride, (k + 1) * fstride, (k + 2) * fstride, (k + 3) * fstride);
            __m128i idx2 = _mm_mullo_epi32(idx1, _mm_set1_epi32(2));

            tw1r = _mm_i32gather_ps(tw1_r, idx1, 4);
            tw1i = _mm_i32gather_ps(tw1_i, idx1, 4);
            tw2r = _mm_i32gather_ps(tw1_r, idx2, 4);
            tw2i = _mm_i32gather_ps(tw1_i, idx2, 4);
        }

        __m128 sr1 = _mm_fmsub_ps(fr1, tw1r, _mm_mul_ps(fi1, tw1i));
        __m128 si1 = _mm_fmadd_ps(fr1, tw1i, _mm_mul_ps(fi1, tw1r));
        __m128 sr2 = _mm_fmsub_ps(fr2, tw2r, _mm_mul_ps(fi2, tw2i));
        __m128 si2 = _mm_fmadd_ps(fr2, tw2i, _mm_mul_ps(fi2, tw2r));

        __m128 sr3 = _mm_add_ps(sr1, sr2);
        __m128 si3 = _mm_add_ps(si1, si2);
        __m128 sr0 = _mm_sub_ps(sr1, sr2);
        __m128 si0 = _mm_sub_ps(si1, si2);

        __m128 fr1o = _mm_sub_ps(fr0, _mm_mul_ps(sr3, v_half_128));
        __m128 fi1o = _mm_sub_ps(fi0, _mm_mul_ps(si3, v_half_128));

        sr0 = _mm_mul_ps(sr0, v_epi3_128);
        si0 = _mm_mul_ps(si0, v_epi3_128);

        __m128 fr0o = _mm_add_ps(fr0, sr3);
        __m128 fi0o = _mm_add_ps(fi0, si3);

        __m128 fr2o = _mm_add_ps(fr1o, si0);
        __m128 fi2o = _mm_sub_ps(fi1o, sr0);

        fr1o = _mm_sub_ps(fr1o, si0);
        fi1o = _mm_add_ps(fi1o, sr0);

        _mm_storeu_ps(Foutr + k, fr0o);
        _mm_storeu_ps(Fouti + k, fi0o);
        _mm_storeu_ps(Foutr + m + k, fr1o);
        _mm_storeu_ps(Fouti + m + k, fi1o);
        _mm_storeu_ps(Foutr + m2 + k, fr2o);
        _mm_storeu_ps(Fouti + m2 + k, fi2o);
    }

    for (; k != m; ++k)
    {
        float tw1r = tw1_r[k * fstride];
        float tw1i = tw1_i[k * fstride];
        float tw2r = tw1_r[k * fstride * 2];
        float tw2i = tw1_i[k * fstride * 2];

        float fr0 = Foutr[k];
        float fi0 = Fouti[k];
        float fr1 = Foutr[m + k];
        float fi1 = Fouti[m + k];
        float fr2 = Foutr[m2 + k];
        float fi2 = Fouti[m2 + k];

        float sr1 = fr1 * tw1r - fi1 * tw1i;
        float si1 = fr1 * tw1i + fi1 * tw1r;
        float sr2 = fr2 * tw2r - fi2 * tw2i;
        float si2 = fr2 * tw2i + fi2 * tw2r;

        float sr3 = sr1 + sr2;
        float si3 = si1 + si2;
        float sr0 = sr1 - sr2;
        float si0 = si1 - si2;

        float fr1o = fr0 - sr3 * 0.5f;
        float fi1o = fi0 - si3 * 0.5f;

        sr0 *= epi3_i;
        si0 *= epi3_i;

        Foutr[k] = fr0 + sr3;
        Fouti[k] = fi0 + si3;
        Foutr[m + k] = fr1o - si0;
        Fouti[m + k] = fi1o + sr0;
        Foutr[m2 + k] = fr1o + si0;
        Fouti[m2 + k] = fi1o - sr0;
    }
}

inline
static void CMUL_256(__m256* ar, __m256* ai, __m256 br, __m256 bi)
{
    __m256 _tr = _mm256_fmsub_ps(*ar, br, _mm256_mul_ps(*ai, bi));
    __m256 _ti = _mm256_fmadd_ps(*ar, bi, _mm256_mul_ps(*ai, br));
    *ar = _tr; *ai = _ti;
}

inline
static void CMUL_128(__m128* ar, __m128* ai, __m128 br, __m128 bi)
{
    __m128 _tr = _mm_fmsub_ps(*ar, br, _mm_mul_ps(*ai, bi));
    __m128 _ti = _mm_fmadd_ps(*ar, bi, _mm_mul_ps(*ai, br));
    *ar = _tr; *ai = _ti;
}

void kf_bfly5(
    float* Foutr, float* Fouti,
    int fstride,
    const float* tw_r, const float* tw_i,
    int m)
{
    int u = 0;

    float yar = tw_r[fstride * m];
    float yai = tw_i[fstride * m];
    float ybr = tw_r[fstride * 2 * m];
    float ybi = tw_i[fstride * 2 * m];

    float* F0r = Foutr, * F1r = Foutr + m, * F2r = Foutr + 2 * m,
        * F3r = Foutr + 3 * m, * F4r = Foutr + 4 * m;
    float* F0i = Fouti, * F1i = Fouti + m, * F2i = Fouti + 2 * m,
        * F3i = Fouti + 3 * m, * F4i = Fouti + 4 * m;

    for (const __m256 v_yar = _mm256_set1_ps(yar),
        v_yai = _mm256_set1_ps(yai),
        v_ybr = _mm256_set1_ps(ybr),
        v_ybi = _mm256_set1_ps(ybi); u + 7 < m; u += 8) {
        __m256i idx1 = _mm256_setr_epi32(
            u * fstride, (u + 1) * fstride, (u + 2) * fstride, (u + 3) * fstride,
            (u + 4) * fstride, (u + 5) * fstride, (u + 6) * fstride, (u + 7) * fstride
        );
        __m256 tw1r = _mm256_i32gather_ps(tw_r, idx1, 4);
        __m256 tw1i = _mm256_i32gather_ps(tw_i, idx1, 4);

        __m256i idx2 = _mm256_mullo_epi32(idx1, _mm256_set1_epi32(2));
        __m256 tw2r = _mm256_i32gather_ps(tw_r, idx2, 4);
        __m256 tw2i = _mm256_i32gather_ps(tw_i, idx2, 4);

        __m256i idx3 = _mm256_mullo_epi32(idx1, _mm256_set1_epi32(3));
        __m256 tw3r = _mm256_i32gather_ps(tw_r, idx3, 4);
        __m256 tw3i = _mm256_i32gather_ps(tw_i, idx3, 4);

        __m256i idx4 = _mm256_mullo_epi32(idx1, _mm256_set1_epi32(4));
        __m256 tw4r = _mm256_i32gather_ps(tw_r, idx4, 4);
        __m256 tw4i = _mm256_i32gather_ps(tw_i, idx4, 4);

        __m256 f0r = _mm256_loadu_ps(F0r + u);
        __m256 f0i = _mm256_loadu_ps(F0i + u);
        __m256 f1r = _mm256_loadu_ps(F1r + u);
        __m256 f1i = _mm256_loadu_ps(F1i + u);
        __m256 f2r = _mm256_loadu_ps(F2r + u);
        __m256 f2i = _mm256_loadu_ps(F2i + u);
        __m256 f3r = _mm256_loadu_ps(F3r + u);
        __m256 f3i = _mm256_loadu_ps(F3i + u);
        __m256 f4r = _mm256_loadu_ps(F4r + u);
        __m256 f4i = _mm256_loadu_ps(F4i + u);

        CMUL_256(&f1r, &f1i, tw1r, tw1i);
        CMUL_256(&f2r, &f2i, tw2r, tw2i);
        CMUL_256(&f3r, &f3i, tw3r, tw3i);
        CMUL_256(&f4r, &f4i, tw4r, tw4i);

        __m256 s7r = _mm256_add_ps(f1r, f4r);
        __m256 s7i = _mm256_add_ps(f1i, f4i);
        __m256 s10r = _mm256_sub_ps(f1r, f4r);
        __m256 s10i = _mm256_sub_ps(f1i, f4i);
        __m256 s8r = _mm256_add_ps(f2r, f3r);
        __m256 s8i = _mm256_add_ps(f2i, f3i);
        __m256 s9r = _mm256_sub_ps(f2r, f3r);
        __m256 s9i = _mm256_sub_ps(f2i, f3i);

        __m256 s5r = _mm256_fmadd_ps(s7r, v_yar, f0r);
        s5r = _mm256_fmadd_ps(s8r, v_ybr, s5r);
        __m256 s5i = _mm256_fmadd_ps(s7i, v_yar, f0i);
        s5i = _mm256_fmadd_ps(s8i, v_ybr, s5i);

        __m256 s6r = _mm256_fmadd_ps(s10i, v_yai, _mm256_mul_ps(s9i, v_ybi));
        __m256 s6i = _mm256_fnmsub_ps(s10r, v_yai, _mm256_mul_ps(s9r, v_ybi));

        __m256 t1r = _mm256_sub_ps(s5r, s6r);
        __m256 t1i = _mm256_sub_ps(s5i, s6i);
        __m256 t4r = _mm256_add_ps(s5r, s6r);
        __m256 t4i = _mm256_add_ps(s5i, s6i);

        __m256 s11r = _mm256_fmadd_ps(s7r, v_ybr, f0r);
        s11r = _mm256_fmadd_ps(s8r, v_yar, s11r);
        __m256 s11i = _mm256_fmadd_ps(s7i, v_ybr, f0i);
        s11i = _mm256_fmadd_ps(s8i, v_yar, s11i);

        __m256 tmp = _mm256_add_ps(s7r, s8r);
        f0r = _mm256_add_ps(f0r, tmp);
        tmp = _mm256_add_ps(s7i, s8i);
        f0i = _mm256_add_ps(f0i, tmp);

        __m256 s12r = _mm256_fmsub_ps(s9i, v_yai, _mm256_mul_ps(s10i, v_ybi));
        __m256 s12i = _mm256_fmsub_ps(s10r, v_ybi, _mm256_mul_ps(s9r, v_yai));

        __m256 t2r = _mm256_add_ps(s11r, s12r);
        __m256 t2i = _mm256_add_ps(s11i, s12i);
        __m256 t3r = _mm256_sub_ps(s11r, s12r);
        __m256 t3i = _mm256_sub_ps(s11i, s12i);

        _mm256_storeu_ps(F0r + u, f0r);
        _mm256_storeu_ps(F0i + u, f0i);
        _mm256_storeu_ps(F1r + u, t1r);
        _mm256_storeu_ps(F1i + u, t1i);
        _mm256_storeu_ps(F2r + u, t2r);
        _mm256_storeu_ps(F2i + u, t2i);
        _mm256_storeu_ps(F3r + u, t3r);
        _mm256_storeu_ps(F3i + u, t3i);
        _mm256_storeu_ps(F4r + u, t4r);
        _mm256_storeu_ps(F4i + u, t4i);
    }

    for (const __m128 v_yar = _mm_set_ps1(yar),
        v_yai = _mm_set_ps1(yai),
        v_ybr = _mm_set_ps1(ybr),
        v_ybi = _mm_set_ps1(ybi); u + 3 < m; u += 4)
    {
        __m128i idx1 = _mm_setr_epi32(
            u * fstride, (u + 1) * fstride, (u + 2) * fstride, (u + 3) * fstride);
        __m128 tw1r = _mm_i32gather_ps(tw_r, idx1, 4);
        __m128 tw1i = _mm_i32gather_ps(tw_i, idx1, 4);

        __m128i idx2 = _mm_mullo_epi32(idx1, _mm_set1_epi32(2));
        __m128 tw2r = _mm_i32gather_ps(tw_r, idx2, 4);
        __m128 tw2i = _mm_i32gather_ps(tw_i, idx2, 4);

        __m128i idx3 = _mm_mullo_epi32(idx1, _mm_set1_epi32(3));
        __m128 tw3r = _mm_i32gather_ps(tw_r, idx3, 4);
        __m128 tw3i = _mm_i32gather_ps(tw_i, idx3, 4);

        __m128i idx4 = _mm_mullo_epi32(idx1, _mm_set1_epi32(4));
        __m128 tw4r = _mm_i32gather_ps(tw_r, idx4, 4);
        __m128 tw4i = _mm_i32gather_ps(tw_i, idx4, 4);

        __m128 f0r = _mm_loadu_ps(F0r + u);
        __m128 f0i = _mm_loadu_ps(F0i + u);
        __m128 f1r = _mm_loadu_ps(F1r + u);
        __m128 f1i = _mm_loadu_ps(F1i + u);
        __m128 f2r = _mm_loadu_ps(F2r + u);
        __m128 f2i = _mm_loadu_ps(F2i + u);
        __m128 f3r = _mm_loadu_ps(F3r + u);
        __m128 f3i = _mm_loadu_ps(F3i + u);
        __m128 f4r = _mm_loadu_ps(F4r + u);
        __m128 f4i = _mm_loadu_ps(F4i + u);

        CMUL_128(&f1r, &f1i, tw1r, tw1i);
        CMUL_128(&f2r, &f2i, tw2r, tw2i);
        CMUL_128(&f3r, &f3i, tw3r, tw3i);
        CMUL_128(&f4r, &f4i, tw4r, tw4i);

        __m128 s7r = _mm_add_ps(f1r, f4r);
        __m128 s7i = _mm_add_ps(f1i, f4i);
        __m128 s10r = _mm_sub_ps(f1r, f4r);
        __m128 s10i = _mm_sub_ps(f1i, f4i);
        __m128 s8r = _mm_add_ps(f2r, f3r);
        __m128 s8i = _mm_add_ps(f2i, f3i);
        __m128 s9r = _mm_sub_ps(f2r, f3r);
        __m128 s9i = _mm_sub_ps(f2i, f3i);

        __m128 s5r = _mm_fmadd_ps(s7r, v_yar, f0r);
        s5r = _mm_fmadd_ps(s8r, v_ybr, s5r);
        __m128 s5i = _mm_fmadd_ps(s7i, v_yar, f0i);
        s5i = _mm_fmadd_ps(s8i, v_ybr, s5i);

        __m128 s6r = _mm_fmadd_ps(s10i, v_yai, _mm_mul_ps(s9i, v_ybi));
        __m128 s6i = _mm_fnmsub_ps(s10r, v_yai, _mm_mul_ps(s9r, v_ybi));

        __m128 t1r = _mm_sub_ps(s5r, s6r);
        __m128 t1i = _mm_sub_ps(s5i, s6i);
        __m128 t4r = _mm_add_ps(s5r, s6r);
        __m128 t4i = _mm_add_ps(s5i, s6i);

        __m128 s11r = _mm_fmadd_ps(s7r, v_ybr, f0r);
        s11r = _mm_fmadd_ps(s8r, v_yar, s11r);
        __m128 s11i = _mm_fmadd_ps(s7i, v_ybr, f0i);
        s11i = _mm_fmadd_ps(s8i, v_yar, s11i);

        __m128 tmp = _mm_add_ps(s7r, s8r);
        f0r = _mm_add_ps(f0r, tmp);
        tmp = _mm_add_ps(s7i, s8i);
        f0i = _mm_add_ps(f0i, tmp);

        __m128 s12r = _mm_fmsub_ps(s9i, v_yai, _mm_mul_ps(s10i, v_ybi));
        __m128 s12i = _mm_fmsub_ps(s10r, v_ybi, _mm_mul_ps(s9r, v_yai));

        __m128 t2r = _mm_add_ps(s11r, s12r);
        __m128 t2i = _mm_add_ps(s11i, s12i);
        __m128 t3r = _mm_sub_ps(s11r, s12r);
        __m128 t3i = _mm_sub_ps(s11i, s12i);

        _mm_storeu_ps(F0r + u, f0r);
        _mm_storeu_ps(F0i + u, f0i);
        _mm_storeu_ps(F1r + u, t1r);
        _mm_storeu_ps(F1i + u, t1i);
        _mm_storeu_ps(F2r + u, t2r);
        _mm_storeu_ps(F2i + u, t2i);
        _mm_storeu_ps(F3r + u, t3r);
        _mm_storeu_ps(F3i + u, t3i);
        _mm_storeu_ps(F4r + u, t4r);
        _mm_storeu_ps(F4i + u, t4i);
    }

    for (; u < m; ++u) {
        float* Fout0r = Foutr + u,
            * Fout1r = Foutr + m + u,
            * Fout2r = Foutr + 2 * m + u,
            * Fout3r = Foutr + 3 * m + u,
            * Fout4r = Foutr + 4 * m + u;
        float* Fout0i = Fouti + u,
            * Fout1i = Fouti + m + u,
            * Fout2i = Fouti + 2 * m + u,
            * Fout3i = Fouti + 3 * m + u,
            * Fout4i = Fouti + 4 * m + u;

        float scratchr[13], scratchi[13];

        scratchr[0] = *Fout0r;
        scratchi[0] = *Fout0i;

        scratchr[1] = *Fout1r * tw_r[u * fstride] - *Fout1i * tw_i[u * fstride];
        scratchi[1] = *Fout1r * tw_i[u * fstride] + *Fout1i * tw_r[u * fstride];
        scratchr[2] = *Fout2r * tw_r[2 * u * fstride] - *Fout2i * tw_i[2 * u * fstride];
        scratchi[2] = *Fout2r * tw_i[2 * u * fstride] + *Fout2i * tw_r[2 * u * fstride];
        scratchr[3] = *Fout3r * tw_r[3 * u * fstride] - *Fout3i * tw_i[3 * u * fstride];
        scratchi[3] = *Fout3r * tw_i[3 * u * fstride] + *Fout3i * tw_r[3 * u * fstride];
        scratchr[4] = *Fout4r * tw_r[4 * u * fstride] - *Fout4i * tw_i[4 * u * fstride];
        scratchi[4] = *Fout4r * tw_i[4 * u * fstride] + *Fout4i * tw_r[4 * u * fstride];

        scratchr[7] = scratchr[1] + scratchr[4];
        scratchi[7] = scratchi[1] + scratchi[4];
        scratchr[10] = scratchr[1] - scratchr[4];
        scratchi[10] = scratchi[1] - scratchi[4];
        scratchr[8] = scratchr[2] + scratchr[3];
        scratchi[8] = scratchi[2] + scratchi[3];
        scratchr[9] = scratchr[2] - scratchr[3];
        scratchi[9] = scratchi[2] - scratchi[3];

        *Fout0r += scratchr[7] + scratchr[8];
        *Fout0i += scratchi[7] + scratchi[8];

        scratchr[5] = scratchr[0] + scratchr[7] * yar + scratchr[8] * ybr;
        scratchi[5] = scratchi[0] + scratchi[7] * yar + scratchi[8] * ybr;

        scratchr[6] = scratchi[10] * yai + scratchi[9] * ybi;
        scratchi[6] = -(scratchr[10] * yai) - scratchr[9] * ybi;

        *Fout1r = scratchr[5] - scratchr[6];
        *Fout1i = scratchi[5] - scratchi[6];
        *Fout4r = scratchr[5] + scratchr[6];
        *Fout4i = scratchi[5] + scratchi[6];

        scratchr[11] = scratchr[0] + scratchr[7] * ybr + scratchr[8] * yar;
        scratchi[11] = scratchi[0] + scratchi[7] * ybr + scratchi[8] * yar;
        scratchr[12] = -(scratchi[10] * ybi) + scratchi[9] * yai;
        scratchi[12] = scratchr[10] * ybi - scratchr[9] * yai;

        *Fout2r = scratchr[11] + scratchr[12];
        *Fout2i = scratchi[11] + scratchi[12];
        *Fout3r = scratchr[11] - scratchr[12];
        *Fout3i = scratchi[11] - scratchi[12];

        ++Fout0r; ++Fout1r; ++Fout2r; ++Fout3r; ++Fout4r;
        ++Fout0i; ++Fout1i; ++Fout2i; ++Fout3i; ++Fout4i;
    }
}


inline
static float hsum128_ps(__m128 v)
{
    v = _mm_hadd_ps(v, v);
    v = _mm_hadd_ps(v, v);
    return _mm_cvtss_f32(v);
}

inline
static float hsum256_ps(__m256 v)
{
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    return hsum128_ps(sum);
}

void kf_bfly_generic(
    float* Foutr,
    float* Fouti,
    int                    fstride,
    const float* twiddles_r,
    const float* twiddles_i,
    int                    Norig,
    int                    m,
    int                    p,
    float* scratchr,
    float* scratchi)
{
    for (int u = 0; u != m; ++u) {
        int k = u;
        for (int q = 0; q != p; ++q) {
            scratchr[q] = Foutr[k];
            scratchi[q] = Fouti[k];
            k += m;
        }

        k = u;
        for (int q1 = 0; q1 != p; ++q1) {

            const int step = k * fstride % Norig;

            float acc_r = scratchr[0];
            float acc_i = scratchi[0];

            alignas(32) int idx[8];
            int q = 1;
            for (; q + 7 < p; q += 8) {
                int qlocal = q;
                for (int i = 0; i != 8; ++i, ++qlocal) {
                    int idxv = qlocal * step;
                    if (idxv >= Norig)
                        idxv %= Norig;
                    idx[i] = idxv;
                }
                //__m256i vindex = _mm256_loadu_si256(reinterpret_cast<__m256i*>(idx));
                __m256i vindex = _mm256_loadu_si256((const __m256i*)idx);


                __m256 twr = _mm256_i32gather_ps(twiddles_r, vindex, 4);
                __m256 twi = _mm256_i32gather_ps(twiddles_i, vindex, 4);

                __m256 sr = _mm256_loadu_ps(scratchr + q);
                __m256 si = _mm256_loadu_ps(scratchi + q);

                __m256 tr = _mm256_fmsub_ps(sr, twr, _mm256_mul_ps(si, twi));
                __m256 ti = _mm256_fmadd_ps(sr, twi, _mm256_mul_ps(si, twr));

                acc_r += hsum256_ps(tr);
                acc_i += hsum256_ps(ti);
            }

            for (; q + 3 < p; q += 4)
            {
                int qlocal = q;
                for (int i = 0; i != 8; ++i, ++qlocal) {
                    int idxv = qlocal * step;
                    if (idxv >= Norig)
                        idxv %= Norig;
                    idx[i] = idxv;
                }
                //__m128i vindex = _mm_loadu_si128(reinterpret_cast<__m128i*>(idx));
                __m128i vindex = _mm_loadu_si128((const __m128i*)idx);

                __m128 twr = _mm_i32gather_ps(twiddles_r, vindex, 4);
                __m128 twi = _mm_i32gather_ps(twiddles_i, vindex, 4);

                __m128 sr = _mm_loadu_ps(scratchr + q);
                __m128 si = _mm_loadu_ps(scratchi + q);

                __m128 tr = _mm_fmsub_ps(sr, twr, _mm_mul_ps(si, twi));
                __m128 ti = _mm_fmadd_ps(sr, twi, _mm_mul_ps(si, twr));

                acc_r += hsum128_ps(tr);
                acc_i += hsum128_ps(ti);
            }

            for (; q != p; ++q) {
                int twidx = (q * step) % Norig;
                float tr = scratchr[q] * twiddles_r[twidx] - scratchi[q] * twiddles_i[twidx];
                float ti = scratchr[q] * twiddles_i[twidx] + scratchi[q] * twiddles_r[twidx];
                acc_r += tr;
                acc_i += ti;
            }

            Foutr[k] = acc_r;
            Fouti[k] = acc_i;

            k += m;
        }
    }
}
#else  //USE_SIMD








/* This file is copied from
 *
 * https://www.kurims.kyoto-u.ac.jp/~ooura/fft.html
 *
 * Copyright Takuya OOURA, 1996-2001
 *
 * You may use, copy, modify and distribute this code for any
 * purpose (include commercial use) and without fee. Please refer to
 * this package when you modify this code.
 */
/*
Fast Fourier/Cosine/Sine Transform
   dimension   :one
   data length :power of 2
   decimation  :frequency
   radix       :split-radix
   data        :inplace
   table       :use
functions
   cdft: Complex Discrete Fourier Transform
   rdft: Real Discrete Fourier Transform
   ddct: Discrete Cosine Transform
   ddst: Discrete Sine Transform
   dfct: Cosine Transform of RDFT (Real Symmetric DFT)
   dfst: Sine Transform of RDFT (Real Anti-symmetric DFT)
function prototypes
   void cdft(int, int, double *, int *, double *);
   void rdft(int, int, double *, int *, double *);
   void ddct(int, int, double *, int *, double *);
   void ddst(int, int, double *, int *, double *);
   void dfct(int, double *, double *, int *, double *);
   void dfst(int, double *, double *, int *, double *);
macro definitions
   USE_CDFT_PTHREADS : default=not defined
       CDFT_THREADS_BEGIN_N  : must be >= 512, default=8192
       CDFT_4THREADS_BEGIN_N : must be >= 512, default=65536
   USE_CDFT_WINTHREADS : default=not defined
       CDFT_THREADS_BEGIN_N  : must be >= 512, default=32768
       CDFT_4THREADS_BEGIN_N : must be >= 512, default=524288


-------- Complex DFT (Discrete Fourier Transform) --------
        [definition]
        <case1>
        X[k] = sum_j=0^n-1 x[j]*exp(2*pi*i*j*k/n), 0<=k<n
            <case2>
            X[k] = sum_j=0^n-1 x[j]*exp(-2*pi*i*j*k/n), 0<=k<n
            (notes: sum_j=0^n-1 is a summation from j=0 to n-1)
                [usage]
            <case1>
            ip[0] = 0; // first time only
cdft(2*n, 1, a, ip, w);
<case2>
   ip[0] = 0; // first time only
cdft(2*n, -1, a, ip, w);
   [parameters]
       2*n            :data length (int)
                       n >= 1, n = power of 2
       a[0...2*n-1]   :input/output data (double *)
                       input data
                           a[2*j] = Re(x[j]),
                           a[2*j+1] = Im(x[j]), 0<=j<n
                       output data
                           a[2*k] = Re(X[k]),
                           a[2*k+1] = Im(X[k]), 0<=k<n
       ip[0...*]      :work area for bit reversal (int *)
                       length of ip >= 2+sqrt(n)
                       strictly,
                       length of ip >=
                           2+(1<<(int)(log(n+0.5)/log(2))/2).
                       ip[0],ip[1] are pointers of the cos/sin table.
       w[0...n/2-1]   :cos/sin table (double *)
                       w[],ip[] are initialized if ip[0] == 0.
   [remark]
       Inverse of
           cdft(2*n, -1, a, ip, w);
   is
       cdft(2*n, 1, a, ip, w);
   for (j = 0; j <= 2 * n - 1; j++) {
       a[j] *= 1.0 / n;
   }
   .


       -------- Real DFT / Inverse of Real DFT --------
       [definition]
       <case1> RDFT
               R[k] = sum_j=0^n-1 a[j]*cos(2*pi*j*k/n), 0<=k<=n/2
                             I[k] = sum_j=0^n-1 a[j]*sin(2*pi*j*k/n), 0<k<n/2
                     <case2> IRDFT (excluding scale)
                       a[k] = (R[0] + R[n/2]*cos(pi*k))/2 +
                              sum_j=1^n/2-1 R[j]*cos(2*pi*j*k/n) +
                                              sum_j=1^n/2-1 I[j]*sin(2*pi*j*k/n), 0<=k<n
                            [usage]
                   <case1>
                   ip[0] = 0; // first time only
   rdft(n, 1, a, ip, w);
   <case2>
       ip[0] = 0; // first time only
   rdft(n, -1, a, ip, w);
   [parameters]
       n              :data length (int)
                       n >= 2, n = power of 2
       a[0...n-1]     :input/output data (double *)
                       <case1>
                           output data
                               a[2*k] = R[k], 0<=k<n/2
                               a[2*k+1] = I[k], 0<k<n/2
                               a[1] = R[n/2]
                       <case2>
                           input data
                               a[2*j] = R[j], 0<=j<n/2
                               a[2*j+1] = I[j], 0<j<n/2
                               a[1] = R[n/2]
       ip[0...*]      :work area for bit reversal (int *)
                       length of ip >= 2+sqrt(n/2)
                       strictly,
                       length of ip >=
                           2+(1<<(int)(log(n/2+0.5)/log(2))/2).
                       ip[0],ip[1] are pointers of the cos/sin table.
       w[0...n/2-1]   :cos/sin table (double *)
                       w[],ip[] are initialized if ip[0] == 0.
   [remark]
       Inverse of
           rdft(n, 1, a, ip, w);
   is
       rdft(n, -1, a, ip, w);
   for (j = 0; j <= n - 1; j++) {
       a[j] *= 2.0 / n;
   }
   .


       -------- DCT (Discrete Cosine Transform) / Inverse of DCT --------
       [definition]
       <case1> IDCT (excluding scale)
           C[k] = sum_j=0^n-1 a[j]*cos(pi*j*(k+1/2)/n), 0<=k<n
               <case2> DCT
               C[k] = sum_j=0^n-1 a[j]*cos(pi*(j+1/2)*k/n), 0<=k<n
                        [usage]
               <case1>
               ip[0] = 0; // first time only
   ddct(n, 1, a, ip, w);
   <case2>
       ip[0] = 0; // first time only
   ddct(n, -1, a, ip, w);
   [parameters]
       n              :data length (int)
                       n >= 2, n = power of 2
       a[0...n-1]     :input/output data (double *)
                       output data
                           a[k] = C[k], 0<=k<n
       ip[0...*]      :work area for bit reversal (int *)
                       length of ip >= 2+sqrt(n/2)
                       strictly,
                       length of ip >=
                           2+(1<<(int)(log(n/2+0.5)/log(2))/2).
                       ip[0],ip[1] are pointers of the cos/sin table.
       w[0...n*5/4-1] :cos/sin table (double *)
                       w[],ip[] are initialized if ip[0] == 0.
   [remark]
       Inverse of
           ddct(n, -1, a, ip, w);
   is
       a[0] *= 0.5;
   ddct(n, 1, a, ip, w);
   for (j = 0; j <= n - 1; j++) {
       a[j] *= 2.0 / n;
   }
   .


       -------- DST (Discrete Sine Transform) / Inverse of DST --------
       [definition]
       <case1> IDST (excluding scale)
           S[k] = sum_j=1^n A[j]*sin(pi*j*(k+1/2)/n), 0<=k<n
               <case2> DST
               S[k] = sum_j=0^n-1 a[j]*sin(pi*(j+1/2)*k/n), 0<k<=n
                        [usage]
               <case1>
               ip[0] = 0; // first time only
   ddst(n, 1, a, ip, w);
   <case2>
       ip[0] = 0; // first time only
   ddst(n, -1, a, ip, w);
   [parameters]
       n              :data length (int)
                       n >= 2, n = power of 2
       a[0...n-1]     :input/output data (double *)
                       <case1>
                           input data
                               a[j] = A[j], 0<j<n
                               a[0] = A[n]
                           output data
                               a[k] = S[k], 0<=k<n
                       <case2>
                           output data
                               a[k] = S[k], 0<k<n
                               a[0] = S[n]
       ip[0...*]      :work area for bit reversal (int *)
                       length of ip >= 2+sqrt(n/2)
                       strictly,
                       length of ip >=
                           2+(1<<(int)(log(n/2+0.5)/log(2))/2).
                       ip[0],ip[1] are pointers of the cos/sin table.
       w[0...n*5/4-1] :cos/sin table (double *)
                       w[],ip[] are initialized if ip[0] == 0.
   [remark]
       Inverse of
           ddst(n, -1, a, ip, w);
   is
       a[0] *= 0.5;
   ddst(n, 1, a, ip, w);
   for (j = 0; j <= n - 1; j++) {
       a[j] *= 2.0 / n;
   }
   .


       -------- Cosine Transform of RDFT (Real Symmetric DFT) --------
       [definition]
       C[k] = sum_j=0^n a[j]*cos(pi*j*k/n), 0<=k<=n
                     [usage]
           ip[0] = 0; // first time only
   dfct(n, a, t, ip, w);
   [parameters]
       n              :data length - 1 (int)
                       n >= 2, n = power of 2
       a[0...n]       :input/output data (double *)
                       output data
                           a[k] = C[k], 0<=k<=n
       t[0...n/2]     :work area (double *)
       ip[0...*]      :work area for bit reversal (int *)
                       length of ip >= 2+sqrt(n/4)
                       strictly,
                       length of ip >=
                           2+(1<<(int)(log(n/4+0.5)/log(2))/2).
                       ip[0],ip[1] are pointers of the cos/sin table.
       w[0...n*5/8-1] :cos/sin table (double *)
                       w[],ip[] are initialized if ip[0] == 0.
   [remark]
       Inverse of
           a[0] *= 0.5;
   a[n] *= 0.5;
   dfct(n, a, t, ip, w);
   is
       a[0] *= 0.5;
   a[n] *= 0.5;
   dfct(n, a, t, ip, w);
   for (j = 0; j <= n; j++) {
       a[j] *= 2.0 / n;
   }
   .


       -------- Sine Transform of RDFT (Real Anti-symmetric DFT) --------
       [definition]
       S[k] = sum_j=1^n-1 a[j]*sin(pi*j*k/n), 0<k<n
                   [usage]
           ip[0] = 0; // first time only
   dfst(n, a, t, ip, w);
   [parameters]
       n              :data length + 1 (int)
                       n >= 2, n = power of 2
       a[0...n-1]     :input/output data (double *)
                       output data
                           a[k] = S[k], 0<k<n
                       (a[0] is used for work area)
       t[0...n/2-1]   :work area (double *)
       ip[0...*]      :work area for bit reversal (int *)
                       length of ip >= 2+sqrt(n/4)
                       strictly,
                       length of ip >=
                           2+(1<<(int)(log(n/4+0.5)/log(2))/2).
                       ip[0],ip[1] are pointers of the cos/sin table.
       w[0...n*5/8-1] :cos/sin table (double *)
                       w[],ip[] are initialized if ip[0] == 0.
   [remark]
       Inverse of
           dfst(n, a, t, ip, w);
   is
       dfst(n, a, t, ip, w);
   for (j = 1; j <= n - 1; j++) {
       a[j] *= 2.0 / n;
   }
   .


           Appendix :
       The cos/sin table is recalculated when the larger table required.
           w[] and ip[] are compatible with all routines.
               */

#ifdef USE_CDFT_PTHREADS
#define USE_CDFT_THREADS
#ifndef CDFT_THREADS_BEGIN_N
#define CDFT_THREADS_BEGIN_N 8192
#endif
#ifndef CDFT_4THREADS_BEGIN_N
#define CDFT_4THREADS_BEGIN_N 65536
#endif
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#define cdft_thread_t pthread_t
#define cdft_thread_create(thp, func, argp)                       \
    {                                                             \
        if (pthread_create(thp, NULL, func, (void *)argp) != 0) { \
            fprintf(stderr, "cdft thread error\n");               \
            exit(1);                                              \
        }                                                         \
    }
#define cdft_thread_wait(th)                        \
    {                                               \
        if (pthread_join(th, NULL) != 0) {          \
            fprintf(stderr, "cdft thread error\n"); \
            exit(1);                                \
        }                                           \
    }
#endif /* USE_CDFT_PTHREADS */

#ifdef USE_CDFT_WINTHREADS
#define USE_CDFT_THREADS
#ifndef CDFT_THREADS_BEGIN_N
#define CDFT_THREADS_BEGIN_N 32768
#endif
#ifndef CDFT_4THREADS_BEGIN_N
#define CDFT_4THREADS_BEGIN_N 524288
#endif
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#define cdft_thread_t HANDLE
#define cdft_thread_create(thp, func, argp)                                                   \
    {                                                                                         \
        DWORD thid;                                                                           \
        *(thp) = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)func, (LPVOID)argp, 0, &thid); \
        if (*(thp) == 0) {                                                                    \
            fprintf(stderr, "cdft thread error\n");                                           \
            exit(1);                                                                          \
        }                                                                                     \
    }
#define cdft_thread_wait(th)               \
    {                                      \
        WaitForSingleObject(th, INFINITE); \
        CloseHandle(th);                   \
    }
#endif /* USE_CDFT_WINTHREADS */

static void makeipt(int nw, int *ip) {
    int j, l, m, m2, p, q;

    ip[2] = 0;
    ip[3] = 16;
    m = 2;
    for (l = nw; l > 32; l >>= 2) {
        m2 = m << 1;
        q = m2 << 3;
        for (j = m; j < m2; j++) {
            p = ip[j] << 2;
            ip[m + j] = p;
            ip[m2 + j] = p + q;
        }
        m = m2;
    }
}

static void makewt(int nw, int *ip, double *w) {
    int j, nwh, nw0, nw1;
    double delta, wn4r, wk1r, wk1i, wk3r, wk3i;

    ip[0] = nw;
    ip[1] = 1;
    if (nw > 2) {
        nwh = nw >> 1;
        delta = atan(1.0) / nwh;
        wn4r = cos(delta * nwh);
        w[0] = 1;
        w[1] = wn4r;
        if (nwh == 4) {
            w[2] = cos(delta * 2);
            w[3] = sin(delta * 2);
        } else if (nwh > 4) {
            makeipt(nw, ip);
            w[2] = 0.5 / cos(delta * 2);
            w[3] = 0.5 / cos(delta * 6);
            for (j = 4; j < nwh; j += 4) {
                w[j] = cos(delta * j);
                w[j + 1] = sin(delta * j);
                w[j + 2] = cos(3 * delta * j);
                w[j + 3] = -sin(3 * delta * j);
            }
        }
        nw0 = 0;
        while (nwh > 2) {
            nw1 = nw0 + nwh;
            nwh >>= 1;
            w[nw1] = 1;
            w[nw1 + 1] = wn4r;
            if (nwh == 4) {
                wk1r = w[nw0 + 4];
                wk1i = w[nw0 + 5];
                w[nw1 + 2] = wk1r;
                w[nw1 + 3] = wk1i;
            } else if (nwh > 4) {
                wk1r = w[nw0 + 4];
                wk3r = w[nw0 + 6];
                w[nw1 + 2] = 0.5 / wk1r;
                w[nw1 + 3] = 0.5 / wk3r;
                for (j = 4; j < nwh; j += 4) {
                    wk1r = w[nw0 + 2 * j];
                    wk1i = w[nw0 + 2 * j + 1];
                    wk3r = w[nw0 + 2 * j + 2];
                    wk3i = w[nw0 + 2 * j + 3];
                    w[nw1 + j] = wk1r;
                    w[nw1 + j + 1] = wk1i;
                    w[nw1 + j + 2] = wk3r;
                    w[nw1 + j + 3] = wk3i;
                }
            }
            nw0 = nw1;
        }
    }
}

static void makect(int nc, int *ip, double *c) {
    int j, nch;
    double delta;

    ip[1] = nc;
    if (nc > 1) {
        nch = nc >> 1;
        delta = atan(1.0) / nch;
        c[0] = cos(delta * nch);
        c[nch] = 0.5 * c[0];
        for (j = 1; j < nch; j++) {
            c[j] = 0.5 * cos(delta * j);
            c[nc - j] = 0.5 * sin(delta * j);
        }
    }
}

/* -------- child routines -------- */

static void bitrv2(int n, int *ip, double *a) {
    int j, j1, k, k1, l, m, nh, nm;
    double xr, xi, yr, yi;

    m = 1;
    for (l = n >> 2; l > 8; l >>= 2) {
        m <<= 1;
    }
    nh = n >> 1;
    nm = 4 * m;
    if (l == 8) {
        for (k = 0; k < m; k++) {
            for (j = 0; j < k; j++) {
                j1 = 4 * j + 2 * ip[m + k];
                k1 = 4 * k + 2 * ip[m + j];
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += nm;
                k1 += 2 * nm;
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += nm;
                k1 -= nm;
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += nm;
                k1 += 2 * nm;
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += nh;
                k1 += 2;
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 -= nm;
                k1 -= 2 * nm;
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 -= nm;
                k1 += nm;
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 -= nm;
                k1 -= 2 * nm;
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += 2;
                k1 += nh;
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += nm;
                k1 += 2 * nm;
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += nm;
                k1 -= nm;
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += nm;
                k1 += 2 * nm;
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 -= nh;
                k1 -= 2;
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 -= nm;
                k1 -= 2 * nm;
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 -= nm;
                k1 += nm;
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 -= nm;
                k1 -= 2 * nm;
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
            }
            k1 = 4 * k + 2 * ip[m + k];
            j1 = k1 + 2;
            k1 += nh;
            xr = a[j1];
            xi = a[j1 + 1];
            yr = a[k1];
            yi = a[k1 + 1];
            a[j1] = yr;
            a[j1 + 1] = yi;
            a[k1] = xr;
            a[k1 + 1] = xi;
            j1 += nm;
            k1 += 2 * nm;
            xr = a[j1];
            xi = a[j1 + 1];
            yr = a[k1];
            yi = a[k1 + 1];
            a[j1] = yr;
            a[j1 + 1] = yi;
            a[k1] = xr;
            a[k1 + 1] = xi;
            j1 += nm;
            k1 -= nm;
            xr = a[j1];
            xi = a[j1 + 1];
            yr = a[k1];
            yi = a[k1 + 1];
            a[j1] = yr;
            a[j1 + 1] = yi;
            a[k1] = xr;
            a[k1 + 1] = xi;
            j1 -= 2;
            k1 -= nh;
            xr = a[j1];
            xi = a[j1 + 1];
            yr = a[k1];
            yi = a[k1 + 1];
            a[j1] = yr;
            a[j1 + 1] = yi;
            a[k1] = xr;
            a[k1 + 1] = xi;
            j1 += nh + 2;
            k1 += nh + 2;
            xr = a[j1];
            xi = a[j1 + 1];
            yr = a[k1];
            yi = a[k1 + 1];
            a[j1] = yr;
            a[j1 + 1] = yi;
            a[k1] = xr;
            a[k1 + 1] = xi;
            j1 -= nh - nm;
            k1 += 2 * nm - 2;
            xr = a[j1];
            xi = a[j1 + 1];
            yr = a[k1];
            yi = a[k1 + 1];
            a[j1] = yr;
            a[j1 + 1] = yi;
            a[k1] = xr;
            a[k1 + 1] = xi;
        }
    } else {
        for (k = 0; k < m; k++) {
            for (j = 0; j < k; j++) {
                j1 = 4 * j + ip[m + k];
                k1 = 4 * k + ip[m + j];
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += nm;
                k1 += nm;
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += nh;
                k1 += 2;
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 -= nm;
                k1 -= nm;
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += 2;
                k1 += nh;
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += nm;
                k1 += nm;
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 -= nh;
                k1 -= 2;
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 -= nm;
                k1 -= nm;
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
            }
            k1 = 4 * k + ip[m + k];
            j1 = k1 + 2;
            k1 += nh;
            xr = a[j1];
            xi = a[j1 + 1];
            yr = a[k1];
            yi = a[k1 + 1];
            a[j1] = yr;
            a[j1 + 1] = yi;
            a[k1] = xr;
            a[k1 + 1] = xi;
            j1 += nm;
            k1 += nm;
            xr = a[j1];
            xi = a[j1 + 1];
            yr = a[k1];
            yi = a[k1 + 1];
            a[j1] = yr;
            a[j1 + 1] = yi;
            a[k1] = xr;
            a[k1 + 1] = xi;
        }
    }
}

static void bitrv2conj(int n, int *ip, double *a) {
    int j, j1, k, k1, l, m, nh, nm;
    double xr, xi, yr, yi;

    m = 1;
    for (l = n >> 2; l > 8; l >>= 2) {
        m <<= 1;
    }
    nh = n >> 1;
    nm = 4 * m;
    if (l == 8) {
        for (k = 0; k < m; k++) {
            for (j = 0; j < k; j++) {
                j1 = 4 * j + 2 * ip[m + k];
                k1 = 4 * k + 2 * ip[m + j];
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += nm;
                k1 += 2 * nm;
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += nm;
                k1 -= nm;
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += nm;
                k1 += 2 * nm;
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += nh;
                k1 += 2;
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 -= nm;
                k1 -= 2 * nm;
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 -= nm;
                k1 += nm;
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 -= nm;
                k1 -= 2 * nm;
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += 2;
                k1 += nh;
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += nm;
                k1 += 2 * nm;
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += nm;
                k1 -= nm;
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += nm;
                k1 += 2 * nm;
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 -= nh;
                k1 -= 2;
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 -= nm;
                k1 -= 2 * nm;
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 -= nm;
                k1 += nm;
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 -= nm;
                k1 -= 2 * nm;
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
            }
            k1 = 4 * k + 2 * ip[m + k];
            j1 = k1 + 2;
            k1 += nh;
            a[j1 - 1] = -a[j1 - 1];
            xr = a[j1];
            xi = -a[j1 + 1];
            yr = a[k1];
            yi = -a[k1 + 1];
            a[j1] = yr;
            a[j1 + 1] = yi;
            a[k1] = xr;
            a[k1 + 1] = xi;
            a[k1 + 3] = -a[k1 + 3];
            j1 += nm;
            k1 += 2 * nm;
            xr = a[j1];
            xi = -a[j1 + 1];
            yr = a[k1];
            yi = -a[k1 + 1];
            a[j1] = yr;
            a[j1 + 1] = yi;
            a[k1] = xr;
            a[k1 + 1] = xi;
            j1 += nm;
            k1 -= nm;
            xr = a[j1];
            xi = -a[j1 + 1];
            yr = a[k1];
            yi = -a[k1 + 1];
            a[j1] = yr;
            a[j1 + 1] = yi;
            a[k1] = xr;
            a[k1 + 1] = xi;
            j1 -= 2;
            k1 -= nh;
            xr = a[j1];
            xi = -a[j1 + 1];
            yr = a[k1];
            yi = -a[k1 + 1];
            a[j1] = yr;
            a[j1 + 1] = yi;
            a[k1] = xr;
            a[k1 + 1] = xi;
            j1 += nh + 2;
            k1 += nh + 2;
            xr = a[j1];
            xi = -a[j1 + 1];
            yr = a[k1];
            yi = -a[k1 + 1];
            a[j1] = yr;
            a[j1 + 1] = yi;
            a[k1] = xr;
            a[k1 + 1] = xi;
            j1 -= nh - nm;
            k1 += 2 * nm - 2;
            a[j1 - 1] = -a[j1 - 1];
            xr = a[j1];
            xi = -a[j1 + 1];
            yr = a[k1];
            yi = -a[k1 + 1];
            a[j1] = yr;
            a[j1 + 1] = yi;
            a[k1] = xr;
            a[k1 + 1] = xi;
            a[k1 + 3] = -a[k1 + 3];
        }
    } else {
        for (k = 0; k < m; k++) {
            for (j = 0; j < k; j++) {
                j1 = 4 * j + ip[m + k];
                k1 = 4 * k + ip[m + j];
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += nm;
                k1 += nm;
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += nh;
                k1 += 2;
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 -= nm;
                k1 -= nm;
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += 2;
                k1 += nh;
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += nm;
                k1 += nm;
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 -= nh;
                k1 -= 2;
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 -= nm;
                k1 -= nm;
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
            }
            k1 = 4 * k + ip[m + k];
            j1 = k1 + 2;
            k1 += nh;
            a[j1 - 1] = -a[j1 - 1];
            xr = a[j1];
            xi = -a[j1 + 1];
            yr = a[k1];
            yi = -a[k1 + 1];
            a[j1] = yr;
            a[j1 + 1] = yi;
            a[k1] = xr;
            a[k1 + 1] = xi;
            a[k1 + 3] = -a[k1 + 3];
            j1 += nm;
            k1 += nm;
            a[j1 - 1] = -a[j1 - 1];
            xr = a[j1];
            xi = -a[j1 + 1];
            yr = a[k1];
            yi = -a[k1 + 1];
            a[j1] = yr;
            a[j1 + 1] = yi;
            a[k1] = xr;
            a[k1 + 1] = xi;
            a[k1 + 3] = -a[k1 + 3];
        }
    }
}

static void bitrv216(double *a) {
    double x1r, x1i, x2r, x2i, x3r, x3i, x4r, x4i, x5r, x5i, x7r, x7i, x8r, x8i, x10r, x10i, x11r, x11i, x12r, x12i,
        x13r, x13i, x14r, x14i;

    x1r = a[2];
    x1i = a[3];
    x2r = a[4];
    x2i = a[5];
    x3r = a[6];
    x3i = a[7];
    x4r = a[8];
    x4i = a[9];
    x5r = a[10];
    x5i = a[11];
    x7r = a[14];
    x7i = a[15];
    x8r = a[16];
    x8i = a[17];
    x10r = a[20];
    x10i = a[21];
    x11r = a[22];
    x11i = a[23];
    x12r = a[24];
    x12i = a[25];
    x13r = a[26];
    x13i = a[27];
    x14r = a[28];
    x14i = a[29];
    a[2] = x8r;
    a[3] = x8i;
    a[4] = x4r;
    a[5] = x4i;
    a[6] = x12r;
    a[7] = x12i;
    a[8] = x2r;
    a[9] = x2i;
    a[10] = x10r;
    a[11] = x10i;
    a[14] = x14r;
    a[15] = x14i;
    a[16] = x1r;
    a[17] = x1i;
    a[20] = x5r;
    a[21] = x5i;
    a[22] = x13r;
    a[23] = x13i;
    a[24] = x3r;
    a[25] = x3i;
    a[26] = x11r;
    a[27] = x11i;
    a[28] = x7r;
    a[29] = x7i;
}

static void bitrv216neg(double *a) {
    double x1r, x1i, x2r, x2i, x3r, x3i, x4r, x4i, x5r, x5i, x6r, x6i, x7r, x7i, x8r, x8i, x9r, x9i, x10r, x10i, x11r,
        x11i, x12r, x12i, x13r, x13i, x14r, x14i, x15r, x15i;

    x1r = a[2];
    x1i = a[3];
    x2r = a[4];
    x2i = a[5];
    x3r = a[6];
    x3i = a[7];
    x4r = a[8];
    x4i = a[9];
    x5r = a[10];
    x5i = a[11];
    x6r = a[12];
    x6i = a[13];
    x7r = a[14];
    x7i = a[15];
    x8r = a[16];
    x8i = a[17];
    x9r = a[18];
    x9i = a[19];
    x10r = a[20];
    x10i = a[21];
    x11r = a[22];
    x11i = a[23];
    x12r = a[24];
    x12i = a[25];
    x13r = a[26];
    x13i = a[27];
    x14r = a[28];
    x14i = a[29];
    x15r = a[30];
    x15i = a[31];
    a[2] = x15r;
    a[3] = x15i;
    a[4] = x7r;
    a[5] = x7i;
    a[6] = x11r;
    a[7] = x11i;
    a[8] = x3r;
    a[9] = x3i;
    a[10] = x13r;
    a[11] = x13i;
    a[12] = x5r;
    a[13] = x5i;
    a[14] = x9r;
    a[15] = x9i;
    a[16] = x1r;
    a[17] = x1i;
    a[18] = x14r;
    a[19] = x14i;
    a[20] = x6r;
    a[21] = x6i;
    a[22] = x10r;
    a[23] = x10i;
    a[24] = x2r;
    a[25] = x2i;
    a[26] = x12r;
    a[27] = x12i;
    a[28] = x4r;
    a[29] = x4i;
    a[30] = x8r;
    a[31] = x8i;
}

static void bitrv208(double *a) {
    double x1r, x1i, x3r, x3i, x4r, x4i, x6r, x6i;

    x1r = a[2];
    x1i = a[3];
    x3r = a[6];
    x3i = a[7];
    x4r = a[8];
    x4i = a[9];
    x6r = a[12];
    x6i = a[13];
    a[2] = x4r;
    a[3] = x4i;
    a[6] = x6r;
    a[7] = x6i;
    a[8] = x1r;
    a[9] = x1i;
    a[12] = x3r;
    a[13] = x3i;
}

static void bitrv208neg(double *a) {
    double x1r, x1i, x2r, x2i, x3r, x3i, x4r, x4i, x5r, x5i, x6r, x6i, x7r, x7i;

    x1r = a[2];
    x1i = a[3];
    x2r = a[4];
    x2i = a[5];
    x3r = a[6];
    x3i = a[7];
    x4r = a[8];
    x4i = a[9];
    x5r = a[10];
    x5i = a[11];
    x6r = a[12];
    x6i = a[13];
    x7r = a[14];
    x7i = a[15];
    a[2] = x7r;
    a[3] = x7i;
    a[4] = x3r;
    a[5] = x3i;
    a[6] = x5r;
    a[7] = x5i;
    a[8] = x1r;
    a[9] = x1i;
    a[10] = x6r;
    a[11] = x6i;
    a[12] = x2r;
    a[13] = x2i;
    a[14] = x4r;
    a[15] = x4i;
}

static void cftf1st(int n, double *a, double *w) {
    int j, j0, j1, j2, j3, k, m, mh;
    double wn4r, csc1, csc3, wk1r, wk1i, wk3r, wk3i, wd1r, wd1i, wd3r, wd3i;
    double x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i, y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;

    mh = n >> 3;
    m = 2 * mh;
    j1 = m;
    j2 = j1 + m;
    j3 = j2 + m;
    x0r = a[0] + a[j2];
    x0i = a[1] + a[j2 + 1];
    x1r = a[0] - a[j2];
    x1i = a[1] - a[j2 + 1];
    x2r = a[j1] + a[j3];
    x2i = a[j1 + 1] + a[j3 + 1];
    x3r = a[j1] - a[j3];
    x3i = a[j1 + 1] - a[j3 + 1];
    a[0] = x0r + x2r;
    a[1] = x0i + x2i;
    a[j1] = x0r - x2r;
    a[j1 + 1] = x0i - x2i;
    a[j2] = x1r - x3i;
    a[j2 + 1] = x1i + x3r;
    a[j3] = x1r + x3i;
    a[j3 + 1] = x1i - x3r;
    wn4r = w[1];
    csc1 = w[2];
    csc3 = w[3];
    wd1r = 1;
    wd1i = 0;
    wd3r = 1;
    wd3i = 0;
    k = 0;
    for (j = 2; j < mh - 2; j += 4) {
        k += 4;
        wk1r = csc1 * (wd1r + w[k]);
        wk1i = csc1 * (wd1i + w[k + 1]);
        wk3r = csc3 * (wd3r + w[k + 2]);
        wk3i = csc3 * (wd3i + w[k + 3]);
        wd1r = w[k];
        wd1i = w[k + 1];
        wd3r = w[k + 2];
        wd3i = w[k + 3];
        j1 = j + m;
        j2 = j1 + m;
        j3 = j2 + m;
        x0r = a[j] + a[j2];
        x0i = a[j + 1] + a[j2 + 1];
        x1r = a[j] - a[j2];
        x1i = a[j + 1] - a[j2 + 1];
        y0r = a[j + 2] + a[j2 + 2];
        y0i = a[j + 3] + a[j2 + 3];
        y1r = a[j + 2] - a[j2 + 2];
        y1i = a[j + 3] - a[j2 + 3];
        x2r = a[j1] + a[j3];
        x2i = a[j1 + 1] + a[j3 + 1];
        x3r = a[j1] - a[j3];
        x3i = a[j1 + 1] - a[j3 + 1];
        y2r = a[j1 + 2] + a[j3 + 2];
        y2i = a[j1 + 3] + a[j3 + 3];
        y3r = a[j1 + 2] - a[j3 + 2];
        y3i = a[j1 + 3] - a[j3 + 3];
        a[j] = x0r + x2r;
        a[j + 1] = x0i + x2i;
        a[j + 2] = y0r + y2r;
        a[j + 3] = y0i + y2i;
        a[j1] = x0r - x2r;
        a[j1 + 1] = x0i - x2i;
        a[j1 + 2] = y0r - y2r;
        a[j1 + 3] = y0i - y2i;
        x0r = x1r - x3i;
        x0i = x1i + x3r;
        a[j2] = wk1r * x0r - wk1i * x0i;
        a[j2 + 1] = wk1r * x0i + wk1i * x0r;
        x0r = y1r - y3i;
        x0i = y1i + y3r;
        a[j2 + 2] = wd1r * x0r - wd1i * x0i;
        a[j2 + 3] = wd1r * x0i + wd1i * x0r;
        x0r = x1r + x3i;
        x0i = x1i - x3r;
        a[j3] = wk3r * x0r + wk3i * x0i;
        a[j3 + 1] = wk3r * x0i - wk3i * x0r;
        x0r = y1r + y3i;
        x0i = y1i - y3r;
        a[j3 + 2] = wd3r * x0r + wd3i * x0i;
        a[j3 + 3] = wd3r * x0i - wd3i * x0r;
        j0 = m - j;
        j1 = j0 + m;
        j2 = j1 + m;
        j3 = j2 + m;
        x0r = a[j0] + a[j2];
        x0i = a[j0 + 1] + a[j2 + 1];
        x1r = a[j0] - a[j2];
        x1i = a[j0 + 1] - a[j2 + 1];
        y0r = a[j0 - 2] + a[j2 - 2];
        y0i = a[j0 - 1] + a[j2 - 1];
        y1r = a[j0 - 2] - a[j2 - 2];
        y1i = a[j0 - 1] - a[j2 - 1];
        x2r = a[j1] + a[j3];
        x2i = a[j1 + 1] + a[j3 + 1];
        x3r = a[j1] - a[j3];
        x3i = a[j1 + 1] - a[j3 + 1];
        y2r = a[j1 - 2] + a[j3 - 2];
        y2i = a[j1 - 1] + a[j3 - 1];
        y3r = a[j1 - 2] - a[j3 - 2];
        y3i = a[j1 - 1] - a[j3 - 1];
        a[j0] = x0r + x2r;
        a[j0 + 1] = x0i + x2i;
        a[j0 - 2] = y0r + y2r;
        a[j0 - 1] = y0i + y2i;
        a[j1] = x0r - x2r;
        a[j1 + 1] = x0i - x2i;
        a[j1 - 2] = y0r - y2r;
        a[j1 - 1] = y0i - y2i;
        x0r = x1r - x3i;
        x0i = x1i + x3r;
        a[j2] = wk1i * x0r - wk1r * x0i;
        a[j2 + 1] = wk1i * x0i + wk1r * x0r;
        x0r = y1r - y3i;
        x0i = y1i + y3r;
        a[j2 - 2] = wd1i * x0r - wd1r * x0i;
        a[j2 - 1] = wd1i * x0i + wd1r * x0r;
        x0r = x1r + x3i;
        x0i = x1i - x3r;
        a[j3] = wk3i * x0r + wk3r * x0i;
        a[j3 + 1] = wk3i * x0i - wk3r * x0r;
        x0r = y1r + y3i;
        x0i = y1i - y3r;
        a[j3 - 2] = wd3i * x0r + wd3r * x0i;
        a[j3 - 1] = wd3i * x0i - wd3r * x0r;
    }
    wk1r = csc1 * (wd1r + wn4r);
    wk1i = csc1 * (wd1i + wn4r);
    wk3r = csc3 * (wd3r - wn4r);
    wk3i = csc3 * (wd3i - wn4r);
    j0 = mh;
    j1 = j0 + m;
    j2 = j1 + m;
    j3 = j2 + m;
    x0r = a[j0 - 2] + a[j2 - 2];
    x0i = a[j0 - 1] + a[j2 - 1];
    x1r = a[j0 - 2] - a[j2 - 2];
    x1i = a[j0 - 1] - a[j2 - 1];
    x2r = a[j1 - 2] + a[j3 - 2];
    x2i = a[j1 - 1] + a[j3 - 1];
    x3r = a[j1 - 2] - a[j3 - 2];
    x3i = a[j1 - 1] - a[j3 - 1];
    a[j0 - 2] = x0r + x2r;
    a[j0 - 1] = x0i + x2i;
    a[j1 - 2] = x0r - x2r;
    a[j1 - 1] = x0i - x2i;
    x0r = x1r - x3i;
    x0i = x1i + x3r;
    a[j2 - 2] = wk1r * x0r - wk1i * x0i;
    a[j2 - 1] = wk1r * x0i + wk1i * x0r;
    x0r = x1r + x3i;
    x0i = x1i - x3r;
    a[j3 - 2] = wk3r * x0r + wk3i * x0i;
    a[j3 - 1] = wk3r * x0i - wk3i * x0r;
    x0r = a[j0] + a[j2];
    x0i = a[j0 + 1] + a[j2 + 1];
    x1r = a[j0] - a[j2];
    x1i = a[j0 + 1] - a[j2 + 1];
    x2r = a[j1] + a[j3];
    x2i = a[j1 + 1] + a[j3 + 1];
    x3r = a[j1] - a[j3];
    x3i = a[j1 + 1] - a[j3 + 1];
    a[j0] = x0r + x2r;
    a[j0 + 1] = x0i + x2i;
    a[j1] = x0r - x2r;
    a[j1 + 1] = x0i - x2i;
    x0r = x1r - x3i;
    x0i = x1i + x3r;
    a[j2] = wn4r * (x0r - x0i);
    a[j2 + 1] = wn4r * (x0i + x0r);
    x0r = x1r + x3i;
    x0i = x1i - x3r;
    a[j3] = -wn4r * (x0r + x0i);
    a[j3 + 1] = -wn4r * (x0i - x0r);
    x0r = a[j0 + 2] + a[j2 + 2];
    x0i = a[j0 + 3] + a[j2 + 3];
    x1r = a[j0 + 2] - a[j2 + 2];
    x1i = a[j0 + 3] - a[j2 + 3];
    x2r = a[j1 + 2] + a[j3 + 2];
    x2i = a[j1 + 3] + a[j3 + 3];
    x3r = a[j1 + 2] - a[j3 + 2];
    x3i = a[j1 + 3] - a[j3 + 3];
    a[j0 + 2] = x0r + x2r;
    a[j0 + 3] = x0i + x2i;
    a[j1 + 2] = x0r - x2r;
    a[j1 + 3] = x0i - x2i;
    x0r = x1r - x3i;
    x0i = x1i + x3r;
    a[j2 + 2] = wk1i * x0r - wk1r * x0i;
    a[j2 + 3] = wk1i * x0i + wk1r * x0r;
    x0r = x1r + x3i;
    x0i = x1i - x3r;
    a[j3 + 2] = wk3i * x0r + wk3r * x0i;
    a[j3 + 3] = wk3i * x0i - wk3r * x0r;
}

static void cftb1st(int n, double *a, double *w) {
    int j, j0, j1, j2, j3, k, m, mh;
    double wn4r, csc1, csc3, wk1r, wk1i, wk3r, wk3i, wd1r, wd1i, wd3r, wd3i;
    double x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i, y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;

    mh = n >> 3;
    m = 2 * mh;
    j1 = m;
    j2 = j1 + m;
    j3 = j2 + m;
    x0r = a[0] + a[j2];
    x0i = -a[1] - a[j2 + 1];
    x1r = a[0] - a[j2];
    x1i = -a[1] + a[j2 + 1];
    x2r = a[j1] + a[j3];
    x2i = a[j1 + 1] + a[j3 + 1];
    x3r = a[j1] - a[j3];
    x3i = a[j1 + 1] - a[j3 + 1];
    a[0] = x0r + x2r;
    a[1] = x0i - x2i;
    a[j1] = x0r - x2r;
    a[j1 + 1] = x0i + x2i;
    a[j2] = x1r + x3i;
    a[j2 + 1] = x1i + x3r;
    a[j3] = x1r - x3i;
    a[j3 + 1] = x1i - x3r;
    wn4r = w[1];
    csc1 = w[2];
    csc3 = w[3];
    wd1r = 1;
    wd1i = 0;
    wd3r = 1;
    wd3i = 0;
    k = 0;
    for (j = 2; j < mh - 2; j += 4) {
        k += 4;
        wk1r = csc1 * (wd1r + w[k]);
        wk1i = csc1 * (wd1i + w[k + 1]);
        wk3r = csc3 * (wd3r + w[k + 2]);
        wk3i = csc3 * (wd3i + w[k + 3]);
        wd1r = w[k];
        wd1i = w[k + 1];
        wd3r = w[k + 2];
        wd3i = w[k + 3];
        j1 = j + m;
        j2 = j1 + m;
        j3 = j2 + m;
        x0r = a[j] + a[j2];
        x0i = -a[j + 1] - a[j2 + 1];
        x1r = a[j] - a[j2];
        x1i = -a[j + 1] + a[j2 + 1];
        y0r = a[j + 2] + a[j2 + 2];
        y0i = -a[j + 3] - a[j2 + 3];
        y1r = a[j + 2] - a[j2 + 2];
        y1i = -a[j + 3] + a[j2 + 3];
        x2r = a[j1] + a[j3];
        x2i = a[j1 + 1] + a[j3 + 1];
        x3r = a[j1] - a[j3];
        x3i = a[j1 + 1] - a[j3 + 1];
        y2r = a[j1 + 2] + a[j3 + 2];
        y2i = a[j1 + 3] + a[j3 + 3];
        y3r = a[j1 + 2] - a[j3 + 2];
        y3i = a[j1 + 3] - a[j3 + 3];
        a[j] = x0r + x2r;
        a[j + 1] = x0i - x2i;
        a[j + 2] = y0r + y2r;
        a[j + 3] = y0i - y2i;
        a[j1] = x0r - x2r;
        a[j1 + 1] = x0i + x2i;
        a[j1 + 2] = y0r - y2r;
        a[j1 + 3] = y0i + y2i;
        x0r = x1r + x3i;
        x0i = x1i + x3r;
        a[j2] = wk1r * x0r - wk1i * x0i;
        a[j2 + 1] = wk1r * x0i + wk1i * x0r;
        x0r = y1r + y3i;
        x0i = y1i + y3r;
        a[j2 + 2] = wd1r * x0r - wd1i * x0i;
        a[j2 + 3] = wd1r * x0i + wd1i * x0r;
        x0r = x1r - x3i;
        x0i = x1i - x3r;
        a[j3] = wk3r * x0r + wk3i * x0i;
        a[j3 + 1] = wk3r * x0i - wk3i * x0r;
        x0r = y1r - y3i;
        x0i = y1i - y3r;
        a[j3 + 2] = wd3r * x0r + wd3i * x0i;
        a[j3 + 3] = wd3r * x0i - wd3i * x0r;
        j0 = m - j;
        j1 = j0 + m;
        j2 = j1 + m;
        j3 = j2 + m;
        x0r = a[j0] + a[j2];
        x0i = -a[j0 + 1] - a[j2 + 1];
        x1r = a[j0] - a[j2];
        x1i = -a[j0 + 1] + a[j2 + 1];
        y0r = a[j0 - 2] + a[j2 - 2];
        y0i = -a[j0 - 1] - a[j2 - 1];
        y1r = a[j0 - 2] - a[j2 - 2];
        y1i = -a[j0 - 1] + a[j2 - 1];
        x2r = a[j1] + a[j3];
        x2i = a[j1 + 1] + a[j3 + 1];
        x3r = a[j1] - a[j3];
        x3i = a[j1 + 1] - a[j3 + 1];
        y2r = a[j1 - 2] + a[j3 - 2];
        y2i = a[j1 - 1] + a[j3 - 1];
        y3r = a[j1 - 2] - a[j3 - 2];
        y3i = a[j1 - 1] - a[j3 - 1];
        a[j0] = x0r + x2r;
        a[j0 + 1] = x0i - x2i;
        a[j0 - 2] = y0r + y2r;
        a[j0 - 1] = y0i - y2i;
        a[j1] = x0r - x2r;
        a[j1 + 1] = x0i + x2i;
        a[j1 - 2] = y0r - y2r;
        a[j1 - 1] = y0i + y2i;
        x0r = x1r + x3i;
        x0i = x1i + x3r;
        a[j2] = wk1i * x0r - wk1r * x0i;
        a[j2 + 1] = wk1i * x0i + wk1r * x0r;
        x0r = y1r + y3i;
        x0i = y1i + y3r;
        a[j2 - 2] = wd1i * x0r - wd1r * x0i;
        a[j2 - 1] = wd1i * x0i + wd1r * x0r;
        x0r = x1r - x3i;
        x0i = x1i - x3r;
        a[j3] = wk3i * x0r + wk3r * x0i;
        a[j3 + 1] = wk3i * x0i - wk3r * x0r;
        x0r = y1r - y3i;
        x0i = y1i - y3r;
        a[j3 - 2] = wd3i * x0r + wd3r * x0i;
        a[j3 - 1] = wd3i * x0i - wd3r * x0r;
    }
    wk1r = csc1 * (wd1r + wn4r);
    wk1i = csc1 * (wd1i + wn4r);
    wk3r = csc3 * (wd3r - wn4r);
    wk3i = csc3 * (wd3i - wn4r);
    j0 = mh;
    j1 = j0 + m;
    j2 = j1 + m;
    j3 = j2 + m;
    x0r = a[j0 - 2] + a[j2 - 2];
    x0i = -a[j0 - 1] - a[j2 - 1];
    x1r = a[j0 - 2] - a[j2 - 2];
    x1i = -a[j0 - 1] + a[j2 - 1];
    x2r = a[j1 - 2] + a[j3 - 2];
    x2i = a[j1 - 1] + a[j3 - 1];
    x3r = a[j1 - 2] - a[j3 - 2];
    x3i = a[j1 - 1] - a[j3 - 1];
    a[j0 - 2] = x0r + x2r;
    a[j0 - 1] = x0i - x2i;
    a[j1 - 2] = x0r - x2r;
    a[j1 - 1] = x0i + x2i;
    x0r = x1r + x3i;
    x0i = x1i + x3r;
    a[j2 - 2] = wk1r * x0r - wk1i * x0i;
    a[j2 - 1] = wk1r * x0i + wk1i * x0r;
    x0r = x1r - x3i;
    x0i = x1i - x3r;
    a[j3 - 2] = wk3r * x0r + wk3i * x0i;
    a[j3 - 1] = wk3r * x0i - wk3i * x0r;
    x0r = a[j0] + a[j2];
    x0i = -a[j0 + 1] - a[j2 + 1];
    x1r = a[j0] - a[j2];
    x1i = -a[j0 + 1] + a[j2 + 1];
    x2r = a[j1] + a[j3];
    x2i = a[j1 + 1] + a[j3 + 1];
    x3r = a[j1] - a[j3];
    x3i = a[j1 + 1] - a[j3 + 1];
    a[j0] = x0r + x2r;
    a[j0 + 1] = x0i - x2i;
    a[j1] = x0r - x2r;
    a[j1 + 1] = x0i + x2i;
    x0r = x1r + x3i;
    x0i = x1i + x3r;
    a[j2] = wn4r * (x0r - x0i);
    a[j2 + 1] = wn4r * (x0i + x0r);
    x0r = x1r - x3i;
    x0i = x1i - x3r;
    a[j3] = -wn4r * (x0r + x0i);
    a[j3 + 1] = -wn4r * (x0i - x0r);
    x0r = a[j0 + 2] + a[j2 + 2];
    x0i = -a[j0 + 3] - a[j2 + 3];
    x1r = a[j0 + 2] - a[j2 + 2];
    x1i = -a[j0 + 3] + a[j2 + 3];
    x2r = a[j1 + 2] + a[j3 + 2];
    x2i = a[j1 + 3] + a[j3 + 3];
    x3r = a[j1 + 2] - a[j3 + 2];
    x3i = a[j1 + 3] - a[j3 + 3];
    a[j0 + 2] = x0r + x2r;
    a[j0 + 3] = x0i - x2i;
    a[j1 + 2] = x0r - x2r;
    a[j1 + 3] = x0i + x2i;
    x0r = x1r + x3i;
    x0i = x1i + x3r;
    a[j2 + 2] = wk1i * x0r - wk1r * x0i;
    a[j2 + 3] = wk1i * x0i + wk1r * x0r;
    x0r = x1r - x3i;
    x0i = x1i - x3r;
    a[j3 + 2] = wk3i * x0r + wk3r * x0i;
    a[j3 + 3] = wk3i * x0i - wk3r * x0r;
}

static void cftmdl1(int n, double *a, double *w) {
    int j, j0, j1, j2, j3, k, m, mh;
    double wn4r, wk1r, wk1i, wk3r, wk3i;
    double x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i;

    mh = n >> 3;
    m = 2 * mh;
    j1 = m;
    j2 = j1 + m;
    j3 = j2 + m;
    x0r = a[0] + a[j2];
    x0i = a[1] + a[j2 + 1];
    x1r = a[0] - a[j2];
    x1i = a[1] - a[j2 + 1];
    x2r = a[j1] + a[j3];
    x2i = a[j1 + 1] + a[j3 + 1];
    x3r = a[j1] - a[j3];
    x3i = a[j1 + 1] - a[j3 + 1];
    a[0] = x0r + x2r;
    a[1] = x0i + x2i;
    a[j1] = x0r - x2r;
    a[j1 + 1] = x0i - x2i;
    a[j2] = x1r - x3i;
    a[j2 + 1] = x1i + x3r;
    a[j3] = x1r + x3i;
    a[j3 + 1] = x1i - x3r;
    wn4r = w[1];
    k = 0;
    for (j = 2; j < mh; j += 2) {
        k += 4;
        wk1r = w[k];
        wk1i = w[k + 1];
        wk3r = w[k + 2];
        wk3i = w[k + 3];
        j1 = j + m;
        j2 = j1 + m;
        j3 = j2 + m;
        x0r = a[j] + a[j2];
        x0i = a[j + 1] + a[j2 + 1];
        x1r = a[j] - a[j2];
        x1i = a[j + 1] - a[j2 + 1];
        x2r = a[j1] + a[j3];
        x2i = a[j1 + 1] + a[j3 + 1];
        x3r = a[j1] - a[j3];
        x3i = a[j1 + 1] - a[j3 + 1];
        a[j] = x0r + x2r;
        a[j + 1] = x0i + x2i;
        a[j1] = x0r - x2r;
        a[j1 + 1] = x0i - x2i;
        x0r = x1r - x3i;
        x0i = x1i + x3r;
        a[j2] = wk1r * x0r - wk1i * x0i;
        a[j2 + 1] = wk1r * x0i + wk1i * x0r;
        x0r = x1r + x3i;
        x0i = x1i - x3r;
        a[j3] = wk3r * x0r + wk3i * x0i;
        a[j3 + 1] = wk3r * x0i - wk3i * x0r;
        j0 = m - j;
        j1 = j0 + m;
        j2 = j1 + m;
        j3 = j2 + m;
        x0r = a[j0] + a[j2];
        x0i = a[j0 + 1] + a[j2 + 1];
        x1r = a[j0] - a[j2];
        x1i = a[j0 + 1] - a[j2 + 1];
        x2r = a[j1] + a[j3];
        x2i = a[j1 + 1] + a[j3 + 1];
        x3r = a[j1] - a[j3];
        x3i = a[j1 + 1] - a[j3 + 1];
        a[j0] = x0r + x2r;
        a[j0 + 1] = x0i + x2i;
        a[j1] = x0r - x2r;
        a[j1 + 1] = x0i - x2i;
        x0r = x1r - x3i;
        x0i = x1i + x3r;
        a[j2] = wk1i * x0r - wk1r * x0i;
        a[j2 + 1] = wk1i * x0i + wk1r * x0r;
        x0r = x1r + x3i;
        x0i = x1i - x3r;
        a[j3] = wk3i * x0r + wk3r * x0i;
        a[j3 + 1] = wk3i * x0i - wk3r * x0r;
    }
    j0 = mh;
    j1 = j0 + m;
    j2 = j1 + m;
    j3 = j2 + m;
    x0r = a[j0] + a[j2];
    x0i = a[j0 + 1] + a[j2 + 1];
    x1r = a[j0] - a[j2];
    x1i = a[j0 + 1] - a[j2 + 1];
    x2r = a[j1] + a[j3];
    x2i = a[j1 + 1] + a[j3 + 1];
    x3r = a[j1] - a[j3];
    x3i = a[j1 + 1] - a[j3 + 1];
    a[j0] = x0r + x2r;
    a[j0 + 1] = x0i + x2i;
    a[j1] = x0r - x2r;
    a[j1 + 1] = x0i - x2i;
    x0r = x1r - x3i;
    x0i = x1i + x3r;
    a[j2] = wn4r * (x0r - x0i);
    a[j2 + 1] = wn4r * (x0i + x0r);
    x0r = x1r + x3i;
    x0i = x1i - x3r;
    a[j3] = -wn4r * (x0r + x0i);
    a[j3 + 1] = -wn4r * (x0i - x0r);
}

static void cftmdl2(int n, double *a, double *w) {
    int j, j0, j1, j2, j3, k, kr, m, mh;
    double wn4r, wk1r, wk1i, wk3r, wk3i, wd1r, wd1i, wd3r, wd3i;
    double x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i, y0r, y0i, y2r, y2i;

    mh = n >> 3;
    m = 2 * mh;
    wn4r = w[1];
    j1 = m;
    j2 = j1 + m;
    j3 = j2 + m;
    x0r = a[0] - a[j2 + 1];
    x0i = a[1] + a[j2];
    x1r = a[0] + a[j2 + 1];
    x1i = a[1] - a[j2];
    x2r = a[j1] - a[j3 + 1];
    x2i = a[j1 + 1] + a[j3];
    x3r = a[j1] + a[j3 + 1];
    x3i = a[j1 + 1] - a[j3];
    y0r = wn4r * (x2r - x2i);
    y0i = wn4r * (x2i + x2r);
    a[0] = x0r + y0r;
    a[1] = x0i + y0i;
    a[j1] = x0r - y0r;
    a[j1 + 1] = x0i - y0i;
    y0r = wn4r * (x3r - x3i);
    y0i = wn4r * (x3i + x3r);
    a[j2] = x1r - y0i;
    a[j2 + 1] = x1i + y0r;
    a[j3] = x1r + y0i;
    a[j3 + 1] = x1i - y0r;
    k = 0;
    kr = 2 * m;
    for (j = 2; j < mh; j += 2) {
        k += 4;
        wk1r = w[k];
        wk1i = w[k + 1];
        wk3r = w[k + 2];
        wk3i = w[k + 3];
        kr -= 4;
        wd1i = w[kr];
        wd1r = w[kr + 1];
        wd3i = w[kr + 2];
        wd3r = w[kr + 3];
        j1 = j + m;
        j2 = j1 + m;
        j3 = j2 + m;
        x0r = a[j] - a[j2 + 1];
        x0i = a[j + 1] + a[j2];
        x1r = a[j] + a[j2 + 1];
        x1i = a[j + 1] - a[j2];
        x2r = a[j1] - a[j3 + 1];
        x2i = a[j1 + 1] + a[j3];
        x3r = a[j1] + a[j3 + 1];
        x3i = a[j1 + 1] - a[j3];
        y0r = wk1r * x0r - wk1i * x0i;
        y0i = wk1r * x0i + wk1i * x0r;
        y2r = wd1r * x2r - wd1i * x2i;
        y2i = wd1r * x2i + wd1i * x2r;
        a[j] = y0r + y2r;
        a[j + 1] = y0i + y2i;
        a[j1] = y0r - y2r;
        a[j1 + 1] = y0i - y2i;
        y0r = wk3r * x1r + wk3i * x1i;
        y0i = wk3r * x1i - wk3i * x1r;
        y2r = wd3r * x3r + wd3i * x3i;
        y2i = wd3r * x3i - wd3i * x3r;
        a[j2] = y0r + y2r;
        a[j2 + 1] = y0i + y2i;
        a[j3] = y0r - y2r;
        a[j3 + 1] = y0i - y2i;
        j0 = m - j;
        j1 = j0 + m;
        j2 = j1 + m;
        j3 = j2 + m;
        x0r = a[j0] - a[j2 + 1];
        x0i = a[j0 + 1] + a[j2];
        x1r = a[j0] + a[j2 + 1];
        x1i = a[j0 + 1] - a[j2];
        x2r = a[j1] - a[j3 + 1];
        x2i = a[j1 + 1] + a[j3];
        x3r = a[j1] + a[j3 + 1];
        x3i = a[j1 + 1] - a[j3];
        y0r = wd1i * x0r - wd1r * x0i;
        y0i = wd1i * x0i + wd1r * x0r;
        y2r = wk1i * x2r - wk1r * x2i;
        y2i = wk1i * x2i + wk1r * x2r;
        a[j0] = y0r + y2r;
        a[j0 + 1] = y0i + y2i;
        a[j1] = y0r - y2r;
        a[j1 + 1] = y0i - y2i;
        y0r = wd3i * x1r + wd3r * x1i;
        y0i = wd3i * x1i - wd3r * x1r;
        y2r = wk3i * x3r + wk3r * x3i;
        y2i = wk3i * x3i - wk3r * x3r;
        a[j2] = y0r + y2r;
        a[j2 + 1] = y0i + y2i;
        a[j3] = y0r - y2r;
        a[j3 + 1] = y0i - y2i;
    }
    wk1r = w[m];
    wk1i = w[m + 1];
    j0 = mh;
    j1 = j0 + m;
    j2 = j1 + m;
    j3 = j2 + m;
    x0r = a[j0] - a[j2 + 1];
    x0i = a[j0 + 1] + a[j2];
    x1r = a[j0] + a[j2 + 1];
    x1i = a[j0 + 1] - a[j2];
    x2r = a[j1] - a[j3 + 1];
    x2i = a[j1 + 1] + a[j3];
    x3r = a[j1] + a[j3 + 1];
    x3i = a[j1 + 1] - a[j3];
    y0r = wk1r * x0r - wk1i * x0i;
    y0i = wk1r * x0i + wk1i * x0r;
    y2r = wk1i * x2r - wk1r * x2i;
    y2i = wk1i * x2i + wk1r * x2r;
    a[j0] = y0r + y2r;
    a[j0 + 1] = y0i + y2i;
    a[j1] = y0r - y2r;
    a[j1 + 1] = y0i - y2i;
    y0r = wk1i * x1r - wk1r * x1i;
    y0i = wk1i * x1i + wk1r * x1r;
    y2r = wk1r * x3r - wk1i * x3i;
    y2i = wk1r * x3i + wk1i * x3r;
    a[j2] = y0r - y2r;
    a[j2 + 1] = y0i - y2i;
    a[j3] = y0r + y2r;
    a[j3 + 1] = y0i + y2i;
}

static int cfttree(int n, int j, int k, double *a, int nw, double *w) {
    int i, isplt, m;

    if ((k & 3) != 0) {
        isplt = k & 1;
        if (isplt != 0) {
            cftmdl1(n, &a[j - n], &w[nw - (n >> 1)]);
        } else {
            cftmdl2(n, &a[j - n], &w[nw - n]);
        }
    } else {
        m = n;
        for (i = k; (i & 3) == 0; i >>= 2) {
            m <<= 2;
        }
        isplt = i & 1;
        if (isplt != 0) {
            while (m > 128) {
                cftmdl1(m, &a[j - m], &w[nw - (m >> 1)]);
                m >>= 2;
            }
        } else {
            while (m > 128) {
                cftmdl2(m, &a[j - m], &w[nw - m]);
                m >>= 2;
            }
        }
    }
    return isplt;
}

static void cftf161(double *a, double *w) {
    double wn4r, wk1r, wk1i, x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i, y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i, y4r, y4i,
        y5r, y5i, y6r, y6i, y7r, y7i, y8r, y8i, y9r, y9i, y10r, y10i, y11r, y11i, y12r, y12i, y13r, y13i, y14r, y14i,
        y15r, y15i;

    wn4r = w[1];
    wk1r = w[2];
    wk1i = w[3];
    x0r = a[0] + a[16];
    x0i = a[1] + a[17];
    x1r = a[0] - a[16];
    x1i = a[1] - a[17];
    x2r = a[8] + a[24];
    x2i = a[9] + a[25];
    x3r = a[8] - a[24];
    x3i = a[9] - a[25];
    y0r = x0r + x2r;
    y0i = x0i + x2i;
    y4r = x0r - x2r;
    y4i = x0i - x2i;
    y8r = x1r - x3i;
    y8i = x1i + x3r;
    y12r = x1r + x3i;
    y12i = x1i - x3r;
    x0r = a[2] + a[18];
    x0i = a[3] + a[19];
    x1r = a[2] - a[18];
    x1i = a[3] - a[19];
    x2r = a[10] + a[26];
    x2i = a[11] + a[27];
    x3r = a[10] - a[26];
    x3i = a[11] - a[27];
    y1r = x0r + x2r;
    y1i = x0i + x2i;
    y5r = x0r - x2r;
    y5i = x0i - x2i;
    x0r = x1r - x3i;
    x0i = x1i + x3r;
    y9r = wk1r * x0r - wk1i * x0i;
    y9i = wk1r * x0i + wk1i * x0r;
    x0r = x1r + x3i;
    x0i = x1i - x3r;
    y13r = wk1i * x0r - wk1r * x0i;
    y13i = wk1i * x0i + wk1r * x0r;
    x0r = a[4] + a[20];
    x0i = a[5] + a[21];
    x1r = a[4] - a[20];
    x1i = a[5] - a[21];
    x2r = a[12] + a[28];
    x2i = a[13] + a[29];
    x3r = a[12] - a[28];
    x3i = a[13] - a[29];
    y2r = x0r + x2r;
    y2i = x0i + x2i;
    y6r = x0r - x2r;
    y6i = x0i - x2i;
    x0r = x1r - x3i;
    x0i = x1i + x3r;
    y10r = wn4r * (x0r - x0i);
    y10i = wn4r * (x0i + x0r);
    x0r = x1r + x3i;
    x0i = x1i - x3r;
    y14r = wn4r * (x0r + x0i);
    y14i = wn4r * (x0i - x0r);
    x0r = a[6] + a[22];
    x0i = a[7] + a[23];
    x1r = a[6] - a[22];
    x1i = a[7] - a[23];
    x2r = a[14] + a[30];
    x2i = a[15] + a[31];
    x3r = a[14] - a[30];
    x3i = a[15] - a[31];
    y3r = x0r + x2r;
    y3i = x0i + x2i;
    y7r = x0r - x2r;
    y7i = x0i - x2i;
    x0r = x1r - x3i;
    x0i = x1i + x3r;
    y11r = wk1i * x0r - wk1r * x0i;
    y11i = wk1i * x0i + wk1r * x0r;
    x0r = x1r + x3i;
    x0i = x1i - x3r;
    y15r = wk1r * x0r - wk1i * x0i;
    y15i = wk1r * x0i + wk1i * x0r;
    x0r = y12r - y14r;
    x0i = y12i - y14i;
    x1r = y12r + y14r;
    x1i = y12i + y14i;
    x2r = y13r - y15r;
    x2i = y13i - y15i;
    x3r = y13r + y15r;
    x3i = y13i + y15i;
    a[24] = x0r + x2r;
    a[25] = x0i + x2i;
    a[26] = x0r - x2r;
    a[27] = x0i - x2i;
    a[28] = x1r - x3i;
    a[29] = x1i + x3r;
    a[30] = x1r + x3i;
    a[31] = x1i - x3r;
    x0r = y8r + y10r;
    x0i = y8i + y10i;
    x1r = y8r - y10r;
    x1i = y8i - y10i;
    x2r = y9r + y11r;
    x2i = y9i + y11i;
    x3r = y9r - y11r;
    x3i = y9i - y11i;
    a[16] = x0r + x2r;
    a[17] = x0i + x2i;
    a[18] = x0r - x2r;
    a[19] = x0i - x2i;
    a[20] = x1r - x3i;
    a[21] = x1i + x3r;
    a[22] = x1r + x3i;
    a[23] = x1i - x3r;
    x0r = y5r - y7i;
    x0i = y5i + y7r;
    x2r = wn4r * (x0r - x0i);
    x2i = wn4r * (x0i + x0r);
    x0r = y5r + y7i;
    x0i = y5i - y7r;
    x3r = wn4r * (x0r - x0i);
    x3i = wn4r * (x0i + x0r);
    x0r = y4r - y6i;
    x0i = y4i + y6r;
    x1r = y4r + y6i;
    x1i = y4i - y6r;
    a[8] = x0r + x2r;
    a[9] = x0i + x2i;
    a[10] = x0r - x2r;
    a[11] = x0i - x2i;
    a[12] = x1r - x3i;
    a[13] = x1i + x3r;
    a[14] = x1r + x3i;
    a[15] = x1i - x3r;
    x0r = y0r + y2r;
    x0i = y0i + y2i;
    x1r = y0r - y2r;
    x1i = y0i - y2i;
    x2r = y1r + y3r;
    x2i = y1i + y3i;
    x3r = y1r - y3r;
    x3i = y1i - y3i;
    a[0] = x0r + x2r;
    a[1] = x0i + x2i;
    a[2] = x0r - x2r;
    a[3] = x0i - x2i;
    a[4] = x1r - x3i;
    a[5] = x1i + x3r;
    a[6] = x1r + x3i;
    a[7] = x1i - x3r;
}

static void cftf162(double *a, double *w) {
    double wn4r, wk1r, wk1i, wk2r, wk2i, wk3r, wk3i, x0r, x0i, x1r, x1i, x2r, x2i, y0r, y0i, y1r, y1i, y2r, y2i, y3r,
        y3i, y4r, y4i, y5r, y5i, y6r, y6i, y7r, y7i, y8r, y8i, y9r, y9i, y10r, y10i, y11r, y11i, y12r, y12i, y13r, y13i,
        y14r, y14i, y15r, y15i;

    wn4r = w[1];
    wk1r = w[4];
    wk1i = w[5];
    wk3r = w[6];
    wk3i = -w[7];
    wk2r = w[8];
    wk2i = w[9];
    x1r = a[0] - a[17];
    x1i = a[1] + a[16];
    x0r = a[8] - a[25];
    x0i = a[9] + a[24];
    x2r = wn4r * (x0r - x0i);
    x2i = wn4r * (x0i + x0r);
    y0r = x1r + x2r;
    y0i = x1i + x2i;
    y4r = x1r - x2r;
    y4i = x1i - x2i;
    x1r = a[0] + a[17];
    x1i = a[1] - a[16];
    x0r = a[8] + a[25];
    x0i = a[9] - a[24];
    x2r = wn4r * (x0r - x0i);
    x2i = wn4r * (x0i + x0r);
    y8r = x1r - x2i;
    y8i = x1i + x2r;
    y12r = x1r + x2i;
    y12i = x1i - x2r;
    x0r = a[2] - a[19];
    x0i = a[3] + a[18];
    x1r = wk1r * x0r - wk1i * x0i;
    x1i = wk1r * x0i + wk1i * x0r;
    x0r = a[10] - a[27];
    x0i = a[11] + a[26];
    x2r = wk3i * x0r - wk3r * x0i;
    x2i = wk3i * x0i + wk3r * x0r;
    y1r = x1r + x2r;
    y1i = x1i + x2i;
    y5r = x1r - x2r;
    y5i = x1i - x2i;
    x0r = a[2] + a[19];
    x0i = a[3] - a[18];
    x1r = wk3r * x0r - wk3i * x0i;
    x1i = wk3r * x0i + wk3i * x0r;
    x0r = a[10] + a[27];
    x0i = a[11] - a[26];
    x2r = wk1r * x0r + wk1i * x0i;
    x2i = wk1r * x0i - wk1i * x0r;
    y9r = x1r - x2r;
    y9i = x1i - x2i;
    y13r = x1r + x2r;
    y13i = x1i + x2i;
    x0r = a[4] - a[21];
    x0i = a[5] + a[20];
    x1r = wk2r * x0r - wk2i * x0i;
    x1i = wk2r * x0i + wk2i * x0r;
    x0r = a[12] - a[29];
    x0i = a[13] + a[28];
    x2r = wk2i * x0r - wk2r * x0i;
    x2i = wk2i * x0i + wk2r * x0r;
    y2r = x1r + x2r;
    y2i = x1i + x2i;
    y6r = x1r - x2r;
    y6i = x1i - x2i;
    x0r = a[4] + a[21];
    x0i = a[5] - a[20];
    x1r = wk2i * x0r - wk2r * x0i;
    x1i = wk2i * x0i + wk2r * x0r;
    x0r = a[12] + a[29];
    x0i = a[13] - a[28];
    x2r = wk2r * x0r - wk2i * x0i;
    x2i = wk2r * x0i + wk2i * x0r;
    y10r = x1r - x2r;
    y10i = x1i - x2i;
    y14r = x1r + x2r;
    y14i = x1i + x2i;
    x0r = a[6] - a[23];
    x0i = a[7] + a[22];
    x1r = wk3r * x0r - wk3i * x0i;
    x1i = wk3r * x0i + wk3i * x0r;
    x0r = a[14] - a[31];
    x0i = a[15] + a[30];
    x2r = wk1i * x0r - wk1r * x0i;
    x2i = wk1i * x0i + wk1r * x0r;
    y3r = x1r + x2r;
    y3i = x1i + x2i;
    y7r = x1r - x2r;
    y7i = x1i - x2i;
    x0r = a[6] + a[23];
    x0i = a[7] - a[22];
    x1r = wk1i * x0r + wk1r * x0i;
    x1i = wk1i * x0i - wk1r * x0r;
    x0r = a[14] + a[31];
    x0i = a[15] - a[30];
    x2r = wk3i * x0r - wk3r * x0i;
    x2i = wk3i * x0i + wk3r * x0r;
    y11r = x1r + x2r;
    y11i = x1i + x2i;
    y15r = x1r - x2r;
    y15i = x1i - x2i;
    x1r = y0r + y2r;
    x1i = y0i + y2i;
    x2r = y1r + y3r;
    x2i = y1i + y3i;
    a[0] = x1r + x2r;
    a[1] = x1i + x2i;
    a[2] = x1r - x2r;
    a[3] = x1i - x2i;
    x1r = y0r - y2r;
    x1i = y0i - y2i;
    x2r = y1r - y3r;
    x2i = y1i - y3i;
    a[4] = x1r - x2i;
    a[5] = x1i + x2r;
    a[6] = x1r + x2i;
    a[7] = x1i - x2r;
    x1r = y4r - y6i;
    x1i = y4i + y6r;
    x0r = y5r - y7i;
    x0i = y5i + y7r;
    x2r = wn4r * (x0r - x0i);
    x2i = wn4r * (x0i + x0r);
    a[8] = x1r + x2r;
    a[9] = x1i + x2i;
    a[10] = x1r - x2r;
    a[11] = x1i - x2i;
    x1r = y4r + y6i;
    x1i = y4i - y6r;
    x0r = y5r + y7i;
    x0i = y5i - y7r;
    x2r = wn4r * (x0r - x0i);
    x2i = wn4r * (x0i + x0r);
    a[12] = x1r - x2i;
    a[13] = x1i + x2r;
    a[14] = x1r + x2i;
    a[15] = x1i - x2r;
    x1r = y8r + y10r;
    x1i = y8i + y10i;
    x2r = y9r - y11r;
    x2i = y9i - y11i;
    a[16] = x1r + x2r;
    a[17] = x1i + x2i;
    a[18] = x1r - x2r;
    a[19] = x1i - x2i;
    x1r = y8r - y10r;
    x1i = y8i - y10i;
    x2r = y9r + y11r;
    x2i = y9i + y11i;
    a[20] = x1r - x2i;
    a[21] = x1i + x2r;
    a[22] = x1r + x2i;
    a[23] = x1i - x2r;
    x1r = y12r - y14i;
    x1i = y12i + y14r;
    x0r = y13r + y15i;
    x0i = y13i - y15r;
    x2r = wn4r * (x0r - x0i);
    x2i = wn4r * (x0i + x0r);
    a[24] = x1r + x2r;
    a[25] = x1i + x2i;
    a[26] = x1r - x2r;
    a[27] = x1i - x2i;
    x1r = y12r + y14i;
    x1i = y12i - y14r;
    x0r = y13r - y15i;
    x0i = y13i + y15r;
    x2r = wn4r * (x0r - x0i);
    x2i = wn4r * (x0i + x0r);
    a[28] = x1r - x2i;
    a[29] = x1i + x2r;
    a[30] = x1r + x2i;
    a[31] = x1i - x2r;
}

static void cftf081(double *a, double *w) {
    double wn4r, x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i, y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i, y4r, y4i, y5r, y5i,
        y6r, y6i, y7r, y7i;

    wn4r = w[1];
    x0r = a[0] + a[8];
    x0i = a[1] + a[9];
    x1r = a[0] - a[8];
    x1i = a[1] - a[9];
    x2r = a[4] + a[12];
    x2i = a[5] + a[13];
    x3r = a[4] - a[12];
    x3i = a[5] - a[13];
    y0r = x0r + x2r;
    y0i = x0i + x2i;
    y2r = x0r - x2r;
    y2i = x0i - x2i;
    y1r = x1r - x3i;
    y1i = x1i + x3r;
    y3r = x1r + x3i;
    y3i = x1i - x3r;
    x0r = a[2] + a[10];
    x0i = a[3] + a[11];
    x1r = a[2] - a[10];
    x1i = a[3] - a[11];
    x2r = a[6] + a[14];
    x2i = a[7] + a[15];
    x3r = a[6] - a[14];
    x3i = a[7] - a[15];
    y4r = x0r + x2r;
    y4i = x0i + x2i;
    y6r = x0r - x2r;
    y6i = x0i - x2i;
    x0r = x1r - x3i;
    x0i = x1i + x3r;
    x2r = x1r + x3i;
    x2i = x1i - x3r;
    y5r = wn4r * (x0r - x0i);
    y5i = wn4r * (x0r + x0i);
    y7r = wn4r * (x2r - x2i);
    y7i = wn4r * (x2r + x2i);
    a[8] = y1r + y5r;
    a[9] = y1i + y5i;
    a[10] = y1r - y5r;
    a[11] = y1i - y5i;
    a[12] = y3r - y7i;
    a[13] = y3i + y7r;
    a[14] = y3r + y7i;
    a[15] = y3i - y7r;
    a[0] = y0r + y4r;
    a[1] = y0i + y4i;
    a[2] = y0r - y4r;
    a[3] = y0i - y4i;
    a[4] = y2r - y6i;
    a[5] = y2i + y6r;
    a[6] = y2r + y6i;
    a[7] = y2i - y6r;
}

static void cftf082(double *a, double *w) {
    double wn4r, wk1r, wk1i, x0r, x0i, x1r, x1i, y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i, y4r, y4i, y5r, y5i, y6r, y6i,
        y7r, y7i;

    wn4r = w[1];
    wk1r = w[2];
    wk1i = w[3];
    y0r = a[0] - a[9];
    y0i = a[1] + a[8];
    y1r = a[0] + a[9];
    y1i = a[1] - a[8];
    x0r = a[4] - a[13];
    x0i = a[5] + a[12];
    y2r = wn4r * (x0r - x0i);
    y2i = wn4r * (x0i + x0r);
    x0r = a[4] + a[13];
    x0i = a[5] - a[12];
    y3r = wn4r * (x0r - x0i);
    y3i = wn4r * (x0i + x0r);
    x0r = a[2] - a[11];
    x0i = a[3] + a[10];
    y4r = wk1r * x0r - wk1i * x0i;
    y4i = wk1r * x0i + wk1i * x0r;
    x0r = a[2] + a[11];
    x0i = a[3] - a[10];
    y5r = wk1i * x0r - wk1r * x0i;
    y5i = wk1i * x0i + wk1r * x0r;
    x0r = a[6] - a[15];
    x0i = a[7] + a[14];
    y6r = wk1i * x0r - wk1r * x0i;
    y6i = wk1i * x0i + wk1r * x0r;
    x0r = a[6] + a[15];
    x0i = a[7] - a[14];
    y7r = wk1r * x0r - wk1i * x0i;
    y7i = wk1r * x0i + wk1i * x0r;
    x0r = y0r + y2r;
    x0i = y0i + y2i;
    x1r = y4r + y6r;
    x1i = y4i + y6i;
    a[0] = x0r + x1r;
    a[1] = x0i + x1i;
    a[2] = x0r - x1r;
    a[3] = x0i - x1i;
    x0r = y0r - y2r;
    x0i = y0i - y2i;
    x1r = y4r - y6r;
    x1i = y4i - y6i;
    a[4] = x0r - x1i;
    a[5] = x0i + x1r;
    a[6] = x0r + x1i;
    a[7] = x0i - x1r;
    x0r = y1r - y3i;
    x0i = y1i + y3r;
    x1r = y5r - y7r;
    x1i = y5i - y7i;
    a[8] = x0r + x1r;
    a[9] = x0i + x1i;
    a[10] = x0r - x1r;
    a[11] = x0i - x1i;
    x0r = y1r + y3i;
    x0i = y1i - y3r;
    x1r = y5r + y7r;
    x1i = y5i + y7i;
    a[12] = x0r - x1i;
    a[13] = x0i + x1r;
    a[14] = x0r + x1i;
    a[15] = x0i - x1r;
}

static void cftf040(double *a) {
    double x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i;

    x0r = a[0] + a[4];
    x0i = a[1] + a[5];
    x1r = a[0] - a[4];
    x1i = a[1] - a[5];
    x2r = a[2] + a[6];
    x2i = a[3] + a[7];
    x3r = a[2] - a[6];
    x3i = a[3] - a[7];
    a[0] = x0r + x2r;
    a[1] = x0i + x2i;
    a[2] = x1r - x3i;
    a[3] = x1i + x3r;
    a[4] = x0r - x2r;
    a[5] = x0i - x2i;
    a[6] = x1r + x3i;
    a[7] = x1i - x3r;
}

static void cftb040(double *a) {
    double x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i;

    x0r = a[0] + a[4];
    x0i = a[1] + a[5];
    x1r = a[0] - a[4];
    x1i = a[1] - a[5];
    x2r = a[2] + a[6];
    x2i = a[3] + a[7];
    x3r = a[2] - a[6];
    x3i = a[3] - a[7];
    a[0] = x0r + x2r;
    a[1] = x0i + x2i;
    a[2] = x1r + x3i;
    a[3] = x1i - x3r;
    a[4] = x0r - x2r;
    a[5] = x0i - x2i;
    a[6] = x1r - x3i;
    a[7] = x1i + x3r;
}

static void cftx020(double *a) {
    double x0r, x0i;

    x0r = a[0] - a[2];
    x0i = a[1] - a[3];
    a[0] += a[2];
    a[1] += a[3];
    a[2] = x0r;
    a[3] = x0i;
}

static void cftfx41(int n, double *a, int nw, double *w) {
    if (n == 128) {
        cftf161(a, &w[nw - 8]);
        cftf162(&a[32], &w[nw - 32]);
        cftf161(&a[64], &w[nw - 8]);
        cftf161(&a[96], &w[nw - 8]);
    } else {
        cftf081(a, &w[nw - 8]);
        cftf082(&a[16], &w[nw - 8]);
        cftf081(&a[32], &w[nw - 8]);
        cftf081(&a[48], &w[nw - 8]);
    }
}

static void cftleaf(int n, int isplt, double *a, int nw, double *w) {
    if (n == 512) {
        cftmdl1(128, a, &w[nw - 64]);
        cftf161(a, &w[nw - 8]);
        cftf162(&a[32], &w[nw - 32]);
        cftf161(&a[64], &w[nw - 8]);
        cftf161(&a[96], &w[nw - 8]);
        cftmdl2(128, &a[128], &w[nw - 128]);
        cftf161(&a[128], &w[nw - 8]);
        cftf162(&a[160], &w[nw - 32]);
        cftf161(&a[192], &w[nw - 8]);
        cftf162(&a[224], &w[nw - 32]);
        cftmdl1(128, &a[256], &w[nw - 64]);
        cftf161(&a[256], &w[nw - 8]);
        cftf162(&a[288], &w[nw - 32]);
        cftf161(&a[320], &w[nw - 8]);
        cftf161(&a[352], &w[nw - 8]);
        if (isplt != 0) {
            cftmdl1(128, &a[384], &w[nw - 64]);
            cftf161(&a[480], &w[nw - 8]);
        } else {
            cftmdl2(128, &a[384], &w[nw - 128]);
            cftf162(&a[480], &w[nw - 32]);
        }
        cftf161(&a[384], &w[nw - 8]);
        cftf162(&a[416], &w[nw - 32]);
        cftf161(&a[448], &w[nw - 8]);
    } else {
        cftmdl1(64, a, &w[nw - 32]);
        cftf081(a, &w[nw - 8]);
        cftf082(&a[16], &w[nw - 8]);
        cftf081(&a[32], &w[nw - 8]);
        cftf081(&a[48], &w[nw - 8]);
        cftmdl2(64, &a[64], &w[nw - 64]);
        cftf081(&a[64], &w[nw - 8]);
        cftf082(&a[80], &w[nw - 8]);
        cftf081(&a[96], &w[nw - 8]);
        cftf082(&a[112], &w[nw - 8]);
        cftmdl1(64, &a[128], &w[nw - 32]);
        cftf081(&a[128], &w[nw - 8]);
        cftf082(&a[144], &w[nw - 8]);
        cftf081(&a[160], &w[nw - 8]);
        cftf081(&a[176], &w[nw - 8]);
        if (isplt != 0) {
            cftmdl1(64, &a[192], &w[nw - 32]);
            cftf081(&a[240], &w[nw - 8]);
        } else {
            cftmdl2(64, &a[192], &w[nw - 64]);
            cftf082(&a[240], &w[nw - 8]);
        }
        cftf081(&a[192], &w[nw - 8]);
        cftf082(&a[208], &w[nw - 8]);
        cftf081(&a[224], &w[nw - 8]);
    }
}

static void cftrec4(int n, double *a, int nw, double *w) {
    int isplt, j, k, m;

    m = n;
    while (m > 512) {
        m >>= 2;
        cftmdl1(m, &a[n - m], &w[nw - (m >> 1)]);
    }
    cftleaf(m, 1, &a[n - m], nw, w);
    k = 0;
    for (j = n - m; j > 0; j -= m) {
        k++;
        isplt = cfttree(m, j, k, a, nw, w);
        cftleaf(m, isplt, &a[j - m], nw, w);
    }
}

static void rftfsub(int n, double *a, int nc, double *c) {
    int j, k, kk, ks, m;
    double wkr, wki, xr, xi, yr, yi;

    m = n >> 1;
    ks = 2 * nc / m;
    kk = 0;
    for (j = 2; j < m; j += 2) {
        k = n - j;
        kk += ks;
        wkr = 0.5 - c[nc - kk];
        wki = c[kk];
        xr = a[j] - a[k];
        xi = a[j + 1] + a[k + 1];
        yr = wkr * xr - wki * xi;
        yi = wkr * xi + wki * xr;
        a[j] -= yr;
        a[j + 1] -= yi;
        a[k] += yr;
        a[k + 1] -= yi;
    }
}

static void rftbsub(int n, double *a, int nc, double *c) {
    int j, k, kk, ks, m;
    double wkr, wki, xr, xi, yr, yi;

    m = n >> 1;
    ks = 2 * nc / m;
    kk = 0;
    for (j = 2; j < m; j += 2) {
        k = n - j;
        kk += ks;
        wkr = 0.5 - c[nc - kk];
        wki = c[kk];
        xr = a[j] - a[k];
        xi = a[j + 1] + a[k + 1];
        yr = wkr * xr + wki * xi;
        yi = wkr * xi - wki * xr;
        a[j] -= yr;
        a[j + 1] -= yi;
        a[k] += yr;
        a[k + 1] -= yi;
    }
}

#ifdef USE_CDFT_THREADS
struct cdft_arg_st {
    int n0;
    int n;
    double *a;
    int nw;
    double *w;
};
typedef struct cdft_arg_st cdft_arg_t;

static void cftrec4_th(int n, double *a, int nw, double *w) {
    int i, idiv4, m, nthread;
    cdft_thread_t th[4];
    cdft_arg_t ag[4];

    nthread = 2;
    idiv4 = 0;
    m = n >> 1;
    if (n > CDFT_4THREADS_BEGIN_N) {
        nthread = 4;
        idiv4 = 1;
        m >>= 1;
    }
    for (i = 0; i < nthread; i++) {
        ag[i].n0 = n;
        ag[i].n = m;
        ag[i].a = &a[i * m];
        ag[i].nw = nw;
        ag[i].w = w;
        if (i != idiv4) {
            cdft_thread_create(&th[i], cftrec1_th, &ag[i]);
        } else {
            cdft_thread_create(&th[i], cftrec2_th, &ag[i]);
        }
    }
    for (i = 0; i < nthread; i++) {
        cdft_thread_wait(th[i]);
    }
}

static void *cftrec1_th(void *p) {
    int cfttree(int n, int j, int k, double *a, int nw, double *w);
    void cftleaf(int n, int isplt, double *a, int nw, double *w);
    void cftmdl1(int n, double *a, double *w);
    int isplt, j, k, m, n, n0, nw;
    double *a, *w;

    n0 = ((cdft_arg_t *)p)->n0;
    n = ((cdft_arg_t *)p)->n;
    a = ((cdft_arg_t *)p)->a;
    nw = ((cdft_arg_t *)p)->nw;
    w = ((cdft_arg_t *)p)->w;
    m = n0;
    while (m > 512) {
        m >>= 2;
        cftmdl1(m, &a[n - m], &w[nw - (m >> 1)]);
    }
    cftleaf(m, 1, &a[n - m], nw, w);
    k = 0;
    for (j = n - m; j > 0; j -= m) {
        k++;
        isplt = cfttree(m, j, k, a, nw, w);
        cftleaf(m, isplt, &a[j - m], nw, w);
    }
    return (void *)0;
}

static void *cftrec2_th(void *p) {
    int cfttree(int n, int j, int k, double *a, int nw, double *w);
    void cftleaf(int n, int isplt, double *a, int nw, double *w);
    void cftmdl2(int n, double *a, double *w);
    int isplt, j, k, m, n, n0, nw;
    double *a, *w;

    n0 = ((cdft_arg_t *)p)->n0;
    n = ((cdft_arg_t *)p)->n;
    a = ((cdft_arg_t *)p)->a;
    nw = ((cdft_arg_t *)p)->nw;
    w = ((cdft_arg_t *)p)->w;
    k = 1;
    m = n0;
    while (m > 512) {
        m >>= 2;
        k <<= 2;
        cftmdl2(m, &a[n - m], &w[nw - m]);
    }
    cftleaf(m, 0, &a[n - m], nw, w);
    k >>= 1;
    for (j = n - m; j > 0; j -= m) {
        k++;
        isplt = cfttree(m, j, k, a, nw, w);
        cftleaf(m, isplt, &a[j - m], nw, w);
    }
    return (void *)0;
}
#endif /* USE_CDFT_THREADS */

static void cftbsub(int n, double *a, int *ip, int nw, double *w) {
    if (n > 8) {
        if (n > 32) {
            cftb1st(n, a, &w[nw - (n >> 2)]);
#ifdef USE_CDFT_THREADS
            if (n > CDFT_THREADS_BEGIN_N) {
                cftrec4_th(n, a, nw, w);
            } else
#endif /* USE_CDFT_THREADS */
                if (n > 512) {
                    cftrec4(n, a, nw, w);
                } else if (n > 128) {
                    cftleaf(n, 1, a, nw, w);
                } else {
                    cftfx41(n, a, nw, w);
                }
            bitrv2conj(n, ip, a);
        } else if (n == 32) {
            cftf161(a, &w[nw - 8]);
            bitrv216neg(a);
        } else {
            cftf081(a, w);
            bitrv208neg(a);
        }
    } else if (n == 8) {
        cftb040(a);
    } else if (n == 4) {
        cftx020(a);
    }
}

static void cftfsub(int n, double *a, int *ip, int nw, double *w) {
    if (n > 8) {
        if (n > 32) {
            cftf1st(n, a, &w[nw - (n >> 2)]);
#ifdef USE_CDFT_THREADS
            if (n > CDFT_THREADS_BEGIN_N) {
                cftrec4_th(n, a, nw, w);
            } else
#endif /* USE_CDFT_THREADS */
                if (n > 512) {
                    cftrec4(n, a, nw, w);
                } else if (n > 128) {
                    cftleaf(n, 1, a, nw, w);
                } else {
                    cftfx41(n, a, nw, w);
                }
            bitrv2(n, ip, a);
        } else if (n == 32) {
            cftf161(a, &w[nw - 8]);
            bitrv216(a);
        } else {
            cftf081(a, w);
            bitrv208(a);
        }
    } else if (n == 8) {
        cftf040(a);
    } else if (n == 4) {
        cftx020(a);
    }
}

void rdft(int n, int isgn, double *a, int *ip, double *w) {
    int nw, nc;
    double xi;

    nw = ip[0];
    if (n > (nw << 2)) {
        nw = n >> 2;
        makewt(nw, ip, w);
    }
    nc = ip[1];
    if (n > (nc << 2)) {
        nc = n >> 2;
        makect(nc, ip, w + nw);
    }
    if (isgn >= 0) {
        if (n > 4) {
            cftfsub(n, a, ip, nw, w);
            rftfsub(n, a, nc, w + nw);
        } else if (n == 4) {
            cftfsub(n, a, ip, nw, w);
        }
        xi = a[0] - a[1];
        a[0] += a[1];
        a[1] = xi;
    } else {
        a[1] = 0.5 * (a[0] - a[1]);
        a[0] -= a[1];
        if (n > 4) {
            rftbsub(n, a, nc, w + nw);
            cftbsub(n, a, ip, nw, w);
        } else if (n == 4) {
            cftbsub(n, a, ip, nw, w);
        }
    }
}
#endif //USE_SIMD

#if 0
void fft()
{

}

void stft()
{

}

void istft()
{


}
#endif



