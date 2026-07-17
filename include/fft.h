#ifndef __FFT_H__
#define __FFT_H__


#ifdef __cplusplus
extern "C" 
{
#endif // __cplusplus


void kf_bfly2(
    float* Foutr,
    float* Fouti,
    int          fstride,
    const float* tw_r,
    const float* tw_i,
    int          m);


void kf_bfly3(
    float* Foutr,
    float* Fouti,
    int    fstride,
    float* tw1_r,
    float* tw1_i,
    int    m);

void kf_bfly4(
    float* Foutr,
    float* Fouti,
    int fstride,
    float* twr,
    float* twi,
    int m);


void kf_bfly5(
    float* Foutr, float* Fouti,
    int fstride,
    const float* tw_r, const float* tw_i,
    int m);


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
    float* scratchi);


#ifdef __cplusplus
}
#endif // __cplusplus


#endif //  __FFT_H__
