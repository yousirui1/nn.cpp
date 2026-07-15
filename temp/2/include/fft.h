#ifndef __FFT_H__
#define __FFT_H__

#ifdef __cplusplus
extern "C" 
{
#endif // __cplusplus


void *fft_alloc(int nfft);
void fft_compute(void *fft_handle, const float* signal, float* buffer);
void fft_free(void *fft_handle);

#ifdef __cplusplus
}
#endif // __cplusplus


#endif //  __FFT_H__
