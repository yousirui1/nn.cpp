#ifndef __MATRIX_H__
#define __MATRIX_H__

#define MAX_DIMS 8
#include <float.h>

#ifdef __cplusplus
extern "C" 
{
#endif // __cplusplus


enum tensor_data_type{
    TENSOR_INT8,
    TENSOR_INT16,
    TENSOR_INT32,
    TENSOR_FLOAT16,
    TENSOR_FLOAT32,
    TENSOR_INT64,
};

enum tensor_format {
    TENSOR_NCHW = 0,                               /* data format is NCHW. */
    TENSOR_NHWC,                                   /* data format is NHWC. */

    TENSOR_FORMAT_MAX
};

enum tensor_layout {
    TENSOR_ROW_MAJOR = 0,
    TENSOR_COL_MAJOR,
};

struct shape_t 
{
    char name[128];

    int num_dims;
    int size;
    int data_type; // 
    int dims[MAX_DIMS];
    int tensor_size;

    int strides[MAX_DIMS];

    int layout;

    int format;

#ifdef RKNN
    int zp;
    float scale; 
#endif
};

typedef struct shape_t shape_t;

struct matrix_t{
	shape_t shape;
    int empty;

    float *data_fp;
    int8_t *data_i8;
    int16_t *data_i16;
    int32_t *data_i32;
    int64_t *data_i64;

    void *data;
};
typedef struct matrix_t matrix_t;

inline static const char *get_matrix_format_string(int fmt)
{
    switch (fmt)
    {   
        case TENSOR_NCHW:
            return "NCHW";
        case TENSOR_NHWC:
            return "NHWC";
        default:
            return "UNKNOW";
    }   
}

inline static const char *get_matrix_type_string(enum tensor_data_type type)
{
    switch (type)
    {
        case TENSOR_FLOAT16:
            return "FP16";
        case TENSOR_FLOAT32:
            return "FP32";
        case TENSOR_INT8:
            return "INT8";
        case TENSOR_INT16:
            return "INT16";
        case TENSOR_INT32:
            return "INT32";
        case TENSOR_INT64:
            return "INT64";
        default:
            return "UNKNOW";
    }
}

static inline int matrix_compute_offset(const matrix_t *mat, int num_dims, const int *indices)
{
    int offset = 0, i = 0;
    for(i = 0; i < num_dims; i++)
    {
        offset += indices[i] * mat->shape.strides[i];
    }
    return offset;
}

//no safe
static inline float get_matrix_fp_fast(const matrix_t *mat, int num_dims, const int *indices)
{
    return mat->data_fp[matrix_compute_offset(mat, num_dims, indices)];
}

//no safe
static inline void set_matrix_fp_fast(const matrix_t *mat, float value, int num_dims, const int *indices)
{
    mat->data_fp[matrix_compute_offset(mat, num_dims, indices)] = value;
}


//index 0 begin not 1
#define get_matrix_fp(mat, ndims, ...) \
    get_matrix_fp_fast(mat, ndims, (int[]){ __VA_ARGS__ })

#define set_matrix_fp(mat, val, ndims, ...) \
    set_matrix_fp_fast(mat, val, ndims, (int[]){ __VA_ARGS__ })


static inline int get_matrix_value_fast(const matrix_t *mat, void *value, int num_dims, const int *indices)
{
    switch(mat->shape.data_type)
    {    
        case TENSOR_INT8:
            *(int8_t *)value = mat->data_i8[matrix_compute_offset(mat, num_dims, indices)];
            break;
        case TENSOR_INT16:
            *(int16_t *)value = mat->data_i16[matrix_compute_offset(mat, num_dims, indices)];
            break;
        case TENSOR_INT32:
            *(int32_t *)value = mat->data_i32[matrix_compute_offset(mat, num_dims, indices)];
            break;
        case TENSOR_INT64:
            *(int64_t *)value = mat->data_i64[matrix_compute_offset(mat, num_dims, indices)];
            break;
        case TENSOR_FLOAT32:
        default:
            *(float *)value = mat->data_fp[matrix_compute_offset(mat, num_dims, indices)];
            break;
    } 
    return SUCCESS;
}

static inline void set_matrix_value_fast(const matrix_t *mat, void* value, int num_dims, const int *indices)
{
    switch(mat->shape.data_type)
    {    
        case TENSOR_INT8:
            mat->data_i8[matrix_compute_offset(mat, num_dims, indices)] = *(int8_t *)value;
            break;
        case TENSOR_INT16:
            mat->data_i16[matrix_compute_offset(mat, num_dims, indices)] = *(int16_t *)value;
            break;
        case TENSOR_INT32:
            mat->data_i32[matrix_compute_offset(mat, num_dims, indices)] = *(int32_t *)value;
            break;
        case TENSOR_INT64:
            mat->data_i64[matrix_compute_offset(mat, num_dims, indices)] = *(int64_t *)value;
            break;
        case TENSOR_FLOAT32:
        default:
            mat->data_fp[matrix_compute_offset(mat, num_dims, indices)] = *(float *)value;
            break;
    }    
}

//index 0 begin not 1
#define get_matrix_value(mat, value, ndims, ...) \
    get_matrix_value_fast(mat, value, ndims, (int[]){ __VA_ARGS__ })

#define set_matrix_value(mat, val, ndims, ...) \
    set_matrix_value_fast(mat, val, ndims, (int[]){ __VA_ARGS__ })


void print_shape(shape_t shape);
void print_matrix_format(matrix_t* mat, int format);
void print_matrix(matrix_t* mat);
int get_shape_size(shape_t *shape);
int set_shape(shape_t *shape, int data_type, int num_dims, ...);
int set_shape_name(shape_t *shape, const char *name);
matrix_t *matrix_alloc(int data_type, int num_dims, ...);
matrix_t *matrix_alloc_shape(shape_t shape);
matrix_t *matrix_empty(void *data, int data_type, int num_dims, ...);
int set_matrix_data(matrix_t *mat, void *data);
int set_matrix_name(matrix_t *mat, const char *name);
matrix_t *matrix_empty_shape(void *data, shape_t shape);
void matrix_free(matrix_t *mat);
matrix_t *matrix_copy(matrix_t *src);
int matrix_apply_copy(matrix_t *dst, int dst_pos, matrix_t *src, int src_pos, int len);
int matrix_alloc_data(matrix_t *mat, void *data);
int matrix_insert(matrix_t *dst, int dst_start_pos, matrix_t *src, int src_start_pos, int src_end_pos); //to do 
int matrix_erase(matrix_t *mat, int start_index, int end_index); //to do 
int matrix_clear(matrix_t *mat);
void matrix_subtract_mean(matrix_t *feature);// to do 
float matrix_log_energy(matrix_t *mat);
matrix_t* matrix_concat(matrix_t *mat1, matrix_t *mat2, int dim);//to do 
int matrix_apply_concat(matrix_t* out, matrix_t *mat1, matrix_t *mat2, int dim); //to do
int matrix_apply_pow(matrix_t *mat, float value);
int matrix_apply_floor(matrix_t *mat, float value);
int matrix_apply_log(matrix_t *mat);
int matrix_zero(matrix_t *mat, int pos, int len);
int matrix_resize_shape(matrix_t *mat, shape_t shape);
int matrix_resize(matrix_t *mat, int data_type, int num_dims, ...);
int matrix_apply_mul_element(matrix_t *mat1, matrix_t *mat2);
float matrix_energy(matrix_t *mat1, matrix_t *mat2);
int matrix_apply_add(matrix_t *dst, matrix_t *src,  float scalar);
int matrix_mul_scalar(matrix_t *mat, float scalar);
matrix_t* matrix_add(matrix_t *mat1, matrix_t *mat2,  float scalar);
matrix_t *matrix_mul(const matrix_t *mat1, const matrix_t *mat2); //to do 
int matrix_apply_mul(const matrix_t *dst, const matrix_t *src); //to do
int matrix_power_spectrum(matrix_t *mat);
int matrix_apply_add_value(matrix_t *mat, float value);
float matrix_sum(matrix_t *mat);
float matrix_mean(matrix_t *mat);
int matrix_apply_cmvn(matrix_t *mat, float *means, float *vars, int size); //to do
matrix_t* matrix_frame_db(float *data, int size, int frame_sample_length, int frame_shift_length);
int matrix_unsqueeze(matrix_t *mat, int index);
int matrix_squeeze(matrix_t *mat, int index);
int matrix_pad(matrix_t *mat, int pad_left, int pad_right, int mode); //to do 
matrix_t* matrix_transpose(matrix_t *mat); //to do 






#ifdef __cplusplus
}
#endif // __cplusplus


#endif //  __MATRIX_H__
