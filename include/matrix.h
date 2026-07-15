#ifndef __MATRIX_H__
#define __MATRIX_H__

#define MAX_DIMS 8
#include <float.h>

#ifdef __cplusplus
extern "C"
{
#endif // __cplusplus


enum tensor_data_type{
    TENSOR_UINT8,
    TENSOR_INT8,
    TENSOR_INT16,
    TENSOR_INT32,
    TENSOR_FLOAT32,
    TENSOR_INT64,
};

enum tensor_format {
    TENSOR_NCHW = 0,                               /* data format is NCHW. */
    TENSOR_NHWC,                                   /* data format is NHWC. */

    TENSOR_FORMAT_MAX
};


struct shape_t
{
    char name[128];

    int num_dims;
    int size;
    int data_type; //
    int dims[MAX_DIMS];
    int stride[MAX_DIMS];
    int tensor_size;

    int format;
    //rknn only
    int zp;
    float scale;
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
        case TENSOR_FLOAT32:
            return "FP32";
        //case TENSOR_FLOAT16:
            //return "FP16";
        case TENSOR_INT8:
            return "INT8";
        case TENSOR_UINT8:
            return "UINT8";
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

void print_shape(shape_t shape);
void print_matrix_format(matrix_t* mat, int format);
int get_shape_size(shape_t *shape);
int set_matrix_data(matrix_t *mat, void *data);
matrix_t *matrix_empty_shape(void *data, shape_t shape);
void matrix_free(matrix_t *mat);
int matrix_resize_shape(matrix_t *mat, shape_t shape);
matrix_t *matrix_alloc_shape(shape_t shape);
int set_shape(shape_t *shape, int data_type, int num_dims, ...);
void print_matrix(matrix_t* mat);

#if 0
int _set_shape_fast(shape_t *shape, int data_type, int n_dims, const int indices[]);


#define get_matrix_fp(mat, ndims, ...) \
    _get_matrix_fp_fast(mat, (int[]){ __VA_ARGS__ })

#define set_matrix_fp(mat, val, ndims, ...) \
    _set_matrix_fp_fast(mat, val, (int[]){ __VA_ARGS__ })

#define set_shape(mat, type, ndims, ...) \
    _set_shape_fast(mat, type, ndims, (int[]){ __VA_ARGS__ })

#define matrix_alloc(type, ndims, ...) \
    _matrix_alloc_fast(type, ndims, (int[]){ __VA_ARGS__ })
#endif

#ifdef __cplusplus
}
#endif // __cplusplus


#endif //  __MATRIX_H__
