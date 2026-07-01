#include "base.h"
#include "matrix.h"
#include <inttypes.h>


#define MATRIX_ALIGN 64

void print_shape(shape_t shape)
{
    int i;
    printf("%s shape  [", shape.name);
    for(i = 0; i < shape.num_dims; i++)
    {   
        printf("%d ", shape.dims[i]);
    }   
    printf("] ");

    printf(" TENSOR DATA TYPE %s \n", get_matrix_type_string(shape.data_type));
    //rk only
#ifdef RKNN
    {
        printf(" TENSOR FORMA: %s ", get_matrix_format_string(shape.format));
        printf(" rknn zp %d scale %.2f \n", shape.zp, shape.scale);
    }
#endif
}

void print_matrix_format(matrix_t* mat, int format) 
{
    int i, count;
    if(NULL == mat)
        return;

    print_shape(mat->shape);

    for(i = 0, count = 0; i < mat->shape.size; i++)
    {
        if(count % format == 0)
        {
            printf("line %d: ", count / format);
        }

        switch(mat->shape.data_type)
        {
            case TENSOR_INT8:
                printf("%d ", mat->data_i8[i]);
                break;
            case TENSOR_FLOAT32:
                printf("%.4f ", mat->data_fp[i]);
                //printf("%e ", mat->data_fp[i]);
                break;
            case TENSOR_INT32:
                printf("%d ", mat->data_i32[i]);
				break;
            case TENSOR_INT64:
                printf("% " PRId64, mat->data_i64[i]);
				break;
        }
        count ++;
        if(count % format == 0)
        {
            printf("\n");
        }
    }
    printf("\n");
}


void print_matrix(matrix_t* mat) 
{
    print_matrix_format(mat, 10);
}

int get_shape_size(shape_t *shape)
{
    int i;
    shape->size = 1;
    for(i = 0; i < shape->num_dims; i++)
    {   
        if(-1 == shape->dims[i])
        {
            shape->size *= 1;       //占位 后续realloc
        }
        else
            shape->size *= shape->dims[i];
    }   

    int stride = 1;
    if(shape->layout == TENSOR_ROW_MAJOR)
    {
        for(i = shape->num_dims - 1; i >= 0; i--)
        {
            shape->strides[i] = stride;
			stride *= shape->dims[i] == -1 ? 1 : shape->dims[i];
        }
    }
    else
    {
        for(i = 0; i < shape->num_dims; i++) 
        {
            shape->strides[i] = stride;
			stride *= shape->dims[i] == -1 ? 1 : shape->dims[i];
        }   
    }


    switch(shape->data_type)
    {   
        case TENSOR_INT8:
            shape->tensor_size = sizeof(int8_t) * shape->size;
            break;
        case TENSOR_INT16:
            shape->tensor_size = sizeof(int16_t) * shape->size;
            break;
        case TENSOR_INT32:
            shape->tensor_size = sizeof(int32_t) * shape->size;
            break;
        case TENSOR_INT64:
            shape->tensor_size = sizeof(int64_t) * shape->size;
            break;
        case TENSOR_FLOAT32:
            shape->tensor_size = sizeof(float) * shape->size;
            break;
        default:
            shape->tensor_size = sizeof(float) * shape->size;
            break;
    }   
    return shape->size;
}

int set_shape(shape_t *shape, int data_type, int num_dims, ...)
{
    int i;
    va_list args;
    shape->num_dims = num_dims;

    if(-1 != data_type)
        shape->data_type = data_type;



    va_start(args, num_dims);
    for(i = 0; i < num_dims; i++)
    {   
        shape->dims[i] = va_arg(args, int);
    }   
    va_end(args);
    get_shape_size(shape);
    return SUCCESS;
}


int set_shape_name(shape_t *shape, const char *name)
{
    if(name && shape)
    {
        strncpy(shape->name, name, sizeof(shape->name));
        return SUCCESS;
    }
    return ERROR;
}






matrix_t *matrix_alloc(int data_type, int num_dims, ...)
{
    int i;
    va_list args;
    matrix_t *out = (struct matrix_t *)malloc(sizeof(matrix_t));
    if(out)
    {
        memset(out, 0, sizeof(matrix_t));
        out->empty = 0;
        out->shape.num_dims = num_dims;
        out->shape.data_type = data_type;
        va_start(args, num_dims);
        for(i = 0; i < num_dims; i++)
        {
            out->shape.dims[i] = va_arg(args, int);
        }
        va_end(args);
        get_shape_size(&out->shape);

#if _POSIX_C_SOURCE >= 200112L
        if(posix_memalign(&out->data, MATRIX_ALIGN, out->shape.tensor_size) != 0)
            out->data = malloc(out->shape.tensor_size);
#else
        out->data = malloc(out->shape.tensor_size);
#endif

		out->data_fp = (float *)out->data;
		out->data_i8 = (int8_t *)out->data;
		out->data_i16 = (int16_t *)out->data;
		out->data_i32 = (int32_t *)out->data;
		out->data_i64 = (int64_t *)out->data;
        memset(out->data, 0, out->shape.tensor_size);
    }
    return out;
}

matrix_t *matrix_alloc_shape(shape_t shape)
{
    matrix_t *out = (struct matrix_t *)malloc(sizeof(matrix_t));
    if(out)
    {
        out->empty = 0;
        out->shape = shape;

        get_shape_size(&out->shape);

#if _POSIX_C_SOURCE >= 200112L
        if(posix_memalign(&out->data, MATRIX_ALIGN, out->shape.tensor_size) != 0)
            out->data = malloc(out->shape.tensor_size);
#else
        out->data = malloc(out->shape.tensor_size);
#endif

        memset(out->data, 0, out->shape.tensor_size);
		out->data_fp = (float *)out->data;
		out->data_i8 = (int8_t *)out->data;
		out->data_i16 = (int16_t *)out->data;
		out->data_i32 = (int32_t *)out->data;
		out->data_i64 = (int64_t *)out->data;
    }
    return out;
}

matrix_t *matrix_empty(void *data, int data_type, int num_dims, ...)
{
    int i;
    va_list args;
    matrix_t *out = (struct matrix_t *)malloc(sizeof(matrix_t));
    if(out)
    {
        memset(out, 0, sizeof(matrix_t));
        out->shape.data_type = data_type;
        out->shape.num_dims = num_dims;
        va_start(args, num_dims);
        for(i = 0; i < num_dims; i++)
        {
            out->shape.dims[i] = va_arg(args, int);
        }
        va_end(args);
        get_shape_size(&out->shape);
        out->empty = 1;
    }

    if(data)
    {
        set_matrix_data(out, data);
    }
    return out;
}

int set_matrix_data(matrix_t *mat, void *data)
{
    if(NULL == mat || 0 == mat->empty || NULL == data)
    {
        LOG_ERROR("mat or data is NULL or mat->empty != 1 mat %p mat->empty %d data %p", mat, mat->empty, data);
        return ERROR;
    }

    mat->data = data;
	mat->data_fp = (float *)mat->data;
	mat->data_i8 = (int8_t *)mat->data;
	mat->data_i32 = (int32_t *)mat->data;
	mat->data_i64 = (int64_t *)mat->data;
    return SUCCESS;
}

int set_matrix_name(matrix_t *mat, const char *name)
{
    if(NULL == mat || NULL == name)
    {
        LOG_ERROR("mat or name is NULL mat %p mat->empty %d name %p", mat, name);
        return ERROR;
    }
    strncpy(mat->shape.name, name, sizeof(mat->shape.name));
    return SUCCESS;
}


matrix_t *matrix_empty_shape(void *data, shape_t shape)
{
    matrix_t *out = (struct matrix_t *)malloc(sizeof(matrix_t));
    if(out)
    {
        memset(out, 0, sizeof(matrix_t));
        out->empty = 1;
        out->shape = shape;
        get_shape_size(&out->shape);
    }

    if(data)
    {
        set_matrix_data(out, data);
    }
    return out;
}

void matrix_free(matrix_t *mat)
{
    if(mat)
    {
        if(mat->empty == 0)
        {
            free(mat->data);
        }
        mat->data = NULL;
	    mat->data_fp = NULL;
	    mat->data_i8 = NULL;
	    mat->data_i32 = NULL;
		mat->data_i64 = NULL;
        free(mat);
    }
}

matrix_t *matrix_copy(matrix_t *src)
{
    matrix_t *out = matrix_alloc_shape(src->shape);
    if(out)
    {
        memcpy(out->data, src->data, src->shape.tensor_size);
    }
    return out;
}

int matrix_apply_copy(matrix_t *dst, int dst_pos, matrix_t *src, int src_pos, int len)
{
    if(len == -1)
    {
        len = src->shape.size - src_pos;
    }

    if(dst_pos + len > dst->shape.size)
    {
        LOG_DEBUG("matrix_apply_copy error dst_pos %d len %d dst size %d", dst_pos, len, dst->shape.size);
        return ERROR;
    }

    if(dst->shape.data_type != src->shape.data_type)
    {
        LOG_DEBUG("matrix_apply_copy error dst data_type %d src data_type  %d", dst->shape.data_type, src->shape.data_type);
        return ERROR;
    }

    switch(src->shape.data_type)
    {   
        case TENSOR_INT8:
    		memcpy(dst->data_i8 + dst_pos, src->data_i8 + src_pos, sizeof(int8_t) * len);
            break;
        case TENSOR_INT16:
    		memcpy(dst->data_i16 + dst_pos, src->data_i16 + src_pos, sizeof(int16_t) * len);
            break;
        case TENSOR_INT32:
    		memcpy(dst->data_i32 + dst_pos, src->data_i32 + src_pos, sizeof(int32_t) * len);
            break;
        case TENSOR_INT64:
    		memcpy(dst->data_i64 + dst_pos, src->data_i64 + src_pos, sizeof(int64_t) * len);
            break;
        default:
        case TENSOR_FLOAT32:
    		memcpy(dst->data_fp + dst_pos, src->data_fp + src_pos, sizeof(float) * len);
            break;
    } 
    return SUCCESS;
}

int matrix_alloc_data(matrix_t *mat, void *data)
{
    float *tmp_fp = NULL;
    if(NULL == mat || mat->empty)
    {
        return ERROR;
    }

    if(NULL != data)
    {
        tmp_fp = (float *)data;
    }
    else
    {
        tmp_fp = (float *)mat->data;
    }
        
    mat->data = malloc(mat->shape.tensor_size);
	mat->data_fp = (float *)mat->data;
	mat->data_i8 = (int8_t *)mat->data;
	mat->data_i32 = (int32_t *)mat->data;
	mat->data_i64 = (int32_t *)mat->data;

    switch(mat->shape.data_type)
    {   
        case TENSOR_INT8:
            break;
        case TENSOR_INT16:
            break;
        case TENSOR_INT32:
            break;
        default:
        case TENSOR_FLOAT32:
        {
            memcpy(mat->data_fp, tmp_fp, mat->shape.tensor_size); 
            break;
        }
    }
    mat->empty = 0;
    return SUCCESS; 
}


int matrix_insert(matrix_t *dst, int dst_start_pos, matrix_t *src, int src_start_pos, int src_end_pos)
{
    int dst_offset = 0;
    int src_offset = 0;
    int dst_size = 0;
    int src_size = 0;

    if(dst->shape.data_type != src->shape.data_type)
    {
        return ERROR;
    }

    if(-1 == src_end_pos)
    {
        src_end_pos = src->shape.dims[0];
    }

    if(-1 == dst_start_pos)
    {
        dst_start_pos = dst->shape.dims[0];
    }

    dst_offset = (dst->shape.size / dst->shape.dims[0]) * dst_start_pos;
    src_offset = (src->shape.size / src->shape.dims[0]) * src_start_pos;
    //LOG_DEBUG("dst_offset %d ",dst_offset);

    //dst_size = dst->shape.size - dst_offset;
    //LOG_DEBUG("src_end_pos - src_start_pos %d", src_end_pos - src_start_pos);
    src_size = (src->shape.size / src->shape.dims[0]) * (src_end_pos - src_start_pos);
    dst_size = dst->shape.size - dst_offset; //to do 

    //LOG_DEBUG("src_size %d", src_size);
    //(dst->empty) // to do
    dst->shape.dims[0] += src_size / (dst->shape.size / dst->shape.dims[0]);

    matrix_resize_shape(dst, dst->shape);

    //LOG_DEBUG("dst_size %d ", dst_size);
    
    //print_shape(dst->shape, NULL);
    //LOG_DEBUG("dst_offset %d dst_size %d ", dst_offset, dst_size);

    switch(dst->shape.data_type)
    {
        case TENSOR_INT8:
            memmove(dst->data_i8 + dst->shape.size - dst_size, dst->data_i8 + dst_offset, dst_size * sizeof(int8_t));
            memcpy(dst->data_i8 + dst_offset, src->data_i8 + src_offset, src_size * sizeof(int8_t));
            break;
        case TENSOR_INT16:
            memmove(dst->data_i16 + dst->shape.size - dst_size, dst->data_i16 + dst_offset, dst_size * sizeof(int16_t));
            memcpy(dst->data_i16 + dst_offset, src->data_i16 + src_offset, src_size * sizeof(int16_t));
            break;
        case TENSOR_INT32:
            memmove(dst->data_i32 + dst->shape.size - dst_size, dst->data_i32 + dst_offset, dst_size * sizeof(int32_t));
            memcpy(dst->data_i32 + dst_offset, src->data_i32 + src_offset, src_size * sizeof(int32_t));
            break;
        case TENSOR_INT64:
            memmove(dst->data_i64 + dst->shape.size - dst_size, dst->data_i64 + dst_offset, dst_size * sizeof(int64_t));
            memcpy(dst->data_i64 + dst_offset, src->data_i64 + src_offset, src_size * sizeof(int64_t));
            break;
        case TENSOR_FLOAT32:
        default:
            memmove(dst->data_fp + dst->shape.size - dst_size, dst->data_fp + dst_offset, dst_size * sizeof(float));
            memcpy(dst->data_fp + dst_offset, src->data_fp + src_offset, src_size * sizeof(float));
            break;
    }
    return SUCCESS;
}


int matrix_erase(matrix_t *mat, int start_index, int end_index)
{
    int start_pos = 0, end_pos = 0;
    int size = 0;
    if(-1 == end_index)
    {
        end_index = mat->shape.dims[0];
    }

    start_pos = (mat->shape.size / mat->shape.dims[0]) * (start_index);
    end_pos = (mat->shape.size / mat->shape.dims[0]) * (end_index);
    size = mat->shape.size - end_pos;

    if(mat->empty)
    { 
        matrix_alloc_data(mat, NULL); 
    }

    mat->shape.dims[0] -= end_index - start_index;

    switch(mat->shape.data_type)
    {
        case TENSOR_INT8:
            memmove(mat->data_i8 + start_pos, mat->data_i8 + end_pos , size * sizeof(int8_t));
            break;
        case TENSOR_INT16:
            memmove(mat->data_i16 + start_pos, mat->data_i16 + end_pos , size * sizeof(int16_t));
            break;
        case TENSOR_INT32:
            memmove(mat->data_i32 + start_pos, mat->data_i32 + end_pos , size * sizeof(int32_t));
            break;
        case TENSOR_INT64:
            memmove(mat->data_i64 + start_pos, mat->data_i64 + end_pos , size * sizeof(int64_t));
            break;
        case TENSOR_FLOAT32:
        default:
            memmove(mat->data_fp + start_pos, mat->data_fp + end_pos , size * sizeof(float));
            break;
    }
    //print_shape(mat->shape, NULL);
    matrix_resize_shape(mat, mat->shape); //to do 
    return SUCCESS;
}

int matrix_clear(matrix_t *mat)
{
    return matrix_erase(mat, 0, -1);
}


void matrix_subtract_mean(matrix_t *feature)
{
    int i, j;
    int feat_dim = feature->shape.dims[1]; //[x, 80]
    matrix_t *means = matrix_alloc(TENSOR_FLOAT32, 1, feat_dim);

    for(i = 0; i < feature->shape.dims[0]; i++)
    {   
        for(j = 0; j < feature->shape.dims[1]; j++)
        {   
            means->data_fp[j] += get_matrix_fp(feature, 2, i, j); 
        }   
    }   

    for(i = 0; i < feat_dim; i++)
    {   
        means->data_fp[i] /= (float)feature->shape.dims[0];
    }   

    for(i = 0; i < feature->shape.dims[0]; i++)
    {   
        for(j = 0; j < feature->shape.dims[1]; j++)
        {
            set_matrix_fp(feature, get_matrix_fp(feature, 2, i, j) - means->data_fp[j], 2, i, j); 
        }   
    }   
    matrix_free(means);
}

float matrix_log_energy(matrix_t *mat)
{
    float energy = 0.0f;
    float epsilon = FLT_EPSILON;
    float max_energy = 0.0f;
    int i = 0;

    //assert(mat->shape.num_dims == 1);   //signal frame

    for(i = 0; i < mat->shape.size; i++)
    {
        energy += mat->data_fp[i] * mat->data_fp[i];
    }
    max_energy = (energy > epsilon) ? energy : epsilon;
    return logf(max_energy);
}

matrix_t* matrix_concat(matrix_t *mat1, matrix_t *mat2, int dim)
{
    //to do 
#if 0
    int i;

    if(mat1->shape.num_dims != mat2->shape.num_dims ||
            mat1->shape.size != mat2->shape.size)
    {
        LOG_ERROR("mat1 or mat2 params is error");
        return NULL;
    }

    for(i = 0; i < mat1->shape.num_dims; i++)
    {

    }
#endif
}

//to do 
int matrix_apply_concat(matrix_t* out, matrix_t *mat1, matrix_t *mat2, int dim)
{
    int i, a, b;
    int h_offset = out->shape.size / 2;
    int v_offset = out->shape.dims[2] / 2;
    for(i = 0, a = 0, b = 0; i < out->shape.size; i++)
    {
        if(0 == dim)
        {
            if(i < h_offset)
            {
                out->data_fp[i] = mat1->data_fp[i];
            }
            else
            {
                out->data_fp[i] = mat2->data_fp[i - h_offset];
            }
        }
        else if(1 == dim)
        {
            if(i % out->shape.dims[2] < v_offset)   //to do 
            {
                out->data_fp[i] = mat1->data_fp[a++];
            }
            else
            {
                out->data_fp[i] = mat2->data_fp[b++];
            }
        }
    }
    return SUCCESS;
}

int matrix_apply_pow(matrix_t *mat, float value)
{
    int i;
    for(i = 0; i < mat->shape.size; i++)
    {
        mat->data_fp[i] = pow(mat->data_fp[i], value);
    }
    return SUCCESS;
}

int matrix_apply_floor(matrix_t *mat, float value)
{
    int i;
    for(i = 0; i < mat->shape.size; i++)
    {
        mat->data_fp[i] = mat->data_fp[i] < value ? value : mat->data_fp[i];
        //mat->data[i] = floor(mat->data[i]); //to do
    }
    return SUCCESS;
}

int matrix_apply_log(matrix_t *mat)
{
    int i;
    for(i = 0; i < mat->shape.size; i++)
    {
        mat->data_fp[i] = logf(mat->data_fp[i]);
    }
    return SUCCESS;
}

int matrix_zero(matrix_t *mat, int pos, int len)
{
    if(NULL == mat)
    {
        return ERROR;
    }
    switch(mat->shape.data_type)
    {    
        case TENSOR_INT8:
            memset(mat->data_i8 + pos, 0, sizeof(int8_t) * len);
            break;
        case TENSOR_INT16:
            memset(mat->data_i16 + pos, 0, sizeof(int16_t) * len);
            break;
        case TENSOR_INT32:
            memset(mat->data_i32 + pos, 0, sizeof(int32_t) * len);
            break;
        case TENSOR_INT64:
            memset(mat->data_i64 + pos, 0, sizeof(int64_t) * len);
            break;
        case TENSOR_FLOAT32:
        default:
            memset(mat->data_fp + pos, 0, sizeof(float) * len);
            break;
    }   
    return SUCCESS;
}

int matrix_resize_shape(matrix_t *mat, shape_t shape)
{
    if(mat && 0 == mat->empty)
    {
        mat->shape = shape;
        get_shape_size(&mat->shape);
        if(mat->shape.tensor_size > 0)
            mat->data = realloc(mat->data, mat->shape.tensor_size);
        mat->data_fp = (float *)mat->data;
        mat->data_i8 = (int8_t *)mat->data;
        mat->data_i16 = (int16_t *)mat->data;
        mat->data_i32 = (int32_t *)mat->data;
        mat->data_i64 = (int64_t *)mat->data;
        return SUCCESS;
    }
    return ERROR;
}

int matrix_resize(matrix_t *mat, int data_type, int num_dims, ...)
{
    int i;
    va_list args;
    shape_t shape;

    shape.num_dims = num_dims;
    shape.data_type = data_type;
    va_start(args, num_dims);
    for(i = 0; i < num_dims; i++)
    {
        shape.dims[i] = va_arg(args, int);
    }
    va_end(args);
    matrix_resize_shape(mat, shape);
    return SUCCESS;
}

int matrix_apply_mul_element(matrix_t *mat1, matrix_t *mat2)
{
    int i;
    if(mat1 && mat2)
    {
        if(mat1->shape.size != mat2->shape.size)
        {
            LOG_DEBUG("mat size %d mat size %d no eq", mat1->shape.size, mat2->shape.size);
            return ERROR;
        }
        for(i = 0; i < mat1->shape.size; i++)
        {
            mat1->data_fp[i] = mat1->data_fp[i] * mat2->data_fp[i];;
        }
        return SUCCESS;
    }
    return ERROR;
}

float matrix_energy(matrix_t *mat1, matrix_t *mat2)
{
    int i;
    float value = 0.0f;
    if(mat1->shape.size != mat2->shape.size)
        return 0.0f;

    for(i = 0; i < mat1->shape.size; i++)
    {
        value += mat1->data_fp[i] * mat2->data_fp[i];
    }
    return value;
}

int matrix_apply_add(matrix_t *dst, matrix_t *src,  float scalar)
{
    int i;

    if(dst && src)
    {
        if(src->shape.size != dst->shape.size)
        {
            LOG_DEBUG("matrix shape error");
            return ERROR;
        }

        for(i = 0; i < src->shape.size; i++)
        {
            dst->data_fp[i] += (src->data_fp[i] * scalar);
        }
        return SUCCESS;
    }
    return ERROR;
}

int matrix_mul_scalar(matrix_t *mat, float scalar)
{
    int i;
    for(i = 0; i < mat->shape.size; i++)
    {
        mat->data_fp[i] = (mat->data_fp[i] * scalar);
    }
    return SUCCESS;
}

matrix_t* matrix_add(matrix_t *mat1, matrix_t *mat2,  float scalar)
{
    int i;
    if(mat1 && mat2)
    {
        if(mat1->shape.size != mat2->shape.size)
        {
            LOG_DEBUG("matrix shape mat1 %d mat2 %derror", mat1->shape.size, mat2->shape.size);
            return NULL;
        }

        matrix_t *out = matrix_alloc_shape(mat1->shape);
        for(i = 0; i < mat1->shape.size; i++)
        {
            out->data_fp[i] =  mat1->data_fp[i] + mat2->data_fp[i];
        }
        return out;
    }
    return NULL;
}

matrix_t *matrix_mul(const matrix_t *mat1, const matrix_t *mat2)
{

    return NULL;
}

int matrix_apply_mul(const matrix_t *dst, const matrix_t *src)
{
    return SUCCESS;
}

int matrix_power_spectrum(matrix_t *mat)
{
    int32_t i;
    int32_t dim = mat->shape.size;
    int32_t half_dim = dim / 2;

    float first_energy = mat->data_fp[0] * mat->data_fp[0];
    float last_energy = mat->data_fp[1] * mat->data_fp[1];
    float real = 0.0f, im = 0.0f;

    /*  
     * now we have in waveform, first half of complex spectrum
     * it's stored as [real0, realN/2, real1, im1, real2, im2, ...]
     */
    for(i = 1; i < half_dim; i++)
    {   
        real = mat->data_fp[i * 2]; 
        im = mat->data_fp[i * 2 + 1]; 

        mat->data_fp[i] = real * real + im * im; 
    }   
    mat->data_fp[0] = first_energy;
    mat->data_fp[half_dim] = last_energy;
    return SUCCESS;
}

int matrix_apply_add_value(matrix_t *mat, float value)
{
    int i;
    if(mat)
    {
        for(i = 0; i < mat->shape.size; i++)
        {   
            mat->data_fp[i] += value;
        } 
        return SUCCESS;
    }
    return ERROR;
}

float matrix_sum(matrix_t *mat)
{
    int i;
    float value = 0.0f;
    if(mat)
    {
        for(i = 0; i < mat->shape.size; i++)
        {   
            value += mat->data_fp[i];
        } 
    }
    return value;
}

float matrix_mean(matrix_t *mat)
{
    return matrix_sum(mat)/mat->shape.size;
}

int matrix_apply_cmvn(matrix_t *mat, float *means, float *vars, int size)
{
    int i, j;
    if(NULL == mat || mat->shape.num_dims != 2 || size != mat->shape.dims[1])
    {
        LOG_ERROR("mat is error");
        return ERROR;
    }
    for(i = 0; i < mat->shape.dims[0]; i++)
    {
        for(j = 0; j < mat->shape.dims[1]; j++)
        {
            set_matrix_fp(mat, (get_matrix_fp(mat, 2, i, j) + means[j]) * vars[j], 2, i, j);
        }
    }
    return SUCCESS;
}

matrix_t* matrix_frame_db(float *data, int size, int frame_sample_length, int frame_shift_length)
{
    int i,j = 0, offset;
    float sum = 0.0;

    matrix_t *decibel = matrix_alloc(TENSOR_FLOAT32, 1, (size - frame_sample_length + 1) / frame_shift_length + 1); 
    for(offset = 0; offset + frame_sample_length - 1 < size; offset += frame_shift_length)
    {
        sum = 0.0;
        for(i = 0; i < frame_sample_length; i++)
        {
            sum += data[offset + i] * data[offset + i];
        }
        decibel->data_fp[j++] = 10 * log10(sum + 0.000001);
    }
    return decibel;
}

int matrix_unsqueeze(matrix_t *mat, int index)
{
    int i;
    shape_t shape = mat->shape;

    if(index > mat->shape.num_dims)
    {
        LOG_ERROR("unsqueeze error num_dims %d index %d ", mat->shape.num_dims, index);
        return ERROR;
    }

    for(i = 0; i < shape.num_dims; i++)
    {   
        if(i < index)
        {
            mat->shape.dims[i] = shape.dims[i];
        }
        else
        {
            mat->shape.dims[i+1] = shape.dims[i];
        }
    }   
    mat->shape.dims[index] = 1;
    mat->shape.num_dims++;    
    get_shape_size(&mat->shape);
    return SUCCESS;
}

int matrix_squeeze(matrix_t *mat, int index)
{
    int i;
    shape_t shape = mat->shape;
    if(mat->shape.dims[index] != 1)
    {
        LOG_ERROR("squeeze index %d value %d != 1", index, mat->shape.dims[index]);
        return ERROR;
    }

    for(i = 0; i < shape.num_dims; i++)
    {   
        if(i < index)
        {
            mat->shape.dims[i] = shape.dims[i];
        }
        else if(i > index)
        {
            mat->shape.dims[i - 1] = shape.dims[i];
        }
    }   
    mat->shape.num_dims--; 
    get_shape_size(&mat->shape);
    return SUCCESS;
}

//F.pad(input, (pad_left, pad_right), mode="reflect")
int matrix_pad(matrix_t *mat, int pad_left, int pad_right, int mode)
{
#if 0
    int size = mat->shape.size;
    if(mat->num_dims != 2)
    {
        LOG_ERROR("matrix_pad dim %d not support ", mat->num_dims);
        return ERROR;
    }
    mat->shape.dims[1] += pad_left + pad_right;
    get_shape_size(&mat->shape):
    matrix_resize_shape(mat, mat->shape);

    switch(mat->shape.data_type)
    {
        case TENSOR_INT8:
            memset(mat->data_i8, 0, sizeof(int8_t) * pad_left);
            memmove(mat->data_i8, mat->data_i8 + pad_left , size * sizeof(int8_t));
            memset(mat->data_i8 + size + pad_left, 0, sizeof(int8_t) * pad_right);
            break;
        case TENSOR_INT16:
            memset(mat->data_i16, 0, sizeof(int16_t) * pad_left);
            memmove(mat->data_i16, mat->data_i16 + pad_left , size * sizeof(int16_t));
            memset(mat->data_i16 + size + pad_left, 0, sizeof(int16_t) * pad_right);
            break;
        case TENSOR_INT32:
            memset(mat->data_i32, 0, sizeof(int32_t) * pad_left);
            memmove(mat->data_i32, mat->data_i32 + pad_left , size * sizeof(int32_t));
            memset(mat->data_i32 + size + pad_left, 0, sizeof(int32_t) * pad_right);
            break;
        case TENSOR_INT64:
            memset(mat->data_i64, 0, sizeof(int64_t) * pad_left);
            memmove(mat->data_i64, mat->data_i64 + pad_left , size * sizeof(int64_t));
            memset(mat->data_i64 + size + pad_left, 0, sizeof(int64_t) * pad_right);
            break;
        case TENSOR_FLOAT32:
        default:
            memset(mat->data_fp, 0, sizeof(float) * pad_left);
            memmove(mat->data_fp, mat->data_fp + pad_left , size * sizeof(float));
            memset(mat->data_fp + size + pad_left, 0, sizeof(float) * pad_right);
            break;
    }
#endif
    return SUCCESS;
}

// 3维以上的转置暂时不支持
matrix_t* matrix_transpose(matrix_t *mat)
{
    int i, j;
    float value = 0.0f;
    matrix_t *output = NULL;
    int dim0 = -1, dim1 = -1;
   
    if(mat->shape.num_dims >= 2 )
    {
        for(i = 0; i < mat->shape.num_dims; i++)
        {
            if(mat->shape.dims[i] != 1)
            {
                if(dim0 == -1)
                {
                    dim0 = i;
                } 
                else if(dim1 == -1)
                {
                    dim1 = i;
                }
                else
                {
                    print_shape(mat->shape);
                    LOG_DEBUG("matrix transponse not support format");
                    return NULL;
                }

            }
        }
    }
    else
    {
        print_shape(mat->shape);
        LOG_DEBUG("matrix transponse not support format");
    }

    output = matrix_copy(mat);
    output->shape.dims[dim0] = mat->shape.dims[dim1];
    output->shape.dims[dim1] = mat->shape.dims[dim0];

    int rows = output->shape.dims[dim0];
    int cols = output->shape.dims[dim1];

    //to do 不懂为什么需要交互顺序才对
    for(i = 0; i < rows; i++)
    {
        for(j = 0; j < cols; j++)
        {
            switch(mat->shape.data_type)
            {
                case TENSOR_INT8:
                    output->data_i8[i * cols + j] = mat->data_i8[j * rows + i];
                break;
                case TENSOR_INT16:
                    output->data_i16[i * cols + j] = mat->data_i16[j * rows + i];
                    break;
                case TENSOR_INT32:
                    output->data_i32[i * cols + j] = mat->data_i32[j * rows + i];
                    break;
                case TENSOR_INT64:
                    output->data_i64[i * cols + j] = mat->data_i64[j * rows + i];
                    break;
                case TENSOR_FLOAT32:
                default:
                    output->data_fp[i * cols + j] = mat->data_fp[j * rows + i];
                    //output->data_fp[j * rows + i] = mat->data_fp[i * cols + j]; // 我觉得这样才是正常的，但是运行会异常
                    break;
            }

        }
    }
    return output;
}



#if 0 //delete code
//index 0 begin not 1  to do 2dim
float get_matrix_fp(matrix_t *mat, int num_dims, ...)
{
    int i, j, index;
    va_list args;
    float value = 0.0f;
    int offset = 0;
    shape_t shape;

    shape.num_dims = num_dims;
    va_start(args, num_dims);
    for(i = 0; i < num_dims; i++)
    {
        shape.dims[i] = va_arg(args, int);
        if(shape.dims[i] > mat->shape.dims[i])
        {
            va_end(args);
            return ERROR;
        }
    }
    va_end(args);

    for(i = 0; i < num_dims; i++)
    {
        index = 1;
        for(j = i + 1; j < num_dims; j++)
        {
            index *= mat->shape.dims[j];
        }
        if(index == 0)
            offset += shape.dims[i];
        else
            offset += shape.dims[i] * index;
    }
    value = mat->data_fp[offset];
    return value;
}


//to do 2dim

int set_matrix_fp(matrix_t *mat, float value, int num_dims, ...)
{
    int i, j, index;
    va_list args;
    int offset = 0;
    shape_t shape;

    shape.num_dims = num_dims;
    va_start(args, num_dims);
    for(i = 0; i < num_dims; i++)
    {
        shape.dims[i] = va_arg(args, int);
        if(shape.dims[i] > mat->shape.dims[i])
        {
            va_end(args);
            return ERROR;
        }
    }
    va_end(args);

    for(i = 0; i < num_dims; i++)
    {
        index = 1;
        for(j = i + 1; j < num_dims; j++)
        {
            index *= mat->shape.dims[j];
        }
        if(index == 0)
            offset += shape.dims[i];
        else
            offset += shape.dims[i] * index;
    }
    mat->data_fp[offset] = value;
    return SUCCESS;
}
#endif


