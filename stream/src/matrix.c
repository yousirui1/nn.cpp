#include "base.h"
#include "matrix.h"

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
    if(0)
    {
        printf(" TENSOR FORMA: %s ", get_matrix_format_string(shape.format));
        printf(" rknn zp %d scale %.2f \n", shape.zp, shape.scale);
    }
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
            case TENSOR_UINT8:
                printf("%u ", mat->data_i8[i]);
                break;
            case TENSOR_FLOAT32:
                //printf("%.15f ", mat->data_fp[i]);
                printf("%e ", mat->data_fp[i]);
                break;
            case TENSOR_INT32:
                printf("%d ", mat->data_i32[i]);
                break;
            case TENSOR_INT64:
                printf("%lld ", mat->data_i64[i]);
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

void print_matrix(matrix_t *mat)
{
    print_matrix_format(mat, 10);
}

float _get_matrix_fp_fast(matrix_t *mat, const int *indices) 
{
    int i,offset = 0;

    //to do 
    for (i = 0; i < mat->shape.num_dims; i++) 
    {
        offset += indices[i] * mat->shape.stride[i];
    }
    return mat->data_fp[offset];
}

int _set_matrix_fp_fast(matrix_t *mat, float value, const int *indices) 
{
    int i, offset = 0;
    //to do 
    for (i = 0; i < mat->shape.num_dims; i++) 
    {
        offset += indices[i] * mat->shape.stride[i];
    }
    mat->data_fp[offset] = value;
    return SUCCESS;
}


int _set_shape_fast(shape_t *shape, int data_type, int n_dims, const int indices[])
{
    int i;
    shape->num_dims = n_dims;
    shape->data_type = data_type;

    for(i = 0; i < shape->num_dims; i++)
    {
        shape->dims[i] = indices[i];
    }
    get_shape_size(shape);
    return SUCCESS;
}

int set_shape(shape_t *shape, int data_type, int num_dims, ...)
{
    int i;
    va_list args;
    shape->num_dims = num_dims;
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

    // 列优先，todo 行优先
    shape->stride[shape->num_dims - 1] = 1;
    for (i = shape->num_dims - 2; i >= 0; i--) 
    {
        shape->stride[i] = shape->stride[i+1] * shape->dims[i+1];
    }

    switch(shape->data_type)
    {
        case TENSOR_UINT8:
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




matrix_t *matrix_alloc_shape(shape_t shape)
{
    matrix_t *out = (struct matrix_t *)malloc(sizeof(matrix_t));
    if(out)
    {
        out->empty = 0;
        out->shape = shape;

        get_shape_size(&out->shape);
        out->data = malloc(out->shape.tensor_size);
        memset(out->data, 0, out->shape.tensor_size);
        out->data_fp = (float *)out->data;
        out->data_i8 = (int8_t *)out->data;
        out->data_i32 = (int32_t *)out->data;
        out->data_i64 = (int64_t *)out->data;
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
        mat->data_i32 = (int32_t *)mat->data;
        mat->data_i64 = (int64_t *)mat->data;
        return SUCCESS;
    }
    return ERROR;
}

matrix_t *_matrix_alloc_fast(int data_type, int num_dims, const int indices[])
{
    int i;
    va_list args;
    matrix_t *out = (struct matrix_t *)malloc(sizeof(matrix_t));
    if(out)
    {
        out->empty = 0;
        out->shape.num_dims = num_dims;
        out->shape.data_type = data_type;
        for(i = 0; i < num_dims; i++)
        {
            out->shape.dims[i] = indices[i];
        }
        get_shape_size(&out->shape);

        out->data = malloc(out->shape.tensor_size);
        out->data_fp = (float *)out->data;
        out->data_i8 = (int8_t *)out->data;
        out->data_i32 = (int32_t *)out->data;
        out->data_i64 = (int64_t *)out->data;
        memset(out->data, 0, out->shape.tensor_size);
    }
    return out;
}


#if 0
int main(int argc, char *argv[])
{
    //shape_t shape;
    //set_shape(&shape, TENSOR_FLOAT32, 3, 1, 2, 4);    

    //print_shape(shape);

    matrix_t *mat = matrix_alloc(TENSOR_FLOAT32, 2, 2, 3);

    for(int i = 0; i < mat->shape.size; i++)
    {
        mat->data_fp[i] = i; 
    }
    print_matrix(mat);

    float val = get_matrix_fp(mat, 2, 2, 1);
    LOG_DEBUG("val %.2f", val);


}
#endif
