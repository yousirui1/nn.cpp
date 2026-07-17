/*
 * gguf_dump_keys.c — 遍历 GGUF 文件所有 metadata key + 张量名
 *
 * 编译:  gcc -O2 -Wall -o gguf_dump_keys gguf_dump_keys.c
 * 用法:  ./gguf_dump_keys model.gguf
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>

#define GGUF_MAGIC 0x46554747 /* "GGUF" little-endian */

/* ---- GGUF value types (与 ggml 定义一致) ---- */
enum gguf_type {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
    GGUF_TYPE_COUNT
};

static const char *type_names[] = {
    "uint8", "int8", "uint16", "int16",
    "uint32", "int32", "float32", "bool",
    "string", "array",
    "uint64", "int64", "float64"
};

/* ---- tensor quantization type names ---- */
static const char *ggml_type_name(int t) {
    switch (t) {
        case 0:  return "F32";
        case 1:  return "F16";
        case 2:  return "Q4_0";
        case 3:  return "Q4_1";
        case 6:  return "Q5_0";
        case 7:  return "Q5_1";
        case 8:  return "Q8_0";
        case 9:  return "Q8_1";
        case 10: return "Q2_K";
        case 11: return "Q3_K_S";
        case 12: return "Q3_K_M";
        case 13: return "Q3_K_L";
        case 14: return "Q4_K_S";
        case 15: return "Q4_K_M";
        case 16: return "Q5_K_S";
        case 17: return "Q5_K_M";
        case 18: return "Q6_K";
        case 24: return "IQ2_XXS";
        case 25: return "IQ2_XS";
        case 26: return "IQ3_XXS";
        case 27: return "IQ1_S";
        case 28: return "IQ4_NL";
        case 29: return "IQ3_S";
        case 30: return "IQ2_S";
        case 31: return "IQ4_XS";
        case 32: return "IQ1_M";
        case 33: return "BF16";
        default: return "UNKNOWN";
    }
}

/* ---- 低层读取辅助 ---- */
static int r_bytes(FILE *f, void *buf, size_t n) {
    return fread(buf, 1, n, f) == n;
}

static int r_u32(FILE *f, uint32_t *v) { return r_bytes(f, v, 4); }
static int r_u64(FILE *f, uint64_t *v) { return r_bytes(f, v, 8); }

static int r_str(FILE *f, char **out) {
    uint64_t len;
    if (!r_u64(f, &len)) return 0;
    char *s = malloc((size_t)len + 1);
    if (!s) return 0;
    if (len > 0 && !r_bytes(f, s, (size_t)len)) { free(s); return 0; }
    s[len] = '\0';
    *out = s;
    return 1;
}

/* ---- 跳过一个值（用于错误恢复等场景） ---- */
static int skip_value(FILE *f, int type);

static size_t type_size(int type) {
    switch (type) {
        case GGUF_TYPE_UINT8:  case GGUF_TYPE_INT8:  case GGUF_TYPE_BOOL: return 1;
        case GGUF_TYPE_UINT16: case GGUF_TYPE_INT16:                      return 2;
        case GGUF_TYPE_UINT32: case GGUF_TYPE_INT32: case GGUF_TYPE_FLOAT32: return 4;
        case GGUF_TYPE_UINT64: case GGUF_TYPE_INT64: case GGUF_TYPE_FLOAT64: return 8;
        default: return 0; /* string / array 需特殊处理 */
    }
}

static int skip_value(FILE *f, int type) {
    if (type == GGUF_TYPE_STRING) {
        char *s;
        int ok = r_str(f, &s);
        free(s);
        return ok;
    }
    if (type == GGUF_TYPE_ARRAY) {
        uint32_t etype; uint64_t n;
        if (!r_u32(f, &etype) || !r_u64(f, &n)) return 0;
        for (uint64_t i = 0; i < n; i++)
            if (!skip_value(f, etype)) return 0;
        return 1;
    }
    size_t sz = type_size(type);
    if (sz == 0) return 0;
    char buf[8];
    return r_bytes(f, buf, sz);
}

/* ---- 读取并打印一个值 ---- */
static int print_value(FILE *f, int type) {
    switch (type) {
        case GGUF_TYPE_UINT8:  { uint8_t  v; if (!r_bytes(f,&v,1)) return 0; printf("%" PRIu8,  v); return 1; }
        case GGUF_TYPE_INT8:   { int8_t   v; if (!r_bytes(f,&v,1)) return 0; printf("%" PRId8,  v); return 1; }
        case GGUF_TYPE_UINT16: { uint16_t v; if (!r_bytes(f,&v,2)) return 0; printf("%" PRIu16, v); return 1; }
        case GGUF_TYPE_INT16:  { int16_t  v; if (!r_bytes(f,&v,2)) return 0; printf("%" PRId16, v); return 1; }
        case GGUF_TYPE_UINT32: { uint32_t v; if (!r_bytes(f,&v,4)) return 0; printf("%" PRIu32, v); return 1; }
        case GGUF_TYPE_INT32:  { int32_t  v; if (!r_bytes(f,&v,4)) return 0; printf("%" PRId32, v); return 1; }
        case GGUF_TYPE_FLOAT32:{ float     v; if (!r_bytes(f,&v,4)) return 0; printf("%g",       v); return 1; }
        case GGUF_TYPE_BOOL:   { uint8_t  v; if (!r_bytes(f,&v,1)) return 0; printf("%s", v?"true":"false"); return 1; }
        case GGUF_TYPE_UINT64: { uint64_t v; if (!r_bytes(f,&v,8)) return 0; printf("%" PRIu64, v); return 1; }
        case GGUF_TYPE_INT64:  { int64_t  v; if (!r_bytes(f,&v,8)) return 0; printf("%" PRId64, v); return 1; }
        case GGUF_TYPE_FLOAT64:{ double    v; if (!r_bytes(f,&v,8)) return 0; printf("%g",       v); return 1; }
        case GGUF_TYPE_STRING: {
            char *s;
            if (!r_str(f, &s)) return 0;
            /* 截断过长字符串方便显示 */
            if (strlen(s) > 200) {
                printf("%.200s...\" (len=%zu)", s, strlen(s));
            } else {
                printf("\"%s\"", s);
            }
            free(s);
            return 1;
        }
        case GGUF_TYPE_ARRAY: {
            uint32_t etype;
            uint64_t n;
            if (!r_u32(f, &etype) || !r_u64(f, &n)) return 0;
            const char *tname = (etype < GGUF_TYPE_COUNT) ? type_names[etype] : "?";
            printf("array<%s>[%" PRIu64 "] = {", tname, n);
            /* 数组过大时只打印前 8 个元素 */
            uint64_t limit = (n > 8) ? 8 : n;
            for (uint64_t i = 0; i < limit; i++) {
                if (i > 0) printf(", ");
                if (!print_value(f, etype)) return 0;
            }
            if (n > 8) {
                printf(", ... (total %" PRIu64 ")", n);
                /* 跳过剩余元素 */
                for (uint64_t i = 8; i < n; i++)
                    if (!skip_value(f, etype)) return 0;
            }
            printf("}");
            return 1;
        }
    }
    fprintf(stderr, "  [unknown type %d]\n", type);
    return 0;
}

/* ================================================================== */
#ifdef GGUF_DUMP

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "用法: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror("fopen"); return 1; }

    /* ---- 读取头部 ---- */
    uint32_t magic, version;
    uint64_t n_tensors, n_kv;

    if (!r_u32(f, &magic) || magic != GGUF_MAGIC) {
        fprintf(stderr, "错误: 不是有效的 GGUF 文件 (magic=0x%08x)\n", magic);
        fclose(f);
        return 1;
    }
    if (!r_u32(f, &version))    goto eof_err;
    if (!r_u64(f, &n_tensors))  goto eof_err;
    if (!r_u64(f, &n_kv))       goto eof_err;

    printf("╔══════════════════════════════════════╗\n");
    printf("║         GGUF 文件信息               ║\n");
    printf("╠══════════════════════════════════════╣\n");
    printf("║ 文件     : %s\n", argv[1]);
    printf("║ 版本     : %" PRIu32 "\n", version);
    printf("║ 张量数   : %" PRIu64 "\n", n_tensors);
    printf("║ 元数据数 : %" PRIu64 "\n", n_kv);
    printf("╚══════════════════════════════════════╝\n\n");

    /* ---- 遍历所有 metadata KV ---- */
    printf("─────────────────────────────────────────────────\n");
    printf("  Metadata Key-Value 对 (共 %" PRIu64 " 个)\n", n_kv);
    printf("─────────────────────────────────────────────────\n");

    for (uint64_t i = 0; i < n_kv; i++) {
        char *key = NULL;
        uint32_t vtype;

        if (!r_str(f, &key))   goto eof_err;
        if (!r_u32(f, &vtype)) { free(key); goto eof_err; }

        printf("\n  [%" PRIu64 "] key = \"%s\"\n", i, key);
        printf("      type  = %s\n",
               vtype < GGUF_TYPE_COUNT ? type_names[vtype] : "???");
        printf("      value = ");

        if (!print_value(f, vtype)) {
            fprintf(stderr, "\n  ** 读取 key \"%s\" 的值时失败 **\n", key);
            free(key);
            goto eof_err;
        }
        printf("\n");
        free(key);
    }

    /* ---- 遍历所有张量信息 ---- */
    printf("\n─────────────────────────────────────────────────\n");
    printf("  张量列表 (共 %" PRIu64 " 个)\n", n_tensors);
    printf("─────────────────────────────────────────────────\n\n");

    for (uint64_t i = 0; i < n_tensors; i++) {
        char *name = NULL;
        uint32_t n_dims, dtype;
        uint64_t offset;

        if (!r_str(f, &name))       goto eof_err;
        if (!r_u32(f, &n_dims))     { free(name); goto eof_err; }

        if (n_dims > 4) {
            fprintf(stderr, "  ** 张量 \"%s\" 维度数 %u 超出预期 **\n", name, n_dims);
            free(name);
            goto eof_err;
        }

        uint64_t dims[4] = {0};
        for (uint32_t d = 0; d < n_dims; d++)
            if (!r_u64(f, &dims[d])) { free(name); goto eof_err; }

        if (!r_u32(f, &dtype))      { free(name); goto eof_err; }
        if (!r_u64(f, &offset))     { free(name); goto eof_err; }

        printf("  [%" PRIu64 "] %s\n", i, name);
        printf("      type   = %s\n", ggml_type_name(dtype));
        printf("      shape  = [");
        for (uint32_t d = 0; d < n_dims; d++) {
            if (d > 0) printf(", ");
            printf("%" PRIu64, dims[d]);
        }
        printf("]\n");
        printf("      offset = %" PRIu64 "\n\n", offset);
        free(name);
    }

    printf("══════════════════════════════════════\n");
    printf("遍历完成。metadata: %" PRIu64 ", tensors: %" PRIu64 "\n",
           n_kv, n_tensors);
    fclose(f);
    return 0;

eof_err:
    fprintf(stderr, "\n错误: 意外的文件结尾或读取失败\n");
    fclose(f);
    return 1;
}
#endif
