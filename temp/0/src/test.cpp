/*
 * example_sched_cache.cpp
 *
 * 编译: g++ -O0 -g -o test example_sched_cache.cpp -lggml -lggml-cuda -lcudart -lm
 * 检测: valgrind --leak-check=full --show-leak-kinds=definite --suppressions=ggml.supp ./test
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>

/* ================================================================
 *  参数
 * ================================================================ */
#define IN_DIM         4
#define OUT_DIM        4
#define CACHE_DIM      4    // 缓存维度（可以和 OUT_DIM 不同）
#define GRAPH_SIZE     128
#define BUILD_BUF_SIZE (256 * 1024)

/* ================================================================
 *  ggml 后端句柄
 * ================================================================ */
typedef struct {
    ggml_backend_t backend;       // 主后端（GPU 或 CPU）
    ggml_backend_t cpu_backend;   // CPU 始终存在
} ggml_handle_t;

/* ================================================================
 *  模型权重（全局唯一，只读）
 *
 *  生命周期: 加载 → 多实例共享 → 最后释放
 * ================================================================ */
typedef struct {
    ggml_context         *ctx;
    ggml_backend_buffer_t buf;       // 承载权重数据
    ggml_tensor          *W;         // [IN_DIM x OUT_DIM]
    ggml_tensor          *bias;      // [OUT_DIM]
} model_t;

/* ================================================================
 *  推理状态（每个实例独立）
 *
 *  包含三类资源:
 *    1. cache  — 持久化中间数据，跨推理调用存活
 *    2. sched  — 管理图执行，自动分配 work buffer
 *    3. build  — 构建计算图的临时上下文
 * ================================================================ */
typedef struct {
    int               id;
    const model_t    *model;

    /* ---- [1] Cache tensors（核心：持久化中间数据） ---- */
    ggml_context         *ctx_cache;
    ggml_backend_buffer_t buf_cache;     // 独立 buffer，不在 sched 管理范围内
    ggml_tensor          *cache_h;       // 缓存 hidden state  [CACHE_DIM]
    ggml_tensor          *cache_count;   // 缓存调用计数      [1]

    /* ---- [2] Scheduler ---- */
    ggml_backend_sched_t sched;

    /* ---- [3] Build context（临时，每次推理重建） ---- */
    ggml_context *ctx_build;
    void         *build_buf;
} state_t;

/* ================================================================
 *  第一步：初始化后端
 * ================================================================ */
static void handle_init(ggml_handle_t *h) {
    memset(h, 0, sizeof(*h));
    h->cpu_backend = ggml_backend_cpu_init();
    h->backend = ggml_backend_cuda_init(0);
    if (!h->backend) {
        printf("  [info] CUDA 不可用，退化为纯 CPU\n");
        h->backend = h->cpu_backend;
    } else {
        printf("  [info] CUDA 可用\n");
    }
}

static void handle_free(ggml_handle_t *h) {
    if (h->backend && h->backend != h->cpu_backend)
        ggml_backend_free(h->backend);
    if (h->cpu_backend)
        ggml_backend_free(h->cpu_backend);
}

/* ================================================================
 *  第二步：加载模型权重（一次）
 * ================================================================ */
static int model_load(model_t *m, ggml_handle_t *h) {
    memset(m, 0, sizeof(*m));

    struct ggml_init_params p = {
        .mem_size = 2 * ggml_tensor_overhead() + 256,
        .no_alloc = true,
    };
    m->ctx = ggml_init(p);
    if (!m->ctx) return -1;

    m->W = ggml_new_tensor_2d(m->ctx, GGML_TYPE_F32, IN_DIM, OUT_DIM);
    ggml_set_name(m->W, "model.W");

    m->bias = ggml_new_tensor_1d(m->ctx, GGML_TYPE_F32, OUT_DIM);
    ggml_set_name(m->bias, "model.bias");

    /* 权重放在 cpu_backend，所有实例共享这个 buf */
    m->buf = ggml_backend_alloc_ctx_tensors(m->ctx, h->cpu_backend);
    if (!m->buf) return -1;

    /* 填充测试数据 */
    float W_data[IN_DIM * OUT_DIM];
    for (int i = 0; i < IN_DIM * OUT_DIM; i++)
        W_data[i] = 0.1f * (i % 7 + 1) - 0.4f;
    ggml_backend_tensor_set(m->W, W_data, 0, sizeof(W_data));

    float bias_data[OUT_DIM] = {0.01f, -0.01f, 0.02f, -0.02f};
    ggml_backend_tensor_set(m->bias, bias_data, 0, sizeof(bias_data));

    printf("  权重: %zu bytes\n", (size_t)ggml_backend_buffer_get_size(m->buf));
    return 0;
}

static void model_free(model_t *m) {
    if (m->buf) ggml_backend_buffer_free(m->buf);
    if (m->ctx) ggml_free(m->ctx);
    memset(m, 0, sizeof(*m));
}

/* ================================================================
 *  第三步：构建计算图
 *
 *  图结构:
 *    h     = W @ input + cache_h     ← cache_h 来自 ctx_cache（持久化）
 *    h     = relu(h)
 *    out   = h + bias
 *
 *  权重 tensor（W, bias）和 cache tensor（cache_h）都是跨 ctx 引用，
 *  它们的 buf 是预分配的，sched 不会重复分配。
 * ================================================================ */
static ggml_cgraph *build_graph(state_t *st) {
    const model_t *m = st->model;

    /* 重建 build context（使用外部 buffer，不泄漏） */
    if (st->ctx_build) ggml_free(st->ctx_build);

    struct ggml_init_params bp = {
        .mem_size   = BUILD_BUF_SIZE,
        .mem_buffer = st->build_buf,      // 外部内存
        .no_alloc   = true,
    };
    st->ctx_build = ggml_init(bp);
    if (!st->ctx_build) return NULL;

    /* ---- 输入 tensor（临时，每次推理不同数据） ---- */
    ggml_tensor *input = ggml_new_tensor_1d(st->ctx_build, GGML_TYPE_F32, IN_DIM);
    ggml_set_name(input, "input");

    /* ---- 计算: h = W^T @ input ---- */
    /* W 来自 model->ctx（共享权重），数据在 model->buf 中 */
    ggml_tensor *h = ggml_mul_mat(st->ctx_build, m->W, input);

    /* ---- 加上缓存: h = h + cache_h ---- */
    /* cache_h 来自 ctx_cache（实例独立），数据在 buf_cache 中 */
    h = ggml_add(st->ctx_build, h, st->cache_h);

    /* ---- ReLU ---- */
    h = ggml_relu(st->ctx_build, h);

    /* ---- 加 bias ---- */
    ggml_tensor *out = ggml_add(st->ctx_build, h, m->bias);
    ggml_set_name(out, "output");

    /* ---- 构建前向图 ---- */
    ggml_cgraph *gf = ggml_new_graph(st->ctx_build);
    ggml_build_forward_expand(gf, out);

    return gf;
}

/* ================================================================
 *  第四步：创建推理实例
 *
 *  关键：cache tensor 的分配方式与权重完全相同
 *    1. ggml_init (ctx_cache)       → 定义 tensor
 *    2. ggml_new_tensor_1d          → 创建 cache tensor
 *    3. ggml_backend_alloc_ctx_tensors → 分配 buffer（数据持久化）
 *    4. ggml_backend_tensor_set     → 初始化数据
 *
 *  与权重的区别：数据在推理过程中会被更新
 * ================================================================ */
static int state_create(state_t *st, const model_t *m,
                        const ggml_handle_t *h, int id)
{
    memset(st, 0, sizeof(*st));
    st->id    = id;
    st->model = m;

    printf("  state[%d]: ", id);

    /* ---- [1] 创建 cache context + tensor ---- */
    struct ggml_init_params cp = {
        .mem_size = 4 * ggml_tensor_overhead() + 256,
        .no_alloc = true,
    };
    st->ctx_cache = ggml_init(cp);
    if (!st->ctx_cache) return -1;

    /* cache_h: 缓存 hidden state，维度可以和模型输出不同 */
    st->cache_h = ggml_new_tensor_1d(st->ctx_cache, GGML_TYPE_F32, CACHE_DIM);
    ggml_set_name(st->cache_h, "cache_h");

    /* cache_count: 记录推理次数，演示多个 cache tensor */
    st->cache_count = ggml_new_tensor_1d(st->ctx_cache, GGML_TYPE_F32, 1);
    ggml_set_name(st->cache_count, "cache_count");

    /* ---- [2] 分配 cache buffer ---- */
    /*
     * 选择 cpu_backend 的原因:
     *   - 权重也在 cpu_backend，避免 sched 做跨后端拷贝
     *   - cache 数据量小，CPU 拷贝开销可忽略
     *
     * 如果 cache 很大且需要频繁 GPU 读写，改用 h->backend
     */
    st->buf_cache = ggml_backend_alloc_ctx_tensors(st->ctx_cache, h->cpu_backend);
    if (!st->buf_cache) return -1;

    /* 初始化为 0 */
    float zeros[CACHE_DIM] = {0};
    ggml_backend_tensor_set(st->cache_h, zeros, 0, sizeof(zeros));
    float zero = 0.0f;
    ggml_backend_tensor_set(st->cache_count, &zero, 0, sizeof(float));

    printf("cache=%zuB ",
           (size_t)ggml_backend_buffer_get_size(st->buf_cache));

    /* ---- [3] 分配 build buffer（外部 malloc，ctx_build 不持有） ---- */
    st->build_buf = malloc(BUILD_BUF_SIZE);
    if (!st->build_buf) return -1;

    /* ---- [4] 创建 sched ---- */
    ggml_backend_t backends[] = { h->backend, h->cpu_backend };
    ggml_backend_buffer_type_t bufts[] = {
        ggml_backend_get_default_buffer_type(h->backend),
        ggml_backend_get_default_buffer_type(h->cpu_backend),
    };
    int n = (h->backend == h->cpu_backend) ? 1 : 2;

    st->sched = ggml_backend_sched_new(
        backends, bufts, n, GRAPH_SIZE, true, true);
    if (!st->sched) return -1;

    /* ---- [5] 预分配: 构建图 → alloc → reset ---- */
    /*
     * 这一步的作用:
     *   - alloc_graph: sched 识别出 cache_h/cache_count/W/bias 已预分配，
     *                  只为中间 tensor 分配 work buffer
     *   - reset: 清除图映射，保留 work buffer
     *
     * 后续推理调用 alloc_graph 时直接复用 work buffer，零分配
     */
    ggml_cgraph *gf = build_graph(st);
    if (!gf) return -1;

    if (!ggml_backend_sched_alloc_graph(st->sched, gf)) {
        fprintf(stderr, "sched_alloc_graph 失败\n");
        return -1;
    }
    ggml_backend_sched_reset(st->sched);

    printf("✓\n");
    return 0;
}

/* ================================================================
 *  第五步：推理
 *
 *  缓存工作流:
 *    输入 → 计算(output = W@input + cache_h + bias)
 *         → 读取 output
 *         → 更新 cache_h = output（为下次推理准备）
 *         → 更新 cache_count++
 *         → reset sched
 * ================================================================ */
static int state_infer(state_t *st,
                       float *input_data,
                       float *output_data)
{
    /* ---- A. 构建图（引用权重 + cache，不拷贝） ---- */
    ggml_cgraph *gf = build_graph(st);
    if (!gf) return -1;

    /* ---- B. 分配（图结构不变 → 复用 work buffer → 零分配） ---- */
    if (!ggml_backend_sched_alloc_graph(st->sched, gf)) return -1;

    /* ---- C. 设置输入 ---- */
    ggml_tensor *input = ggml_graph_get_tensor(gf, "input");
    ggml_backend_tensor_set(input, input_data, 0, sizeof(float) * IN_DIM);

    /* ---- D. 执行 ---- */
    ggml_backend_sched_graph_compute(st->sched, gf);

    /* ---- E. 读取输出 ---- */
    ggml_tensor *output = ggml_graph_get_tensor(gf, "output");
    ggml_backend_tensor_get(output, output_data, 0, sizeof(float) * OUT_DIM);

    /* ---- F. 更新 cache（核心：将输出写回 cache tensor） ---- */
    /*
     * cache_h 的数据在 buf_cache 中（不在 sched 的 work buffer 中）。
     * 用 ggml_backend_tensor_set 将 output 写入 cache_h，
     * 下次推理时 build_graph 会引用更新后的 cache_h。
     */
    ggml_backend_tensor_set(st->cache_h, output_data, 0,
                            sizeof(float) * CACHE_DIM);

    /* 更新调用计数 */
    float count;
    ggml_backend_tensor_get(st->cache_count, &count, 0, sizeof(float));
    count += 1.0f;
    ggml_backend_tensor_set(st->cache_count, &count, 0, sizeof(float));

    /* ---- G. 重置 sched（保留 work buffer，清除图映射） ---- */
    ggml_backend_sched_reset(st->sched);

    return 0;
}

/* ================================================================
 *  第六步：读取 cache 当前值（调试用）
 * ================================================================ */
static void state_dump_cache(const state_t *st) {
    float h[CACHE_DIM];
    float count;
    ggml_backend_tensor_get(st->cache_h, h, 0, sizeof(float) * CACHE_DIM);
    ggml_backend_tensor_get(st->cache_count, &count, 0, sizeof(float));

    printf("      cache_h=[");
    for (int i = 0; i < CACHE_DIM; i++)
        printf("%.4f%s", h[i], i < CACHE_DIM - 1 ? "," : "");
    printf("] calls=%.0f\n", count);
}

/* ================================================================
 *  第七步：释放实例
 *
 *  释放顺序: sched → build → cache（依赖关系：后分配先释放）
 * ================================================================ */
static void state_free(state_t *st) {
    printf("  state[%d]:\n", st->id);

    if (st->sched) {
        ggml_backend_sched_free(st->sched);
        printf("    [✓] sched (含 work buffer)\n");
    }
    if (st->ctx_build) {
        ggml_free(st->ctx_build);
        printf("    [✓] ctx_build\n");
    }
    if (st->build_buf) {
        free(st->build_buf);
        printf("    [✓] build_buf\n");
    }
    if (st->buf_cache) {
        ggml_backend_buffer_free(st->buf_cache);
        printf("    [✓] buf_cache\n");
    }
    if (st->ctx_cache) {
        ggml_free(st->ctx_cache);
        printf("    [✓] ctx_cache\n");
    }

    memset(st, 0, sizeof(*st));
}

/* ================================================================
 *  主函数
 * ================================================================ */
int main(void) {
    printf("========================================\n");
    printf("  Sched + Cache Tensor 多实例泄漏检测\n");
    printf("========================================\n\n");

    #define N_INST  3
    #define N_ROUNDS 4

    /* 1. 后端 */
    ggml_handle_t handle;
    handle_init(&handle);

    /* 2. 模型 */
    printf("加载模型:\n");
    model_t model;
    model_load(&model, &handle);

    /* 3. 多实例 */
    printf("\n创建 %d 个实例:\n", N_INST);
    state_t states[N_INST];
    for (int i = 0; i < N_INST; i++) {
        if (state_create(&states[i], &model, &handle, i) != 0) {
            fprintf(stderr, "state[%d] 创建失败\n", i);
            goto cleanup;
        }
    }

    /* 4. 推理循环 */
    printf("\n推理 (%d 轮 × %d 实例):\n", N_ROUNDS, N_INST);

    for (int r = 0; r < N_ROUNDS; r++) {
        printf("  ── 轮次 %d ──\n", r + 1);

        for (int i = 0; i < N_INST; i++) {
            float in[IN_DIM], out[OUT_DIM];

            /* 每个实例不同的输入（模拟不同用户/场景） */
            for (int j = 0; j < IN_DIM; j++)
                in[j] = (r + 1) * 0.1f + i * 0.5f + j * 0.01f;

            state_infer(&states[i], in, out);

            printf("    state[%d]: in=[%.2f,%.2f,%.2f,%.2f] "
                   "-> out=[%.4f,%.4f,%.4f,%.4f]\n",
                   i, in[0], in[1], in[2], in[3],
                   out[0], out[1], out[2], out[3]);

            /* 每轮最后一轮打印 cache 状态 */
            if (r == N_ROUNDS - 1) {
                state_dump_cache(&states[i]);
            }
        }
    }

    /* 5. 释放 */
cleanup:
    printf("\n释放:\n");
    for (int i = 0; i < N_INST; i++)
        state_free(&states[i]);

    printf("模型:\n");
    model_free(&model);

    printf("后端:\n");
    handle_free(&handle);

    printf("\n完成。\n");
    return 0;
}
