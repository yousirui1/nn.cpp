/*
 * example_sched_leak_test.c
 *
 * 模拟你的实际场景：backend_sched + 双后端 + VAD 模型
 *
 * 编译: g++ -O0 -g -o test example_sched_leak_test.cpp -lggml -lggml-cuda -lcudart -lm
 * 检测: valgrind --leak-check=full --show-leak-kinds=definite --suppressions=ggml.supp ./test
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>

/* ================================================================
 *  模拟你的 ggml_handle 结构
 * ================================================================ */
typedef struct {
    ggml_backend_t backend;      // GPU（或 fallback 到 CPU）
    ggml_backend_t cpu_backend;  // CPU 始终存在
} ggml_handle_t;

/* ================================================================
 *  模拟你的 model / state 结构
 * ================================================================ */
typedef struct {
    ggml_context     *ctx;
    ggml_backend_buffer_t buf_weights;
    ggml_tensor      *weight;    // 共享权重
} model_t;

typedef struct {
    ggml_context       *ctx;
    ggml_backend_buffer_t buf_state;  // state tensor（如 KV cache）
    ggml_tensor        *input;
    ggml_tensor        *output;
    ggml_backend_sched_t sched;       // 调度器
} state_t;

#define IN_DIM   4
#define OUT_DIM  4
#define GRAPH_SIZE 256

/* ================================================================
 *  初始化 handle（模拟 GPU + CPU 双后端）
 * ================================================================ */
static void handle_init(ggml_handle_t *h) {
    h->cpu_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);

    h->backend = h->cpu_backend;
    /* 尝试 CUDA，失败就退化为纯 CPU */
    //h->backend = ggml_backend_cuda_init(0);
    if (!h->backend) {
        printf("  CUDA 不可用，退化为 CPU\n");
    } else {
        printf("  CUDA 可用\n");
    }
}

static void handle_free(ggml_handle_t *h) {
    if (h->backend && h->backend != h->cpu_backend) {
        ggml_backend_free(h->backend);
        printf("  [✓] GPU backend freed\n");
    }
    if (h->cpu_backend) {
        ggml_backend_free(h->cpu_backend);
        printf("  [✓] CPU backend freed\n");
    }
}

/* ================================================================
 *  模型加载（权重，只分配一次）
 * ================================================================ */
static int model_load(model_t *m, ggml_handle_t *h) {
    memset(m, 0, sizeof(*m));

    struct ggml_init_params p = {
        .mem_size   = 2 * ggml_tensor_overhead() + 256,
        .no_alloc   = true,
    };
    m->ctx = ggml_init(p);
    if (!m->ctx) return -1;

    m->weight = ggml_new_tensor_2d(m->ctx, GGML_TYPE_F32, IN_DIM, OUT_DIM);
    ggml_set_name(m->weight, "weight");

    /* 权重数据在 CPU backend 的 buffer 中 */
    m->buf_weights = ggml_backend_alloc_ctx_tensors(m->ctx, h->cpu_backend);
    if (!m->buf_weights) return -1;

    float data[IN_DIM * OUT_DIM];
    for (int i = 0; i < IN_DIM * OUT_DIM; i++) data[i] = 0.1f * (i + 1);
    ggml_backend_tensor_set(m->weight, data, 0, sizeof(data));

    printf("模型加载完成 ✓\n");
    return 0;
}

static void model_free(model_t *m) {
    if (m->buf_weights) {
        ggml_backend_buffer_free(m->buf_weights);
        printf("  [✓] buf_weights freed\n");
    }
    if (m->ctx) {
        ggml_free(m->ctx);
        printf("  [✓] model ctx freed\n");
    }
}

/* ================================================================
 *  构建计算图（模拟 silero_build_cgraph）
 * ================================================================ */
static ggml_cgraph * build_graph(model_t *model, state_t *state) {
    /*
     * state->ctx 作为构建 context（no_alloc=true，外部内存）
     * 这里每次重新构建前需要重置 ctx
     */
    struct ggml_init_params p = {
        .mem_size   = 256 * 1024,
        .mem_buffer = NULL,         // 实际项目中可以用外部 buffer
        .no_alloc   = true,
    };

    /* 如果 ctx 已存在，先释放 */
    if (state->ctx) {
        ggml_free(state->ctx);
        state->ctx = NULL;
    }

    state->ctx = ggml_init(p);
    if (!state->ctx) return NULL;

    /* 创建 state tensor（如输入/输出） */
    state->input = ggml_new_tensor_1d(state->ctx, GGML_TYPE_F32, IN_DIM);
    ggml_set_name(state->input, "input");

    /* 计算: output = input * W（用共享权重） */
    state->output = ggml_mul_mat(state->ctx, model->weight, state->input);
    ggml_set_name(state->output, "output");

    /* 构建前向图 */
    ggml_cgraph *gf = ggml_new_graph(state->ctx);
    ggml_build_forward_expand(gf, state->output);

    return gf;
}

/* ================================================================
 *  创建 state（包含 sched）
 * ================================================================ */
static int state_create(state_t *st, model_t *model, ggml_handle_t *h) {
    memset(st, 0, sizeof(*st));

    /* ---- 1. 创建 backend_sched -------------------------------- */
    ggml_backend_t backends[] = { h->backend, h->cpu_backend };
    int n_backends = (h->backend == h->cpu_backend) ? 1 : 2;

    st->sched = ggml_backend_sched_new(
        backends,
        NULL,              // bufts — NULL 让 sched 自动选择
        n_backends,
        GRAPH_SIZE,        // max_nodes
        true,              // parallel
        true               // measure — 首次运行时自动 benchmark
    );
    if (!st->sched) return -1;

    printf("  sched 创建完成, n_backends=%d\n", n_backends);

    /* ---- 2. 构建图 + 分配 sched buffer ----------------------- */
    ggml_cgraph *gf = build_graph(model, st);
    if (!gf) return -1;

    /* alloc_graph: 为图中所有 tensor 分配 buffer（跨后端自动分配） */
    if (!ggml_backend_sched_alloc_graph(st->sched, gf)) {
        fprintf(stderr, "错误: sched_alloc_graph 失败\n");
        return -1;
    }

    /* set_graph_backend: 可选，手动指定某些 tensor 放在哪个后端 */
    /* 这里不手动指定，让 sched 自动决定 */
    // ggml_backend_sched_set_tensor_backend(st->sched, tensor, backend);

    /* reset: 重置调度器状态（保留 buffer，释放 graph 映射） */
    ggml_backend_sched_reset(st->sched);

    printf("  state 创建完成 ✓\n");
    return 0;
}

/* ================================================================
 *  推理一次
 * ================================================================ */
static int state_infer(state_t *st, model_t *model, ggml_handle_t *h,
                       float *in, float *out)
{
    /* ---- 1. 重新构建图（模拟每帧调用） ----------------------- */
    ggml_cgraph *gf = build_graph(model, st);
    if (!gf) return -1;

    /* ---- 2. 分配图 buffer ------------------------------------- */
    /*
     * alloc_graph 内部会判断：
     *   - 如果图结构没变 → 复用已有 buffer，零分配
     *   - 如果图结构变了 → 重新分配
     *
     * 所以不需要手动管理，直接调用即可
     */
    if (!ggml_backend_sched_alloc_graph(st->sched, gf)) {
        fprintf(stderr, "错误: alloc_graph 失败\n");
        return -1;
    }

    /* ---- 3. 设置输入 ------------------------------------------ */
    ggml_backend_tensor_set(st->input, in, 0, sizeof(float) * IN_DIM);

    /* ---- 4. 执行（sched 自动分配到 GPU/CPU） ----------------- */
    ggml_backend_sched_graph_compute(st->sched, gf);

    /* ---- 5. 读取输出 ------------------------------------------ */
    ggml_backend_tensor_get(st->output, out, 0, sizeof(float) * OUT_DIM);

    /* ---- 6. reset（保留 buffer，释放图映射，下次可复用） ---- */
    ggml_backend_sched_reset(st->sched);

    return 0;
}

/* ================================================================
 *  释放 state
 * ================================================================ */
static void state_free(state_t *st) {
    printf("  释放 state:\n");

    /*
     * 释放顺序：
     *   1. sched — 它内部持有 work buffer
     *   2. ctx   — 构建上下文
     *
     * 注意：先释放 sched，再释放 ctx
     * 因为 sched 可能引用了 ctx 中的 tensor 信息
     */
    if (st->sched) {
        ggml_backend_sched_free(st->sched);
        printf("    [✓] sched freed\n");
    }

    if (st->ctx) {
        ggml_free(st->ctx);
        printf("    [✓] ctx freed\n");
    }

    memset(st, 0, sizeof(*st));
}

/* ================================================================
 *  主函数
 * ================================================================ */
int main(void) {
    printf("========================================\n");
    printf("  backend_sched 多实例泄漏检测\n");
    printf("========================================\n\n");

    #define N_INST  2
    #define N_ROUNDS 3

    /* ===== 1. 初始化后端 ======================================== */
    ggml_handle_t handle;
    handle_init(&handle);

    /* ===== 2. 加载模型（一次） ================================== */
    model_t model;
    model_load(&model, &handle);

    /* ===== 3. 创建多个 state =================================== */
    printf("\n创建 %d 个 state:\n", N_INST);
    state_t states[N_INST];

    for (int i = 0; i < N_INST; i++) {
        printf("  state[%d]:\n", i);
        if (state_create(&states[i], &model, &handle) != 0) {
            fprintf(stderr, "state[%d] 创建失败\n", i);
            goto cleanup;
        }
    }

    /* ===== 4. 多实例推理 ======================================= */
    printf("\n推理 (%d 轮, %d 实例):\n", N_ROUNDS, N_INST);

    for (int r = 0; r < N_ROUNDS; r++) {
        printf("  轮次 %d:\n", r + 1);
        for (int i = 0; i < N_INST; i++) {
            float in[IN_DIM], out[OUT_DIM];
            for (int j = 0; j < IN_DIM; j++)
                in[j] = (r + 1) * 0.1f + i * 1.0f + j * 0.01f;

            if (state_infer(&states[i], &model, &handle, in, out) != 0) {
                fprintf(stderr, "    state[%d] 推理失败\n", i);
                continue;
            }

            printf("    state[%d]: in=[%.2f,%.2f,%.2f,%.2f] "
                   "-> out=[%.3f,%.3f,%.3f,%.3f]\n",
                   i, in[0], in[1], in[2], in[3],
                   out[0], out[1], out[2], out[3]);
        }
    }

    /* ===== 5. 释放（严格逆序） ================================== */
cleanup:
    printf("\n释放资源:\n");

    /* 5a. 先释放所有 state（sched） */
    for (int i = 0; i < N_INST; i++) {
        printf("  state[%d]:\n", i);
        state_free(&states[i]);
    }

    /* 5b. 再释放模型 */
    printf("  模型:\n");
    model_free(&model);

    /* 5c. 最后释放后端 */
    printf("  后端:\n");
    handle_free(&handle);

    printf("\n完成。\n");
    return 0;
}
