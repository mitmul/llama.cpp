#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

/**
 * This the arbitrary data which will be passed to each callback.
 * Later on we can for example add operation or tensor name filter from the CLI arg, or a file descriptor to dump the tensor.
 */
struct callback_data {
    std::vector<uint8_t> data;
    std::vector<std::string> dump_prefixes;
    int64_t dump_n = 0;
};

static std::vector<std::string> get_dump_prefixes_from_env() {
    const char * env = std::getenv("LLAMA_EVAL_CALLBACK_TENSOR_PREFIX");
    if (!env || env[0] == '\0') {
        return {};
    }

    auto parts = string_split<std::string>(env, ',');
    std::vector<std::string> prefixes;
    prefixes.reserve(parts.size());

    for (auto & part : parts) {
        part = string_strip(part);
        if (!part.empty()) {
            prefixes.push_back(part);
        }
    }

    return prefixes;
}

static bool should_dump_tensor(const callback_data * cb_data, const ggml_tensor * t) {
    if (cb_data->dump_prefixes.empty()) {
        return true;
    }

    for (const auto & prefix : cb_data->dump_prefixes) {
        const size_t n = prefix.size();
        if (n == 0) {
            continue;
        }
        // Suffix '$' means exact match (handy to avoid 'ffn_residual-1' matching 'ffn_residual-10').
        if (prefix.back() == '$') {
            const std::string exact = prefix.substr(0, n - 1);
            if (!exact.empty() && std::strcmp(t->name, exact.c_str()) == 0) {
                return true;
            }
            continue;
        }

        if (std::strncmp(t->name, prefix.c_str(), n) == 0) {
            return true;
        }
    }

    return false;
}

static int64_t get_dump_n_from_env() {
    const char * env = std::getenv("LLAMA_EVAL_CALLBACK_DUMP_N");
    if (!env || env[0] == '\0') {
        // The user requested "GGML_MAX_DIMS=256"; interpret it as "dump first N values"
        // for this example (not the compile-time ggml tensor rank limit).
        env = std::getenv("GGML_MAX_DIMS");
    }
    if (!env || env[0] == '\0') {
        return 0;
    }

    char * end = nullptr;
    const long long v = std::strtoll(env, &end, 10);
    if (end == env || v <= 0) {
        return 0;
    }
    return (int64_t) v;
}

static std::string ggml_ne_string(const ggml_tensor * t) {
    std::string str;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        str += std::to_string(t->ne[i]);
        if (i + 1 < GGML_MAX_DIMS) {
            str += ", ";
        }
    }
    return str;
}

static inline float ggml_compute_bf16_to_fp32(ggml_bf16_t h) {
    union {
        float f;
        uint32_t i;
    } u;
    u.i = (uint32_t)h.bits << 16;
    return u.f;
}

static float ggml_get_float_value(const uint8_t * data, ggml_type type, const size_t * nb, size_t i0, size_t i1, size_t i2, size_t i3) {
    size_t i = i3 * nb[3] + i2 * nb[2] + i1 * nb[1] + i0 * nb[0];
    float v;
    if (type == GGML_TYPE_F16) {
        v = ggml_fp16_to_fp32(*(const ggml_fp16_t *) &data[i]);
    } else if (type == GGML_TYPE_F32) {
        v = *(const float *) &data[i];
    } else if (type == GGML_TYPE_I64) {
        v = (float) *(const int64_t *) &data[i];
    } else if (type == GGML_TYPE_I32) {
        v = (float) *(const int32_t *) &data[i];
    } else if (type == GGML_TYPE_I16) {
        v = (float) *(const int16_t *) &data[i];
    } else if (type == GGML_TYPE_I8) {
        v = (float) *(const int8_t *) &data[i];
    } else if (type == GGML_TYPE_BF16) {
        v = ggml_compute_bf16_to_fp32(*(const ggml_bf16_t *) &data[i]);
    } else {
        GGML_ABORT("fatal error");
    }
    return v;
}

static void ggml_dump_tensor_first_n(const ggml_tensor * t,
                                    uint8_t * data,
                                    ggml_type type,
                                    const int64_t * ne,
                                    const size_t * nb,
                                    int64_t dump_n) {
    int64_t total = 1;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        total *= ne[i] > 0 ? ne[i] : 0;
    }

    float sum = 0.0f;
    for (int64_t i3 = 0; i3 < ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < ne[2]; i2++) {
            for (int64_t i1 = 0; i1 < ne[1]; i1++) {
                for (int64_t i0 = 0; i0 < ne[0]; i0++) {
                    sum += ggml_get_float_value(data, type, nb, i0, i1, i2, i3);
                }
            }
        }
    }

    const int64_t take = dump_n > 0 ? std::min(dump_n, total) : 0;
    LOG("EVALCB %s type=%s shape={%s} sum=%.9g first_n=%lld total=%lld\n",
        t->name, ggml_type_name(type), ggml_ne_string(t).c_str(), sum, (long long) take, (long long) total);

    LOG("EVALCB_VALS %s", t->name);
    for (int64_t idx = 0; idx < take; ++idx) {
        int64_t tmp = idx;
        const int64_t i0 = ne[0] > 0 ? (tmp % ne[0]) : 0; tmp = ne[0] > 0 ? (tmp / ne[0]) : 0;
        const int64_t i1 = ne[1] > 0 ? (tmp % ne[1]) : 0; tmp = ne[1] > 0 ? (tmp / ne[1]) : 0;
        const int64_t i2 = ne[2] > 0 ? (tmp % ne[2]) : 0; tmp = ne[2] > 0 ? (tmp / ne[2]) : 0;
        const int64_t i3 = tmp;
        const float v = ggml_get_float_value(data, type, nb, i0, i1, i2, i3);
        LOG(" %.9g", v);
    }
    LOG("\n");

    if (std::isnan(sum)) {
        LOG_ERR("encountered NaN - aborting\n");
        exit(0);
    }
}

/**
 * GGML operations callback during the graph execution.
 *
 * @param t current tensor
 * @param ask when ask is true, the scheduler wants to know if we are interested in data from this tensor
 *            if we return true, a follow-up call will be made with ask=false in which we can do the actual collection.
 *            see ggml_backend_sched_eval_callback
 * @param user_data user data to pass at each call back
 * @return true to receive data or continue the graph, false otherwise
 */
static bool ggml_debug(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cb_data = (callback_data *) user_data;

    const struct ggml_tensor * src0 = t->src[0];
    const struct ggml_tensor * src1 = t->src[1];

    if (ask) {
        return should_dump_tensor(cb_data, t);
    }

    char src1_str[128] = {0};
    if (src1) {
        snprintf(src1_str, sizeof(src1_str), "%s{%s}", src1->name, ggml_ne_string(src1).c_str());
    }

    LOG("%s: %24s = (%s) %10s(%s{%s}, %s}) = {%s}\n", __func__,
         t->name, ggml_type_name(t->type), ggml_op_desc(t),
         src0->name, ggml_ne_string(src0).c_str(),
         src1 ? src1_str : "",
         ggml_ne_string(t).c_str());


    // copy the data from the GPU memory if needed
    const bool is_host = ggml_backend_buffer_is_host(t->buffer);

    if (!is_host) {
        auto n_bytes = ggml_nbytes(t);
        cb_data->data.resize(n_bytes);
        ggml_backend_tensor_get(t, cb_data->data.data(), 0, n_bytes);
    }

    if (!ggml_is_quantized(t->type)) {
        uint8_t * data = is_host ? (uint8_t *) t->data : cb_data->data.data();
        if (cb_data->dump_n > 0) {
            ggml_dump_tensor_first_n(t, data, t->type, t->ne, t->nb, cb_data->dump_n);
        } else {
            ggml_dump_tensor_first_n(t, data, t->type, t->ne, t->nb, 3);
        }
    }

    return true;
}

static bool run(llama_context * ctx, const common_params & params) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const bool add_bos = llama_vocab_get_add_bos(vocab);

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos);

    if (tokens.empty()) {
        LOG_ERR("%s : there are not input tokens to process - (try to provide a prompt with '-p')\n", __func__);
        return false;
    }

    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()))) {
        LOG_ERR("%s : failed to eval\n", __func__);
        return false;
    }

    return true;
}

int main(int argc, char ** argv) {
    callback_data cb_data;
    cb_data.dump_prefixes = get_dump_prefixes_from_env();
    cb_data.dump_n = get_dump_n_from_env();

    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    common_init();

    llama_backend_init();
    llama_numa_init(params.numa);

    if (!cb_data.dump_prefixes.empty()) {
        LOG("eval-callback: LLAMA_EVAL_CALLBACK_TENSOR_PREFIX=%s\n", std::getenv("LLAMA_EVAL_CALLBACK_TENSOR_PREFIX"));
    }
    if (cb_data.dump_n > 0) {
        LOG("eval-callback: dumping first %lld values per tensor\n", (long long) cb_data.dump_n);
    }

    // pass the callback to the backend scheduler
    // it will be executed for each node during the graph computation
    params.cb_eval = ggml_debug;
    params.cb_eval_user_data = &cb_data;
    params.warmup = false;

    // init
    auto llama_init = common_init_from_params(params);

    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s : failed to init\n", __func__);
        return 1;
    }

    // print system information
    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
        LOG_INF("\n");
    }

    bool OK = run(ctx, params);
    if (!OK) {
        return 1;
    }

    LOG("\n");
    llama_perf_context_print(ctx);

    llama_backend_free();

    return 0;
}
