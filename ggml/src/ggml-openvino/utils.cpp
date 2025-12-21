#include "utils.h"

#include "ggml-impl.h"
#include "ggml-backend.h"
#include "ggml-openvino/ggml-decoder.h"
#include "ggml.h"
#include "openvino/frontend.hpp"
#include "openvino/input_model.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <openvino/core/any.hpp>
#include <openvino/core/graph_util.hpp>
#include <openvino/core/shape.hpp>
#include <openvino/core/type/float16.hpp>
#include <openvino/frontend/manager.hpp>
#include <openvino/openvino.hpp>
#include <openvino/runtime/compiled_model.hpp>
#include <openvino/runtime/infer_request.hpp>
#include <openvino/runtime/intel_npu/properties.hpp>
#include <openvino/runtime/properties.hpp>
#include <openvino/runtime/tensor.hpp>
#include <string>
#include <unordered_map>
#include <vector>

// Suppress deprecation warning for ov::Tensor::data()
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

static ov::Core core;

enum ggml_status ov_graph_compute(ggml_cgraph * cgraph) {
    auto get_device = [&] {
        std::string device = getenv("GGML_OPENVINO_DEVICE") ? getenv("GGML_OPENVINO_DEVICE") : "CPU";
        auto available_devices = core.get_available_devices();
        if (std::find(available_devices.begin(), available_devices.end(), device) == available_devices.end()) {
            GGML_LOG_WARN("GGML OpenVINO Backend: device %s is not available, fallback to CPU\n", device.c_str());
            device = "CPU";
        }
        return device;
    };

    if (getenv("GGML_OPENVINO_DUMP_CGRAPH")) {
        std::string filename = "cgraph.txt";
        GgmlOvDecoder::dump_cgraph(cgraph, filename);
    }

    static const auto device = get_device();
    static const bool force_dynamic = [] {
        const char * env = getenv("GGML_OPENVINO_FORCE_DYNAMIC");
        return env && std::string(env) != "0";
    }();

    const bool is_static = device == "NPU" && !force_dynamic;
    return is_static ? ov_graph_compute_static(cgraph) : ov_graph_compute_dynamic(cgraph, device);
}

enum ggml_status ov_graph_compute_dynamic(ggml_cgraph * cgraph, const std::string & device) {
    try {
    static auto is_static = false;
    static auto config = get_ov_compile_config(device);

    // if (is_naive(cgraph)) {
    //     return naive_compute(cgraph, core, device, config);
    // }

    auto start_time = ggml_time_us();
    GGML_LOG_INFO("GGML OpenVINO Backend: dynamic graph compute start (device=%s)\n", device.c_str());

    static std::mutex cache_mutex;
    static std::unordered_map<graph_key, std::shared_ptr<GgmlOvDecoder>, graph_key_hash> decoder_cache;
    static std::unordered_map<graph_key, std::shared_ptr<ov::InferRequest>, graph_key_hash> infer_request_cache;
    static std::unordered_map<graph_key, std::vector<std::string>, graph_key_hash> ov_input_names_cache;
    static std::unordered_map<graph_key, std::vector<std::string>, graph_key_hash> ov_output_names_cache;

    std::shared_ptr<GgmlOvDecoder> ggml_decoder;
    std::shared_ptr<ov::InferRequest> infer_request;
    ModelParams m_params;
    ComputeParams c_params;
    std::tie(m_params, c_params) = GgmlOvDecoder::compute_llm_params(cgraph, is_static);

    const auto key = compute_graph_key(cgraph);
    bool cache_hit = false;
    bool force_rebuild = false;

    int64_t decoder_end_time;
    int64_t conversion_end_time;
    int64_t compile_end_time;
    int64_t infer_end_time;

    for (int attempt = 0; attempt < 2; ++attempt) {
        {
            std::lock_guard<std::mutex> lock(cache_mutex);

            if (force_rebuild) {
                decoder_cache.erase(key);
                infer_request_cache.erase(key);
                ov_input_names_cache.erase(key);
                ov_output_names_cache.erase(key);
            }

            auto it = decoder_cache.find(key);

            cache_hit = !force_rebuild && it != decoder_cache.end();
            if (cache_hit) {
                ggml_decoder = it->second;
                cache_hit = ggml_decoder->get_model_params().can_reuse_dynamically(m_params);
            }

            if (cache_hit) {
                GGML_LOG_INFO("GGML OpenVINO Backend: reusing cached decoder/infer request\n");
                ggml_decoder = decoder_cache[key];
                ggml_decoder->set_compute_params(c_params);
                ggml_decoder->set_model_params(m_params);
                ggml_decoder->add_extra_inputs();
                infer_request = infer_request_cache[key];

                decoder_end_time = ggml_time_us();
                conversion_end_time = decoder_end_time;
                compile_end_time = decoder_end_time;
            } else {
                infer_request_cache.erase(key);

                std::shared_ptr<ov::Model> model;
                auto model_weights = GgmlOvDecoder::create_weight_nodes(cgraph, get_types_to_requant(device));

                GGML_LOG_INFO("GGML OpenVINO Backend: creating decoder\n");
                ggml_decoder = std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, model_weights, is_static);
                decoder_end_time = ggml_time_us();

                auto input_model = std::make_shared<ov::frontend::ggml::InputModel>(ggml_decoder);
                GGML_LOG_INFO("GGML OpenVINO Backend: converting graph to OpenVINO model\n");
                model = ov::frontend::ggml::FrontEnd::convert(input_model);
                ggml_decoder->clear_model_weights();
                conversion_end_time = ggml_time_us();

                if (getenv("GGML_OPENVINO_DUMP_IR")) {
                    char timestamped_filename[64];
                    auto timestamp = (long long) ggml_time_us();
                    snprintf(timestamped_filename, sizeof(timestamped_filename), "model_%lld.xml", timestamp);
                    ov::serialize(model, timestamped_filename);
                }

                GGML_LOG_INFO("GGML OpenVINO Backend: compiling model\n");
                auto compiled_model = core.compile_model(model, device, config);
                compile_end_time = ggml_time_us();
                infer_request = std::make_shared<ov::InferRequest>(compiled_model.create_infer_request());
                infer_request_cache[key] = infer_request;
                decoder_cache[key] = ggml_decoder;

                std::vector<std::string> ov_input_names;
                std::vector<std::string> ov_output_names;
                for (const auto & ov_param : model->get_parameters()) {
                    ov_input_names.push_back(ov_param->get_friendly_name());
                }
                for (const auto & ov_output : model->get_results()) {
                    ov_output_names.push_back(ov_output->get_friendly_name());
                }
                ov_input_names_cache[key] = std::move(ov_input_names);
                ov_output_names_cache[key] = std::move(ov_output_names);
                GGML_LOG_INFO("GGML OpenVINO Backend: model compiled (inputs=%zu, outputs=%zu)\n",
                              ov_input_names_cache[key].size(), ov_output_names_cache[key].size());
            }
        }

        auto ov_input_names = ov_input_names_cache[key];
        auto ov_output_names = ov_output_names_cache[key];

        try {
            for (size_t i = 0; i < ov_input_names.size(); i++) {
                auto param_name = ov_input_names[i];
                auto input_tensor = get_ov_input_tensor(ggml_decoder, param_name);
                infer_request->set_input_tensor(i, input_tensor);

                if (getenv("GGML_OPENVINO_DEBUG_INPUT")) {
                    print_input_tensor_info(param_name, input_tensor);
                }
            }

            for (size_t i = 0; i < ov_output_names.size(); i++) {
                auto output_tensor = get_ov_output_tensor(ggml_decoder, ov_output_names[i]);
                infer_request->set_output_tensor(i, output_tensor);
            }

            infer_request->infer();
            infer_end_time = ggml_time_us();
            break;
        } catch (const std::exception & e) {
            if (cache_hit && !force_rebuild) {
                GGML_LOG_WARN("GGML OpenVINO Backend: cached model input mismatch, rebuilding: %s\n", e.what());
                force_rebuild = true;
                continue;
            }
            throw;
        }
    }

    auto ov_input_names = ov_input_names_cache[key];
    auto ov_output_names = ov_output_names_cache[key];

    if (getenv("GGML_OPENVINO_DEBUG_OUTPUT")) {
        for (size_t i = 0; i < ov_output_names.size(); i++) {
            const auto output_tensor = infer_request->get_output_tensor(i);
            print_output_tensor_info(ov_output_names[i], output_tensor, output_tensor.data());
        }
    }

    if (getenv("GGML_OPENVINO_PROFILING")) {
        GGML_LOG_INFO("\nGGML OpenVINO Backend: \n");
        GGML_LOG_INFO("  - Graph decoder Time: %ld ms \n", (decoder_end_time - start_time) / 1000);
        if (!cache_hit) {
            GGML_LOG_INFO("  - Graph conversion Time: %ld ms \n", (conversion_end_time - decoder_end_time) / 1000);
            GGML_LOG_INFO("  - Graph compile Time: %ld ms \n", (compile_end_time - conversion_end_time) / 1000);
        }
        GGML_LOG_INFO("  - Graph Inference Time: %ld ms \n", (infer_end_time - compile_end_time) / 1000);
    }

    GGML_LOG_INFO("GGML OpenVINO Backend: inference completed\n");

    return GGML_STATUS_SUCCESS;
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("GGML OpenVINO Backend: dynamic compute exception: %s\n", e.what());
    } catch (...) {
        GGML_LOG_ERROR("GGML OpenVINO Backend: dynamic compute encountered an unknown exception\n");
    }
    return GGML_STATUS_FAILED;
}

enum ggml_status ov_graph_compute_static(ggml_cgraph * cgraph) {
    const char * stage = "init";
    const auto key = compute_graph_key(cgraph);
    try {
    auto get_prefill_chunk_size = [] {
        const char * chunk_size_str = getenv("GGML_OPENVINO_PREFILL_CHUNK_SIZE");
        if (chunk_size_str && atoi(chunk_size_str) > 0) {
            return atoi(chunk_size_str);
        }
        return 256;
    };

    static std::string device = "NPU";
    static auto is_static = true;
    static auto prefill_chunk_size = get_prefill_chunk_size();
    static auto config = get_ov_compile_config(device);

    auto find_inp_pos_tensor = [](ggml_cgraph * graph) -> const ggml_tensor * {
        for (int i = 0; i < graph->n_nodes; ++i) {
            auto * op = graph->nodes[i];
            for (int j = 0; j < GGML_MAX_SRC; ++j) {
                auto * src = op->src[j];
                if (src == nullptr) {
                    break;
                }
                if (std::string(src->name) == "inp_pos") {
                    return src;
                }
            }
        }
        return nullptr;
    };

    const auto * inp_pos = find_inp_pos_tensor(cgraph);
    if (inp_pos == nullptr) {
        // Some backend-scheduled subgraphs (e.g. SSM-only paths) do not reference inp_pos.
        // The static OpenVINO frontend path can be unstable for these graphs, so fall back
        // to the naive converter on the CPU plugin.
        stage = "fallback_naive_no_inp_pos";
        return naive_compute(cgraph, core, "CPU", get_ov_compile_config("CPU"));
    }

    if (is_naive(cgraph)) {
        // The Intel NPU plugin is strict about IO tensor element types for these tiny "naive" graphs and
        // can fail with f32/f16 mismatches. Run them on the CPU plugin instead so the main LLM graphs can
        // still execute on the NPU.
        stage = "fallback_naive_small_graph";
        return naive_compute(cgraph, core, "CPU", get_ov_compile_config("CPU"));
    }

    auto start_time = ggml_time_us();

    static std::mutex cache_mutex;
    static std::unordered_map<graph_key, std::shared_ptr<GgmlOvDecoder>, graph_key_hash> decoder_cache;
    static std::unordered_map<graph_key, std::shared_ptr<ov::InferRequest>, graph_key_hash> infer_request_cache;
    static std::unordered_map<graph_key, std::shared_ptr<ov::InferRequest>, graph_key_hash> infer_request_cache_prefill;
    static std::unordered_map<graph_key, std::vector<std::string>, graph_key_hash> ov_input_names_cache;
    static std::unordered_map<graph_key, std::vector<std::string>, graph_key_hash> ov_output_names_cache;

    std::shared_ptr<GgmlOvDecoder> ggml_decoder;
    std::shared_ptr<ov::InferRequest> infer_request;
    ModelParams m_params;
    ComputeParams c_params;
    stage = "compute_llm_params";
    std::tie(m_params, c_params) = GgmlOvDecoder::compute_llm_params(cgraph, is_static);
    stage = "cache_lock";

    const bool is_prefill = get_is_prefill(inp_pos);
    bool cache_hit = false;

    int64_t decoder_end_time;
    int64_t conversion_end_time;
    int64_t compile_end_time;
    int64_t infer_end_time;

    std::vector<std::string> ov_input_names;
    std::vector<std::string> ov_output_names;

    stage = "cache_lookup";
    {
        std::lock_guard<std::mutex> lock(cache_mutex);

        auto it = decoder_cache.find(key);
        cache_hit = it != decoder_cache.end();
        if (cache_hit) {
            ggml_decoder = it->second;
            cache_hit = ggml_decoder->get_model_params().can_reuse_statically(m_params);
            if (!cache_hit) {
                decoder_cache.erase(key);
                infer_request_cache.erase(key);
                infer_request_cache_prefill.erase(key);
                ov_input_names_cache.erase(key);
                ov_output_names_cache.erase(key);
            }
        }

        if (cache_hit) {
            stage = "cache_hit";
            ggml_decoder = decoder_cache[key];
            ggml_decoder->m_is_prefill = is_prefill;
            ggml_decoder->set_model_params(m_params);
            ggml_decoder->set_compute_params(c_params);
            ggml_decoder->add_extra_inputs();
            infer_request = is_prefill ? infer_request_cache_prefill[key] : infer_request_cache[key];

            ov_input_names = ov_input_names_cache[key];
            ov_output_names = ov_output_names_cache[key];

            decoder_end_time = ggml_time_us();
            conversion_end_time = decoder_end_time;
            compile_end_time = decoder_end_time;
        }
    }

    if (!cache_hit) {
        stage = "build_models";

        std::shared_ptr<ov::InferRequest> infer_request_prefill;
        std::shared_ptr<ov::InferRequest> infer_request_decode;

        try {
            std::shared_ptr<ov::Model> model;

            stage = "build_model_weights";
            auto model_weights = GgmlOvDecoder::create_weight_nodes(cgraph, get_types_to_requant(device));

            auto ggml_decoder_prefill = std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, model_weights,
                                                                        is_static, true, prefill_chunk_size);
            auto ggml_decoder_decode = std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, model_weights,
                                                                       is_static, false, prefill_chunk_size);
            decoder_end_time = ggml_time_us();

            auto input_model_prefill = std::make_shared<ov::frontend::ggml::InputModel>(ggml_decoder_prefill);
            auto input_model_decode = std::make_shared<ov::frontend::ggml::InputModel>(ggml_decoder_decode);

            stage = "convert_models";
            auto model_prefill = ov::frontend::ggml::FrontEnd::convert(input_model_prefill);
            ggml_decoder_prefill->clear_model_weights();
            auto model_decode = ov::frontend::ggml::FrontEnd::convert(input_model_decode);
            ggml_decoder_decode->clear_model_weights();
            conversion_end_time = ggml_time_us();

            if (getenv("GGML_OPENVINO_DUMP_IR")) {
                char timestamped_filename[64];
                auto timestamp = (long long) ggml_time_us();
                snprintf(timestamped_filename, sizeof(timestamped_filename), "model_prefill_%lld.xml", timestamp);
                ov::serialize(model_prefill, timestamped_filename);
                snprintf(timestamped_filename, sizeof(timestamped_filename), "model_decode_%lld.xml", timestamp);
                ov::serialize(model_decode, timestamped_filename);
            }

            stage = "compile_models";
            auto compiled_model_prefill = core.compile_model(model_prefill, device, get_ov_compile_config(device));
            auto compiled_model_decode = core.compile_model(model_decode, device, get_ov_compile_config(device));

            stage = "create_infer_requests";
            infer_request_prefill = std::make_shared<ov::InferRequest>(compiled_model_prefill.create_infer_request());
            infer_request_decode = std::make_shared<ov::InferRequest>(compiled_model_decode.create_infer_request());
            compile_end_time = ggml_time_us();

            model = is_prefill ? model_prefill : model_decode;
            ggml_decoder = is_prefill ? ggml_decoder_prefill : ggml_decoder_decode;
            infer_request = is_prefill ? infer_request_prefill : infer_request_decode;

            for (const auto & ov_param : model->get_parameters()) {
                ov_input_names.push_back(ov_param->get_friendly_name());
            }
            for (const auto & ov_output : model->get_results()) {
                ov_output_names.push_back(ov_output->get_friendly_name());
            }
        } catch (const std::exception & e) {
            GGML_LOG_WARN("GGML OpenVINO Backend: static path failed (stage=%s first=%s last=%s), fallback to naive/CPU: %s\n",
                          stage, key.first_node_name.c_str(), key.last_node_name.c_str(), e.what());
            return naive_compute(cgraph, core, "CPU", get_ov_compile_config("CPU"));
        } catch (...) {
            GGML_LOG_WARN("GGML OpenVINO Backend: static path failed (stage=%s first=%s last=%s), fallback to naive/CPU\n",
                          stage, key.first_node_name.c_str(), key.last_node_name.c_str());
            return naive_compute(cgraph, core, "CPU", get_ov_compile_config("CPU"));
        }

        stage = "cache_store";
        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            infer_request_cache_prefill[key] = infer_request_prefill;
            infer_request_cache[key] = infer_request_decode;
            decoder_cache[key] = ggml_decoder;
            ov_input_names_cache[key] = ov_input_names;
            ov_output_names_cache[key] = ov_output_names;
        }
    }

    if (is_prefill) {
        stage = "set_inputs_prefill";
        auto inp_len = inp_pos->ne[0];
        for (int chunk_index = 0; chunk_index * prefill_chunk_size < inp_len; chunk_index++) {
            for (size_t i = 0; i < ov_input_names.size(); i++) {
                auto param_name = ov_input_names[i];
                auto input_tensor = get_ov_input_tensor_static_prefill(ggml_decoder, param_name, chunk_index);
                infer_request->set_input_tensor(i, input_tensor);

                if (getenv("GGML_OPENVINO_DEBUG_INPUT")) {
                    const auto input_tensor = infer_request->get_input_tensor(i);
                    print_input_tensor_info(param_name, input_tensor);
                }
            }

            stage = "set_outputs_prefill";
            for (size_t i = 0; i < ov_output_names.size(); i++) {
                auto output_tensor = get_ov_output_tensor(ggml_decoder, ov_output_names[i]);
                infer_request->set_output_tensor(i, output_tensor);
            }

            stage = "infer_prefill";
            infer_request->infer();

            if (getenv("GGML_OPENVINO_DEBUG_OUTPUT")) {
                for (size_t i = 0; i < ov_output_names.size(); i++) {
                    const auto output_tensor = infer_request->get_output_tensor(i);
                    print_output_tensor_info(ov_output_names[i], output_tensor, output_tensor.data());
                }
            }
        }
        infer_end_time = ggml_time_us();
    } else {
        stage = "set_inputs_decode";
        for (size_t i = 0; i < ov_input_names.size(); i++) {
            auto param_name = ov_input_names[i];
            auto input_tensor = get_ov_input_tensor_static_decode(ggml_decoder, param_name);
            infer_request->set_input_tensor(i, input_tensor);

            if (getenv("GGML_OPENVINO_DEBUG_INPUT")) {
                const auto input_tensor = infer_request->get_input_tensor(i);
                print_input_tensor_info(param_name, input_tensor);
            }
        }

        stage = "set_outputs_decode";
        for (size_t i = 0; i < ov_output_names.size(); i++) {
            auto output_tensor = get_ov_output_tensor(ggml_decoder, ov_output_names[i]);
            infer_request->set_output_tensor(i, output_tensor);
        }

        stage = "infer_decode";
        infer_request->infer();
        infer_end_time = ggml_time_us();

        if (getenv("GGML_OPENVINO_DEBUG_OUTPUT")) {
            for (size_t i = 0; i < ov_output_names.size(); i++) {
                const auto output_tensor = infer_request->get_output_tensor(i);
                print_output_tensor_info(ov_output_names[i], output_tensor, output_tensor.data());
            }
        }
    }

    if (getenv("GGML_OPENVINO_PROFILING")) {
        GGML_LOG_INFO("\nGGML OpenVINO Backend: \n");
        GGML_LOG_INFO("  - Graph decoder Time: %ld ms \n", (decoder_end_time - start_time) / 1000);
        if (!cache_hit) {
            GGML_LOG_INFO("  - Graph conversion Time: %ld ms \n", (conversion_end_time - decoder_end_time) / 1000);
            GGML_LOG_INFO("  - Graph compile Time: %ld ms \n", (compile_end_time - conversion_end_time) / 1000);
        }
        GGML_LOG_INFO("  - Graph Inference Time: %ld ms \n", (infer_end_time - compile_end_time) / 1000);
    }

    return GGML_STATUS_SUCCESS;
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("GGML OpenVINO Backend: static compute exception (stage=%s n_nodes=%zu first=%s last=%s): %s\n",
                       stage, key.n_nodes, key.first_node_name.c_str(), key.last_node_name.c_str(), e.what());
    } catch (...) {
        GGML_LOG_ERROR("GGML OpenVINO Backend: static compute unknown exception (stage=%s n_nodes=%zu first=%s last=%s)\n",
                       stage, key.n_nodes, key.first_node_name.c_str(), key.last_node_name.c_str());
    }
    return GGML_STATUS_FAILED;
}

ov::AnyMap get_ov_compile_config(const std::string & device) {
    ov::AnyMap config;
    auto * cache_dir = getenv("GGML_OPENVINO_CACHE_DIR");
    if (device == "NPU") {
        config = {
            {"NPU_COMPILER_DYNAMIC_QUANTIZATION", "YES"   },
            {"NPU_USE_NPUW",                      "YES"   },
            {"NPUW_DEVICES",                      "NPU"   },
            {"NPUW_FOLD",                         "YES"   },
            {"NPUW_WEIGHTS_BANK",                 "shared"},
            {"NPUW_FUNCALL_FOR_ALL",              "YES"   },
            {"NPUW_FUNCALL_ASYNC",                "YES"   },
            {"NPUW_DQ",                           "YES"   },
            {"NPUW_DQ_FULL",                      "NO"    },
        };
        if (cache_dir) {
            config["NPUW_CACHE_DIR"] = cache_dir;
        }
    } else if (cache_dir) {
        core.set_property(ov::cache_dir(cache_dir));
    }
    return config;
}

std::map<ggml_type, ExtraQuantType> get_types_to_requant(const std::string & device) {
    if (device == "NPU") {
        return {
            {GGML_TYPE_Q4_0, ExtraQuantType::Q4_0_128},
            {GGML_TYPE_Q4_1, ExtraQuantType::Q4_0_128},
            {GGML_TYPE_Q4_K, ExtraQuantType::Q4_0_128},
            {GGML_TYPE_Q6_K, ExtraQuantType::F16     },
            {GGML_TYPE_Q5_K, ExtraQuantType::F16     },
        };
    }
    return {};
}

bool is_naive(ggml_cgraph * cgraph) {
    constexpr int naive_graph_size_threshold = 20;
    return cgraph->n_nodes < naive_graph_size_threshold;
}

static void copy_ov_tensor_to_ggml(const ov::Tensor & src, ggml_tensor * dst, const std::string & name) {
    if (dst == nullptr) {
        return;
    }

    if (ggml_is_empty(dst)) {
        return;
    }

    const size_t n_src = src.get_size();
    const size_t n_dst = (size_t) ggml_nelements(dst);
    if (n_src != n_dst) {
        GGML_LOG_ERROR("GGML OpenVINO Backend: output '%s' element count mismatch: ov=%zu ggml=%zu\n",
                       name.c_str(), n_src, n_dst);
        throw std::runtime_error("OpenVINO output element count mismatch for " + name);
    }

    {
        // Guard against writing past the underlying allocation for VIEW tensors.
        const ggml_tensor * root = dst;
        size_t root_off = 0;
        while (root->view_src != nullptr) {
            root_off += root->view_offs;
            root = root->view_src;
        }

        // Avoid writing into buffers that are flagged as weights (often mmapped / read-only).
        if (root->buffer != nullptr && ggml_backend_buffer_get_usage(root->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            if (dst->op == GGML_OP_VIEW || dst->view_src != nullptr) {
                return;
            }
            GGML_LOG_ERROR("GGML OpenVINO Backend: output '%s' targets weights buffer (dst=%s root=%s)\n",
                           name.c_str(), dst->name, root->name);
            throw std::runtime_error("OpenVINO attempted to write into weights buffer for " + name);
        }

        const size_t type_size = ggml_type_size(dst->type);
        const size_t max_off =
            ((size_t) (dst->ne[0] > 0 ? (dst->ne[0] - 1) : 0) * dst->nb[0]) +
            ((size_t) (dst->ne[1] > 0 ? (dst->ne[1] - 1) : 0) * dst->nb[1]) +
            ((size_t) (dst->ne[2] > 0 ? (dst->ne[2] - 1) : 0) * dst->nb[2]) +
            ((size_t) (dst->ne[3] > 0 ? (dst->ne[3] - 1) : 0) * dst->nb[3]);
        const size_t needed = root_off + max_off + type_size;
        const size_t root_bytes = ggml_nbytes(root);
        if (needed > root_bytes) {
            GGML_LOG_ERROR("GGML OpenVINO Backend: output '%s' write would exceed allocation: needed=%zu root_bytes=%zu (dst=%s root=%s)\n",
                           name.c_str(), needed, root_bytes, dst->name, root->name);
            throw std::runtime_error("OpenVINO output write exceeds allocation for " + name);
        }
    }

    const ov::element::Type src_type = src.get_element_type();
    const bool dst_contiguous = ggml_is_contiguous(dst);

    if (dst->type == GGML_TYPE_F32) {
        if (src_type == ov::element::f32) {
            if (dst_contiguous) {
                std::memcpy(dst->data, src.data(), ggml_nbytes(dst));
                return;
            }

            const auto * src_data = src.data<const float>();
            for (size_t i3 = 0; i3 < (size_t) dst->ne[3]; ++i3) {
                for (size_t i2 = 0; i2 < (size_t) dst->ne[2]; ++i2) {
                    for (size_t i1 = 0; i1 < (size_t) dst->ne[1]; ++i1) {
                        for (size_t i0 = 0; i0 < (size_t) dst->ne[0]; ++i0) {
                            const size_t src_idx =
                                (((i3 * (size_t) dst->ne[2] + i2) * (size_t) dst->ne[1] + i1) * (size_t) dst->ne[0] +
                                 i0);
                            const size_t dst_off = i3 * dst->nb[3] + i2 * dst->nb[2] + i1 * dst->nb[1] + i0 * dst->nb[0];
                            *(float *) ((char *) dst->data + dst_off) = src_data[src_idx];
                        }
                    }
                }
            }
            return;
        }
        if (src_type == ov::element::f16) {
            const auto * src_data = src.data<const ov::float16>();
            if (dst_contiguous) {
                auto * dst_data = (float *) dst->data;
                for (size_t i = 0; i < n_dst; ++i) {
                    dst_data[i] = (float) src_data[i];
                }
                return;
            }

            for (size_t i3 = 0; i3 < (size_t) dst->ne[3]; ++i3) {
                for (size_t i2 = 0; i2 < (size_t) dst->ne[2]; ++i2) {
                    for (size_t i1 = 0; i1 < (size_t) dst->ne[1]; ++i1) {
                        for (size_t i0 = 0; i0 < (size_t) dst->ne[0]; ++i0) {
                            const size_t src_idx =
                                (((i3 * (size_t) dst->ne[2] + i2) * (size_t) dst->ne[1] + i1) * (size_t) dst->ne[0] +
                                 i0);
                            const size_t dst_off = i3 * dst->nb[3] + i2 * dst->nb[2] + i1 * dst->nb[1] + i0 * dst->nb[0];
                            *(float *) ((char *) dst->data + dst_off) = (float) src_data[src_idx];
                        }
                    }
                }
            }
            return;
        }
    }

    if (dst->type == GGML_TYPE_F16) {
        if (src_type == ov::element::f16) {
            if (dst_contiguous) {
                std::memcpy(dst->data, src.data(), ggml_nbytes(dst));
                return;
            }

            const auto * src_data = src.data<const ov::float16>();
            for (size_t i3 = 0; i3 < (size_t) dst->ne[3]; ++i3) {
                for (size_t i2 = 0; i2 < (size_t) dst->ne[2]; ++i2) {
                    for (size_t i1 = 0; i1 < (size_t) dst->ne[1]; ++i1) {
                        for (size_t i0 = 0; i0 < (size_t) dst->ne[0]; ++i0) {
                            const size_t src_idx =
                                (((i3 * (size_t) dst->ne[2] + i2) * (size_t) dst->ne[1] + i1) * (size_t) dst->ne[0] +
                                 i0);
                            const size_t dst_off = i3 * dst->nb[3] + i2 * dst->nb[2] + i1 * dst->nb[1] + i0 * dst->nb[0];
                            *(ggml_fp16_t *) ((char *) dst->data + dst_off) = ggml_fp32_to_fp16((float) src_data[src_idx]);
                        }
                    }
                }
            }
            return;
        }
        if (src_type == ov::element::f32) {
            if (dst_contiguous) {
                ggml_fp32_to_fp16_row(src.data<const float>(), (ggml_fp16_t *) dst->data, (int64_t) n_dst);
                return;
            }

            const auto * src_data = src.data<const float>();
            for (size_t i3 = 0; i3 < (size_t) dst->ne[3]; ++i3) {
                for (size_t i2 = 0; i2 < (size_t) dst->ne[2]; ++i2) {
                    for (size_t i1 = 0; i1 < (size_t) dst->ne[1]; ++i1) {
                        for (size_t i0 = 0; i0 < (size_t) dst->ne[0]; ++i0) {
                            const size_t src_idx =
                                (((i3 * (size_t) dst->ne[2] + i2) * (size_t) dst->ne[1] + i1) * (size_t) dst->ne[0] +
                                 i0);
                            const size_t dst_off = i3 * dst->nb[3] + i2 * dst->nb[2] + i1 * dst->nb[1] + i0 * dst->nb[0];
                            *(ggml_fp16_t *) ((char *) dst->data + dst_off) = ggml_fp32_to_fp16(src_data[src_idx]);
                        }
                    }
                }
            }
            return;
        }
    }

    if (dst->type == GGML_TYPE_I32 && src_type == ov::element::i32) {
        if (dst_contiguous) {
            std::memcpy(dst->data, src.data(), ggml_nbytes(dst));
            return;
        }

        const auto * src_data = src.data<const int32_t>();
        for (size_t i3 = 0; i3 < (size_t) dst->ne[3]; ++i3) {
            for (size_t i2 = 0; i2 < (size_t) dst->ne[2]; ++i2) {
                for (size_t i1 = 0; i1 < (size_t) dst->ne[1]; ++i1) {
                    for (size_t i0 = 0; i0 < (size_t) dst->ne[0]; ++i0) {
                        const size_t src_idx =
                            (((i3 * (size_t) dst->ne[2] + i2) * (size_t) dst->ne[1] + i1) * (size_t) dst->ne[0] + i0);
                        const size_t dst_off = i3 * dst->nb[3] + i2 * dst->nb[2] + i1 * dst->nb[1] + i0 * dst->nb[0];
                        *(int32_t *) ((char *) dst->data + dst_off) = src_data[src_idx];
                    }
                }
            }
        }
        return;
    }

    if (dst->type == GGML_TYPE_I64 && src_type == ov::element::i64) {
        if (dst_contiguous) {
            std::memcpy(dst->data, src.data(), ggml_nbytes(dst));
            return;
        }

        const auto * src_data = src.data<const int64_t>();
        for (size_t i3 = 0; i3 < (size_t) dst->ne[3]; ++i3) {
            for (size_t i2 = 0; i2 < (size_t) dst->ne[2]; ++i2) {
                for (size_t i1 = 0; i1 < (size_t) dst->ne[1]; ++i1) {
                    for (size_t i0 = 0; i0 < (size_t) dst->ne[0]; ++i0) {
                        const size_t src_idx =
                            (((i3 * (size_t) dst->ne[2] + i2) * (size_t) dst->ne[1] + i1) * (size_t) dst->ne[0] + i0);
                        const size_t dst_off = i3 * dst->nb[3] + i2 * dst->nb[2] + i1 * dst->nb[1] + i0 * dst->nb[0];
                        *(int64_t *) ((char *) dst->data + dst_off) = src_data[src_idx];
                    }
                }
            }
        }
        return;
    }

    GGML_LOG_ERROR("GGML OpenVINO Backend: unsupported output type conversion for '%s': ov=%s ggml=%s\n",
                   name.c_str(), src_type.get_type_name().c_str(), ggml_type_name(dst->type));
    throw std::runtime_error("OpenVINO output type mismatch for " + name);
}

static ov::Tensor convert_ov_tensor_to_type(const ov::Tensor & src, const ov::element::Type & dst_type, const std::string & name) {
    const auto src_type = src.get_element_type();
    if (src_type == dst_type) {
        return src;
    }

    ov::Tensor dst(dst_type, src.get_shape());
    const size_t n = src.get_size();

    if (src_type == ov::element::f32 && dst_type == ov::element::f16) {
        const float * src_data = src.data<const float>();
        auto * dst_data = dst.data<ov::float16>();
        for (size_t i = 0; i < n; ++i) {
            dst_data[i] = ov::float16(src_data[i]);
        }
        return dst;
    }

    if (src_type == ov::element::f16 && dst_type == ov::element::f32) {
        const auto * src_data = src.data<const ov::float16>();
        auto * dst_data = dst.data<float>();
        for (size_t i = 0; i < n; ++i) {
            dst_data[i] = (float) src_data[i];
        }
        return dst;
    }

    GGML_LOG_ERROR("GGML OpenVINO Backend: unsupported input type conversion for '%s': ov=%s -> %s\n",
                   name.c_str(), src_type.get_type_name().c_str(), dst_type.get_type_name().c_str());
    throw std::runtime_error("OpenVINO input type mismatch for " + name);
}

enum ggml_status naive_compute(ggml_cgraph * cgraph,
                               ov::Core & core,
                               const std::string & device,
                               const ov::AnyMap & config) {
    const char * stage = "init";
    try {
    if (cgraph->n_nodes == 1 && (cgraph->nodes[0]->op == GGML_OP_NONE || cgraph->nodes[0]->op == GGML_OP_VIEW)) {
        return GGML_STATUS_SUCCESS;
    }
    if (cgraph->nodes[0]->op == GGML_OP_FLASH_ATTN_EXT) {
        return GGML_STATUS_FAILED;
    }

    stage = "create_weight_nodes";
    auto model_weights = GgmlOvDecoder::create_weight_nodes(cgraph);
    stage = "create_decoder";
    auto decoder = std::make_shared<GgmlOvDecoder>(cgraph, model_weights);
    stage = "create_input_model";
    auto input_model = std::make_shared<ov::frontend::ggml::InputModel>(decoder);
    auto naive = true;
    stage = "frontend_convert";
    auto model = ov::frontend::ggml::FrontEnd::convert(input_model, naive);
    if (getenv("GGML_OPENVINO_DUMP_IR")) {
        ov::serialize(model, "IR_naive.xml");
    }
    stage = "compile_model";
    auto compiled_model = core.compile_model(model, device, config);
    stage = "create_infer_request";
    auto infer_request = compiled_model.create_infer_request();

    stage = "set_inputs";
    auto ov_params = model->get_parameters();
    for (size_t i = 0; i < ov_params.size(); i++) {
        auto param_name = ov_params[i]->get_friendly_name();
        auto input_tensor = get_ov_input_tensor(decoder, param_name);
        input_tensor = convert_ov_tensor_to_type(input_tensor, compiled_model.input(i).get_element_type(), param_name);
        infer_request.set_input_tensor(i, input_tensor);
    }

    stage = "prepare_outputs";
    auto ov_results = model->get_results();
    std::vector<std::string> ov_result_names;
    ov_result_names.reserve(ov_results.size());
    for (size_t i = 0; i < ov_results.size(); i++) {
        ov_result_names.push_back(ov_results[i]->get_friendly_name());
    }

    for (size_t i = 0; i < ov_result_names.size(); i++) {
        const auto expected_type = compiled_model.output(i).get_element_type();
        const auto current = infer_request.get_output_tensor(i);
        if (current.get_element_type() == expected_type) {
            continue;
        }
        infer_request.set_output_tensor(i, ov::Tensor(expected_type, current.get_shape()));
    }

    stage = "infer";
    infer_request.infer();

    stage = "copy_outputs";
    for (size_t i = 0; i < ov_result_names.size(); i++) {
        const auto & result_name = ov_result_names[i];

        const auto & outputs = decoder->get_model_outputs();
        auto it = outputs.find(result_name);
        if (it == outputs.end()) {
            GGML_LOG_ERROR("GGML OpenVINO Backend: output '%s' not found; available outputs: ", result_name.c_str());
            for (const auto & kv : outputs) {
                GGML_LOG_ERROR("%s ", kv.first.c_str());
            }
            GGML_LOG_ERROR("\n");
            throw std::out_of_range("missing model output " + result_name);
        }

        copy_ov_tensor_to_ggml(infer_request.get_output_tensor(i), it->second, result_name);
    }
    return GGML_STATUS_SUCCESS;
    } catch (const std::exception & e) {
        const auto key = compute_graph_key(cgraph);
        GGML_LOG_ERROR("GGML OpenVINO Backend: naive compute exception (device=%s stage=%s n_nodes=%d first=%s last=%s): %s\n",
                       device.c_str(), stage, cgraph->n_nodes, key.first_node_name.c_str(), key.last_node_name.c_str(),
                       e.what());
    } catch (...) {
        const auto key = compute_graph_key(cgraph);
        GGML_LOG_ERROR("GGML OpenVINO Backend: naive compute unknown exception (device=%s stage=%s n_nodes=%d first=%s last=%s)\n",
                       device.c_str(), stage, cgraph->n_nodes, key.first_node_name.c_str(), key.last_node_name.c_str());
    }
    return GGML_STATUS_FAILED;
}

namespace {
ov::Tensor convert_ggml_input_to_ov(std::shared_ptr<GgmlOvDecoder> ggml_decoder, const std::string & name) {
    const auto * ggml_tensor = ggml_decoder->get_input_ggml_tensor(name);
    auto * input_data = ggml_tensor->data;
    ov::Shape input_shape;
    // Respect VIEW shapes for model inputs: many graphs pass reshaped views (e.g. [64,1,33,1]) as inputs.
    // Using view_src here causes parameter/tensor shape mismatches at inference time.
    input_shape = ggml_decoder->get_shape(ggml_tensor);
    auto input_tensor = ov::Tensor(ggml_decoder->get_ov_type(ggml_tensor), input_shape, input_data);
    return input_tensor;
}
}  // namespace

ov::Tensor get_ov_input_tensor(std::shared_ptr<GgmlOvDecoder> ggml_decoder, const std::string & param_name) {
    ov::Tensor input_tensor;
    if (ggml_decoder->get_model_extra_inputs().find(param_name) != ggml_decoder->get_model_extra_inputs().end()) {
        const auto & extra_inputs = ggml_decoder->get_model_extra_input_values();
        auto it = extra_inputs.find(param_name);
        if (it == extra_inputs.end()) {
            GGML_LOG_ERROR("GGML OpenVINO Backend: extra input '%s' requested but not found; available extra inputs: ",
                           param_name.c_str());
            for (const auto & kv : extra_inputs) {
                GGML_LOG_ERROR("%s ", kv.first.c_str());
            }
            GGML_LOG_ERROR("\n");
            throw std::out_of_range("missing extra input " + param_name);
        }
        input_tensor = *it->second;
    } else {
        input_tensor = convert_ggml_input_to_ov(ggml_decoder, param_name);
    }
    return input_tensor;
}

ov::Tensor get_ov_input_tensor_static_decode(std::shared_ptr<GgmlOvDecoder> ggml_decoder,
                                             const std::string & param_name) {
    const auto * ggml_tensor = ggml_decoder->get_input_ggml_tensor(param_name);
    const auto * op = ggml_decoder->get_tensor_used_op(ggml_tensor);

    if (param_name == "inp_pos" || param_name == "inp_tokens" ||
        (op->op == GGML_OP_SET_ROWS && op->src[1] == ggml_tensor)) {
        assert(ggml_tensor->ne[0] == 1);
        ov::Shape input_shape = {1, 1, 1, 1};
        ov::Tensor input_tensor(ggml_decoder->get_ov_type(ggml_tensor), input_shape);
        if (ggml_tensor->type == GGML_TYPE_I32) {
            *input_tensor.data<int32_t>() = *((int32_t *) ggml_tensor->data);
        } else if (ggml_tensor->type == GGML_TYPE_I64) {
            *input_tensor.data<int64_t>() = *((int64_t *) ggml_tensor->data);
        } else {
            throw std::runtime_error("Unexpected tensor type for " + param_name);
        }
        return input_tensor;
    }

    if (param_name == "inp_out_ids") {
        ov::Shape input_shape = {1, 1, 1, 1};
        ov::Tensor input_tensor(ggml_decoder->get_ov_type(ggml_tensor), input_shape);
        int32_t inp_out_id = *((int32_t *) ggml_tensor->data);
        assert(ggml_tensor->ne[0] == 1);
        assert(inp_out_id == 0);
        *input_tensor.data<int32_t>() = inp_out_id;
        return input_tensor;
    }

    if (param_name.find("KQ_mask") == 0) {
        size_t context_size = ggml_decoder->get_ctx_size();
        std::vector<float> padded_data = pad_input<float>(ggml_tensor, 1, context_size, -INFINITY);
        ov::Tensor input_tensor(ov::element::f32, ov::Shape{1, 1, 1, context_size});
        auto * data_ptr = input_tensor.data<float>();
        std::copy(padded_data.begin(), padded_data.begin() + context_size, data_ptr);
        return input_tensor;
    }

    return get_ov_input_tensor(ggml_decoder, param_name);
}

ov::Tensor get_ov_input_tensor_static_prefill(std::shared_ptr<GgmlOvDecoder> ggml_decoder,
                                              const std::string & param_name,
                                              int chunk_index) {
    const auto * ggml_tensor = ggml_decoder->get_input_ggml_tensor(param_name);
    const auto * op = ggml_decoder->get_tensor_used_op(ggml_tensor);

    const size_t input_len = ggml_decoder->get_input_len();
    const size_t chunk_size = ggml_decoder->m_prefill_chunk_size;
    const size_t chunk_valid_size = std::min(chunk_size, input_len - chunk_index * chunk_size);
    const size_t chunk_pad_size = chunk_size - chunk_valid_size;

    if (param_name == "inp_pos" || param_name == "inp_tokens" ||
        (op->op == GGML_OP_SET_ROWS && op->src[1] == ggml_tensor)) {
        ov::Shape input_shape = {1, 1, 1, chunk_size};
        ov::Tensor input_tensor(ggml_decoder->get_ov_type(ggml_tensor), input_shape);
        // copy the chunk_index-th chunk from ggml_tensor
        size_t element_size = ggml_type_size(ggml_tensor->type);
        void * input_data = (char *) ggml_tensor->data + chunk_index * chunk_size * element_size;
        std::memcpy(input_tensor.data(), input_data, chunk_valid_size * element_size);
        // pad the rest with last_value + 1, so that kv's of padded positions are inserted
        // to the next row after the valids row in the kvcache
        if (chunk_pad_size > 0) {
            if (ggml_tensor->type == GGML_TYPE_I32) {
                int32_t last_value =
                    *((int32_t *) ggml_tensor->data + (chunk_index * chunk_size + chunk_valid_size - 1));
                int32_t * output_data = input_tensor.data<int32_t>();
                std::fill(output_data + chunk_valid_size, output_data + chunk_size, last_value + 1);
            } else if (ggml_tensor->type == GGML_TYPE_I64) {
                int64_t last_value =
                    *((int64_t *) ggml_tensor->data + (chunk_index * chunk_size + chunk_valid_size - 1));
                int64_t * output_data = input_tensor.data<int64_t>();
                std::fill(output_data + chunk_valid_size, output_data + chunk_size, last_value + 1);
            } else {
                throw std::runtime_error("Unexpected tensor type for " + param_name);
            }
        }
        return input_tensor;
    }

    if (param_name == "inp_out_ids") {
        size_t output_len = ggml_decoder->get_compute_params().output_len;
        ov::Shape input_shape = {1, 1, 1, output_len};
        ov::Tensor input_tensor(ggml_decoder->get_ov_type(ggml_tensor), input_shape);
        if (ggml_tensor->ne[0] == 0) {
            *input_tensor.data<int32_t>() = 0;
        } else {
            auto * data_addr = input_tensor.data<int32_t>();
            for (size_t i = 0; i < output_len; i++) {
                data_addr[i] = ((int32_t *) ggml_tensor->data)[i] % chunk_size;
            }
        }
        return input_tensor;
    }

    if (param_name.find("KQ_mask") == 0) {
        size_t cols = ggml_tensor->ne[0];
        size_t rows = ggml_tensor->ne[1];
        float * ggml_data = (float *) ggml_tensor->data + chunk_index * chunk_size * cols;
        size_t chunk_valid_rows = std::min(chunk_size, rows - chunk_index * chunk_size);
        size_t context_size = ggml_decoder->get_ctx_size();
        std::vector<float> padded_data =
            pad_input<float>(ggml_data, chunk_valid_rows, cols, chunk_size, context_size, -INFINITY);
        set_zero_diagonal(padded_data, chunk_size, context_size);
        ov::Tensor input_tensor(ov::element::f32, ov::Shape{1, 1, chunk_size, context_size});
        auto * data_ptr = input_tensor.data<float>();
        std::copy(padded_data.begin(), padded_data.begin() + chunk_size * context_size, data_ptr);
        return input_tensor;
    }

    return get_ov_input_tensor(ggml_decoder, param_name);
}

ov::Tensor get_ov_output_tensor(std::shared_ptr<GgmlOvDecoder> ggml_decoder, const std::string & result_name) {
    const auto & outputs = ggml_decoder->get_model_outputs();
    auto it = outputs.find(result_name);
    if (it == outputs.end()) {
        GGML_LOG_ERROR("GGML OpenVINO Backend: output '%s' not found; available outputs: ", result_name.c_str());
        for (const auto & kv : outputs) {
            GGML_LOG_ERROR("%s ", kv.first.c_str());
        }
        GGML_LOG_ERROR("\n");
        throw std::out_of_range("missing model output " + result_name);
    }
    auto * ggml_tensor = it->second;
    auto output_type = ggml_decoder->get_ov_type(ggml_tensor);
    auto output_shape = ggml_decoder->get_shape(ggml_tensor);

    if (ggml_decoder->is_static() && result_name == "result_output" && output_shape[2] == 0) {
        output_shape[2] = 1;
    }
    ov::Tensor output_tensor(output_type, output_shape, ggml_tensor->data);
    return output_tensor;
}

size_t checksum(const void * data, size_t size) {
    const uint8_t * bytes = static_cast<const uint8_t *>(data);
    size_t sum = 0;
    for (size_t i = 0; i < size; ++i) {
        sum += (uint8_t) i;
        sum += bytes[i];
    }
    return sum;
}

void print_input_tensor_info(const std::string & name, const ov::Tensor & tensor) {
    std::cout << "Input name: " << name << ", Input shape: " << tensor.get_shape() << ", Address: " << tensor.data()
              << std::endl;
    switch (tensor.get_element_type()) {
    case ov::element::f32: {
        if (name.find("KQ_mask") == std::string::npos) {
            std::cout << *(tensor.data<float>()) << std::endl;
        } else {
            size_t rows = tensor.get_shape()[2];
            size_t cols = tensor.get_shape()[3];
            auto * data = tensor.data<float>();
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    float val = data[i * cols + j];
                    if (std::isinf(val) && val < 0) {
                        std::cout << std::setw(5) << "-inf";
                    } else {
                        std::cout << std::setw(5) << val;
                    }
                }
                std::cout << std::endl;
            }
        }

        break;
    }
    case ov::element::f16:
        std::cout << *(tensor.data<ov::float16>()) << std::endl;
        break;
    case ov::element::i32:
        for (size_t i = 0; i < tensor.get_size(); ++i) {
            std::cout << tensor.data<int32_t>()[i] << " ";
        }
        std::cout << std::endl;
        break;
    case ov::element::i64:
        for (size_t i = 0; i < tensor.get_size(); ++i) {
            std::cout << tensor.data<int64_t>()[i] << " ";
        }
        std::cout << std::endl;
        break;
    default:
        break;
    }
}

void print_output_tensor_info(const std::string & name, const ov::Tensor & tensor, const void * output_dst) {
    std::cout << "Output name: " << name << ", Output shape: " << tensor.get_shape() << ", Address: " << output_dst
              << std::endl;

    auto print_float_stats = [](const std::string & type_name, size_t size, auto get_value) {
        if (size == 0) {
            return;
        }

        float first = get_value(0);
        float min = first;
        float max = first;
        double sum = first;

        for (size_t i = 1; i < size; ++i) {
            float v = get_value(i);
            if (v < min) {
                min = v;
            }
            if (v > max) {
                max = v;
            }
            sum += v;
        }
        double mean = sum / size;

        std::cout << std::right << std::setw(6) << type_name << std::right << std::setw(12) << "First" << std::setw(12)
                  << "Min" << std::setw(12) << "Max" << std::setw(12) << "Mean" << std::endl;
        std::cout << std::right << std::setw(6) << "" << std::right << std::setw(12) << first << std::setw(12) << min
                  << std::setw(12) << max << std::setw(12) << mean << std::endl;
    };

    switch (tensor.get_element_type()) {
    case ov::element::f32: {
        const float * data = tensor.data<float>();
        size_t size = tensor.get_size();
        print_float_stats("[f32]", size, [data](size_t i) { return data[i]; });
        break;
    }
    case ov::element::f16: {
        const ov::float16 * data = tensor.data<ov::float16>();
        size_t size = tensor.get_size();
        print_float_stats("[f16]", size, [data](size_t i) { return static_cast<float>(data[i]); });
        break;
    }
    default:
        break;
    }
}

void set_zero_diagonal(std::vector<float> & matrix, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; ++i) {
        size_t diag_col = std::min(i, cols - 1);
        matrix[i * cols + diag_col] = 0.0f;
    }
}

const ggml_tensor * get_inp_pos_tensor(ggml_cgraph * cgraph) {
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        auto * op = cgraph->nodes[i];
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            auto * src = op->src[j];
            if (src == nullptr) {
                break;
            }
            if (std::string(src->name) == "inp_pos") {
                return src;
            }
        }
    }
    GGML_LOG_ERROR("get_inp_pos_tensor: inp_pos not found in cgraph");
    throw std::runtime_error("get_inp_pos_tensor: inp_pos not found in cgraph");
}

bool get_is_prefill(const ggml_tensor * inp_pos) {
    return inp_pos->ne[0] > 1;
}

graph_key compute_graph_key(ggml_cgraph * cgraph) {
    graph_key key;
    key.n_nodes = cgraph->n_nodes;

    if (cgraph->n_nodes > 0) {
        key.first_node_name = std::string(cgraph->nodes[0]->name);
        key.last_node_name = std::string(cgraph->nodes[cgraph->n_nodes - 1]->name);
    } else {
        key.first_node_name = "";
        key.last_node_name = "";
    }

    return key;
}

#pragma GCC diagnostic pop
