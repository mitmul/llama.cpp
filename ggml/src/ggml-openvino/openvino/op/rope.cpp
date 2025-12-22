#include "../node_context.hpp"
#include "../op_table.hpp"
#include "../utils.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <openvino/core/node.hpp>
#include <openvino/core/node_output.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/shape_of.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/split.hpp>
#include <openvino/op/subtract.hpp>
#include <openvino/op/unsqueeze.hpp>
#include <vector>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_rope(const NodeContext & context) {
    num_inputs_check(context, 2, 3);

    int op_case = context.get_op_case();

    ov::Output<Node> res;

    auto data_node = context.get_input(0).get_node_shared_ptr();
    auto output_shape = context.get_output_shape().to_shape();
    int32_t * op_params = context.get_output_op_params();

    Output<Node> cos_theta_node;
    Output<Node> sin_theta_node;
    if (context.has_input("rope_cos")) {
        cos_theta_node = context.get_input("rope_cos");
        sin_theta_node = context.get_input("rope_sin");
    } else {
        auto inp_pos = context.get_input(1).get_node_shared_ptr();
        std::shared_ptr<ov::Node> rope_freqs_weight;
        if (context.get_input_size() == 3) {
            rope_freqs_weight = context.get_input(2).get_node_shared_ptr();
        }
        // ggml encodes ROPE params as int32_t[], but some models pass n_dims=0 to mean "use full head size".
        // Our sin/cos generator needs an explicit n_dims, so infer it from the tensor shape when needed.
        int32_t params_local[15];
        std::memcpy(params_local, op_params, sizeof(params_local));
        const int32_t inferred_dims = static_cast<int32_t>(output_shape.empty() ? 0 : output_shape.back());
        if (params_local[1] <= 0) {
            params_local[1] = inferred_dims;
        }
        if (const char * trace = std::getenv("GGML_OPENVINO_TRACE_ROPE"); trace && std::strcmp(trace, "0") != 0) {
            float freq_base;
            float freq_scale;
            float ext_factor;
            float attn_factor;
            float beta_fast;
            float beta_slow;
            std::memcpy(&freq_base, params_local + 5, sizeof(float));
            std::memcpy(&freq_scale, params_local + 6, sizeof(float));
            std::memcpy(&ext_factor, params_local + 7, sizeof(float));
            std::memcpy(&attn_factor, params_local + 8, sizeof(float));
            std::memcpy(&beta_fast, params_local + 9, sizeof(float));
            std::memcpy(&beta_slow, params_local + 10, sizeof(float));
            GGML_LOG_INFO(
                "GGML OpenVINO Backend: ROPE name=%s op_case=%d n_dims=%d inferred=%d mode=%d n_ctx_orig=%d freq_base=%g freq_scale=%g ext_factor=%g attn_factor=%g beta_fast=%g beta_slow=%g out_shape=[%zu,%zu,%zu,%zu]\n",
                context.get_name().c_str(), op_case, params_local[1], inferred_dims, params_local[2], params_local[4],
                freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow,
                output_shape.size() > 0 ? output_shape[0] : 0, output_shape.size() > 1 ? output_shape[1] : 0,
                output_shape.size() > 2 ? output_shape[2] : 0, output_shape.size() > 3 ? output_shape[3] : 0);
        }

        auto sin_cos = make_sin_cos(params_local, inp_pos, rope_freqs_weight);
        sin_theta_node = sin_cos.first;
        cos_theta_node = sin_cos.second;
    }

    if (op_case == 2) {
        // The input comes from a VIEW
        int slice_len = output_shape[2] * output_shape[3];
        data_node = process_view_input(context, 0, slice_len).get_node_shared_ptr();
        auto data_shape = ov::op::v0::Constant::create(
            ov::element::i64, {4}, std::vector<int64_t>{1, -1, (int64_t) output_shape[2], (int64_t) output_shape[3]});
        data_node = std::make_shared<ov::op::v1::Reshape>(data_node, data_shape, false);
    }

    const int mode = op_params[2];
    constexpr int ROPE_TYPE_NEOX = 2;
    constexpr int ROPE_TYPE_NORM = 0;

    if (mode == ROPE_TYPE_NORM) {
        auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        auto two = ov::op::v0::Constant::create(ov::element::i64, {1}, {2});
        auto three = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
        auto end = ov::op::v0::Constant::create(ov::element::i64, {1}, {output_shape[3]});
        auto even_slice = std::make_shared<ov::op::v8::Slice>(data_node, zero, end, two, three);
        auto odd_slice = std::make_shared<ov::op::v8::Slice>(data_node, one, end, two, three);

        Output<Node> first_half =
            std::make_shared<ov::op::v1::Subtract>(std::make_shared<ov::op::v1::Multiply>(even_slice, cos_theta_node),
                                                   std::make_shared<ov::op::v1::Multiply>(odd_slice, sin_theta_node));
        Output<Node> second_half =
            std::make_shared<ov::op::v1::Add>(std::make_shared<ov::op::v1::Multiply>(even_slice, sin_theta_node),
                                              std::make_shared<ov::op::v1::Multiply>(odd_slice, cos_theta_node));

        first_half = std::make_shared<ov::op::v0::Unsqueeze>(first_half,
                                                             ov::op::v0::Constant::create(ov::element::i64, {1}, {4}));
        second_half = std::make_shared<ov::op::v0::Unsqueeze>(second_half,
                                                              ov::op::v0::Constant::create(ov::element::i64, {1}, {4}));
        auto stack = std::make_shared<ov::op::v0::Concat>(OutputVector{first_half, second_half}, 4);

        auto data_shape = ov::op::v0::Constant::create(
            ov::element::i64, {4}, std::vector<int64_t>{1, -1, (int64_t) output_shape[2], (int64_t) output_shape[3]});
        res = std::make_shared<ov::op::v1::Reshape>(stack, data_shape, false);
    } else if (mode == ROPE_TYPE_NEOX) {
        auto data_split = std::make_shared<ov::op::v1::Split>(
            data_node, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {3}), 2);
        Output<Node> slice_data_node_0 = data_split->outputs()[0];
        Output<Node> slice_data_node_1 = data_split->outputs()[1];

        auto first_half_node = std::make_shared<ov::op::v1::Subtract>(
            std::make_shared<ov::op::v1::Multiply>(slice_data_node_0, cos_theta_node),
            std::make_shared<ov::op::v1::Multiply>(slice_data_node_1, sin_theta_node));

        auto second_half_node = std::make_shared<ov::op::v1::Add>(
            std::make_shared<ov::op::v1::Multiply>(slice_data_node_0, sin_theta_node),
            std::make_shared<ov::op::v1::Multiply>(slice_data_node_1, cos_theta_node));

        res = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{first_half_node, second_half_node}, 3);
    }

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
