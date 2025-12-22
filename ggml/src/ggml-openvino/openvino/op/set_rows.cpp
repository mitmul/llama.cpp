#include "../node_context.hpp"
#include "../op_table.hpp"
#include "../utils.hpp"

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <openvino/core/node.hpp>
#include <openvino/core/node_output.hpp>
#include <openvino/frontend/exception.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/scatter_update.hpp>
#include <openvino/op/shape_of.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/squeeze.hpp>
#include <openvino/op/transpose.hpp>
#include <sstream>
#include <vector>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_set_rows(const NodeContext & context) {
    num_inputs_check(context, 3, 3);

    auto data = context.get_input(0);
    auto indices = context.get_input(1);
    auto dst = context.get_input(2);

    if (const char * trace = std::getenv("GGML_OPENVINO_TRACE_SET_ROWS"); trace && std::strcmp(trace, "0") != 0) {
        std::ostringstream oss;
        oss << "GGML OpenVINO Backend: SET_ROWS name=" << context.get_name()
            << " data=" << context.get_input_names()[0] << " data_shape=" << data.get_partial_shape()
            << " indices=" << context.get_input_names()[1] << " indices_shape=" << indices.get_partial_shape()
            << " dst=" << context.get_input_names()[2] << " dst_shape=" << dst.get_partial_shape()
            << " ggml_data_shape=" << context.get_input_shape(0)
            << " ggml_indices_shape=" << context.get_input_shape(1)
            << " ggml_dst_shape=" << context.get_input_shape(2)
            << " ggml_out_shape=" << context.get_output_shape()
            << "\n";
        GGML_LOG_INFO("%s", oss.str().c_str());
    }

    data = std::make_shared<ov::op::v0::Convert>(data, context.get_output_type());

    const auto out_shape = context.get_output_shape().to_shape();
    auto dst_shape = context.get_input_shape(2).to_shape();

    auto ind_squeezed =
        std::make_shared<ov::op::v0::Squeeze>(indices, ov::op::v0::Constant::create(ov::element::i64, {3}, {0, 1, 2}));

    // When updating flattened cache views (e.g. cache_v_l* reshaped to [1, 32768, 1, 1]), ggml uses two patterns:
    // - indices is a vector: scatter individual scalar elements into the flattened buffer
    // - indices is a scalar: update one logical "row" (e.g. [1, 1, 1, head_size]) at a given position
    // The OpenVINO ScatterUpdate expects the updates tensor to match indices rank/length along the scatter axis.
    const size_t indices_len = context.get_input_shape(1).to_shape().back();
    if (indices_len == 1 && dst_shape.size() == 4 && out_shape.size() == 4 && dst_shape[3] == 1 && out_shape[3] > 1) {
        // Prefer row update on the logical destination shape.
        std::vector<int64_t> out_shape_vec(out_shape.begin(), out_shape.end());
        dst = std::make_shared<ov::op::v1::Reshape>(
            dst, ov::op::v0::Constant::create(ov::element::i64, {out_shape_vec.size()}, out_shape_vec), false);
        dst_shape = out_shape;
    }

    auto data_reshaped = std::make_shared<ov::op::v1::Reshape>(
        data,
        ov::op::v0::Constant::create(ov::element::i64, {4},
                                     {(int64_t) 1, (int64_t) 1, (int64_t) -1, (int64_t) dst_shape[3]}),
        false);
    auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {2});

    Output<Node> res = std::make_shared<ov::op::v3::ScatterUpdate>(dst, ind_squeezed, data_reshaped, axes);

    if (auto dst_reshape = std::dynamic_pointer_cast<ov::op::v1::Reshape>(dst.get_node_shared_ptr())) {
        // Fix the case of multiple sequences, reshape back to original shape [1, n_seq, ctx_per_seq, emb]
        // ctx_per_seq is not fixed due to llama-bench compatibility
        auto dst_shape_partial = dst_reshape->get_input_partial_shape(0);
        std::vector<int64_t> dst_shape = {dst_shape_partial[0].get_length(), dst_shape_partial[1].get_length(),
                                          dst_shape_partial[2].is_static() ? dst_shape_partial[2].get_length() : -1,
                                          dst_shape_partial[3].get_length()};
        res = std::make_shared<ov::op::v1::Reshape>(res, ov::op::v0::Constant::create(ov::element::i64, {4}, dst_shape),
                                                    false);
    }
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
