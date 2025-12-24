#include "../node_context.hpp"
#include "../op_table.hpp"
#include "../utils.hpp"

#include <cstdint>
#include <memory>
#include <openvino/core/node.hpp>
#include <openvino/core/node_output.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/group_conv.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/transpose.hpp>
#include <vector>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_ssm_conv(const NodeContext & context) {
    num_inputs_check(context, 2, 2);

    auto src = context.get_input(0);
    auto weights = context.get_input(1);

    const auto out_type = context.get_output_type();
    if (src.get_element_type() != out_type) {
        src = std::make_shared<ov::op::v0::Convert>(src, out_type);
    }
    if (weights.get_element_type() != out_type) {
        weights = std::make_shared<ov::op::v0::Convert>(weights, out_type);
    }

    const auto weight_shape = context.get_input_shape(1).to_shape();
    const int64_t d_inner = static_cast<int64_t>(weight_shape[2]);
    const int64_t d_conv = static_cast<int64_t>(weight_shape[3]);

    // GGML uses reversed dimensions in OpenVINO; transpose to NCHW for depthwise convolution.
    auto input_perm = ov::op::v0::Constant::create(ov::element::i64, {4}, {1, 2, 0, 3});
    auto src_transposed = std::make_shared<ov::op::v1::Transpose>(src, input_perm);

    const std::vector<int64_t> weights_shape_values = {d_inner, 1, 1, 1, d_conv};
    auto weights_shape =
        ov::op::v0::Constant::create(ov::element::i64, {weights_shape_values.size()}, weights_shape_values);
    auto weights_reshaped = std::make_shared<ov::op::v1::Reshape>(weights, weights_shape, false);

    auto conv = std::make_shared<ov::op::v1::GroupConvolution>(
        src_transposed,
        weights_reshaped,
        ov::Strides{1, 1},
        ov::CoordinateDiff{0, 0},
        ov::CoordinateDiff{0, 0},
        ov::Strides{1, 1});

    // Restore GGML dimension order.
    auto output_perm = ov::op::v0::Constant::create(ov::element::i64, {4}, {2, 0, 3, 1});
    auto res = std::make_shared<ov::op::v1::Transpose>(conv, output_perm);

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
