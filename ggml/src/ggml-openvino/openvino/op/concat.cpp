#include "../node_context.hpp"
#include "../op_table.hpp"
#include "../utils.hpp"

#include <openvino/op/concat.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/reshape.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_concat(const NodeContext & context) {
    num_inputs_check(context, 2, 2);

    int32_t * params = context.get_output_op_params();
    int64_t axis = params ? static_cast<int64_t>(params[0]) : 0;

    auto in0 = context.get_input(0);
    auto in1 = context.get_input(1);
    const auto out_type = context.get_output_type();

    if (in0.get_element_type() != out_type) {
        in0 = std::make_shared<ov::op::v0::Convert>(in0, out_type);
    }
    if (in1.get_element_type() != out_type) {
        in1 = std::make_shared<ov::op::v0::Convert>(in1, out_type);
    }

    // ggml axes are ordered as [ne0..ne3], while OpenVINO uses [ne3..ne0].
    // Map concat axis from ggml to OpenVINO order when the output rank is known.
    bool axis_mapped = false;
    try {
        const auto out_shape = context.get_output_shape().to_shape();
        if (!out_shape.empty()) {
            const int64_t rank = static_cast<int64_t>(out_shape.size());
            axis = rank - 1 - axis;
            axis_mapped = true;
        }
    } catch (const ov::Exception &) {
    }

    const auto name = context.get_name();
    if (!axis_mapped && name.find("mamba_conv1d_input") != std::string::npos) {
        // Fallback for legacy mamba concat when rank is not available.
        axis = 3;
    }

    auto res = std::make_shared<ov::op::v0::Concat>(OutputVector{in0, in1}, axis);
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
