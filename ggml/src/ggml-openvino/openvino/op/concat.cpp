#include "../node_context.hpp"
#include "../op_table.hpp"
#include "../utils.hpp"

#include <openvino/op/concat.hpp>
#include <openvino/op/convert.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_concat(const NodeContext & context) {
    num_inputs_check(context, 2, 2);

    int32_t * params = context.get_output_op_params();
    const int64_t axis = params ? static_cast<int64_t>(params[0]) : 0;

    auto in0 = context.get_input(0);
    auto in1 = context.get_input(1);
    const auto out_type = context.get_output_type();

    if (in0.get_element_type() != out_type) {
        in0 = std::make_shared<ov::op::v0::Convert>(in0, out_type);
    }
    if (in1.get_element_type() != out_type) {
        in1 = std::make_shared<ov::op::v0::Convert>(in1, out_type);
    }

    auto res = std::make_shared<ov::op::v0::Concat>(OutputVector{in0, in1}, axis);
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov

