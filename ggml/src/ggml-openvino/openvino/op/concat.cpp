#include "../node_context.hpp"
#include "../op_table.hpp"
#include "../utils.hpp"

#include <cstdint>
#include <cstring>
#include <memory>
#include <openvino/core/node.hpp>
#include <openvino/core/node_output.hpp>
#include <openvino/frontend/exception.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/convert.hpp>
#include <vector>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_concat(const NodeContext & context) {
    num_inputs_check(context, 2, 2);

    auto input0 = context.get_input(0);
    auto input1 = context.get_input(1);

    const auto out_type = context.get_output_type();
    if (input0.get_element_type() != out_type) {
        input0 = std::make_shared<ov::op::v0::Convert>(input0, out_type);
    }
    if (input1.get_element_type() != out_type) {
        input1 = std::make_shared<ov::op::v0::Convert>(input1, out_type);
    }

    int32_t dim = 0;
    memcpy(&dim, context.get_output_op_params(), sizeof(int32_t));
    auto out_rank = context.get_output_shape().rank();
    FRONT_END_OP_CONVERSION_CHECK(out_rank.is_static(), "CONCAT requires static rank");
    const int64_t rank = out_rank.get_length();
    const int64_t axis = rank - 1 - static_cast<int64_t>(dim);
    FRONT_END_OP_CONVERSION_CHECK(axis >= 0 && axis < rank, "CONCAT axis out of range");

    auto res = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{input0, input1}, axis);

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
