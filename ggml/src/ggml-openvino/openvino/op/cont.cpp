
#include "../node_context.hpp"
#include "../op_table.hpp"
#include "../utils.hpp"

#include <climits>
#include <cstdint>
#include <memory>
#include <openvino/op/reshape.hpp>
#include <openvino/op/slice.hpp>
#include <vector>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_cont(const NodeContext & context) {
    num_inputs_check(context, 1, 1);

    int op_case = context.get_op_case();
    FRONT_END_CHECK_IMPLEMENTED(op_case == 1 || op_case == 2 || op_case == 3, "Unsupported CONT case");

    auto src_shape = context.get_input_shape(0).to_shape();
    auto dst_shape = context.get_output_shape().to_shape();
    ov::Output<Node> res;

    if (op_case == 1) {
        // The input comes from a PERMUTE
        std::vector<int64_t> target_shape(dst_shape.begin(), dst_shape.end());
        res = std::make_shared<ov::op::v1::Reshape>(
            context.get_input(0),
            ov::op::v0::Constant::create(ov::element::i64, {target_shape.size()}, target_shape),
            false);
    } else if (op_case == 2) {
        // The input comes from a TRANSPOSE
        return {context.get_input(0)};
    } else {
        // The input comes from a VIEW
        auto input = context.get_input(0);
        auto out_shape = context.get_output_shape().to_shape();
        if (!out_shape.empty()) {
            const auto in_shape = context.get_input_shape(0).to_shape();
            auto numel = [](const std::vector<size_t> & dims) {
                size_t total = 1;
                for (auto d : dims) {
                    total *= d;
                }
                return total;
            };
            if (numel(in_shape) == numel(out_shape)) {
                std::vector<int64_t> target_shape(out_shape.begin(), out_shape.end());
                auto shape_const =
                    ov::op::v0::Constant::create(ov::element::i64, {target_shape.size()}, target_shape);
                res = std::make_shared<ov::op::v1::Reshape>(input, shape_const, false);
            } else {
                res = process_view_input(context, 0);
            }
        } else {
            res = process_view_input(context, 0);
        }
    }

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
