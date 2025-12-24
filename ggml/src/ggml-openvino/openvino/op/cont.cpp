
#include "../node_context.hpp"
#include "../op_table.hpp"
#include "../utils.hpp"

#include <climits>
#include <cstdint>
#include <memory>
#include <openvino/op/constant.hpp>
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

    ov::Output<Node> res = context.get_input(0);

    if (op_case == 3) {
        // The input comes from a VIEW that is translated as a no-op; materialize the sliced region.
        res = process_view_input(context, 0);
    }

    // ggml CONT may also change shape (e.g. ggml_cont_2d/3d/4d). Represent it as a reshape.
    const auto dst_ps = context.get_output_shape();
    if (dst_ps.is_static()) {
        const auto dst_shape = dst_ps.to_shape();
        auto dst_shape_const = ov::op::v0::Constant::create(ov::element::i64, {dst_shape.size()}, dst_shape);
        res = std::make_shared<ov::op::v1::Reshape>(res, dst_shape_const, false);
    } else if (dst_ps.rank().is_static()) {
        std::vector<int64_t> pattern;
        pattern.reserve(dst_ps.rank().get_length());
        int n_dynamic = 0;
        for (size_t i = 0; i < (size_t) dst_ps.rank().get_length(); ++i) {
            const auto & d = dst_ps[i];
            if (d.is_static()) {
                pattern.push_back(d.get_length());
            } else {
                pattern.push_back(-1);
                ++n_dynamic;
            }
        }

        if (n_dynamic <= 1) {
            auto dst_shape_const = ov::op::v0::Constant::create(ov::element::i64, {pattern.size()}, pattern);
            res = std::make_shared<ov::op::v1::Reshape>(res, dst_shape_const, false);
        }
    }

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
