#include "../node_context.hpp"
#include "../op_table.hpp"
#include "../utils.hpp"

#include <climits>
#include <cstdint>
#include <memory>
#include <openvino/core/node.hpp>
#include <openvino/core/node_output.hpp>
#include <openvino/op/broadcast.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/tile.hpp>
#include <openvino/op/transpose.hpp>
#include <openvino/op/unsqueeze.hpp>
#include <openvino/op/util/op_types.hpp>
#include <vector>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_mulmat(const NodeContext & context) {
    num_inputs_check(context, 2, 2);

    int op_case = context.get_op_case();

    ov::Output<Node> res;
    ov::Output<ov::Node> B = context.get_input(0);
    ov::Output<ov::Node> A = context.get_input(1);

    bool transpose_b = true;
    if (op_case == 2) {
        auto b_node = B.get_node_shared_ptr();
        if (b_node && b_node->get_input_size() > 0) {
            B = b_node->input_value(0);
        }
    } else if (op_case == 3) {
        B = process_view_input(context, 0);
        A = process_view_input(context, 1);
    }

    // Some MUL_MAT inputs are ggml VIEW tensors that only reshape (same elements) and are otherwise translated as no-ops.
    // Materialize those shapes here so OpenVINO MatMul sees compatible dimensions (e.g., flattening [128,64] -> 8192).
    auto materialize_reshape_only = [&](const ov::Output<ov::Node> & node, size_t input_index) -> ov::Output<ov::Node> {
        const auto expected_ps = context.get_input_shape(input_index);
        if (!expected_ps.is_static()) {
            return node;
        }
        const auto actual_ps = node.get_partial_shape();
        if (!actual_ps.is_static()) {
            return node;
        }

        const auto expected = expected_ps.to_shape();
        const auto actual = actual_ps.to_shape();
        if (expected == actual) {
            return node;
        }

        auto shape_elems = [](const ov::Shape & shape) -> size_t {
            size_t n = 1;
            for (const auto d : shape) {
                n *= d;
            }
            return n;
        };

        if (shape_elems(expected) != shape_elems(actual)) {
            return node;
        }

        auto expected_const = ov::op::v0::Constant::create(ov::element::i64, {expected.size()}, expected);
        return std::make_shared<ov::op::v1::Reshape>(node, expected_const, false);
    };

    B = materialize_reshape_only(B, 0);
    A = materialize_reshape_only(A, 1);

    if (A.get_element_type() != B.get_element_type()) {
        B = std::make_shared<ov::op::v0::Convert>(B, context.get_input_type(1));
    }

    // Broadcast batch/head dimension for GQA-style MatMuls where inputs differ by an integer factor (e.g. 32 vs 4).
    // The OpenVINO MatMul requires broadcastable batch dims; ggml uses implicit repetition.
    {
        const auto A_ps = A.get_partial_shape();
        const auto B_ps = B.get_partial_shape();
        if (A_ps.rank().is_static() && B_ps.rank().is_static() && A_ps.is_static() && B_ps.is_static() &&
            A_ps.rank().get_length() == 4 && B_ps.rank().get_length() == 4) {
            const auto a = A_ps.to_shape();
            const auto b = B_ps.to_shape();

            // Only handle the common layout: [1, heads, tokens, dim] (or similar with heads in dim 1).
            if (a[0] == b[0] && a[1] != b[1]) {
                const size_t big = std::max(a[1], b[1]);
                const size_t small = std::min(a[1], b[1]);
                if (small > 0 && big % small == 0) {
                    const size_t factor = big / small;
                    auto repeats = ov::op::v0::Constant::create(ov::element::i64, {4},
                                                               std::vector<int64_t>{1, (int64_t) factor, 1, 1});
                    if (a[1] > b[1]) {
                        B = std::make_shared<ov::op::v0::Tile>(B, repeats);
                    } else {
                        A = std::make_shared<ov::op::v0::Tile>(A, repeats);
                    }
                }
            }
        }
    }

    // Choose whether to transpose the RHS based on the actual (post-materialization) shapes.
    // This is needed when backend splitting turns MUL_MAT inputs into Parameters and the original "cont(transpose(..))"
    // producer chain is not available in the OpenVINO graph.
    {
        const auto A_ps = A.get_partial_shape();
        const auto B_ps = B.get_partial_shape();
        if (A_ps.rank().is_static() && B_ps.rank().is_static() && A_ps.is_static() && B_ps.is_static() &&
            A_ps.rank().get_length() >= 2 && B_ps.rank().get_length() >= 2) {
            const auto a = A_ps.to_shape();
            const auto b = B_ps.to_shape();

            const size_t a_k = a.back();
            const size_t b_k_no_transpose = b[b.size() - 2];
            const size_t b_k_transpose = b.back();

            if (a_k == b_k_no_transpose) {
                transpose_b = false;
            } else if (a_k == b_k_transpose) {
                transpose_b = true;
            }
        }
    }

    res = std::make_shared<ov::op::v0::MatMul>(A, B, false, transpose_b);

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
