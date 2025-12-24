#include "../node_context.hpp"
#include "../op_table.hpp"
#include "../utils.hpp"

#include <cstdint>
#include <memory>
#include <openvino/core/model.hpp>
#include <openvino/core/node.hpp>
#include <openvino/core/node_output.hpp>
#include <openvino/frontend/exception.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/exp.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/reduce_sum.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/result.hpp>
#include <openvino/op/softplus.hpp>
#include <openvino/op/squeeze.hpp>
#include <openvino/op/tensor_iterator.hpp>
#include <openvino/op/transpose.hpp>
#include <openvino/op/unsqueeze.hpp>
#include <vector>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_ssm_scan(const NodeContext & context) {
    num_inputs_check(context, 7, 7);

    auto s = context.get_input(0);
    auto x = context.get_input(1);
    auto dt = context.get_input(2);
    auto A = context.get_input(3);
    auto B = context.get_input(4);
    auto C = context.get_input(5);
    auto ids = context.get_input(6);

    const auto out_type = context.get_output_type();
    auto cast_to_out = [&](const ov::Output<ov::Node> & node) -> ov::Output<ov::Node> {
        if (node.get_element_type() != out_type) {
            return std::make_shared<ov::op::v0::Convert>(node, out_type);
        }
        return node;
    };

    s = cast_to_out(s);
    x = cast_to_out(x);
    dt = cast_to_out(dt);
    A = cast_to_out(A);
    B = cast_to_out(B);
    C = cast_to_out(C);

    auto shape_elems = [](const ov::Shape & shape) -> size_t {
        size_t n = 1;
        for (const auto d : shape) {
            n *= d;
        }
        return n;
    };

    auto materialize_view_like_input = [&](size_t input_index, ov::Output<ov::Node> node) -> ov::Output<ov::Node> {
        const auto expected_ps = context.get_input_shape(input_index);
        if (!expected_ps.rank().is_static() || expected_ps.rank().get_length() != 4) {
            return node;
        }
        const auto expected = expected_ps.to_shape();

        const auto actual_ps = node.get_partial_shape();
        if (!actual_ps.is_static()) {
            return node;
        }
        const auto actual = actual_ps.to_shape();
        if (actual == expected) {
            return node;
        }

        const size_t expected_n = shape_elems(expected);
        const size_t actual_n = shape_elems(actual);

        auto expected_const = ov::op::v0::Constant::create(ov::element::i64, {expected.size()}, expected);
        if (expected_n == actual_n) {
            return std::make_shared<ov::op::v1::Reshape>(node, expected_const, false);
        }

        // If the ggml input tensor is a VIEW that slices at the lowest dimension, materialize it via Slice + Reshape.
        if (expected_n < actual_n) {
            auto sliced = process_view_input(context, static_cast<int>(input_index));
            return std::make_shared<ov::op::v1::Reshape>(sliced, expected_const, false);
        }

        return node;
    };

    // Several SSM_SCAN inputs are ggml VIEW tensors (slice/reshape) and must be materialized before shape-sensitive ops.
    x = materialize_view_like_input(1, x);
    B = materialize_view_like_input(4, B);
    C = materialize_view_like_input(5, C);

    const std::vector<int64_t> ids_axes_vals = {0, 1, 2};
    auto ids_axes = ov::op::v0::Constant::create(ov::element::i64, {ids_axes_vals.size()}, ids_axes_vals);
    auto ids_squeezed = std::make_shared<ov::op::v0::Squeeze>(ids, ids_axes);
    auto gather_axis0 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, std::vector<int64_t>{0});
    auto s0 = std::make_shared<ov::op::v8::Gather>(s, ids_squeezed, gather_axis0);
    auto s0_out = s0->output(0);

    auto a_shape = context.get_input_shape(3);
    auto b_shape = context.get_input_shape(4);
    FRONT_END_OP_CONVERSION_CHECK(a_shape.rank().is_static() && b_shape.rank().is_static(),
                                  "SSM_SCAN requires static input ranks");
    FRONT_END_OP_CONVERSION_CHECK(a_shape.rank().get_length() == 4 && b_shape.rank().get_length() == 4,
                                  "SSM_SCAN expects 4D inputs");
    FRONT_END_OP_CONVERSION_CHECK(a_shape[2].is_static() && b_shape[2].is_static(),
                                  "SSM_SCAN requires static n_head and n_group");
    const size_t n_head = a_shape[2].get_length();
    const size_t n_group = b_shape[2].get_length();
    FRONT_END_OP_CONVERSION_CHECK(n_group > 0 && n_head % n_group == 0,
                                  "SSM_SCAN requires n_head % n_group == 0");

    const size_t heads_per_group = n_head / n_group;
    std::vector<int64_t> group_indices(n_head);
    for (size_t h = 0; h < n_head; ++h) {
        group_indices[h] = static_cast<int64_t>(h / heads_per_group);
    }
    auto group_indices_const =
        ov::op::v0::Constant::create(ov::element::i64, {group_indices.size()}, group_indices);
    auto gather_axis2 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, std::vector<int64_t>{2});
    auto B_head = std::make_shared<ov::op::v8::Gather>(B, group_indices_const, gather_axis2);
    auto C_head = std::make_shared<ov::op::v8::Gather>(C, group_indices_const, gather_axis2);
    auto B_head_out = B_head->output(0);
    auto C_head_out = C_head->output(0);

    const std::vector<int64_t> dt_perm_vals = {1, 2, 3, 0};
    auto dt_perm = ov::op::v0::Constant::create(ov::element::i64, {dt_perm_vals.size()}, dt_perm_vals);
    auto dt_t = std::make_shared<ov::op::v1::Transpose>(dt, dt_perm);
    auto dt_t_out = dt_t->output(0);

    const std::vector<int64_t> a_perm_vals = {0, 2, 1, 3};
    auto a_perm = ov::op::v0::Constant::create(ov::element::i64, {a_perm_vals.size()}, a_perm_vals);
    auto A_t = std::make_shared<ov::op::v1::Transpose>(A, a_perm);
    auto A_t_out = A_t->output(0);

    auto x_slice_shape = x.get_partial_shape();
    if (x_slice_shape.rank().is_static() && x_slice_shape.rank().get_length() > 1) {
        x_slice_shape[1] = 1;
    }
    auto dt_slice_shape = dt_t_out.get_partial_shape();
    if (dt_slice_shape.rank().is_static() && dt_slice_shape.rank().get_length() > 1) {
        dt_slice_shape[1] = 1;
    }
    auto b_slice_shape = B_head_out.get_partial_shape();
    if (b_slice_shape.rank().is_static() && b_slice_shape.rank().get_length() > 1) {
        b_slice_shape[1] = 1;
    }
    auto c_slice_shape = C_head_out.get_partial_shape();
    if (c_slice_shape.rank().is_static() && c_slice_shape.rank().get_length() > 1) {
        c_slice_shape[1] = 1;
    }

    auto state_param = std::make_shared<ov::op::v0::Parameter>(out_type, s0_out.get_partial_shape());
    auto x_param = std::make_shared<ov::op::v0::Parameter>(out_type, x_slice_shape);
    auto dt_param = std::make_shared<ov::op::v0::Parameter>(out_type, dt_slice_shape);
    auto b_param = std::make_shared<ov::op::v0::Parameter>(out_type, b_slice_shape);
    auto c_param = std::make_shared<ov::op::v0::Parameter>(out_type, c_slice_shape);
    auto a_param = std::make_shared<ov::op::v0::Parameter>(out_type, A_t_out.get_partial_shape());

    const std::vector<int64_t> axis1_vals = {1};
    auto axis1 = ov::op::v0::Constant::create(ov::element::i64, {axis1_vals.size()}, axis1_vals);
    const std::vector<int64_t> axis2_vals = {2};
    auto axis2 = ov::op::v0::Constant::create(ov::element::i64, {axis2_vals.size()}, axis2_vals);
    const std::vector<int64_t> axis3_vals = {3};
    auto axis3 = ov::op::v0::Constant::create(ov::element::i64, {axis3_vals.size()}, axis3_vals);
    const std::vector<int64_t> axes13_vals = {1, 3};
    auto axes13 = ov::op::v0::Constant::create(ov::element::i64, {axes13_vals.size()}, axes13_vals);
    const std::vector<int64_t> axes23_vals = {2, 3};
    auto axes23 = ov::op::v0::Constant::create(ov::element::i64, {axes23_vals.size()}, axes23_vals);

    auto x_sq = std::make_shared<ov::op::v0::Squeeze>(x_param, axis1);
    auto x_ex = std::make_shared<ov::op::v0::Unsqueeze>(x_sq, axis3);
    auto dt_sq = std::make_shared<ov::op::v0::Squeeze>(dt_param, axes13);
    auto dt_softplus = std::make_shared<ov::op::v4::SoftPlus>(dt_sq);
    auto dt_ex = std::make_shared<ov::op::v0::Unsqueeze>(dt_softplus, axes23);

    auto b_sq = std::make_shared<ov::op::v0::Squeeze>(b_param, axis1);
    auto b_ex = std::make_shared<ov::op::v0::Unsqueeze>(b_sq, axis2);
    auto c_sq = std::make_shared<ov::op::v0::Squeeze>(c_param, axis1);
    auto c_ex = std::make_shared<ov::op::v0::Unsqueeze>(c_sq, axis2);

    auto dA = std::make_shared<ov::op::v0::Exp>(std::make_shared<ov::op::v1::Multiply>(dt_ex, a_param));
    auto x_dt = std::make_shared<ov::op::v1::Multiply>(x_ex, dt_ex);
    auto state_mul = std::make_shared<ov::op::v1::Multiply>(state_param, dA);
    auto bx = std::make_shared<ov::op::v1::Multiply>(b_ex, x_dt);
    auto state_new = std::make_shared<ov::op::v1::Add>(state_mul, bx);

    auto y_mul = std::make_shared<ov::op::v1::Multiply>(state_new, c_ex);
    const std::vector<int64_t> reduce_axes_vals = {3};
    auto reduce_axes = ov::op::v0::Constant::create(ov::element::i64, {reduce_axes_vals.size()}, reduce_axes_vals);
    auto y_sum = std::make_shared<ov::op::v1::ReduceSum>(y_mul, reduce_axes, false);
    auto y_out = std::make_shared<ov::op::v0::Unsqueeze>(y_sum, axis1);

    auto body = std::make_shared<ov::Model>(
        ov::ResultVector{std::make_shared<ov::op::v0::Result>(y_out),
                         std::make_shared<ov::op::v0::Result>(state_new)},
        ov::ParameterVector{state_param, x_param, dt_param, b_param, c_param, a_param});

    auto ti = std::make_shared<ov::op::v0::TensorIterator>();
    ti->set_body(body);
    ti->set_merged_input(state_param, s0_out, state_new);
    ti->set_sliced_input(x_param, x, 0, 1, 1, -1, 1);
    ti->set_sliced_input(dt_param, dt_t_out, 0, 1, 1, -1, 1);
    ti->set_sliced_input(b_param, B_head_out, 0, 1, 1, -1, 1);
    ti->set_sliced_input(c_param, C_head_out, 0, 1, 1, -1, 1);
    ti->set_invariant_input(a_param, A_t_out);

    auto y_concat = ti->get_concatenated_slices(y_out, 0, 1, 1, -1, 1);
    auto state_last = ti->get_iter_value(state_new, -1);

    const std::vector<int64_t> flat_shape_vals = {1, 1, 1, -1};
    auto flat_shape = ov::op::v0::Constant::create(ov::element::i64, {flat_shape_vals.size()}, flat_shape_vals);
    auto y_flat = std::make_shared<ov::op::v1::Reshape>(y_concat, flat_shape, false);
    auto state_flat = std::make_shared<ov::op::v1::Reshape>(state_last, flat_shape, false);
    auto res = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{y_flat, state_flat}, 3);

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
