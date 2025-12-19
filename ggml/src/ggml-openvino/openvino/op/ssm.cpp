#include "../node_context.hpp"
#include "../op_table.hpp"
#include "../utils.hpp"

#include <vector>

#include <openvino/frontend/exception.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/broadcast.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/exp.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/group_conv.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/reduce_sum.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/softplus.hpp>
#include <openvino/op/squeeze.hpp>
#include <openvino/op/transpose.hpp>
#include <openvino/op/unsqueeze.hpp>
#include <openvino/op/slice.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

namespace {

std::shared_ptr<ov::op::v0::Constant> make_const(const ov::Shape & shape, const std::vector<int64_t> & values) {
    return ov::op::v0::Constant::create(ov::element::i64, shape, values);
}

}  // namespace

OutputVector translate_ssm_conv(const NodeContext & context) {
    num_inputs_check(context, 2, 2);

    const auto sx = context.get_input(0);
    const auto c = context.get_input(1);

    const auto sx_shape_ps = context.get_input_shape(0);
    const auto c_shape_ps = context.get_input_shape(1);
    FRONT_END_GENERAL_CHECK(sx_shape_ps.is_static() && c_shape_ps.is_static(), "SSM_CONV requires static shapes");

    const auto sx_shape = sx_shape_ps.to_shape();  // {d_conv - 1 + n_t, d_inner, n_s}
    const auto c_shape = c_shape_ps.to_shape();    // {d_conv, d_inner}
    FRONT_END_GENERAL_CHECK(sx_shape.size() == 3 && c_shape.size() == 2, "SSM_CONV expects 3D input and 2D kernel");

    const int64_t d_conv = static_cast<int64_t>(c_shape[0]);
    const int64_t d_inner = static_cast<int64_t>(c_shape[1]);
    const int64_t n_t = static_cast<int64_t>(sx_shape[0]) - d_conv + 1;
    const int64_t n_s = static_cast<int64_t>(sx_shape[2]);

    FRONT_END_GENERAL_CHECK(sx_shape[1] == c_shape[1], "SSM_CONV channel size mismatch");
    FRONT_END_GENERAL_CHECK(sx_shape[0] == static_cast<size_t>(d_conv - 1 + n_t),
                            "SSM_CONV time dimension mismatch");

    // reorder input to [n_s, d_inner, time]
    auto data_perm = make_const({3}, {2, 1, 0});
    auto data_ncw = std::make_shared<ov::op::v1::Transpose>(sx, data_perm);

    // weights: [d_conv, d_inner] -> [d_inner, 1, 1, d_conv] for depthwise group convolution
    auto weights_t = std::make_shared<ov::op::v1::Transpose>(c, make_const({2}, {1, 0}));
    auto weights_shape = make_const({4}, {d_inner, 1, 1, d_conv});
    auto weights_dw = std::make_shared<ov::op::v1::Reshape>(weights_t, weights_shape, false);

    auto conv = std::make_shared<ov::op::v1::GroupConvolution>(
        data_ncw, weights_dw, ov::Strides{1}, ov::CoordinateDiff{0}, ov::CoordinateDiff{0}, ov::Strides{1});

    // back to ggml layout [d_inner, n_t, n_s]
    auto out_perm = make_const({3}, {1, 2, 0});
    auto result = std::make_shared<ov::op::v1::Transpose>(conv, out_perm);

    return rename_outputs_with_suffix({result}, context.get_name());
}

OutputVector translate_ssm_scan(const NodeContext & context) {
    num_inputs_check(context, 7, 7);

    auto state_in = context.get_input(0);
    auto x_in = context.get_input(1);
    auto dt_in = context.get_input(2);
    auto A_in = context.get_input(3);
    auto B_in = context.get_input(4);
    auto C_in = context.get_input(5);
    auto ids_in = context.get_input(6);

    const auto state_shape_ps = context.get_input_shape(0);
    const auto x_shape_ps = context.get_input_shape(1);
    const auto dt_shape_ps = context.get_input_shape(2);
    const auto A_shape_ps = context.get_input_shape(3);
    const auto B_shape_ps = context.get_input_shape(4);
    const auto C_shape_ps = context.get_input_shape(5);

    FRONT_END_GENERAL_CHECK(state_shape_ps.is_static() && x_shape_ps.is_static() && dt_shape_ps.is_static() &&
                                A_shape_ps.is_static() && B_shape_ps.is_static() && C_shape_ps.is_static(),
                            "SSM_SCAN requires static input shapes");

    const auto state_shape = state_shape_ps.to_shape();  // {d_state, dim, n_head, n_state_seqs}
    const auto x_shape = x_shape_ps.to_shape();          // {dim, n_head, n_seq_tokens, n_seqs}
    const auto dt_shape = dt_shape_ps.to_shape();        // {n_head, n_seq_tokens, n_seqs}
    const auto A_shape = A_shape_ps.to_shape();          // {d_state or 1, n_head}
    const auto B_shape = B_shape_ps.to_shape();          // {d_state, n_group, n_seq_tokens, n_seqs}
    const auto C_shape = C_shape_ps.to_shape();          // same as B

    FRONT_END_GENERAL_CHECK(state_shape.size() == 4 && x_shape.size() == 4 && dt_shape.size() == 3 &&
                                B_shape.size() == 4 && C_shape.size() == 4 && A_shape.size() == 2,
                            "Unexpected rank for SSM_SCAN inputs");

    const int64_t d_state = static_cast<int64_t>(state_shape[0]);
    const int64_t head_dim = static_cast<int64_t>(state_shape[1]);
    const int64_t n_head = static_cast<int64_t>(state_shape[2]);
    const int64_t n_state_seqs = static_cast<int64_t>(state_shape[3]);
    const int64_t n_tokens = static_cast<int64_t>(x_shape[2]);
    const int64_t n_seqs = static_cast<int64_t>(x_shape[3]);
    const int64_t n_group = static_cast<int64_t>(B_shape[1]);

    FRONT_END_GENERAL_CHECK(n_state_seqs >= n_seqs, "SSM_SCAN state buffer smaller than batch");
    FRONT_END_GENERAL_CHECK(x_shape[0] == state_shape[1] && x_shape[1] == state_shape[2],
                            "SSM_SCAN head shape mismatch");
    FRONT_END_GENERAL_CHECK(dt_shape[0] == static_cast<size_t>(n_head) && dt_shape[1] == x_shape[2] &&
                                dt_shape[2] == x_shape[3],
                            "SSM_SCAN dt shape mismatch");
    FRONT_END_GENERAL_CHECK(B_shape[0] == static_cast<size_t>(d_state) && C_shape == B_shape,
                            "SSM_SCAN B/C shape mismatch");
    FRONT_END_GENERAL_CHECK(n_head % n_group == 0 && n_group > 0, "SSM_SCAN invalid group configuration");

    const bool is_mamba2 = A_shape[0] == 1;

    auto axis0 = make_const({1}, {0});
    auto axis2 = make_const({1}, {2});
    auto axis3 = make_const({1}, {3});

    // Gather initial state by ids along sequence dimension (axis = 3)
    auto ids_i64 = std::make_shared<ov::op::v0::Convert>(ids_in, ov::element::i64);
    auto gather_axis = make_const({}, {3});
    auto gathered_state = std::make_shared<ov::op::v1::Gather>(state_in, ids_i64, gather_axis);

    // Reorder layout to [n_seqs, n_head, dim, d_state]
    auto state_perm = make_const({4}, {3, 2, 1, 0});
    ov::Output<ov::Node> state_cur = std::make_shared<ov::op::v1::Transpose>(gathered_state, state_perm);

    // Transpose inputs for easier token slicing
    auto x_perm = std::make_shared<ov::op::v1::Transpose>(x_in, make_const({4}, {3, 1, 0, 2}));
    auto dt_perm = std::make_shared<ov::op::v1::Transpose>(dt_in, make_const({3}, {2, 0, 1}));
    auto B_perm = std::make_shared<ov::op::v1::Transpose>(B_in, make_const({4}, {3, 1, 0, 2}));
    auto C_perm = std::make_shared<ov::op::v1::Transpose>(C_in, make_const({4}, {3, 1, 0, 2}));

    // group mapping for B/C gather
    std::vector<int64_t> group_map(static_cast<size_t>(n_head));
    const int64_t heads_per_group = n_head / n_group;
    for (int64_t h = 0; h < n_head; ++h) {
        group_map[static_cast<size_t>(h)] = h / heads_per_group;
    }
    auto group_indices = make_const({static_cast<size_t>(n_head)}, group_map);
    auto gather_group_axis = make_const({}, {1});

    std::vector<ov::Output<ov::Node>> y_tokens;
    y_tokens.reserve(static_cast<size_t>(n_tokens));

    for (int64_t t = 0; t < n_tokens; ++t) {
        // Slice token t
        auto token_begin4 = make_const({4}, {0, 0, 0, t});
        auto token_end4 = make_const({4}, {n_seqs, n_head, head_dim, t + 1});
        auto strides4 = make_const({4}, {1, 1, 1, 1});
        auto x_slice = std::make_shared<ov::op::v8::Slice>(x_perm, token_begin4, token_end4, strides4);
        auto x_step = std::make_shared<ov::op::v0::Squeeze>(x_slice, axis3);  // [n_seqs, n_head, head_dim]

        auto token_begin3 = make_const({3}, {0, 0, t});
        auto token_end3 = make_const({3}, {n_seqs, n_head, t + 1});
        auto strides3 = make_const({3}, {1, 1, 1});
        auto dt_slice = std::make_shared<ov::op::v8::Slice>(dt_perm, token_begin3, token_end3, strides3);
        auto dt_step = std::make_shared<ov::op::v0::Squeeze>(dt_slice, axis2);  // [n_seqs, n_head]
        auto dt_softplus = std::make_shared<ov::op::v4::SoftPlus>(dt_step);

        auto BC_begin = make_const({4}, {0, 0, 0, t});
        auto BC_end = make_const({4}, {n_seqs, n_group, d_state, t + 1});
        auto BC_slice = std::make_shared<ov::op::v8::Slice>(B_perm, BC_begin, BC_end, strides4);
        auto B_step = std::make_shared<ov::op::v0::Squeeze>(BC_slice, axis3);  // [n_seqs, n_group, d_state]
        BC_slice = std::make_shared<ov::op::v8::Slice>(C_perm, BC_begin, BC_end, strides4);
        auto C_step = std::make_shared<ov::op::v0::Squeeze>(BC_slice, axis3);  // [n_seqs, n_group, d_state]

        auto B_head = std::make_shared<ov::op::v1::Gather>(B_step, group_indices, gather_group_axis);
        auto C_head = std::make_shared<ov::op::v1::Gather>(C_step, group_indices, gather_group_axis);

        auto dt_exp = std::make_shared<ov::op::v0::Unsqueeze>(dt_softplus, axis2);  // [n_seqs, n_head, 1]
        auto x_dt = std::make_shared<ov::op::v1::Multiply>(x_step, dt_exp);         // [n_seqs, n_head, head_dim]
        auto x_dt_expanded = std::make_shared<ov::op::v0::Unsqueeze>(x_dt, axis3);  // [n_seqs, n_head, head_dim, 1]

        ov::Output<ov::Node> dA_broadcast;
        if (is_mamba2) {
            auto A_squeezed = std::make_shared<ov::op::v0::Squeeze>(A_in, axis0);  // [n_head]
            auto A_unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(A_squeezed, axis0); // [1, n_head]
            auto target = make_const({2}, {n_seqs, n_head});
            auto A_broadcast = std::make_shared<ov::op::v3::Broadcast>(A_unsqueezed, target);
            auto dtA = std::make_shared<ov::op::v1::Multiply>(dt_softplus, A_broadcast);
            auto dA = std::make_shared<ov::op::v0::Exp>(dtA);  // [n_seqs, n_head]
            dA_broadcast = std::make_shared<ov::op::v0::Unsqueeze>(dA, make_const({2}, {2, 3}));
        } else {
            auto A_perm = std::make_shared<ov::op::v1::Transpose>(A_in, make_const({2}, {1, 0}));  // [n_head, d_state]
            auto A_expand = std::make_shared<ov::op::v0::Unsqueeze>(A_perm, axis0);               // [1, n_head, d_state]
            auto dt_expanded = std::make_shared<ov::op::v0::Unsqueeze>(dt_softplus, axis2);       // [n_seqs, n_head, 1]
            auto dtA = std::make_shared<ov::op::v1::Multiply>(dt_expanded, A_expand);             // [n_seqs, n_head, d_state]
            auto dA = std::make_shared<ov::op::v0::Exp>(dtA);
            dA_broadcast = std::make_shared<ov::op::v0::Unsqueeze>(dA, axis2);                    // [n_seqs, n_head, 1, d_state]
        }

        auto state_scaled = std::make_shared<ov::op::v1::Multiply>(state_cur, dA_broadcast);
        auto B_expanded = std::make_shared<ov::op::v0::Unsqueeze>(B_head, axis2);  // [n_seqs, n_head, 1, d_state]
        auto update = std::make_shared<ov::op::v1::Multiply>(B_expanded, x_dt_expanded);
        auto state_next = std::make_shared<ov::op::v1::Add>(state_scaled, update);  // [n_seqs, n_head, head_dim, d_state]

        auto C_expanded = std::make_shared<ov::op::v0::Unsqueeze>(C_head, axis2);  // [n_seqs, n_head, 1, d_state]
        auto y_mul = std::make_shared<ov::op::v1::Multiply>(state_next, C_expanded);
        auto reduce_axes = make_const({1}, {3});
        auto y_step = std::make_shared<ov::op::v1::ReduceSum>(y_mul, reduce_axes, false);  // [n_seqs, n_head, head_dim]
        auto y_step_time = std::make_shared<ov::op::v0::Unsqueeze>(y_step, axis2);         // [n_seqs, n_head, 1, head_dim]
        y_tokens.push_back(y_step_time);

        state_cur = state_next;
    }

    FRONT_END_GENERAL_CHECK(!y_tokens.empty(), "SSM_SCAN requires at least one token");

    auto y_concat = std::make_shared<ov::op::v0::Concat>(y_tokens, 2);  // [n_seqs, n_head, n_tokens, head_dim]
    auto y_perm = make_const({4}, {3, 1, 2, 0});                        // -> [head_dim, n_head, n_tokens, n_seqs]
    auto y_out = std::make_shared<ov::op::v1::Transpose>(y_concat, y_perm);

    const int64_t y_elements = head_dim * n_head * n_tokens * n_seqs;
    auto y_flat = std::make_shared<ov::op::v1::Reshape>(y_out, make_const({1}, {y_elements}), false);

    auto state_out = std::make_shared<ov::op::v1::Transpose>(state_cur, make_const({4}, {3, 2, 1, 0}));
    const int64_t state_elements = d_state * head_dim * n_head * n_seqs;
    auto state_flat = std::make_shared<ov::op::v1::Reshape>(state_out, make_const({1}, {state_elements}), false);

    auto result = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{y_flat, state_flat}, 0);

    return rename_outputs_with_suffix({result}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
