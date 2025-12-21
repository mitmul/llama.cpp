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
#include <sstream>

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

    auto sx = context.get_input(0);
    auto c = context.get_input(1);

    auto sx_shape = context.get_input_shape(0);
    auto c_shape = context.get_input_shape(1);
    FRONT_END_GENERAL_CHECK(sx_shape.is_static() && c_shape.is_static(), "SSM_CONV requires static shapes");

    auto sx_shape_vec = sx_shape.to_shape();
    auto c_shape_vec = c_shape.to_shape();

    auto to_ggml_shape = [](const std::vector<size_t> & shape) {
        return std::vector<int64_t>(shape.rbegin(), shape.rend());
    };
    auto trim_trailing_ones = [](std::vector<int64_t> & shape, size_t target_rank) {
        while (shape.size() > target_rank && !shape.empty() && shape.back() == 1) {
            shape.pop_back();
        }
    };

    auto sx_shape_ggml = to_ggml_shape(sx_shape_vec);  // ggml order: [ne0, ne1, ...]
    auto c_shape_ggml = to_ggml_shape(c_shape_vec);
    trim_trailing_ones(sx_shape_ggml, 3);
    trim_trailing_ones(c_shape_ggml, 2);

    FRONT_END_GENERAL_CHECK(sx_shape_ggml.size() == 3 && c_shape_ggml.size() == 2,
                            "SSM_CONV expects 3D input and 2D kernel");

    if (sx_shape_vec.size() != sx_shape_ggml.size() ||
        !std::equal(sx_shape_vec.rbegin(), sx_shape_vec.rbegin() + static_cast<long>(sx_shape_ggml.size()),
                    sx_shape_ggml.begin())) {
        auto target = ov::op::v0::Constant::create(
            ov::element::i64, {sx_shape_ggml.size()}, std::vector<int64_t>(sx_shape_ggml.begin(), sx_shape_ggml.end()));
        sx = std::make_shared<ov::op::v1::Reshape>(sx, target, false);
    }
    if (c_shape_vec.size() != c_shape_ggml.size() ||
        !std::equal(c_shape_vec.rbegin(), c_shape_vec.rbegin() + static_cast<long>(c_shape_ggml.size()),
                    c_shape_ggml.begin())) {
        auto target = ov::op::v0::Constant::create(
            ov::element::i64, {c_shape_ggml.size()}, std::vector<int64_t>(c_shape_ggml.begin(), c_shape_ggml.end()));
        c = std::make_shared<ov::op::v1::Reshape>(c, target, false);
    }

    const int64_t d_conv = static_cast<int64_t>(c_shape_ggml[0]);
    const int64_t d_inner = static_cast<int64_t>(c_shape_ggml[1]);
    const int64_t n_t = static_cast<int64_t>(sx_shape_ggml[0]) - d_conv + 1;
    const int64_t n_s = static_cast<int64_t>(sx_shape_ggml[2]);

    FRONT_END_GENERAL_CHECK(sx_shape_ggml[1] == c_shape_ggml[1], "SSM_CONV channel size mismatch");
    FRONT_END_GENERAL_CHECK(sx_shape_ggml[0] == static_cast<size_t>(d_conv - 1 + n_t),
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

    // back to layout expected by downstream ops: [n_s, n_t, d_inner]
    auto out_perm = make_const({3}, {0, 2, 1});
    ov::Output<ov::Node> result = std::make_shared<ov::op::v1::Transpose>(conv, out_perm);

    const auto output_shape = context.get_output_shape().to_shape();
    if (!output_shape.empty()) {
        std::vector<int64_t> target_shape(output_shape.begin(), output_shape.end());
        auto shape_const = ov::op::v0::Constant::create(ov::element::i64, {target_shape.size()}, target_shape);
        result = std::make_shared<ov::op::v1::Reshape>(result, shape_const, false);
    }

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

    // OpenVINO shapes are reversed from ggml: [ne3, ne2, ne1, ne0].
    auto state_shape = state_shape_ps.to_shape();  // {n_state_seqs, n_head, head_dim, d_state}
    auto x_shape = x_shape_ps.to_shape();          // {n_seqs, n_seq_tokens, n_head, head_dim}
    auto dt_shape = dt_shape_ps.to_shape();        // {n_seqs, n_seq_tokens, n_head}
    auto A_shape = A_shape_ps.to_shape();          // {n_head, d_state or 1}
    auto B_shape = B_shape_ps.to_shape();          // {n_seqs, n_seq_tokens, n_group, d_state}
    auto C_shape = C_shape_ps.to_shape();          // same as B

    auto normalize_rank = [](ov::Output<ov::Node> node, ov::Shape & shape, size_t target_rank) -> ov::Output<ov::Node> {
        if (shape.size() == target_rank) {
            return node;
        }
        ov::Shape adjusted;
        if (shape.size() > target_rank) {
            // keep the trailing dimensions (token/head ordering) and drop leading ones if any
            adjusted.insert(adjusted.end(), shape.end() - static_cast<std::ptrdiff_t>(target_rank), shape.end());
        } else {
            // pad leading ones to preserve trailing ordering
            adjusted.insert(adjusted.end(), target_rank - shape.size(), 1);
            adjusted.insert(adjusted.end(), shape.begin(), shape.end());
        }
        shape = adjusted;
        auto shape_const = ov::op::v0::Constant::create(
            ov::element::i64, {shape.size()}, std::vector<int64_t>(shape.begin(), shape.end()));
        return std::make_shared<ov::op::v1::Reshape>(node, shape_const, false);
    };

    state_in = normalize_rank(state_in, state_shape, 4);
    x_in = normalize_rank(x_in, x_shape, 4);
    dt_in = normalize_rank(dt_in, dt_shape, 3);
    A_in = normalize_rank(A_in, A_shape, 2);
    B_in = normalize_rank(B_in, B_shape, 4);
    C_in = normalize_rank(C_in, C_shape, 4);

    const int64_t n_state_seqs = static_cast<int64_t>(state_shape[0]);
    const int64_t n_head = static_cast<int64_t>(state_shape[1]);
    const int64_t head_dim = static_cast<int64_t>(state_shape[2]);
    const int64_t d_state = static_cast<int64_t>(state_shape[3]);
    const int64_t n_seqs = static_cast<int64_t>(x_shape[0]);
    const int64_t n_tokens = static_cast<int64_t>(x_shape[1]);
    const int64_t n_group = static_cast<int64_t>(B_shape[2]);

    FRONT_END_GENERAL_CHECK(n_state_seqs >= n_seqs, "SSM_SCAN state buffer smaller than batch");
    FRONT_END_GENERAL_CHECK(x_shape[2] == state_shape[1] && x_shape[3] == state_shape[2],
                            "SSM_SCAN head shape mismatch");
    FRONT_END_GENERAL_CHECK(dt_shape[0] == static_cast<size_t>(n_seqs) && dt_shape[1] == x_shape[1] &&
                                dt_shape[2] == x_shape[2],
                            "SSM_SCAN dt shape mismatch");
    FRONT_END_GENERAL_CHECK(B_shape[0] == static_cast<size_t>(n_seqs) && B_shape[1] == x_shape[1] &&
                                B_shape[3] == static_cast<size_t>(d_state) && C_shape == B_shape,
                            "SSM_SCAN B/C shape mismatch");
    FRONT_END_GENERAL_CHECK(n_head % n_group == 0 && n_group > 0, "SSM_SCAN invalid group configuration");

    const bool is_mamba2 = A_shape[1] == 1;

    auto axis0 = make_const({1}, {0});
    auto axis1 = make_const({1}, {1});
    auto axis2 = make_const({1}, {2});
    auto axis3 = make_const({1}, {3});

    // Gather initial state by ids along sequence dimension (axis = 0)
    ov::Output<ov::Node> ids_i64 = std::make_shared<ov::op::v0::Convert>(ids_in, ov::element::i64);
    ids_i64 = std::make_shared<ov::op::v1::Reshape>(ids_i64, make_const({1}, {n_seqs}), false);
    auto gather_axis = make_const({}, {0});
    auto gathered_state = std::make_shared<ov::op::v1::Gather>(state_in, ids_i64, gather_axis);

    // Layout already matches [n_seqs, n_head, head_dim, d_state]
    ov::Output<ov::Node> state_cur = gathered_state;

    // Transpose inputs for easier token slicing
    auto x_perm = std::make_shared<ov::op::v1::Transpose>(x_in, make_const({4}, {0, 2, 3, 1}));
    auto dt_perm = std::make_shared<ov::op::v1::Transpose>(dt_in, make_const({3}, {0, 2, 1}));
    auto B_perm = std::make_shared<ov::op::v1::Transpose>(B_in, make_const({4}, {0, 2, 3, 1}));
    auto C_perm = std::make_shared<ov::op::v1::Transpose>(C_in, make_const({4}, {0, 2, 3, 1}));

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
        // Select token t
        auto token_idx = make_const({1}, {t});
        auto axis_tokens4 = make_const({}, {3});
        auto axis_tokens3 = make_const({}, {2});

        auto x_gather = std::make_shared<ov::op::v1::Gather>(x_perm, token_idx, axis_tokens4);
        auto x_step = std::make_shared<ov::op::v0::Squeeze>(x_gather, axis3);  // [n_seqs, n_head, head_dim]

        auto dt_gather = std::make_shared<ov::op::v1::Gather>(dt_perm, token_idx, axis_tokens3);
        auto dt_step = std::make_shared<ov::op::v0::Squeeze>(dt_gather, axis2);  // [n_seqs, n_head]
        auto dt_softplus = std::make_shared<ov::op::v4::SoftPlus>(dt_step);

        auto axis_tokens_bc = make_const({}, {3});
        auto B_gather = std::make_shared<ov::op::v1::Gather>(B_perm, token_idx, axis_tokens_bc);
        auto B_step = std::make_shared<ov::op::v0::Squeeze>(B_gather, axis3);  // [n_seqs, n_group, d_state]

        auto C_gather = std::make_shared<ov::op::v1::Gather>(C_perm, token_idx, axis_tokens_bc);
        auto C_step = std::make_shared<ov::op::v0::Squeeze>(C_gather, axis3);  // [n_seqs, n_group, d_state]

        auto B_head = std::make_shared<ov::op::v1::Gather>(B_step, group_indices, gather_group_axis);
        auto C_head = std::make_shared<ov::op::v1::Gather>(C_step, group_indices, gather_group_axis);

        auto dt_exp = std::make_shared<ov::op::v0::Unsqueeze>(dt_softplus, axis2);  // [n_seqs, n_head, 1]
        auto x_dt = std::make_shared<ov::op::v1::Multiply>(x_step, dt_exp);         // [n_seqs, n_head, head_dim]
        auto x_dt_expanded = std::make_shared<ov::op::v0::Unsqueeze>(x_dt, axis3);  // [n_seqs, n_head, head_dim, 1]

        ov::Output<ov::Node> dA_broadcast;
        if (is_mamba2) {
            auto A_squeezed = std::make_shared<ov::op::v0::Squeeze>(A_in, axis1);  // [n_head]
            auto A_unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(A_squeezed, axis0); // [1, n_head]
            auto target = make_const({2}, {n_seqs, n_head});
            auto A_broadcast = std::make_shared<ov::op::v3::Broadcast>(A_unsqueezed, target);
            auto dtA = std::make_shared<ov::op::v1::Multiply>(dt_softplus, A_broadcast);
            auto dA = std::make_shared<ov::op::v0::Exp>(dtA);  // [n_seqs, n_head]
            dA_broadcast = std::make_shared<ov::op::v0::Unsqueeze>(dA, make_const({2}, {2, 3}));
        } else {
            auto A_expand = std::make_shared<ov::op::v0::Unsqueeze>(A_in, axis0);                 // [1, n_head, d_state]
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
    auto y_perm = make_const({4}, {0, 2, 1, 3});                        // -> [n_seqs, n_tokens, n_head, head_dim]
    auto y_out = std::make_shared<ov::op::v1::Transpose>(y_concat, y_perm);

    const int64_t y_elements = head_dim * n_head * n_tokens * n_seqs;
    auto y_flat = std::make_shared<ov::op::v1::Reshape>(y_out, make_const({1}, {y_elements}), false);

    auto state_out = state_cur;
    const int64_t state_elements = d_state * head_dim * n_head * n_seqs;
    auto state_flat = std::make_shared<ov::op::v1::Reshape>(state_out, make_const({1}, {state_elements}), false);

    ov::Output<ov::Node> result = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{y_flat, state_flat}, 0);
    const auto output_shape = context.get_output_shape().to_shape();
    if (!output_shape.empty()) {
        std::vector<int64_t> target_shape(output_shape.begin(), output_shape.end());
        auto shape_const = ov::op::v0::Constant::create(ov::element::i64, {target_shape.size()}, target_shape);
        result = std::make_shared<ov::op::v1::Reshape>(result, shape_const, false);
    }

    return rename_outputs_with_suffix({result}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
