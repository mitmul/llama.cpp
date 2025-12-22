#include "../op_table.hpp"
#include "../utils.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>

#include <openvino/op/constant.hpp>
#include <openvino/op/reshape.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_view(const NodeContext & context) {
    num_inputs_check(context, 1, 1);

    // Fast-path: VIEW is frequently used as a pure reshape (same element count, zero offset).
    // Using Slice here is incorrect and can lead to invalid shapes in downstream ops.
    try {
        const auto input_shape = context.get_input_shape(0).to_shape();
        const auto output_shape = context.get_output_shape().to_shape();
        const auto * op_params = reinterpret_cast<const size_t *>(context.get_output_op_params());

        if (!input_shape.empty() && !output_shape.empty() && input_shape != output_shape && op_params != nullptr &&
            op_params[0] == 0) {
            const auto input_elems =
                std::accumulate(input_shape.begin(), input_shape.end(), size_t{1}, std::multiplies<size_t>());
            const auto output_elems =
                std::accumulate(output_shape.begin(), output_shape.end(), size_t{1}, std::multiplies<size_t>());

            if (input_elems == output_elems) {
                std::vector<int64_t> target_shape(output_shape.begin(), output_shape.end());
                auto shape_const =
                    ov::op::v0::Constant::create(ov::element::i64, {target_shape.size()}, target_shape);
                auto reshaped = std::make_shared<ov::op::v1::Reshape>(context.get_input(0), shape_const, false);
                return rename_outputs_with_suffix({reshaped}, context.get_name());
            }
        }
    } catch (const std::exception &) {
        // Fall through to legacy handling.
    }

    // Fast-path: VIEW that selects a contiguous slice from the last dimension and then reshapes.
    // This covers common cases like:
    // - Q/K/V views into a fused WQKV projection for both decode (token_len=1) and prefill (token_len>1)
    // - Mamba SSM_SCAN outputs where `y` and `state` are concatenated
    try {
        const auto input_shape = context.get_input_shape(0).to_shape();
        const auto output_shape = context.get_output_shape().to_shape();
        const auto * op_params = reinterpret_cast<const size_t *>(context.get_output_op_params());

        if (!input_shape.empty() && !output_shape.empty() && op_params != nullptr && input_shape.size() == 4 &&
            input_shape[0] == 1 && input_shape[1] == 1) {
            const auto input_elems =
                std::accumulate(input_shape.begin(), input_shape.end(), size_t{1}, std::multiplies<size_t>());
            const auto output_elems =
                std::accumulate(output_shape.begin(), output_shape.end(), size_t{1}, std::multiplies<size_t>());

            if (output_elems > 0 && output_elems < input_elems) {
                const size_t token_len = input_shape[2];
                if (token_len == 0 || (output_elems % token_len) != 0) {
                    throw std::runtime_error("VIEW fast-path expects output_elems divisible by token_len");
                }
                const size_t slice_elems = output_elems / token_len;
                if (slice_elems == 0) {
                    throw std::runtime_error("VIEW fast-path produced empty slice");
                }
                auto strides = context.get_input_stride(0);
                if (strides.size() >= 4 && strides[3] > 0) {
                    const int64_t axis_dim = static_cast<int64_t>(input_shape[3]);
                    const int64_t start = static_cast<int64_t>(op_params[0] / strides[3]);
                    const int64_t end = start + static_cast<int64_t>(slice_elems);
                    if (start >= 0 && end >= start && end <= axis_dim) {
                        auto begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {start});
                        auto end_const = ov::op::v0::Constant::create(ov::element::i64, {1}, {end});
                        auto stride = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
                        auto axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
                        auto sliced =
                            std::make_shared<ov::op::v8::Slice>(context.get_input(0), begin, end_const, stride, axes);

                        std::vector<int64_t> target_shape(output_shape.begin(), output_shape.end());
                        auto shape_const =
                            ov::op::v0::Constant::create(ov::element::i64, {target_shape.size()}, target_shape);
                        auto reshaped = std::make_shared<ov::op::v1::Reshape>(sliced, shape_const, false);
                        return rename_outputs_with_suffix({reshaped}, context.get_name());
                    }
                }
            }
        }
    } catch (const std::exception &) {
        // Fall through to legacy handling.
    }

    try {
        const auto input_shape = context.get_input_shape(0).to_shape();
        const auto output_shape = context.get_output_shape().to_shape();
        if (input_shape != output_shape && !input_shape.empty() && !output_shape.empty()) {
            const auto * op_params = reinterpret_cast<const size_t *>(context.get_output_op_params());
            auto strides = context.get_input_stride(0);
            size_t axis = output_shape.size() - 1;
            const size_t rank = std::min(input_shape.size(), output_shape.size());
            for (size_t i = 0; i < rank; ++i) {
                if (input_shape[i] != output_shape[i]) {
                    axis = i;
                    break;
                }
            }
            if (axis >= input_shape.size() && !input_shape.empty()) {
                axis = input_shape.size() - 1;
            }
            if (op_params != nullptr && axis < strides.size() && strides[axis] > 0 && axis < input_shape.size()) {
                const auto offset_bytes = op_params[0];
                const auto axis_stride = strides[axis];
                const int64_t axis_dim = static_cast<int64_t>(input_shape[axis]);

                int64_t length = static_cast<int64_t>(output_shape[axis]);
                if (length <= 0 && !output_shape.empty()) {
                    length = static_cast<int64_t>(output_shape.back());
                }

                if (axis_dim > 0 && length > 0) {
                    int64_t start = static_cast<int64_t>(offset_bytes / axis_stride);
                    if (start >= axis_dim) {
                        // Clamp oversized offsets to the last valid slice to avoid zero-length outputs.
                        start = std::max<int64_t>(0, axis_dim - length);
                    }
                    int64_t end = std::min<int64_t>(axis_dim, start + length);
                    if (end > start) {
                        auto begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {start});
                        auto end_const = ov::op::v0::Constant::create(ov::element::i64, {1}, {end});
                        auto stride = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
                        auto axes =
                            ov::op::v0::Constant::create(ov::element::i64, {1}, {static_cast<int64_t>(axis)});
                        auto sliced =
                            std::make_shared<ov::op::v8::Slice>(context.get_input(0), begin, end_const, stride, axes);
                        return rename_outputs_with_suffix({sliced}, context.get_name());
                    }
                }
            }
        }
    } catch (const std::exception &) {
        // Fall through to existing handling when shape data is incomplete.
    }

    if (context.get_op_case() == 2) {
        // VIEW-of-VIEW with the same element count: the upstream VIEW already accounts for any offset.
        // Applying process_view_input() here would apply the input VIEW's offset a second time and can
        // create empty slices (e.g. Vcur views into a fused WQKV projection).
        return {context.get_input(0)};
    }
    return {context.get_input(0)};
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
