#pragma once

#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace cosyvoice_internal {


std::vector<std::string> split_into_fragments(std::string_view text);


using token_count_fn = std::function<std::size_t(std::string_view)>;


std::vector<std::string> reassemble_by_token_budget(
    const std::vector<std::string>& fragments,
    std::size_t max_tokens,
    const token_count_fn& count);


std::vector<std::vector<int>> reassemble_by_token_budget(
    std::vector<std::vector<int>>& fragment_tokens,
    std::size_t max_tokens);

}  // namespace cosyvoice_internal
