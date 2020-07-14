#pragma once

namespace mpl {

static const int pattern[8][2] = {{0, 0}, {-1, -1}, {-1, 1}, {1, -1},
                                  {0, 2}, {0, -2},  {2, 0},  {-2, 0}};

static const int PATTERN_WIDTH = 5;

static const int PATTERN_SIZE = 8;
}  // namespace mpl