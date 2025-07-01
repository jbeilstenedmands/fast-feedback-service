#ifndef BASELINE_INDEXER_COMBINATIONS_H
#define BASELINE_INDEXER_COMBINATIONS_H
#include <Eigen/Dense>
#include <dx2/crystal.hpp>
#include <optional>
#include <vector>

using Eigen::Vector3d;
using Eigen::Vector3i;

// A class to determine candadite orientation matrices by combining potential lattice vectors.

class CandidateOrientationMatrices {
  public:
    CandidateOrientationMatrices(const std::vector<Vector3d>& basis_vectors,
                                 int max_combinations = -1);
    bool has_next();
    // Generate the next valid combination that meets a set of criteria.
    std::optional<Crystal> next();

  private:
    std::vector<Vector3d> truncated_basis_vectors{};
    std::vector<Vector3i> combinations{};
    std::vector<Vector3i> truncated_combinations{};
    int n;
    int max_combinations;
    size_t index;
};

#endif // BASELINE_INDEXER_COMBINATIONS_H