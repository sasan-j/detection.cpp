// Copyright 2020-present pytorch-cpp Authors
#pragma once

#include <torch/torch.h>
#include <vector>

#include "pillar_vfe.h"

namespace pointpillars {


struct PointPillars : public torch::nn::Module {
    //parameters: voxel_size, point_cloud_range, max_points_voxel, max_num_voxels
  PointPillars(std::vector<float> voxel_size, std::vector<float> point_cloud_range, int max_points_voxel, int max_num_voxels) {
    // Construct and register two Linear submodules.

    std::vector<int64_t> num_filters = {64};

    // PillarVFE
    vfe = register_module("vfe", PillarVFE(num_filters, true, true, true, voxel_size, point_cloud_range, 4));
  }

  // Implement the Net's algorithm.
    std::unordered_map<std::string, torch::Tensor> forward(std::unordered_map<std::string, torch::Tensor> batch_dict) {
    
    auto out = vfe->forward(batch_dict);

    // Use one of many tensor manipulation functions.
    // x = torch::relu(fc1->forward(x.reshape({x.size(0), 784})));
    // x = torch::dropout(x, /*p=*/0.5, /*train=*/is_training());
    // x = torch::relu(fc2->forward(x));
    // x = torch::log_softmax(fc3->forward(x), /*dim=*/1);
    return out;
  }

  // Use one of many "standard library" modules.
  PillarVFE vfe{nullptr};
};

}