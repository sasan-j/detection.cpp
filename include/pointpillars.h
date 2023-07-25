// Copyright 2020-present pytorch-cpp Authors
#pragma once

#include <torch/torch.h>
#include <vector>

#include "pillar_vfe.h"
#include "pointpillar_scatter.h"
#include "backbone2d.h"

namespace pointpillars {


struct PointPillars : public torch::nn::Module {
    //parameters: voxel_size, point_cloud_range, max_points_voxel, max_num_voxels
  PointPillars(std::vector<float> voxel_size, std::vector<float> point_cloud_range, int max_points_voxel, int max_num_voxels, torch::Tensor grid_size) {
    // Construct and register two Linear submodules.

    std::vector<int32_t> num_filters = {64};

    // Backbone Settings
    std::vector<int32_t> LAYER_NUMS = {3, 5, 5};
    std::vector<int32_t> LAYER_STRIDES = {2, 2, 2};
    std::vector<int32_t> NUM_FILTERS = {64, 128, 256};
    std::vector<int32_t> UPSAMPLE_STRIDES = {1, 2, 4};
    std::vector<int32_t> NUM_UPSAMPLE_FILTERS = {128, 128, 128};

    // PillarVFE
    vfe = register_module("vfe", PillarVFE(num_filters, true, true, true, voxel_size, point_cloud_range, 4));
    pp_scatter = register_module("map_to_bev", PointPillarScatter(num_filters[0], grid_size[0].item<int64_t>(), grid_size[1].item<int64_t>(),  grid_size[2].item<int64_t>()));
    // not so sure about the first parameter (num_channels)
    backbone2d = register_module("backbone2d", BaseBEVBackbone(num_filters[0], LAYER_NUMS, LAYER_STRIDES, NUM_FILTERS, UPSAMPLE_STRIDES, NUM_UPSAMPLE_FILTERS));

  }

  // Implement the Net's algorithm.
    std::unordered_map<std::string, torch::Tensor> forward(std::unordered_map<std::string, torch::Tensor> batch_dict) {
    
    auto out = vfe->forward(batch_dict);
    out = pp_scatter->forward(out);
    out = backbone2d->forward(out);

    // Use one of many tensor manipulation functions.
    // x = torch::relu(fc1->forward(x.reshape({x.size(0), 784})));
    // x = torch::dropout(x, /*p=*/0.5, /*train=*/is_training());
    // x = torch::relu(fc2->forward(x));
    // x = torch::log_softmax(fc3->forward(x), /*dim=*/1);
    return out;
  }

  // Use one of many "standard library" modules.
  PillarVFE vfe{nullptr};
  PointPillarScatter pp_scatter{nullptr};
  BaseBEVBackbone backbone2d{nullptr};
};

}