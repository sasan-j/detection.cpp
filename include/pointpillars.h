// Copyright 2020-present pytorch-cpp Authors
#pragma once

#include <torch/torch.h>
#include <vector>

#include "pillar_vfe.h"
#include "pointpillar_scatter.h"
#include "backbone2d.h"
#include "anchor_head_single.h"

namespace pointpillars
{

  struct PointPillars : public torch::nn::Module
  {
    // parameters: voxel_size, point_cloud_range, max_points_voxel, max_num_voxels
    PointPillars(std::vector<float> voxel_size, std::vector<float> point_cloud_range, int max_points_voxel, int max_num_voxels, torch::Tensor grid_size)
    {
      // Construct and register two Linear submodules.

      std::vector<int32_t> num_filters = {64};

      // Backbone Settings
      std::vector<int32_t> LAYER_NUMS = {3, 5, 5};
      std::vector<int32_t> LAYER_STRIDES = {2, 2, 2};
      std::vector<int32_t> NUM_FILTERS = {64, 128, 256};
      std::vector<int32_t> UPSAMPLE_STRIDES = {1, 2, 4};
      std::vector<int32_t> NUM_UPSAMPLE_FILTERS = {128, 128, 128};

      // CLASS_AGNOSTIC: False

      // USE_DIRECTION_CLASSIFIER: True
      // DIR_OFFSET: 0.78539
      // DIR_LIMIT_OFFSET: 0.0
      // NUM_DIR_BINS: 2
      // Anchor Head Settings
      bool USE_DIRECTION_CLASSIFIER = true;
      bool USE_MULTIHEAD = false;
      int NUM_CLASS = 3;
      int NUM_DIR_BINS = 2;
      float DIR_OFFSET = 0.78539;
      float DIR_LIMIT_OFFSET = 0.0;
      std::vector<std::string> CLASS_NAMES = {"Car", "Pedestrian", "Cyclist"};

      // PillarVFE
      vfe = PillarVFE(num_filters, true, false, true, voxel_size, point_cloud_range, 4);
      pp_scatter = PointPillarScatter(64, grid_size.index({0}).item<int64_t>(), grid_size.index({1}).item<int64_t>(), grid_size.index({2}).item<int64_t>());
      register_module("vfe", vfe);
      register_module("map_to_bev", pp_scatter);
      // not so sure about the first parameter (num_channels)
      backbone2d = BaseBEVBackbone(num_filters[0], LAYER_NUMS, LAYER_STRIDES, NUM_FILTERS, UPSAMPLE_STRIDES, NUM_UPSAMPLE_FILTERS);
      register_module("backbone2d", backbone2d);
      anchor_head = AnchorHeadSingle(NUM_DIR_BINS, USE_DIRECTION_CLASSIFIER, 384, NUM_CLASS, CLASS_NAMES, grid_size, point_cloud_range, DIR_OFFSET, DIR_LIMIT_OFFSET, true);
      register_module("anchor_head", anchor_head);
    }

    BatchMap forward(std::unordered_map<std::string, torch::Tensor> batch_dict)
    {

      auto out = vfe->forward(batch_dict);
      std::cout << "After PillarVFE!\n";
      print_shapes(out);
      out = pp_scatter->forward(out);
      std::cout << "After Scatter!\n";
      print_shapes(out);
      std::cout << "spatial_feature max: " << out["spatial_features"].max() << '\n';
      out = backbone2d->forward(out);
      print_shapes(out);
      out = anchor_head->forward(out);
      print_shapes(out);

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
    AnchorHeadSingle anchor_head{nullptr};
  };
}