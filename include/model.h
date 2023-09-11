#pragma once

#include <torch/torch.h>
#include <vector>
#include "anchor_head.h"

// Create a struct to contain parameters
struct ModelConfig
{
  // Data Config
  std::vector<float> voxel_size;
  std::vector<float> point_cloud_range;
  int max_points_voxel;
  int max_num_voxels;

  // Backbone 2d Config
  std::vector<int32_t> backbone_layer_nums;
  std::vector<int32_t> backbone_layer_strides;
  std::vector<int32_t> backbone_num_filters;
  std::vector<float> backbone_upsample_strides;
  std::vector<int32_t> backbone_num_upsample_filters;

  AnchorHeadConfig anchor_head_config;
};