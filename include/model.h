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
  int num_point_features;

  // Backbone 2d Config
  std::vector<int32_t> backbone_layer_nums;
  std::vector<int32_t> backbone_layer_strides;
  std::vector<int32_t> backbone_num_filters;
  std::vector<float> backbone_upsample_strides;
  std::vector<int32_t> backbone_num_upsample_filters;

  AnchorHeadConfig anchor_head_config;
};


// struct BatchData{
//     int batch_size;
//     std::vector<torch::Tensor> batch_cls_preds;
//     std::vector<torch::Tensor> batch_box_preds;
//     std::vector<torch::Tensor> multihead_label_mapping;
//     bool class_preds_normalized;
// };

struct BatchData{
    std::unordered_map<std::string, torch::Tensor> tensor_dict;
    std::unordered_map<std::string, std::vector<torch::Tensor>> vector_dict;
};