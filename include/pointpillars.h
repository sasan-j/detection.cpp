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
      register_module("backbone_2d", backbone2d);
      anchor_head = AnchorHeadSingle(NUM_DIR_BINS, USE_DIRECTION_CLASSIFIER, 384, NUM_CLASS, CLASS_NAMES, grid_size, point_cloud_range, DIR_OFFSET, DIR_LIMIT_OFFSET, true);
      register_module("dense_head", anchor_head);

      std::cout << "Backend2d" << '\n';
      std::cout << backbone2d << '\n';

      load_parameters("pointpillars_weights_simplified.pt");
    }

    BatchMap forward(std::unordered_map<std::string, torch::Tensor> batch_dict)
    {
      auto out = vfe->forward(batch_dict);
      std::cout << "After PillarVFE!\n";
      std::cout << "##############################!\n";
      print_shapes(out);
      out = pp_scatter->forward(out);
      std::cout << "After Scatter!\n";
      std::cout << "##############################!\n";
      print_shapes(out);
      std::cout << "spatial_feature max: " << out["spatial_features"].max() << '\n';
      out = backbone2d->forward(out);
      std::cout << "##############################!\n";
      print_shapes(out);
      out = anchor_head->forward(out);
      std::cout << "##############################!\n";
      print_shapes(out);

      // Use one of many tensor manipulation functions.
      // x = torch::relu(fc1->forward(x.reshape({x.size(0), 784})));
      // x = torch::dropout(x, /*p=*/0.5, /*train=*/is_training());
      // x = torch::relu(fc2->forward(x));
      // x = torch::log_softmax(fc3->forward(x), /*dim=*/1);
      return out;
    }

    std::vector<char> get_the_bytes(std::string filename) {
        std::ifstream input(filename, std::ios::binary);
        std::vector<char> bytes(
            (std::istreambuf_iterator<char>(input)),
            (std::istreambuf_iterator<char>()));

        input.close();
        return bytes;
    }

    void load_parameters(std::string pt_pth) {
      std::vector<char> f = this->get_the_bytes(pt_pth);
      c10::Dict<torch::IValue, torch::IValue> weights = torch::pickle_load(f).toGenericDict();

      std::cout << "Loaded weights OpenPCDet" << std::endl;
      for (auto const& w : weights) {
          std::cout << w.key().toStringRef() << w.value().toTensor().sizes() << '\n';
      }

      const torch::OrderedDict<std::string, at::Tensor>& model_params = this->named_parameters();

      std::cout << "Our model weights" << std::endl;
      for (auto const& w : model_params) {
          std::cout << w.key() << w.value().sizes() << '\n';
      }

      std::vector<std::string> param_names;
      for (auto const& w : model_params) {
        param_names.push_back(w.key());
      }

      // 

      torch::NoGradGuard no_grad;
      for (auto const& w : weights) {
          std::string name = w.key().toStringRef();
          at::Tensor param = w.value().toTensor();

          if (std::find(param_names.begin(), param_names.end(), name) != param_names.end()){
            std::cout << name << " exists: " << param.sizes() << model_params.find(name)->
            sizes() << std::endl;
            model_params.find(name)->copy_(param);
          } else {
            std::cout << name << " does not exist among model parameters." << std::endl;
          };

      }
    }

    PillarVFE vfe{nullptr};
    PointPillarScatter pp_scatter{nullptr};
    BaseBEVBackbone backbone2d{nullptr};
    AnchorHeadSingle anchor_head{nullptr};
  };
}