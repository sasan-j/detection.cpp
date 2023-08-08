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

    std::pair<std::vector<BatchMap>, BatchMap> forward(std::unordered_map<std::string, torch::Tensor> batch_dict)
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

      auto processed_out = postprocess(out);

      return processed_out;
    }

    std::vector<char> get_the_bytes(std::string filename)
    {
      std::ifstream input(filename, std::ios::binary);
      std::vector<char> bytes(
          (std::istreambuf_iterator<char>(input)),
          (std::istreambuf_iterator<char>()));

      input.close();
      return bytes;
    }

    void load_parameters(std::string pt_pth)
    {
      std::vector<char> f = this->get_the_bytes(pt_pth);
      c10::Dict<torch::IValue, torch::IValue> weights = torch::pickle_load(f).toGenericDict();

      std::cout << "Loaded weights OpenPCDet" << std::endl;
      for (auto const &w : weights)
      {
        std::cout << w.key().toStringRef() << w.value().toTensor().sizes() << '\n';
      }

      const torch::OrderedDict<std::string, at::Tensor> &model_params = this->named_parameters();

      std::cout << "Our model weights" << std::endl;
      for (auto const &w : model_params)
      {
        std::cout << w.key() << w.value().sizes() << '\n';
      }

      std::vector<std::string> param_names;
      for (auto const &w : model_params)
      {
        param_names.push_back(w.key());
      }

      //

      torch::NoGradGuard no_grad;
      for (auto const &w : weights)
      {
        std::string name = w.key().toStringRef();
        at::Tensor param = w.value().toTensor();

        if (std::find(param_names.begin(), param_names.end(), name) != param_names.end())
        {
          std::cout << name << " exists: " << param.sizes() << model_params.find(name)->sizes() << std::endl;
          model_params.find(name)->copy_(param);
        }
        else
        {
          std::cout << name << " does not exist among model parameters." << std::endl;
        };
      }
    }

    // POST_PROCESSING:
    //     RECALL_THRESH_LIST: [0.3, 0.5, 0.7]
    //     SCORE_THRESH: 0.1
    //     OUTPUT_RAW_SCORE: False

    //     EVAL_METRIC: kitti

    //     NMS_CONFIG:
    //         MULTI_CLASSES_NMS: False
    //         NMS_TYPE: nms_gpu
    //         NMS_THRESH: 0.01
    //         NMS_PRE_MAXSIZE: 4096
    //         NMS_POST_MAXSIZE: 500

struct PostProcessingConfig {
    std::vector<double> RECALL_THRESH_LIST = {0.3, 0.5, 0.7};
    double SCORE_THRESH = 0.1;
    bool OUTPUT_RAW_SCORE = false;
    std::string EVAL_METRIC = "kitti";
};


    std::pair<std::vector<BatchMap>, BatchMap> postprocess(BatchMap batch_dict)
    {
      auto nms_config = NMSConfig{false, "nms_gpu", 0.01, 4096, 500};
      auto post_processing_config = PostProcessingConfig{std::vector<double>{0.3, 0.5, 0.7}, 0.1, false, "kitti"};

      auto batch_size = batch_dict["batch_size"][0].item<int>();
      std::unordered_map<std::string, torch::Tensor> recall_dict;
      std::vector<BatchMap> pred_dicts;

      for (int index = 0; index < batch_size; index++)
      {

        torch::Tensor batch_mask;

        if (batch_dict.find("batch_index") != batch_dict.end()) {
            // This assumes batch_box_preds is a tensor
            TORCH_CHECK(batch_dict["batch_box_preds"].dim() == 2, "Expected batch_box_preds to have 2 dimensions");
            batch_mask = (batch_dict["batch_index"] == index);
        } else {
            TORCH_CHECK(batch_dict["batch_box_preds"].dim() == 3, "Expected batch_box_preds to have 3 dimensions");
            batch_mask = torch::tensor(index);
        }

        torch::Tensor box_preds = batch_dict["batch_box_preds"].index_select(0, batch_mask);
        torch::Tensor src_box_preds = box_preds;

        torch::Tensor cls_preds, src_cls_preds, label_preds;

        cls_preds = batch_dict["batch_cls_preds"].index_select(0, batch_mask);
        src_cls_preds = cls_preds;
        
        TORCH_CHECK(cls_preds.size(1) == 1 || cls_preds.size(1) == NUM_CLASS, "cls_preds shape mismatch");

        if (!batch_dict["cls_preds_normalized"].item<bool>()) {
            cls_preds = torch::sigmoid(cls_preds);
        }

        std::tie(cls_preds, label_preds) = torch::max(cls_preds, -1);

        if (batch_dict.find("has_class_labels") != batch_dict.end() && batch_dict["has_class_labels"].item<bool>()) {
            std::string label_key = batch_dict.find("roi_labels") != batch_dict.end() ? "roi_labels" : "batch_pred_labels";
            label_preds = batch_dict[label_key].index({index});
        } else {
            label_preds = label_preds + 1;
        }

        auto nms_result = class_agnostic_nms(
            cls_preds, box_preds, nms_config, 0.1);

        torch::Tensor selected = std::get<0>(nms_result);
        torch::Tensor selected_scores = std::get<1>(nms_result);

        if (post_processing_config.OUTPUT_RAW_SCORE)
        {
          auto max_result = torch::max(src_cls_preds, -1);
          torch::Tensor max_cls_preds = std::get<0>(max_result);
          selected_scores = max_cls_preds.index({selected});
        }

        torch::Tensor final_scores = selected_scores;
        torch::Tensor final_labels = label_preds.index({selected});
        torch::Tensor final_boxes = box_preds.index({selected});

        recall_dict = this->generate_recall_record(
            final_boxes, recall_dict, index, batch_dict, post_processing_config.RECALL_THRESH_LIST);

        std::unordered_map<std::string, torch::Tensor> record_dict = {
            {"pred_boxes", final_boxes},
            {"pred_scores", final_scores},
            {"pred_labels", final_labels}};
        pred_dicts.push_back(record_dict);
      }
      return std::make_pair(pred_dicts,recall_dict);
    }


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

    PillarVFE vfe{nullptr};
    PointPillarScatter pp_scatter{nullptr};
    BaseBEVBackbone backbone2d{nullptr};
    AnchorHeadSingle anchor_head{nullptr};
  };
}