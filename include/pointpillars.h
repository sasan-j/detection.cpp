// Copyright 2020-present pytorch-cpp Authors
#pragma once

#include <torch/torch.h>
#include <vector>

#include "pillar_vfe.h"
#include "pointpillar_scatter.h"
#include "backbone2d.h"
#include "anchor_head.h"
#include "anchor_head_single.h"
#include "anchor_head_multi.h"
#include "model.h"

namespace pointpillars
{

  struct PointPillars : public torch::nn::Module
  {

    // parameters: voxel_size, point_cloud_range, max_points_voxel, max_num_voxels
    PointPillars(ModelConfig config) : config(config)
    {
      use_multihead = config.anchor_head_config.use_multihead;
      at::Tensor point_cloud_range = torch::tensor(config.point_cloud_range, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
      at::Tensor voxel_size = torch::tensor(config.voxel_size, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
      torch::Tensor grid_size = (point_cloud_range.slice(0, 3, 6) - point_cloud_range.slice(0, 0, 3)) / voxel_size;
      grid_size = grid_size.round().to(torch::kLong).reshape({-1});

      // Construct and register two Linear submodules.

      std::vector<int32_t> num_filters = {64};

      // CLASS_AGNOSTIC: False

      // PillarVFE
      vfe = PillarVFE(num_filters, true, false, true, config.voxel_size, config.point_cloud_range, config.num_point_features);
      pp_scatter = PointPillarScatter(64, grid_size.index({0}).item<int64_t>(), grid_size.index({1}).item<int64_t>(), grid_size.index({2}).item<int64_t>());
      register_module("vfe", vfe);
      register_module("map_to_bev", pp_scatter);
      // not so sure about the first parameter (num_channels)
      backbone2d = BaseBEVBackbone(config, num_filters[0]);
      register_module("backbone_2d", backbone2d);
      if (use_multihead)
      {
        anchor_heads = AnchorHeadMulti(config, config.point_cloud_range, grid_size, 384);
        register_module("dense_head", anchor_heads);
        num_class = anchor_heads->num_class;
      }
      else
      {
        anchor_head = AnchorHeadSingle(config.anchor_head_config, config.point_cloud_range, grid_size, 384);
        register_module("dense_head", anchor_head);
        num_class = anchor_head->num_class;
      }

      std::cout << "PointPillarVFE" << '\n';
      std::cout << vfe << '\n';

      std::cout << "PointPillarScatter" << '\n';
      std::cout << pp_scatter << '\n';

      std::cout << "Backend2d" << '\n';
      std::cout << backbone2d << '\n';

      std::cout << "DenseHead" << '\n';
      if (use_multihead)
      {
        std::cout << anchor_heads << '\n';
      }
      else
      {
        std::cout << anchor_head << '\n';
      }

      // Single Head
      // load_parameters("pointpillars_weights_simplified.pt");
      // Multi Head
      load_parameters("pp_multi_weights_simplified.pt");
    }

    std::vector<BatchMap> forward(std::unordered_map<std::string, torch::Tensor> batch_dict)
    {
      BatchData out_multi;

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
      std::cout << "After Backbone 2d!\n";
      std::cout << "##############################!\n";
      print_shapes(out);
      if (use_multihead)
      {
        out_multi = anchor_heads->forward(out);
      }
      else
      {
        out = anchor_head->forward(out);
      }

      std::cout << "##############################!\n";
      if (use_multihead)
      {
        print_shapes(out_multi.tensor_dict);
      }
      else
      {
        print_shapes(out);
      }

      // Use one of many tensor manipulation functions.
      // x = torch::relu(fc1->forward(x.reshape({x.size(0), 784})));
      // x = torch::dropout(x, /*p=*/0.5, /*train=*/is_training());
      // x = torch::relu(fc2->forward(x));
      // x = torch::log_softmax(fc3->forward(x), /*dim=*/1);

      auto processed_out = postprocess(out_multi);

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
        std::cout << w.key().toStringRef() << " shape: " << w.value().toTensor().sizes() << '\n';
      }
      std::cout << "##############################!\n";

      const torch::OrderedDict<std::string, at::Tensor> &model_params = this->named_parameters();

      std::cout << "Our model weights" << std::endl;
      for (auto const &w : model_params)
      {
        std::cout << w.key() << " shape: " << w.value().sizes() << '\n';
      }
      std::cout << "##############################!\n";

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
          // std::cout << name << " exists: " << param.sizes() << model_params.find(name)->sizes() << std::endl;
          if (param.sizes() != model_params.find(name)->sizes())
          {
            std::cout << name << " exists but difference sizes theirs vs ours: " << param.sizes() << model_params.find(name)->sizes() << std::endl;
          }
          model_params.find(name)->copy_(param);
        }
        else
        {
          // don't print running_mean and running_var and num_batches_tracked
          if (name.find("running_mean") == std::string::npos && name.find("running_var") == std::string::npos && name.find("num_batches_tracked") == std::string::npos)
          {
            std::cout << name << " does not exist among model parameters." << std::endl;
          }
          else
          {
          }
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

    struct PostProcessingConfig
    {
      std::vector<double> RECALL_THRESH_LIST = {0.3, 0.5, 0.7};
      double SCORE_THRESH = 0.1;
      bool OUTPUT_RAW_SCORE = false;
      std::string EVAL_METRIC = "kitti";
    };

    std::vector<BatchMap> postprocess(BatchData batch_data)
    {
      BatchMap batch_dict = batch_data.tensor_dict;
      // auto nms_config = NMSConfig{false, "nms_gpu", 0.01, 4096, 500};
      auto nms_config = NMSConfig{true, "nms_gpu", 0.02, 1000, 83};
      auto post_processing_config = PostProcessingConfig{std::vector<double>{0.3, 0.5, 0.7}, 0.1, false, "kitti"};

      auto batch_size = batch_dict["batch_size"][0].item<int>();
      std::vector<BatchMap> pred_dicts;

      for (int index = 0; index < batch_size; index++)
      {

        torch::Tensor batch_mask;

        if (batch_dict.find("batch_index") != batch_dict.end())
        {
          // This assumes batch_box_preds is a tensor
          TORCH_CHECK(batch_dict["batch_box_preds"].dim() == 2, "Expected batch_box_preds to have 2 dimensions");
          batch_mask = (batch_dict["batch_index"] == index);
        }
        else
        {
          TORCH_CHECK(batch_dict["batch_box_preds"].dim() == 3, "Expected batch_box_preds to have 3 dimensions");
          batch_mask = torch::tensor(index);
        }

        std::cout << "batch_mask: " << batch_mask << std::endl;
        std::cout << "batch box_preds: " << batch_dict["batch_box_preds"].sizes() << std::endl;

        torch::Tensor box_preds = batch_dict["batch_box_preds"].index({batch_mask});
        std::cout << "box_preds: " << box_preds.sizes() << std::endl;

        std::cout << "points orig: " << batch_dict["points"].sizes() << std::endl;
        std::cout << "points preview before: " << batch_dict["points"].index({torch::indexing::Slice(0, 5)}) << std::endl;
        torch::Tensor points = batch_dict["points"].index_select(0, (batch_dict["points"].select(1, 0) == index).nonzero().squeeze()).slice(1, 1);
        std::cout << "points: " << points.sizes() << std::endl;
        std::cout << "points preview after: " << points.index({torch::indexing::Slice(0, 5)}) << std::endl;

        torch::Tensor src_box_preds = box_preds;

        torch::Tensor cls_preds, src_cls_preds, label_preds;


        /////////////////
        std::vector<torch::Tensor> cls_preds_list;

        // Identify if this is a case of multihead
        if (batch_data.vector_dict.find("batch_cls_preds") != batch_data.vector_dict.end())
        {
          for (auto cls_preds_item : batch_data.vector_dict["batch_cls_preds"])
          {
            if (!batch_dict["cls_preds_normalized"].item<bool>())
            {
              cls_preds_list.push_back(torch::sigmoid(cls_preds_item.index({batch_mask})));
            }
            else {
              cls_preds_list.push_back(cls_preds_item.index({batch_mask}));
            }
          }
        }
        else
        {
          std::cout << "batch_dict - batch_cls_preds: " << batch_dict["batch_cls_preds"].sizes() << std::endl;
          std::cout << "batch_mask: " << batch_mask << std::endl;
          cls_preds = batch_dict["batch_cls_preds"].index({batch_mask});
          src_cls_preds = cls_preds;

          std::cout << "cls_preds: " << cls_preds.sizes() << std::endl;
          TORCH_CHECK(cls_preds.size(1) == 1 || cls_preds.size(1) == this->num_class, "cls_preds shape mismatch");

          if (!batch_dict["cls_preds_normalized"].item<bool>())
          {
            cls_preds = torch::sigmoid(cls_preds);
          }
        }

        //////////////////////////////////////////////////
        std::vector<torch::Tensor> multihead_label_mapping;
        torch::Tensor final_scores, final_labels, final_boxes;

        if (nms_config.MULTI_CLASSES_NMS) {
          if (cls_preds_list.size() == 0) { // If cls_preds is not a list
              cls_preds_list.push_back(cls_preds);  // Add an additional dimension to mimic list
              multihead_label_mapping.push_back(torch::arange(1, num_class, cls_preds_list[0].device()));
          } else {
              multihead_label_mapping = batch_data.vector_dict["multihead_label_mapping"];  // Assuming this is a vector of tensors
          }

          int64_t cur_start_idx = 0;
          std::vector<torch::Tensor> pred_scores, pred_labels, pred_boxes;

          for (int i = 0; i < cls_preds_list.size(); ++i) {
              torch::Tensor cur_cls_preds = cls_preds_list[i];
              torch::Tensor cur_label_mapping = multihead_label_mapping[i];
              
              assert(cur_cls_preds.size(1) == cur_label_mapping.size(0));

              torch::Tensor cur_box_preds = box_preds.slice(0, cur_start_idx, cur_start_idx + cur_cls_preds.size(0));

              std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> nms_result = 
                  multi_classes_nms(
                      cur_cls_preds, cur_box_preds, 
                      nms_config, 
                      post_processing_config.SCORE_THRESH
                  );

              torch::Tensor cur_pred_scores = std::get<0>(nms_result);
              torch::Tensor cur_pred_labels = cur_label_mapping.index_select(0, std::get<1>(nms_result));
              torch::Tensor cur_pred_boxes = std::get<2>(nms_result);

              pred_scores.push_back(cur_pred_scores);
              pred_labels.push_back(cur_pred_labels);
              pred_boxes.push_back(cur_pred_boxes);

              cur_start_idx += cur_cls_preds.size(0);
          }

          final_scores = torch::cat(pred_scores, 0);
          final_labels = torch::cat(pred_labels, 0);
          final_boxes = torch::cat(pred_boxes, 0);
        } else {
          // Single Head Kinda thing
          std::tie(cls_preds, label_preds) = torch::max(cls_preds, -1);

          if (batch_dict.find("has_class_labels") != batch_dict.end() && batch_dict["has_class_labels"].item<bool>())
          {
            std::string label_key = batch_dict.find("roi_labels") != batch_dict.end() ? "roi_labels" : "batch_pred_labels";
            label_preds = batch_dict[label_key].index({index});
          }
          else
          {
            label_preds = label_preds + 1;
          }

          auto nms_result = class_agnostic_nms(
              cls_preds, box_preds, nms_config, post_processing_config.SCORE_THRESH);

          torch::Tensor selected = std::get<0>(nms_result);
          torch::Tensor selected_scores = std::get<1>(nms_result);

          if (post_processing_config.OUTPUT_RAW_SCORE)
          {
            auto max_result = torch::max(src_cls_preds, -1);
            torch::Tensor max_cls_preds = std::get<0>(max_result);
            selected_scores = max_cls_preds.index({selected});
          }

          final_scores = selected_scores;
          final_labels = label_preds.index({selected});
          final_boxes = box_preds.index({selected});
        }


        std::unordered_map<std::string, torch::Tensor> record_dict = {
              {"points", points},
              {"pred_boxes", final_boxes},
              {"pred_scores", final_scores},
              {"pred_labels", final_labels}};
        pred_dicts.push_back(record_dict);
      }
      return pred_dicts;
    }

    ModelConfig config;
    int num_class;
    bool use_multihead = false;
    PillarVFE vfe{nullptr};
    PointPillarScatter pp_scatter{nullptr};
    BaseBEVBackbone backbone2d{nullptr};
    AnchorHeadSingle anchor_head{nullptr};
    AnchorHeadMulti anchor_heads{nullptr};
  };
}