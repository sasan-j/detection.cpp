#pragma once

#include <torch/torch.h>
#include <cmath>

#include "anchor_head.h"


class AnchorHeadSingleImpl : public torch::nn::Module
{
public:
    AnchorHeadSingleImpl(AnchorHeadConfig config, std::vector<float> point_cloud_range, torch::Tensor grid_size, int input_channels)
        : box_coder(ResidualCoder())
    {
        this->config = config;
        this->num_class = config.anchor_generator_configs.size();
        this->class_names = config.getClassNames();

        // this->box_coder = ResidualCoder();

        auto anchors = generate_anchors(grid_size, point_cloud_range, config.anchor_generator_configs, box_coder.code_size);
        this->anchors = anchors.first;
        this->num_anchors_per_location = anchors.second;
        for (auto &anchor : this->anchors)
        {
            anchor = anchor.to(torch::kCUDA);
        }

        // TARGET_ASSIGNER_CONFIG:
        //     NAME: AxisAlignedTargetAssigner
        //     POS_FRACTION: -1.0
        //     SAMPLE_SIZE: 512
        //     NORM_BY_NUM_EXAMPLES: False
        //     MATCH_HEIGHT: False
        //     BOX_CODER: ResidualCoder

        if (config.target_assigner_config.name == "AxisAlignedTargetAssigner")
        {
            this->target_assigner = AxisAlignedTargetAssigner(this->config.target_assigner_config, this->config.getClassNames(), this->box_coder);
        }
        else
        {
            std::cout << "Target Assigner not implemented" << '\n';
        }

        // Ignore for now, probably only for training
        // build_losses(model_cfg["LOSS_CONFIG"].toDict());

        std::cout << "num_anchors_per_location " << this->num_anchors_per_location.sizes() << this->num_anchors_per_location << '\n';

        // AnchorHeadSingle Constructor
        this->sum_num_anchors_per_location = this->num_anchors_per_location.sum().item<int64_t>();

        this->conv_cls = torch::nn::Conv2d(torch::nn::Conv2dOptions(input_channels, this->sum_num_anchors_per_location * this->num_class, 1));
        this->conv_box = torch::nn::Conv2d(torch::nn::Conv2dOptions(input_channels, this->sum_num_anchors_per_location * this->box_coder.code_size, 1));
        this->conv_dir_cls = torch::nn::Conv2d(torch::nn::Conv2dOptions(input_channels, this->sum_num_anchors_per_location * this->config.num_dir_bins, 1));

        std::cout << "conv_cls " << this->conv_cls << '\n';
        std::cout << "conv_box " << this->conv_box << '\n';
        std::cout << "conv_dir_cls " << this->conv_dir_cls << '\n';

        register_module("conv_cls", this->conv_cls);
        register_module("conv_box", this->conv_box);
        register_module("conv_dir_cls", this->conv_dir_cls);

        init_weights();
    }

    void init_weights()
    {
        float pi = 0.01;
        torch::nn::init::constant_(this->conv_cls->bias, -std::log((1 - pi) / pi));
        torch::nn::init::normal_(this->conv_box->weight, 0, 0.001);
    }

    std::unordered_map<std::string, torch::Tensor> forward(std::unordered_map<std::string, torch::Tensor> data_dict)
    {
        torch::Tensor spatial_features_2d = data_dict["spatial_features_2d"];

        torch::Tensor cls_preds = this->conv_cls(spatial_features_2d);
        torch::Tensor box_preds = this->conv_box(spatial_features_2d);

        cls_preds = cls_preds.permute({0, 2, 3, 1}).contiguous();
        box_preds = box_preds.permute({0, 2, 3, 1}).contiguous();

        this->forward_ret_dict["cls_preds"] = cls_preds;
        this->forward_ret_dict["box_preds"] = box_preds;

        torch::Tensor dir_cls_preds = this->conv_dir_cls(spatial_features_2d);
        dir_cls_preds = dir_cls_preds.permute({0, 2, 3, 1}).contiguous();
        this->forward_ret_dict["dir_cls_preds"] = dir_cls_preds;

        if (this->is_training())
        {
            std::unordered_map<std::string, torch::Tensor> targets_dict = this->assign_targets(data_dict["gt_boxes"]);
            for (const auto &pair : targets_dict)
            {
                this->forward_ret_dict[pair.first] = pair.second;
            }
        }

        if (this->predict_boxes_when_training)
        {
            // assuming generate_predicted_boxes is a function defined elsewhere in your code
            auto [batch_cls_preds, batch_box_preds] = generate_predicted_boxes(this->config, this->anchors, this->box_coder, data_dict["batch_size"].item<int>(), cls_preds, box_preds, dir_cls_preds);
            data_dict["batch_cls_preds"] = batch_cls_preds;
            data_dict["batch_box_preds"] = batch_box_preds;
            data_dict["cls_preds_normalized"] = torch::tensor({false});
        }

        return data_dict;
    }

    std::unordered_map<std::string, torch::Tensor> assign_targets(torch::Tensor gt_boxes)
    {
        auto targets_dict = this->target_assigner.assign_targets(
            this->anchors, gt_boxes);
        return targets_dict;
    }

private:
    AnchorHeadConfig config;
    int num_class;
    std::unordered_map<std::string, torch::Tensor> forward_ret_dict = {};
    std::vector<std::string> class_names;
    bool predict_boxes_when_training;
    std::vector<torch::Tensor> anchors;
    torch::Tensor num_anchors_per_location;
    int64_t sum_num_anchors_per_location;
    ResidualCoder box_coder;
    AxisAlignedTargetAssigner target_assigner;
    torch::nn::Conv2d conv_cls{nullptr};
    torch::nn::Conv2d conv_box{nullptr};
    torch::nn::Conv2d conv_dir_cls{nullptr};
};

TORCH_MODULE(AnchorHeadSingle);
