#include <torch/torch.h>
#include <torch/script.h>
#include <cmath>

#include "utils.h"

// TARGET_ASSIGNER_CONFIG:
//     NAME: AxisAlignedTargetAssigner
//     POS_FRACTION: -1.0
//     SAMPLE_SIZE: 512
//     NORM_BY_NUM_EXAMPLES: False
//     MATCH_HEIGHT: False
//     BOX_CODER: ResidualCoder

// Assuming AnchorHeadTemplate is a parent class you've defined elsewhere
class AnchorHeadSingleImpl : public torch::nn::Module
{
public:
    AnchorHeadSingleImpl(int num_dir_bins, bool use_direction_classifier, int input_channels, int num_class, std::vector<std::string> class_names, torch::Tensor grid_size, std::vector<float> point_cloud_range,
                         bool predict_boxes_when_training = true, bool use_multihead = false)
    {
        this->use_multihead = use_multihead;
        this->num_class = num_class;
        this->class_names = class_names;
        this->predict_boxes_when_training = predict_boxes_when_training;
        this->num_dir_bins = num_dir_bins;

        this->box_coder = ResidualCoder();

        auto anchors = generate_anchors(grid_size, point_cloud_range, box_coder.code_size);
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

        this->target_assigner = AxisAlignedTargetAssigner(this->class_names, this->box_coder);

        // Ignore for now, probably only for training
        // build_losses(model_cfg["LOSS_CONFIG"].toDict());

        this->num_anchors_per_location = std::accumulate(this->num_anchors_per_location.begin(), this->num_anchors_per_location.end(), 0);

        this->conv_cls = register_module("conv_cls", torch::nn::Conv2d(torch::nn::Conv2dOptions(input_channels, this->num_anchors_per_location * this->num_class, 1)));
        this->conv_box = register_module("conv_box", torch::nn::Conv2d(torch::nn::Conv2dOptions(input_channels, this->num_anchors_per_location * this->box_coder.code_size, 1)));

        if (use_direction_classifier)
        {
            this->conv_dir_cls = register_module("conv_dir_cls", torch::nn::Conv2d(torch::nn::Conv2dOptions(input_channels, this->num_anchors_per_location * this->model_cfg["NUM_DIR_BINS"].toInt(), 1)));
        }
        else
        {
            this->conv_dir_cls = nullptr;
        }
        init_weights();
    }

    std::pair<std::vector<torch::Tensor>, std::vector<int>> generate_anchors(
        torch::Tensor grid_size,
        std::vector<float> point_cloud_range,
        int anchor_ndim = 7)
    {

        AnchorGenerator anchor_generator = AnchorGenerator(point_cloud_range);

        // From Yaml Config
        std::vector<int> feature_map_stride = {2,2,2};

        std::vector<torch::Tensor> feature_map_size;
        for (auto &stride : feature_map_stride)
        {
            feature_map_size.push_back(grid_size.slice(0, 0, 2) / stride);
        }

        auto [anchors_list, num_anchors_per_location_list] = anchor_generator.generate_anchors(feature_map_size);

        if (anchor_ndim != 7)
        {
            for (auto &anchors : anchors_list)
            {
                auto pad_zeros = torch::zeros({anchors.size(0), anchor_ndim - 7});
                auto new_anchors = torch::cat({anchors, pad_zeros}, -1);
                anchors = new_anchors;
            }
        }

        return {anchors_list, num_anchors_per_location_list};
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

        if (this->conv_dir_cls != nullptr)
        {
            torch::Tensor dir_cls_preds = this->conv_dir_cls(spatial_features_2d);
            dir_cls_preds = dir_cls_preds.permute({0, 2, 3, 1}).contiguous();
            this->forward_ret_dict["dir_cls_preds"] = dir_cls_preds;
        }
        else
        {
            torch::Tensor dir_cls_preds = nullptr;
        }

        if (this->is_training())
        {
            // assuming assign_targets is a function defined elsewhere in your code
            torch::Tensor targets_dict = this->assign_targets(data_dict["gt_boxes"]);
            this->forward_ret_dict.update(targets_dict);
        }

        if (!this->is_training() || this->predict_boxes_when_training)
        {
            // assuming generate_predicted_boxes is a function defined elsewhere in your code
            auto [batch_cls_preds, batch_box_preds] = this->generate_predicted_boxes(data_dict["batch_size"].toInt(), cls_preds, box_preds, dir_cls_preds);
            data_dict["batch_cls_preds"] = batch_cls_preds;
            data_dict["batch_box_preds"] = batch_box_preds;
            data_dict["cls_preds_normalized"] = false;
        }

        return data_dict;
    }

private:
    bool use_multihead;
    int num_class;
    int num_dir_bins;
    std::map<std::string, torch::Tensor> forward_ret_dict = {};
    std::vector<std::string> class_names;
    bool predict_boxes_when_training;
    std::vector<torch::Tensor> anchors;
    std::vector<int> num_anchors_per_location;
    ResidualCoder box_coder;
    AxisAlignedTargetAssigner target_assigner;
    torch::nn::Conv2d conv_cls{nullptr};
    torch::nn::Conv2d conv_box{nullptr};
    torch::nn::Conv2d conv_dir_cls{nullptr};
};

TORCH_MODULE(AnchorHeadSingle);