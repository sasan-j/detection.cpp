#include <torch/torch.h>
#include <torch/script.h>
#include <cmath>

#include "utils.h"
#include "box_utils.h"

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
                         float dir_offset = 0.78539, float dir_limit_offset=0.0, bool predict_boxes_when_training = true, bool use_multihead = false)
    {
        this->use_multihead = use_multihead;
        this->num_class = num_class;
        this->class_names = class_names;
        this->predict_boxes_when_training = predict_boxes_when_training;
        this->num_dir_bins = num_dir_bins;
        this->dir_offset = dir_offset;
        this->dir_limit_offset = dir_limit_offset;

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

        // AnchorHeadSingle Constructor
        this->sum_num_anchors_per_location = this->num_anchors_per_location.sum().item<int>();

        this->conv_cls = register_module("conv_cls", torch::nn::Conv2d(torch::nn::Conv2dOptions(input_channels, this->sum_num_anchors_per_location * this->num_class, 1)));
        this->conv_box = register_module("conv_box", torch::nn::Conv2d(torch::nn::Conv2dOptions(input_channels, this->sum_num_anchors_per_location * this->box_coder.code_size, 1)));
        this->conv_dir_cls = register_module("conv_dir_cls", torch::nn::Conv2d(torch::nn::Conv2dOptions(input_channels, this->sum_num_anchors_per_location * this->num_dir_bins, 1)));
        init_weights();
    }

    std::pair<std::vector<torch::Tensor>, torch::Tensor> generate_anchors(
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

        return std::make_pair(anchors_list, num_anchors_per_location_list);
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
            for (const auto& pair : targets_dict) {
                this->forward_ret_dict[pair.first] = pair.second;
            }
        }

        if (!this->is_training() || this->predict_boxes_when_training)
        {
            // assuming generate_predicted_boxes is a function defined elsewhere in your code
            auto [batch_cls_preds, batch_box_preds] = this->generate_predicted_boxes(data_dict["batch_size"].item<int>(), cls_preds, box_preds, dir_cls_preds);
            data_dict["batch_cls_preds"] = batch_cls_preds;
            data_dict["batch_box_preds"] = batch_box_preds;
            data_dict["cls_preds_normalized"] = torch::tensor({false});
        }

        return data_dict;
    }

    std::unordered_map<std::string, torch::Tensor> assign_targets(torch::Tensor gt_boxes)
    {
        auto targets_dict = this->target_assigner.assign_targets(
            this->anchors, gt_boxes
        );
        return targets_dict;
    }

std::pair<torch::Tensor, torch::Tensor> generate_predicted_boxes(int batch_size, torch::Tensor cls_preds, torch::Tensor box_preds, torch::Tensor dir_cls_preds = torch::Tensor()) {
    std::vector<torch::Tensor> anchors = this->anchors;

    // Only Single Head
    torch::Tensor anchors_tensor = torch::cat(anchors, -3);

    int num_anchors = anchors_tensor.reshape({-1, anchors_tensor.size(-1)}).size(0);
    torch::Tensor batch_anchors = anchors_tensor.view({1, -1, anchors_tensor.size(-1)}).repeat({batch_size, 1, 1});
    
    torch::Tensor batch_cls_preds = cls_preds.view({batch_size, num_anchors, -1}).toType(torch::kFloat);
    torch::Tensor batch_box_preds = box_preds.view({batch_size, num_anchors, -1});
    batch_box_preds = this->box_coder.decode_torch(batch_box_preds, batch_anchors);

    if (dir_cls_preds.defined()) {
        float dir_offset = this->dir_offset;
        float dir_limit_offset = this->dir_limit_offset;
        dir_cls_preds = dir_cls_preds.view({batch_size, num_anchors, -1});

        torch::Tensor dir_labels = std::get<1>(dir_cls_preds.max(-1));

        float period = (2 * M_PI / this->num_dir_bins);
        torch::Tensor dir_rot = limit_period(
            batch_box_preds.index({torch::indexing::Slice(), torch::indexing::Slice(), 6}) - dir_offset, dir_limit_offset, period
        );

        batch_box_preds.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), 6}, dir_rot + dir_offset + period * dir_labels.to(batch_box_preds.dtype()));
    }

    return {batch_cls_preds, batch_box_preds};
}



private:
    bool use_multihead;
    int num_class;
    int num_dir_bins;
    float dir_offset;
    float dir_limit_offset;
    std::unordered_map<std::string, torch::Tensor> forward_ret_dict = {};
    std::vector<std::string> class_names;
    bool predict_boxes_when_training;
    std::vector<torch::Tensor> anchors;
    torch::Tensor num_anchors_per_location;
    int sum_num_anchors_per_location;
    ResidualCoder box_coder;
    AxisAlignedTargetAssigner target_assigner;
    torch::nn::Conv2d conv_cls{nullptr};
    torch::nn::Conv2d conv_box{nullptr};
    torch::nn::Conv2d conv_dir_cls{nullptr};
};

TORCH_MODULE(AnchorHeadSingle);