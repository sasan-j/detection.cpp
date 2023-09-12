#pragma once

#include "utils.h"
#include "box_utils.h"
#include "target_assigner.h"
#include "anchor_generator.h"

struct SeparateRegConfig {
    int num_middle_conv = 1;
    int num_middle_filter = 64;
    std::vector<std::string> reg_list = {"reg:2", "height:1", "size:3", "angle:2", "velo:2"};
};

struct AnchorHeadConfig
{
    int num_dir_bins;
    bool use_direction_classifier;
    // int input_channels;
    float dir_offset;
    float dir_limit_offset;
    TargetAssignerConfig target_assigner_config;
    std::vector<AnchorGeneratorConfig> anchor_generator_configs;
    SeparateRegConfig separate_reg_config;
    bool use_multihead = false;
    bool separate_multihead = false;
    int shared_conv_num_filter = 0;
    bool class_agnostic = false;
    std::vector<std::vector<std::string>> rpn_head_config;

    std::vector<std::string> getClassNames()
    {
        std::vector<std::string> class_names;
        for (const AnchorGeneratorConfig &config : anchor_generator_configs)
        {
            class_names.push_back(config.class_name);
        }
        return class_names;
    }
};


std::pair<std::vector<torch::Tensor>, torch::Tensor> generate_anchors(
    torch::Tensor grid_size,
    std::vector<float> point_cloud_range,
    std::vector<AnchorGeneratorConfig> anchor_generator_configs,
    int anchor_ndim = 7)
{

    AnchorGenerator anchor_generator = AnchorGenerator(
        point_cloud_range,
        anchor_generator_configs);

    std::vector<torch::Tensor> feature_map_size;
    for (auto anchor_gen_conf : anchor_generator_configs)
    {
        feature_map_size.push_back(grid_size.slice(0, 0, 2) / anchor_gen_conf.feature_map_stride);
    }

    std::cout << "feature_map_size" << feature_map_size << '\n';
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


    std::pair<torch::Tensor, torch::Tensor> generate_predicted_boxes(AnchorHeadConfig config, std::vector<torch::Tensor> anchors, ResidualCoder box_coder, int batch_size, torch::Tensor cls_preds, torch::Tensor box_preds, torch::Tensor dir_cls_preds = torch::Tensor())
    {
        // Only Single Head
        torch::Tensor anchors_tensor = torch::cat(anchors, -3);

        int num_anchors = anchors_tensor.reshape({-1, anchors_tensor.size(-1)}).size(0);
        torch::Tensor batch_anchors = anchors_tensor.view({1, -1, anchors_tensor.size(-1)}).repeat({batch_size, 1, 1});

        torch::Tensor batch_cls_preds = cls_preds.view({batch_size, num_anchors, -1}).toType(torch::kFloat);
        torch::Tensor batch_box_preds = box_preds.view({batch_size, num_anchors, -1});
        // FAIL -   what():  The size of tensor a (7) must match the size of tensor b (12) at non-singleton dimension 1
        batch_box_preds = box_coder.decode_torch(batch_box_preds, batch_anchors);

        if (dir_cls_preds.defined())
        {
            float dir_offset = config.dir_offset;
            float dir_limit_offset = config.dir_limit_offset;
            dir_cls_preds = dir_cls_preds.view({batch_size, num_anchors, -1});

            torch::Tensor dir_labels = std::get<1>(dir_cls_preds.max(-1));

            float period = (2 * M_PI / config.num_dir_bins);
            torch::Tensor dir_rot = limit_period(
                batch_box_preds.index({torch::indexing::Slice(), torch::indexing::Slice(), 6}) - dir_offset, dir_limit_offset, period);

            batch_box_preds.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), 6}, dir_rot + dir_offset + period * dir_labels.to(batch_box_preds.dtype()));
        }

        return {batch_cls_preds, batch_box_preds};
    }


    std::pair<std::vector<torch::Tensor>, torch::Tensor> generate_predicted_boxes(AnchorHeadConfig config, std::vector<torch::Tensor> anchors, ResidualCoder box_coder, int batch_size, std::vector<torch::Tensor> cls_preds, std::vector<torch::Tensor> box_preds, std::vector<torch::Tensor> dir_cls_preds = std::vector<torch::Tensor>())
    {
        // We assume we are only handling multihead
        std::vector<torch::Tensor> batch_cls_preds;
        torch::Tensor batch_box_preds;

        // torch::TensorList anchors_list(anchors);
        torch::Tensor anchors_tensor;
        std::vector<torch::Tensor> permuted_anchors;
        for (auto anchor_item : anchors){
            torch::Tensor permuted = anchor_item.permute({3, 4, 0, 1, 2, 5}).contiguous().view({-1, anchor_item.size(-1)});
            permuted_anchors.push_back(permuted);
        }
        anchors_tensor = torch::cat(permuted_anchors, 0);

        int num_anchors = anchors_tensor.reshape({-1, anchors_tensor.size(-1)}).size(0);
        torch::Tensor batch_anchors = anchors_tensor.view({1, -1, anchors_tensor.size(-1)}).repeat({batch_size, 1, 1});


        batch_cls_preds = cls_preds;
        batch_box_preds = torch::cat(torch::TensorList(box_preds), 1).view({batch_size, num_anchors, -1});

        // FAIL -   what():  The size of tensor a (7) must match the size of tensor b (12) at non-singleton dimension 1
        batch_box_preds = box_coder.decode_torch(batch_box_preds, batch_anchors);

        float dir_offset = config.dir_offset;
        float dir_limit_offset = config.dir_limit_offset;
        torch::Tensor dir_cls_preds_viewed;

        dir_cls_preds_viewed = torch::cat(torch::TensorList(dir_cls_preds), 1).view({batch_size, num_anchors, -1});

        torch::Tensor dir_labels = std::get<1>(torch::max(dir_cls_preds_viewed, -1));
        float period = (2 * M_PI / config.num_dir_bins);

        torch::Tensor dir_rot = limit_period(
            batch_box_preds.index({torch::indexing::Slice(), torch::indexing::Slice(), 6}) - dir_offset, dir_limit_offset, period);

        batch_box_preds.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), 6}, dir_rot + dir_offset + period * dir_labels.to(batch_box_preds.dtype()));

        return {batch_cls_preds, batch_box_preds};
    }