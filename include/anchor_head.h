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
    std::vector<std::vector<std::string>> rpn_head_config = {
        {{"car"}},
        {{"truck", "construction_vehicle"}},
        {{"bus", "trailer"}},
        {{"barrier"}},
        {{"motorcycle", "bicycle"}},
        {{"pedestrian", "traffic_cone"}},
    };

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

    // From Yaml Config
    std::vector<int> feature_map_stride = {2, 2, 2};

    std::vector<torch::Tensor> feature_map_size;
    for (auto &stride : feature_map_stride)
    {
        feature_map_size.push_back(grid_size.slice(0, 0, 2) / stride);
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
