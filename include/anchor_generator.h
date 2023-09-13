#pragma once

#include <torch/torch.h>
#include <vector>
#include <map>


struct AnchorGeneratorConfig
{
    std::string class_name;
    std::vector<std::vector<float>> anchor_sizes;
    std::vector<float> anchor_rotations;
    std::vector<float> anchor_bottom_heights;
    bool align_center;
    int feature_map_stride;
    float matched_threshold;
    float unmatched_threshold;
};

class AnchorGenerator
{
public:
    std::vector<std::vector<std::vector<float>>> anchor_sizes;
    std::vector<std::vector<float>> anchor_rotations;
    std::vector<std::vector<float>> anchor_bottom_heights;
    std::vector<bool> align_center;
    int num_of_anchor_sets;
    std::vector<float> anchor_range;

    AnchorGenerator(std::vector<float> anchor_range,
                    std::vector<AnchorGeneratorConfig> anchor_generator_configs)
    {
        this->anchor_range = anchor_range;

        for (const AnchorGeneratorConfig &config : anchor_generator_configs)
        {
            this->anchor_sizes.push_back(config.anchor_sizes);
        }

        for (const AnchorGeneratorConfig &config : anchor_generator_configs)
        {
            this->anchor_rotations.push_back(config.anchor_rotations);
        }

        for (const AnchorGeneratorConfig &config : anchor_generator_configs)
        {
            this->anchor_bottom_heights.push_back(config.anchor_bottom_heights);
        }

        for (const AnchorGeneratorConfig &config : anchor_generator_configs)
        {
            this->align_center.push_back(config.align_center);
        }

        assert(anchor_sizes.size() == anchor_rotations.size() && anchor_sizes.size() == anchor_bottom_heights.size());
        this->num_of_anchor_sets = anchor_sizes.size();
    }

    std::pair<std::vector<torch::Tensor>, torch::Tensor> generate_anchors(std::vector<torch::Tensor> grid_sizes)
    {
        // std::cout << "grid_size.size(): " << grid_sizes.size() << " num_of_anchor_sets: " << num_of_anchor_sets << '\n';
        assert(grid_sizes.size() == num_of_anchor_sets);

        std::vector<torch::Tensor> all_anchors;
        std::vector<int32_t> num_anchors_per_location;

        // std::cout << "grid_sizes" << grid_sizes << '\n';

        for (int i = 0; i < num_of_anchor_sets; i++)
        {
            auto grid_size = grid_sizes[i];
            auto anchor_size = anchor_sizes[i];
            auto anchor_rotation = anchor_rotations[i];
            auto anchor_height = anchor_bottom_heights[i];
            auto align_center_i = align_center[i];

            // std::cout << "anchor_size " << anchor_size << " size: " << anchor_size.size() << '\n';
            // std::cout << "anchor_height " << anchor_height << " size: " << anchor_height.size() << '\n';
            // std::cout << "anchor_rotation " << anchor_rotation << " size: " << anchor_rotation.size() << '\n';
            // std::cout << "grid_size " << grid_size << " size: " << grid_size.sizes() << '\n';
            // std::cout << "align_center_i " << align_center_i << '\n';

            num_anchors_per_location.push_back(anchor_rotation.size() * anchor_size.size() * anchor_height.size());

            float x_stride, y_stride, x_offset, y_offset;
            if (align_center_i)
            {
                x_stride = (anchor_range[3] - anchor_range[0]) / grid_size[0].item<float>();
                y_stride = (anchor_range[4] - anchor_range[1]) / grid_size[1].item<float>();
                x_offset = x_stride / 2;
                y_offset = y_stride / 2;
            }
            else
            {
                x_stride = (anchor_range[3] - anchor_range[0]) / (grid_size[0].item<float>() - 1);
                y_stride = (anchor_range[4] - anchor_range[1]) / (grid_size[1].item<float>() - 1);
                x_offset = 0;
                y_offset = 0;
            }

            auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
            auto x_shifts = torch::arange(anchor_range[0] + x_offset, anchor_range[3] + 1e-5, x_stride, options);
            auto y_shifts = torch::arange(anchor_range[1] + y_offset, anchor_range[4] + 1e-5, y_stride, options);
            auto z_shifts = torch::tensor(anchor_height, options);

            int num_anchor_size = anchor_size.size();
            int num_anchor_rotation = anchor_rotation.size();
            auto anchor_rotation_tensor = torch::tensor(anchor_rotation, options);
            auto anchor_size_tensor = torch::tensor(anchor_size[0], options).reshape({1, -1});
            auto anchor_grid = torch::meshgrid({x_shifts, y_shifts, z_shifts});
            auto anchors = torch::stack(anchor_grid, -1);
            anchors = anchors.unsqueeze(3).expand({-1, -1, -1, num_anchor_size, -1});
            anchor_size_tensor = anchor_size_tensor.view({1, 1, 1, -1, 3}).expand_as(anchors);
            anchors = torch::cat({anchors, anchor_size_tensor}, -1);
            anchors = anchors.unsqueeze(4).expand({-1, -1, -1, -1, num_anchor_rotation, -1});
            anchor_rotation_tensor = anchor_rotation_tensor.view({1, 1, 1, 1, -1, 1}).expand_as(anchors);
            anchors = torch::cat({anchors, anchor_rotation_tensor}, -1);
            anchors = anchors.permute({2, 1, 0, 3, 4, 5}).contiguous();
            anchors.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), 2}) += anchors.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), 5}) / 2;
            all_anchors.push_back(anchors);
            // std::cout << "bottom of loop" << '\n';
        }

        auto options = torch::TensorOptions().dtype(torch::kInt32); // or whatever data type num_anchors_per_location holds
        // auto out = torch::from_blob(num_anchors_per_location.data(), {num_anchors_per_location.size()}, options);
        auto out = torch::tensor(num_anchors_per_location, options);

        // std::cout << "out " << out << '\n';
        return {all_anchors, out};
    }
};
