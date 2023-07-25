#include <torch/torch.h>
#include <vector>

class PFNLayer : public torch::nn::Module
{

public:
    PFNLayer(int64_t in_channels,
             int64_t out_channels,
             bool use_norm = true,
             bool last_layer = false) : use_norm(use_norm), last_vfe(last_layer)
    {
        if (!last_layer)
        {
            out_channels = out_channels / 2;
        }

        if (use_norm)
        {

            // BatchNorm1d model(BatchNorm1dOptions(out_channels).eps(1e-3).momentum(0.01).affine(false).track_running_stats(true));
            this->linear = register_module("linear", torch::nn::Linear(torch::nn::LinearOptions(in_channels, out_channels).bias(false)));
            this->norm = register_module("norm", torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(out_channels).eps(1e-3).momentum(0.01)));
        }
        else
        {
            this->linear = register_module("linear", torch::nn::Linear(torch::nn::LinearOptions(in_channels, out_channels).bias(true)));
        }

        part = 50000;
    }

    torch::Tensor forward(torch::Tensor inputs)
    {
        torch::Tensor x, x_max;

        if (inputs.size(0) > part)
        {
            // nn.Linear performs randomly when batch size is too large
            int64_t num_parts = inputs.size(0) / part;
            std::vector<torch::Tensor> part_linear_out;
            for (int64_t num_part = 0; num_part <= num_parts; ++num_part)
            {
                part_linear_out.push_back(this->linear->forward(inputs.slice(0, num_part * part, (num_part + 1) * part)));
            }
            x = torch::cat(part_linear_out, 0);
        }
        else
        {
            x = this->linear->forward(inputs);
        }
        torch::manual_seed(0);
        x = use_norm ? this->norm->forward(x.transpose(0, 2)).transpose(0, 2) : x;
        torch::manual_seed(1);
        x = torch::relu(x);
        x_max = std::get<0>(x.max(1, true));

        if (last_vfe)
        {
            return x_max;
        }
        else
        {
            torch::Tensor x_repeat = x_max.repeat({1, inputs.size(1), 1});
            torch::Tensor x_concatenated = torch::cat({x, x_repeat}, 2);
            return x_concatenated;
        }
    }

private:
    torch::nn::Linear linear{nullptr};
    torch::nn::BatchNorm1d norm{nullptr};
    bool use_norm;
    bool last_vfe;
    int64_t part;

};

class PillarVFEImpl : public torch::nn::Module
{
private:
    std::vector<int32_t> num_filters;
    std::vector<std::shared_ptr<PFNLayer>> pfn_layers;
    bool use_norm;
    bool with_distance;
    bool use_absolute_xyz;
    int64_t voxel_x;
    int64_t voxel_y;
    int64_t voxel_z;
    int64_t x_offset;
    int64_t y_offset;
    int64_t z_offset;


public:
    PillarVFEImpl(std::vector<int32_t> num_filters,
              bool use_norm = true,
              bool with_distance = true,
              bool use_absolute_xyz = true,
              const std::vector<float>& voxel_size = {0.05, 0.05, 0.1},
              const std::vector<float>& point_cloud_range = {0, -40, -3, 70.4, 40, 1},
              int64_t num_point_features = 4)
    {

        this->num_filters = num_filters;
        this->use_norm = use_norm;
        this->with_distance = with_distance;
        this->use_absolute_xyz = use_absolute_xyz;

        num_point_features += use_absolute_xyz ? 6 : 3;
        if (with_distance)
        {
            num_point_features += 1;
        }

        assert(this->num_filters.size() > 0);
        this->num_filters.insert(num_filters.begin(), num_point_features);

        for (size_t i = 0; i < num_filters.size() - 1; ++i)
        {
            int64_t in_filters = num_filters[i];
            int64_t out_filters = num_filters[i + 1];
            pfn_layers.push_back(
                std::make_shared<PFNLayer>(in_filters, out_filters, this->use_norm, i >= num_filters.size() - 2));
        }

        for (size_t i = 0; i < pfn_layers.size(); ++i)
        {
            register_module("pfn_" + std::to_string(i), pfn_layers[i]);
        }
        

        this->voxel_x = voxel_size[0];
        this->voxel_y = voxel_size[1];
        this->voxel_z = voxel_size[2];

        this->x_offset = voxel_x / 2 + point_cloud_range[0];
        this->y_offset = voxel_y / 2 + point_cloud_range[1];
        this->z_offset = voxel_z / 2 + point_cloud_range[2];
    }

    int64_t get_output_feature_dim()
    {
        return this->num_filters.back();
    }

    torch::Tensor get_paddings_indicator(torch::Tensor actual_num, int max_num, int axis = 0) {
        actual_num = actual_num.unsqueeze(axis + 1);
        auto options = torch::TensorOptions().dtype(torch::kInt).device(actual_num.device());
        auto max_num_shape = std::vector<int64_t>(actual_num.sizes().size(), 1);
        max_num_shape[axis + 1] = -1;
        auto max_num_tensor = torch::arange(max_num, options).view(torch::IntArrayRef(max_num_shape.data(), max_num_shape.size()));
        auto paddings_indicator = actual_num.to(torch::kInt) > max_num_tensor;
        return paddings_indicator;
    }

    // torch::Tensor get_paddings_indicator(torch::Tensor actual_num, int64_t max_num, int64_t axis = 0)
    // {
    //     actual_num = actual_num.unsqueeze(axis + 1);
    //     torch::Tensor max_num_shape = torch::ones({actual_num.dim()}).to(torch::kLong);
    //     max_num_shape[axis + 1] = -1;
    //     torch::Tensor max_num_t = torch::arange(max_num, actual_num.device()).view(max_num_shape);
    //     torch::Tensor paddings_indicator = actual_num.to(torch::kInt) > max_num_t;
    //     return paddings_indicator;
    // }

    std::unordered_map<std::string, torch::Tensor> forward(std::unordered_map<std::string, torch::Tensor> batch_dict)
    {
        auto voxel_features = batch_dict["voxels"];
        auto voxel_num_points = batch_dict["voxel_num_points"];
        auto coords = batch_dict["voxel_coords"];

        auto points_mean = voxel_features.slice(2, 0, 3).sum(1, true) / voxel_num_points.to(voxel_features.dtype()).view({-1, 1, 1});
        auto f_cluster = voxel_features.slice(2, 0, 3) - points_mean;

        auto f_center = torch::zeros_like(voxel_features.slice(2, 0, 3));
        f_center.slice(2, 0, 1) = voxel_features.slice(2, 0, 1) - (coords.slice(1, 3, 4).to(voxel_features.dtype()).unsqueeze(1) * voxel_x + x_offset);
        f_center.slice(2, 1, 2) = voxel_features.slice(2, 1, 2) - (coords.slice(1, 2, 3).to(voxel_features.dtype()).unsqueeze(1) * voxel_y + y_offset);
        f_center.slice(2, 2, 3) = voxel_features.slice(2, 2, 3) - (coords.slice(1, 1, 2).to(voxel_features.dtype()).unsqueeze(1) * voxel_z + z_offset);

        std::vector<torch::Tensor> features;
        if (use_absolute_xyz)
        {
            features = {voxel_features, f_cluster, f_center};
        }
        else
        {
            features = {voxel_features.slice(2, 3), f_cluster, f_center};
        }

        if (with_distance)
        {
            auto points_dist = torch::norm(voxel_features.slice(2, 0, 3), 2, 2, true);
            features.push_back(points_dist);
        }
        auto combined_features = torch::cat(features, -1);

        auto voxel_count = combined_features.size(1);
        auto mask = get_paddings_indicator(voxel_num_points, voxel_count, 0);
        mask = mask.unsqueeze(-1).to(voxel_features.dtype());
        combined_features *= mask;

        for (auto &pfn : pfn_layers)
        {
            combined_features = pfn->forward(combined_features);
        }
        combined_features = combined_features.squeeze();

        batch_dict["pillar_features"] = combined_features;
        return batch_dict;
    }
};

TORCH_MODULE(PillarVFE);