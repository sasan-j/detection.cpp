#pragma once

#include <torch/torch.h>


class PointPillarScatterImpl : public torch::nn::Module {

public:
    PointPillarScatterImpl(int num_bev_features, int nx, int ny, int nz) {
        assert(nz == 1);
        this->num_bev_features_ = num_bev_features;
        this->nx_ = nx;
        this->ny_ = ny;
        this->nz_ = nz;
    }

    std::unordered_map<std::string, torch::Tensor> forward(std::unordered_map<std::string, torch::Tensor> batch_dict) {
        auto pillar_features = batch_dict["pillar_features"];
        auto coords = batch_dict["voxel_coords"];

        std::vector<torch::Tensor> batch_spatial_features;
        int batch_size = coords.index({torch::indexing::Slice(), 0}).max().item<int>() + 1;
        for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            auto spatial_feature = torch::zeros({this->num_bev_features_, nz_ * nx_ * ny_},
                                                pillar_features.options());

            auto batch_mask = coords.index({torch::indexing::Slice(), 0}) == batch_idx;
            auto this_coords = coords.index({batch_mask, torch::indexing::Slice()});
            auto indices = this_coords.index({torch::indexing::Slice(), 1}) +
                           this_coords.index({torch::indexing::Slice(), 2}) * nx_ +
                           this_coords.index({torch::indexing::Slice(), 3});
            indices = indices.to(torch::kLong);
            auto pillars = pillar_features.index({batch_mask, torch::indexing::Slice()});
            pillars = pillars.transpose(0, 1);
            spatial_feature.index_put_({torch::indexing::Slice(), indices}, pillars);
            batch_spatial_features.push_back(spatial_feature);
        }

        auto batch_spatial_features_tensor = torch::stack(batch_spatial_features, 0);
        batch_spatial_features_tensor = batch_spatial_features_tensor.view({batch_size, this->num_bev_features_ * nz_, ny_, nx_});
        batch_dict["spatial_features"] = batch_spatial_features_tensor;
        return batch_dict;
    }

private:
    int num_bev_features_;
    int nx_;
    int ny_;
    int nz_;
};


TORCH_MODULE(PointPillarScatter);