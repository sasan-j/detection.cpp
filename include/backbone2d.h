#include <torch/torch.h>

#include "utils.h"

// NAME: BaseBEVBackbone
// LAYER_NUMS: [3, 5, 5]
// LAYER_STRIDES: [2, 2, 2]
// NUM_FILTERS: [64, 128, 256]
// UPSAMPLE_STRIDES: [1, 2, 4]
// NUM_UPSAMPLE_FILTERS: [128, 128, 128]

class BaseBEVBackboneImpl : public torch::nn::Module
{
public:
    BaseBEVBackboneImpl(int input_channels, std::vector<int> layer_nums, std::vector<int> layer_strides, std::vector<int> num_filters,
                        std::vector<float> upsample_strides, std::vector<int> num_upsample_filters)
        : layer_nums(layer_nums), layer_strides(layer_strides), num_filters(num_filters),
          upsample_strides(upsample_strides), num_upsample_filters(num_upsample_filters)
    {
        std::cout << layer_nums.size() << layer_strides.size() << num_filters.size() << std::endl;
        assert(layer_nums.size() == layer_strides.size() && layer_strides.size() == num_filters.size());
        assert(upsample_strides.size() == num_upsample_filters.size());

        // this->layer_nums = layer_nums;
        // this->layer_strides = layer_strides;
        // this->num_filters = num_filters;
        // this->upsample_strides = upsample_strides;
        // this->num_upsample_filters = num_upsample_filters;

        int num_levels = layer_nums.size();
        std::vector<int> c_in_list = {input_channels};
        c_in_list.insert(c_in_list.end(), num_filters.begin(), num_filters.end() - 1);

        this->blocks = torch::nn::ModuleList();
        this->deblocks = torch::nn::ModuleList();

        for (int idx = 0; idx < num_levels; ++idx)
        {
            auto cur_layers = torch::nn::Sequential(
                torch::nn::ZeroPad2d(1),
                torch::nn::Conv2d(torch::nn::Conv2dOptions(c_in_list[idx], num_filters[idx], 3).stride(layer_strides[idx]).padding(0).bias(false)),
                torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(num_filters[idx]).eps(1e-3).momentum(0.01)),
                torch::nn::ReLU());
            for (int k = 0; k < layer_nums[idx]; ++k)
            {
                cur_layers->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(num_filters[idx], num_filters[idx], 3).padding(1).bias(false)));
                cur_layers->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(num_filters[idx]).eps(1e-3).momentum(0.01)));
                cur_layers->push_back(torch::nn::ReLU());
            }
            // register_module("blocks_" + std::to_string(idx), cur_layers);
            // this->blocks.push_back(cur_layers);
            this->blocks->push_back(cur_layers);

            if (upsample_strides.size() > 0)
            {
                float stride = upsample_strides[idx];
                if (stride > 1 || (stride == 1. && ~this->USE_CONV_FOR_NO_STRIDE))
                {
                    auto deblock_layers = torch::nn::Sequential(
                        torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(num_filters[idx], num_upsample_filters[idx], upsample_strides[idx]).stride(upsample_strides[idx]).bias(false)),
                        torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(num_upsample_filters[idx]).eps(1e-3).momentum(0.01)),
                        torch::nn::ReLU());
                    // register_module("deblocks_" + std::to_string(idx), deblock_layers);
                    auto named_params = deblock_layers->named_parameters();
                    for (auto iter = named_params.begin(); iter != named_params.end(); ++iter)
                    {
                        std::cout << iter->key() << iter->value().sizes() << std::endl;
                    }
                    std::cout << deblock_layers << std::endl;
                    this->deblocks->push_back(deblock_layers);
                }
                else
                {
                    stride = static_cast<int>(round(1.0 / stride));
                    auto deblock_layers = torch::nn::Sequential(
                        torch::nn::Conv2d(torch::nn::Conv2dOptions(num_filters[idx], num_upsample_filters[idx], stride).stride(stride).bias(false)),
                        torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(num_upsample_filters[idx]).eps(1e-3).momentum(0.01)),
                        torch::nn::ReLU());
                    // register_module("deblocks_" + std::to_string(idx), deblock_layers);

                    auto named_params = deblock_layers->named_parameters();
                    for (auto iter = named_params.begin(); iter != named_params.end(); ++iter)
                    {
                        std::cout << iter->key() << iter->value().sizes() << std::endl;
                    }
                    std::cout << deblock_layers << std::endl;
                    this->deblocks->push_back(deblock_layers);
                }
            }
        }

        int c_in = std::accumulate(num_upsample_filters.begin(), num_upsample_filters.end(), 0);
        if (upsample_strides.size() > num_levels)
        {
            auto deblock_layers = torch::nn::Sequential(
                torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(c_in, c_in, upsample_strides.back()).stride(upsample_strides.back()).bias(false)),
                torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(c_in).eps(1e-3).momentum(0.01)),
                torch::nn::ReLU());
            // register_module("deblock_last", deblock_layers);
            this->deblocks->push_back(deblock_layers);
        }

        std::cout << "blocks: " << this->blocks << std::endl;
        std::cout << "deblocks: " << this->deblocks << std::endl;

        register_module("blocks", this->blocks);
        register_module("deblocks", this->deblocks);

        num_bev_features = c_in;

        std::cout << "NAMED PARAMETERS: " << std::endl;
        auto named_params = this->named_parameters();
        for (auto iter = named_params.begin(); iter != named_params.end(); ++iter)
        {
            std::cout << iter->key() << std::endl;
        }
    }

    BatchMap forward(BatchMap data_dict)
    {
        torch::Tensor spatial_features = data_dict["spatial_features"];
        std::vector<torch::Tensor> ups;
        std::unordered_map<std::string, torch::Tensor> ret_dict;
        torch::Tensor x = spatial_features;

        // for (auto layer : this->blocks)
        // {
        //     x = layer->forward(x);
        //     ups.push_back(x);
        // }

        for (size_t i = 0; i < this->blocks->size(); ++i)
        {
            x = blocks[i]->as<torch::nn::Sequential>()->forward(x);

            std::cout << "spatial_features: size(2) and sizes()" << spatial_features.size(2) << spatial_features.sizes() << std::endl;
            int stride = static_cast<int>(spatial_features.size(2) / x.size(2));
            ret_dict["spatial_features_" + std::to_string(stride) + "x"] = x;

            if (this->deblocks->size() > 0)
            {
                ups.push_back(this->deblocks[i]->as<torch::nn::Sequential>()->forward(x));
            }
            else
            {
                ups.push_back(x);
            }
        }

        if (ups.size() > 1)
        {
            x = torch::cat(ups, 1);
        }
        else if (ups.size() == 1)
        {
            x = ups[0];
        }

        if (this->deblocks->size() > this->blocks->size())
        {
            x = this->deblocks[this->deblocks->size() - 1]->as<torch::nn::Sequential>()->forward(x);
        }

        data_dict["spatial_features_2d"] = x;

        return data_dict;
    }

private:
    bool USE_CONV_FOR_NO_STRIDE = false;
    std::vector<int> layer_nums;
    std::vector<int> layer_strides;
    std::vector<int> num_filters;
    std::vector<float> upsample_strides;
    std::vector<int> num_upsample_filters;
    torch::nn::ModuleList blocks{nullptr};
    torch::nn::ModuleList deblocks{nullptr};
    int num_bev_features;
};

TORCH_MODULE(BaseBEVBackbone);