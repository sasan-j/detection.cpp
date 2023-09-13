#pragma once

#include <torch/torch.h>
#include <cmath>

#include "utils.h"
#include "anchor_head.h"
#include "backbone2d.h"
#include "model.h"

class SingleHeadImpl : public torch::nn::Module
{
public:
    SingleHeadImpl(ModelConfig model_config, int input_channels, int num_class, int num_anchors_per_location, int code_size, torch::Tensor head_label_indices, SeparateRegConfig separate_reg_config) : num_anchors_per_location(num_anchors_per_location),
                                                                                                                                                                                                    num_class(num_class),
                                                                                                                                                                                                    code_size(code_size), head_label_indices(head_label_indices), separate_reg_config(separate_reg_config)
    {
        this->num_dir_bins = model_config.anchor_head_config.num_dir_bins;
        this->use_direction_classifier = model_config.anchor_head_config.use_direction_classifier;
        register_buffer("head_label_indices", head_label_indices);


        ModelConfig backbone_config = ModelConfig();


        // Initialize Backbone
        this->backbone = BaseBEVBackbone(backbone_config, input_channels);

        int code_size_cnt = 0;
        // torch::nn::Sequential conv_box;
        // std::unordered_map<std::string, torch::nn::Module> conv_box;
        int num_middle_conv = separate_reg_config.num_middle_conv;
        int num_middle_filter = separate_reg_config.num_middle_filter;
        int c_in = input_channels;

        conv_cls = torch::nn::Sequential();

        for (int k = 0; k < num_middle_conv; ++k)
        {
            conv_cls->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(c_in, num_middle_filter, 3)
                                                             .stride(1)
                                                             .padding(1)
                                                             .bias(false)));
            conv_cls->push_back(
                torch::nn::BatchNorm2d(num_middle_filter));
            conv_cls->push_back(torch::nn::ReLU());
            c_in = num_middle_filter;
        }

        conv_cls->push_back(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(c_in, num_anchors_per_location * num_class, 3)
                                  .stride(1)
                                  .padding(1)));

        register_module("conv_cls", this->conv_cls);

        // std::vector<std::pair<std::string, std::shared_ptr<torch::nn::Module>>> conv_box_vect;

        torch::OrderedDict<std::string, std::shared_ptr<torch::nn::Module>> conv_box_od;
        for (const std::string &reg_config : separate_reg_config.reg_list)
        {
            // auto conv_seq =  std::shared_ptr<CustomSeqModuleImpl>();
            auto conv_seq = torch::nn::Sequential();
            // auto conv_seq = std::shared_ptr<torch::nn::Sequential>();

            std::vector<std::string> reg_split = split_string(reg_config, ':');
            std::string reg_name = reg_split[0];
            int reg_channel = std::stoi(reg_split[1]);

            std::vector<torch::nn::AnyModule> cur_conv_list;
            c_in = input_channels;

            for (int k = 0; k < num_middle_conv; ++k)
            {
                auto tmp_conv2d = torch::nn::Conv2d(torch::nn::Conv2dOptions(c_in, num_middle_filter, 3)
                                                        .stride(1)
                                                        .padding(1)
                                                        .bias(false));
                torch::nn::init::kaiming_normal_(tmp_conv2d->weight, 0, torch::kFanOut, torch::kReLU);
                // conv_seq->sequential->push_back(tmp_conv2d);
                // conv_seq->sequential->push_back(torch::nn::BatchNorm2d(num_middle_filter));
                // conv_seq->sequential->push_back(torch::nn::ReLU());
                conv_seq->push_back(tmp_conv2d);
                conv_seq->push_back(torch::nn::BatchNorm2d(num_middle_filter));
                conv_seq->push_back(torch::nn::ReLU());
                c_in = num_middle_filter;
            }
            auto tmp_conv2d_2 = torch::nn::Conv2d(torch::nn::Conv2dOptions(c_in, num_anchors_per_location * reg_channel, 3)
                                                      .stride(1)
                                                      .padding(1)
                                                      .bias(true));
            torch::nn::init::kaiming_normal_(tmp_conv2d_2->weight, 0, torch::kFanOut, torch::kReLU);
            torch::nn::init::constant_(tmp_conv2d_2->bias, 0);
            // conv_seq->sequential->push_back(tmp_conv2d_2);
            conv_seq->push_back(tmp_conv2d_2);

            code_size_cnt += reg_channel;
            char buffer[255];
            std::sprintf(buffer, "conv_%s", reg_name.c_str());
            const std::string name(buffer);
            // auto conv_seq = torch::nn::Sequential(cur_conv_list.begin(), cur_conv_list.end());
            // conv_box->insert(name, std::static_pointer_cast<torch::nn::Module>(conv_seq));
            conv_box_od.insert(name, conv_seq.ptr());
            // conv_box_vect.push_back(std::make_pair(name, conv_seq->as<torch::nn::Module>());
            // register_module(name, conv_seq);
            // Working
            // conv_box->push_back(conv_seq);
            conv_box_names.push_back(name);
            conv_box_map[name] = conv_box->size() - 1;
            // register_module(name, conv_seq);
        }
        // conv_box->update(conv_box_vect);
        conv_box->update(conv_box_od);

        register_module("conv_box", this->conv_box);

        assert(code_size_cnt == code_size);

        if (model_config.anchor_head_config.use_direction_classifier)
        {
            this->conv_dir_cls = torch::nn::Conv2d(torch::nn::Conv2dOptions(input_channels, num_anchors_per_location * model_config.anchor_head_config.num_dir_bins, 1));
            register_module("conv_dir_cls", this->conv_dir_cls);
        }

        this->use_multihead = model_config.anchor_head_config.use_multihead;

        // Perform weight initialization as in Python code
        init_weights();
    }

    void init_weights()
    {
        float pi = 0.01;
        if (conv_cls)
        {
            auto conv_cls_last = 
                conv_cls->children().at(conv_cls->children().size() - 1)->as<torch::nn::Conv2d>();

            torch::nn::init::constant_(conv_cls_last->bias, -std::log((1 - pi) / pi));
        }
    }

    // std::map<std::string, torch::Tensor> forward(torch::Tensor spatialFeatures2D)
    // {
    //     std::map<std::string, torch::Tensor> retDict;

    //     // Implement the forward pass as in Python code
    //     // ...
    //     // Calculate cls_preds, box_preds, and dir_cls_preds
    //     // ...

    //     return retDict;
    // }

    BatchMap forward(torch::Tensor spatial_features_2d)
    {
        BatchMap ret_dict;

        std::cout << "before backbone spatial_features_2d: " << spatial_features_2d.sizes() << '\n';

        // Call the base class's forward method
        auto super_output = this->backbone->forward({BatchMap{{"spatial_features", spatial_features_2d}}});
        spatial_features_2d = super_output.at("spatial_features_2d");

        // Printout the spatial_features_2d tensor
        std::cout << "after backbone spatial_features_2d: " << spatial_features_2d.sizes() << '\n';

        torch::Tensor cls_preds = this->conv_cls->forward(spatial_features_2d);

        torch::Tensor box_preds;

        std::vector<torch::Tensor> box_preds_list;
        for (const std::string& reg_name : this->conv_box_names) {
            box_preds_list.push_back(this->conv_box[reg_name]->as<torch::nn::Sequential>()->forward(spatial_features_2d));
            // box_preds_list.push_back(this->conv_box[conv_box_map[reg_name]]->as<torch::nn::Sequential>()->forward(spatial_features_2d));
        }
        box_preds = torch::cat(box_preds_list, 1);
    

        if (!this->use_multihead) {
            box_preds = box_preds.permute({0, 2, 3, 1}).contiguous();
            cls_preds = cls_preds.permute({0, 2, 3, 1}).contiguous();
        } else {
            int H = box_preds.size(2);
            int W = box_preds.size(3);
            int batch_size = box_preds.size(0);

            box_preds = box_preds.view({-1, this->num_anchors_per_location, this->code_size, H, W})
                            .permute({0, 1, 3, 4, 2})
                            .contiguous();
            cls_preds = cls_preds.view({-1, this->num_anchors_per_location, this->num_class, H, W})
                            .permute({0, 1, 3, 4, 2})
                            .contiguous();
            box_preds = box_preds.view({batch_size, -1, this->code_size});
            cls_preds = cls_preds.view({batch_size, -1, this->num_class});
        }

        torch::Tensor dir_cls_preds;
        if (this->use_direction_classifier){
            dir_cls_preds = this->conv_dir_cls->forward(spatial_features_2d);
            if (this->use_multihead) {
                int H = box_preds.size(2);
                int W = box_preds.size(3);
                int batch_size = box_preds.size(0);
                dir_cls_preds = dir_cls_preds.view({-1, this->num_anchors_per_location, this->num_dir_bins, H, W})
                                    .permute({0, 1, 3, 4, 2})
                                    .contiguous();
                dir_cls_preds = dir_cls_preds.view({batch_size, -1, this->num_dir_bins});
            } else {
                dir_cls_preds = dir_cls_preds.permute({0, 2, 3, 1}).contiguous();
            }
        }

        ret_dict["cls_preds"] = cls_preds;
        ret_dict["box_preds"] = box_preds;
        ret_dict["dir_cls_preds"] = dir_cls_preds;

        return ret_dict;
    }

    torch::Tensor head_label_indices;
    int num_class;
private:
    int num_dir_bins;
    int num_anchors_per_location;
    int code_size;
    bool use_multihead;
    bool use_direction_classifier;
    std::vector<std::string> conv_box_names;
    BaseBEVBackbone backbone{nullptr};
    SeparateRegConfig separate_reg_config;
    torch::nn::Sequential conv_cls;
    torch::nn::ModuleDict conv_box;
    // torch::nn::ModuleList conv_box;
    torch::nn::Conv2d conv_dir_cls{nullptr};
    std::unordered_map<std::string, int> conv_box_map;
};

TORCH_MODULE(SingleHead);

class AnchorHeadMultiImpl : public torch::nn::Module
{
public:
    AnchorHeadMultiImpl(ModelConfig model_config, std::vector<float> point_cloud_range, torch::Tensor grid_size, int input_channels) : box_coder(ResidualCoder())
    {
        this->config = model_config.anchor_head_config;
        this->model_config = model_config;
        this->num_class = model_config.anchor_head_config.getClassNames().size();

        // Initialization as per template

        auto anchors = generate_anchors(grid_size, point_cloud_range, config.anchor_generator_configs, box_coder.code_size);
        this->anchors = anchors.first;
        this->num_anchors_per_location = anchors.second;
        for (auto &anchor : this->anchors)
        {
            anchor = anchor.to(torch::kCUDA);
        }

        if (config.target_assigner_config.name == "AxisAlignedTargetAssigner")
        {
            this->target_assigner = AxisAlignedTargetAssigner(this->config.target_assigner_config, this->config.getClassNames(), this->box_coder);
        }
        else
        {
            std::cout << "Target Assigner not implemented" << '\n';
        }

        // Multihead Constructor
        if (config.shared_conv_num_filter != 0)
        {
            this->shared_conv = torch::nn::Sequential(
                torch::nn::Conv2d(torch::nn::Conv2dOptions(input_channels, config.shared_conv_num_filter, 3).stride(1).padding(1).bias(false)),
                torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(config.shared_conv_num_filter).eps(1e-3).momentum(0.01)),
                torch::nn::ReLU());
            register_module("shared_conv", this->shared_conv);
        }
        else
        {
            this->shared_conv = nullptr;
            config.shared_conv_num_filter = input_channels;
        }

        // std::cout << "num_anchors_per_location" << this->num_anchors_per_location.sizes() << this->num_anchors_per_location << '\n';

        // this->rpn_heads = nullptr;
        this->make_multihead(config.shared_conv_num_filter);
    }


    void make_multihead(int input_channels)
    {
        std::vector<std::string> classNames = this->config.getClassNames();

        for (const auto &rpnHeadCfg : this->config.rpn_head_config)
        {
            int num_anchors_per_location = 0;
            std::vector<int> head_label_indices_vect;

            for (const auto &headClsName : rpnHeadCfg)
            {
                int index = -1;
                for (int i = 0; i < classNames.size(); ++i)
                {
                    if (classNames[i] == headClsName)
                    {
                        index = i;
                        break;
                    }
                }
                if (index != -1)
                {
                    num_anchors_per_location += this->num_anchors_per_location[index].item<int>();
                    head_label_indices_vect.push_back(index + 1);
                }
            }

            torch::Tensor head_label_indices = torch::from_blob(head_label_indices_vect.data(), {static_cast<long int>(head_label_indices_vect.size())}, torch::kInt);

            int num_class = this->num_class;
            if (this->config.use_multihead)
            {
                num_class = rpnHeadCfg.size();
            }

            SingleHead rpnHead = SingleHead(
                this->model_config,
                input_channels, num_class, num_anchors_per_location, this->box_coder.code_size,
                head_label_indices,
                this->config.separate_reg_config);

            rpn_heads->push_back(rpnHead);
        }
        register_module("rpn_heads", rpn_heads);
    }


    BatchData forward(BatchMap data_dict) {
        torch::Tensor spatial_features_2d = data_dict["spatial_features_2d"];
        if (this->config.shared_conv_num_filter != 0){
            spatial_features_2d = shared_conv->forward(spatial_features_2d);
        }

        std::vector<BatchMap> ret_dicts;
        for (auto rpn_head : rpn_heads->children()) {
            ret_dicts.push_back(rpn_head->as<SingleHead>()->forward(spatial_features_2d));
        }

        std::vector<torch::Tensor> cls_preds;
        std::vector<torch::Tensor> box_preds;
        std::vector<torch::Tensor> dir_cls_preds;

        for (auto ret_dict : ret_dicts) {
            cls_preds.push_back(ret_dict["cls_preds"]);
            box_preds.push_back(ret_dict["box_preds"]);
            if (config.use_direction_classifier) {
                dir_cls_preds.push_back(ret_dict["dir_cls_preds"]);
            }
        }

        std::unordered_map<std::string, std::vector<torch::Tensor>> ret;

        if (config.separate_multihead) {
            ret["cls_preds"] = cls_preds;
            ret["box_preds"] = box_preds;
            if (config.use_direction_classifier) {
                ret["dir_cls_preds"] = dir_cls_preds;
            }
        }

        std::cout << "cls_preds: " << cls_preds.size() << '\n';
        std::cout << "box_preds: " << box_preds.size() << '\n';
        std::cout << "dir_cls_preds: " << dir_cls_preds.size() << '\n';

        //  else {
        //     ret["cls_preds"] = torch::cat(cls_preds, 1);
        //     ret["box_preds"] = torch::cat(box_preds, 1);
        //     if (config.use_direction_classifier) {
        //         ret["dir_cls_preds"] = torch::cat(dir_cls_preds, 1);
        //     }
        // }

        std::pair<std::vector<torch::Tensor>, torch::Tensor> result = generate_predicted_boxes(
            this->config,
            this->anchors,
            this->box_coder,
            data_dict["batch_size"].item<int>(),
            ret["cls_preds"],
            ret["box_preds"],
            ret["dir_cls_preds"]
        );

        auto batch_cls_preds = result.first;

        std::vector<torch::Tensor> multihead_label_mapping;
        for (size_t idx = 0; idx < batch_cls_preds.size(); ++idx) {
            std::cout << "head_label_indices: " << rpn_heads[idx]->as<SingleHead>()->head_label_indices << '\n';
            multihead_label_mapping.push_back(rpn_heads[idx]->as<SingleHead>()->head_label_indices);
        }

        BatchData output;
        output.tensor_dict = data_dict;
        output.tensor_dict["cls_preds_normalized"] = torch::tensor({false});
        output.vector_dict["multihead_label_mapping"] = multihead_label_mapping;
        output.vector_dict["batch_cls_preds"] = batch_cls_preds;
        output.tensor_dict["batch_box_preds"] = result.second;

        return output;
    }

    int num_class;
private:
    AnchorHeadConfig config;
    ModelConfig model_config;
    torch::nn::Sequential shared_conv;
    std::vector<torch::Tensor> anchors;
    torch::Tensor num_anchors_per_location;
    AxisAlignedTargetAssigner target_assigner;
    ResidualCoder box_coder;
    torch::nn::ModuleList rpn_heads;
    // Define other member variables here
};

TORCH_MODULE(AnchorHeadMulti);
