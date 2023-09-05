#pragma once

#include <torch/torch.h>
#include <vector>
#include <map>

#include "utils.h"

struct TargetAssignerConfig {
    std::string name = "AxisAlignedTargetAssigner";
    float pos_fraction = -1.0;
    int sample_size = 512;
    bool norm_by_num_examples = false;
    bool match_height = false;
    std::string box_coder = "ResidualCoder";
    int box_coder_code_size = -1;
    bool box_coder_encode_angle_by_sincos = false;
};

// TARGET_ASSIGNER_CONFIG:
//     NAME: AxisAlignedTargetAssigner
//     POS_FRACTION: -1.0
//     SAMPLE_SIZE: 512
//     NORM_BY_NUM_EXAMPLES: False
//     MATCH_HEIGHT: False
//     BOX_CODER: ResidualCoder

class AxisAlignedTargetAssigner
{
public:
    AxisAlignedTargetAssigner()
    {
    }

    AxisAlignedTargetAssigner(
        TargetAssignerConfig config,
        std::vector<std::string> class_names,
        ResidualCoder box_coder,
        int sample_size = 512,
        bool norm_by_num_examples = false,
        bool use_multihead = false) : config(config)
    {
        this->box_coder = box_coder;
        this->class_names = class_names;
        this->anchor_class_names = class_names;
        this->sample_size = sample_size;
        this->norm_by_num_examples = norm_by_num_examples;
        this->use_multihead = use_multihead;

        this->matched_thresholds = {
            {"Car", 0.6},
            {"Pedestrian", 0.5},
            {"Cyclist", 0.5}};
        this->unmatched_thresholds = {{"Car", 0.45}, {"Pedestrian", 0.35}, {"Cyclist", 0.35}};

        this->pos_fraction = (this->config.pos_fraction >= 0) ? this->config.pos_fraction : -1;
    }

    std::unordered_map<std::string, torch::Tensor> assign_targets(std::vector<torch::Tensor> all_anchors, torch::Tensor gt_boxes_with_classes)
    {
        std::vector<torch::Tensor> bbox_targets;
        std::vector<torch::Tensor> cls_labels;
        std::vector<torch::Tensor> reg_weights;

        int batch_size = gt_boxes_with_classes.size(0);
        auto gt_classes = gt_boxes_with_classes.index({torch::indexing::Slice(), torch::indexing::Slice(), -1});
        auto gt_boxes = gt_boxes_with_classes.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice({torch::indexing::None}, -1)});

        for (int k = 0; k < batch_size; ++k)
        {
            auto cur_gt = gt_boxes[k];
            int cnt = cur_gt.size(0) - 1;
            while (cnt > 0 && cur_gt[cnt].sum().item<float>() == 0)
            {
                --cnt;
            }
            cur_gt = cur_gt.slice(0, 0, cnt + 1);
            auto cur_gt_classes = gt_classes[k].slice(0, 0, cnt + 1).to(torch::kInt);
            torch::Tensor feature_map_size;

            std::vector<std::unordered_map<std::string, torch::Tensor>> target_list;
            for (int i = 0; i < anchor_class_names.size(); ++i)
            {
                auto anchor_class_name = anchor_class_names[i];
                auto anchors = all_anchors[i];

                torch::Tensor mask;

                if (cur_gt_classes.size(0) > 1)
                {
                    std::vector<torch::Tensor> mask_elems;
                    for (int64_t i = 0; i < cur_gt_classes.size(0); ++i)
                    {
                        int64_t c = cur_gt_classes[i].item<int64_t>() - 1;
                        mask_elems.push_back(torch::full({}, this->class_names[c] == anchor_class_name, torch::kBool));
                    }
                    mask = torch::stack(mask_elems);
                }
                else
                {
                    std::vector<torch::Tensor> mask_data;
                    for (int64_t i = 0; i < cur_gt_classes.size(0); ++i)
                    {
                        int64_t c = cur_gt_classes[i].item<int64_t>() - 1;
                        mask_data.push_back(torch::full({}, this->class_names[c] == anchor_class_name, torch::kBool));
                    }
                    mask = torch::stack(mask_data);
                }

                feature_map_size = torch::tensor((anchors.sizes().slice(0, 3)).data(), torch::kLong);
                anchors = anchors.reshape({-1, anchors.size(-1)});
                auto selected_classes = cur_gt_classes.index({mask});

                auto single_target = assign_targets_single(
                    anchors,
                    cur_gt.index({mask}),
                    selected_classes,
                    matched_thresholds[anchor_class_name],
                    unmatched_thresholds[anchor_class_name]);
                target_list.push_back(single_target);
            }

            std::vector<torch::Tensor> box_cls_labels, box_reg_targets, reg_weights;

            for (auto &t : target_list)
            {
                box_cls_labels.push_back(t["box_cls_labels"].view({-1}));
                box_reg_targets.push_back(t["box_reg_targets"].view({-1, this->box_coder.code_size}));
                reg_weights.push_back(t["reg_weights"].view({-1}));
            }

            std::unordered_map<std::string, std::vector<torch::Tensor>> target_dict = {
                {"box_cls_labels", box_cls_labels},
                {"box_reg_targets", box_reg_targets},
                {"reg_weights", reg_weights}};

            std::unordered_map<std::string, torch::Tensor> target_dict_refined;

            target_dict_refined["box_reg_targets"] = torch::cat(target_dict["box_reg_targets"], -2).view({-1, this->box_coder.code_size});
            target_dict_refined["box_cls_labels"] = torch::cat(target_dict["box_cls_labels"], -1).view({-1});
            target_dict_refined["reg_weights"] = torch::cat(target_dict["reg_weights"], -1).view({-1});

            bbox_targets.push_back(target_dict_refined["box_reg_targets"]);
            cls_labels.push_back(target_dict_refined["box_cls_labels"]);
            reg_weights.push_back(target_dict_refined["reg_weights"]);
        }

        auto bbox_targets_t = torch::stack(bbox_targets, 0);
        auto cls_labels_t = torch::stack(cls_labels, 0);
        auto reg_weights_t = torch::stack(reg_weights, 0);

        std::unordered_map<std::string, torch::Tensor> all_targets_dict = {
            {"box_cls_labels", cls_labels_t},
            {"box_reg_targets", bbox_targets_t},
            {"reg_weights", reg_weights_t}};
        return all_targets_dict;
    }

    std::unordered_map<std::string, torch::Tensor> assign_targets_single(
        torch::Tensor anchors, torch::Tensor gt_boxes, torch::Tensor gt_classes,
        double matched_threshold, double unmatched_threshold)
    {
        // Initialize tensors
        int num_anchors = anchors.size(0);
        int num_gt = gt_boxes.size(0);

        torch::Tensor labels = torch::ones({num_anchors}).to(anchors.device()) * -1;
        torch::Tensor gt_ids = torch::ones({num_anchors}).to(anchors.device()) * -1;

        torch::Tensor anchor_to_gt_argmax, anchor_to_gt_max, gt_to_anchor_argmax, gt_to_anchor_max, anchors_with_max_overlap;
        torch::Tensor gt_inds_force, pos_inds, gt_inds_over_thresh, bg_inds;

        if (gt_boxes.size(0) > 0 && anchors.size(0) > 0)
        {
            torch::Tensor anchor_by_gt_overlap;
            if (this->config.match_height)
            {
                anchor_by_gt_overlap = boxes_iou3d_gpu(anchors.index({torch::indexing::Slice(), torch::indexing::Slice(0, 7)}),
                                                       gt_boxes.index({torch::indexing::Slice(), torch::indexing::Slice(0, 7)}));
            }
            else
            {
                anchor_by_gt_overlap = boxes3d_nearest_bev_iou(anchors.index({torch::indexing::Slice(), torch::indexing::Slice(0, 7)}),
                                                               gt_boxes.index({torch::indexing::Slice(), torch::indexing::Slice(0, 7)}));
            }

            anchor_to_gt_argmax = anchor_by_gt_overlap.argmax(1);
            anchor_to_gt_max = anchor_by_gt_overlap.index({torch::arange(num_anchors, anchors.options().device()), anchor_to_gt_argmax});

            gt_to_anchor_argmax = anchor_by_gt_overlap.argmax(0);
            gt_to_anchor_max = anchor_by_gt_overlap.index({gt_to_anchor_argmax, torch::arange(num_gt, anchors.options().device())});

            torch::Tensor empty_gt_mask = gt_to_anchor_max == 0;
            gt_to_anchor_max.index_put_({empty_gt_mask}, -1);

            anchors_with_max_overlap = (anchor_by_gt_overlap == gt_to_anchor_max).nonzero().index({torch::indexing::Slice(), 0});
            gt_inds_force = anchor_to_gt_argmax.index({anchors_with_max_overlap});
            labels.index_put_({anchors_with_max_overlap}, gt_classes.index({gt_inds_force}));
            gt_ids.index_put_({anchors_with_max_overlap}, gt_inds_force.to(torch::kInt));

            pos_inds = anchor_to_gt_max >= matched_threshold;
            gt_inds_over_thresh = anchor_to_gt_argmax.index({pos_inds});
            labels.index_put_({pos_inds}, gt_classes.index({gt_inds_over_thresh}));
            gt_ids.index_put_({pos_inds}, gt_inds_over_thresh.to(torch::kInt));
            bg_inds = (anchor_to_gt_max < unmatched_threshold).nonzero().index({torch::indexing::Slice(), 0});
        }
        else
        {
            bg_inds = torch::arange(num_anchors, anchors.options().device());
        }

        auto fg_inds = torch::nonzero(labels > 0).select(1, 0);

        int64_t num_fg = static_cast<int64_t>(this->pos_fraction * this->sample_size);
        if (fg_inds.size(0) > num_fg)
        {
            int64_t num_disabled = fg_inds.size(0) - num_fg;
            auto disable_inds = torch::randperm(fg_inds.size(0)).slice(0, 0, num_disabled);
            labels.index_put_({disable_inds}, -1);
            fg_inds = torch::nonzero(labels > 0).select(1, 0);
        }

        int64_t num_bg = this->sample_size - (labels > 0).sum().item<int64_t>();
        if (bg_inds.size(0) > num_bg)
        {
            auto enable_inds = bg_inds.index({torch::randint(0, bg_inds.size(0), {num_bg})});
            labels.index_put_({enable_inds}, 0);
        }

        auto bbox_targets = torch::zeros({num_anchors, this->box_coder.code_size}, anchors.options());
        if (gt_boxes.size(0) > 0 && anchors.size(0) > 0)
        {
            auto fg_gt_boxes = gt_boxes.index({anchor_to_gt_argmax.index({fg_inds}), torch::indexing::Slice()});
            auto fg_anchors = anchors.index({fg_inds, torch::indexing::Slice()});
            bbox_targets.index_put_({fg_inds, torch::indexing::Slice()}, this->box_coder.encode_torch(fg_gt_boxes, fg_anchors));
        }

        auto reg_weights = torch::zeros({num_anchors}, anchors.options());

        if (this->norm_by_num_examples)
        {
            auto num_examples = (labels >= 0).sum().item<float>();
            num_examples = num_examples > 1.0 ? num_examples : 1.0;
            reg_weights.index_put_({labels > 0}, 1.0 / num_examples);
        }
        else
        {
            reg_weights.index_put_({labels > 0}, 1.0);
        }

        std::unordered_map<std::string, torch::Tensor> ret_dict = {
            {"box_cls_labels", labels},
            {"box_reg_targets", bbox_targets},
            {"reg_weights", reg_weights},
        };

        return ret_dict;
    }

private:
    std::vector<std::string> class_names;
    ResidualCoder box_coder;
    bool match_height;
    std::vector<std::string> anchor_class_names;
    float pos_fraction;
    int sample_size;
    bool norm_by_num_examples;
    TargetAssignerConfig config;
    std::map<std::string, float> matched_thresholds;
    std::map<std::string, float> unmatched_thresholds;
    bool use_multihead;
};

void print_shapes(BatchMap data)
{
    for (auto &item : data)
    {
        std::cout << item.first << " " << item.second.sizes() << '\n';
    }
};