#include <torch/torch.h>
#include <vector>
#include <map>

#include "iou3d_cpu.h"
#include "iou3d_nms.h"

class ResidualCoder
{
public:
    ResidualCoder(int code_size = 7, bool encode_angle_by_sincos = false)
        : code_size(code_size), encode_angle_by_sincos(encode_angle_by_sincos)
    {
        if (this->encode_angle_by_sincos)
        {
            this->code_size += 1;
        }
    }

    torch::Tensor encode_torch(torch::Tensor boxes, torch::Tensor anchors)
    {
        anchors.slice(1, 3, 6).clamp_min_(1e-5);
        boxes.slice(1, 3, 6).clamp_min_(1e-5);

        auto xa = anchors.select(1, 0);
        auto ya = anchors.select(1, 1);
        auto za = anchors.select(1, 2);
        auto dxa = anchors.select(1, 3);
        auto dya = anchors.select(1, 4);
        auto dza = anchors.select(1, 5);
        auto ra = anchors.select(1, 6);
        auto cas = anchors.slice(1, 7);

        auto xg = boxes.select(1, 0);
        auto yg = boxes.select(1, 1);
        auto zg = boxes.select(1, 2);
        auto dxg = boxes.select(1, 3);
        auto dyg = boxes.select(1, 4);
        auto dzg = boxes.select(1, 5);
        auto rg = boxes.select(1, 6);
        auto cgs = boxes.slice(1, 7);

        auto diagonal = torch::sqrt(dxa.pow(2) + dya.pow(2));
        auto xt = (xg - xa) / diagonal;
        auto yt = (yg - ya) / diagonal;
        auto zt = (zg - za) / dza;
        auto dxt = torch::log(dxg / dxa);
        auto dyt = torch::log(dyg / dya);
        auto dzt = torch::log(dzg / dza);

        torch::Tensor rt_cos, rt_sin, rts;
        if (this->encode_angle_by_sincos)
        {
            rt_cos = torch::cos(rg) - torch::cos(ra);
            rt_sin = torch::sin(rg) - torch::sin(ra);
            rts = torch::stack({rt_cos, rt_sin});
        }
        else
        {
            rts = rg - ra;
        }

        auto cts = torch::stack({cgs - cas});
        return torch::cat({xt, yt, zt, dxt, dyt, dzt, rts, cts}, 1);
    }

    torch::Tensor decode_torch(torch::Tensor box_encodings, torch::Tensor anchors)
    {
        auto xa = anchors.select(1, 0);
        auto ya = anchors.select(1, 1);
        auto za = anchors.select(1, 2);
        auto dxa = anchors.select(1, 3);
        auto dya = anchors.select(1, 4);
        auto dza = anchors.select(1, 5);
        auto ra = anchors.select(1, 6);
        auto cas = anchors.slice(1, 7);

        torch::Tensor xt, yt, zt, dxt, dyt, dzt, cost, sint, cts;
        if (!this->encode_angle_by_sincos)
        {
            xt = box_encodings.select(1, 0);
            yt = box_encodings.select(1, 1);
            zt = box_encodings.select(1, 2);
            dxt = box_encodings.select(1, 3);
            dyt = box_encodings.select(1, 4);
            dzt = box_encodings.select(1, 5);
            cost = box_encodings.select(1, 6);
            sint = box_encodings.select(1, 7);
            cts = box_encodings.slice(1, 8);
        }
        else
        {
            xt = box_encodings.select(1, 0);
            yt = box_encodings.select(1, 1);
            zt = box_encodings.select(1, 2);
            dxt = box_encodings.select(1, 3);
            dyt = box_encodings.select(1, 4);
            dzt = box_encodings.select(1, 5);
            cost = box_encodings.select(1, 6);
            sint = box_encodings.select(1, 7);
            cts = box_encodings.slice(1, 9);
        }

        auto diagonal = torch::sqrt(dxa.pow(2) + dya.pow(2));
        auto xg = xt * diagonal + xa;
        auto yg = yt * diagonal + ya;
        auto zg = zt * dza + za;
        auto dxg = torch::exp(dxt) * dxa;
        auto dyg = torch::exp(dyt) * dya;
        auto dzg = torch::exp(dzt) * dza;

        torch::Tensor rg_cos, rg_sin, rg;
        if (this->encode_angle_by_sincos)
        {
            rg_cos = cost + torch::cos(ra);
            rg_sin = sint + torch::sin(ra);
            rg = torch::atan2(rg_sin, rg_cos);
        }
        else
        {
            rg = cost + ra;
        }

        auto cgs = torch::stack({cts + cas});
        return torch::cat({xg, yg, zg, dxg, dyg, dzg, rg, cgs}, 1);
    }

private:
    int code_size;
    bool encode_angle_by_sincos;
};

// ANCHOR_GENERATOR_CONFIG: [
//     {
//         'class_name': 'Car',
//         'anchor_sizes': [[3.9, 1.6, 1.56]],
//         'anchor_rotations': [0, 1.57],
//         'anchor_bottom_heights': [-1.78],
//         'align_center': False,
//         'feature_map_stride': 2,
//         'matched_threshold': 0.6,
//         'unmatched_threshold': 0.45
//     },
//     {
//         'class_name': 'Pedestrian',
//         'anchor_sizes': [[0.8, 0.6, 1.73]],
//         'anchor_rotations': [0, 1.57],
//         'anchor_bottom_heights': [-0.6],
//         'align_center': False,
//         'feature_map_stride': 2,
//         'matched_threshold': 0.5,
//         'unmatched_threshold': 0.35
//     },
//     {
//         'class_name': 'Cyclist',
//         'anchor_sizes': [[1.76, 0.6, 1.73]],
//         'anchor_rotations': [0, 1.57],
//         'anchor_bottom_heights': [-0.6],
//         'align_center': False,
//         'feature_map_stride': 2,
//         'matched_threshold': 0.5,
//         'unmatched_threshold': 0.35
//     }
// ]

class AnchorGenerator
{
public:
    std::vector<std::vector<float>> anchor_sizes;
    std::vector<std::vector<float>> anchor_rotations;
    std::vector<std::vector<float>> anchor_bottom_heights;
    std::vector<bool> align_center;
    int num_of_anchor_sets;
    std::vector<float> anchor_range;

    AnchorGenerator(std::vector<float> anchor_range,
                    std::vector<std::vector<float>> anchor_sizes = {{3.9, 1.6, 1.56}, {0.8, 0.6, 1.73}, {1.76, 0.6, 1.73}},
                    std::vector<std::vector<float>> anchor_rotations = {{0, 1.57}, {0, 1.57}, {0, 1.57}},
                    std::vector<std::vector<float>> anchor_bottom_heights = {{-1.78}, {-0.6}, {-0.6}},
                    std::vector<bool> align_center = {false, false, false})
    {
        this->anchor_range = anchor_range;
        this->anchor_sizes = anchor_sizes;
        this->anchor_rotations = anchor_rotations;
        this->anchor_bottom_heights = anchor_bottom_heights;
        this->align_center = align_center;

        assert(anchor_sizes.size() == anchor_rotations.size() && anchor_sizes.size() == anchor_bottom_heights.size());
        this->num_of_anchor_sets = anchor_sizes.size();
    }

    std::pair<std::vector<torch::Tensor>, torch::Tensor> generate_anchors(std::vector<torch::Tensor> grid_sizes)
    {
        assert(grid_sizes.size() == num_of_anchor_sets);
        std::vector<torch::Tensor> all_anchors;
        std::vector<int> num_anchors_per_location;

        for (int i = 0; i < num_of_anchor_sets; i++)
        {
            auto grid_size = grid_sizes[i];
            auto anchor_size = anchor_sizes[i];
            auto anchor_rotation = anchor_rotations[i];
            auto anchor_height = anchor_bottom_heights[i];
            auto align_center_i = align_center[i];

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
            auto anchor_size_tensor = torch::tensor(anchor_size, options);
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
        }

        auto out = torch::from_blob(num_anchors_per_location.data(), {num_anchors_per_location.size()});

        return {all_anchors, out};
    }
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
    AxisAlignedTargetAssigner(
        std::vector<std::string> class_names, 
        ResidualCoder box_coder, 
        bool match_height = false, 
        int sample_size = 512, 
        float pos_fraction = -1, 
        bool norm_by_num_examples = false, 
        bool use_multihead = false)
    {

        this->box_coder = box_coder;
        this->match_height = match_height;
        this->class_names = class_names;
        this->anchor_class_names = class_names;
        this->sample_size = sample_size;
        this->norm_by_num_examples = norm_by_num_examples;
        this->use_multihead = use_multihead;

        this->matched_thresholds = {
            {"Car", 0.6}, 
            {"Pedestrian", 0.5}, 
            {"Cyclist", 0.5}
        };
        this->unmatched_thresholds = {{"Car", 0.45}, {"Pedestrian", 0.35}, {"Cyclist", 0.35}};

        this->pos_fraction = (pos_fraction >= 0) ? pos_fraction : -1;
    }

    std::map<std::string, torch::Tensor> assign_targets(std::vector<torch::Tensor> all_anchors, torch::Tensor gt_boxes_with_classes)
    {
        std::vector<torch::Tensor> bbox_targets;
        std::vector<torch::Tensor> cls_labels;
        std::vector<torch::Tensor> reg_weights;

        int batch_size = gt_boxes_with_classes.size(0);
        auto gt_classes = gt_boxes_with_classes.index({torch::indexing::Slice(), torch::indexing::Slice(), -1});
        auto gt_boxes = gt_boxes_with_classes.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(None, -1)});

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

            std::vector<Target> target_list;
            for (int i = 0; i < anchor_class_names.size(); ++i)
            {
                auto anchor_class_name = anchor_class_names[i];
                auto anchors = all_anchors[i];

                // The following is a placeholder for the mask creation
                // The Python functionality does not have a direct C++ equivalent
                torch::Tensor mask;
                // ... create mask ...

                auto feature_map_size = anchors.sizes().slice(0, 3);
                anchors = anchors.view(-1, anchors.size(-1));
                auto selected_classes = cur_gt_classes.index({mask});

                // The assign_targets_single function needs to be implemented
                auto single_target = assign_targets_single(
                    anchors,
                    cur_gt.index({mask}),
                    selected_classes,
                    matched_thresholds[anchor_class_name],
                    unmatched_thresholds[anchor_class_name]);
                target_list.push_back(single_target);

                // ... continue translation ...
            }

            // ... continue translation ...
            std::vector<torch::Tensor> box_cls_labels;
            std::vector<torch::Tensor> box_reg_targets;
            std::vector<torch::Tensor> reg_weights;

            for (const auto &t : target_list)
            {
                box_cls_labels.push_back(t.at("box_cls_labels").view(feature_map_size).view(-1));
                box_reg_targets.push_back(t.at("box_reg_targets").view(feature_map_size).view(-1, this->box_coder.code_size));
                reg_weights.push_back(t.at("reg_weights").view(feature_map_size).view(-1));
            }

            std::map<std::string, std::vector<torch::Tensor>> target_dict = {
                {"box_cls_labels", box_cls_labels},
                {"box_reg_targets", box_reg_targets},
                {"reg_weights", reg_weights}};

            target_dict["box_reg_targets"] = torch::cat(target_dict["box_reg_targets"], -2).view(-1, this->box_coder.code_size);
            target_dict["box_cls_labels"] = torch::cat(target_dict["box_cls_labels"], -1).view(-1);
            target_dict["reg_weights"] = torch::cat(target_dict["reg_weights"], -1).view(-1);

            bbox_targets.push_back(target_dict["box_reg_targets"]);
            cls_labels.push_back(target_dict["box_cls_labels"]);
            reg_weights.push_back(target_dict["reg_weights"]);
        }

        auto bbox_targets_t = torch::stack(bbox_targets, 0);
        auto cls_labels_t = torch::stack(cls_labels, 0);
        auto reg_weights_t = torch::stack(reg_weights, 0);

        std::map<std::string, torch::Tensor> all_targets_dict = {
            {"box_cls_labels", cls_labels_t},
            {"box_reg_targets", bbox_targets_t},
            {"reg_weights", reg_weights_t}};
        return all_targets_dict;
    }


    std::map<std::string, torch::Tensor> assign_targets_single(
        torch::Tensor anchors, torch::Tensor gt_boxes, torch::Tensor gt_classes,
        double matched_threshold, double unmatched_threshold
    ) {
        // Initialize tensors
        int num_anchors = anchors.size(0);
        int num_gt = gt_boxes.size(0);

        torch::Tensor labels = torch::ones({num_anchors}).to(anchors.device()) * -1;
        torch::Tensor gt_ids = torch::ones({num_anchors}).to(anchors.device()) * -1;

        torch::Tensor anchor_to_gt_argmax, anchor_to_gt_max, gt_to_anchor_argmax, gt_to_anchor_max, anchors_with_max_overlap;
        torch::Tensor gt_inds_force, pos_inds, gt_inds_over_thresh, bg_inds;

        if (gt_boxes.size(0) > 0 && anchors.size(0) > 0) {
            torch::Tensor anchor_by_gt_overlap;
            if (this->match_height) {
                anchor_by_gt_overlap = boxes_iou3d_gpu(anchors.index({torch::indexing::Slice(), torch::indexing::Slice(0,7)}), 
                                                                        gt_boxes.index({torch::indexing::Slice(), torch::indexing::Slice(0,7)}));
            } else {
                anchor_by_gt_overlap = boxes3d_nearest_bev_iou(anchors.index({torch::indexing::Slice(), torch::indexing::Slice(0,7)}), 
                                                                        gt_boxes.index({torch::indexing::Slice(), torch::indexing::Slice(0,7)}));
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
        } else {
            bg_inds = torch::arange(num_anchors, anchors.options().device());
        }




        // Actual implementation needed here
        // This involves operations like IoU calculations, array manipulations, etc.
        // which needs to be implemented using appropriate C++ libraries or custom functions.


        if (this->norm_by_num_examples) {
            auto num_examples = (labels >= 0).sum();
            num_examples = num_examples.item<float>() > 1.0 ? num_examples : torch::tensor(1.0);
            reg_weights.masked_fill_(labels > 0, 1.0 / num_examples.item<float>());
        } else {
            reg_weights.masked_fill_(labels > 0, 1.0);
        }



        std::map<std::string, torch::Tensor> ret_dict = {
            {"box_cls_labels", labels},
            {"box_reg_targets", bbox_targets},
            {"reg_weights", reg_weights}};
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
    std::map<std::string, float> matched_thresholds;
    std::map<std::string, float> unmatched_thresholds;
    bool use_multihead;
};


torch::Tensor boxes_iou_normal(torch::Tensor boxes_a, torch::Tensor boxes_b) {
    // Ensure the shapes of boxes_a and boxes_b
    assert(boxes_a.size(1) == 4 && boxes_b.size(1) == 4);

    // Calculate overlap in x and y directions
    torch::Tensor x_min = torch::max(boxes_a.index({torch::indexing::Slice(), 0}).unsqueeze(1), boxes_b.index({torch::indexing::Slice(), 0}).unsqueeze(0));
    torch::Tensor x_max = torch::min(boxes_a.index({torch::indexing::Slice(), 2}).unsqueeze(1), boxes_b.index({torch::indexing::Slice(), 2}).unsqueeze(0));
    torch::Tensor y_min = torch::max(boxes_a.index({torch::indexing::Slice(), 1}).unsqueeze(1), boxes_b.index({torch::indexing::Slice(), 1}).unsqueeze(0));
    torch::Tensor y_max = torch::min(boxes_a.index({torch::indexing::Slice(), 3}).unsqueeze(1), boxes_b.index({torch::indexing::Slice(), 3}).unsqueeze(0));

    // Calculate overlap area
    torch::Tensor x_len = torch::clamp_min(x_max - x_min, 0);
    torch::Tensor y_len = torch::clamp_min(y_max - y_min, 0);
    torch::Tensor area_a = (boxes_a.index({torch::indexing::Slice(), 2}) - boxes_a.index({torch::indexing::Slice(), 0})) * (boxes_a.index({torch::indexing::Slice(), 3}) - boxes_a.index({torch::indexing::Slice(), 1}));
    torch::Tensor area_b = (boxes_b.index({torch::indexing::Slice(), 2}) - boxes_b.index({torch::indexing::Slice(), 0})) * (boxes_b.index({torch::indexing::Slice(), 3}) - boxes_b.index({torch::indexing::Slice(), 1}));

    // Calculate intersection and IoU
    torch::Tensor a_intersect_b = x_len * y_len;
    torch::Tensor iou = a_intersect_b / torch::clamp_min(area_a.unsqueeze(1) + area_b.unsqueeze(0) - a_intersect_b, 1e-6);

    return iou;
}


torch::Tensor boxes3d_lidar_to_aligned_bev_boxes(torch::Tensor boxes3d) {
    // Make sure boxes3d has at least 7 columns
    assert(boxes3d.size(1) >= 7);

    // Limit the period of the heading angle
    torch::Tensor rot_angle = limit_period(boxes3d.index({torch::indexing::Slice(), 6}), 0.5, M_PI).abs();

    // Choose dimensions based on rotation angle
    torch::Tensor choose_dims = torch::where(rot_angle.unsqueeze(1) < M_PI / 4, 
                                             boxes3d.index({torch::indexing::Slice(), torch::indexing::Slice(3, 5)}), 
                                             boxes3d.index({torch::indexing::Slice(), torch::indexing::Slice(4, 6)}));

    // Create aligned BEV boxes
    torch::Tensor aligned_bev_boxes = torch::cat({boxes3d.index({torch::indexing::Slice(), torch::indexing::Slice(0, 2)}) - choose_dims / 2, 
                                                  boxes3d.index({torch::indexing::Slice(), torch::indexing::Slice(0, 2)}) + choose_dims / 2}, 1);

    return aligned_bev_boxes;
}


torch::Tensor boxes3d_nearest_bev_iou(torch::Tensor boxes_a, torch::Tensor boxes_b) {
    // Ensure the input tensors have exactly 7 columns
    assert(boxes_a.size(1) == 7 && boxes_b.size(1) == 7);
    
    // Convert 3D boxes to BEV boxes
    torch::Tensor boxes_bev_a = boxes3d_lidar_to_aligned_bev_boxes(boxes_a);
    torch::Tensor boxes_bev_b = boxes3d_lidar_to_aligned_bev_boxes(boxes_b);
    
    // Compute the IoU using the converted BEV boxes
    return boxes_iou_normal(boxes_bev_a, boxes_bev_b);
}


torch::Tensor limit_period(torch::Tensor val, float offset=0.5, float period=3.14159) {
    return val - torch::floor(val / period + offset) * period;
}
