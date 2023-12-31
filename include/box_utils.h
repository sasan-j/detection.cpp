#pragma once

#include <torch/torch.h>
#include "utils.h"
#include "iou3d_nms.h"



// Defs
torch::Tensor boxes3d_lidar_to_aligned_bev_boxes(torch::Tensor boxes3d);
torch::Tensor boxes_iou_normal(torch::Tensor boxes_a, torch::Tensor boxes_b);
torch::Tensor limit_period(torch::Tensor val, float offset=0.5, float period=3.14159);


torch::Tensor boxes3d_nearest_bev_iou(torch::Tensor boxes_a, torch::Tensor boxes_b) {
    // Ensure the input tensors have exactly 7 columns
    assert(boxes_a.size(1) == 7 && boxes_b.size(1) == 7);
    
    // Convert 3D boxes to BEV boxes
    torch::Tensor boxes_bev_a = boxes3d_lidar_to_aligned_bev_boxes(boxes_a);
    torch::Tensor boxes_bev_b = boxes3d_lidar_to_aligned_bev_boxes(boxes_b);
    
    // Compute the IoU using the converted BEV boxes
    return boxes_iou_normal(boxes_bev_a, boxes_bev_b);
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

torch::Tensor limit_period(torch::Tensor val, float offset, float period) {
    return val - torch::floor(val / period + offset) * period;
}

torch::Tensor boxes_iou3d_gpu(torch::Tensor boxes_a, torch::Tensor boxes_b) {
    // Ensure the shapes of boxes_a and boxes_b
    assert(boxes_a.size(1) == 7 && boxes_b.size(1) == 7);

    // Calculate height overlap
    torch::Tensor boxes_a_height_max = (boxes_a.index({torch::indexing::Slice(), 2}) + boxes_a.index({torch::indexing::Slice(), 5}) / 2).view({-1, 1});
    torch::Tensor boxes_a_height_min = (boxes_a.index({torch::indexing::Slice(), 2}) - boxes_a.index({torch::indexing::Slice(), 5}) / 2).view({-1, 1});
    torch::Tensor boxes_b_height_max = (boxes_b.index({torch::indexing::Slice(), 2}) + boxes_b.index({torch::indexing::Slice(), 5}) / 2).view({1, -1});
    torch::Tensor boxes_b_height_min = (boxes_b.index({torch::indexing::Slice(), 2}) - boxes_b.index({torch::indexing::Slice(), 5}) / 2).view({1, -1});

    // Calculate BEV overlap
    torch::Tensor overlaps_bev = torch::zeros({boxes_a.size(0), boxes_b.size(0)}, boxes_a.options().device(torch::kCUDA));
    // Note: boxes_overlap_bev_gpu() should be a defined function in your C++/CUDA code
    boxes_overlap_bev_gpu(boxes_a.contiguous(), boxes_b.contiguous(), overlaps_bev);

    // Calculate height overlap
    torch::Tensor max_of_min = torch::max(boxes_a_height_min, boxes_b_height_min);
    torch::Tensor min_of_max = torch::min(boxes_a_height_max, boxes_b_height_max);
    torch::Tensor overlaps_h = torch::clamp(min_of_max - max_of_min, 0);

    // Calculate 3D overlap
    torch::Tensor overlaps_3d = overlaps_bev * overlaps_h;

    // Calculate volume of boxes_a and boxes_b
    torch::Tensor vol_a = (boxes_a.index({torch::indexing::Slice(), 3}) * boxes_a.index({torch::indexing::Slice(), 4}) * boxes_a.index({torch::indexing::Slice(), 5})).view({-1, 1});
    torch::Tensor vol_b = (boxes_b.index({torch::indexing::Slice(), 3}) * boxes_b.index({torch::indexing::Slice(), 4}) * boxes_b.index({torch::indexing::Slice(), 5})).view({1, -1});

    // Calculate IoU
    torch::Tensor iou3d = overlaps_3d / torch::clamp(vol_a + vol_b - overlaps_3d, 1e-6);

    return iou3d;
}

struct NMSConfig
{
  bool MULTI_CLASSES_NMS = false;
  std::string NMS_TYPE = "nms_gpu";
  float NMS_THRESH = 0.01;
  int NMS_PRE_MAXSIZE = 4096;
  int NMS_POST_MAXSIZE = 500;
};



torch::Tensor nms_gpu_wrapper(
    /* This is a high level function */
    torch::Tensor boxes, torch::Tensor scores, float thresh, 
    c10::optional<int> pre_maxsize = c10::nullopt) {

    TORCH_CHECK(boxes.size(1) == 7, "Expected boxes to have shape (N, 7)");

    torch::Tensor order = std::get<1>(scores.sort(0, true));
    
    if (pre_maxsize.has_value()) {
        order = order.slice(0, 0, pre_maxsize.value());
    }

    boxes = boxes.index_select(0, order).contiguous();
    torch::Tensor keep = torch::empty({boxes.size(0)}, torch::kInt64);
    
    int num_out = nms_gpu(boxes, keep, thresh);
    
    return order.index_select(0, keep.slice(0, 0, num_out).to(torch::kCUDA)).contiguous();
}



std::pair<torch::Tensor, torch::Tensor> class_agnostic_nms(
    torch::Tensor box_scores, torch::Tensor box_preds, 
    NMSConfig nms_config, torch::optional<double> score_thresh = torch::nullopt) {

    torch::Tensor src_box_scores = box_scores.clone();

    torch::Tensor scores_mask;


    if(score_thresh.has_value()) {
        scores_mask = box_scores.ge(score_thresh.value());
        box_scores = torch::masked_select(box_scores, scores_mask);
        box_preds = box_preds.index({scores_mask});
    }

    torch::Tensor selected;

    if(box_scores.size(0) > 0) {
        std::tuple<torch::Tensor, torch::Tensor> topk_result = 
            torch::topk(box_scores, std::min(nms_config.NMS_PRE_MAXSIZE, static_cast<int>(box_scores.size(0))));

        torch::Tensor box_scores_nms = std::get<0>(topk_result).to(torch::kCUDA).contiguous();
        torch::Tensor indices = std::get<1>(topk_result);

        torch::Tensor boxes_for_nms = box_preds.index({indices}).to(torch::kCUDA).contiguous();

        if (nms_config.NMS_TYPE == "nms_gpu") {
            auto keep_idx = nms_gpu_wrapper(
                boxes_for_nms.slice(1, 0, 7), box_scores_nms, nms_config.NMS_THRESH /*, Other arguments in nms_config */);

            selected = indices.to(torch::kCUDA).index({keep_idx.slice(0, 0, nms_config.NMS_POST_MAXSIZE)});
        // } else if (nms_config.NMS_TYPE == "nms_cpu") {
        //     auto keep_idx_and_selected_scores = iou3d_nms_utils::nms_cpu(
        //         boxes_for_nms.slice(1, 0, 7), box_scores_nms, nms_config.NMS_THRESH /*, Other arguments in nms_config */);

        //     torch::Tensor keep_idx = keep_idx_and_selected_scores.first;
        //     torch::Tensor selected_scores = keep_idx_and_selected_scores.second;

        //     selected = indices.index_select(0, keep_idx.slice(0, 0, nms_config.NMS_POST_MAXSIZE));
        } else {
            throw std::runtime_error("Invalid nms type");
        } 
    }

    selected = selected.to(torch::kCPU);

    if(score_thresh.has_value()) {
        torch::Tensor original_idxs = torch::nonzero(scores_mask).view(-1);
        selected = original_idxs.index({selected});
    }

    return std::make_pair(selected, src_box_scores.index_select(0, selected));
}


#include <torch/torch.h>
#include <vector>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> multi_classes_nms(
    const torch::Tensor& cls_scores, 
    const torch::Tensor& box_preds, 
    const NMSConfig& nms_config,  // Assuming NMSConfig is a struct/class you have
    const c10::optional<float>& score_thresh = c10::nullopt
) {
    std::vector<torch::Tensor> pred_scores, pred_labels, pred_boxes;

    for (int64_t k = 0; k < cls_scores.size(1); ++k) {
        torch::Tensor scores_mask;
        scores_mask = cls_scores.index({torch::indexing::Slice(), k}).ge(score_thresh.value());
        auto box_scores = cls_scores.index({scores_mask, k});
        auto cur_box_preds = box_preds.index({scores_mask});


        torch::Tensor selected;
        if (box_scores.size(0) > 0) {
            std::tuple<torch::Tensor, torch::Tensor> topk_result = 
                torch::topk(box_scores, std::min(nms_config.NMS_PRE_MAXSIZE, static_cast<int>(box_scores.size(0))));
            torch::Tensor box_scores_nms = std::get<0>(topk_result).to(torch::kCUDA).contiguous();
            torch::Tensor indices = std::get<1>(topk_result);
            torch::Tensor boxes_for_nms = cur_box_preds.index_select(0, indices).to(torch::kCUDA).contiguous();

            if (nms_config.NMS_TYPE == "nms_gpu") {
                auto keep_idx = nms_gpu_wrapper(
                    boxes_for_nms.slice(1, 0, 7), box_scores_nms, nms_config.NMS_THRESH /*, Other arguments in nms_config */);

                selected = indices.to(torch::kCUDA).index({keep_idx.slice(0, 0, nms_config.NMS_POST_MAXSIZE)});
                selected = selected.to(torch::kCPU);
                box_scores = box_scores.to(torch::kCPU);
            // } else if (nms_config.NMS_TYPE == "nms_cpu") {
            //     auto keep_idx_and_selected_scores = iou3d_nms_utils::nms_cpu(
            //         boxes_for_nms.slice(1, 0, 7), box_scores_nms, nms_config.NMS_THRESH /*, Other arguments in nms_config */);

            //     torch::Tensor keep_idx = keep_idx_and_selected_scores.first;
            //     torch::Tensor selected_scores = keep_idx_and_selected_scores.second;

            //     selected = indices.index_select(0, keep_idx.slice(0, 0, nms_config.NMS_POST_MAXSIZE));
            } else {
                throw std::runtime_error("Invalid nms type");
            } 
        }

        pred_scores.push_back(box_scores.index_select(0, selected));
        pred_labels.push_back(torch::ones({selected.sizes()}, box_scores.options().dtype(torch::kLong)) * k);
        pred_boxes.push_back(cur_box_preds.index_select(0, selected));
    }

    torch::Tensor final_scores = torch::cat(pred_scores, 0);
    torch::Tensor final_labels = torch::cat(pred_labels, 0);
    torch::Tensor final_boxes = torch::cat(pred_boxes, 0);

    return std::make_tuple(final_scores, final_labels, final_boxes);
}
