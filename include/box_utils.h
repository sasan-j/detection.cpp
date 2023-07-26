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