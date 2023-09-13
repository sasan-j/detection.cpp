#pragma once

#include <torch/torch.h>
#include <vector>
#include <map>

// #include "iou3d_nms.h"
#include "box_utils.h"

// Type Aliases
using BatchMap = std::unordered_map<std::string, torch::Tensor>;

class ResidualCoder
{
public:
    int code_size;
    ResidualCoder(int code_size = 9, bool encode_angle_by_sincos = true)
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

        anchors = anchors.to(box_encodings.device());

        std::cout << box_encodings.device() << anchors.device() << '\n';

        auto anchors_split = anchors.split(1, /*dim=*/-1);
        torch::Tensor xa = anchors_split[0];
        torch::Tensor ya = anchors_split[1];
        torch::Tensor za = anchors_split[2];
        torch::Tensor dxa = anchors_split[3];
        torch::Tensor dya = anchors_split[4];
        torch::Tensor dza = anchors_split[5];
        torch::Tensor ra = anchors_split[6];
        std::vector<torch::Tensor> cas(anchors_split.begin() + 7, anchors_split.end());

        std::vector<torch::Tensor> box_encodings_split;
        torch::Tensor xt, yt, zt, dxt, dyt, dzt, rt, cost, sint;
        std::vector<torch::Tensor> cts;

        if (!encode_angle_by_sincos)
        {
            box_encodings_split = box_encodings.split(1, /*dim=*/-1);
            xt = box_encodings_split[0];
            yt = box_encodings_split[1];
            zt = box_encodings_split[2];
            dxt = box_encodings_split[3];
            dyt = box_encodings_split[4];
            dzt = box_encodings_split[5];
            rt = box_encodings_split[6];
            cts = std::vector<torch::Tensor>(box_encodings_split.begin() + 7, box_encodings_split.end());
        }
        else
        {
            box_encodings_split = box_encodings.split(1, /*dim=*/-1);
            xt = box_encodings_split[0];
            yt = box_encodings_split[1];
            zt = box_encodings_split[2];
            dxt = box_encodings_split[3];
            dyt = box_encodings_split[4];
            dzt = box_encodings_split[5];
            cost = box_encodings_split[6];
            sint = box_encodings_split[7];
            cts = std::vector<torch::Tensor>(box_encodings_split.begin() + 8, box_encodings_split.end());
        }

        torch::Tensor diagonal = torch::sqrt(torch::pow(dxa, 2) + torch::pow(dya, 2));
        torch::Tensor xg = xt * diagonal + xa;
        torch::Tensor yg = yt * diagonal + ya;
        torch::Tensor zg = zt * dza + za;

        torch::Tensor dxg = torch::exp(dxt) * dxa;
        torch::Tensor dyg = torch::exp(dyt) * dya;
        torch::Tensor dzg = torch::exp(dzt) * dza;

        torch::Tensor rg;
        if (encode_angle_by_sincos)
        {
            torch::Tensor rg_cos = cost + torch::cos(ra);
            torch::Tensor rg_sin = sint + torch::sin(ra);
            rg = torch::atan2(rg_sin, rg_cos);
        }
        else
        {
            rg = rt + ra;
        }

        std::vector<torch::Tensor> cgs;
        for (int i = 0; i < cts.size(); ++i)
        {
            cgs.push_back(cts[i] + cas[i]);
        }

        std::vector<torch::Tensor> cat_tensors = {xg, yg, zg, dxg, dyg, dzg, rg};
        cat_tensors.insert(cat_tensors.end(), cgs.begin(), cgs.end());
        torch::Tensor result = torch::cat(cat_tensors, /*dim=*/-1);

        return result;
    }

private:
    bool encode_angle_by_sincos;
};


// Function to split a string by a delimiter and return a vector of substrings
std::vector<std::string> split_string(const std::string& input, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream token_stream(input);

    while (std::getline(token_stream, token, delimiter)) {
        tokens.push_back(token);
    }

    return tokens;
}


torch::Tensor stackAndPad(const std::vector<torch::Tensor>& tensors) {
    // Find the maximum length
    int64_t max_length = 0;
    for (const auto& tensor : tensors) {
        max_length = std::max(max_length, tensor.size(0));
    }

    // Create new tensors with padding
    std::vector<torch::Tensor> padded_tensors;
    for (const auto& tensor : tensors) {
        torch::Tensor padded_tensor = torch::zeros({max_length});
        padded_tensor.slice(0, 0, tensor.size(0)).copy_(tensor);
        padded_tensors.push_back(padded_tensor);
    }

    // Stack the tensors
    return torch::stack(padded_tensors);
}
