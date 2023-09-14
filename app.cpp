#include <torch/torch.h>
#include <torch/script.h> // One-stop header.
#include <iostream>
#include <tuple>
#include <vector>
#include "voxelization.h"
#include "pointpillars.h"
#include "utils.h"

torch::Tensor add_padding(torch::Tensor tensor, int padding_value, bool right=false) {
    auto options = torch::TensorOptions().dtype(tensor.dtype()).device(tensor.device());
    auto padding_tensor = torch::full({tensor.size(0), 1}, padding_value, options);
    if (right) {
        return torch::cat({tensor, padding_tensor}, /*dim=*/1);
    }
    return torch::cat({padding_tensor, tensor}, /*dim=*/1);
}


void save_map(const std::string& filename, const std::unordered_map<std::string, torch::Tensor>& map) {
  c10::Dict<std::string, torch::Tensor> ivalue_map;
  for (const auto& kv : map) {
      ivalue_map.insert(kv.first, kv.second);
  }

  torch::IValue ivalue(ivalue_map);

  auto pickle_bytes = torch::pickle_save(ivalue);

  auto pickle_out = torch::pickle_save(ivalue.toGenericDict());
  std::ofstream outputFile(filename, std::ios::binary);

  if (outputFile.is_open()) {
      outputFile.write(pickle_out.data(), pickle_out.size());
      outputFile.close();
  }
}


BatchMap fake_collate(BatchMap batch_dict){
  // Collate Batch - Done (untested)
  // We just pretend to have a batch of size 1
  batch_dict["points"] = add_padding(batch_dict["points"], 0);
  std::cout << "points" << batch_dict["points"].index({torch::indexing::Slice(0, 3), torch::indexing::Slice(0, torch::indexing::None)}) << '\n';
  std::cout << "points shape:" << batch_dict["points"].sizes() << '\n';

  batch_dict["voxel_coords"] = add_padding(batch_dict["voxel_coords"], 0);
  std::cout << "voxel_coords" << batch_dict["voxel_coords"].index({torch::indexing::Slice(0, 3), torch::indexing::Slice(0, torch::indexing::None)}) << '\n';
  std::cout << "voxel_coords shape:" << batch_dict["voxel_coords"].sizes() << '\n';

  batch_dict["batch_size"] = torch::tensor({1}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
  return batch_dict;
}



int main()
{

TargetAssignerConfig target_assigner_conf_single = TargetAssignerConfig();


// For Multi - NuScenes
std::vector<AnchorGeneratorConfig> anchor_configs_multi = {
    {"car", {{4.63, 1.97, 1.74}}, {0, 1.57}, {-0.95}, false, 4, 0.6, 0.45},
    {"truck", {{6.93, 2.51, 2.84}}, {0, 1.57}, {-0.6}, false, 4, 0.55, 0.4},
    {"construction_vehicle", {{6.37, 2.85, 3.19}}, {0, 1.57}, {-0.225}, false, 4, 0.5, 0.35},
    {"bus", {{10.5, 2.94, 3.47}}, {0, 1.57}, {-0.085}, false, 4, 0.55, 0.4},
    {"trailer", {{12.29, 2.90, 3.87}}, {0, 1.57}, {0.115}, false, 4, 0.5, 0.35},
    {"barrier", {{0.50, 2.53, 0.98}}, {0, 1.57}, {-1.33}, false, 4, 0.55, 0.4},
    {"motorcycle", {{2.11, 0.77, 1.47}}, {0, 1.57}, {-1.085}, false, 4, 0.5, 0.3},
    {"bicycle", {{1.70, 0.60, 1.28}}, {0, 1.57}, {-1.18}, false, 4, 0.5, 0.35},
    {"pedestrian", {{0.73, 0.67, 1.77}}, {0, 1.57}, {-0.935}, false, 4, 0.6, 0.4},
    {"traffic_cone", {{0.41, 0.41, 1.07}}, {0, 1.57}, {-1.285}, false, 4, 0.6, 0.4}
};

// For Single - Kitti
std::vector<AnchorGeneratorConfig> anchor_configs_single = {
    {"Car", {{3.9, 1.6, 1.56}}, {0, 1.57}, {-1.78}, false, 2, 0.6, 0.45},
    {"Pedestrian", {{0.8, 0.6, 1.73}}, {0, 1.57}, {-0.6}, false, 2, 0.5, 0.35},
    {"Cyclist", {{1.76, 0.6, 1.73}}, {0, 1.57}, {-0.6}, false, 2, 0.5, 0.35}
};


// Has problems but not needed atm
// AnchorHeadConfig pp_single_head = {
//   2, // num_dir_bins;
//   true, // use_direction_classifier;
//   0.78539, // dir_offset;
//   0.0, // dir_limit_offset;
//   target_assigner_conf_single, // target_assigner_config
//   anchor_configs_single, // anchor_generator_configs
//   SeparateRegConfig(),
// };

TargetAssignerConfig target_assigner_conf_multi = TargetAssignerConfig();
target_assigner_conf_multi.box_coder_code_size = 9;
target_assigner_conf_multi.box_coder_encode_angle_by_sincos = true;

AnchorHeadConfig pp_multi_head = {
  2, // num_dir_bins;
  false, // use_direction_classifier;
  0.78539, // dir_offset;
  0.0, // dir_limit_offset;
  target_assigner_conf_multi, // target_assigner_config
  anchor_configs_multi, // anchor_generator_configs
  SeparateRegConfig(), // separate_reg_config
  true, // use_multihead
  true, // separate_multihead
  64, // shared_conv_num_filter
  false, // class_agnostic
  {
        {"car"},
        {"truck", "construction_vehicle"},
        {"bus", "trailer"},
        {"barrier"},
        {"motorcycle", "bicycle"},
        {"pedestrian", "traffic_cone"},
    } // rpn_head_config
};

// PointPillar Single Head
// ModelConfig model_config = {
//     {0.16, 0.16, 4.0}, // voxel_size
//     {-39.68, -39.68,  -3.  ,  39.68,  39.68,   1.}, // point_cloud_range
//     32, // max_points_voxel
//     40000, // max_num_voxels
//     4, // num_point_features
//     {3, 5, 5}, // backbone_layer_nums
//     {2, 2, 2}, // backbone_layer_strides
//     {64, 128, 256}, // backbone_num_filters
//     {1, 2, 4}, // backbone_upsample_strides
//     {128, 128, 128}, // backbone_num_upsample_filters
//     pp_single_head // anchor_head_config
// };

// PointPillar Multi Head
ModelConfig model_config = {
    {0.2, 0.2, 8.0}, // voxel_size
    {-51.2, -51.2, -5.0, 51.2, 51.2, 3.0}, // point_cloud_range
    20, // max_points_voxel
    40000, // max_num_voxels
    5, // num_point_features
    {3, 5, 5}, // backbone_layer_nums
    {2, 2, 2}, // backbone_layer_strides
    {64, 128, 256}, // backbone_num_filters
    {0.5, 1.0, 2.0}, // backbone_upsample_strides
    {128, 128, 128}, // backbone_num_upsample_filters
    pp_multi_head // anchor_head_config
};


  std::cout << "rpn_head_config:" << '\n';
  for (const auto& kv : model_config.anchor_head_config.rpn_head_config) {
    std::cout << kv << '\n';
  }

  // torch::Tensor dummy_pcd = torch::rand({65536, 4});
  // std::string file_path = "rc_scaled.bin";
  // std::string file_path = "1687261555.576275000.bin"; 
  std::string file_path = "1687261608.576809000.bin";
  auto pcd = torch::from_file(file_path, false, 65536 * 4, torch::kFloat32);
  pcd = pcd.view({-1, 4});

  if (model_config.num_point_features == 5) {
    // add a column of ones to the point cloud
    pcd = add_padding(pcd, 1, true);
  }


  std::cout << "points" << pcd.sizes() << '\n';

  pcd.select(1, 2) -= 1.75;
  pcd.select(1, 2).clamp_(-3, 1);

  std::cout << pcd.index({torch::indexing::Slice(0, 5), torch::indexing::Slice()}) << std::endl;
  std::cout << pcd.index({torch::indexing::Slice(-5), torch::indexing::Slice()}) << std::endl;

  // Call Voxelization on CPU
  at::Tensor voxel_size = torch::tensor(model_config.voxel_size, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  at::Tensor point_cloud_range = torch::tensor(model_config.point_cloud_range, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));

  torch::Tensor grid_size = (point_cloud_range.slice(0, 3, 6) - point_cloud_range.slice(0, 0, 3)) / voxel_size;
  grid_size = grid_size.round().to(torch::kLong).reshape({-1});
  torch::Tensor input_feat_shape = grid_size.slice(/*dim=*/0, /*start=*/0, /*end=*/2);

  auto voxel_size_vect = std::vector(voxel_size.data_ptr<float>(), voxel_size.data_ptr<float>() + voxel_size.numel());
  auto point_cloud_range_vect = std::vector(point_cloud_range.data_ptr<float>(), point_cloud_range.data_ptr<float>() + point_cloud_range.numel());

  at::Tensor voxels = torch::zeros({model_config.max_num_voxels, model_config.max_points_voxel, pcd.size(1)}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  at::Tensor coors = torch::zeros({model_config.max_num_voxels, 3}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
  at::Tensor num_points_per_voxel = torch::zeros({model_config.max_num_voxels}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));

  std::cout << "voxels before voxelization" << voxels.sizes() << '\n';

  auto vox_out = voxelization::hard_voxelize(pcd, voxels, coors, num_points_per_voxel, voxel_size_vect, point_cloud_range_vect, model_config.max_points_voxel, model_config.max_num_voxels, 3, true);
  voxels = voxels.index({torch::indexing::Slice(0, vox_out)});
  coors = coors.index({torch::indexing::Slice(0, vox_out)});//.flip({-1});
  num_points_per_voxel = num_points_per_voxel.index({torch::indexing::Slice(0, vox_out)});



  std::cout << "points" << pcd.index({torch::indexing::Slice(0, 3), torch::indexing::Slice(0, 3)}) << '\n';
  std::cout << "points shape:" << pcd.sizes() << '\n';
  std::cout << "grid_size" << grid_size << '\n';
  std::cout << "grid_size shape" << grid_size.sizes() << '\n';
  std::cout << "grid_size elements: " << grid_size.index({0}).item<int64_t>() << ", " << grid_size.index({1}).item<int64_t>() << ", " << grid_size.index({2}).item<int64_t>() << '\n';
  std::cout << "voxels" << voxels.index({torch::indexing::Slice(0, 3), torch::indexing::Slice(0, 3)}) << '\n';
  std::cout << "voxels shape:" << voxels.sizes() << '\n';
  std::cout << "coors" << coors.index({torch::indexing::Slice(0, 5)}) << '\n';
  std::cout << "coors shape:" << coors.sizes() << '\n';
  std::cout << "points_per_voxels" << num_points_per_voxel.index({torch::indexing::Slice(0, 3)}) << '\n';
  std::cout << "points_per_voxels shape:" << num_points_per_voxel.sizes() << '\n';


  // Create a vector of inputs.
  BatchMap batch_dict;
  batch_dict["points"] = pcd;
  batch_dict["voxels"] = voxels;
  batch_dict["voxel_coords"] = coors;
  batch_dict["voxel_num_points"] = num_points_per_voxel;
  batch_dict["batch_size"] = torch::tensor({1}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
  batch_dict["use_lead_xyz"] = torch::tensor({false}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
  batch_dict["frame_id"] = torch::tensor({0}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));


  std::cout << "batch_dict before instantiating the model" << '\n';
  std::cout << "##############################!\n";
  print_shapes(batch_dict);

  batch_dict = fake_collate(batch_dict);
  
  // PointPillar
  pointpillars::PointPillars model(model_config);
  std::cout << "Model was instantiated" << '\n';

  // Execute the model and turn its output into a tensor.
  auto pred_dicts = model.forward(batch_dict);

  std::cout << "pred_dict" << '\n';

  auto batch_preds = pred_dicts[0];

  // for (const auto& kv : batch_preds) {
  //   std::cout << kv.first << kv.second << '\n';
  // }


  std::cout << "output" << '\n';

  save_map("output.pt", batch_preds);


  // torch::jit::script::Module voxelizer;
  // try {
  //   // Deserialize the ScriptModule from a file using torch::jit::load().
  //   voxelizer = torch::jit::load("traced_voxelizer.pt");
  // }
  // catch (const c10::Error& e) {
  //   std::cerr << "error loading the model\n";
  //   return -1;
  // }

  // // Create a vector of inputs.
  // std::vector<torch::jit::IValue> inputs;
  // inputs.push_back(pcd);

  // // Execute the model and turn its output into a tensor.
  // auto output = voxelizer.forward(inputs).toTuple();
  // std::cout << "voxels" << output->elements()[0].toTensor() << '\n';
  // std::cout << "coors" << output->elements()[1].toTensor() << '\n';
  // std::cout << "points_per_voxels" << output->elements()[2].toTensor() << '\n';

  std::cout << "ok\n";
}

