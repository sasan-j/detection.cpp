#include <torch/torch.h>
#include <torch/script.h> // One-stop header.
#include <iostream>
#include <tuple>
#include <vector>
#include "voxelization.h"
#include "pointpillars.h"

int main()
{
  // torch::Tensor dummy_pcd = torch::rand({65536, 4});
  std::string file_path = "rc_scaled.bin";
  auto pcd = torch::from_file(file_path, false, 65536 * 4, torch::kFloat32);
  pcd = pcd.view({-1, 4});
  std::cout << pcd.index({torch::indexing::Slice(0, 5), torch::indexing::Slice()}) << std::endl;
  std::cout << pcd.index({torch::indexing::Slice(-5), torch::indexing::Slice()}) << std::endl;

  // Call Voxelization on CPU
  at::Tensor voxel_size = torch::tensor({0.16, 0.16, 4.0}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  at::Tensor point_cloud_range = torch::tensor({-39.68, -39.68,  -3.  ,  39.68,  39.68,   1.}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));

  // Assuming point_cloud_range and voxel_size are already defined torch::Tensors
  torch::Tensor grid_size = (point_cloud_range.slice(/*dim=*/0, /*start=*/3) -
                           point_cloud_range.slice(/*dim=*/0, /*start=*/0, /*end=*/3)) / voxel_size;

  grid_size = grid_size.round().to(torch::kLong);
  torch::Tensor input_feat_shape = grid_size.slice(/*dim=*/0, /*start=*/0, /*end=*/2);

  auto voxel_size_vect = std::vector(voxel_size.data_ptr<float>(), voxel_size.data_ptr<float>() + voxel_size.numel());
  auto point_cloud_range_vect = std::vector(point_cloud_range.data_ptr<float>(), point_cloud_range.data_ptr<float>() + point_cloud_range.numel());

  int max_points_voxel = 32;
  int max_num_voxels = 40000;

  at::Tensor voxels = torch::zeros({max_num_voxels, max_points_voxel, pcd.size(1)}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  at::Tensor coors = torch::zeros({max_num_voxels, 3}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
  at::Tensor num_points_per_voxel = torch::zeros({max_num_voxels}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));

  auto vox_out = voxelization::hard_voxelize(pcd, voxels, coors, num_points_per_voxel, voxel_size_vect, point_cloud_range_vect, max_points_voxel, max_num_voxels, 3, true);
  voxels = voxels.index({torch::indexing::Slice(0, vox_out)});
  coors = coors.index({torch::indexing::Slice(0, vox_out)}).flip({-1});
  num_points_per_voxel = num_points_per_voxel.index({torch::indexing::Slice(0, vox_out)});


  std::cout << "grid_size" << grid_size << '\n';
  std::cout << "voxels" << voxels.index({torch::indexing::Slice(0, 3), torch::indexing::Slice(0, 3)}) << '\n';
  std::cout << "coors" << coors.index({torch::indexing::Slice(0, 3)}) << '\n';
  std::cout << "points_per_voxels" << num_points_per_voxel.index({torch::indexing::Slice(0, 3)}) << '\n';



  // PointPillar
  pointpillars::PointPillars model(voxel_size_vect, point_cloud_range_vect, max_points_voxel, max_num_voxels, grid_size);

  // Create a vector of inputs.
  std::unordered_map<std::string, torch::Tensor> batch_dict;
  batch_dict["voxels"] = voxels;
  batch_dict["coors"] = coors;
  batch_dict["num_points_per_voxel"] = num_points_per_voxel;

  // Execute the model and turn its output into a tensor.
  auto output = model.forward(batch_dict);

  //// VFE - Done (untested)
  //// Map to BEV
  //// Backbone 2d
  //// Dense Head



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
