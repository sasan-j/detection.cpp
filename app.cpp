#include <torch/torch.h>
#include <torch/script.h> // One-stop header.
#include <iostream>
#include <tuple>
#include <vector>
#include "voxelization.h"
#include "pointpillars.h"
#include "utils.h"

torch::Tensor add_padding(torch::Tensor tensor, int padding_value) {
    auto options = torch::TensorOptions().dtype(tensor.dtype()).device(tensor.device());
    auto padding_tensor = torch::full({tensor.size(0), 1}, padding_value, options);
    return torch::cat({padding_tensor, tensor}, /*dim=*/1);
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
  // torch::Tensor dummy_pcd = torch::rand({65536, 4});
  std::string file_path = "rc_scaled.bin";
  auto pcd = torch::from_file(file_path, false, 65536 * 4, torch::kFloat32);
  pcd = pcd.view({-1, 4});
  std::cout << pcd.index({torch::indexing::Slice(0, 5), torch::indexing::Slice()}) << std::endl;
  std::cout << pcd.index({torch::indexing::Slice(-5), torch::indexing::Slice()}) << std::endl;

  // Call Voxelization on CPU
  at::Tensor voxel_size = torch::tensor({0.16, 0.16, 4.0}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  at::Tensor point_cloud_range = torch::tensor({-39.68, -39.68,  -3.  ,  39.68,  39.68,   1.}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));

  torch::Tensor grid_size = (point_cloud_range.slice(0, 3, 6) - point_cloud_range.slice(0, 0, 3)) / voxel_size;
  grid_size = grid_size.round().to(torch::kLong).reshape({-1});
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

  // Verifications to see if we're getting the right shapes
  assert(pcd.sizes() == torch::IntArrayRef({65536, 4}));
  assert(torch::equal(grid_size, torch::tensor({496, 496, 1}, torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU))));
  assert(grid_size.sizes() == torch::IntArrayRef({3}));
  assert(voxels.sizes() == torch::IntArrayRef({6941, max_points_voxel, pcd.size(1)}));
  assert(coors.sizes() == torch::IntArrayRef({6941, 3}));
  assert(num_points_per_voxel.sizes() == torch::IntArrayRef({6941}));

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
  print_shapes(batch_dict);

  batch_dict = fake_collate(batch_dict);
  
  // PointPillar
  pointpillars::PointPillars model(voxel_size_vect, point_cloud_range_vect, max_points_voxel, max_num_voxels, grid_size);
  std::cout << "Model was instantiated" << '\n';

  // Execute the model and turn its output into a tensor.
  auto output = model.forward(batch_dict);

  //// VFE - Done (untested)
  //// Map to BEV - Done (untested)
  //// Backbone 2d - Done (untested)
  //// Dense Head - Done (untested)



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
