#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace kharin_m_multidimensional_integral_calc_all {

class TaskALL : public ppc::core::Task {
 public:
  explicit TaskALL(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  double ComputeLocalSum();
  std::vector<double> input_;
  std::vector<double> local_input_;
  std::vector<size_t> grid_sizes_;
  std::vector<double> step_sizes_;
  bool validation_result_{true};
  bool preprocessing_result_{true};
  bool local_steps_valid_{true};
  double output_result_;
  boost::mpi::communicator world_;
  size_t num_threads_{1};
};

}  // namespace kharin_m_multidimensional_integral_calc_all