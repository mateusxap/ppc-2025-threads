#include "all/kharin_m_multidimensional_integral_calc/include/ops_all.hpp"

#include <algorithm>
#include <boost/mpi/collectives/all_reduce.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/scatterv.hpp>
#include <boost/serialization/vector.hpp>  // NOLINT(misc-include-cleaner)
#include <cstddef>
#include <functional>
#include <thread>
#include <utility>
#include <vector>

#include "boost/mpi/collectives/reduce.hpp"
#include "core/util/include/util.hpp"

bool kharin_m_multidimensional_integral_calc_all::TaskALL::ValidationImpl() {
  // Проверка только на локальном процессе, без MPI-коммуникаций
  bool local_is_valid = true;
  if (world_.rank() == 0) {
    if (task_data->inputs.size() != 3 || task_data->outputs.size() != 1 ||
        task_data->inputs_count[1] != task_data->inputs_count[2] || task_data->outputs_count[0] != 1) {
      local_is_valid = false;
    }
  }

  // Сохраняем локальный результат для использования в RunImpl
  validation_result_ = local_is_valid;
  return true; // Всегда возвращаем true, реальная проверка будет в RunImpl
}

bool kharin_m_multidimensional_integral_calc_all::TaskALL::PreProcessingImpl() {
  // Локальная предобработка без MPI-коммуникаций
  bool local_is_valid = true;

  if (world_.rank() == 0) {
    auto* input_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
    size_t input_size = task_data->inputs_count[0];
    input_ = std::vector<double>(input_ptr, input_ptr + input_size);

    auto* sizes_ptr = reinterpret_cast<size_t*>(task_data->inputs[1]);
    size_t d = task_data->inputs_count[1];
    grid_sizes_ = std::vector<size_t>(sizes_ptr, sizes_ptr + d);
  
    auto* steps_ptr = reinterpret_cast<double*>(task_data->inputs[2]);
    step_sizes_ = std::vector<double>(steps_ptr, steps_ptr + d);

    // Проверка размера входных данных
    size_t total_size = 1;
    for (auto n : grid_sizes_) {
      total_size *= n;
    }
    if (total_size != input_size) {
      local_is_valid = false;
    }
  }

  // Локальная проверка шагов
  local_steps_valid_ = true; // Инициализируем, но будем использовать только на 0-ом процессе до RunImpl
  if (world_.rank() == 0 && !step_sizes_.empty()) {
    local_steps_valid_ = std::ranges::all_of(step_sizes_, [](double h) { return h > 0.0; });
  }

  // Сохраняем локальный результат для использования в RunImpl
  preprocessing_result_ = local_is_valid;
  return true; // Всегда возвращаем true, реальная проверка будет в RunImpl
}

bool kharin_m_multidimensional_integral_calc_all::TaskALL::RunImpl() {
  // Сначала выполняем отложенную валидацию
  bool is_valid = validation_result_;
  boost::mpi::broadcast(world_, is_valid, 0);
  if (!is_valid) {
    return false;
  }

  // Выполняем отложенную предобработку
  is_valid = preprocessing_result_;
  boost::mpi::broadcast(world_, is_valid, 0);
  if (!is_valid) {
    return false;
  }

  // Транслируем grid_sizes_ и step_sizes_ на все процессы
  boost::mpi::broadcast(world_, grid_sizes_, 0);
  boost::mpi::broadcast(world_, step_sizes_, 0);

  // Проверка шагов с использованием MPI
  boost::mpi::broadcast(world_, local_steps_valid_, 0);
  bool all_steps_valid = false;
  boost::mpi::all_reduce(world_, local_steps_valid_, all_steps_valid, std::logical_and<>());
  if (!all_steps_valid) {
    return false;
  }

  // Распределение данных между процессами
  size_t total_size = 1;
  for (auto n : grid_sizes_) {
    total_size *= n;
  }

  size_t p = world_.size();
  size_t chunk_size = total_size / p;
  size_t remainder = total_size % p;
  size_t rank = world_.rank();
  size_t local_size = (rank < remainder) ? chunk_size + 1 : chunk_size;

  local_input_.resize(local_size);

  // Распределение данных
  if (world_.rank() == 0) {
    std::vector<int> send_counts(p);
    std::vector<int> displacements(p);
    size_t offset = 0;
    for (size_t i = 0; i < p; ++i) {
      size_t size = (i < remainder) ? chunk_size + 1 : chunk_size;
      send_counts[i] = static_cast<int>(size);
      displacements[i] = static_cast<int>(offset);
      offset += size;
    }
    boost::mpi::scatterv(world_, input_, send_counts, displacements, local_input_.data(),
                         static_cast<int>(local_input_.size()), 0);
  } else {
    boost::mpi::scatterv(world_, local_input_.data(), static_cast<int>(local_input_.size()), 0);
  }

  // Основные вычисления
  double local_sum = ComputeLocalSum();

  // Сбор результатов от всех процессов
  double total_sum = 0.0;
  boost::mpi::reduce(world_, local_sum, total_sum, std::plus<>(), 0);

  // Обработка результата
  if (world_.rank() == 0) {
    double volume_element = 1.0;
    for (const auto& h : step_sizes_) {
      volume_element *= h;
    }
    output_result_ = total_sum * volume_element;
  }
  return true;
}

bool kharin_m_multidimensional_integral_calc_all::TaskALL::PostProcessingImpl() {
  // Отправка результата всем процессам
  boost::mpi::broadcast(world_, output_result_, 0);

  // Запись результата в выходные данные
  if (!task_data->outputs.empty() && !task_data->outputs_count.empty() && task_data->outputs_count[0] > 0) {
    reinterpret_cast<double*>(task_data->outputs[0])[0] = output_result_;
  }

  return true;
}