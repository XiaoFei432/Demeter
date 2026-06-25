#ifndef INCLUDE_COORDINATOR_DEMETER_SCHEDULER_HPP_
#define INCLUDE_COORDINATOR_DEMETER_SCHEDULER_HPP_

#include <limits>
#include "coord_handlers.hpp"
#include "yaml-cpp/yaml.h"

struct DemeterAllocation {
  string name_;
  double cpu_cores_;
  unsigned memory_mb_;
};

struct DemeterNodeProfile {
  string dc_;
  double warm_weight_;
  double load_weight_;
  double locality_weight_;
};

struct DemeterSchedulerConfig {
  bool enabled_;
  double memory_loss_factor_;
  vector<DemeterAllocation> allocations_;
  map<Address, DemeterNodeProfile> node_profiles_;

  DemeterSchedulerConfig() {
    enabled_ = false;
    memory_loss_factor_ = 0.8;
  }
};

DemeterSchedulerConfig load_demeter_scheduler_config(const YAML::Node &conf);

map<Address, string> demeter_schedule_function_call(
    logger log,
    const FunctionCall &call_msg,
    const string &serialized,
    const string &source,
    map<Address, NodeStatus> &node_status_map,
    const DemeterSchedulerConfig &config,
    unsigned &seed);

#endif // INCLUDE_COORDINATOR_DEMETER_SCHEDULER_HPP_
