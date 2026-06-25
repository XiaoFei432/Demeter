#include "demeter_scheduler.hpp"

namespace {

double clamp_positive(double value, double fallback) {
  return value > 0 ? value : fallback;
}

double estimate_request_input_mb(const FunctionRequest &req) {
  double total_bytes = 0.0;
  for (const auto &arg : req.arguments()) {
    if (!arg.body().empty()) {
      total_bytes += arg.body().size();
    }
  }
  // Function calls that carry only object references have tiny protobuf bodies.
  // Keep a non-zero estimate so the allocation selector remains well-defined.
  return std::max(total_bytes / (1024.0 * 1024.0), 1.0);
}

DemeterAllocation choose_allocation(
    const FunctionRequest &req,
    const DemeterSchedulerConfig &config) {
  if (config.allocations_.empty()) {
    return DemeterAllocation{"default", 1.0, 128};
  }

  const double input_mb = estimate_request_input_mb(req);
  const double required_mb = input_mb / clamp_positive(config.memory_loss_factor_, 0.8);
  DemeterAllocation best = config.allocations_.front();
  bool found = false;
  for (auto alloc : config.allocations_) {
    if (alloc.memory_mb_ >= required_mb) {
      if (!found ||
          alloc.memory_mb_ < best.memory_mb_ ||
          (alloc.memory_mb_ == best.memory_mb_ && alloc.cpu_cores_ < best.cpu_cores_)) {
        best = alloc;
        found = true;
      }
    }
  }
  return found ? best : config.allocations_.back();
}

double node_affinity(
    const Address &addr,
    const NodeStatus &status,
    const FunctionRequest &req,
    const DemeterAllocation &alloc,
    const DemeterSchedulerConfig &config) {
  auto profile_it = config.node_profiles_.find(addr);
  DemeterNodeProfile profile;
  profile.dc_ = addr;
  profile.warm_weight_ = 0.15;
  profile.load_weight_ = 0.35;
  profile.locality_weight_ = 1.0;
  if (profile_it != config.node_profiles_.end()) {
    profile = profile_it->second;
  }

  const bool warm = status.functions_.find(req.name()) != status.functions_.end();
  const double warm_bonus = warm ? profile.warm_weight_ : 0.0;
  const double load_score = profile.load_weight_ * status.avail_executors_;
  const double resource_score = 0.01 * alloc.cpu_cores_ + 0.00001 * alloc.memory_mb_;
  return load_score + warm_bonus + resource_score + profile.locality_weight_;
}

}  // namespace

DemeterSchedulerConfig load_demeter_scheduler_config(const YAML::Node &conf) {
  DemeterSchedulerConfig config;
  YAML::Node demeter = conf["demeter"];
  if (!demeter) {
    return config;
  }

  if (demeter["enabled"]) {
    config.enabled_ = demeter["enabled"].as<unsigned>() != 0;
  }
  if (demeter["memory_loss_factor"]) {
    config.memory_loss_factor_ = demeter["memory_loss_factor"].as<double>();
  }

  if (demeter["allocations"]) {
    for (const auto &item : demeter["allocations"]) {
      DemeterAllocation alloc;
      alloc.name_ = item["name"] ? item["name"].as<string>() : string("custom");
      alloc.cpu_cores_ = item["cpu_cores"] ? item["cpu_cores"].as<double>() : 1.0;
      alloc.memory_mb_ = item["memory_mb"] ? item["memory_mb"].as<unsigned>() : 128;
      config.allocations_.push_back(alloc);
    }
  }
  if (config.allocations_.empty()) {
    vector<unsigned> memory = {128, 512, 1024, 2048, 4096, 8192, 10240};
    vector<double> cpu = {1, 2, 4, 8};
    for (auto c : cpu) {
      for (auto m : memory) {
        config.allocations_.push_back(
            DemeterAllocation{std::to_string(static_cast<int>(c)) + "c-" + std::to_string(m) + "m", c, m});
      }
    }
  }

  if (demeter["nodes"]) {
    for (const auto &item : demeter["nodes"]) {
      if (!item["ip"]) {
        continue;
      }
      Address ip = item["ip"].as<Address>();
      DemeterNodeProfile profile;
      profile.dc_ = item["dc"] ? item["dc"].as<string>() : ip;
      profile.warm_weight_ = item["warm_weight"] ? item["warm_weight"].as<double>() : 0.15;
      profile.load_weight_ = item["load_weight"] ? item["load_weight"].as<double>() : 0.35;
      profile.locality_weight_ = item["locality_weight"] ? item["locality_weight"].as<double>() : 1.0;
      config.node_profiles_[ip] = profile;
    }
  }
  return config;
}

map<Address, string> demeter_schedule_function_call(
    logger log,
    const FunctionCall &call_msg,
    const string &serialized,
    const string &source,
    map<Address, NodeStatus> &node_status_map,
    const DemeterSchedulerConfig &config,
    unsigned &seed) {
  map<Address, string> scheduled_node_msg;
  if (!config.enabled_ || node_status_map.empty()) {
    return scheduled_node_msg;
  }

  map<Address, vector<FunctionRequest>> node_reqs;
  for (const auto &req : call_msg.requests()) {
    double best_score = -std::numeric_limits<double>::infinity();
    Address best_node;
    auto alloc = choose_allocation(req, config);

    for (auto &pair : node_status_map) {
      if (!source.empty() && source == pair.first) {
        continue;
      }
      if (pair.second.avail_executors_ <= 0) {
        continue;
      }
      double score = node_affinity(pair.first, pair.second, req, alloc, config);
      if (score > best_score) {
        best_score = score;
        best_node = pair.first;
      }
    }

    if (best_node.empty() && !node_status_map.empty()) {
      auto it = node_status_map.begin();
      std::advance(it, rand_r(&seed) % node_status_map.size());
      best_node = it->first;
    }
    if (!best_node.empty()) {
      node_reqs[best_node].push_back(req);
      if (node_status_map[best_node].avail_executors_ > 0) {
        node_status_map[best_node].avail_executors_--;
      }
    }
  }

  for (auto &pair : node_reqs) {
    FunctionCall routed_call;
    routed_call.set_app_name(call_msg.app_name());
    routed_call.set_source(call_msg.source());
    routed_call.set_request_id(call_msg.request_id());
    routed_call.set_resp_address(call_msg.resp_address());
    routed_call.set_session_id(call_msg.session_id());
    routed_call.set_sync_data_status(true);
    for (auto &req : pair.second) {
      auto out_req = routed_call.add_requests();
      out_req->CopyFrom(req);
      auto alloc = choose_allocation(*out_req, config);
      out_req->set_cpu_cores(alloc.cpu_cores_);
      out_req->set_memory_mb(alloc.memory_mb_);
    }
    string part_serialized;
    routed_call.SerializeToString(&part_serialized);
    scheduled_node_msg[pair.first] = part_serialized;
  }

  log->info("Demeter scheduler produced {} routed call(s) for app {}",
            scheduled_node_msg.size(), call_msg.app_name());
  return scheduled_node_msg;
}
