// SPDX-License-Identifier: GPL-3.0-or-later
// Minimal ROS-free EuRoC folder replay for the fixed OpenVINS checkout.
// No simulator, synthetic features, GT loading, or GT initialization.
// Normal VioManager KLT/initialization/propagation/update paths are used.
// Camera delivery waits for one later IMU sample, matching ROS1Visualizer.
#include "core/VioManager.h"
#include "core/VioManagerOptions.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "utils/print.h"
#include "utils/sensor_data.h"
#include <Eigen/Geometry>
#include <algorithm>
#include <boost/filesystem.hpp>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgcodecs.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
namespace fs = boost::filesystem;
struct Image {
  std::int64_t ns;
  std::string name;
};
struct Stereo {
  std::int64_t ns;
  std::string left, right;
};
struct Imu {
  std::int64_t ns;
  ov_core::ImuData data;
};
double ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}
double seconds(std::int64_t ns) { return static_cast<double>(ns) * 1e-9; }
std::string trim(std::string s) {
  const auto first = s.find_first_not_of(" \t\r\n");
  if (first == std::string::npos)
    return {};
  return s.substr(first, s.find_last_not_of(" \t\r\n") - first + 1);
}
std::vector<std::string> split(const std::string &line) {
  std::istringstream in(line);
  std::string field;
  std::vector<std::string> out;
  while (std::getline(in, field, ','))
    out.push_back(trim(field));
  return out;
}
std::vector<std::vector<std::string>> csv(const fs::path &path,
                                          std::size_t columns) {
  std::ifstream input(path.string());
  if (!input)
    throw std::runtime_error("cannot read " + path.string());
  std::vector<std::vector<std::string>> rows;
  std::string line;
  std::size_t number = 0;
  while (std::getline(input, line)) {
    ++number;
    line = trim(line);
    if (line.empty() || line.front() == '#')
      continue;
    auto fields = split(line);
    if (fields.size() != columns)
      throw std::runtime_error("bad CSV width at " + path.string() + ":" +
                               std::to_string(number));
    rows.push_back(std::move(fields));
  }
  if (rows.empty())
    throw std::runtime_error("empty CSV " + path.string());
  return rows;
}
template <class T>
void order_unique(std::vector<T> &rows, const std::string &name) {
  std::sort(rows.begin(), rows.end(),
            [](const T &a, const T &b) { return a.ns < b.ns; });
  for (std::size_t i = 1; i < rows.size(); ++i)
    if (rows[i - 1].ns == rows[i].ns)
      throw std::runtime_error("duplicate timestamp in " + name);
}
std::vector<Image> load_images(const fs::path &path) {
  std::vector<Image> out;
  for (const auto &row : csv(path, 2)) {
    fs::path filename(row[1]);
    if (filename.is_absolute() || filename.has_parent_path())
      throw std::runtime_error("camera filename must be a basename: " + row[1]);
    out.push_back({std::stoll(row[0]), row[1]});
  }
  order_unique(out, path.string());
  return out;
}
std::vector<Imu> load_imu(const fs::path &path) {
  std::vector<Imu> out;
  for (const auto &row : csv(path, 7)) {
    Imu sample;
    sample.ns = std::stoll(row[0]);
    sample.data.timestamp = seconds(sample.ns);
    for (int k = 0; k < 3; ++k) {
      sample.data.wm(k) = std::stod(row[1 + k]);
      sample.data.am(k) = std::stod(row[4 + k]);
    }
    if (!sample.data.wm.allFinite() || !sample.data.am.allFinite())
      throw std::runtime_error("nonfinite IMU sample");
    out.push_back(sample);
  }
  order_unique(out, path.string());
  return out;
}
std::string quote(const std::string &text) {
  std::string out = "\"";
  for (char c : text) {
    if (c == '"' || c == '\\')
      out += '\\';
    if (c == '\n')
      out += "\\n";
    else if (c == '\r')
      out += "\\r";
    else if (c == '\t')
      out += "\\t";
    else
      out += c;
  }
  return out + '"';
}
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 4 || argc > 5) {
      std::cerr << "usage: run_euroc_folder CONFIG_YAML DATASET_ROOT "
                   "OUTPUT_PREFIX [DURATION_SECONDS]\n";
      return 2;
    }
    const fs::path config = fs::absolute(argv[1]);
    fs::path mav = fs::absolute(argv[2]);
    if (fs::is_directory(mav / "mav0"))
      mav /= "mav0";
    const fs::path prefix = fs::absolute(argv[3]);
    if (!prefix.parent_path().empty())
      fs::create_directories(prefix.parent_path());
    const double duration = argc == 5 ? std::stod(argv[4]) : -1.0;
    if (!std::isfinite(duration) || duration == 0)
      throw std::invalid_argument(
          "duration must be positive, or negative for full replay");

    const auto cam0 = load_images(mav / "cam0/data.csv");
    const auto cam1 = load_images(mav / "cam1/data.csv");
    const auto imu = load_imu(mav / "imu0/data.csv");
    std::vector<Stereo> pairs;
    std::size_t left = 0, right = 0, unmatched = 0;
    while (left < cam0.size() && right < cam1.size()) {
      if (cam0[left].ns < cam1[right].ns) {
        ++left;
        ++unmatched;
      } else if (cam1[right].ns < cam0[left].ns) {
        ++right;
        ++unmatched;
      } else {
        pairs.push_back({cam0[left].ns, cam0[left].name, cam1[right].name});
        ++left;
        ++right;
      }
    }
    unmatched += cam0.size() - left + cam1.size() - right;
    if (pairs.empty())
      throw std::runtime_error("no exactly synchronized EuRoC stereo pairs");
    const double first_camera = seconds(pairs.front().ns);
    const double last_requested =
        duration < 0 ? seconds(pairs.back().ns) : first_camera + duration;

    auto parser = std::make_shared<ov_core::YamlParser>(config.string());
    std::string verbosity = "WARNING";
    parser->parse_config("verbosity", verbosity);
    ov_core::Printer::setPrintLevel(verbosity);
    ov_msckf::VioManagerOptions params;
    params.print_and_load(parser);
    if (!parser->successful())
      throw std::runtime_error("OpenVINS configuration parse failed");
    if (params.state_options.num_cameras != 2 || !params.use_stereo)
      throw std::runtime_error(
          "runner requires the original stereo EuRoC configuration");
    // Only execution controls change. Estimator/calibration/noise/feature/
    // initialization mathematics stay exactly as loaded from CONFIG_YAML.
    params.num_opencv_threads = 1;
    params.use_multi_threading_pubs = false;
    params.use_multi_threading_subs = false;
    params.init_options.init_dyn_mle_max_threads = 1;
    Eigen::setNbThreads(1);
    cv::setNumThreads(1);
    cv::ocl::setUseOpenCL(false);
    auto vio = std::make_shared<ov_msckf::VioManager>(params);

    std::ofstream states(prefix.string() + "_states.csv");
    std::ofstream trajectory(prefix.string() + "_trajectory.txt");
    if (!states || !trajectory)
      throw std::runtime_error("cannot create output files");
    states << std::setprecision(17)
           << "frame,cam_time,state_time,imu_time,last_fed_imu_time,elapsed_s,"
              "initialized,cam_ms,imu_ms,decode_ms,"
           << "px,py,pz,qx,qy,qz,qw,vx,vy,vz,bgx,bgy,bgz,bax,bay,baz,pos_var_x,"
              "pos_var_y,pos_var_z,dt_camimu\n";
    trajectory << std::setprecision(17)
               << "# time_in_imu_clock tx ty tz qx qy qz qw; Hamilton "
                  "IMU-to-estimator-world; NO GT alignment\n";
    const char *selected = std::getenv("OV_GAUSSIAN_BACKEND");
    const std::string backend = selected ? selected : "original";
#ifndef OV_GAUSSIAN_INTEGRATION
    if (backend != "original")
      throw std::runtime_error(
          "clean upstream executable only supports original");
#endif
    std::cerr << "EuRoC replay: backend=" << backend
              << ", GT initialization=DISABLED, stereo=" << pairs.size()
              << ", unmatched image records=" << unmatched << '\n';

    std::deque<std::size_t> pending;
    std::size_t ci = 0, ii = 0, frames = 0, initialized_frames = 0,
                throttled = 0;
    double last_accepted_cam = -std::numeric_limits<double>::infinity();
    double imu_ms = 0, total_cam_ms = 0, total_decode_ms = 0, total_imu_ms = 0;
    double first_initialized = std::numeric_limits<double>::quiet_NaN();
    double final_camera = first_camera;
    const auto replay_start = Clock::now();
    while (ii < imu.size()) {
      const bool camera_remains =
          ci < pairs.size() && seconds(pairs[ci].ns) <= last_requested;
      // Merge raw sensor timestamps; camera records are only queued here.
      if (camera_remains && pairs[ci].ns <= imu[ii].ns) {
        const double time = seconds(pairs[ci].ns);
        if (params.track_frequency > 0 &&
            time < last_accepted_cam + 1.0 / params.track_frequency)
          ++throttled;
        else {
          pending.push_back(ci);
          last_accepted_cam = time;
        }
        ++ci;
        continue;
      }
      if (!camera_remains && pending.empty())
        break;
      const auto imu_start = Clock::now();
      vio->feed_measurement_imu(imu[ii].data);
      const double this_imu_ms = ms(imu_start);
      imu_ms += this_imu_ms;
      total_imu_ms += this_imu_ms;
      const double last_fed_imu = imu[ii].data.timestamp;
      ++ii;

      while (!pending.empty()) {
        const Stereo &pair = pairs[pending.front()];
        const double cam_time = seconds(pair.ns);
        const double dt = vio->get_state()->_calib_dt_CAMtoIMU->value()(0);
        // Strict inequality supplies the right interpolation endpoint,
        // exactly as the upstream single-thread ROS callback queue does.
        if (!(cam_time + dt < last_fed_imu))
          break;
        ov_core::CameraData camera;
        camera.timestamp = cam_time;
        camera.sensor_ids = {0, 1};
        const auto decode_start = Clock::now();
        camera.images.push_back(cv::imread(
            (mav / "cam0/data" / pair.left).string(), cv::IMREAD_GRAYSCALE));
        camera.images.push_back(cv::imread(
            (mav / "cam1/data" / pair.right).string(), cv::IMREAD_GRAYSCALE));
        for (int cam = 0; cam < 2; ++cam) {
          if (camera.images[cam].empty())
            throw std::runtime_error("image decode failed at " +
                                     std::to_string(pair.ns));
          if (params.use_mask)
            camera.masks.push_back(params.masks.at(cam));
          else
            camera.masks.push_back(
                cv::Mat::zeros(camera.images[cam].size(), CV_8UC1));
        }
        const double decode_ms = ms(decode_start);
        total_decode_ms += decode_ms;
        const auto camera_start = Clock::now();
        vio->feed_measurement_camera(camera);
        const double camera_ms = ms(camera_start);
        total_cam_ms += camera_ms;
        const auto state = vio->get_state();
        const bool initialized = vio->initialized();
        const double estimated_dt = state->_calib_dt_CAMtoIMU->value()(0);
        states << frames << ',' << cam_time << ',' << state->_timestamp << ','
               << state->_timestamp + estimated_dt << ',' << last_fed_imu << ','
               << cam_time - first_camera << ',' << initialized << ','
               << camera_ms << ',' << imu_ms << ',' << decode_ms;
        if (initialized) {
          if (!initialized_frames)
            first_initialized = cam_time - first_camera;
          ++initialized_frames;
          const auto p = state->_imu->pos();
          // Convert from OpenVINS' JPL G->I matrix to an explicit Hamilton
          // quaternion for I->G. This avoids ambiguous quaternion conventions.
          Eigen::Quaterniond q(state->_imu->Rot().transpose());
          q.normalize();
          const auto v = state->_imu->vel();
          const auto bg = state->_imu->bias_g(), ba = state->_imu->bias_a();
          for (int k = 0; k < 3; ++k)
            states << ',' << p(k);
          states << ',' << q.x() << ',' << q.y() << ',' << q.z() << ','
                 << q.w();
          for (const auto &value : {v, bg, ba})
            for (int k = 0; k < 3; ++k)
              states << ',' << value(k);
          const Eigen::MatrixXd covariance =
              ov_msckf::StateHelper::get_marginal_covariance(state,
                                                             {state->_imu});
          for (int k = 0; k < 3; ++k)
            states << ',' << covariance(3 + k, 3 + k);
          trajectory << state->_timestamp + estimated_dt << ' ' << p.transpose()
                     << ' ' << q.x() << ' ' << q.y() << ' ' << q.z() << ' '
                     << q.w() << '\n';
        } else {
          for (int k = 0; k < 19; ++k)
            states << ",nan";
        }
        states << ',' << estimated_dt << '\n';
        pending.pop_front();
        ++frames;
        imu_ms = 0;
        final_camera = cam_time;
      }
    }
    states.close();
    trajectory.close();
    const double replay_wall_ms = ms(replay_start);
#ifdef OV_GAUSSIAN_INTEGRATION
    if (vio->get_state()->_gaussian)
      vio->get_state()->_gaussian->write_statistics(prefix.string() +
                                                    "_backend.json");
#endif
    std::ofstream summary(prefix.string() + "_run.json");
    summary << std::setprecision(17) << "{\n  \"backend\": " << quote(backend)
            << ",\n  \"config\": " << quote(config.string())
            << ",\n  \"mav0\": " << quote(mav.string())
            << ",\n  \"data_kind\": \"real EuRoC grayscale images and IMU\""
            << ",\n  \"gt_initialized\": false,\n  \"gt_loaded\": false"
            << ",\n  \"frames\": " << frames
            << ",\n  \"initialized_frames\": " << initialized_frames
            << ",\n  \"unmatched_camera_records\": " << unmatched
            << ",\n  \"throttled_pairs\": " << throttled
            << ",\n  \"tail_pairs_without_future_imu\": " << pending.size()
            << ",\n  \"imu_samples_fed\": " << ii
            << ",\n  \"elapsed_camera_s\": " << final_camera - first_camera
            << ",\n  \"first_initialized_elapsed_s\": ";
    if (std::isfinite(first_initialized))
      summary << first_initialized;
    else
      summary << "null";
    summary << ",\n  \"total_camera_ms\": " << total_cam_ms
            << ",\n  \"total_decode_ms\": " << total_decode_ms
            << ",\n  \"total_imu_feed_ms\": " << total_imu_ms
            << ",\n  \"replay_wall_ms\": " << replay_wall_ms << "\n}\n";
    std::cerr << "processed=" << frames
              << ", initialized=" << initialized_frames
              << ", remaining camera queue=" << pending.size() << '\n';
    if (!initialized_frames) {
      std::cerr << "VIO never initialized from its actual measurements; run is "
                   "a failure.\n";
      return 3;
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "EuRoC replay failed: " << error.what() << '\n';
    return 1;
  }
}
