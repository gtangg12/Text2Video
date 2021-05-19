#include <math.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

using std::vector;
using std::string;
using std::pair;
using cv::Vec2d;
using cv::Mat;
using cv::Point2f;

const int NUM_LANDMARKS = 20;

class CandidateFrame {
public:
  Mat image;
  vector<Vec2d> landmarks; // 20 mouth landmarks

  CandidateFrame(Mat image, vector<Vec2d> &landmarks):
    image(image), landmarks(landmarks) {}
};

// ********************************************
//
//         Candidate Frame Selection
//
// ********************************************

double compute_weight(CandidateFrame* candidate, vector<Vec2d> &target, double sigma) {
  double sum = 0;
  for (int j = 0; j < target.size(); j++) {
    sum += pow(candidate->landmarks[j][0] - target[j][0], 2);
    sum += pow(candidate->landmarks[j][1] - target[j][1], 2);
  }
  //std::cout << sum << std::endl;
  return exp(-sum / (2 * sigma * sigma));
}

vector<CandidateFrame*> top_n_candidates(int n, vector<Vec2d> &target, vector<CandidateFrame*> video) {
   double sigma = 10; // anything works
   vector<pair<double, CandidateFrame*>> stats;
   for (int i = 0; i < video.size(); i++) {
      stats.push_back({-compute_weight(video[i], target, sigma), video[i]});
   }
   std::sort(stats.begin(), stats.end());
   vector<CandidateFrame*> candidates;
   for (int i = 0; i < n; i++) {
      candidates.push_back(stats[i].second);
   }
   return candidates;
}

double compute_sigma(vector<CandidateFrame*> candidates, vector<CandidateFrame*> video,
    vector<Vec2d> &target, double alpha) {
  double lo = 0.01, hi = 50, mid;
  int iters = 30;
  for (int i = 0; i < iters; i++) {
    mid = (lo + hi) / 2;

    double candidate_sum = 0, total_sum = 0;
    for (int j = 0; j < candidates.size(); j++) {
      candidate_sum += compute_weight(candidates[j], target, mid);
    }
    for (int j = 0; j < video.size(); j++) {
      total_sum += compute_weight(video[j], target, mid);
    }

    if (candidate_sum > alpha * total_sum) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return lo;
}

vector<double> compute_weights(vector<CandidateFrame*> candidates, vector<CandidateFrame*> video,
    vector<Vec2d> &target, double alpha) {
  double sigma = compute_sigma(candidates, video, target, alpha);
  vector<double> weights;
  for (int i = 0; i < candidates.size(); i++) {
    weights.push_back(compute_weight(candidates[i], target, sigma));
  }
  return weights;
}

// solve with https://ita.skanev.com/09/problems/02.html (post-office location problem)
Mat weighted_median(vector<CandidateFrame*> candidates, vector<CandidateFrame*> video,
    vector<Vec2d> &target, double alpha) {
  vector<double> weights = compute_weights(candidates, video, target, alpha);

  Mat output(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));

  int h = candidates[0]->image.rows, w = candidates[0]->image.cols, ch = 3;
  std::cout << h << ' ' << w  << std::endl;
  for (int c = 0; c < 3; c++) {
    for (int i = 0; i < h; i++) {
      for (int j = 0; j < w; j++) {

        vector<pair<double, double>> pixel_params; // pixel value, weight of candidate
        for (int k = 0; k < candidates.size(); k++) {
          pixel_params.push_back({candidates[k]->image.at<cv::Vec3b>(i, j)[c], weights[k]});
        }
        std::sort(pixel_params.begin(), pixel_params.end());

        double med_sum = 0;
        for (int k = 0; k < pixel_params.size(); k++) {
          med_sum += pixel_params[i].second;
        }
        med_sum /= 2;
        med_sum = 0;

        double median, sum = 0;
        for (int k = 0; k < pixel_params.size(); k++) {
          sum += pixel_params[i].second;
          if (sum >= med_sum) {
            median = pixel_params[k].first;
            break;
          }
        }

        median = candidates[0]->image.at<cv::Vec3b>(i, j)[c];
        output.at<cv::Vec3b>(i, j)[c] = median;
      }
    }
  }
  return output;
}

/*
Mat warp_adjacent_frames(int i, vector<CandidateFrame> &video) {
  assert (i > 0 && i < video.size() - 1);

  Mat flow; // backward flow
  cv::calcOpticalFlowFarneback(video[i - 1], video[i + 1], flow); // backward flow
  Mat map_f(flow.size(), CV_32FC2);
  Mat map_b(flow.size(), CV_32FC2);

  for (int y = 0; y < flow.rows; ++y) {
    for (int x = 0; x < flow.cols; ++x) {
        Point2f f = flow.at<Point2f>(y, x);
        map_b.at<Point2f>(y, x) = Point2f(x + 0.5 * f.x, y + 0.5 * f.y);
        map_f.at<Point2f>(y, x) = Point2f(x - 0.5 * f.x, y - 0.5 * f.y);
    }
  }

  Mat mid_f, mid_b;
  cv::remap(video[i - 1], mid_f, map_f);
  cv::remap(video[i + 1], mid_b, map_b);

  return 0.5 * (mid_f + mid_b);
}
*/

vector<CandidateFrame*> load_data(string path) {
   vector<CandidateFrame*> ret;

   int start_frame = 10, end_frame = 80;
   for (int frame = start_frame; frame <= end_frame; frame++) {
      string imgpath = path + "/output_frame_" + std::to_string(frame) + ".jpg";
      string landmarks_path = imgpath + "_mouth_landmarks.csv";
      // std::cout << landmarks_path << std::endl;

      Mat image = cv::imread(imgpath, cv::IMREAD_COLOR);

      std::ifstream fin;
      fin.open(landmarks_path);
      vector<Vec2d> points;
      int x, y;
      for (int i = 0; i < NUM_LANDMARKS; i++) {
         fin >> x >> y;
         points.push_back(Vec2d(y, x));
      }
      CandidateFrame *f = new CandidateFrame(image, points);
      ret.push_back(f);
   }
   return ret;
}

int main() {
   string path = "frontalized";
   vector<CandidateFrame*> video = load_data(path);
   /*
   for (int i = 0; i < candidates.size(); i++) {
      for (Vec2d &v : candidates[i]->landmarks) {
         // std::cout << v[0] << ' ' << v[1] << std::endl;
         cv::circle(candidates[i]->image, cv::Point(v[1], v[0]), 5, cv::Scalar(255,255,255));
      }
      cv::imshow("Frame", candidates[i]->image);
      cv::waitKey(0);
   }*/

   CandidateFrame* target = video.back();
   video.pop_back();
   for (int i = 0; i < 20; i++) video.pop_back();
   /*
   for (int i = 0; i < candidates.size(); i++) {
      double w = compute_weight(target, candidates[i]->landmarks, 8);
      std::cout << i << ' ' << w << std::endl;
   }*/
   int n = 8;
   vector<CandidateFrame*> candidates = top_n_candidates(n, target->landmarks, video);
   /*
   for (CandidateFrame* f : candidates) {
      cv::imshow("Frame", f->image);
      cv::waitKey(0);
   }*/
   /*
   double sigma = compute_sigma(candidates, video, target->landmarks, 0.9);
   std::cout << sigma << std::endl;
   */
   /*
   vector<double> weights = compute_weights(candidates, video, target->landmarks, 0.9);
   for (double w : weights)
      std::cout << w << std::endl;
   */
   Mat output = weighted_median(candidates, video, target->landmarks, 0.9);
   cv::imshow("Frame", output);
   cv::waitKey(0);

   for (CandidateFrame* f : video) {
      free(f);
   }
}
