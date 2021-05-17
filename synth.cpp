#include <math.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <opencv2/opencv.hpp>

using std::vector;
using std::pair;
using cv::Vec2d;
using cv::Mat;

class CandidateFrame {
public:
  Mat image;
  vector<Vec2d> landmarks; // total 68

  CandidateFrame(Mat image, vector<Vec2d> &landmarks):
    image(image), landmarks(landmarks) {}

  vector<Vec2d> mouth_landmarks() {
    return landmarks.begin() + 48; // mouth landmark ranges from 48 to 67 (61 ,65 not used)
  }

  double is_blinking() {
    vector<Vec2d> leye, reye;
    for (int i = 36; i < 42; i++) {
      leye.push_back(landmarks[i]);
    }
    for (int i = 42; i < 48; i++) {
      reye.push_back(landmarks[i]);
    }
    return (contourArea(leye) + contourArea(reye)) / 2.0 > 20; // threshold for blinking
  }

  static int num_landmarks = 64;
  static int num_mouth_landmarks = 16;
};

// ********************************************
//
//         Candidate Frame Selection
//
// ********************************************

double compute_weight(CandidateFrame &candidate, vector<Vec2d> &target, double sigma) {
  double sum = 0;
  for (int j = 0; j < target.size(); j++) {
    sum += pow(candidate.mouth_landmarks()[j][0] - target[j][0], 2)
    sum += pow(candidate.mouth_landmarks()[j][1] - target[j][1], 2)
  }
  return exp(-sum / (2 * sigma * sigma));
}

double compute_sigma(vector<CandidateFrame> &candidates, vector<CandidateFrame> &video,
    vector<Vec2d> &target, double alpha) {
  double lo = 0.01, hi = 10, mid;
  int iters = 25;
  for (int i = 0; i < iters; i++) {
    mid = (lo + hi) / 2;

    candidate_sum, total_sum = 0, 0;
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

vector<double> compute_weights(vector<CandidateFrame> &candidates, vector<CandidateFrame> &video,
    vector<Vec2d> &target, double alpha) {
  double sigma = compute_sigma(candidates, video, target, alpha);
  vector<double> weights;
  for (int i = 0; i < candidates.size(); i++) {
    weights.push_back(compute_weight(candidates[i], target, sigma));
  }
  return weights;
}

Mat weighted_median(vector<CandidateFrame> &candidates, vector<CandidateFrame> &video,
    vector<Vec2d> &target, double alpha) {
  vector<double> weights = compute_weights(candidates, video, target, alpha);

  int h = candidates[0].height, w = candidates[0].width, ch = candidates[0].channels;
  Mat output(h, w, ch);

  for (int c = 0; c < ch; c++) {
    for (int i = 0; i < h; i++) {
      for (int j = 0; j < w; j++) {

        vector<pair<double, double>> pixel_params; // pixel value, weight of candidate
        for (int k = 0; k < candidates.size(); k++) {
          pixel_params.push_back(candidates[k].at<cv::Vec3b>(i, j)[c], weights[k]);
        }
        std::sort(pixel_params.begin(), value_idx.end());

        double med_sum = 0;
        for (int k = 0; k < pixel_params.size(); k++) {
          med_sum += pixel_params[i].first;
        }
        med_sum /= 2;

        double median, sum = 0;
        for (int k = 0; k < pixel_params.size(); k++) {
          sum += pixel_params[i].first;
          if (sum >= med_sum) {
            median = pixel_params[k];
            break;
          }
        }
        output.at<cv::Vec3b>(i, j)[c] = median;
      }
    }
  }
  return output;
}

Mat warp_adjacent_frames(int i, vector<CandidateFrame> &video) {
  assert (i > 0 && i < video.size() - 1);

  Mat flow; // backward flow
  cv::calcOpticalFlowFarneback(video[i - 1], vide[i + 1], flow); // backward flow
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

int main() {

}
