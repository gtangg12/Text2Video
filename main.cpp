#include <math.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <opencv2/opencv.hpp>

using std::vector;
using std::pair;
using cv::Vec2d;

class CandidateFrame {
public:
  cv::Mat image;
  vector<Vec2d> landmarks; // total 68

  CandidateFrame(cv::Mat image, vector<Vec2d> &landmarks):
    image(image), landmarks(landmarks) {}

  vector<Vec2d> mouth_landmarks() {
    return landmarks.begin() + 48; // mouth landmark ranges from 48 to 67
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

// ********************************************
//
//              Video Retiming
//
// ********************************************

double compute_motion(CandidateFrame &prev, CandidateFrame &cur) {
  double motion = 0;
  for (int i = 0; i < a.num_mouth_landmarks; i++) {
    double tmp = 0;
    tmp += pow(prev.mouth_landmarks()[i][0] - cur.mouth_landmarks()[i][0], 2);
    tmp += pow(prev.mouth_landmarks()[i][0] - cur.mouth_landmarks()[i][0], 2);
    motion += sqrt(tmp);
  }
  return motion;
}

double compute_V(int j, double a_B, vector<CandidateFrame> &video) {
  assert (j > 0);
  return compute_motion(video[j - 1], video[j]) + a_B * cur.is_blinking();
}

double compute_G(int i, int j, double a_B, double a_u, vector<CandidateFrame> &video,
    vector<bool> &A_arr) {
  assert (i > 0 && j > 0);
  double G = A_arr[i] == 0 ? compute_V(j, a_B, video) : 0;
  if (i >= 3 && A_arr[i - 3] == 0 && A_arr[i - 2] == 1) {
    G -= a_u * compute_V(j);
  }
  return G;
}

vector<pair<int, int>> min_cost_path(vector<CandidateFrame> &synth, vector<CandidateFrame> &video,
    vector<bool> &A_arr) {
  double a_B = 1;
  double a_u = 2;
  double a_s = 2;

  int N = synth.size()
  int M = video.size();
  double **dp = new double[N][M][2];
  bool **parent = new bool[N][M];

  for (int i = 0; i < M; i++) {
    dp[0][i][0] = A_arr[i] == 0 ? compute_V(i, a_B, video) : 0;
    dp[0][i][1] = DBL_MAX;
  }
  for (int i = 0; i < N; i++) {
    dp[i][0][0] = DBL_MAX;
    dp[i][0][1] = DBL_MAX;
  }
  dp[0][0][0] = 0;

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      double G = compute_G(i, j, a_B, a_u, video), mn;

      if (dp[i - 1][j - 1][0] < dp[i - 1][j - 1][1]) {
        mn = dp[i - 1][j - 1][0];
        parent[i][j] = 0;
      } else {
        mn = dp[i - 1][j - 1][1];
        parent[i][j] = 1;
      }

      dp[i][j][0] = mn + G;
      dp[i][j][1] = dp[i][j][0] + a_s * V(j, a_B, video) + G;
    }
  }

  double mn_cost = DBL_MAX;
  int mn_idx, ch;
  for (int i = 0; i < M; i++) {
    if (dp[N - 1][i][0] < mn_cost) {
      mn_cost = dp[N - 1][i][0];
      mn_idx = i;
      ch = 0;
    }
    if (dp[N - 1][i][1] < mn_cost) {
      mn_cost = dp[N - 1][i][1];
      mn_idx = i;
      ch = 1;
    }
  }



  delete [] dp;
}

Mat warp_frame(int i, vector<CandidateFrame> &video) {
  assert (i > 0 && i < video.size() - 1);
}

int main() {

}
