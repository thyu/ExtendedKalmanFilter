#include <cstdio>
#include <cstdint>
#include <random>
#include <chrono>

#include <opencv2/opencv.hpp>

#include "ExtendedKalmanFilter.h"

// implement the EKF system here, we need to define several functions
class RectEKF : public thyu::ExtendedKalmanFilter<float, 4, 2, 2> {

  public:

  virtual State getPrediction(const State& state, const Control& control){

      State newState = State::Zero();  
      newState(2) = std::cos(control(0)) * control(1); // dx
      newState(3) = std::sin(control(0)) * control(1); // dy
      newState(0) = state(0) + state(2); // x
      newState(1) = state(1) + state(3); // y
      return newState;
  }
  
  virtual Measurement getMeasurement(const State& state) {
    Measurement M;
    // center point location
    M(0) = state(0);
    M(1) = state(1);
    return M;
  }
  
  protected:

  virtual StateJacobian getStateJacobian(const State& state, const Control& control){
    StateJacobian J;
    J.setZero();
    J(0,0) = 1; 
    J(0,2) = 1;
    J(1,1) = 1;
    J(1,3) = 1;
    J(2,2) = 0;
    J(3,3) = 0;
    return J;
  }

  virtual StateError getStateError(const State& state, const Control& control){
    return StateError::Identity() * 1;
  }

  virtual MeasurementJacobian getMeasurementJacobian(const State& state, const Measurement& measurement){
    MeasurementJacobian J;
    J.setZero();
    J(0,0) = 1;
    J(1,1) = 1;
    return J;

  }

  virtual MeasurementError getMeasurementError(const State& state, const Measurement& measurement){
    return MeasurementError::Identity() * 100;
  }

};
  
// create an cv::Rect from State vector
cv::Rect rectFromState(const RectEKF::State& s, int32_t offset_x, int32_t offset_y){
  const int size = 100;
  auto r = cv::Rect(s(0) - size + offset_x, s(1) - size + offset_y, 2 * size, 2 * size);
  return r;
}

int main(int argc, char** argv){

  // some constant
  constexpr int32_t imageWidth = 800;
  constexpr int32_t imageHeight = 800;
  constexpr int32_t rotateRadius = 200;
  constexpr float PI = 3.14159;

  // random number generator
  std::default_random_engine generator;
  generator.seed( std::chrono::system_clock::now().time_since_epoch().count() );
  std::normal_distribution<float> noise(0, 1);

  int frame = 0;

  RectEKF ekf;
  RectEKF::State groundTruthState;
  groundTruthState << rotateRadius, 0, 0, 0;
  ekf.setState(groundTruthState);


  while (1) {
  
    cv::Mat image = cv::Mat::zeros(imageHeight, imageWidth, CV_8UC3);

    // generate control - some random number
    RectEKF::Control control;
    control(0) = (90 + frame) * (PI/180.0);
    control(1) = rotateRadius * (PI/180.0);
   
    // update groundtruth state 
    groundTruthState = ekf.getPrediction(groundTruthState, control);
   
    // compute simulated noisy state
    RectEKF::State noisyState = groundTruthState;
    noisyState(0) += (noise(generator) * 10);
    noisyState(1) += (noise(generator) * 10);
    noisyState(2) += (noise(generator) * 3);
    noisyState(3) += (noise(generator) * 3);
    RectEKF::Measurement noisyMeasurement = ekf.getMeasurement(noisyState);

    // run EKF
    ekf.predict(control);
    ekf.update(noisyMeasurement);

    // Visualize
    cv::rectangle(image, rectFromState(groundTruthState, imageWidth/2, imageHeight/2), cv::Scalar(255, 0, 0), 2);
    cv::rectangle(image, rectFromState(noisyState, imageWidth/2, imageHeight/2), cv::Scalar(0, 255, 0), 2);
    cv::rectangle(image, rectFromState(ekf.getState(), imageWidth/2, imageHeight/2), cv::Scalar(0, 0, 255), 2);

    cv::imshow("Result", image);

    auto k = cv::waitKey(30);
    if (k == 27){
      break;
    }

    frame++;

  }

  return 0;
}
