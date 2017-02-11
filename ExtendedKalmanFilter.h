#ifndef _THYU_EKF_H_
#define _THYU_EKF_H_

#include <Eigen/Dense>

namespace thyu {

  template<class DType, int32_t StateSize, int32_t ControlSize, int32_t MeasurementSize >
    class FilterBase {

      public:

        using State = Eigen::Matrix<DType, StateSize, 1>;
        using Control = Eigen::Matrix<DType, ControlSize, 1>;
        using StateCovariance = Eigen::Matrix<DType, StateSize, StateSize>;
        using Measurement = Eigen::Matrix<DType, MeasurementSize, 1>;

        FilterBase() :
          state_(State::Zero()),
          stateCov_(StateCovariance::Identity()) { }

        virtual ~FilterBase() { /* do nothing  */ }

        State getState(){
          return this->state_;
        }

        StateCovariance getStateCovariance() {
          return this->stateCov_;
        }

        void setState(const State& newState){
          this->state_ = newState;
        }

        void setStateCovariance(const StateCovariance& newStateCov){
          this->stateCov_ = newStateCov;
        }

        virtual void predict(const Control& control) = 0;
        virtual void update(const Measurement& measurement) = 0;

      protected: 

        State state_;
        StateCovariance stateCov_;

    };

  template<class DType, int32_t StateSize, int32_t ControlSize, int32_t MeasurementSize >
    class ExtendedKalmanFilter : public FilterBase<DType, StateSize, ControlSize, MeasurementSize> {

      public:

        using State = Eigen::Matrix<DType, StateSize, 1>;
        using Control = Eigen::Matrix<DType, ControlSize, 1>;
        using StateCovariance = Eigen::Matrix<DType, StateSize, StateSize>;
        using StateJacobian = Eigen::Matrix<DType, StateSize, StateSize>;
        using StateError = Eigen::Matrix<DType, StateSize, StateSize>;
        using Measurement = Eigen::Matrix<DType, MeasurementSize, 1>;
        using MeasurementJacobian = Eigen::Matrix<DType, MeasurementSize, StateSize>;
        using MeasurementError = Eigen::Matrix<DType, MeasurementSize, MeasurementSize>;

        ExtendedKalmanFilter() : FilterBase<DType, StateSize, ControlSize, MeasurementSize>() { }

        virtual ~ExtendedKalmanFilter(){ /* do nothing */ }

        virtual void predict(const Control& control = Control::Zero()){

          auto stateJacobian = getStateJacobian(this->state_, control);
          auto stateNoise = getStateError(this->state_, control);

          // update state and state cov
          this->state_ = getPrediction(this->state_, control);
          this->stateCov_ = (stateJacobian * this->stateCov_ * stateJacobian.transpose()) + stateNoise;

        };

        virtual void update(const Measurement& measurement = Measurement::Zero()){
          // Measurement jacobian
          auto mj = getMeasurementJacobian(this->state_, measurement);
          // Innocation Covariance
          auto innoCov = (mj * this->stateCov_ * mj.transpose()) + getMeasurementError(this->state_, measurement);
          // Kalman gain
          auto kGain = this->stateCov_ * mj.transpose() * innoCov.inverse();
          // update state and state cov
          this->state_ += kGain * (measurement - getMeasurement(this->state_));
          this->stateCov_ -= kGain * mj * this->stateCov_;
        };
       
        // You need to implement them
        virtual Measurement getMeasurement(const State& state) = 0;
        virtual State getPrediction(const State& state, const Control& control) = 0;

      protected:

        // You need to implement them
        virtual StateJacobian getStateJacobian(const State& state, const Control& control) = 0; 
        virtual StateError getStateError(const State& state, const Control& control) = 0;
        virtual MeasurementJacobian getMeasurementJacobian(const State& state, const Measurement& measurement) = 0;
        virtual MeasurementError getMeasurementError(const State& state, const Measurement& measurement) = 0;

    };

}

#endif // _THYU_EKF_H_
