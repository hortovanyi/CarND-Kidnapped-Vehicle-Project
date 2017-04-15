/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <map>
#include <cctype>

#include "particle_filter.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
  //   x, y, theta and their uncertainties from GPS) and all weights to 1.
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  // set the number of particles
  num_particles = 1000;

  // measurement yaw
  m_yaw = theta;

  // Set standard deviations for x, y, and psi
  double std_x = std[0];
  double std_y = std[1];
  double std_theta = std[2];

  // initialise the particles
  default_random_engine gen;

  // create gaussian distributions
  normal_distribution<double> dist_x(x, std_x);
  normal_distribution<double> dist_y(y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);

  // set particle weights to 1
  weights.clear();
  weights.resize(num_particles, 1.0);

  // initialise particles
  particles.clear();
  particles.resize(num_particles);
  int id = 0;
  for (auto &p : particles) {
    p.id = id++;
    p.x = dist_x(gen);
    p.y = dist_x(gen);
    p.theta = dist_theta(gen);
    p.weight = 1.0;
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/

  // Set standard deviations for x, y, and theta
  double std_x = std_pos[0];
  double std_y = std_pos[1];
  double std_theta = std_pos[2];

  // measurement yaw
  m_yaw = yaw_rate * delta_t;

  // initialise the random number generator
  random_device rd;
  mt19937 gen(rd());

  for (auto &p : particles) {
    double yaw = p.theta;

    if (fabs(yaw_rate) > 0.001) {
      // update x and y position and yaw rate
      p.x += velocity / yaw_rate * (sin(yaw + m_yaw) - sin(yaw));
      p.y += velocity / yaw_rate * (cos(yaw) - cos(yaw + m_yaw));
      p.theta += m_yaw;
    } else {
      // update x and y position, yaw stays the same
      p.x += velocity * delta_t * cos(yaw);
      p.y += velocity * delta_t * sin(yaw);
    }

    // create gausian distributions
    normal_distribution<double> dist_x(p.x, std_x);
    normal_distribution<double> dist_y(p.y, std_y);
    normal_distribution<double> dist_theta(p.theta, std_theta);

    // add noise
    p.x += dist_x(gen);
    p.y += dist_y(gen);
    p.theta += dist_theta(gen);
  }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted,
                                     std::vector<LandmarkObs>& observations) {
  // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
  //   implement this method and use it as a helper during the updateWeights phase.

  // for each observation find the nearest landmark
  for (auto &observation : observations) {
    double closest_dist = numeric_limits<double>::max();
    for (auto prediction : predicted) {

      double d = dist(observation.x, observation.y, prediction.x, prediction.y);

      // found a predicted landmark closer to the observation
      if (d < closest_dist) {
        // reassign this predicted landmark id to the current
        observation.id = prediction.id;

        // save current as closest
        closest_dist = d;
      }
    }
  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations,
                                   Map map_landmarks) {
  // TODO: Update the weights of each particle using a multi-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation
  //   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account
  //   for the fact that the map's y-axis actually points downwards.)
  //   http://planning.cs.uiuc.edu/node99.html

  // initialise measurement covariance matrix
  MatrixXd measurementCovar;
  measurementCovar << std_landmark[0], 0, 0, std_landmark[1];

  // initialise multi-variate gaussian distribution
  VectorXd x, mu;
  MatrixXd inverseMeasurementCovar = measurementCovar.inverse();
  MatrixXd c2 = 2 * M_PI * measurementCovar;
  double c3 = sqrt(c2.determinant());


  // for each particle
  for (auto &p : particles) {
    // convert observations to map space http://planning.cs.uiuc.edu/node99.html
    vector<LandmarkObs> observations_map(observations);
    for (auto &o : observations_map) {
      o.x = o.x * cos(p.theta) - o.y * sin(p.theta) + p.x;
      o.y = o.y * sin(p.theta) + o.y * cos(p.theta) - p.y;
    }

    // predict landmarks within sensor range of this particle
    vector<LandmarkObs> predicted;
    LandmarkObs lo;
    for (auto landmark : map_landmarks.landmark_list) {
      if (dist(p.x, p.y, landmark.x_f, landmark.y_f) < sensor_range) {
        lo = {landmark.id_i, landmark.x_f, landmark.y_f};
        predicted.push_back(lo);
      }
    }

    // find nearest neighbor - updates observations map with landmark id
    dataAssociation(predicted, observations_map);

    // predicted landmark lookup
    map<int, LandmarkObs> predicted_lookup;
    for (auto prediction : predicted) {
      predicted_lookup[prediction.id] = prediction;
    }

    // calculate multi variate gaussian weight
    double weight_product;
    bool first_measurement = true;
    for (auto measurement : observations_map) {
      // nearest neighbor observation to the predicted landmark
      LandmarkObs predicted_measurement = predicted_lookup[measurement.id];  // id points to landmark

      // multi-variate gaussian distribution calculation
      x << measurement.x, measurement.y;
      mu << predicted_measurement.x, predicted_measurement.y;

      double weight = exp(double(-0.5 * (x - mu).transpose() * inverseMeasurementCovar * (x - mu))) / c3;

      if (first_measurement) {
        weight_product = weight;
        first_measurement = false;
      } else {
        weight_product *= weight;
      }
    }

    p.weight = weight_product;
  }

}

// TODO remove later once the above code works
double ParticleFilter::multiVariateGaussianWeight(
    LandmarkObs predicted_measurement, LandmarkObs measurement,
    const MatrixXd& measurementCovar) {
  VectorXd x, mu;
  x << measurement.x, measurement.y;
  mu << predicted_measurement.x, predicted_measurement.y;

  double c1 = -0.5 * (x - mu).transpose() * measurementCovar.inverse()
      * (x - mu);
  MatrixXd c2 = 2 * M_PI * measurementCovar;
  double c3 = c2.determinant();
  return exp(c1) / sqrt(c3);
}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to their weight.
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  // calculate the sum of weights
//  double weights_sum = 0.0;
//  for (auto p : particles)
//    weights_sum += p.weight;
//
//  // normalize weights
//  weights.clear();
//  weights.resize(num_particles);
//  for (auto p : particles) {
//    weights.push_back(p.weight/weights_sum);
//  }


  // intialise resampling wheel
  double beta = 0.0;
  random_device rd;
  mt19937 genindex(rd());
  uniform_int_distribution<int> dis(0, num_particles);
  int index = dis(genindex);

  vector<Particle> resampled_particles;
  resampled_particles.resize(num_particles);


  // find max weight and create particle weights vectors
  double max_weight = 0.0;
  weights.clear();
  weights.resize(num_particles);
  for (auto particle: particles) {
    if (particle.weight > max_weight)
      max_weight = particle.weight;
    weights.push_back(particle.weight);
  }

  // create random number generator for two times max weight
  mt19937 gen(rd());
  uniform_real_distribution<double> dis_real(0, 1);

  // resample
  for (auto particle: particles) {
    beta += dis_real(gen) * 2.0 * max_weight;
    while (beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    resampled_particles.push_back(particles[index]);
  }

  // update particles
  particles = resampled_particles;

}

void ParticleFilter::write(std::string filename) {
  // You don't need to modify this file.
  std::ofstream dataFile;
  dataFile.open(filename, std::ios::app);
  for (int i = 0; i < num_particles; ++i) {
    dataFile << particles[i].x << " " << particles[i].y << " "
             << particles[i].theta << "\n";
  }
  dataFile.close();
}
