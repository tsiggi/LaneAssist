//
// Created by giannishorgos on 04/01/2024.
//

#pragma once

#include <opencv2/opencv.hpp>
#include "../LaneDetection/detect.hpp"
#include <stdio.h>
#include <iostream>
#include <string>
#include <vector>
#include <Eigen/Dense>


class laneKeeping{
public:
    laneKeeping(int width, int height, std::string &camera);
    ~laneKeeping();
    double lane_keeping(LaneDetectionResults results);

private:
    void choose_405();
    void choose_455();
    double pid_controller(Polynomial* left, Polynomial* right, cv::Mat &frame);
    int* calculate_desire_lane(Polynomial* left, Polynomial* right);
    double get_error(int* desire_lane);
    double get_mean(int* desire_lane);
    void visualize_desire_lane(cv::Mat &frame, int* desire_lane);
    double* copy_coeffs(double* , double* );
    double* linspace(double start, double end, int numPoints);


    int img_width;
    int img_height;
    std::string camera;

    // steer_value_list ??
    int median_constant;
    bool print_desire_lane ;
    int bottom_width_455 ;
    int bottom_width_405; 
    int top_width_455 ;
    int top_width_405 ;
    float max_coef_of_sharp_turn ;
    float min_coef_of_sharp_turn; 
    int sharp_turning_factor ;
    float mu ;
    float sigma ;
    float min_smoothing_factor ;
    float max_smoothing_factor ;

    float max_lk_steer;


    int slices;
    int bottom_offset;

    float bottom_perc;
    float bottom_perc_455;
    float bottom_perc_405;

    int peaks_min_width ;
    int peaks_min_width_455;
    int peaks_min_width_405;

    int peaks_max_width ;
    int peaks_max_width_455 ;
    int peaks_max_width_405 ;


    // HELP
    float angle;
    float last_angle;
    int bottom_row_index;
    int step;
    int real_slices;
    int top_row_index;
    int bottom_width;
    int top_width;
    
    bool already_initialized;

    int *desire_lane, *desire_lane_ptr;
    double *plot_y;
    double *plot_y_norm;

    Polynomial *left_lane, *right_lane;
};
