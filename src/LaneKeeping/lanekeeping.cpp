//
// Created by giannishorgos on 04/01/2024.
//
#include <opencv2/opencv.hpp>
#include <stdio.h>
// #include "../laneDetection/helpers.hpp"
#include "lanekeeping.hpp"
#include <iostream>
#include <vector>
#include <eigen/Eigen/Dense>


laneKeeping::laneKeeping(int width, int height, std::string &camera){
    std::cout << ">>> LaneKeeping Constructor called..." << std::endl;

    this->img_width = width;
    this->img_height = height;
    this->camera = camera;

    // LK
    this->median_constant = 2;
    this->print_desire_lane = false;
    this->bottom_width_455 = 190;
    this->bottom_width_405 = 145;
    this->top_width_455 = 85;
    this->top_width_455 = 80;
    this->max_coef_of_sharp_turn = 0.01;
    this->min_coef_of_sharp_turn = 0.0001;
    this->sharp_turning_factor = 2;
    this->mu = 0.5;
    this->sigma = 0.4;
    this->min_smoothing_factor = 0.2;
    this->max_smoothing_factor = 0.8;

    // PARAMS
    this->max_lk_steer = 23;

    
    // LD 
    this->slices = 20;
    this->bottom_offset = 10;
    this->bottom_perc_455 = 0.5;
    this->bottom_perc_405 = 0.45;
    this->peaks_min_width_455 = 3;
    this->peaks_max_width_405 = 2;
    this->peaks_max_width_455 = 20;
    this->peaks_max_width_405 = 15;

    // HELP
    this->already_initialized = false;
    this->angle = 0.0;
    this->last_angle = 0.;
   
    this->left_lane = new Polynomial;
    this->right_lane = new Polynomial;
    this->desire_lane_ptr = nullptr;
    
    if (this->camera == "405")
        laneKeeping::choose_405();
    else if (this->camera == "455")
        laneKeeping::choose_455();
    else
        throw std::runtime_error(">>> LANEKEEPING ERROR on Initialization :\n\t Can't choose a camera. No correct camera selected!");

    // std::cout << "Real Slices: " << this->real_slices << ", Slices: " << this->slices << std::endl;
}

void laneKeeping::choose_405(){
    std::cout << "405 camera selected successfully" << std::endl;

    this->bottom_perc = this->bottom_perc_405;
    this->peaks_min_width = this->peaks_min_width_405;
    this->peaks_max_width = this->peaks_max_width_405;

    this->bottom_row_index = this->img_height - this->bottom_offset;
    int end = int(float(this->img_height) * (1 - this->bottom_perc));
    this->step = int(-(float(this->img_height) * this->bottom_perc / float(this->slices)));
    this->real_slices = int((end - this->bottom_row_index) / this->step);
    this->top_row_index = this->bottom_row_index + this->real_slices * this->step;

    if (this->already_initialized){
        delete[] this->plot_y;
        delete[] this->plot_y_norm;
        delete[] this->desire_lane;
    }
    // this->plot_y = linspace(this->bottom_row_index, this->top_row_index, this->real_slices + 1);
    this->plot_y = linspace(this->bottom_row_index, this->top_row_index, this->real_slices);
    this->plot_y_norm = linspace(0,1,this->real_slices);
    this->desire_lane = new int[this->real_slices];

    this->bottom_width = this->bottom_width_405;
    this->top_width = this->top_width_405;

    this->already_initialized = true;
}

void laneKeeping::choose_455(){
    std::cout << "455 camera selected successfully" << std::endl;

    this->bottom_perc = this->bottom_perc_455;
    this->peaks_min_width = this->peaks_min_width_455;
    this->peaks_max_width = this->peaks_max_width_455;

    this->bottom_row_index = this->img_height - this->bottom_offset;
    int end = int(float(this->img_height) * (1 - this->bottom_perc));
    this->step = int(-(float(this->img_height) * this->bottom_perc / float(this->slices)));
    this->real_slices = int((end - this->bottom_row_index) / this->step);
    this->top_row_index = this->bottom_row_index + this->real_slices * this->step;

    if (this->already_initialized){
        delete[] this->plot_y;
        delete[] this->plot_y_norm;
        delete[] this->desire_lane;
    }
    // this->plot_y = linspace(this->bottom_row_index, this->top_row_index, this->real_slices + 1);
    this->plot_y = linspace(this->bottom_row_index, this->top_row_index, this->real_slices);
    this->plot_y_norm = linspace(0, 1, this->real_slices);
    this->desire_lane = new int[this->real_slices];

    this->bottom_width = this->bottom_width_455;
    this->top_width = this->top_width_455;

    this->already_initialized = true;
}

laneKeeping::~laneKeeping(){
    std::cout << ">>> LK Destructor called..." << std::endl;
    delete [] this->plot_y;
    delete[] this->plot_y_norm;
    delete[] this->desire_lane;
}

double laneKeeping::lane_keeping(LaneDetectionResults results){

    // results.print_lane_detection_result();
    cv::Mat frame = results.frame;
    this->left_lane->copyPolynomial(results.left_poly);
    this->right_lane->copyPolynomial(results.right_poly);

    // Set lanes as accepted if they are trusted
    this->left_lane->accepted = results.trust_left;
    this->right_lane->accepted = results.trust_right;
        
    this->angle = laneKeeping::pid_controller(this->left_lane, this->right_lane, frame);
    this->last_angle = this->angle;

    // std::cout << ">>> Angle: " << this->angle <<std::endl;
    return this->angle;
}

double laneKeeping::pid_controller(Polynomial* left, Polynomial* right, cv::Mat &frame){
    double angle;

    if(not left->accepted and not right->accepted){
        angle = this->last_angle;
        std::cout<< ">>> (LK WARNING!!!) NO LANES FOUND"<< std::endl;
    }else{
        this->desire_lane_ptr = laneKeeping::calculate_desire_lane(left, right);
        if(this->print_desire_lane){
            laneKeeping::visualize_desire_lane(frame, this->desire_lane);
        }
        
        double error = laneKeeping::get_error(this->desire_lane_ptr);
        
        angle = 90 - (180 / M_PI) * atan2(this->img_height, error);

    }
    return angle;
}

int* laneKeeping::calculate_desire_lane(Polynomial* left, Polynomial* right){
    if(left->accepted and right->accepted){
        // BOTH LANES FOUND
        for(int i=0; i< this->real_slices; i++){
            this->desire_lane[i] = (
             (left->a + right->a)/2 * this->plot_y[i] * this->plot_y[i] +
             (left->b + right->b)/2 * this->plot_y[i] + 
             (left->c + right->c)/2
             );
        }
    }else if(not left->accepted){
        // Create the desired lane depending only on the right lane , the bottom_width and top_width params
        
        int top_width = this->top_width;
        int fix;

        if(right->a < -(this->max_smoothing_factor)){
            top_width = int(top_width * this->sharp_turning_factor);
        }else if(right->a < -(this->min_coef_of_sharp_turn)){
                int add = (-right->a - this->min_coef_of_sharp_turn) / (
                        this->max_coef_of_sharp_turn - this->min_coef_of_sharp_turn);
                top_width = int(top_width * (1 + add * (this->sharp_turning_factor - 1)));
        }

        for(int i=0; i< this->real_slices; i++){
            fix = this->bottom_width - ((this->bottom_width - top_width)* this->plot_y_norm[i]);
            this->desire_lane[i] = right->evaluate(plot_y[i]) - fix;
        }

    }else{
        int top_width = this->top_width;
        int fix;

        if(left->a > this->max_smoothing_factor){
            top_width = int(top_width * this->sharp_turning_factor);
        }else if(left->a > this->min_coef_of_sharp_turn){
            int add = (left->a - this->min_coef_of_sharp_turn) / (
                    this->max_coef_of_sharp_turn - this->min_coef_of_sharp_turn);
            top_width = int(top_width * (1 + add * (this->sharp_turning_factor - 1)));
        }


        for(int i=0; i< this->real_slices; i++){            
            fix = this->bottom_width - ((this->bottom_width - top_width)* this->plot_y_norm[i]);
            this->desire_lane[i] = left->evaluate(plot_y[i]) + fix;
        }
    }

    return this->desire_lane;
}

double laneKeeping::get_error(int* desired){
    if(desired!=nullptr){
        double mean = get_mean(desired);

        // The actual lane is the middle lane
        int center = this->img_width/2;

        double error = mean - center;
        return error;
    }
    return 0;
}

double laneKeeping::get_mean(int* desire){
    double sum=0;
    for(int i=0; i< this->real_slices; i++){
        sum += desire[i];
    }
    return sum/this->real_slices;
}

void laneKeeping::visualize_desire_lane(cv::Mat &frame, int* plot_x){
    if(this->desire_lane_ptr == nullptr)
        return;

    cv::Point point_1, point_2;
    cv::Scalar colour = cv::Scalar(50, 205 ,50);

    for(int i=1; i<this->real_slices; i++){
        point_1.x = plot_x[i];
        point_1.y = plot_y[i];
        point_2.x = plot_x[i-1];
        point_2.y = plot_y[i-1];

        cv::line(frame, point_1, point_2, colour, 3);
    }
    cv::Point p1(int(this->img_width/2), this->img_height);
    point_1.x = int(this->img_width/2);
    cv::line(frame, p1, point_1, cv::Scalar(50,50,205), 3);
}

double* laneKeeping::linspace(double start, double end, int numPoints) {
    if (numPoints <= 1) {
        double* result = new double[1];
        result[0] = start;
        return result;
    }
    
    double* result = new double[numPoints];
    double step = (end - start) / (numPoints - 1);
    for (int i = 0; i < numPoints; ++i) {
        result[i] = start + i * step;
    }
    
    return result;
}

double* laneKeeping::copy_coeffs(double* coef_dest, double* coef){
    /**
     * Coef ptr = null if coef == null
     * else coef_data
    */
    if(coef != nullptr){
        coef_dest[0] = coef[0];
        coef_dest[1] = coef[1];
        coef_dest[2] = coef[2];
        return coef_dest;
    }else{
        return nullptr;
    }
}
