//
// Created by tsiggi on 23/12/2023.
//
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include "detect.hpp"
#include <iostream>
#include <vector>
#include "helpers.hpp"
#include <eigen/Eigen/Dense>

detect::~detect(){
    std::cout << ">>> Deconstructor called..." << std::endl;

    delete[] this->histogram;
    delete[] this->histogram_peaks;

    int max_peaks_on_one_image = int(this->max_peaks_per_hist * this->slices);

    this->peaks->deleteMemory();
    delete this->peaks;

    this->lanes->deleteMemory();
    delete this->lanes;
    this->left_lane->deleteMemory();
    delete this->left_lane;
    this->right_lane->deleteMemory();
    delete this->right_lane;

    delete[] this->height_norm;

    delete this->left_lane_poly;
    delete this->right_lane_poly;
    delete this->prev_left_lane_poly;
    delete this->prev_right_lane_poly;
}

detect::detect(int width, int height, std::string camera){
    std::cout << ">>> LaneDetection Constructor called..." << std::endl;

    // PARAMS
    this->img_width = width;
    this->img_height = height;
    this->camera = camera;
    this->print_lanes = true;
    this->print_peaks = true;
    this->print_lane_certainty = true;
    this->print_if_dashed = true;
    this->print_horizontal = true;
    this->slices = 20;
    this->bottom_offset = 10;
    this->bottom_perc_455 = 0.5;
    this->bottom_perc_405 = 0.45;
    this->peaks_min_width_455 = 3;
    this->peaks_min_width_405 = 4;
    this->peaks_max_width_455 = 20;
    this->peaks_max_width_405 = 30;
    float max_allowed_width_perc = 0.15;
    this->max_allowed_dist = max_allowed_width_perc * this->img_width;
    this->weight_for_width_distance = 0.3;
    this->weight_for_expected_value_distance = 0.7;
    this->min_peaks_for_lane = 3;
    this->optimal_peak_perc = 0.3;
    this->min_lane_dist_perc = 0.5;
    this->max_lane_dist_perc = 1.5;
    this->percentage_for_first_degree = 0.4;
    this->allowed_certainty_perc_dif = 20.0;
    this->certainty_perc_from_peaks = 0.5;
    this->extreme_coef_second_deg = 0.1;
    this->extreme_coef_first_deg = 3;
    this->min_single_lane_certainty = 80;
    this->min_dual_lane_certainty = 60;
    this->square_pulses_min_height = 170;
    this->square_pulses_pix_dif = 2;
    this->square_pulses_min_height_dif = 60;
    this->square_pulses_allowed_peaks_width_error = 10;
    this->hor_peaks_max_width = 8;
    this->hor_peaks_min_width = 3;
    this->hor_perc_from_mid = 0.15;
    this->hor_square_pulses_allowed_peaks_width_error = 10;
    this->dashed_max_dash_points_perc = 0.4;
    this->dashed_min_dash_points_perc = 0.15;
    this->dashed_min_space_points_perc = 0.1;
    this->dashed_min_count_of_dashed_lanes = 4;
    // PARAMS From Lane Keeping
    this->bottom_width_455 = 190;
    this->bottom_width_405 = 400;
    this->top_width_455 = 85;
    this->top_width_405 = 250;

    this->max_peaks_per_hist = 10;
    this->max_different_lanes = 20;

    // HELPING PARAMS (Store the data of the algorithms)
    this->already_initialized = false;

    int size_of_histogram = sizeof(int) * this->img_width;
    this->histogram = new int[size_of_histogram];

    int max_peaks_on_one_histogram = sizeof(int) * this->max_peaks_per_hist;
    this->histogram_peaks = new int[max_peaks_on_one_histogram];


    int size_of_lane = this->slices;
    int max_peaks_on_one_image = int(this->max_peaks_per_hist * size_of_lane);
    this->peaks = new ArrayOfImagePoints(max_peaks_on_one_image);
    // this->peaks_size = 0;

    this->lanes = new Lanes(this->max_different_lanes, size_of_lane);

    this->left_lane = new ArrayOfImagePoints(size_of_lane);
    this->right_lane = new ArrayOfImagePoints(size_of_lane);

    this->left_lane_poly = new Polynomial;
    this->right_lane_poly = new Polynomial;
    this->prev_left_lane_poly = new Polynomial;
    this->prev_right_lane_poly = new Polynomial;

    this->prev_trust_lk = false;

    // Horizontal Params 
    this->hor_max_allowed_slope = 0.25;
    this->horizontal = HorizontalLine();

    // TODO: CHANGE 405 and 455
    if (this->camera == "405")
        detect::choose_405();
    else if (this->camera == "455")
        detect::choose_455();
    else
        throw std::runtime_error(">>> LANEDETECTION ERROR on Initialization :\n\t Can't choose a camera. No correct camera selected!");
}

void detect::choose_405(){
    this->bottom_perc = this->bottom_perc_405;
    this->peaks_min_width = this->peaks_min_width_405;
    this->peaks_max_width = this->peaks_max_width_405;

    this->bottom_row_index = this->img_height - this->bottom_offset;
    int end = int(float(this->img_height) * (1 - this->bottom_perc));
    this->step = int(-(float(this->img_height) * this->bottom_perc / float(this->slices)));
    this->real_slices = int((end - this->bottom_row_index) / this->step);
    this->top_row_index = this->bottom_row_index + this->real_slices * this->step;

    if (this->already_initialized)
        delete[] this->height_norm;
    this->height_norm = linspace(0, 1, this->real_slices + 1);

    this->bottom_width = this->bottom_width_405;
    this->top_width = this->top_width_405;

    this->min_top_width_dif = float(this->top_width) * this->min_lane_dist_perc;
    this->max_top_width_dif = float(this->top_width) * this->max_lane_dist_perc;
    this->min_bot_width_dif = float(this->bottom_width) * this->min_lane_dist_perc;
    this->max_bot_width_dif = float(this->bottom_width) * this->max_lane_dist_perc;

    this->min_count_of_dashed_lanes = this->dashed_min_count_of_dashed_lanes;

    // Checks if len of dashed lanes and len of spaces are between the following margins
    this->max_points = this->dashed_max_dash_points_perc * float(this->real_slices);
    this->min_points = this->dashed_min_dash_points_perc * float(this->real_slices);
    this->min_points_space = this->dashed_min_space_points_perc * float(this->real_slices);

    this->already_initialized = true;
    
    std::cout << "405 camera selected successfully" << std::endl;
}

void detect::choose_455(){
    this->bottom_perc = this->bottom_perc_455;
    this->peaks_min_width = this->peaks_min_width_455;
    this->peaks_max_width = this->peaks_max_width_455;

    this->bottom_row_index = this->img_height - this->bottom_offset;
    int end = int(float(this->img_height) * (1 - this->bottom_perc));
    this->step = int(-(float(this->img_height) * this->bottom_perc / float(this->slices)));
    this->real_slices = int((end - this->bottom_row_index) / this->step);
    this->top_row_index = this->bottom_row_index + this->real_slices * this->step;

    if (this->already_initialized)
        delete[] this->height_norm;
    this->height_norm = linspace(0, 1, this->real_slices + 1);

    this->bottom_width = this->bottom_width_455;
    this->top_width = this->top_width_455;

    this->min_top_width_dif = float(this->top_width) * this->min_lane_dist_perc;
    this->max_top_width_dif = float(this->top_width) * this->max_lane_dist_perc;
    this->min_bot_width_dif = float(this->bottom_width) * this->min_lane_dist_perc;
    this->max_bot_width_dif = float(this->bottom_width) * this->max_lane_dist_perc;

    this->min_count_of_dashed_lanes = this->dashed_min_count_of_dashed_lanes;

    // Checks if len of dashed lanes and len of spaces are between the following margins
    this->max_points = this->dashed_max_dash_points_perc * float(this->real_slices);
    this->min_points = this->dashed_min_dash_points_perc * float(this->real_slices);
    this->min_points_space = this->dashed_min_space_points_perc * float(this->real_slices);

    this->already_initialized = true;

    std::cout << "455 camera selected successfully" << std::endl;
}

LaneDetectionResults detect::lanes_detection(cv::Mat &img){
    this->left_lane->reset();
    this->right_lane->reset();
    this->lanes->reset();
    this->peaks->reset();
    this->horizontal.reset();

    detect::peaks_detection(img);
    
    detect::horizontal_detection(img);
    // detect::printlanes();

    detect::choose_correct_lanes();

    detect::create_polyfits_from_peaks(img);

    detect::lanes_post_processing(img, this->allowed_certainty_perc_dif);

    if (this->print_peaks)
    {
        detect::visualize_all_peaks(img);
        detect::visualize_lane_peaks(img);
    }
    if (this->print_horizontal)
        detect::visualize_horizontal_line(img, this->horizontal);
    LaneDetectionResults result = detect::get_results(img);
    // result.print_lane_detection_result();


    return result;
}

void detect::printlanes(){
    int lastLaneIndex = this->lanes->length;
    for (int i = 0; i < lastLaneIndex; i++)
    {
        ArrayOfImagePoints *lane = this->lanes->getLane(i);

        std::cout << i << ". [";
        for (int p = 0; p < lane->length; p++)
        {
            std::cout << "[" << lane->getPointWidth(p) << "," << lane->getPointHeight(p) << "], ";
        }
        std::cout << "]" << std::endl;
    }
}

// -------------------------------------------------------------------------------- //
// -------------------------------- Main functions -------------------------------- //
// -------------------------------------------------------------------------------- //

void detect::peaks_detection(cv::Mat &src){
    /**
     *
     *
     *
     */
    cv::Mat img;
    cv::cvtColor(src, img, cv::COLOR_BGR2GRAY);

    int cnt = 0, width, height;
    int sum = 0;
    for (int height = this->bottom_row_index; height > this->top_row_index; height += this->step)
    {
        // initialize histogram
        const unsigned char *row = img.ptr<unsigned char>(height);
        
        for (int i = 0; i < this->img_width; i++){
            this->histogram[i] = int(row[i]);
        }
        double norm_height = this->height_norm[cnt++];

        int ps_size = detect::find_lane_peaks(this->histogram, norm_height, this->img_width);
        sum += ps_size;

        for (int i = 0; i < ps_size; i++)
        {            
            width = this->histogram_peaks[i];
            // update peaks. lastIndex is Updated automatically
            this->peaks->addPoint(width, height);
        }

        // add detected peaks to lanes
        detect::peaks_clustering(ps_size, height);

    }
}

void detect::choose_correct_lanes(){

    // initialize the indexes to -1
    int left_lane_index = -1, right_lane_index = -1;

    ImagePoint* point;

    int index, lane_length, point_index;
    int allowed_length = this->real_slices * this->optimal_peak_perc;

    // Dont use lanes with few points
    int lanes_size = this->lanes->length;
    int allowed_lane_indexes[lanes_size];
    int array_size = 0;
    for (int lane_index = 0; lane_index < lanes_size; lane_index++)
        if (this->lanes->getLaneLength(lane_index) >= this->min_peaks_for_lane)
        {
            allowed_lane_indexes[array_size++] = lane_index;
        }

    // Choose left and right lanes
    for (int i = 0; i < array_size; i++)
    {
        // Lanes are stores from left to right (Checking from left to right)
        index = allowed_lane_indexes[i];
        lane_length = this->lanes->getLaneLength(index);

        // For the left side
        point = this->lanes->getLane(index)->getFirstPoint();
        if (point->width <= this->img_width / 2)
        { // if first point of the lane is in the left side => lane is from the left side
            // if lane index -1 initialize it
            if (left_lane_index == -1)
            {
                left_lane_index = index;
                continue;
            }
            // (left lane = the most right lane of the left side with most points (similar for the right lane)
            if (lane_length > allowed_length || lane_length > this->lanes->getLaneLength(left_lane_index))
                left_lane_index = index;
        }
        // For the right side
        else
        {
            if (right_lane_index == -1)
            {
                right_lane_index = index;
                // if the leftest lane (of the right side) has more peaks than this WE FOUND OUR RIGHT LANE
                if (lane_length > allowed_length)
                    break;
                continue;
            }
            // here we dont want to add (or length > len(right)) cause right lane is the most left lane of the right side
            if (lane_length > allowed_length)
            {
                right_lane_index = index;
                break;
            }
        }
    }

    // Lane becomes accepted if it's copied successfully   
    this->left_lane->copyFrom(this->lanes->getLane(left_lane_index));
    this->right_lane->copyFrom(this->lanes->getLane(right_lane_index));

    // Check if these 2 lanes are correct (else keep only one lane)
    if (this->left_lane->accepted and this->right_lane->accepted)
    {
        double left_top, left_bot, right_top, right_bot;
        detect::calculate_lane_boundaries(left_lane_index, left_top, left_bot);
        detect::calculate_lane_boundaries(right_lane_index, right_top, right_bot);

        int top_dif = (right_top - left_top) / 2;
        int bot_dif = (right_bot - left_bot) / 2;

        // if lanes width difference is small/big delete a lane
        if (!((top_dif > this->min_top_width_dif) && (top_dif < this->max_top_width_dif) && (bot_dif > this->min_bot_width_dif) && (bot_dif < this->max_bot_width_dif)))
        {
            // keep lane with most peaks
            if (this->left_lane->length > this->right_lane->length)
            {
                this->right_lane->reset();
                right_lane_index = -1;
            }
            else
            {
                this->left_lane->reset();
                left_lane_index = -1;
            }
        }
    }
}

void detect::create_polyfits_from_peaks(cv::Mat &frame){

    detect::fit_polyfit(this->left_lane_poly, this->left_lane, this->percentage_for_first_degree);

    detect::fit_polyfit(this->right_lane_poly, this->right_lane, this->percentage_for_first_degree);

    if (this->print_lanes)
    {
        detect::visualize_lane(this->left_lane_poly, frame, cv::Scalar(255, 128, 0));
        // detect::visualize_lane(this->left_lane_poly, frame, cv::Scalar(0, 128, 255));
        detect::visualize_lane(this->right_lane_poly, frame, cv::Scalar(0, 128, 255));
    }
}

void detect::lanes_post_processing(cv::Mat &frame, double allowed_difference){
    // std::cout << "Left certainty from coeffs: " ;
    this->left_certainty_perc = detect::find_lane_certainty(this->left_lane_poly, this->prev_left_lane_poly, this->left_lane);
    // std::cout << "Right certainty from coeffs: " ;
    this->right_certainty_perc = detect::find_lane_certainty(this->right_lane_poly, this->prev_right_lane_poly, this->right_lane);

    if (this->print_lane_certainty)
    {
        detect::visualize_lane_certainty(frame, this->left_certainty_perc, this->right_certainty_perc);
    }

    this->prev_left_lane_poly->copyPolynomial(this->left_lane_poly);
    this->prev_right_lane_poly->copyPolynomial(this->right_lane_poly);

    detect::check_lane_certainties(frame, this->allowed_certainty_perc_dif);

    this->trust_laneDetection = detect::trust_lane_detection(this->left_certainty_perc, this->right_certainty_perc);
}

LaneDetectionResults detect::get_results(cv::Mat &img){
    LaneDetectionResults result;
    result.frame = img;
    result.left = this->left_lane;
    result.right = this->right_lane;
    result.left_poly = this->left_lane_poly;
    result.right_poly = this->right_lane_poly;
    result.horizontal = this->horizontal;
    result.left_perc = this->left_certainty_perc;
    result.right_perc = this->right_certainty_perc;
    result.trust_left = this->trust_left_lane;
    result.trust_right = this->trust_right_lane;
    result.trust_lk = this->trust_laneDetection;
    return result;
}

// -------------------------------------------------------------------------------- //
// ---------------------------------- HORIZONTAL ---------------------------------- //
// -------------------------------------------------------------------------------- //

double calcSlope(ImagePoint* point1, ImagePoint* point2){
    return (point1->height - point2->height)*1.0 / (point1->width - point2->width);
}

void detect::horizontal_detection(cv::Mat &frame){

    // Finds the main point and the line of the horizontal and stores them in horLine
    HorizontalLine* horTmp = detect::detect_main_point(frame);

    // Finds the start and the end of the horLine and stores them there.
    // if slope > thresshold it deletes horTmp and makes it nullptr
    detect::detect_lane_line_endpoints(frame, horTmp);
    
    this->horizontal.copyFrom(horTmp);
    delete horTmp;
}

HorizontalLine* detect::detect_main_point(cv::Mat &frame){
    cv::Mat gray_frame;
    cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);

    int width, num_of_slopes;
    double slope, intercept;

    int iterations = 3;
    float shearch_in_these_widths_perc[iterations] = {0.5, 0.45, 0.55};

    float top_height_perc = 0.4;
    float bot_height_perc = 0.95;
    int height_start = int(top_height_perc * this->img_height);
    int height_end = int(bot_height_perc * this->img_height);
    
    int size_of_histogram = sizeof(int) * (height_end - height_start);
    int* hor_histogram = new int[size_of_histogram];

    ImagePoint* main_point = nullptr;
    HorizontalLine* line = nullptr;

    for(int i=0; i<iterations; i++){
        width = int(shearch_in_these_widths_perc[i] * this->img_width);
                
        // initialize histogram        
        for (int j = height_start; j < height_end; j++){
            hor_histogram[j-height_start] = (int)gray_frame.at<unsigned char>(j,width);
        }

        int ps_size = detect::find_lane_peaks(hor_histogram, 0, height_end-height_start);

        // cv::line(frame, cv::Point(width, height_start), cv::Point(width, height_end), cv::Scalar(60,20,220));

        for (int j = 0; j < ps_size; j++)
        {            
            
            // cv::Point tmp = cv::Point(width, histogram_peaks[j] + height_start);
            // cv::circle(frame, tmp, 2, cv::Scalar(0,0,0),2);

            ImagePoint basePoint = ImagePoint(width, histogram_peaks[j] + height_start);
            ImagePoint* pointL = detect::search_for_near_point(frame, gray_frame, basePoint, std::string("left"), nullptr, 0.015);
            ImagePoint* pointR = detect::search_for_near_point(frame, gray_frame, basePoint, std::string("right"), nullptr, 0.015);
            
            if(pointL!=nullptr and pointR!=nullptr){
                double slopeL = calcSlope(&basePoint, pointL);
                double slopeR = calcSlope(&basePoint, pointR);
                slope = (slopeR+slopeL)/2;
                num_of_slopes = 2;
            } else 
            if (pointR!=nullptr){
                slope = calcSlope(&basePoint, pointR);
            }else 
            if (pointL!=nullptr){
                slope = calcSlope(&basePoint, pointL);
            } else 
                continue;
            
            if(std::abs(slope) > this->hor_max_allowed_slope)
                continue;

            bool bigger_num_of_slopes = line!=nullptr and num_of_slopes > line->num_of_slopes;
            bool same_num_of_slopes_but_smaller_slope = line!=nullptr and num_of_slopes == line->num_of_slopes and std::abs(line->slope) > std::abs(slope);

            if(line==nullptr or bigger_num_of_slopes or same_num_of_slopes_but_smaller_slope){
                intercept = basePoint.height - slope * basePoint.width;

                line = new HorizontalLine(slope, intercept, num_of_slopes, basePoint);
            }
            delete pointL, pointR;
        }

        if(line!=nullptr)
            return line;
    }
    return nullptr;
}

ImagePoint* detect::search_for_near_point(cv::Mat &frame, cv::Mat &grayFrame, ImagePoint &basePoint, std::string diraction, HorizontalLine* line, double width_step_perc){

    float max_allowed_height_dif_from_line = 0.03 * this->img_height;
    int operation = (diraction == "right") ? 1 : -1 ;

    int width_step = int(width_step_perc * this->img_width);
    int sliding_window_height = int(0.15 * this->img_height);

    int width = int(basePoint.width + operation * width_step);
    int height = (line == nullptr) ? basePoint.height : line->evaluate(width);

    int start = height - sliding_window_height/2;
    int end = height + sliding_window_height/2;

    int size_of_histogram = sizeof(int) * (end - start);
    int* hor_sw_histogram = new int[size_of_histogram];


    if((width < 0 or width >= this->img_width) or (start < 0 or end >= this->img_height))
        return nullptr;

    // initialize histogram        
    for (int j = start; j < end; j++){
        hor_sw_histogram[j-start] = (int)grayFrame.at<unsigned char>(j,width);
    }
    

    int ps_size = detect::find_lane_peaks(hor_sw_histogram, 0, end-start);

    // cv::line(frame, cv::Point(width, start), cv::Point(width, end), cv::Scalar(60,20,220));

    if(ps_size<=0){
        return nullptr;
    }
    ImagePoint* point = nullptr;
    if(ps_size==1){
        
        ImagePoint tmpPoint(width, (this->histogram_peaks[0] + start));
        
        double slope_with_base = std::abs(calcSlope(&basePoint, &tmpPoint));
        bool is_height_accepted = (line==nullptr) ? true : std::abs(this->histogram_peaks[0]+start - height) <= round(max_allowed_height_dif_from_line);

        if(slope_with_base<= this->hor_max_allowed_slope and is_height_accepted){
            point = new ImagePoint(tmpPoint);
            // cv::circle(frame, cv::Point(width, histogram_peaks[0] + start), 2, cv::Scalar(0,0,0),2);
        }
        // else{
        //     cv::circle(frame, cv::Point(width, histogram_peaks[0] + start), 2, cv::Scalar(0,250,0),2);
        // }
    }   

    return point;
}

void detect::detect_lane_line_endpoints(cv::Mat &frame, HorizontalLine* horLine){
    
    if(horLine == nullptr)
        return;
    
    cv::Mat grayFrame;
    cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
    
    int hor_min_width_dist = round(0.2 * this->img_width);

    ImagePoint leftPoint(horLine->basePoint), rightPoint(horLine->basePoint);
    ImagePoint *pointL, *pointR, prevRight, prevLeft;
    double slope, intercept;

    bool newLeftPoint = true, newRightPoint = true;
    while (newLeftPoint or newRightPoint)
    {
        // Continue Searching if there was a previous detection
        if(newLeftPoint){
            pointL = detect::search_for_near_point(frame, grayFrame, leftPoint, std::string("left"), horLine, 0.03);
            // Update basePoint and cnt (copy fails if pointL is nullptr)
            leftPoint.copyFrom(pointL);
            if(pointL == nullptr)
                newLeftPoint = false;
        }

        if(newRightPoint){
            pointR = detect::search_for_near_point(frame, grayFrame, rightPoint, std::string("right"), horLine, 0.03);
            if(pointR == nullptr)
                newRightPoint = false;
            rightPoint.copyFrom(pointR);
        }
    }
    
    // Return nullptr if the slope of the line is less than the thresshold
    if(std::abs(slope) > this->hor_max_allowed_slope){
        delete horLine;
        horLine = nullptr;
        return;
    }

    horLine->setEndPoints(leftPoint, rightPoint);
    horLine->exist = (horLine!=nullptr) and ((horLine->endPoint.width - horLine->startPoint.width) >= hor_min_width_dist);
}


// -------------------------------------------------------------------------------- //
// -------------------------- peaks_detection() HELPERS --------------------------- //
// -------------------------------------------------------------------------------- //

int detect::find_lane_peaks(const int *histogram, double norm_height, int hist_size) const{

    double temp_peaks_max_width = this->peaks_max_width - (this->peaks_max_width - this->peaks_min_width) * norm_height;
    double upper_limit = temp_peaks_max_width + this->square_pulses_allowed_peaks_width_error;

    bool inside_a_peak = false;
    int pix_num = 0;
    int height_dif_start = 0;
    int height_dif_end = 0;
    int peak;

    int detected_peaks_cnt = 0;

    int pix_dif = this->square_pulses_pix_dif;
    // search for peaks
    for (int i = pix_dif; i < hist_size - pix_dif; i++)
    {

        int pixel = histogram[i];

        height_dif_start = pixel - histogram[i - pix_dif];
        height_dif_end = pixel - histogram[i + pix_dif];

        if (inside_a_peak)
        {
            // peak finished
            if (height_dif_end > this->square_pulses_min_height_dif)
            {
                inside_a_peak = false;
                // check if it's a lane peak
                if ((pix_num >= this->peaks_min_width) && (pix_num <= upper_limit))
                {
                    peak = i - (pix_num - pix_dif) / 2;
                    this->histogram_peaks[detected_peaks_cnt++] = peak;
                    // if(detected_peaks_cnt < this->max_ps_size){

                    // } else {
                    //     throw runtime_error("ERROR on Find_peaks function :\n\t Memory is not enough!! Detected peaks more than 10 !!!");
                    // }
                }
                pix_num = 0;
            }

            // still inside
            else
            {
                if (pixel > this->square_pulses_min_height)
                    pix_num++;
                else
                    inside_a_peak = false;
            }
        }
        // not inside a peak
        else
        {
            // go inside
            if ((pixel > this->square_pulses_min_height) && (height_dif_start > this->square_pulses_min_height_dif))
            {
                inside_a_peak = true;
                pix_num++;
            }
            // else stay outside
        }
    }
    // returns the number of detected peaks in this histogram
    return detected_peaks_cnt;
}

void detect::peaks_clustering(int ps_size, int height){
    /**
     * ...
     * if lanes array is empty it initializes them and exits.
     * ...
     */
    int width, point_index;

    if (not this->lanes->length)
    {
        for (int i = 0; i < ps_size; i++)
        {
            width = histogram_peaks[i];
            this->lanes->addLanePoint(i, width, height);
        }
        return;
    }

    // For each lane we store the index of the best qualified point (int) and the distance (float). Initialization: index=-1, distance=-1.
    int lanes_point_index[this->lanes->length];
    double lanes_distances[this->lanes->length];
    for (int lane_index = 0; lane_index < this->lanes->length; lane_index++)
    {
        lanes_point_index[lane_index] = -1;
        lanes_distances[lane_index] = -1.0;
    }

    // For each point we store respectively if it's a qualified point for a lane (bool) and the index of this lane (int). Initialization: used=false, index=-1.
    bool points_used[ps_size];
    int points_lane_index[ps_size];
    for (int point_index = 0; point_index < ps_size; point_index++)
    {
        points_used[point_index] = false;
        points_lane_index[point_index] = -1;
    }

    // 1) Finds the best matchups between the detected points and the lanes.
    bool run_again = detect::find_best_qualified_points(lanes_point_index, lanes_distances, points_used, points_lane_index, ps_size, height);
    // if some data were corrupted on the first run => run again
    if (run_again)
        run_again = detect::find_best_qualified_points(lanes_point_index, lanes_distances, points_used, points_lane_index, ps_size, height);

    // 2) Appends the points that have a matchup into the corresponding lane.
    // After finding the best (unique) point for each lane (if there is), append it and delete it from the list
    int cnt = 0, l_index, insert_cnt = 0;
    int points_to_be_inserted[ps_size];
    for (int i = 0; i < ps_size; i++)
    {
        if (points_used[i])
        {
            l_index = points_lane_index[i];
            width = this->histogram_peaks[cnt];
            this->lanes->addLanePoint(l_index, width, height);
        }
        else
        {
            points_to_be_inserted[insert_cnt++] = this->histogram_peaks[cnt];
        }
        cnt++;
    }

    // 3) Insert the remaining points (that do not have matchup) [new lanes].
    int lanes_inserted = 0, lanes_index = 0, lane_point_width, lane_point2;
    int size_of_points_to_be_inserted = insert_cnt;

    for (int i = 0; i < size_of_points_to_be_inserted; i++)
    {
        width = points_to_be_inserted[i];
        // insert at the beginning
        // Python : if point < lanes[lanes_inserted][-1][0] :
        point_index = this->lanes->getLaneLength(lanes_inserted) - 1;
        lane_point_width = this->lanes->getLanePointWidth(lanes_inserted, point_index);
        if (width < lane_point_width)
        {
            this->lanes->addPointToNewLane(0, width, height);
            lanes_inserted++;
            lanes_index = lanes_inserted;
        }else {
            l_index = this->lanes->length - 1;

            point_index = this->lanes->getLaneLength(l_index) - 1;
            lane_point_width = this->lanes->getLanePointWidth(l_index, point_index);

            // insert at the end
            if (width > lane_point_width)
            {
                this->lanes->addPointToNewLane(this->lanes->length, width, height);
                lanes_inserted++;
            }

            // insert somewhere in the middle
            else
            {
                // find where we should insert it
                for (int j = lanes_index; j < this->lanes->length + lanes_inserted - 1; j++)
                {
                    l_index = j;
                    point_index = this->lanes->getLaneLength(l_index) -1;
                    lane_point_width = this->lanes->getLanePointWidth(l_index, point_index);
                    l_index = j + 1;
                    point_index = this->lanes->getLaneLength(l_index) -1;
                    lane_point2 = this->lanes->getLanePointWidth(l_index, point_index);
                    if (width > lane_point_width && width < lane_point2)
                    {
                        this->lanes->addPointToNewLane(j+1, width, height);
                        lanes_inserted++;
                        lanes_index = j + 1;
                        break;
                    }
                }
            }
        }
    }
}

bool detect::find_best_qualified_points(int lanes_point_index[], double lanes_dist[], bool points_used[], int points_lane_index[], int ps_size, int height){
    
    // TODO : Create a stack for the points => No need for run_again param, Function should only be called once.
    // if a point is more qualified, add the other point in the top of the stack. Do this until there is no points left.    
    
    
    bool run_again = false, at_least_one, flag;
    int x0, y0, x1, y1, prev_lane_index, prev_point_index;
    double width_dist, width_dist_error, temp_dist;
    int point_index;

    for (int lane_index = 0; lane_index < this->lanes->length; lane_index++)
    {
        at_least_one = false;

        // if we already found a point for this lane => means that this function is running for the second time => no need to check for this lane
        if (lanes_point_index[lane_index] != -1)
            continue;

        // for every lane find the best qualified point
        for (int peak_index = 0; peak_index < ps_size; peak_index++)
        {

            x0 = this->histogram_peaks[peak_index];
            y0 = height;
            // PYTHON: x1, y1 = lanes[lane_index][-1][0], lanes[lane_index][-1][1]
            point_index = this->lanes->getLaneLength(lane_index) - 1;
            x1 = this->lanes->getLanePointWidth(lane_index, point_index);
            y1 = this->lanes->getLanePointHeight(lane_index, point_index);

            width_dist = abs((x0 - x1) * this->step * 1.0 / (y0 - y1));

            // Check only for near points
            if (width_dist < this->max_allowed_dist)
            {

                flag = detect::verify_with_expected_value(lane_index, height, x0, width_dist_error);

                if (flag)
                {
                    at_least_one = true;
                    temp_dist = this->weight_for_width_distance * width_dist + this->weight_for_expected_value_distance * width_dist_error;

                    // if point not used by any lane and lane has not been initialized with best qualified point
                    if ((!points_used[peak_index]) && lanes_point_index[lane_index] == -1)
                        detect::add_qualified_point(lanes_point_index, lanes_dist, points_used, points_lane_index, lane_index, peak_index, temp_dist);

                    // if point used by other lane and lane has not a qualified point (1 point 2 lane) => Check which lane is the best fit for this lane
                    else if (points_used[peak_index] && lanes_point_index[lane_index] == -1)
                    {
                        // get prev lane
                        prev_lane_index = points_lane_index[peak_index];

                        // if distance from prev lane is greater than this lane
                        if (lanes_dist[prev_lane_index] > temp_dist)
                        {
                            run_again = true;

                            // delete point from prev lane
                            lanes_point_index[prev_lane_index] = -1;
                            lanes_dist[prev_lane_index] = -1;
                            // For this lane : initialize with the point, For point : update only the used lane (it's already used)
                            detect::add_qualified_point(lanes_point_index, lanes_dist, points_used, points_lane_index, lane_index, peak_index, temp_dist);
                        }
                    }

                    // if point not used by any lane and lane has already a qualified point (2 points 1 lane) => Check which point is the best
                    else if ((!points_used[peak_index]) && lanes_point_index[lane_index] != -1)
                    {

                        // get prev point
                        prev_point_index = lanes_point_index[lane_index];

                        // for the same lane, if the already qualified point has greater distance than this point => New point is the best
                        if (lanes_dist[lane_index] > temp_dist)
                        {
                            run_again = true;

                            // delete data from prev point (not used and lane_index)
                            points_used[prev_point_index] = false;
                            points_lane_index[prev_point_index] = -1;
                            // Update lane data (new point_index and new distance) and Initialize data for new point
                            detect::add_qualified_point(lanes_point_index, lanes_dist, points_used, points_lane_index, lane_index, peak_index, temp_dist);
                        }
                    }

                    // if point is used by another lane and lane has already another best fit (2 points 2 lanes) => Check if the new combination is better
                    else if (points_used[peak_index] && lanes_point_index[lane_index] != -1)
                    {
                        // 2 cases :
                        // 1) this combination is already stored (exit)
                        // 2) else (check)
                        if (points_lane_index[peak_index] == lane_index && lanes_point_index[lane_index] == peak_index)
                            continue;

                        prev_lane_index = points_lane_index[peak_index];
                        prev_point_index = lanes_point_index[lane_index];

                        // if new distance is smaller than the 2 others (delete data from prev point, delete data from prev lane,update data for this point and this lane)
                        if (lanes_dist[lane_index] > temp_dist && lanes_dist[prev_lane_index] > temp_dist)
                        {
                            run_again = true;

                            // delete data from prev point (not used and lane_index)
                            points_used[prev_point_index] = false;
                            points_lane_index[prev_point_index] = -1;
                            // delete point from prev lane
                            lanes_point_index[prev_lane_index] = -1;
                            lanes_dist[prev_lane_index] = -1;
                            // Update lane data (new point_index and new distance) and update data for new point
                            detect::add_qualified_point(lanes_point_index, lanes_dist, points_used, points_lane_index, lane_index, peak_index, temp_dist);
                        }
                    }
                }
            }
            else if (at_least_one)
                break;
        }
    }

    // Check if we realy need to run again
    if (run_again)
    {
        bool all_points = true;
        for (int i = 0; i < ps_size; i++)
        {
            if (!points_used[i])
            {
                all_points = false;
                break;
            }
        }

        bool all_lanes = true;
        for (int i = 0; i < this->lanes->length; i++)
        {
            if (lanes_point_index[i] == -1)
            {
                all_lanes = false;
                break;
            }
        }
        if (all_lanes || all_points)
            run_again = false;
    }

    return run_again;
}

bool detect::verify_with_expected_value(int lane_index, int height, int x_value, double &dist){

    int point_index;
    bool flag;
    // create a lane from the first and last lane points, then calculate the expected peak (when y=height)

    // We will calculate the expected peak = x0.
    point_index = this->lanes->getLaneLength(lane_index) - 1;
    int x1 = this->lanes->getLanePointWidth(lane_index, point_index);
    int y1 = this->lanes->getLanePointHeight(lane_index, point_index);

    point_index = 0;
    int x0 = this->lanes->getLanePointWidth(lane_index, point_index);
    int y0 = this->lanes->getLanePointHeight(lane_index, point_index);

    double x_dif = x1 - x0;
    double y_dif = y1 - y0;
    double x_div_y = y_dif != 0 ? (x_dif / y_dif) : 0;

    // x/y = (x0 - x1) / (y0 - y1)               we solve for x0 = expected_peak
    // x/y = x_div_y , x0 = EXPECTED,  x1 = lane[-1][0], y0 = height, y1 = lane[-1][1]
    double expected_peak = (height - y1) * x_div_y + x1;
    dist = abs(x_value - expected_peak);

    // punishment if we only have one point
    double punish = x_div_y == 0 ? this->max_allowed_dist / 2 : 0;
    // punishment depending on the count of the slices that preceded from the last point of the lane
    double perc_of_preceded_slices = ((height - y1) / this->step - 1) / this->real_slices;
    punish += perc_of_preceded_slices * this->max_allowed_dist;

    double max_allowed_dist = x_div_y == 0 ? this->max_allowed_dist : this->max_allowed_dist / 2;

    if (this->lanes->getLaneLength(lane_index) > 3)
    {
        // create a lane from the last 2 lane points and calculate the expected peak.
        point_index = this->lanes->getLaneLength(lane_index) - 2;
        int x2 = this->lanes->getLanePointWidth(lane_index, point_index);
        int y2 = this->lanes->getLanePointHeight(lane_index, point_index);

        x_dif = x1 - x2;
        y_dif = y1 - y2;
        x_div_y = y_dif != 0 ? (x_dif / y_dif) : 0;

        expected_peak = (height - y1) * x_div_y + x1;
        double temp = abs(x_value - expected_peak);
        dist = temp < dist ? temp : dist;

        // same for the second-to-last and third-to-last points.
        point_index = this->lanes->getLaneLength(lane_index) - 3;
        int x3 = this->lanes->getLanePointWidth(lane_index, point_index);
        int y3 = this->lanes->getLanePointHeight(lane_index, point_index);

        x_dif = x2 - x3;
        y_dif = y2 - y3;
        x_div_y = y_dif != 0 ? (x_dif / y_dif) : 0;

        expected_peak = (height - y2) * x_div_y + x2;
        temp = abs(x_value - expected_peak);
        dist = temp < dist ? temp : dist;
    }

    flag = dist < max_allowed_dist ? true : false;
    dist += punish;

    return flag;
}

void detect::add_qualified_point(int lanes_point_index[], double lanes_dist[], bool points_used[], int points_lane_index[], int lane_index, int point_index, double distance){
    lanes_point_index[lane_index] = point_index;
    lanes_dist[lane_index] = distance;
    points_used[point_index] = true;
    points_lane_index[point_index] = lane_index;
}

// -------------------------------------------------------------------------------- //
// ------------------------ choose_corect_lanes HELPERS() ------------------------- //
// -------------------------------------------------------------------------------- //


void detect::calculate_lane_boundaries(int lane_index, double &out_top, double &out_bot){
    int point_index, x0, x1, y0, y1;
    point_index = 0;
    x0 = this->lanes->getLanePointWidth(lane_index, point_index);
    y0 = this->lanes->getLanePointHeight(lane_index, point_index);
    point_index = this->lanes->getLaneLength(lane_index) - 1;
    x1 = this->lanes->getLanePointWidth(lane_index, point_index);
    y1 = this->lanes->getLanePointHeight(lane_index, point_index);

    // x = Slope * y + b
    // c = Width, y = Height
    double slope = (x1 - x0) * 1.0 / (y1 - y0);

    double b = x0 - slope * y0;

    out_top = slope * this->top_row_index + b;
    out_bot = slope * this->bottom_row_index + b;
}

// -------------------------------------------------------------------------------- //
// --------------------- create_polyfits_from_peaks() HELPERS --------------------- //
// -------------------------------------------------------------------------------- //

void detect::fit_polyfit(Polynomial *poly, ArrayOfImagePoints *lane, float percentage_for_first_degree){
    Eigen::VectorXd laneCoef;

    bool lane_exist = lane && lane->length > 0;
    if (lane_exist)
    {

        Eigen::VectorXd x(lane->length);
        Eigen::VectorXd y(lane->length);

        for (int point = 0; point < lane->length; point++)
        {
            // REVERTED because we want to create a polyfit with independent variable the height (Y) and dependent the width (X)
            // So the output looks like that f(y) = x. This way, in the same height we can only have one width.
            // The problem that occurs if we try f(x) = y is that in the same width we can have multiples heights.
            x(point) = lane->getPointHeight(point);
            y(point) = lane->getPointWidth(point);
        }
        // TODO: CHECK IF WE NEED THE check_for_extreme_coefs FUNCTION
        double max_allowed_size_for_first_degree = this->real_slices * percentage_for_first_degree;
        if (lane->length > max_allowed_size_for_first_degree)
        {
            laneCoef = detect::fit_polynomial_with_degree(x, y, 2);
            poly->setCoefficients(laneCoef);
            poly->check_extreme_coeffs(this->extreme_coef_second_deg);
        }
        else
        {
            laneCoef = detect::fit_polynomial_with_degree(x, y, 1);
            poly->setCoefficients(laneCoef);
            poly->check_extreme_coeffs(this->extreme_coef_first_deg);
        }
    }
    else
    {
        poly->reset();
    }
}

Eigen::VectorXd detect::fit_polynomial_with_degree(const Eigen::VectorXd &x, const Eigen::VectorXd &y, int degree){

    int numPoints = x.size();

    Eigen::MatrixXd A(numPoints, degree + 1);
    Eigen::VectorXd b(numPoints);

    for (int i = 0; i < numPoints; ++i)
    {
        for (int j = 0; j <= degree; ++j)
        {
            A(i, j) = std::pow(x(i), j);
        }
        b(i) = y(i);
    }

    Eigen::VectorXd coef = A.colPivHouseholderQr().solve(b);

    return coef;
}

// -------------------------------------------------------------------------------- //
// ----------------------- lanes_post_processing() HELPERS ------------------------ //
// -------------------------------------------------------------------------------- //

double detect::find_lane_certainty(Polynomial *new_poly, Polynomial *prev_poly, ArrayOfImagePoints* lane){

    if (!new_poly->accepted || !prev_poly->accepted){
        double certainty = 0.0;
        return certainty;
    }

    int point_index;

    // Similarity
    double error;
    error = (pow(new_poly->a - prev_poly->a, 4) + pow((new_poly->b - prev_poly->b), 2) + abs(new_poly->c - prev_poly->c)) / 3;
    error = sqrt(error);
    double similarity = 100 - error;
    // std::cout << similarity << std::endl;
    if (similarity < 0)
        similarity = 0;
    // Lane Peaks
    // 1)
    // double peaks_percentage = len(peaks) / self.slices * 100;
    // 2)
    point_index = lane->length - 1;
    int y0 = lane->getPointHeight(point_index);

    point_index = 0;
    int y1 = lane->getPointHeight(point_index);

    double peaks_percentage = ((y0 - y1) * 1.0 / (this->top_row_index - this->bottom_row_index)) * 100;

    double certainty = (this->certainty_perc_from_peaks * peaks_percentage + (1 - this->certainty_perc_from_peaks) * similarity);

    return certainty;
}

void detect::check_lane_certainties(cv::Mat &frame, double allowed_difference){
    this->trust_left_lane = true, this->trust_right_lane = true;

    if (abs(this->left_certainty_perc - this->right_certainty_perc) > allowed_difference)
    {
        if (this->left_certainty_perc > this->right_certainty_perc)
        {
            if (this->print_lanes)
                detect::visualize_lane(this->right_lane_poly, frame, cv::Scalar(169, 169, 169));
            this->trust_left_lane = true;
            this->trust_right_lane = false;
        }
        else
        {
            if (this->print_lanes)
                detect::visualize_lane(this->left_lane_poly, frame, cv::Scalar(169, 169, 169));
            this->trust_left_lane = false;
            this->trust_right_lane = true;
        }
    }
}

bool detect::trust_lane_detection(double l_certainty, double r_certainty){
    // Is lane keeping trusted?
    bool single_flag = (l_certainty > this->min_single_lane_certainty || r_certainty > this->min_single_lane_certainty);
    bool dual_flag = (l_certainty > this->min_dual_lane_certainty && r_certainty > this->min_dual_lane_certainty);
    bool lane_keeping_ok = single_flag || dual_flag;

    // Return this
    bool result = this->prev_trust_lk && lane_keeping_ok;
    // Update previous trust
    this->prev_trust_lk = lane_keeping_ok ? true : false;

    return result;
}

// -------------------------------------------------------------------------------- //
// -------------------------------- Visualizations -------------------------------- //
// -------------------------------------------------------------------------------- //

void detect::visualize_lane(Polynomial *lane, cv::Mat &frame, cv::Scalar bgr_colour){

    if (! lane->accepted){
        return;
    }

    int k, end = (1 - this->bottom_perc) * this->img_height;
    cv::Point point_start, point_finish;
    for (int i = this->img_height; i > end; i -= 5){

        point_start.x = lane->evaluate(i);
        point_start.y = i;
        k = i - 5;
        point_finish.x = lane->evaluate(k);
        point_finish.y = k;
        cv::line(frame, point_start, point_finish, bgr_colour, 3);
    }
}

void detect::visualize_lane_certainty(cv::Mat &frame, double l_perc, double r_perc){
    cv::Point point;
    point.x = int(0.2 * this->img_width);
    point.y = int(this->img_height / 2.5);
    cv::putText(frame, std::to_string(l_perc), point, cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 128, 0), 1);
    // cv::putText(frame, std::to_string(l_perc), point, cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 128, 255), 1);

    point.x = int(0.7 * this->img_width);
    cv::putText(frame, std::to_string(r_perc), point, cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 128, 255), 1);
}

void detect::visualize_all_peaks(cv::Mat &img){
    cv::Point point;
    for (int point_index = 0; point_index < this->peaks->length; point_index++)
    {
        point.x = this->peaks->getPointWidth(point_index);
        point.y = this->peaks->getPointHeight(point_index);
        cv::circle(img, point, 2, cv::Scalar(255, 0, 255), 2);
    }
}

void detect::visualize_lane_peaks(cv::Mat &img){

    cv::Point point;
    if (this->left_lane->accepted)
    {
        for (int point_index = 0; point_index < this->left_lane->length; point_index++)
        {
            point.x = this->left_lane->getPointWidth(point_index);
            point.y = this->left_lane->getPointHeight(point_index);
            cv::circle(img, point, 2, cv::Scalar(0, 0, 0), 2);
        }
    }
    if (this->right_lane->accepted)
    {
        for (int point_index = 0; point_index < this->right_lane->length; point_index++)
        {
            point.x = this->right_lane->getPointWidth(point_index);
            point.y = this->right_lane->getPointHeight(point_index);
            cv::circle(img, point, 2, cv::Scalar(0, 0, 0), 2);
        }
    }
}

void detect::visualize_horizontal_line(cv::Mat &frame, HorizontalLine hor){
    if(hor.exist){
        cv::Point start(hor.startPoint.width, hor.startPoint.height), end(hor.endPoint.width, hor.endPoint.height);
        cv::line(frame, start, end, cv::Scalar(226,43,138), 10);
        
        // start.y = int(hor.slope * start.x + hor.intercept);
        // end.y = int(hor.slope * end.x + hor.intercept);

        // cv::line(frame, start, end, cv::Scalar(150,150,150), 3);

    }
}
