//
// Created by tsiggi on 23/12/2023.
//
#pragma once

#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <iostream>
#include <string>
#include <vector>
#include <Eigen/Dense>

struct Polynomial {
    double a; // Coefficient of x^2 term
    double b; // Coefficient of x term
    double c; // Coefficient of constant term
    int degree;
    bool accepted;

    // Constructor to initialize the coefficients  
    Polynomial() : a(0.0), b(0.0), c(0.0), degree(0), accepted(false) {}

    void setCoefficients(const Eigen::VectorXd& coeffs){
        int size = coeffs.size();
        if (size == 2) {
            a = 0.0;
            b = coeffs[1];
            c = coeffs[0];
            degree = 1;
            accepted = true;
        } else if (size == 3) {
            a = coeffs[2];
            b = coeffs[1];
            c = coeffs[0];
            degree = 2;
            accepted = true;
        } else {
            std::cerr << "Invalid coefficient vector size!" << std::endl;
            a = b = c = 0.0;
            degree = 0;
            accepted = false;
        }
    }

    void copyPolynomial(const Polynomial * polynomial_){
        if(polynomial_!=nullptr){
            a = polynomial_->a;
            b = polynomial_->b;
            c = polynomial_->c;
            degree = polynomial_->degree;
            accepted = polynomial_->accepted;
        }
    }

    // Function to evaluate the polynomial for a given x
    double evaluate(double x) {
        return a * x * x + b * x + c;
    }

    
    void check_extreme_coeffs(double limit){
        if(accepted){
            if(degree == 1 && std::fabs(b) > limit){
                this->isNotAccepted();
            }else if (degree == 2 && std::fabs(a) > limit){
                this->isNotAccepted();
            }
        }
    }

    // Function to reset the coefficients and accepted flag
    void reset() {
        a = b = c = 0.0;
        degree = 0;
        accepted = false;
    }

    void isNotAccepted(){accepted = false;}

    // Function to display the polynomial
    void display() {
        std::cout << "Polynomial: " << a << "x^2 + " << b << "x + " << c << ", Degree: " << degree << " Accepted: " << accepted << "\n";
    }
};

struct ImagePoint {
    int width; // the witdh of an image point
    int height; // the height of the image point

    // Constructor to initialize a default point
    ImagePoint() : width(0), height(0) {}
    // Constructor to initialize a point with values
    ImagePoint(int width_, int height_) {
        this->width = width_;
        this->height = height_;
    } 
    
    void copyFrom(const ImagePoint * point_){
        if(point_ != nullptr){
            this->width = point_->width;
            this->height = point_->height;
        }
    }

    void setValues(int width_, int height_){
        this->width = width_;
        this->height = height_;
    }

    // Function to display the point
    void display(){
        std::cout<< "Point: (width:"<< width <<", height:"<< height <<")\n";
    }
    

};

struct ArrayOfImagePoints {
    ImagePoint** arrayOfPoints; // Array of Points (A LANE)
    int size;                   // Size of the Lane
    int length;                 // Stores the number of stored points
    bool accepted;

    ArrayOfImagePoints(int arraySize_){
        this->size = arraySize_;
        this->length = 0;
        this->accepted = false;
        this->arrayOfPoints = new ImagePoint* [sizeof(ImagePoint*) * arraySize_];
        for(int point_index=0; point_index < arraySize_; point_index++){
            this->arrayOfPoints[point_index] = new ImagePoint();
        }
    }

    ImagePoint* getPoint(int index){
        if(index >=0 && index < this->size){
            return this->arrayOfPoints[index];
        }
        return nullptr;
    }

    int getPointHeight(int index){
        return index >=0 && index < this->size ? this->arrayOfPoints[index]->height : -1; 
    }

    int getPointWidth(int index){
        return (index >=0 && index < this->size) ? this->arrayOfPoints[index]->width : -1;
    }

    ImagePoint* getFirstPoint(){
        if(length >0){
            return this->arrayOfPoints[0];
        }
        return nullptr;
    }

    int addPoint(int width, int height){
        if(length < size){
            arrayOfPoints[length]->setValues(width, height);
            length += 1;
        }
        return length;
    }

    void copyFrom(const ArrayOfImagePoints * lane_){
        if(lane_ != nullptr && this->size == lane_->size){
            // store the number of point data
            this->length = lane_->length;
            this->accepted = true;
            // copy data from input lane to this lane
            for(int pointIndex=0; pointIndex<this->length; pointIndex++){
                arrayOfPoints[pointIndex]->copyFrom(lane_->arrayOfPoints[pointIndex]);
            }
        }
    }

    void reset(){
        this->length = 0;
        this->accepted = false;
    }

    void deleteMemory(){
        for(int i=0; i < size; i++){
            delete arrayOfPoints[i];
        }
        delete[] this->arrayOfPoints;
    }
};

struct Lanes{
    ArrayOfImagePoints** lanes; // Array of Lanes (Stores all the Lanes)
    int size;                   // Size of Lanes Array
    int length;                 // Stores the number of stored lanes

    // Creates the Array of Lanes. Input : numberOfLanes, numberOfPointsInLane
    Lanes(int numberOfLanes, int numberOfPointsInLane){
        this->size = numberOfLanes;
        this->length = 0;
        this->lanes = new ArrayOfImagePoints* [sizeof(ArrayOfImagePoints*) * numberOfLanes];

        for(int lane_index=0; lane_index < numberOfLanes; lane_index++){
            this->lanes[lane_index] = new ArrayOfImagePoints(numberOfPointsInLane);
        }
    }

    ArrayOfImagePoints* getLane(int index){
        // return a lane
        if(index >=0 && index < this->length){
            return this->lanes[index];
        }
        return nullptr;
    }

    int getLanePointWidth(int laneIndex, int pointIndex){
        return (laneIndex >=0 && laneIndex < size) ? this->lanes[laneIndex]->getPointWidth(pointIndex) : -1;
    }

    int getLanePointHeight(int laneIndex, int pointIndex){
        return laneIndex >=0 && laneIndex < size ? this->lanes[laneIndex]->getPointHeight(pointIndex) : -1;
    }

    void addLanePoint(int laneIndex, int width, int height){
        // Add a point to a lane and returns the length of that lane
        // if it's 1 => we created a new lane (so length += 1)
        if(lanes[laneIndex]->addPoint(width, height) == 1){
            length++;
        }
    }

    void addPointToNewLane(int laneIndex, int width, int height){
        int point_index;

        if (this->length + 1 >= this->size){
            std::cout << std::endl
                << "Warning: Point can not be inserted to new lane. There is not a free lane." << std::endl
                << std::endl;
            return;
        }

        // append
        if (this->length == laneIndex){
            addLanePoint(this->length, width, height);
        }
        // insert
        else
        {
            ArrayOfImagePoints *insertion_lane = this->lanes[this->length];

            // Move/Slide all data one index to the right (create space for the insertion)
            for (int i = this->length; i > laneIndex; i--)
                this->lanes[i] = this->lanes[i - 1];
   
            // (Insert Lane) Add Point to the laneIndex
            this->lanes[laneIndex] = insertion_lane;            
            addLanePoint(laneIndex, width, height);
        }
    }

    int getLaneLength(int laneIndex){
            return lanes[laneIndex]->length;
        }

    void reset(){
        // reset data of each lane
        for(int point_index=0; point_index < this->length; point_index++){
            this->lanes[point_index]->reset();
        }
        this->length = 0;
    }

    void deleteMemory(){
        for(int point_index=0; point_index < size; point_index++){
            this->lanes[point_index]->deleteMemory(); 
        }
        delete[] this->lanes;
        
    }
};

struct HorizontalLine{
    double slope, intercept;
    int num_of_slopes;
    ImagePoint basePoint;
    ImagePoint startPoint, endPoint;
    bool exist;

    HorizontalLine(){
      reset();
    } 
    
    HorizontalLine(double slope_, double intercept_, int num_of_slopes_, ImagePoint basePoint_){
        slope = slope_;
        intercept = intercept_;
        num_of_slopes = num_of_slopes_;
        basePoint.copyFrom(&basePoint_);
    }

    void copyFrom(HorizontalLine *hor){
        if(hor!=nullptr){
            slope = hor->slope;
            intercept = hor->intercept;
            num_of_slopes = hor->num_of_slopes;
            basePoint.copyFrom(& hor->basePoint);
            startPoint.copyFrom(& hor->startPoint);
            endPoint.copyFrom(& hor->endPoint);
            exist = hor->exist;
        }
    }

    void setValues(double slope_, double intercept_, int num_of_slopes_,  ImagePoint point){
        slope = slope_;
        intercept = intercept_;
        num_of_slopes = num_of_slopes_;
        basePoint.setValues(point.width, point.height);
    }

    void updateLine(double slope_, double intercept_){
        slope = slope_;
        intercept = intercept_;
    }

    void setEndPoints(ImagePoint start, ImagePoint end){
        startPoint.copyFrom(& start);
        endPoint.copyFrom(& end);
    }

    double evaluate(double x) {
        return slope * x + intercept;
    }

    void reset(){
        slope = 0;
        intercept = 0;
        num_of_slopes = 0;
        basePoint.setValues(0,0);
        startPoint.setValues(0,0);
        endPoint.setValues(0,0);
        exist = false;
    }

    void displayLine(){
        std::cout << "slope: " << slope << ", intercept: " << intercept << "\n";
    }

};

struct LaneDetectionResults{
    cv::Mat frame;                                  // frame 
    ArrayOfImagePoints *left, *right;               // left and right lane
    Polynomial *left_poly, *right_poly;             // Polynomial of the 2 lanes
    HorizontalLine horizontal;                      // Data of Horizontal line
    double left_perc, right_perc;                   // lane certainty percentage
    bool trust_left, trust_right, trust_lk;         // trusted lanes and laneKeeping
    void print_lane_detection_result(){
        std::cout << "LD Result: "<< std::endl;
        std::cout << "Left coef: ";
        if(left_poly->accepted){
            left_poly->display();
        }else
            std::cout << "NULL"<<std::endl;
        
        std::cout << "right coef: ";
        if(right_poly->accepted){
            right_poly->display();
        }else
            std::cout << "NULL"<<std::endl;   
    }
};

class detect
{    
public:
    ~detect();
    detect(int width, int height, std::string camera);
    void choose_405();
    void choose_455();
    LaneDetectionResults lanes_detection(cv::Mat &);
    LaneDetectionResults lanes_detection_sliding_window(cv::Mat &img);
        
private:

    // HORIZONTAL DETECTION
    void horizontal_detection(cv::Mat &);
    HorizontalLine* detect_main_point(cv::Mat &);
    ImagePoint* search_for_near_point(cv::Mat &, cv::Mat &, ImagePoint &, std::string , HorizontalLine*, double);
    void detect_lane_line_endpoints(cv::Mat &, HorizontalLine*);
    void visualize_horizontal_line(cv::Mat &, HorizontalLine);

    // void peaks_detection_sliding_window(cv::Mat &);
    // void detect::vertical_search(cv::Mat &, cv::Mat &, int, int[2]);


    // Order of the FUNCTIONS called in lanes_detection
    void peaks_detection(cv::Mat &);
        int find_lane_peaks(const int*,double, int) const;
        void peaks_clustering(int, int);
            bool find_best_qualified_points(int[], double[], bool[], int[], int, int);
                bool verify_with_expected_value(int, int, int, double&);
                void add_qualified_point(int[], double[], bool[], int[], int, int, double);
        
    void choose_correct_lanes();
        void update_lane_data(int**, int***, int, int&);
        void calculate_lane_boundaries(int, double&, double&);

    void create_polyfits_from_peaks(cv::Mat &);
        void fit_polyfit(Polynomial*, ArrayOfImagePoints* , float);
            Eigen::VectorXd fit_polynomial_with_degree(const Eigen::VectorXd&, const Eigen::VectorXd&, int);
        void visualize_lane(Polynomial*, cv::Mat &, cv::Scalar);


    void lanes_post_processing(cv::Mat &, double);
        double find_lane_certainty(Polynomial*, Polynomial*, ArrayOfImagePoints*);
        void visualize_lane_certainty(cv::Mat &, double, double);
        void check_lane_certainties(cv::Mat &, double);
        bool trust_lane_detection(double, double);

    void visualize_all_peaks(cv::Mat &);
    void visualize_lane_peaks(cv::Mat &);

    LaneDetectionResults get_results(cv::Mat &);

    void printlanes();
    // PARAMS

    bool already_initialized;
    int img_width;
    int img_height;
    std::string camera;
    bool print_lanes;
    bool print_peaks;
    bool print_lane_certainty;
    bool print_if_dashed;
    bool print_horizontal;
    int slices;
    int bottom_offset;
    double bottom_perc_455;
    double bottom_perc_405;
    int peaks_min_width_455;
    int peaks_min_width_405;
    int peaks_max_width_455;
    int peaks_max_width_405;
    double max_allowed_dist;
    double weight_for_width_distance;
    double weight_for_expected_value_distance;
    int min_peaks_for_lane;
    double optimal_peak_perc;
    double min_lane_dist_perc;
    double max_lane_dist_perc;
    double percentage_for_first_degree;
    double allowed_certainty_perc_dif;
    double certainty_perc_from_peaks;
    double extreme_coef_second_deg;
    int extreme_coef_first_deg;
    int min_single_lane_certainty;
    int min_dual_lane_certainty;
    int square_pulses_min_height;
    int square_pulses_pix_dif;
    int square_pulses_min_height_dif;
    int square_pulses_allowed_peaks_width_error;
    int hor_peaks_max_width;
    int hor_peaks_min_width;
    double hor_perc_from_mid;
    int hor_square_pulses_allowed_peaks_width_error;
    double dashed_max_dash_points_perc;
    double dashed_min_dash_points_perc;
    double dashed_min_space_points_perc;
    int dashed_min_count_of_dashed_lanes;
    // From Lane Keeping
    int bottom_width_455;
    int bottom_width_405;
    int top_width_455;
    int top_width_405;

    double bottom_perc;
    int peaks_min_width;
    int peaks_max_width;
    int bottom_row_index;
    int step;
    int real_slices;
    int top_row_index;

    int top_width, bottom_width;

    double min_top_width_dif, max_top_width_dif, min_bot_width_dif, max_bot_width_dif;

    int min_count_of_dashed_lanes;
    double max_points, min_points, min_points_space;

    int max_peaks_per_hist, max_different_lanes;

    bool prev_trust_lk;


    // ALL POINTERS
    Lanes* lanes;
    ArrayOfImagePoints* peaks, *left_lane, *right_lane;
    Polynomial *left_lane_poly, *right_lane_poly, *prev_left_lane_poly, *prev_right_lane_poly;

    int *histogram, *histogram_peaks;
    double *height_norm;

    // RESULTS
    double left_certainty_perc, right_certainty_perc;
    bool trust_left_lane, trust_right_lane, trust_laneDetection;
    
    // HORIZONTAL
    float hor_max_allowed_slope;
    HorizontalLine horizontal;
};

