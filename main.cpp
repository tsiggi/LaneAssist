#include <iostream>
#include <chrono>
#include <string>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include "src/LaneDetection/detect.hpp"
#include "src/LaneKeeping/lanekeeping.hpp"

#define FRAME_WIDTH 510
#define FRAME_HEIGHT 280


int main()
{

    LaneDetectionResults ld_results;
    bool has_video = true;
    if (has_video)
    {
        std::string camera("455");
        std::string full_path_to_videos = "/mnt/c/Users/giann/Desktop/Work Experience/LaneAssist/"; // put your full path to repo
        std::string video_path = cv::String(full_path_to_videos + "video_repository/highway.mp4");
        cv::VideoCapture cap(video_path);

        // Check if the video file is opened successfully
        if (!cap.isOpened()) {
            std::cerr << ">>> Error: Could not open the video file. Check the path!!!" << std::endl;
            return -1;
        }
        int sf = 0, cnt = 0, skipped_frames = 3, angle;
        std::chrono::duration<double> elapsed;
        double sum = 0.0;
        cv::Mat image;
        bool ret;

        detect laneDetection(FRAME_WIDTH, FRAME_HEIGHT, camera);
        laneKeeping lane_keeping_obj(FRAME_WIDTH, FRAME_HEIGHT, camera);

        while (cap.isOpened())
        {
            // Skip frames 
            while(sf < skipped_frames)
            {
                ret = cap.read(image);
                sf++;
            }
            sf = 0;
            ret = cap.read(image);

            if (ret)
            {
                cnt++;  // image number

                auto start = std::chrono::high_resolution_clock::now();
                ld_results = laneDetection.lanes_detection(image);                
                auto finish = std::chrono::high_resolution_clock::now();

                double angle = lane_keeping_obj.lane_keeping(ld_results);
                // std::cout << "Angle " << angle << std::endl;

                elapsed = finish - start;
                sum += elapsed.count();

                cv::imshow("image", ld_results.frame);
                cv::waitKey(10);
            }
            else
            {
                break;
            }
        }
        if (cnt > 0){
            double av_time = sum / (1.0 * cnt);
            std::cout << std::endl
                      << "Average time: " << av_time << "s" << std::endl 
                      << "Average Fps:  " << round(1/av_time) << "fps" << std::endl;
        }
    }

    else
    {
        cv::Mat image;
        std::string image_path = cv::String("path/image_repository/Starting_point.png");
        // Check if the image is read successfully
        if (image.empty()) {
            std::cerr << ">>> Error: Could not open or read the image file. Check the path!!!" << std::endl;
            return -1;
        }
        image = cv::imread(image_path, 1);
        cv::resize(image, image, cv::Size(), 0.5, 0.5);
        std::string camera("405");

        detect laneDetection(image.cols, image.rows, camera);

        laneDetection.lanes_detection(image);
        imshow("ld", image);
        cv::waitKey(0);
    }

    return 0;
}
