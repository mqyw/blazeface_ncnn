#pragma once

#include<stdio.h>
#include<string>
#include "ncnn/net.h"
#include "plate_color_recog.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

struct Object
{
    cv::Rect_<float> rect;
    int label;
    std::vector<cv::Point2f> landmarks;
    float prob;
};

class palteDet
{
public:
    palteDet();
    palteDet(std::string param_path, std::string bin_path, int w, int h);
    ~palteDet();
    void init(std::string param_path, std::string bin_path, int w, int h);
    int detect(const cv::Mat& bgr, std::vector<Object>& objects);
    void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects, int labelflag);

private:
    int target_size_w = 0;
    int target_size_h = 0;
    ncnn::Net mobilenet;
};

class faceDet
{
public:
    faceDet();
    faceDet(std::string param_path, std::string bin_path, int w, int h);
    ~faceDet();
    void init(std::string param_path, std::string bin_path, int w, int h);
    int detect(const cv::Mat& bgr, std::vector<Object>& objects);

private:
    int target_size_w = 0;
    int target_size_h = 0;
    float anchors[1000][4];
    int x_scale = 128;
    int y_scale = 128;
    int w_scale = 128;
    int h_scale = 128;
    ncnn::Net blazeface;
};