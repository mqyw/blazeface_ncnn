#include<iostream>
#include <fstream>
#include<stdio.h>
#include<string>
#include "ncnn/net.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "face_net.h"


float sigmoid(float x)
{
    return (1 / (1 + exp(-x)));
}
std::vector<std::vector<float>> sort(std::vector<std::vector<float>>& _list)
{
    std::vector<std::vector<float>> list(_list);
    int list_num = list.size();

    for (size_t i = 0; i < list_num; i++)
    {
        int val_num = list[i].size();
        float conf_max = list[i][val_num - 1];
        int index = i;
        for (size_t j = i + 1; j < list_num; j++)
        {
            val_num = list[j].size();
            float conf_val = list[j][val_num - 1];
            if (conf_val > conf_max) {
                conf_max = conf_val;
                index = j;
            }
        }
        if (index != i) {
            std::swap(list[i], list[index]);
        }
    }
  
    list.swap(_list);
    return _list;
}

float overlap_similarity(std::vector<float> r1, std::vector<float>r2)
{
    float xmin1 = r1[1];
    float ymin1 = r1[0];
    float xmax1 = r1[3];
    float ymax1 = r1[2];

    float w1 = xmax1 - xmin1;
    float h1 = ymax1 - ymin1;

    float xmin2 = r2[1];
    float ymin2 = r2[0];
    float xmax2 = r2[3];
    float ymax2 = r2[2];

    float w2 = xmax2 - xmin2;
    float h2 = ymax2 - ymin2;

    float overlapW = std::min(xmax1, xmax2) - std::max(xmin1, xmin2);
    float overlapH = std::min(ymax1, ymax2) - std::max(ymin1, ymin2);

    return (overlapW * overlapH) / ((w1 * h1) + (w2 * h2) - (overlapW * overlapH));

}

inline void op_divide(float& a, float b) { a = a / b; }
inline void op_ride(float &a, float b) { a = a * b; }
inline float op_add(float a, float b) { return a + b; }
std::vector<std::vector<float>> WeightedNonMaxSuppression(std::vector<std::vector<float>> _list)
{
    std::vector<std::vector<float>> res;
    std::vector<std::vector<float>> list(_list);
    int list_num = list.size();
    for (size_t i = 0; i < list_num; i++)
    {
        float conf = list[i][ list[i].size() - 1 ];
        if (conf < 0){ continue; }

        std::vector<std::vector<float>> temp_face;
        temp_face.push_back(_list[i]);
        for (size_t j = i + 1; j < list_num; j++)
        {
            if (list[j][list[j].size() - 1] < 0) { continue; }
            float iou_val = overlap_similarity(list[i], list[j]);
            if (iou_val > 0.3) 
            {
                list[j][list[j].size() - 1] = -1;
                temp_face.push_back(_list[j]);
            }
        }
  
        if (temp_face.size() > 0) 
        {
            for (size_t j = 0; j < temp_face.size(); j++)
                for (size_t k = 0; k < temp_face[j].size() - 1; k++)
                    op_ride(temp_face[j][k], temp_face[j][temp_face[j].size() - 1]);

            std::vector<float> temp_total_val(temp_face[0]);
            for (size_t j = 1; j < temp_face.size(); j++) 
                std::transform(temp_face[j].begin(), temp_face[j].end(), temp_total_val.begin(), temp_total_val.begin(), op_add);

            for (size_t j = 0; j < temp_total_val.size() - 1; j++)
                op_divide(temp_total_val[j], temp_total_val[temp_total_val.size() - 1]);


            temp_total_val[temp_total_val.size() - 1] /= temp_face.size();
            res.push_back(temp_total_val);
        }
    }
    return res;
}

faceDet::faceDet()
{

}
faceDet::faceDet(std::string param_path, std::string bin_path, int w, int h)
{
    init(param_path, bin_path, w, h);
    std::string line;
    std::ifstream in("data/anchor.txt");
    if (in) 
    {
        int ln = 0;
        while (getline(in, line))
        {
            std::vector<std::string> res;
            std::string result;
            std::stringstream input;
            input << line;
            while (input >> result)
                res.push_back(result);
            for (int j = 0; j < res.size(); j++) {
                anchors[ln][j] = std::stof(res[j]);
            }
            ln++;
        }
    }
}
faceDet::~faceDet()
{
}

void faceDet::init(std::string param_path, std::string bin_path, int w, int h)
{
    target_size_w = w;
    target_size_h = h;
    blazeface.load_param(param_path.c_str());
    blazeface.load_model(bin_path.c_str());
}

int faceDet::detect(const cv::Mat& bgr, std::vector<Object>& objects)
{

    int img_w = bgr.cols;
    int img_h = bgr.rows;
   
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, target_size_w, target_size_h);

    const float mean_vals[3] = { 127.5f, 127.5f, 127.5f };
    const float norm_vals[3] = { 1.0 / 127.5, 1.0 / 127.5, 1.0 / 127.5 };
    in.substract_mean_normalize(mean_vals, norm_vals);
    
    ncnn::Extractor ex = blazeface.create_extractor();
    ex.set_num_threads(1);
    ex.input("x.1", in);
    
    ncnn::Mat out1;
    ex.extract("180", out1);
    ncnn::Mat out2;
    ex.extract("199", out2);

    //--------------获取定位

    std::vector< std::vector< float > > det_list;
    for (size_t h = 0; h < out1.h; h++)
    {
        for (size_t w = 0; w < out1.w; w++)
        {
            if (sigmoid(out1[h * out1.w + w]) > 0.65) {
                std::vector<float> val;
                
                float x_center = out2[h * out2.w + 0] / x_scale * anchors[h][2] + anchors[h][0];
                float y_center = out2[h * out2.w + 1] / y_scale * anchors[h][3] + anchors[h][1];
                float bw = out2[h * out2.w + 2] / w_scale * anchors[h][2];
                float bh = out2[h * out2.w + 3] / h_scale * anchors[h][3];

                val.push_back(y_center - bh / 2.0);
                val.push_back(x_center - bw / 2.0);
                val.push_back(y_center + bh / 2.0);
                val.push_back(x_center + bw / 2.0);

                for (size_t k = 0; k < 6; k++)
                {
                    int offset = 4 + k * 2;
                    float keypoint_x = out2[h * out2.w + offset] / x_scale * anchors[h][2] + anchors[h][0];
                    float keypoint_y = out2[h * out2.w + offset + 1] / y_scale * anchors[h][3] + anchors[h][1];
                    val.push_back(keypoint_x);
                    val.push_back(keypoint_y);
                }
                val.push_back(sigmoid(out1[h * out1.w + w]));
                det_list.push_back(val);
            }

        }
    }
    
    sort(det_list);
    std::vector< std::vector< float > > res_face = WeightedNonMaxSuppression(det_list);
    for (size_t i = 0; i < res_face.size(); i++)
    {

        Object face;
        face.rect.x = res_face[i][1];
        face.rect.y = res_face[i][0];
        face.rect.width = res_face[i][3] - res_face[i][1];
        face.rect.height = res_face[i][2] - res_face[i][0];
        face.prob = res_face[i][16];
        for (size_t k = 0; k < 6; k++)
        {
            int offset = 4 + k * 2;
            cv::Point2f kp;

            kp.x = res_face[i][offset];
            kp.y = res_face[i][offset + 1];
            face.landmarks.push_back(kp);
        }
        objects.push_back(face);
    }

    return 0;
}
