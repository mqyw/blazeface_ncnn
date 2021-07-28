#include "ncnn/net.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include<iostream>
#include <algorithm>
#include <fstream>
#include <string>
#include <direct.h>
#include <vector>
#include <io.h> 
#include <windows.h>

#include "face_net.h"



using namespace std;
using namespace cv;

int64_t GetSysTimeMicros()
{

#define EPOCHFILETIME   (116444736000000000UL)

	FILETIME ft;
	LARGE_INTEGER li;
	int64_t tt = 0;
	GetSystemTimeAsFileTime(&ft);
	li.LowPart = ft.dwLowDateTime;
	li.HighPart = ft.dwHighDateTime;
	// 从1970年1月1日0:0:0:000到现在的微秒数(UTC时间)
	tt = (li.QuadPart - EPOCHFILETIME) / 10;
	return tt;
}
int64_t Tstart1, Tend1;
float TimeCost;


int main() 
{
#if 1
	faceDet fd("data/blazeface.param", "data/blazeface.bin", 128, 128);
	cv::Mat src = cv::imread("111.jpeg");
	Tstart1 = GetSysTimeMicros();
	std::vector<Object> objects;
	fd.detect(src, objects);
	Tend1 = GetSysTimeMicros();
	TimeCost = (float)(Tend1 - Tstart1) / 1000;
	std::cout << "car detect times:" << TimeCost << " ms" << std::endl;

    for (size_t i = 0; i < objects.size(); i++)
    {
        int x1 = objects[i].rect.x * src.cols;
        int y1 = objects[i].rect.y * src.rows;
        int width1 = objects[i].rect.width * src.cols;
        int height1 = objects[i].rect.height * src.rows;
        printf("%f %f %f %f\n", objects[i].rect.x, objects[i].rect.y, objects[i].rect.width, objects[i].rect.height);
        cv::rectangle(src, cv::Rect(cv::Point(x1, y1), cv::Size(width1, height1)), cv::Scalar(255, 255, 255));

        char text[256];
        sprintf_s(text, "%.1f%%", objects[i].prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        cv::putText(src, text, cv::Point(x1, y1 - label_size.height),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));

        for (size_t k = 0; k < 6; k++)
        {
            cv::Point kp;
            kp.x = objects[i].landmarks[k].x * src.cols;
            kp.y = objects[i].landmarks[k].y * src.rows;
            circle(src, Point(kp.x, kp.y), 3, Scalar(0, 0, 255), -1);
        }
    }

	cv::imshow("src", src);
	cv::waitKey();
#else
    faceDet fd("data/blazeface.param", "data/blazeface.bin", 128, 128);
    std::string path = "img";
    std::cout << "--------------------------------- dir path:" << path << " ---------------------------------" << std::endl;
    //文件句柄 
    long long hFile = 0;	//文件信息，_finddata_t需要io.h头文件 
    struct _finddata_t fileinfo;
    std::string p;
    int pnum = 0;
    if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
    {
        do
        {
            //如果是目录,迭代之  		
            //如果不是,加入列表  	
            if ((fileinfo.attrib & _A_SUBDIR))
            {

            }
            else
            {
                Tstart1 = GetSysTimeMicros();
                p.assign(path).append("\\").append(fileinfo.name);
                cv::Mat src;
                src = cv::imread(p.c_str());
                if (src.empty())
                {
                    printf_s("continue image path:%s \n", p.c_str());
                    continue;
                }
                pnum++;
                std::vector<Object> objects;
                fd.detect(src, objects);

                Tend1 = GetSysTimeMicros();
                TimeCost = (float)(Tend1 - Tstart1) / 1000;
                std::cout << "car detect times:" << TimeCost << " ms" << std::endl;

                cv::Mat showimg = src.clone();

                for (size_t i = 0; i < objects.size(); i++)
                {
                    int x1 = objects[i].rect.x * showimg.cols;
                    int y1 = objects[i].rect.y * showimg.rows;
                    int width1 = objects[i].rect.width * showimg.cols;
                    int height1 = objects[i].rect.height * showimg.rows;
                    printf("%f %f %f %f\n", objects[i].rect.x, objects[i].rect.y, objects[i].rect.width, objects[i].rect.height);
                    cv::rectangle(showimg, cv::Rect(cv::Point(x1, y1), cv::Size(width1, height1)), cv::Scalar(255, 255, 255));

                    char text[256];
                    sprintf_s(text, "%.1f%%", objects[i].prob * 100);

                    int baseLine = 0;
                    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                    cv::putText(showimg, text, cv::Point(x1, y1 - label_size.height),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));

                    for (size_t k = 0; k < 6; k++)
                    {
                        cv::Point kp;
                        kp.x = objects[i].landmarks[k].x * showimg.cols;
                        kp.y = objects[i].landmarks[k].y * showimg.rows;
                        circle(showimg, Point(kp.x, kp.y), 3, Scalar(0, 0, 255),-1);
                    }
                }
                cv::imshow("src", showimg);
                cv::waitKey();

            }
        } while (_findnext(hFile, &fileinfo) == 0);
        _findclose(hFile);
    }
#endif
	return 0;
}