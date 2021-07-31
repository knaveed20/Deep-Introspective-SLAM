#include <ros/ros.h>
#include <iostream>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

using namespace std;

int main(int argc, char* argv[])
{
    ros::init(argc,argv,"minos_image_publisher");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    image_transport::Publisher pub=it.advertise("/usb_cam/image_raw",12);
    stringstream ss;
    std::string image_name;
    stringstream exception_text;
    ss<<"/home/frames/color_.jpg";
    image_name=ss.str();
    cv::Mat image;
    sensor_msgs::ImagePtr msg;
    ros::Rate loop_rate(5);
    
    while(nh.ok()) {
        
    	image=cv::imread(image_name,CV_LOAD_IMAGE_COLOR);
        if(!image.empty()) {
            msg=cv_bridge::CvImage(std_msgs::Header(),"bgr8",image).toImageMsg();
            pub.publish(msg);
        }
    	//ROS_INFO_STREAM(image_name);
    	ros::spinOnce();
    	loop_rate.sleep();
    }

}
