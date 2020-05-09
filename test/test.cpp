#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <chrono>
#include <ctime>
#include <climits>
 
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
 
#include <ceres/ceres.h>
#include <ceres/rotation.h>
 
using namespace std;
 
 
//一次测量值，只是一个点的三维坐标和这个点对应灰度图上的灰度值
struct Measurement
{
  Measurement (Eigen::Vector3d p, float g) : pos_world(p), grayscale(g) {}
  Eigen::Vector3d pos_world;
  float grayscale;
};
 
//像素坐标到空间点三维坐标计算
inline Eigen::Vector3d project2Dto3D (int x, int y, int d, float fx, float fy, float cx, float cy, float scale)
{
  float zz = float(d) / scale;
  float xx = zz * (x - cx) / fx;
  float yy = zz * (y - cy) / fy;
  return Eigen::Vector3d(xx, yy, zz);
}
 
inline Eigen::Vector2d project3Dto2D (float x, float y, float z, float fx, float fy, float cx, float cy)
{
  float u = fx * x / z + cx;
  float v = fy * y / z + cy;
  return Eigen::Vector2d (u,v);
}
 
bool poseEstimationDirect(const vector<Measurement>& measurements, cv::Mat* gray, Eigen::Matrix3f& intrinsics, Eigen::Isometry3d& Tcw);
 
 
class SparseBA : public ceres::SizedCostFunction<1,6>
{
public:
  cv::Mat * gray_;
  double cx_, cy_;
  double fx_, fy_;
  
  double pixelValue_;
  double X_, Y_, Z_;
  
SparseBA(cv::Mat *gray, double cx, double cy, double fx, double fy, double X, double Y, double Z, double pixelValue)
{
  gray_ = gray;
  cx_ = cx;
  cy_ = cy;
  fx_ = fx;
  fy_ = fy;
  X_ = X;
  Y_ = Y;
  Z_ = Z;
  pixelValue_ = pixelValue;
}
 
virtual bool Evaluate (double const *const *pose, double *residual, double **jacobians) const{
  //存储p的坐标
  double p[3];
  p[0] = X_; 
  p[1] = Y_;
  p[2] = Z_;
  
  //存储新的p'的坐标
  double newP[3];
  double R[3];
  R[0] = pose[0][0];
  R[1] = pose[0][1];
  R[2] = pose[0][2];
  ceres::AngleAxisRotatePoint(R, p, newP);
  
  newP[0] += pose[0][3];
  newP[1] += pose[0][4];
  newP[2] += pose[0][5];
  
  //新的p‘点投影到像素坐标系
  double ux = fx_ * newP[0] / newP[2] + cx_;
  double uy = fy_ * newP[1] / newP[2] + cy_;
  
  residual[0] = getPixelValue(ux, uy) - pixelValue_;
  
  if (jacobians)
  {
    double invz = 1.0 / newP[2];
    double invz_2 = invz * invz;
    
    //公式8.15
    Eigen::Matrix<double, 2, 6> jacobian_uv_ksai;
    jacobian_uv_ksai(0,0) = -newP[0] * newP[1] * invz_2 * fx_;
    jacobian_uv_ksai(0,1) = (1 + (newP[0] * newP[0] * invz_2)) * fx_;
    jacobian_uv_ksai(0,2) = -newP[1] * invz * fx_;
    jacobian_uv_ksai(0,3) = invz * fx_;
    jacobian_uv_ksai(0,4) = 0;
    jacobian_uv_ksai(0,5) = -newP[0] * invz_2 * fx_;
    
    jacobian_uv_ksai(1,0) = -(1 + newP[1] * newP[1] * invz_2) * fy_;
    jacobian_uv_ksai(1,1) = newP[0] * newP[1] * invz_2 * fy_;
    jacobian_uv_ksai(1,2) = newP[0] * invz * fy_;
    jacobian_uv_ksai(1,3) = 0;
    jacobian_uv_ksai(1,4) = invz * fy_;
    jacobian_uv_ksai(1,5) = -newP[1] * invz_2 * fy_;
    
    //像素梯度
    Eigen::Matrix<double, 1, 2> jacobian_pixel_uv;
    jacobian_pixel_uv(0,0) = (getPixelValue(ux+1, uy) - getPixelValue(ux-1, uy))/2;
    jacobian_pixel_uv(0,1) = (getPixelValue(ux, uy+1) - getPixelValue(ux, uy-1))/2;
    
    //公式8.16
    Eigen::Matrix<double, 1, 6> jacobian = jacobian_pixel_uv * jacobian_uv_ksai;
    
    jacobians[0][0] = jacobian(0);
    jacobians[0][1] = jacobian(1);
    jacobians[0][2] = jacobian(2);
    jacobians[0][3] = jacobian(3);
    jacobians[0][4] = jacobian(4);
    jacobians[0][5] = jacobian(5);
  }
  
  return true;
  
}
 
double getPixelValue (double x, double y) const
{
  uchar* data = & gray_->data[int(y) * gray_->step + int(x)];
  double xx = x - floor(x);
  double yy = y - floor(y);
  return double (
    (1 - xx) * (1 - yy) * data[0] + xx * (1 - yy) * data[1] + (1 - xx) * yy * data[gray_->step] + xx * yy * data[gray_->step + 1]
  );
}
};
 
 
 
int main(int argc, char** argv)
{
  //定位数据文件
  string path_to_dataset = "/home/fuhang/projects/useLK/data";
  string associate_file = path_to_dataset + "/associate.txt";
  
  //读入到文件输入流
  ifstream fin (associate_file);
  
  string rgb_file, time_rgb, depth_file, time_depth;
  
  cv::Mat color, depth, gray;
  //测量值数组，就是一堆灰度值
  vector<Measurement> measurements;
  
  //相机内参值
  float cx = 325.5;
  float cy = 253.5;
  float fx = 518.0;
  float fy = 519.0;
  
  //单位换算值，相机空间单位为毫米，三维空间单位为米
  float depth_scale = 1000.0;
  Eigen::Matrix3f K;
  K << fx, 0.f, cx, 0.f, fy, cy, 0.f, 0.f, 1.0f;
  
  //位姿矩阵，在这里初始化为单位矩阵
  Eigen::Isometry3d Tcw = Eigen::Isometry3d::Identity();
  
  //prev_color也就是图像流的第一帧，在整个程序中，也只有第一帧对这变量进行了赋值，所有帧的参考帧都为第一帧
  cv::Mat prev_color;
  
  for (int index=0; index<10; index++)
  {
    cout << "*************************loop " << index << "*************************************" << endl;
    //从输入流中中读取这四个变量
    fin >> time_rgb >> rgb_file >> time_depth >> depth_file;
    //读取彩色图和深度图
    color = cv::imread(path_to_dataset + "/" + rgb_file);
    depth = cv::imread(path_to_dataset + "/" + depth_file, -1);
    
    //空指针说明这一帧有损坏，直接跳过
    if (color.data == nullptr || depth.data == nullptr)
      continue;
    
    //将彩色图转化为灰度图
    cv::cvtColor(color, gray, cv::COLOR_BGR2GRAY);
    //printf("a");
    //程序开始对第一帧进行处理，也就是参考帧
    if (index == 0)
    {
     //双层循环遍历像素点，不要图片边缘
      for (int x=10; x<gray.cols-10; x++)
	for (int y=10; y<gray.rows-10; y++)
	{
	  //梯度向量delta，定义为（x,y）像素右减左梯度值和下减上梯度值，和本身灰度无关
	  Eigen::Vector2d delta(
	    gray.ptr<uchar>(y)[x+1] - gray.ptr<uchar>(y)[x-1],
	    gray.ptr<uchar>(y+1)[x] - gray.ptr<uchar>(y-1)[x]
	  );
	  //模长小于50则梯度不明显就跳过
	  if (delta.norm() < 50)
	    continue;
	  ushort d = depth.ptr<ushort>(y)[x];
	  if (d==0)
	    continue;
	  Eigen::Vector3d p3d = project2Dto3D(x, y, d, fx, fy, cx, cy, depth_scale);
	  float grayscale = float(gray.ptr<uchar>(y)[x]);
	  measurements.push_back(Measurement(p3d, grayscale));
	}
 
      prev_color = color.clone();
      cout << "add total " << measurements.size() << " measurements. " << endl; 
      continue;
    }
  
    //使用直接法计算相机运动
    //开始计算时间
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    //在这个函数执行时measurements是不变的，只有不断读入的&gray灰度图是变化的
    poseEstimationDirect (measurements, &gray, K, Tcw);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
    cout << "Tcw = " << endl << Tcw.matrix() << endl;
  
    //画出特征点
    cv::Mat img_show (color.rows * 2, color.cols, CV_8UC3);
    prev_color.copyTo(img_show (cv::Rect (0,0,color.cols, color.rows)));
    color.copyTo(img_show (cv::Rect(0,color.rows,color.cols, color.rows)));
  
    //遍历数组measurements并进行操作
    for (Measurement m:measurements)
    {
      //随机选20%的关键点
      if (rand() > RAND_MAX/5)
        continue;
      //取得空间点世界坐标系下坐标
      Eigen::Vector3d p = m.pos_world;
      //求一下这个空间点在第一帧像素中的的坐标，所以坐标每张显示图中上半部分一样
      Eigen::Vector2d pixel_prev = project3Dto2D (p(0,0), p(1,0), p(2,0), fx, fy, cx, cy);
      //空间点乘以位姿估计
      Eigen::Vector3d p2 = Tcw*m.pos_world;
      //在新的一帧中寻找像素位置
      Eigen::Vector2d pixel_now = project3Dto2D (p2(0,0), p2(1,0), p2(2,0), fx, fy, cx, cy);
      //如果像素超出平面外就舍去
      if (pixel_now(0,0)<0 || pixel_now(0,0)>=color.cols || pixel_now(1,0)<0 || pixel_now(1,0)>=color.rows)
        continue;
    
      //随机色使用
      float b = 255*float (rand()) / RAND_MAX;
      float g = 255*float (rand()) / RAND_MAX;
      float r = 255*float (rand()) / RAND_MAX;
    
      //追踪特征圆和匹配直线
      cv::circle(img_show, cv::Point2d(pixel_prev(0,0), pixel_prev(1,0)), 8, cv::Scalar(b,g,r), 2);
      cv::circle(img_show, cv::Point2d(pixel_now(0,0), pixel_now(1,0) + color.rows), 8, cv::Scalar(b,g,r), 2);
      cv::line (img_show, cv::Point2d(pixel_prev(0,0), pixel_prev(1,0)), cv::Point2d(pixel_now(0,0),pixel_now(1,0) + color.rows), cv::Scalar(b,g,r)
      , 1);
    
    }
  
  //输出图像
    cv::imshow("result", img_show);
    cv::waitKey(0);
  }
  return 0;
}
 
//用直接法求位姿的函数poseEstimationDirect
bool poseEstimationDirect(const std::vector< Measurement >& measurements, cv::Mat* gray, Eigen::Matrix3f& K, Eigen::Isometry3d& Tcw)
{
  ceres::Problem problem;
  //定义位姿数组
  double pose[6];
  //用轴角进行优化
  Eigen::AngleAxisd rotationVector(Tcw.rotation());
  pose[0] = rotationVector.angle() * rotationVector.axis()(0);
  pose[1] = rotationVector.angle() * rotationVector.axis()(1);
  pose[2] = rotationVector.angle() * rotationVector.axis()(2);
  pose[3] = Tcw.translation()(0);
  pose[4] = Tcw.translation()(1);
  pose[5] = Tcw.translation()(2);
  
  //构建Ceres问题
  for (Measurement m:measurements)
  {
    ceres::CostFunction * costFunction = new SparseBA(gray, K(0,2), K(1,2), K(0,0), K(1,1), m.pos_world(0), m.pos_world(1), m.pos_world(2), double(m.grayscale));
    problem.AddResidualBlock(costFunction, NULL, pose);
  }
  
  ceres::Solver::Options options;
  options.num_threads = 4;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  
  cv::Mat rotateVectorCV = cv::Mat::zeros(3, 1, CV_64FC1);
  rotateVectorCV.at<double>(0) = pose[0];
  rotateVectorCV.at<double>(1) = pose[1];
  rotateVectorCV.at<double>(2) = pose[2];
  
  cv::Mat RCV;
  cv::Rodrigues(rotateVectorCV, RCV);
  Tcw(0,0) = RCV.at<double>(0,0); Tcw(0,1) = RCV.at<double>(0,1); Tcw(0,2) = RCV.at<double>(0,2);
  Tcw(1,0) = RCV.at<double>(1,0); Tcw(1,1) = RCV.at<double>(1,1); Tcw(1,2) = RCV.at<double>(1,2);
  Tcw(2,0) = RCV.at<double>(2,0); Tcw(2,1) = RCV.at<double>(2,1); Tcw(2,2) = RCV.at<double>(2,2);
 
  Tcw(0,3) = pose[3];
  Tcw(1,3) = pose[4];
  Tcw(2,3) = pose[5];
}