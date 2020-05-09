#include "../util/ply.h"
#include "../util/tictoc.h"
#include <iostream>
int main(){

std::string file_path = "/home/zhaoran/thesis_ws/mpl/project/mesh.ply";
std::string out_file_path = "/home/zhaoran/thesis_ws/mpl/project/out.ply";

tictoc::tic();
mpl::PlyFile ply = mpl::readPlyFile(file_path);
int t1 = tictoc::toc();
std::cout <<" t1 :" << t1;
//mpl::convertNormal(ply);
mpl::writeBinaryPlyFile(out_file_path,ply);
tictoc::tic();
mpl::PlyFile ply2 = mpl::readPlyFile(out_file_path);
int t2 = tictoc::toc();
std::cout <<" t2 :" << t2;




}