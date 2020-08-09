#include "PhotometricBA.h"
#include "cam_data.h"
#include "candidate_manager.h"
#include "config.h"
#include "debug.h"
#include "synetic_image.h"
#include <future>
// clang-format off
#include "../visualizer/visualizer.h"
// clang-format on
#include "tracker.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace mpl;
void add_last_tracking_result_to_window(PointCloudPyramid::ptr pcp, Frame::ptr newst_KF, CandidateManager& cm) {
    auto& cam = CamData::getInstance();
    for (auto& voxel : pcp->operator[](0)) {
        if ((voxel.can->status != CandidateStatus::ACTIVE && voxel.can->status != CandidateStatus::OOB)) continue;

        if (voxel.can->point_block == nullptr) {  // can,that never been added to sw
            voxel.can->point_block = std::make_unique<PointParameterBlock>(voxel.can->d_inv_synetic_im);

            // add obs on the frame, that added into sw before
            auto& kf_vec = cm.get_key_frames();
            for (auto it = kf_vec.begin(); *it != voxel.can->host_frame; ++it) {
                Eigen::Vector2f projection = unproject_trans_project(voxel.can, voxel.can->host_frame, *it);

                // todo check energy
                if (is_in_img(cam, projection)) {
                    std::unique_ptr<PhotometricResidual> obs =
                        std::make_unique<PhotometricResidual>(voxel.can, it->get());
                    voxel.can->observations[*it] = std::move(obs);
                }
            }
        }

        // add all active point the new observation on newst KF
        std::unique_ptr<PhotometricResidual> obs_newst_KF =
            std::make_unique<PhotometricResidual>(voxel.can, newst_KF.get());
        voxel.can->observations[newst_KF] = std::move(obs_newst_KF);
    }
}

bool is_newframe_KF(PointCloudPyramid::ptr pcp, const Sophus::SE3f& T_KF_curr, float energy) {
    std::cout << "@@@@@@@@ log norm: " << T_KF_curr.log().norm() << "  angle Y : " << abs(T_KF_curr.angleY())
              << "   energy :" << energy << '\n';

    // should use depth/movement ratio and candidate num cnt
    return (abs(T_KF_curr.angleY()) > 0.02 || T_KF_curr.log().norm() > 0.7 || energy > 7.f ||
            T_KF_curr.log().tail(3).norm() > 0.01);
}

void show_debug_key_frame_synetic_img_alignment(cv::Mat frame_img, cv::Mat synetic_img, cv::Mat mask,
                                                cv::Mat alignment_weight_mask) {
    cv::Mat mask_32F;
    cv::Mat zero_mat(mask.size(), CV_8UC1, cv::Scalar(0.0f));
    cv::Mat synetic_8U;
    synetic_img.convertTo(synetic_8U, CV_8UC1);

    alignment_weight_mask.copyTo(zero_mat, synetic_8U);

    zero_mat.convertTo(mask_32F, CV_32FC1);

    cv::normalize(frame_img, frame_img, 0, 1, cv::NORM_MINMAX);
    cv::normalize(mask_32F, mask_32F, 0, 1, cv::NORM_MINMAX);
    cv::normalize(synetic_img, synetic_img, 0, 1, cv::NORM_MINMAX);

    double num_pixel = cv::sum(mask)[0];

    double diff_sum = cv::sum(mask_32F)[0];

    std::cout << "  diff sum :" << diff_sum << " num pixel : " << num_pixel << " average diff "
              << 255 * diff_sum / num_pixel << '\n';

    cv::vconcat(mask_32F, frame_img, frame_img);
    cv::vconcat(frame_img, synetic_img, frame_img);

    cv::resize(frame_img, frame_img, cv::Size(frame_img.cols / 1.5, frame_img.rows / 1.5));

    cv::imshow("alignment result", frame_img);
    cv::waitKey(0);
}

cv::Mat make_alignment_weight_mask(cv::Mat frame_img, cv::Mat synetic_img, const AffineLight& aff_map_c) {
    Config& config = Config::getInstance();

    cv::Mat abs_diff_img;
    cv::Mat synetic_img_single_channel, frame_img_single_channel;

    cv::cvtColor(synetic_img, synetic_img_single_channel, CV_RGBA2GRAY);
    cv::Mat mask = synetic_img_single_channel.clone();

    synetic_img_single_channel.convertTo(synetic_img_single_channel, CV_32FC1);
    frame_img.convertTo(frame_img_single_channel, CV_32FC1);

    cv::Mat zero_mat = cv::Mat::zeros(synetic_img_single_channel.size(), CV_32FC1);
    cv::addWeighted(frame_img_single_channel, exp(aff_map_c.alpha()), zero_mat, 0, aff_map_c.beta(),
                    frame_img_single_channel);

    cv::absdiff(frame_img_single_channel, synetic_img_single_channel, abs_diff_img);

    cv::threshold(mask, mask, 1, 1, cv::THRESH_BINARY);

    cv::Mat result(mask.size(), CV_8UC1, cv::Scalar(255));
    cv::Mat gauss_blured_result = result.clone();
    cv::Mat abs_8U;
    cv::normalize(abs_diff_img, abs_8U, 0, 255, cv::NORM_MINMAX);
    abs_8U.convertTo(abs_8U, CV_8UC1);
    abs_8U.copyTo(result, mask);

    cv::GaussianBlur(result, gauss_blured_result, cv::Size(3, 3), 3);

    debug::execute_func_according_to_config(config.DEBUG_KEY_FRAME_SYNETIC_IMAGE_ALIGNMENT,
                                            config.debug_key_frame_synetci_img_alignment_mutex,
                                            show_debug_key_frame_synetic_img_alignment, frame_img_single_channel,
                                            synetic_img_single_channel, mask, gauss_blured_result);

    return gauss_blured_result;
}

int main() {
    std::string project_path = "/home/zhaoran/thesis_ws/mpl/project/";
    // read yaml
    Config& config = Config::getInstance();
    std::string config_path = project_path + "config.yaml";
    config.readYamlFile(config_path);

    CamData& cam = CamData::getInstance();
    std::string calib_path = project_path + "camera.yaml";
    cam.readYamlFile(calib_path);

    // read mesh model, texture,initial synetic image
    SyneticImage synetic_image(project_path);

    std::string pose_file = project_path + "ground_truth.txt";
    trajectory_io::Trajectory pose;
    pose.read(pose_file, trajectory_io::Trajectory::FORMAT_MAT);

    // read 12 image date for test
    std::string data_path = "/home/zhaoran/dataset/KITTI/sequences/00/image_0/";
    std::vector<cv::Mat> img_vec;
    std::vector<cv::Mat> img_draw_vec;
    for (int i = 0; i < 520; ++i) {
        int im_id = 80 + i;
        string im_name = (im_id < 100) ? "0000" + to_string(im_id) + ".png" : "000" + to_string(im_id) + ".png";
        cv::Mat im_tmp = cv::imread(data_path + im_name);

        assert(im_tmp.isContinuous());
        // cv::putText(im_tmp,                                                // target image
        //            "T",                                                   // text
        //            cv::Point(0, im_tmp.rows),                             // down-left position
        //            cv::FONT_HERSHEY_DUPLEX, 13.0, CV_RGB(255, 255, 255),  // font color
        //            14);

        img_draw_vec.push_back(im_tmp.clone());

        cv::cvtColor(im_tmp, im_tmp, CV_BGR2GRAY);
        assert(im_tmp.channels() == 1);
        img_vec.push_back(im_tmp);
    }
    // 1360.png

    // test :
    // build a window with 2 old KF and a new added frame
    // use very simple strategy : add KF after 2 frames
    // 1. activate candidate of 2 old KF
    // 2. generate point cloud pyramid on newst frame for tracking

    // system begin
    int start_idx = 0;  // 20
    synetic_image.set_start_pose(pose.atIndex(start_idx));
    Sophus::SE3f render_pose = Sophus::SE3f::transZ(0.0);

    int off_set = start_idx * 4;                                                        // 80
    Frame::ptr init_frame = Frame::create(cv::Mat::zeros(img_vec[0].size(), CV_8UC1));  // only has synetic im
    std::vector<cv::Mat> syn_im_vec_curr = synetic_image.renderingAt(render_pose);

    std::shared_ptr<Visualizer> vis(new Visualizer);
    vis->stop_or_start_according_to_pangolin_menu();
    CandidateManager cm(vis);

    init_frame->set_synetic_photometirc_im(syn_im_vec_curr[0]);
    cv::Mat alignment_weight_mask = cv::Mat::zeros(syn_im_vec_curr[1].size(), CV_8UC1);
    cm.select_candidate(init_frame, syn_im_vec_curr[1], alignment_weight_mask);

    PointCloudPyramid::ptr pcp = cm.get_point_cloud_pyramid();

    Tracker tracker;
    tracker.set_tracking_ref(init_frame, pcp);

    // given initial pose is not precise, so estimate it(tracking itself)
    Frame::ptr key_frame = Frame::create(img_vec[off_set]);
    tracker.tracking(key_frame);

    // add_last_tracking_result_to_window(pcp, key_frame, cm);

    // refine the initial state
    syn_im_vec_curr = synetic_image.renderingAt(key_frame->get_pose());
    key_frame->set_synetic_photometirc_im(syn_im_vec_curr[0]);
    alignment_weight_mask = make_alignment_weight_mask(img_vec[0], syn_im_vec_curr[0], key_frame->get_aff_light());
    cm.select_candidate(key_frame, syn_im_vec_curr[1], alignment_weight_mask);

    pcp = cm.get_point_cloud_pyramid();
    tracker.set_tracking_ref(key_frame, pcp);
    tracker.refine_pose(key_frame);

    cm.remove_frame(init_frame);
    Eigen::Isometry3f T_c_c0 = static_cast<Eigen::Isometry3f>(key_frame->get_pose().matrix());
    synetic_image.set_start_pose(pose.atIndex(start_idx) * T_c_c0.inverse());
    key_frame->set_pose(Sophus::SE3f::transZ(0.0f));
    key_frame->set_rendering_pose(Sophus::SE3f::transZ(0.0f));
    // debug::execute_func_according_to_config(
    //    config.DEBUG_KEY_FRAME_SYNETIC_IMAGE_ALIGNMENT, config.debug_key_frame_synetci_img_alignment_mutex,
    //    show_debug_key_frame_synetic_img_alignment, img_draw_vec[0], syn_im_vec_curr[0],
    //    key_frame->get_aff_light());
    std::thread th_draw(&Visualizer::publish_curr_frame_tracking_info, std::ref(*vis), pcp, img_draw_vec[off_set],
                        key_frame->get_pose(), key_frame->get_pose().inverse(), true);
    th_draw.detach();

    key_frame->get_frame_block() =
        std::make_unique<FrameParameterBlock>(key_frame->get_pose().cast<double>(), key_frame->get_aff_light());

    // draw candidates before and after tracking
    cv::Mat key_frame_vis = img_draw_vec[off_set];

    PhotometricBA pba;

    for (size_t i = off_set + 1; i < img_vec.size(); ++i) {
        vis->stop_or_start_according_to_pangolin_menu();
        // cv::imshow("frame", key_frame_vis);
        // cv::imshow("synetic", syn_im_vec_curr[0]);
        // debug::execute_func_according_to_config(
        //    config.DEBUG_KEY_FRAME_SYNETIC_IMAGE_ALIGNMENT, config.debug_key_frame_synetci_img_alignment_mutex,
        //    show_debug_key_frame_synetic_img_alignment, key_frame_vis, syn_im_vec_curr[0],
        //    key_frame->get_aff_light());
        // cv::waitKey(0);

        Frame::ptr curr_frame = Frame::create(img_vec[i]);

        // if tracking failed(due to e.g over exposure), throw this frame just away
        if (!tracker.tracking(curr_frame)) continue;

        Sophus::SE3f T_curr_KF = get_src_to_dst_transform(curr_frame->get_ref_frame(), curr_frame);
        // visualize tracking result
        std::thread th_draw(&Visualizer::publish_curr_frame_tracking_info, std::ref(*vis), pcp, img_draw_vec[i],
                            curr_frame->get_pose(), T_curr_KF, false);
        th_draw.detach();

        // todo : a best way to choose KF
        bool is_KF = true;  // is_newframe_KF(pcp, T_curr_KF, tracker.get_per_pixel_energy());
        if (is_KF) {
            key_frame = curr_frame;
            key_frame_vis = img_draw_vec[i];

            // rendering at new key frame's pose
            render_pose = curr_frame->get_pose();
            syn_im_vec_curr = synetic_image.renderingAt(render_pose);

            // add new frame to the window, and select candidate on it
            curr_frame->set_synetic_photometirc_im(syn_im_vec_curr[0]);
            cv::Mat alignment_weight_mask =
                make_alignment_weight_mask(img_vec[i], syn_im_vec_curr[0], key_frame->get_aff_light());

            key_frame->get_frame_block() =
                std::make_unique<FrameParameterBlock>(key_frame->get_pose().cast<double>(), key_frame->get_aff_light());

            // add new tracking result(residual)
            // add_last_tracking_result_to_window(pcp, key_frame, cm);

            cm.select_candidate(curr_frame, syn_im_vec_curr[1], alignment_weight_mask);

            // optimize key frame window
            // pba.solve(cm);

            // get and set new tracking reference point cloud after optimization
            pcp = cm.get_point_cloud_pyramid();

            // set pcp for next tracking
            tracker.set_tracking_ref(curr_frame, pcp);

            // refine itself
            tracker.refine_pose(curr_frame);

            // visualize: depth map
            std::thread th_draw_depth(&Visualizer::draw_and_publish_key_frame_depth, std::ref(*vis),
                                      syn_im_vec_curr[1]);
            th_draw_depth.detach();

            // sleep at last image
            if (i == img_vec.size() - 1) {
                while (true) std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        }
    }
}