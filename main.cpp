#include "helpers.hpp"

#include "eos/core/Mesh.hpp"
#include "eos/core/read_obj.hpp"
#include "eos/fitting/fitting.hpp"
#include "eos/fitting/contour_correspondence.hpp"
#include "eos/fitting/closest_edge_fitting.hpp"
#include "eos/fitting/RenderingParameters.hpp"
#include "eos/render/utils.hpp"
#include "eos/render/render.hpp"
#include "eos/render/texture_extraction.hpp"
#include "eos/render/draw_utils.hpp"
#include "eos/core/Image_opencv_interop.hpp"

#include "rcr/model.hpp"
#include "cereal/cereal.hpp"

#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/quaternion.hpp"

#include "Eigen/Dense"

#include "opencv2/highgui.hpp"

#include <vector>
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h>

#include "FaceDetection.h"

class Face3d
{
public:
    Face3d();
    
    //输入图像并计算
    void inputFrame(const cv::Mat &input);
    
    //输出
    void output();
    
    //得到五个不同视角的脸部图像
    static void get5faces(std::vector<cv::Mat> &faces);
    
    //测试用
    void test();
    
private:
    eos::morphablemodel::MorphableModel morphable_model;
    eos::core::LandmarkMapper landmark_mapper;
    
    eos::fitting::ModelContour model_contour;
    eos::fitting::ContourLandmarks ibug_contour;
    
    rcr::detection_model rcr_model;
    
    eos::morphablemodel::Blendshapes blendshapes;
    
    eos::morphablemodel::EdgeTopology edge_topology;
    
    cv::Mat frame, unmodified_frame;
    
    bool have_face;
    rcr::LandmarkCollection<cv::Vec2f> current_landmarks;
    cv::Rect current_facebox;
    WeightedIsomapAveraging isomap_averaging;
    cv::Mat isomap;
    cv::Mat merged_isomap;
    PcaCoefficientMerging pca_shape_merging;
    
    eos::fitting::RenderingParameters rendering_params;
    std::vector<float> shape_coefficients, blendshape_coefficients;
    std::vector<Eigen::Vector2f> image_points;
    eos::core::Mesh mesh;
};

Face3d g_face3d;

glm::mat4x4 p0(0.99,0.014,0.2,0,
               0,0.98,-0.2,0,
               -0.03,0.2,0.98,0,
               0,0,0,1);
glm::mat4x4 p1(0.95,-0.1,-0.3,0,
               0.036,0.98,-0.2,0,
               0.33,0.17,0.93,0,
               0,0,0,1);
glm::mat4x4 p2(0.96,0,0.3,0,
               0.04,0.99,-0.15,0,
               -0.3,0.15,0.94,0,
               0,0,0,1);
glm::mat4x4 p3(0.99,-0.04,-0.05,0,
               0.01,0.87,-0.5,0,
               0.06,0.5,0.87,0,
               0,0,0,1);
glm::mat4x4 p4(0.99,0.06,-0.03,0,
               -0.04,0.94,0.34,0,
               0.04,-0.34,0.94,0,
               0,0,0,1);

using namespace eos;
using cv::Mat;
using cv::Vec2f;
using cv::Vec3f;
using cv::Vec4f;
using cv::Rect;
using std::cout;
using std::endl;
using std::vector;
using std::string;

Face3d::Face3d()
{
    std::string modelfile = "./share/sfm_shape_3448.bin";
    std::string mappingsfile = "./share/ibug_to_sfm.txt";
    std::string contourfile = "./share/sfm_model_contours.json";
    std::string edgetopologyfile = "./share/sfm_3448_edge_topology.json";
    std::string blendshapesfile = "./share/expression_blendshapes_3448.bin";
    std::string landmarkdetector = "./share/face_landmarks_model_rcr_68.bin";
    
    morphable_model = morphablemodel::load_model(modelfile);
    landmark_mapper = mappingsfile.empty() ? core::LandmarkMapper() : core::LandmarkMapper(mappingsfile);
    
    model_contour = contourfile.empty() ? fitting::ModelContour() : fitting::ModelContour::load(contourfile);
    ibug_contour = fitting::ContourLandmarks::load(mappingsfile);
    
    // Load the landmark detection model:
    try {
        rcr_model = rcr::load_detection_model(landmarkdetector);
    }
    catch (const cereal::Exception& e) {
        cout << "Error reading the RCR model " << landmarkdetector << ": " << e.what() << endl;
        exit(0);
    }
    
    blendshapes = morphablemodel::load_blendshapes(blendshapesfile);
    
    edge_topology = morphablemodel::load_edge_topology(edgetopologyfile);
    
    have_face = false;
    isomap_averaging = WeightedIsomapAveraging(60.f);
}

void Face3d::inputFrame(const cv::Mat &input)
{
    frame = input.clone();
    if (frame.empty()) {
        return;
    }
    
    // We do a quick check if the current face's width is <= 50 pixel. If it is, we re-initialise the tracking with the face detector.
    if (have_face && get_enclosing_bbox(rcr::to_row(current_landmarks)).width <= 50) {
        cout << "Reinitialising because the face bounding-box width is <= 50 px" << endl;
        have_face = false;
    }
    
    unmodified_frame = frame.clone();
    
    if (!have_face) {
        vector<Rect> detected_faces;
        
        g_faceDT.detect(unmodified_frame,detected_faces);
        if (detected_faces.empty()) {
            return;
        }
        cv::rectangle(frame, detected_faces[0], { 255, 0, 0 });
        // Rescale the V&J facebox to make it more like an ibug-facebox:
        // (also make sure the bounding box is square, V&J's is square)
        Rect ibug_facebox = rescale_facebox(detected_faces[0], 0.85, 0.2);
        current_landmarks = rcr_model.detect(unmodified_frame, ibug_facebox);
        have_face = true;
    }
    else {
        // We already have a face - track and initialise using the enclosing bounding
        // box from the landmarks from the last frame:
        auto enclosing_bbox = get_enclosing_bbox(rcr::to_row(current_landmarks));
        enclosing_bbox = make_bbox_square(enclosing_bbox);
        current_landmarks = rcr_model.detect(unmodified_frame, enclosing_bbox);
    }
    
    //time
    //double st = cv::getTickCount();
    
    // Fit the 3DMM:
    std::tie(mesh, rendering_params) = fitting::fit_shape_and_pose(
                morphable_model, blendshapes, rcr_to_eos_landmark_collection(current_landmarks), 
                landmark_mapper, unmodified_frame.cols, unmodified_frame.rows, edge_topology, 
                ibug_contour, model_contour, 3, 5, 15.0f, cpp17::nullopt, shape_coefficients, 
                blendshape_coefficients, image_points);
    
    //time
    //double rt = (cv::getTickCount()-st)/cv::getTickFrequency();
    //std::cout<<"fit 3dmm runtime:"<<rt<<std::endl;
    //st = cv::getTickCount();
    
    // Extract the texture using the fitted mesh from this frame:
    const Eigen::Matrix<float, 3, 4> affine_cam = fitting::get_3x4_affine_camera_matrix(rendering_params, frame.cols, frame.rows);
    
    //time
    //rt = (cv::getTickCount()-st)/cv::getTickFrequency();
    //std::cout<<"rest0 runtime:"<<rt<<std::endl;
    //st = cv::getTickCount();
    
    isomap = core::to_mat(render::extract_texture(mesh, affine_cam, core::from_mat(unmodified_frame), true, render::TextureInterpolation::NearestNeighbour, 512));
    
    //time
    //rt = (cv::getTickCount()-st)/cv::getTickFrequency();
    //std::cout<<"rest0.5 runtime:"<<rt<<std::endl;
    //st = cv::getTickCount();
    
    // Merge the isomaps - add the current one to the already merged ones:
    merged_isomap = isomap_averaging.add_and_merge(isomap);
    
    //time
    //rt = (cv::getTickCount()-st)/cv::getTickFrequency();
    //std::cout<<"rest1 runtime:"<<rt<<std::endl;
    //st = cv::getTickCount();
    
    // Same for the shape:
    shape_coefficients = pca_shape_merging.add_and_merge(shape_coefficients);
    
    //time
    //rt = (cv::getTickCount()-st)/cv::getTickFrequency();
    //std::cout<<"rest2 runtime:"<<rt<<std::endl;
    
    /*
    //test
    for(int i=0;i<4;i++)
    {
        for(int j=0;j<4;j++)
            std::cout<<modelview_no_translation[i][j]<<" ";
        std::cout<<"\n"<<std::endl;
    }
    */
}

void Face3d::test()
{
    if(!have_face)
        return;
    
    // Wireframe rendering of mesh of this frame (non-averaged):
    render::draw_wireframe(frame, mesh, rendering_params.get_modelview(), rendering_params.get_projection(), fitting::get_opencv_viewport(frame.cols, frame.rows));
    rcr::draw_landmarks(frame, current_landmarks, { 0, 0, 255 }); // red, initial landmarks
    
    cv::namedWindow("mesh",0);
    cv::imshow("mesh",frame);
    
    cv::namedWindow("isomap",0);
    cv::imshow("isomap",isomap);
    
    cv::namedWindow("merged_isomap",0);
    cv::imshow("merged_isomap",merged_isomap);
    
    std::vector<cv::Mat> faces(5);
    //save("");
    //get5faces(faces);
    std::vector<glm::mat4x4> mvnts(5);
    mvnts[0] = p0; mvnts[1] = p1; mvnts[2] = p2; mvnts[3] = p3; mvnts[4] = p4;
    for(size_t i=0;i<5;i++)
    {
        core::Image4u rd;
        std::tie(rd, std::ignore) = render::render(mesh, mvnts[i], 
                                                   glm::ortho(-130.0f, 130.0f, -130.0f, 130.0f), 256, 256, 
                                                   render::create_mipmapped_texture(merged_isomap), true, false, false);
        faces[i] = core::to_mat(rd);
    }
    for(size_t i=0;i<faces.size();i++)
        cv::imshow("face"+std::to_string(i),faces[i]);
    
    cv::waitKey();
}

void Face3d::get5faces(std::vector<cv::Mat> &faces)
{
    core::Mesh mesh = core::read_obj("current_merged.obj");
    cv::Mat isomap = cv::imread("merged_isomap.png");
    
    faces.resize(5);
    std::vector<glm::mat4x4> mvnts(5);
    mvnts[0] = p0; mvnts[1] = p1; mvnts[2] = p2; mvnts[3] = p3; mvnts[4] = p4;
    for(size_t i=0;i<5;i++)
    {
        core::Image4u rd;
        std::tie(rd, std::ignore) = render::render(mesh, mvnts[i], 
                                                   glm::ortho(-130.0f, 130.0f, -130.0f, 130.0f), 256, 256, 
                                                   render::create_mipmapped_texture(isomap), true, false, false);
        faces[i] = core::to_mat(rd);
    }
}

void Face3d::output()
{
    /*
    // save an obj + current merged isomap to the disk:
    const core::Mesh neutral_expression = 
            morphablemodel::sample_to_mesh(morphable_model.get_shape_model().draw_sample(shape_coefficients), 
                                           morphable_model.get_color_model().get_mean(), 
                                           morphable_model.get_shape_model().get_triangle_list(), 
                                           morphable_model.get_color_model().get_triangle_list(), 
                                           morphable_model.get_texture_coordinates());
    
    core::write_textured_obj(neutral_expression, "current_merged.obj");
    */
    
    std::string outPath = "./output";
    
    if(access(outPath.data(),F_OK) == -1)
        mkdir(outPath.data(),00777);
    
    std::vector<glm::mat4x4> mvnts(5);
    mvnts[0] = p0; mvnts[1] = p1; mvnts[2] = p2; mvnts[3] = p3; mvnts[4] = p4;
    for(size_t i=0;i<5;i++)
    {
        core::Image4u rd;
        std::tie(rd, std::ignore) = render::render(mesh, mvnts[i], 
                                                   glm::ortho(-130.0f, 130.0f, -130.0f, 130.0f), 256, 256, 
                                                   render::create_mipmapped_texture(isomap), true, false, false);
        cv::imwrite(outPath+"/"+"face"+std::to_string(i)+".png",core::to_mat(rd));
    }
    
    cv::imwrite(outPath+"/"+"merged_isomap.png", merged_isomap);
}

int main(int argc,char **argv)
{
    cv::Mat img = cv::imread(argv[1]);
    
    g_face3d.inputFrame(img);
    g_face3d.output();
    
    return 0;
}
