#include "helpers.hpp"

#include "eos/core/Landmark.hpp"
#include "eos/core/LandmarkMapper.hpp"
#include "eos/core/Mesh.hpp"
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

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/objdetect.hpp"

#include <vector>
#include <iostream>

#include "FaceDetection.h"

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

void draw_axes_topright(float r_x, float r_y, float r_z, cv::Mat image);

int main()
{
    std::string modelfile, landmarkdetector, mappingsfile, contourfile, edgetopologyfile, blendshapesfile;
    
    modelfile = "../share/sfm_shape_3448.bin";
    mappingsfile = "../share/ibug_to_sfm.txt";
    contourfile = "../share/sfm_model_contours.json";
    edgetopologyfile = "../share/sfm_3448_edge_topology.json";
    blendshapesfile = "../share/expression_blendshapes_3448.bin";
    
    landmarkdetector = "../share/face_landmarks_model_rcr_68.bin";
    
    // Load the Morphable Model and the LandmarkMapper:
    const morphablemodel::MorphableModel morphable_model = morphablemodel::load_model(modelfile);
    const core::LandmarkMapper landmark_mapper = mappingsfile.empty() ? core::LandmarkMapper() : core::LandmarkMapper(mappingsfile);
    
    const fitting::ModelContour model_contour = contourfile.empty() ? fitting::ModelContour() : fitting::ModelContour::load(contourfile);
    const fitting::ContourLandmarks ibug_contour = fitting::ContourLandmarks::load(mappingsfile);
    
    rcr::detection_model rcr_model;
    // Load the landmark detection model:
    try {
        rcr_model = rcr::load_detection_model(landmarkdetector);
    }
    catch (const cereal::Exception& e) {
        cout << "Error reading the RCR model " << landmarkdetector << ": " << e.what() << endl;
        return EXIT_FAILURE;
    }
    
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "Couldn't open the given file or camera 0." << endl;
        return EXIT_FAILURE;
    }
    
    const morphablemodel::Blendshapes blendshapes = morphablemodel::load_blendshapes(blendshapesfile);
    
    const morphablemodel::EdgeTopology edge_topology = morphablemodel::load_edge_topology(edgetopologyfile);
    
    cv::namedWindow("video", 1);
    cv::namedWindow("render", 0);
    
    Mat frame, unmodified_frame;
    
    bool have_face = false;
    rcr::LandmarkCollection<Vec2f> current_landmarks;
    Rect current_facebox;
    WeightedIsomapAveraging isomap_averaging(60.f); // merge all triangles that are facing <60° towards the camera
    PcaCoefficientMerging pca_shape_merging;
    
    for (;;)
    {
        cap >> frame; // get a new frame from camera
        if (frame.empty()) { // stop if we're at the end of the video
            break;
        }
        
        // We do a quick check if the current face's width is <= 50 pixel. If it is, we re-initialise the tracking with the face detector.
        if (have_face && get_enclosing_bbox(rcr::to_row(current_landmarks)).width <= 50) {
            cout << "Reinitialising because the face bounding-box width is <= 50 px" << endl;
            have_face = false;
        }
        
        unmodified_frame = frame.clone();
        
        if (!have_face) {
            // Run the face detector and obtain the initial estimate using the mean landmarks:
            vector<Rect> detected_faces;
            //face_cascade.detectMultiScale(unmodified_frame, detected_faces, 1.2, 2, 0, cv::Size(110, 110));
            g_faceDT.detect(unmodified_frame,detected_faces);
            if (detected_faces.empty()) {
                cv::imshow("video", frame);
                cv::waitKey(30);
                continue;
            }
            cv::rectangle(frame, detected_faces[0], { 255, 0, 0 });
            // Rescale the V&J facebox to make it more like an ibug-facebox:
            // (also make sure the bounding box is square, V&J's is square)
            Rect ibug_facebox = rescale_facebox(detected_faces[0], 0.85, 0.2);
            
            current_landmarks = rcr_model.detect(unmodified_frame, ibug_facebox);
            rcr::draw_landmarks(frame, current_landmarks, { 0, 0, 255 }); // red, initial landmarks
            
            have_face = true;
        }
        else {
            // We already have a face - track and initialise using the enclosing bounding
            // box from the landmarks from the last frame:
            auto enclosing_bbox = get_enclosing_bbox(rcr::to_row(current_landmarks));
            enclosing_bbox = make_bbox_square(enclosing_bbox);
            current_landmarks = rcr_model.detect(unmodified_frame, enclosing_bbox);
            rcr::draw_landmarks(frame, current_landmarks, { 255, 0, 0 }); // blue, the new optimised landmarks
        }
        
        // Fit the 3DMM:
        fitting::RenderingParameters rendering_params;
        vector<float> shape_coefficients, blendshape_coefficients;
        vector<Eigen::Vector2f> image_points;
        core::Mesh mesh;
        std::tie(mesh, rendering_params) = fitting::fit_shape_and_pose(morphable_model, blendshapes, rcr_to_eos_landmark_collection(current_landmarks), landmark_mapper, unmodified_frame.cols, unmodified_frame.rows, edge_topology, ibug_contour, model_contour, 3, 5, 15.0f, cpp17::nullopt, shape_coefficients, blendshape_coefficients, image_points);
        
        // Draw the 3D pose of the face:
        draw_axes_topright(glm::eulerAngles(rendering_params.get_rotation())[0], glm::eulerAngles(rendering_params.get_rotation())[1], glm::eulerAngles(rendering_params.get_rotation())[2], frame);
        
        // Wireframe rendering of mesh of this frame (non-averaged):
        render::draw_wireframe(frame, mesh, rendering_params.get_modelview(), rendering_params.get_projection(), fitting::get_opencv_viewport(frame.cols, frame.rows));
        
        // Extract the texture using the fitted mesh from this frame:
        const Eigen::Matrix<float, 3, 4> affine_cam = fitting::get_3x4_affine_camera_matrix(rendering_params, frame.cols, frame.rows);
        const Mat isomap = core::to_mat(render::extract_texture(mesh, affine_cam, core::from_mat(unmodified_frame), true, render::TextureInterpolation::NearestNeighbour, 512));
        
        // Merge the isomaps - add the current one to the already merged ones:
        const Mat merged_isomap = isomap_averaging.add_and_merge(isomap);
        // Same for the shape:
        shape_coefficients = pca_shape_merging.add_and_merge(shape_coefficients);
        const Eigen::VectorXf merged_shape =
                morphable_model.get_shape_model().draw_sample(shape_coefficients) +
                to_matrix(blendshapes) *
                Eigen::Map<const Eigen::VectorXf>(blendshape_coefficients.data(), blendshape_coefficients.size());
        const core::Mesh merged_mesh = morphablemodel::sample_to_mesh(merged_shape, morphable_model.get_color_model().get_mean(), morphable_model.get_shape_model().get_triangle_list(), morphable_model.get_color_model().get_triangle_list(), morphable_model.get_texture_coordinates());
        
        // Render the model in a separate window using the estimated pose, shape and merged texture:
        core::Image4u rendering;
        auto modelview_no_translation = rendering_params.get_modelview();
        modelview_no_translation[3][0] = 0;
        modelview_no_translation[3][1] = 0;
        std::tie(rendering, std::ignore) = render::render(merged_mesh, modelview_no_translation, glm::ortho(-130.0f, 130.0f, -130.0f, 130.0f), 256, 256, render::create_mipmapped_texture(merged_isomap), true, false, false);
        cv::imshow("render", core::to_mat(rendering));
        
        cv::imshow("video", frame);
        auto key = cv::waitKey(30);
        if (key == 'q') break;
        if (key == 'r') {
            have_face = false;
            isomap_averaging = WeightedIsomapAveraging(60.f);
        }
        if (key == 's') {
            // save an obj + current merged isomap to the disk:
            const core::Mesh neutral_expression = morphablemodel::sample_to_mesh(morphable_model.get_shape_model().draw_sample(shape_coefficients), morphable_model.get_color_model().get_mean(), morphable_model.get_shape_model().get_triangle_list(), morphable_model.get_color_model().get_triangle_list(), morphable_model.get_texture_coordinates());
            core::write_textured_obj(neutral_expression, "current_merged.obj");
            cv::imwrite("current_merged.isomap.png", merged_isomap);
        }
    }
    
    return EXIT_SUCCESS;
};

/**
 * @brief Draws 3D axes onto the top-right corner of the image. The
 * axes are oriented corresponding to the given angles.
 *
 * @param[in] r_x Pitch angle, in radians.
 * @param[in] r_y Yaw angle, in radians.
 * @param[in] r_z Roll angle, in radians.
 * @param[in] image The image to draw onto.
 */
void draw_axes_topright(float r_x, float r_y, float r_z, cv::Mat image)
{
    const glm::vec3 origin(0.0f, 0.0f, 0.0f);
    const glm::vec3 x_axis(1.0f, 0.0f, 0.0f);
    const glm::vec3 y_axis(0.0f, 1.0f, 0.0f);
    const glm::vec3 z_axis(0.0f, 0.0f, 1.0f);
    
    const auto rot_mtx_x = glm::rotate(glm::mat4(1.0f), r_x, glm::vec3{ 1.0f, 0.0f, 0.0f });
    const auto rot_mtx_y = glm::rotate(glm::mat4(1.0f), r_y, glm::vec3{ 0.0f, 1.0f, 0.0f });
    const auto rot_mtx_z = glm::rotate(glm::mat4(1.0f), r_z, glm::vec3{ 0.0f, 0.0f, 1.0f });
    const auto modelview = rot_mtx_z * rot_mtx_x * rot_mtx_y;
    
    const auto viewport = fitting::get_opencv_viewport(image.cols, image.rows);
    const float aspect = static_cast<float>(image.cols) / image.rows;
    const auto ortho_projection = glm::ortho(-3.0f * aspect, 3.0f * aspect, -3.0f, 3.0f);
    const auto translate_topright = glm::translate(glm::mat4(1.0f), glm::vec3(0.7f, 0.65f, 0.0f));
    const auto o_2d = glm::project(origin, modelview, translate_topright * ortho_projection, viewport);
    const auto x_2d = glm::project(x_axis, modelview, translate_topright * ortho_projection, viewport);
    const auto y_2d = glm::project(y_axis, modelview, translate_topright * ortho_projection, viewport);
    const auto z_2d = glm::project(z_axis, modelview, translate_topright * ortho_projection, viewport);
    cv::line(image, cv::Point2f{ o_2d.x, o_2d.y }, cv::Point2f{ x_2d.x, x_2d.y }, { 0, 0, 255 });
    cv::line(image, cv::Point2f{ o_2d.x, o_2d.y }, cv::Point2f{ y_2d.x, y_2d.y }, { 0, 255, 0 });
    cv::line(image, cv::Point2f{ o_2d.x, o_2d.y }, cv::Point2f{ z_2d.x, z_2d.y }, { 255, 0, 0 });
};
