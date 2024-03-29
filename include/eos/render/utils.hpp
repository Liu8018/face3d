#pragma once

#ifndef RENDER_UTILS_HPP_
#define RENDER_UTILS_HPP_

#include "glm/vec3.hpp"
#include "glm/geometric.hpp"

#include "Eigen/Core"

namespace eos {
namespace render {

/**
 * Transforms a point from clip space ([-1, 1] x [-1, 1]) to
 * image (screen) coordinates, i.e. the window transform.
 * Note that the y-coordinate is flipped because the image origin
 * is top-left while in clip space top is +1 and bottom is -1.
 * No z-division is performed.
 * Note: It should rather be called from NDC to screen space?
 *
 * Exactly conforming to the OpenGL viewport transform, except that
 * we flip y at the end.
 * Qt: Origin top-left. OpenGL: bottom-left. OCV: top-left.
 *
 * @param[in] clip_coordinates A point in clip coordinates.
 * @param[in] screen_width Width of the screen or window.
 * @param[in] screen_height Height of the screen or window.
 * @return A vector with x and y coordinates transformed to screen space.
 */
inline glm::vec2 clip_to_screen_space(const glm::vec2& clip_coordinates, int screen_width, int screen_height)
{
    // Window transform:
    const float x_ss = (clip_coordinates[0] + 1.0f) * (screen_width / 2.0f);
    const float y_ss =
        screen_height - (clip_coordinates[1] + 1.0f) *
                            (screen_height / 2.0f); // also flip y; Qt: Origin top-left. OpenGL: bottom-left.
    return glm::vec2(x_ss, y_ss);
    /* Note: What we do here is equivalent to
       x_w = (x *  vW/2) + vW/2;
       However, Shirley says we should do:
       x_w = (x *  vW/2) + (vW-1)/2;
       (analogous for y)
       Todo: Check the consequences.
    */
};

/**
 * Transforms a point from image (screen) coordinates to
 * clip space ([-1, 1] x [-1, 1]).
 * Note that the y-coordinate is flipped because the image origin
 * is top-left while in clip space top is +1 and bottom is -1.
 *
 * @param[in] screen_coordinates A point in screen coordinates.
 * @param[in] screen_width Width of the screen or window.
 * @param[in] screen_height Height of the screen or window.
 * @return A vector with x and y coordinates transformed to clip space.
 */
/*inline cv::Vec2f screen_to_clip_space(const cv::Vec2f& screen_coordinates, int screen_width, int screen_height)
{
    const float x_cs = screen_coordinates[0] / (screen_width / 2.0f) - 1.0f;
    float y_cs = screen_coordinates[1] / (screen_height / 2.0f) - 1.0f;
    y_cs *= -1.0f;
    return cv::Vec2f(x_cs, y_cs);
};*/

template <typename T, glm::precision P = glm::defaultp>
glm::tvec2<T, P> clip_to_screen_space(const T clip_coord_x, const T clip_coord_y, int screen_width,
                                      int screen_height)
{
    // Todo: See/copy notes from utils.hpp/clip_to_screen_space.
    const T x_ss = (clip_coord_x + T(1)) * (screen_width / 2.0);
    const T y_ss =
        screen_height - (clip_coord_y + T(1)) *
                            (screen_height / 2.0); // also flip y; Qt: Origin top-left. OpenGL: bottom-left.
    return glm::tvec2<T, P>(x_ss, y_ss);
};

/**
 * Calculates the normal of a face (or triangle), i.e. the
 * per-face normal. Return normal will be normalised.
 * Assumes the triangle is given in CCW order, i.e. vertices
 * in counterclockwise order on the screen are front-facing.
 *
 * @param[in] v0 First vertex.
 * @param[in] v1 Second vertex.
 * @param[in] v2 Third vertex.
 * @return The unit-length normal of the given triangle.
 */
inline Eigen::Vector3f compute_face_normal(const Eigen::Vector3f& v0, const Eigen::Vector3f& v1,
                                           const Eigen::Vector3f& v2)
{
    Eigen::Vector3f n = (v1 - v0).cross(v2 - v0); // v0-to-v1 x v0-to-v2
    return n.normalized();
};

// Todo: Doxygen. Actually this is the overload that's probably most used?
inline Eigen::Vector3f compute_face_normal(const Eigen::Vector4f& v0, const Eigen::Vector4f& v1,
                                           const Eigen::Vector4f& v2)
{
    Eigen::Vector4f n = (v1 - v0).cross3(v2 - v0); // v0-to-v1 x v0-to-v2
    return n.head<3>().normalized();
};

/**
 * Computes the normal of a face (or triangle), i.e. the
 * per-face normal. Return normal will be normalised.
 * Assumes the triangle is given in CCW order, i.e. vertices
 * in counterclockwise order on the screen are front-facing.
 *
 * @param[in] v0 First vertex.
 * @param[in] v1 Second vertex.
 * @param[in] v2 Third vertex.
 * @return The unit-length normal of the given triangle.
 */
inline glm::vec3 compute_face_normal(const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2)
{
    glm::vec3 n = glm::cross(v1 - v0, v2 - v0); // v0-to-v1 x v0-to-v2
    n = glm::normalize(n);
    return n;
};

} /* namespace render */
} /* namespace eos */

#endif /* RENDER_UTILS_HPP_ */
