#ifndef CAMERA_CUH
#define CAMERA_CUH
#include"Vec.cuh"
#include"Mat.cuh"

//Camera definition
struct Camera {
    Vec3 position;
    Vec3 target;
    Vec3 up;
    float fov;
    float aspect;
    float near;
    float far;

    Mat4 getViewMatrix() const {
        return Mat4::lookAt(position, target, up);
    }

    Mat4 getProjectionMatrix() const {
        return Mat4::perspective(fov, aspect, near, far);
    }

    //Combined view-projection matrix
    Mat4 getViewProjectionMatrix() const {
        return getProjectionMatrix() * getViewMatrix();
    }
};

#endif