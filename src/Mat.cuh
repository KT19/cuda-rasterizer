#ifndef MAT_CUH
#define MAT_CUH
#include"Vec.cuh"

struct Mat4 {
    float m[16]; //column-major order

    __host__ __device__ Mat4() {
        for(int i = 0;i < 16;i++) {
            m[i] = (i % 5 == 0) ? 1.0f : 0.0f; //Identity matrix
        }
    }

    __host__ __device__ Vec4 operator*(const Vec4& v) const {
        return Vec4(
            m[0] * v.x + m[4] * v.y + m[8] * v.z + m[12] * v.w,
            m[1] * v.x + m[5] * v.y + m[9] * v.z + m[13] * v.w,
            m[2] * v.x + m[6] * v.y + m[10] * v.z + m[14] * v.w,
            m[3] * v.x + m[7] * v.y + m[11] * v.z + m[15] * v.w
        );
    }

    __host__ __device__ Mat4 operator*(const Mat4& other) const {
        Mat4 result;
        for(int row = 0;row < 4;row++) {
            for(int col = 0; col < 4;col++) {
                result.m[row + col * 4] = 0;
                for(int k = 0;k < 4;k++) {
                    result.m[row + col * 4] += m[row + k * 4] * other.m[k + col * 4];
                }
            }
        }

        return result;
    }

    static Mat4 translate(float x, float y, float z) {
        Mat4 result;
        result.m[12] = x;
        result.m[13] = y;
        result.m[14] = z;
        return result;
    }

    static Mat4 scale(float x, float y, float z) {
        Mat4 result;
        result.m[0] = x;
        result.m[5] = y;
        result.m[10] = z;

        return result;
    }

    static Mat4 rotateX(float radians) {
        Mat4 result;
        float c = cosf(radians);
        float s = sinf(radians);
        result.m[5] = c;
        result.m[9] = -s;
        result.m[6] = s;
        result.m[10] = c;
        return result;
    }

    static Mat4 rotateY(float radians) {
        Mat4 result;
        float c = cosf(radians);
        float s = sinf(radians);
        result.m[0] = c;
        result.m[8] = s;
        result.m[2] = -s;
        result.m[10] = c;

        return result;
    }

    static Mat4 rotateZ(float radians) {
        Mat4 result;
        float c = cosf(radians);
        float s = sinf(radians);
        result.m[0] = c;
        result.m[4] = -s;
        result.m[1] = s;
        result.m[5] = c;
        return result;
    }

    static Mat4 perspective(float fovY, float aspect, float near, float far) {
        Mat4 result;
        float f = 1.0f / tanf(fovY * 0.5f);
        float nf = 1.0f / (far - near);

        result.m[0] = f / aspect;
        result.m[5] = f;
        result.m[10] = -(far + near) * nf;
        result.m[11] = -1.0f;
        result.m[14] = -2 * far * near * nf;
        result.m[15] = 0.0f;

        return result;
    }

    static Mat4 lookAt(const Vec3& eye, const Vec3& target, const Vec3& up) {
        Mat4 result;

        Vec3 f = (target - eye).normalize();
        Vec3 s = up.cross(f).normalize();
        Vec3 u = f.cross(s);

        result.m[0] = s.x;
        result.m[4] = s.y;
        result.m[8] = s.z;

        result.m[1] = u.x;
        result.m[5] = u.y;
        result.m[9] = u.z;

        result.m[2] = -f.x;
        result.m[6] = -f.y;
        result.m[10] = -f.z;

        result.m[12] = -s.dot(eye);
        result.m[13] = -u.dot(eye);
        result.m[14] = f.dot(eye);

        return result;
    }
};

#endif