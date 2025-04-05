#ifndef VEC_CUH
#define VEC_CUH
struct Vec3 {
    float x, y, z;

    __host__ __device__ Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ Vec3(float x, float y, float z) : x(x), y(y), z(z) {}

    __host__ __device__ Vec3 operator+(const Vec3& v) const { return Vec3(x + v.x, y + v.y, z + v.z); }
    __host__ __device__ Vec3 operator-(const Vec3& v) const { return Vec3(x - v.x, y - v.y, z - v.z); }
    __host__ __device__ Vec3 operator-(void) const { return Vec3(-x, -y, -z); }
    __host__ __device__ Vec3 operator*(float s) const { return Vec3(x * s, y * s, z * s); }
    __host__ __device__ float dot(const Vec3& v) const { return x * v.x + y * v.y + z * v.z; }
    __host__ __device__ Vec3 cross(const Vec3& v) const { return Vec3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);}

    __host__ __device__ float length() const { return sqrtf(x * x + y * y + z * z);}
    __host__ __device__ Vec3 normalize() const {
        float len = length();
        return Vec3(x / len, y / len, z / len);
    }
};

struct Vec4 {
    float x, y, z, w;

    __host__ __device__ Vec4() : x(0), y(0), z(0), w(1) {}
    __host__ __device__ Vec4(float x, float y, float z, float w = 1.0f) : x(x), y(y), z(z), w(w) {}
    __host__ __device__ Vec4(const Vec3& v, float w = 1.0f) : x(v.x), y(v.y), z(v.z), w(w) {}

    __host__ __device__ Vec3 toVec3() const { return Vec3(x / w, y / w, z / w); }
};


#endif