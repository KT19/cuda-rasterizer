#ifndef GEOMETRY_CUH
#define GEOMETRY_CUH
#include<stdio.h>
#include"Vec.cuh"
#include"Mat.cuh"

//Define vertex
struct Vertex {
    Vec3 position;
    Vec3 color;
    Vec3 normal; //For lighting calculations
};

//Triangle definition
struct Triangle {
    Vertex v0, v1, v2;
};

//Edge functions
__device__ float edgeFunction(const Vec3& v0, const Vec3& v1, const Vec3& v2) {
    return (v2.x - v0.x) * (v1.y - v0.y) - (v2.y - v0.y) * (v1.x - v0.x);
}

__device__ bool pointInTriangle(float x, float y, const Vec3& v0, const Vec3& v1, const Vec3& v2) {
    Vec3 p(x, y, 0);
    float edge0 = edgeFunction(v0, v1, p);
    float edge1 = edgeFunction(v1, v2, p);
    float edge2 = edgeFunction(v2, v0, p);

    return (edge0 >= 0 && edge1 >= 0 && edge2 >= 0);
}

__device__ void computeBarycentricCoordinates(
    float x, float y,
    const Vec3& v0, const Vec3& v1, const Vec3& v2,
    float& u, float&v, float& w
) {
    float area = edgeFunction(v0, v1, v2);

    Vec3 p(x, y, 0);
    u = edgeFunction(v1, v2, p) / area;
    v = edgeFunction(v2, v0, p) / area;
    w = 1 - u - v;

    return;
}

//Transform vertices from world space to screen space
__global__ void transformVertices(
    Triangle* triangles,
    Vec3* screenVertices,
    int numTriangles,
    Mat4 viewProjectionMatrix,
    int screenWidth,
    int screenHeight
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < numTriangles * 3) {
        int triangleIdx = idx / 3;
        int vertexIdx = idx % 3;

        Vertex vertex;
        if(vertexIdx == 0) vertex = triangles[triangleIdx].v0;
        else if(vertexIdx == 1) vertex = triangles[triangleIdx].v1;
        else vertex = triangles[triangleIdx].v2;

        //Transform to clip space
        Vec4 clipPos = viewProjectionMatrix * Vec4(vertex.position);

        //Perspective divide to get normalized device coordinates (NDC)
        Vec3 ndcPos = Vec3(clipPos.x / clipPos.w, clipPos.y / clipPos.w, clipPos.z / clipPos.w);
    
        screenVertices[idx] = Vec3(
            (1.0f - (ndcPos.x + 1.0f) * 0.5f) * screenWidth,
            (1.0f - ndcPos.y) * 0.5f * screenHeight,
            (ndcPos.z+1)/2
        );
    }

    return;
}

#endif