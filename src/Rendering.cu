#ifndef RENDERING_CU
#define RENDERING_CU
#include<cuda_runtime.h>
#include<stdio.h>
#include"Vec.cuh"
#include"Mat.cuh"
#include"Camera.cuh"
#include"Geometry.cuh"

//Frame buffer
struct FrameBuffer {
    float* colorBuffer;
    float* depthBuffer;
    int width;
    int height;
};


//Main rasterization kernel
__global__ void rasterizerTriangles(
    Triangle* triangles,
    Vec3* screenVertices,
    int numTriangles,
    FrameBuffer frameBuffer
) {
    int triangleIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if(triangleIdx < numTriangles) {
        Vec3 v0 = screenVertices[triangleIdx * 3];
        Vec3 v1 = screenVertices[triangleIdx * 3 + 1];
        Vec3 v2 = screenVertices[triangleIdx * 3 + 2];

        float area = edgeFunction(v0, v1, v2);
        if(area <= 0) return; //counter-clockwise triangles are front-facing

        //Vertex color
        Vec3 c0 = triangles[triangleIdx].v0.color;
        Vec3 c1 = triangles[triangleIdx].v1.color;
        Vec3 c2 = triangles[triangleIdx].v2.color;

        //bounding box of the triangle
        int minX = min(0, (int)floorf(min(min(v0.x, v1.x), v2.x)));
        int maxX = min(frameBuffer.width - 1, (int)ceilf(max(max(v0.x, v1.x), v2.x)));
        int minY = min(0, (int)floorf(min(min(v0.y, v1.y), v2.y)));
        int maxY = min(frameBuffer.height - 1, (int)ceilf(max(max(v0.y, v1.y), v2.y)));

        //Iterate over each pixel
        for(int y = minY; y <= maxY;y++) {
            for(int x = minX; x <= maxX;x++) {
                float pixelX = x + 0.5f;
                float pixelY = y + 0.5f;

                //check if the pixel is inside the triangle
                if(pointInTriangle(pixelX, pixelY, v0, v1, v2)) {
                    float u, v, w;
                    computeBarycentricCoordinates(pixelX, pixelY, v0, v1, v2, u, v, w);

                    //Interpolate z for depth
                    float z = u * v0.z + v * v1.z + w * v2.z;

                    //Interpolate color
                    Vec3 color = c0 * u + c1 * v + c2 * w;

                    int idx = y * frameBuffer.width + x;
        
                    if(z < frameBuffer.depthBuffer[idx]) {
                        atomicMin((int*)&frameBuffer.depthBuffer[idx], __float_as_int(z));
                        
                        //Only update the color
                        if(z == __int_as_float(atomicMin((int*)&frameBuffer.depthBuffer[idx], __float_as_int(z)))) {
                            frameBuffer.colorBuffer[idx * 3] = min(255,(unsigned char)(255.99*color.x));
                            frameBuffer.colorBuffer[idx * 3 + 1] = min(255,(unsigned char)(255.99*color.y));
                            frameBuffer.colorBuffer[idx * 3 + 2] = min(255,(unsigned char)(255.99*color.z));
                        }
                    }

                }
            }
        }
    }

    return;
}


//Host function to setup and launch rasterization
void renderScene(
    Triangle* hostTriangles,
    int numTriangles,
    Camera camera,
    float* hostColorBuffer,
    float* hostDepthBuffer,
    int width,
    int height
) {
    //Device memory allocation
    Triangle* deviceTriangles;
    Vec3* deviceScreenVertices;
    float* deviceColorBuffer;
    float* deviceDepthBuffer;

    cudaMalloc((void**)&deviceTriangles, numTriangles * sizeof(Triangle));
    cudaMalloc((void**)&deviceScreenVertices, numTriangles * 3 * sizeof(Vec3));
    cudaMalloc((void**)&deviceColorBuffer, width * height * 3 * sizeof(float));
    cudaMalloc((void**)&deviceDepthBuffer, width * height * sizeof(float));

    for(int i = 0;i < width * height;i++) {
        hostDepthBuffer[i] = __FLT_MAX__;
    }
    cudaMemcpy(deviceDepthBuffer, hostDepthBuffer, width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(deviceColorBuffer, 0, width * height * 3 * sizeof(float));
    cudaMemcpy(deviceTriangles, hostTriangles, numTriangles * sizeof(Triangle), cudaMemcpyHostToDevice);

    //Setup frame buffer
    FrameBuffer deviceFrameBuffer;

    deviceFrameBuffer.colorBuffer = deviceColorBuffer;
    deviceFrameBuffer.depthBuffer = deviceDepthBuffer;
    deviceFrameBuffer.width = width;
    deviceFrameBuffer.height = height;

    //View projection
    Mat4 viewProjectionMatrix = camera.getViewProjectionMatrix();

    int threadsPerBlock = 16 * 16;
    int blocksPerGrid = (numTriangles * 3 + threadsPerBlock - 1) / threadsPerBlock;

    transformVertices<<<blocksPerGrid, threadsPerBlock>>>(
        deviceTriangles,
        deviceScreenVertices,
        numTriangles,
        viewProjectionMatrix,
        width,
        height
    );

    //launch rasterization kernel
    blocksPerGrid = (numTriangles + threadsPerBlock - 1) / threadsPerBlock;

    rasterizerTriangles<<<blocksPerGrid, threadsPerBlock>>>(deviceTriangles, deviceScreenVertices, numTriangles, deviceFrameBuffer);

    //wait
    cudaDeviceSynchronize();

    cudaMemcpy(hostColorBuffer, deviceColorBuffer, width * height * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostDepthBuffer, deviceDepthBuffer, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    //Free
    cudaFree(deviceTriangles);
    cudaFree(deviceScreenVertices);
    cudaFree(deviceColorBuffer);
    cudaFree(deviceDepthBuffer);

    return;
}

#endif